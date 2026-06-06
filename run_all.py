"""
一键运行完整实验：数据生成 → 训练 → 评估 → 消融实验 → 基线对比
"""
import os
import sys
import json
import time
import torch
import numpy as np

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def step1_generate_data():
    """步骤1：生成数据集"""
    print("\n" + "="*60)
    print("步骤1：生成数据集")
    print("="*60)

    if os.path.exists(config.DATA_FILE):
        print(f"数据文件已存在: {config.DATA_FILE}")
        with open(config.DATA_FILE, "r") as f:
            data = json.load(f)
        print(f"  总样本数: {len(data)}")
        print(f"  训练集: {sum(1 for d in data if d['split'] == 'train')}")
        print(f"  测试集: {sum(1 for d in data if d['split'] == 'test')}")
        return

    from generate_data import main as generate_main
    generate_main()


def step2_train_and_evaluate():
    """步骤2：训练完整模型并评估"""
    print("\n" + "="*60)
    print("步骤2：训练双流融合模型 (dual)")
    print("="*60)

    from train import train
    from evaluate import evaluate

    model, history = train(mode="dual", save_name="model_dual")
    results = evaluate(model, mode="dual", result_prefix="")

    return model, history, results


def step3_ablation():
    """步骤3：消融实验"""
    print("\n" + "="*60)
    print("步骤3：消融实验")
    print("="*60)

    from evaluate import ablation_study
    all_results = ablation_study()
    return all_results


def step4_baselines():
    """步骤4：基线方法对比"""
    print("\n" + "="*60)
    print("步骤4：基线方法对比")
    print("="*60)

    from evaluate import evaluate_retrieval, evaluate_repair, compute_bleu
    from dataset import load_data
    from ast_graph import code_to_ast_graph
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    from train import get_tokenizer, check_codebert_available
    from models import APRContrastiveModel

    train_data = load_data(split="train")
    test_data = load_data(split="test")
    device = config.DEVICE

    # ---- 基线1: BM25/TF-IDF检索 ----
    print("\n基线1: TF-IDF检索")
    train_buggy_codes = [d["buggy_code"] for d in train_data]
    test_buggy_codes = [d["buggy_code"] for d in test_data]
    train_fixed_codes = [d["fixed_code"] for d in train_data]
    train_labels = np.array([d["defect_type_id"] for d in train_data])
    test_labels = np.array([d["defect_type_id"] for d in test_data])

    vectorizer = TfidfVectorizer(max_features=5000)
    all_codes = train_buggy_codes + test_buggy_codes
    tfidf_matrix = vectorizer.fit_transform(all_codes)
    train_tfidf = tfidf_matrix[:len(train_buggy_codes)]
    test_tfidf = tfidf_matrix[len(train_buggy_codes):]

    tfidf_retrieval = evaluate_retrieval(
        test_tfidf.toarray(), train_tfidf.toarray(),
        test_labels, train_labels, top_k=config.TOP_K
    )
    print(f"  TF-IDF检索结果: {tfidf_retrieval}")

    # TF-IDF修复
    tfidf_sim = cos_sim(test_tfidf, train_tfidf)
    tfidf_repair = evaluate_repair_with_sim(
        test_data, train_data, tfidf_sim, top_k=config.TOP_K
    )
    print(f"  TF-IDF修复结果: {tfidf_repair}")

    # ---- 基线2: CodeBERT原始向量（无对比学习微调） ----
    print("\n基线2: CodeBERT原始向量 (无微调)")
    try:
        use_cb = check_codebert_available()
        tokenizer = get_tokenizer(use_cb)
        if tokenizer is not None:
            from dataset import BugFixDataset, collate_fn
            from evaluate import encode_all

            model_no_finetune = APRContrastiveModel(use_codebert=use_cb).to(device)
            model_no_finetune.eval()

            train_dataset = BugFixDataset(train_data, tokenizer, split="train")
            test_dataset = BugFixDataset(test_data, tokenizer, split="test")

            train_buggy_reprs, train_fixed_reprs, train_l = encode_all(
                model_no_finetune, train_dataset, device, "token_only"
            )
            test_buggy_reprs, test_fixed_reprs, test_l = encode_all(
                model_no_finetune, test_dataset, device, "token_only"
            )

            bert_retrieval = evaluate_retrieval(
                test_buggy_reprs, train_buggy_reprs,
                test_l, train_l, top_k=config.TOP_K
            )
            bert_repair = evaluate_repair(
                test_data, train_data,
                test_buggy_reprs, train_fixed_reprs,
                top_k=config.TOP_K
            )
            print(f"  CodeBERT检索结果: {bert_retrieval}")
            print(f"  CodeBERT修复结果: {bert_repair}")
        else:
            bert_retrieval = {"MRR": 0, f"Recall@{config.TOP_K}": 0, f"nDCG@{config.TOP_K}": 0}
            bert_repair = {"BLEU-4": 0, "Exact Match": 0, f"Top-{config.TOP_K} Hit Rate": 0}
            print("  CodeBERT不可用，跳过此基线")
    except Exception as e:
        print(f"  CodeBERT基线失败: {e}")
        bert_retrieval = {"MRR": 0, f"Recall@{config.TOP_K}": 0, f"nDCG@{config.TOP_K}": 0}
        bert_repair = {"BLEU-4": 0, "Exact Match": 0, f"Top-{config.TOP_K} Hit Rate": 0}

    # 汇总基线结果
    baseline_results = {
        "TF-IDF": {**tfidf_retrieval, **tfidf_repair},
        "CodeBERT (no fine-tune)": {**bert_retrieval, **bert_repair},
    }

    # 加载本文方法结果
    dual_result_path = os.path.join(config.RESULT_DIR, "eval_dual.json")
    if os.path.exists(dual_result_path):
        with open(dual_result_path, "r") as f:
            baseline_results["Ours (Dual)"] = json.load(f)

    # 保存
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    with open(os.path.join(config.RESULT_DIR, "baseline_comparison.json"), "w") as f:
        json.dump(baseline_results, f, indent=2)

    # 打印对比表
    print_comparison_table(baseline_results)

    return baseline_results


def evaluate_repair_with_sim(test_data, train_data, sim_matrix, top_k=5):
    """基于相似度矩阵的修复评估（用于TF-IDF基线）"""
    from evaluate import compute_bleu

    bleu_scores = []
    exact_matches = []
    successful = 0

    for i in range(len(test_data)):
        scores = sim_matrix[i] if hasattr(sim_matrix[i], 'A') else sim_matrix[i].flatten()
        ranked = np.argsort(-np.array(scores).flatten())
        best_idx = ranked[0]
        predicted_fix = train_data[best_idx]["fixed_code"]
        true_fix = test_data[i]["fixed_code"]

        bleu = compute_bleu(predicted_fix, true_fix)
        bleu_scores.append(bleu)
        em = 1.0 if predicted_fix.strip() == true_fix.strip() else 0.0
        exact_matches.append(em)

        for idx in ranked[:top_k]:
            if train_data[idx]["fixed_code"].strip() == true_fix.strip():
                successful += 1
                break

    return {
        "BLEU-4": np.mean(bleu_scores),
        "Exact Match": np.mean(exact_matches),
        f"Top-{top_k} Hit Rate": successful / len(test_data),
    }


def print_comparison_table(results):
    """打印方法对比表"""
    print(f"\n{'='*70}")
    print(f"{'方法对比':^70}")
    print(f"{'='*70}")

    methods = list(results.keys())
    if not methods:
        return

    metrics = [k for k in results[methods[0]] if k != "mode"]
    header = f"{'指标':<20}" + "".join(f"{m:>15}" for m in methods)
    print(header)
    print("-" * 70)

    for metric in metrics:
        row = f"{metric:<20}"
        for method in methods:
            val = results[method].get(metric, 0)
            row += f"{val:>15.4f}"
        print(row)
    print("=" * 70)


def step5_final_report():
    """步骤5：生成实验结果汇总"""
    print("\n" + "="*60)
    print("步骤5：实验结果汇总")
    print("="*60)

    result_files = {
        "本文方法(dual)": "eval_dual.json",
        "消融-token_only": "eval_ablation_token_only.json",
        "消融-graph_only": "eval_ablation_graph_only.json",
        "基线对比": "baseline_comparison.json",
        "消融汇总": "ablation_summary.json",
    }

    all_results = {}
    for name, fname in result_files.items():
        fpath = os.path.join(config.RESULT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                all_results[name] = json.load(f)

    # 打印完整结果
    print("\n所有实验结果:")
    for name, data in all_results.items():
        print(f"\n--- {name} ---")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    # 保存最终报告
    report_path = os.path.join(config.RESULT_DIR, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n最终报告已保存至: {report_path}")

    return all_results


def main():
    """一键运行所有实验"""
    total_start = time.time()

    print("="*60)
    print("  基于缺陷感知对比学习的检索增强程序修复实验")
    print("  Defect-Aware Contrastive Learning for")
    print("  Retrieval-Augmented Program Repair")
    print("="*60)
    print(f"设备: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Embedding Dim: {config.EMBEDDING_DIM}")

    # 步骤1：数据生成
    step1_generate_data()

    # 步骤2：训练和评估完整模型
    model, history, results = step2_train_and_evaluate()

    # 步骤3：消融实验
    ablation_results = step3_ablation()

    # 步骤4：基线对比
    baseline_results = step4_baselines()

    # 步骤5：汇总
    final_results = step5_final_report()

    total_time = time.time() - total_start
    print(f"\n总实验时间: {total_time:.1f}s ({total_time/60:.1f}min)")
    print("实验完成!")


if __name__ == "__main__":
    main()
