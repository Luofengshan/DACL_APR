"""
评估模块：
1. 检索评估：MRR, Recall@K, nDCG@K
2. 修复评估：基于检索的修复，计算BLEU/Exact Match
3. 消融实验对比
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import config
from models import APRContrastiveModel
from dataset import BugFixDataset, collate_fn, load_data
from train import get_tokenizer, check_codebert_available


# ============ 编码所有样本 ============
@torch.no_grad()
def encode_all(model, dataset, device, mode="dual"):
    """将数据集中所有样本编码为向量"""
    model.eval()
    all_buggy_reprs = []
    all_fixed_reprs = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    for batch in tqdm(dataloader, desc="Encoding"):
        buggy_ids = batch["buggy_ids"].to(device)
        buggy_mask = batch["buggy_mask"].to(device)
        fixed_ids = batch["fixed_ids"].to(device)
        fixed_mask = batch["fixed_mask"].to(device)
        defect_labels = batch["defect_labels"]

        buggy_graphs = []
        for node_types, adj_matrix, num_nodes in batch["buggy_graphs"]:
            buggy_graphs.append((node_types.to(device), adj_matrix.to(device), num_nodes))
        fixed_graphs = []
        for node_types, adj_matrix, num_nodes in batch["fixed_graphs"]:
            fixed_graphs.append((node_types.to(device), adj_matrix.to(device), num_nodes))

        buggy_repr = model.encode(buggy_ids, buggy_mask, buggy_graphs, mode)
        fixed_repr = model.encode(fixed_ids, fixed_mask, fixed_graphs, mode)

        all_buggy_reprs.append(buggy_repr.cpu().numpy())
        all_fixed_reprs.append(fixed_repr.cpu().numpy())
        all_labels.append(defect_labels.numpy())

    buggy_reprs = np.concatenate(all_buggy_reprs, axis=0)
    fixed_reprs = np.concatenate(all_fixed_reprs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return buggy_reprs, fixed_reprs, labels


# ============ 检索评估 ============
def evaluate_retrieval(query_reprs, corpus_reprs, query_labels, corpus_labels, top_k=5):
    """
    评估检索质量：给定query（buggy代码），从corpus中检索最相似的样本

    Metrics: MRR, Recall@K, nDCG@K
    """
    sim_matrix = cosine_similarity(query_reprs, corpus_reprs)  # [Q, C]

    mrr_list = []
    recall_k_list = []
    ndcg_k_list = []

    for i in range(len(query_reprs)):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(-scores)

        # MRR: 第一个同类型样本的排名倒数
        mrr = 0.0
        for rank, idx in enumerate(ranked_indices):
            if corpus_labels[idx] == query_labels[i]:
                mrr = 1.0 / (rank + 1)
                break
        mrr_list.append(mrr)

        # Recall@K: 前K个中同类型的比例
        top_k_indices = ranked_indices[:top_k]
        relevant = sum(1 for idx in top_k_indices if corpus_labels[idx] == query_labels[i])
        total_relevant = sum(1 for cl in corpus_labels if cl == query_labels[i])
        recall_k = relevant / max(total_relevant, 1)
        recall_k_list.append(recall_k)

        # nDCG@K
        dcg = 0.0
        idcg = 0.0
        for rank, idx in enumerate(top_k_indices):
            rel = 1.0 if corpus_labels[idx] == query_labels[i] else 0.0
            dcg += rel / np.log2(rank + 2)
        # 理想情况
        ideal_rels = sorted([1.0 if cl == query_labels[i] else 0.0
                            for cl in corpus_labels], reverse=True)[:top_k]
        for rank, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(rank + 2)
        ndcg = dcg / max(idcg, 1e-8)
        ndcg_k_list.append(ndcg)

    return {
        "MRR": np.mean(mrr_list),
        f"Recall@{top_k}": np.mean(recall_k_list),
        f"nDCG@{top_k}": np.mean(ndcg_k_list),
    }


# ============ 修复评估 ============
def evaluate_repair(test_data, train_data, buggy_reprs_test, fixed_reprs_train,
                    top_k=config.TOP_K):
    """
    基于检索的修复评估：
    1. 对测试集buggy代码，从训练集中检索Top-K最相似的bug-fix对
    2. 使用检索到的fixed代码作为修复参考
    3. 计算BLEU和Exact Match
    """
    sim_matrix = cosine_similarity(buggy_reprs_test, fixed_reprs_train)

    bleu_scores = []
    exact_matches = []
    successful_repairs = 0

    for i in range(len(test_data)):
        scores = sim_matrix[i]
        top_indices = np.argsort(-scores)[:top_k]

        # 取最相似样本的fixed代码作为修复候选
        best_idx = top_indices[0]
        predicted_fix = train_data[best_idx]["fixed_code"]
        true_fix = test_data[i]["fixed_code"]

        # 计算BLEU-4 (简化版)
        bleu = compute_bleu(predicted_fix, true_fix)
        bleu_scores.append(bleu)

        # Exact Match
        em = 1.0 if predicted_fix.strip() == true_fix.strip() else 0.0
        exact_matches.append(em)

        # Top-K中是否有正确修复
        for idx in top_indices:
            if train_data[idx]["fixed_code"].strip() == true_fix.strip():
                successful_repairs += 1
                break

    return {
        "BLEU-4": np.mean(bleu_scores),
        "Exact Match": np.mean(exact_matches),
        f"Top-{top_k} Hit Rate": successful_repairs / len(test_data),
    }


def compute_bleu(predicted, reference):
    """简化版BLEU-4计算"""
    from collections import Counter

    def tokenize(code):
        # 简单的代码token化：按空白和标点分割
        tokens = code.replace("(", " ( ").replace(")", " ) ")
        tokens = tokens.replace("[", " [ ").replace("]", " ] ")
        tokens = tokens.replace(":", " : ").replace(",", " , ")
        tokens = tokens.replace(".", " . ").replace("=", " = ")
        return tokens.split()

    pred_tokens = tokenize(predicted)
    ref_tokens = tokenize(reference)

    if len(pred_tokens) == 0:
        return 0.0

    # BLEU-1到BLEU-4
    precisions = []
    for n in range(1, 5):
        pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])

        matches = sum((pred_ngrams & ref_ngrams).values())
        total = max(sum(pred_ngrams.values()), 1)
        precisions.append(matches / total)

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(np.log(p) for p in precisions) / 4
    return bp * np.exp(log_avg)


# ============ 完整评估流程 ============
def evaluate(model, mode="dual", result_prefix=""):
    """完整的评估流程"""
    device = config.DEVICE

    # 加载数据
    train_data = load_data(split="train")
    test_data = load_data(split="test")
    tokenizer = get_tokenizer(use_codebert=check_codebert_available())

    train_dataset = BugFixDataset(train_data, tokenizer, split="train")
    test_dataset = BugFixDataset(test_data, tokenizer, split="test")

    # 编码
    print(f"\n编码训练集 ({mode})...")
    train_buggy_reprs, train_fixed_reprs, train_labels = encode_all(
        model, train_dataset, device, mode
    )
    print(f"编码测试集 ({mode})...")
    test_buggy_reprs, test_fixed_reprs, test_labels = encode_all(
        model, test_dataset, device, mode
    )

    # 检索评估：用测试集buggy代码检索训练集中的相似样本
    print(f"\n检索评估 ({mode})...")
    retrieval_results = evaluate_retrieval(
        test_buggy_reprs, train_buggy_reprs,
        test_labels, train_labels, top_k=config.TOP_K
    )

    # 修复评估
    print(f"修复评估 ({mode})...")
    repair_results = evaluate_repair(
        test_data, train_data,
        test_buggy_reprs, train_fixed_reprs,
        top_k=config.TOP_K
    )

    results = {**retrieval_results, **repair_results}
    results["mode"] = mode

    # 打印结果
    print(f"\n{'='*50}")
    print(f"评估结果 ({mode}):")
    for k, v in results.items():
        if k != "mode":
            print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")

    # 保存
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    save_path = os.path.join(config.RESULT_DIR, f"eval_{result_prefix}{mode}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============ 消融实验 ============
def ablation_study():
    """
    消融实验：对比三种编码模式
    - dual: Token+Graph双流融合（完整方法）
    - token_only: 仅Token流
    - graph_only: 仅Graph流
    """
    from train import train

    all_results = {}

    for mode in ["dual", "token_only", "graph_only"]:
        print(f"\n{'#'*60}")
        print(f"# 消融实验: {mode}")
        print(f"{'#'*60}")

        model, _ = train(mode=mode, epochs=config.NUM_EPOCHS, save_name=f"ablation_{mode}")
        results = evaluate(model, mode=mode, result_prefix="ablation_")
        all_results[mode] = results

    # 保存汇总
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    summary_path = os.path.join(config.RESULT_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # 打印对比表
    print(f"\n{'='*70}")
    print(f"{'消融实验结果对比':^70}")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'dual':>15} {'token_only':>15} {'graph_only':>15}")
    print(f"{'-'*70}")
    metrics = [k for k in all_results["dual"] if k != "mode"]
    for metric in metrics:
        vals = [all_results[mode].get(metric, 0) for mode in ["dual", "token_only", "graph_only"]]
        print(f"{metric:<20} {vals[0]:>15.4f} {vals[1]:>15.4f} {vals[2]:>15.4f}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    ablation_study()
