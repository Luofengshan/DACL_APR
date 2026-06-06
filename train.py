"""
训练模块：对比学习训练循环
支持三种模式训练（用于消融实验）：dual, token_only, graph_only
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from models import APRContrastiveModel
from dataset import BugFixDataset, collate_fn, load_data


def check_codebert_available():
    """检查CodeBERT模型和tokenizer是否都可加载"""
    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained(config.CODEBERT_MODEL)
        AutoModel.from_pretrained(config.CODEBERT_MODEL)
        return True
    except Exception as e:
        print(f"CodeBERT不可用({e})，将使用轻量Token编码器")
        return False


def get_tokenizer(use_codebert=True):
    """加载tokenizer：CodeBERT可用时返回其tokenizer，否则返回None（使用简单tokenize）"""
    if not use_codebert:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(config.CODEBERT_MODEL)
    except Exception:
        return None


def train_one_epoch(model, dataloader, optimizer, scheduler, device, mode="dual"):
    model.train()
    total_loss = 0
    total_contrast = 0
    total_align = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc=f"Training ({mode})", leave=False):
        buggy_ids = batch["buggy_ids"].to(device)
        buggy_mask = batch["buggy_mask"].to(device)
        fixed_ids = batch["fixed_ids"].to(device)
        fixed_mask = batch["fixed_mask"].to(device)
        defect_labels = batch["defect_labels"].to(device)

        # 将图数据移到device
        buggy_graphs = []
        for node_types, adj_matrix, num_nodes in batch["buggy_graphs"]:
            buggy_graphs.append((
                node_types.to(device),
                adj_matrix.to(device),
                num_nodes
            ))
        fixed_graphs = []
        for node_types, adj_matrix, num_nodes in batch["fixed_graphs"]:
            fixed_graphs.append((
                node_types.to(device),
                adj_matrix.to(device),
                num_nodes
            ))

        optimizer.zero_grad()
        loss, l_contrast, l_align = model.compute_loss(
            buggy_ids, buggy_mask, buggy_graphs,
            fixed_ids, fixed_mask, fixed_graphs,
            defect_labels, mode=mode
        )

        if loss.requires_grad and loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_contrast += l_contrast.item()
        total_align += l_align.item()
        num_batches += 1

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(num_batches, 1),
        "contrastive_loss": total_contrast / max(num_batches, 1),
        "alignment_loss": total_align / max(num_batches, 1),
    }


def train(mode="dual", epochs=None, save_name=None):
    """完整的训练流程"""
    epochs = epochs or config.NUM_EPOCHS
    save_name = save_name or f"model_{mode}"
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"训练模式: {mode} | 设备: {device} | Epochs: {epochs}")
    print(f"{'='*60}")

    # 数据
    train_data = load_data(split="train")
    use_codebert = check_codebert_available()
    tokenizer = get_tokenizer(use_codebert)
    train_dataset = BugFixDataset(train_data, tokenizer, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    print(f"训练样本数: {len(train_dataset)}")

    # 模型
    model = APRContrastiveModel(use_codebert=use_codebert).to(device)

    # 优化器
    # CodeBERT层用较小的学习率，其他层用较大学习率
    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "codebert" in name:
            bert_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": config.LEARNING_RATE},
        {"params": other_params, "lr": config.LEARNING_RATE * 10},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    history = []
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_start = time.time()
        metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, mode)
        epoch_time = time.time() - epoch_start

        metrics["epoch"] = epoch + 1
        metrics["time"] = epoch_time
        history.append(metrics)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Contrast: {metrics['contrastive_loss']:.4f} | "
              f"Align: {metrics['alignment_loss']:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            save_path = os.path.join(config.MODEL_DIR, f"{save_name}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "mode": mode,
            }, save_path)

    # 保存训练历史
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    hist_path = os.path.join(config.RESULT_DIR, f"train_history_{mode}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    train(mode="dual")
