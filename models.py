"""
核心模型模块：
1. TokenStreamEncoder - CodeBERT token流编码器
2. GraphStreamEncoder - GNN结构流编码器
3. DualStreamEncoder - 双流融合编码器（交叉注意力）
4. DefectAwareContrastiveLoss - 缺陷感知对比学习损失
5. CrossModalAlignmentLoss - 跨模态对齐损失
6. SimpleTokenEncoder - CodeBERT不可用时的轻量fallback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from ast_graph import NUM_NODE_TYPES


# ============ GCN层（纯PyTorch实现，无需PyG） ============
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        """
        x: [N, in_dim]  节点特征
        adj: [N, N]     归一化邻接矩阵
        """
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = self.layer_norm(x)
        return F.relu(x)


# ============ Token流编码器 ============
class SimpleTokenEncoder(nn.Module):
    """轻量Token编码器（CodeBERT不可用时的fallback）"""
    def __init__(self, vocab_size=50000, embed_dim=config.EMBEDDING_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, config.EMBEDDING_DIM)

    def forward(self, input_ids, attention_mask):
        # 将超出vocab范围的id截断到合法范围
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        x = self.embedding(input_ids)
        x = x * attention_mask.unsqueeze(-1)
        x = x.mean(dim=1)  # mean pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.proj(x)


class CodeBERTTokenEncoder(nn.Module):
    """CodeBERT token流编码器"""
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.codebert = AutoModel.from_pretrained(config.CODEBERT_MODEL)
        self.proj = nn.Linear(config.CODEBERT_HIDDEN, config.EMBEDDING_DIM)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] + mean pooling的混合表示
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        token_repr = outputs.last_hidden_state  # [B, L, 768]
        mask = attention_mask.unsqueeze(-1).float()
        mean_repr = (token_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, 768]
        combined = (cls_repr + mean_repr) / 2
        return self.proj(combined)  # [B, D]


# ============ 图流编码器 ============
class GraphStreamEncoder(nn.Module):
    """GNN结构流编码器：从AST图学习结构化代码表示"""
    def __init__(self, node_type_dim=64, hidden_dim=config.GNN_HIDDEN,
                 out_dim=config.EMBEDDING_DIM, num_layers=config.GNN_LAYERS):
        super().__init__()
        self.node_embedding = nn.Embedding(NUM_NODE_TYPES, node_type_dim, padding_idx=0)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = node_type_dim if i == 0 else hidden_dim
            self.gnn_layers.append(GCNLayer(in_d, hidden_dim))
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_types, adj_matrix, num_nodes=None):
        """
        node_types: [N]  节点类型ID
        adj_matrix: [N, N] 归一化邻接矩阵
        num_nodes: int   实际节点数（用于mask）
        """
        x = self.node_embedding(node_types)  # [N, node_type_dim]
        for gnn in self.gnn_layers:
            x = gnn(x, adj_matrix)  # [N, hidden_dim]

        # 图级别池化：对实际节点取平均
        if num_nodes is not None and num_nodes > 0:
            x = x[:num_nodes].mean(dim=0)  # [hidden_dim]
        else:
            x = x.mean(dim=0)

        return self.proj(x)  # [out_dim]


# ============ 双流融合编码器 ============
class DualStreamEncoder(nn.Module):
    """
    Token + Graph 双流编码器，通过交叉注意力融合

    创新点：不是简单拼接，而是通过交叉注意力让token表示关注结构信息、
    让结构表示关注语义信息，实现深层交互融合。
    """
    def __init__(self, use_codebert=True):
        super().__init__()
        self.use_codebert = use_codebert

        if use_codebert:
            try:
                self.token_encoder = CodeBERTTokenEncoder()
                print("成功加载CodeBERT token编码器")
            except Exception as e:
                print(f"CodeBERT加载失败({e})，使用轻量Token编码器")
                self.token_encoder = SimpleTokenEncoder()
                self.use_codebert = False
        else:
            self.token_encoder = SimpleTokenEncoder()

        self.graph_encoder = GraphStreamEncoder()

        # 交叉注意力融合层
        self.cross_attn_t2g = nn.MultiheadAttention(
            embed_dim=config.EMBEDDING_DIM, num_heads=config.NUM_HEADS,
            batch_first=True
        )
        self.cross_attn_g2t = nn.MultiheadAttention(
            embed_dim=config.EMBEDDING_DIM, num_heads=config.NUM_HEADS,
            batch_first=True
        )

        # 融合投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.EMBEDDING_DIM * 2, config.EMBEDDING_DIM),
            nn.LayerNorm(config.EMBEDDING_DIM),
            nn.GELU(),
            nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM),
        )

        # 仅Token流分支（消融实验用）
        self.token_only_proj = nn.Sequential(
            nn.LayerNorm(config.EMBEDDING_DIM),
            nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM),
        )

    def forward(self, input_ids, attention_mask, graphs, mode="dual"):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        graphs: list of (node_types, adj_matrix, num_nodes)
        mode: "dual" | "token_only" | "graph_only"
        """
        # Token流编码（批处理）
        token_repr = self.token_encoder(input_ids, attention_mask)  # [B, D]

        if mode == "token_only":
            return self.token_only_proj(token_repr)

        # Graph流编码（逐样本）
        graph_reprs = []
        for node_types, adj_matrix, num_nodes in graphs:
            gr = self.graph_encoder(node_types, adj_matrix, num_nodes)  # [D]
            graph_reprs.append(gr)
        graph_repr = torch.stack(graph_reprs, dim=0)  # [B, D]

        if mode == "graph_only":
            return graph_repr

        # 交叉注意力融合
        # token_repr: [B, D] → [B, 1, D]  作为query
        # graph_repr: [B, D] → [B, 1, D]  作为key/value
        t2g_input = token_repr.unsqueeze(1)    # [B, 1, D]
        g2t_input = graph_repr.unsqueeze(1)    # [B, 1, D]

        t_attended, _ = self.cross_attn_t2g(t2g_input, g2t_input, g2t_input)  # [B, 1, D]
        g_attended, _ = self.cross_attn_g2t(g2t_input, t2g_input, t2g_input)  # [B, 1, D]

        t_attended = t_attended.squeeze(1)  # [B, D]
        g_attended = g_attended.squeeze(1)  # [B, D]

        # 拼接 + 投影
        fused = torch.cat([t_attended, g_attended], dim=-1)  # [B, 2D]
        output = self.fusion_proj(fused)  # [B, D]

        return output


# ============ 缺陷感知对比学习损失 ============
class DefectAwareContrastiveLoss(nn.Module):
    """
    缺陷感知对比学习损失

    创新点：
    - 正样本对：同一缺陷类型的不同代码实例（缺陷模式感知）
    - 硬负样本：语法相似但缺陷类型不同的代码
    - 与标准SimCLR/MoCo的关键区别：利用缺陷类型标签构造更有意义的正负样本
    """
    def __init__(self, temperature=config.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, defect_labels):
        """
        embeddings: [B, D]  归一化后的嵌入向量
        defect_labels: [B]   缺陷类型ID
        """
        embeddings = F.normalize(embeddings, dim=-1)
        sim = embeddings @ embeddings.T / self.temperature  # [B, B]

        # 同一缺陷类型 = 正样本对
        pos_mask = defect_labels.unsqueeze(0) == defect_labels.unsqueeze(1)  # [B, B]
        pos_mask.fill_diagonal_(False)  # 排除自身

        # 没有正样本对则返回0
        has_pos = pos_mask.sum(dim=1) > 0
        if not has_pos.any():
            return embeddings.new_tensor(0.0, requires_grad=True)

        # 数值稳定性
        logits_max = sim.max(dim=1, keepdim=True).values.detach()
        sim = sim - logits_max

        # Log概率
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 对正样本对取平均log概率
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (pos_mask.float().sum(dim=1).clamp(min=1))

        loss = -mean_log_prob_pos[has_pos].mean()
        return loss


# ============ 跨模态对齐损失 ============
class CrossModalAlignmentLoss(nn.Module):
    """
    跨模态对齐损失：将buggy代码嵌入空间与fixed代码嵌入空间对齐

    创新点：类似CLIP的跨模态对齐，但用于代码修复场景——
    学习buggy代码和fixed代码之间的语义对应关系，使得相似缺陷的
    修复方案在嵌入空间中距离接近。
    """
    def __init__(self, temperature=config.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, buggy_embeddings, fixed_embeddings):
        """
        buggy_embeddings: [B, D]
        fixed_embeddings: [B, D]
        对角线上的对为正样本（同一个bug-fix对）
        """
        buggy_embeddings = F.normalize(buggy_embeddings, dim=-1)
        fixed_embeddings = F.normalize(fixed_embeddings, dim=-1)

        sim = buggy_embeddings @ fixed_embeddings.T / self.temperature  # [B, B]
        labels = torch.arange(len(sim), device=sim.device)

        loss_forward = F.cross_entropy(sim, labels)
        loss_backward = F.cross_entropy(sim.T, labels)

        return (loss_forward + loss_backward) / 2


# ============ 完整训练模型 ============
class APRContrastiveModel(nn.Module):
    """
    自动化程序修复的对比学习模型

    总损失 = L_contrastive + α * L_alignment
    - L_contrastive: 缺陷感知对比学习损失
    - L_alignment: 跨模态对齐损失
    """
    def __init__(self, use_codebert=True):
        super().__init__()
        self.encoder = DualStreamEncoder(use_codebert=use_codebert)
        self.contrastive_loss = DefectAwareContrastiveLoss()
        self.alignment_loss = CrossModalAlignmentLoss()
        self.alignment_weight = config.ALIGNMENT_WEIGHT

    def encode(self, input_ids, attention_mask, graphs, mode="dual"):
        return self.encoder(input_ids, attention_mask, graphs, mode)

    def compute_loss(self, buggy_ids, buggy_mask, buggy_graphs,
                     fixed_ids, fixed_mask, fixed_graphs,
                     defect_labels, mode="dual"):
        # 编码buggy和fixed代码
        buggy_repr = self.encode(buggy_ids, buggy_mask, buggy_graphs, mode)
        fixed_repr = self.encode(fixed_ids, fixed_mask, fixed_graphs, mode)

        # 对比学习损失（在buggy代码上计算缺陷感知对比）
        l_contrast = self.contrastive_loss(buggy_repr, defect_labels)

        # 跨模态对齐损失
        l_align = self.alignment_loss(buggy_repr, fixed_repr)

        total_loss = l_contrast + self.alignment_weight * l_align
        return total_loss, l_contrast, l_align
