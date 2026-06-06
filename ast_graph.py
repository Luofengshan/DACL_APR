"""
AST图构建模块：将Python代码解析为AST，转换为GNN可用的图结构
输出：节点类型序列 + 邻接矩阵（含父子边和兄弟边）
"""
import ast
import numpy as np
import torch

import config

# ============ AST节点类型映射 ============
# Python 3 AST节点类型 → 整数ID
AST_NODE_TYPES = sorted({name for name in dir(ast) if name[0].isupper() and isinstance(getattr(ast, name, None), type)})
NODE_TYPE_TO_ID = {name: i + 1 for i, name in enumerate(AST_NODE_TYPES)}  # 0留给padding
NUM_NODE_TYPES = len(AST_NODE_TYPES) + 1


def code_to_ast_graph(code, max_nodes=config.MAX_AST_NODES):
    """
    将代码字符串转换为AST图

    Returns:
        node_types: [N] 节点类型ID张量
        adj_matrix: [N, N] 邻接矩阵（自环+父子+兄弟）
        num_nodes: int 实际节点数
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 解析失败返回空图
        node_types = torch.zeros(max_nodes, dtype=torch.long)
        adj_matrix = torch.eye(max_nodes)
        return node_types, adj_matrix, 1

    nodes = []
    edges_parent_child = []  # (parent_idx, child_idx)
    edges_sibling = []       # (sibling_a, sibling_b)

    def _visit(node, parent_idx=None):
        idx = len(nodes)
        type_name = type(node).__name__
        type_id = NODE_TYPE_TO_ID.get(type_name, 0)
        nodes.append(type_id)

        if parent_idx is not None:
            edges_parent_child.append((parent_idx, idx))
            edges_parent_child.append((idx, parent_idx))  # 双向

        children = list(ast.iter_child_nodes(node))
        prev_child_idx = None
        for child in children:
            if len(nodes) >= max_nodes:
                break
            child_idx = len(nodes)
            _visit(child, idx)
            if prev_child_idx is not None:
                edges_sibling.append((prev_child_idx, child_idx))
                edges_sibling.append((child_idx, prev_child_idx))  # 双向
            prev_child_idx = child_idx if child_idx < len(nodes) else None

    _visit(tree)

    num_nodes = min(len(nodes), max_nodes)
    nodes = nodes[:max_nodes]

    # 构建邻接矩阵
    adj = np.eye(num_nodes, dtype=np.float32)  # 自环
    for (i, j) in edges_parent_child:
        if i < max_nodes and j < max_nodes:
            adj[i][j] = 1.0
    for (i, j) in edges_sibling:
        if i < max_nodes and j < max_nodes:
            adj[i][j] = 1.0

    # 归一化: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5, where=(degree > 0), out=np.zeros_like(degree))
    adj_normalized = degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]

    # Padding到max_nodes
    node_types = torch.zeros(max_nodes, dtype=torch.long)
    node_types[:num_nodes] = torch.tensor(nodes[:num_nodes], dtype=torch.long)

    adj_full = torch.eye(max_nodes, dtype=torch.float32)
    adj_full[:num_nodes, :num_nodes] = torch.tensor(adj_normalized, dtype=torch.float32)

    return node_types, adj_full, num_nodes
