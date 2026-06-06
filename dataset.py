"""
PyTorch Dataset：加载bug-fix代码对，进行tokenize和AST图构建
"""
import json
import torch
from torch.utils.data import Dataset

import config
from ast_graph import code_to_ast_graph


class BugFixDataset(Dataset):
    def __init__(self, data, tokenizer=None, split="train"):
        """
        data: list of dicts with buggy_code, fixed_code, defect_type_id
        tokenizer: transformers tokenizer
        split: "train" or "test"
        """
        self.data = data
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        buggy_code = item["buggy_code"]
        fixed_code = item["fixed_code"]
        defect_type_id = item["defect_type_id"]

        # Tokenize
        if self.tokenizer is not None:
            buggy_enc = self.tokenizer(
                buggy_code, max_length=config.MAX_CODE_LENGTH,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            fixed_enc = self.tokenizer(
                fixed_code, max_length=config.MAX_CODE_LENGTH,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            buggy_ids = buggy_enc["input_ids"].squeeze(0)
            buggy_mask = buggy_enc["attention_mask"].squeeze(0)
            fixed_ids = fixed_enc["input_ids"].squeeze(0)
            fixed_mask = fixed_enc["attention_mask"].squeeze(0)
        else:
            # Fallback: 简单字符级tokenize
            buggy_ids = self._simple_tokenize(buggy_code)
            buggy_mask = (buggy_ids != 0).long()
            fixed_ids = self._simple_tokenize(fixed_code)
            fixed_mask = (fixed_ids != 0).long()

        # AST图构建
        buggy_graph = code_to_ast_graph(buggy_code)
        fixed_graph = code_to_ast_graph(fixed_code)

        return {
            "buggy_ids": buggy_ids,
            "buggy_mask": buggy_mask,
            "buggy_graph": buggy_graph,
            "fixed_ids": fixed_ids,
            "fixed_mask": fixed_mask,
            "fixed_graph": fixed_graph,
            "defect_label": torch.tensor(defect_type_id, dtype=torch.long),
        }

    def _simple_tokenize(self, code):
        """简单的字符级tokenize（无tokenizer时的fallback）"""
        tokens = code.encode("utf-8")
        token_ids = [min(b, 255) + 1 for b in tokens[:config.MAX_CODE_LENGTH]]
        token_ids = token_ids + [0] * (config.MAX_CODE_LENGTH - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch):
    """自定义collate函数，处理变长AST图"""
    buggy_ids = torch.stack([b["buggy_ids"] for b in batch])
    buggy_mask = torch.stack([b["buggy_mask"] for b in batch])
    fixed_ids = torch.stack([b["fixed_ids"] for b in batch])
    fixed_mask = torch.stack([b["fixed_mask"] for b in batch])
    defect_labels = torch.stack([b["defect_label"] for b in batch])

    # 图数据保持为list（变长）
    buggy_graphs = [b["buggy_graph"] for b in batch]
    fixed_graphs = [b["fixed_graph"] for b in batch]

    return {
        "buggy_ids": buggy_ids,
        "buggy_mask": buggy_mask,
        "buggy_graphs": buggy_graphs,
        "fixed_ids": fixed_ids,
        "fixed_mask": fixed_mask,
        "fixed_graphs": fixed_graphs,
        "defect_labels": defect_labels,
    }


def load_data(data_file=config.DATA_FILE, split=None):
    """加载数据"""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if split is not None:
        data = [d for d in data if d["split"] == split]
    return data
