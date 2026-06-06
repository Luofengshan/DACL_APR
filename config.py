import os
import torch

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_FILE = os.path.join(DATA_DIR, "bug_fix_pairs.json")

# ============ 模型配置 ============
CODEBERT_MODEL = "microsoft/codebert-base"
EMBEDDING_DIM = 128          # 最终嵌入维度
CODEBERT_HIDDEN = 768        # CodeBERT输出维度
GNN_HIDDEN = 128             # GNN隐藏层维度
GNN_LAYERS = 3               # GNN层数
NUM_HEADS = 4                # 交叉注意力头数
MAX_CODE_LENGTH = 256        # token最大长度
MAX_AST_NODES = 128          # AST最大节点数

# ============ 训练配置 ============
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
WARMUP_RATIO = 0.1
TEMPERATURE = 0.07           # 对比学习温度
ALIGNMENT_WEIGHT = 0.5       # 跨模态对齐损失权重

# ============ 数据配置 ============
NUM_FUNCTIONS = 80           # 生成的基础函数数量
BUGS_PER_FUNCTION = 4        # 每个函数的bug变体数量
TEST_RATIO = 0.2             # 测试集比例
RANDOM_SEED = 42

# ============ 检索配置 ============
TOP_K = 5                    # 检索Top-K相似样本

# ============ 设备配置 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
