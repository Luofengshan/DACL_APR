\# DACL-APR: 基于缺陷感知对比学习的检索增强自动化程序修复



\## 项目简介



本项目实现了一种基于深度学习的自动化程序修复（APR）方法，包含三项核心创新：



1\. \*\*缺陷感知对比学习框架\*\* — 利用缺陷类型标签构造语义有意义的正负样本对，学习缺陷模式感知的代码表示

2\. \*\*Token-Graph 双流编码器\*\* — 通过交叉注意力融合代码的序列语义（Token流）与AST结构语义（Graph流）

3\. \*\*跨模态对齐损失\*\* — 将缺陷代码空间与修复代码空间进行语义对齐（类CLIP思想）



\---



\## 环境配置



\### 硬件要求



| 配置项 | 最低要求 | 推荐配置 |

|--------|----------|----------|

| CPU | 4核 x86/ARM | 8核以上 |

| 内存 | 8 GB | 16 GB+ |

| GPU | 无（可CPU运行） | NVIDIA GPU, CUDA 11.7+, 显存 4GB+ |



\- 无GPU时自动使用CPU训练，并切换为轻量Token编码器（SimpleTokenEncoder）

\- 有GPU时自动加载CodeBERT预训练模型，效果更优



\### 软件依赖



```

Python >= 3.9

PyTorch >= 1.12.0

transformers >= 4.20.0

numpy >= 1.21.0

scikit-learn >= 1.0.0

tqdm >= 4.60.0

```



\### 安装步骤



```bash

\# 1. 进入项目目录

cd DACL\_APR\_PROJECT

\# 2. 安装核心依赖

pip install -r requirements.txt



\# 3. 如有GPU，安装对应版本的PyTorch

\# pip install torch --index-url https://download.pytorch.org/whl/cu118



\# 4. 安装transformers（用于CodeBERT支持）

pip install transformers



\# 5. 验证环境

python -c "import torch; print(f'PyTorch {torch.\_\_version\_\_}, CUDA: {torch.cuda.is\_available()}')"

python -c "from transformers import AutoModel; print('CodeBERT: OK')"

```



> 如transformers/CodeBERT安装失败，程序会自动降级为轻量Token编码器，不影响运行。



\---



\## 快速开始



\### 一键运行全部实验



```bash

python run\_all.py

```



该脚本依次执行以下5个步骤：



| 步骤 | 内容 | 产出 |

|------|------|------|

| Step 1 | 生成数据集 | `data/bug\_fix\_pairs.json` |

| Step 2 | 训练Dual模型并评估 | `checkpoints/model\_dual.pt`, `results/eval\_dual.json` |

| Step 3 | 消融实验（token\_only / graph\_only） | `results/eval\_ablation\_\*.json` |

| Step 4 | 基线对比（TF-IDF / CodeBERT无微调） | `results/baseline\_comparison.json` |

| Step 5 | 汇总所有结果 | `results/final\_report.json` |



\### 分步运行



```bash

\# 仅生成数据

python generate\_data.py



\# 训练指定模式（dual / token\_only / graph\_only）

python train.py                    # 默认dual模式，15个epoch

python -c "from train import train; train(mode='token\_only', epochs=10)"



\# 评估已训练的模型

python -c "

from train import train

from evaluate import evaluate

model, \_ = train(mode='dual', epochs=15)

evaluate(model, mode='dual')

"



\# 仅运行消融实验

python evaluate.py



```



\---



\## 代码文件结构



```

DACL\_APR\_PROJECT/

│

├── config.py              # 全局配置

├── requirements.txt       # 安装依赖

│

├── generate\_data.py       # 数据生成模块

│   ├── CORRECT\_FUNCTIONS  # 63个正确Python函数库

│   ├── 5种AST Bug注入器    # OffByOne / WrongOperator / WrongVariable

│   │                      #   / MissingCondition / MissingStatement

│   └── generate\_dataset() # 生成bug-fix对并划分训练/测试集

│

├── ast\_graph.py           # AST图构建模块

│   ├── code\_to\_ast\_graph()# 代码 → 节点类型序列 + 归一化邻接矩阵

│   └── NODE\_TYPE\_TO\_ID    # 116种AST节点类型的映射表

│

├── models.py              # 模型

│   ├── GCNLayer           # 图卷积层

│   ├── SimpleTokenEncoder # 轻量Token编码器（fallback）

│   ├── CodeBERTTokenEncoder # CodeBERT Token流编码器

│   ├── GraphStreamEncoder # GNN结构流编码器（3层GCN + 池化）

│   ├── DualStreamEncoder  # 双流融合编码器（交叉注意力）

│   ├── DefectAwareContrastiveLoss # 缺陷感知对比学习损失

│   ├── CrossModalAlignmentLoss    # 跨模态对齐损失

│   └── APRContrastiveModel # 完整训练模型

│

├── dataset.py             # PyTorch数据集

│   ├── BugFixDataset      # 加载bug-fix对，tokenize + 构建AST图

│   └── collate\_fn         # 处理变长AST图的自定义collate函数

│

├── train.py               # 训练模块

│   ├── check\_codebert\_available() # 检测CodeBERT是否可用

│   ├── get\_tokenizer()    # 获取tokenizer

│   ├── train\_one\_epoch()  # 单epoch训练循环

│   └── train()            # 完整训练流程（含checkpoint保存）

│

├── evaluate.py            # 评估模块

│   ├── encode\_all()       # 将数据集编码为向量

│   ├── evaluate\_retrieval()# 检索评估（MRR, Recall@K, nDCG@K）

│   ├── evaluate\_repair()  # 修复评估（BLEU-4, Exact Match, Top-K Hit）

│   ├── compute\_bleu()     # 简化版BLEU-4计算

│   ├── evaluate()         # 完整评估流程

│   └── ablation\_study()   # 消融实验（dual vs token\_only vs graph\_only）

│

├── run\_all.py             # 一键运行入口

│   ├── step1\_generate\_data()

│   ├── step2\_train\_and\_evaluate()

│   ├── step3\_ablation()

│   ├── step4\_baselines()  # TF-IDF + CodeBERT(no fine-tune)

│   └── step5\_final\_report()

│

├── data/                  # 自动生成的数据目录

│   └── bug\_fix\_pairs.json # 244个缺陷-修复对

│

├── checkpoints/           # 模型checkpoint目录

│   ├── model\_dual.pt

│   ├── ablation\_token\_only.pt

│   └── ablation\_graph\_only.pt

│

└── results/               # 实验结果目录

&#x20;   ├── eval\_dual.json            # Dual模型评估结果

&#x20;   ├── eval\_ablation\_\*.json      # 消融实验结果

&#x20;   ├── baseline\_comparison.json  # 基线对比结果

&#x20;   ├── train\_history\_\*.json      # 训练损失历史

&#x20;   └── final\_report.json         # 最终汇总报告

```



\---



\## 核心模型架构



```

输入: 缺陷代码 buggy\_code + 修复代码 fixed\_code



┌──────────────────────────────────────────────────┐

│                DualStreamEncoder                  │

│                                                   │

│  ┌─────────────────┐    ┌─────────────────┐      │

│  │   Token Stream   │    │   Graph Stream   │      │

│  │                  │    │                  │      │

│  │  CodeBERT /      │    │  AST Parse       │      │

│  │  SimpleEncoder   │    │  Node Embedding  │      │

│  │       ↓          │    │       ↓          │      │

│  │  h\_token \[B,D]   │    │  3× GCN Layer    │      │

│  │                  │    │       ↓          │      │

│  │                  │    │  Graph Pooling    │      │

│  │                  │    │       ↓          │      │

│  │                  │    │  h\_graph \[B,D]   │      │

│  └────────┬─────────┘    └────────┬─────────┘      │

│           │                       │                │

│           ↓                       ↓                │

│     Cross-Attention Fusion                        │

│     (双向: token→graph + graph→token)             │

│           ↓                       ↓                │

│     h\_t2g \[B,D]           h\_g2t \[B,D]            │

│           │                       │                │

│           └───────────┬───────────┘                │

│                       ↓                            │

│              Concat + MLP Projection               │

│                       ↓                            │

│              h\_fused \[B, D=128]                    │

└──────────────────────────────────────────────────┘

&#x20;                       ↓

┌──────────────────────────────────────────────────┐

│                  损失计算                          │

│                                                   │

│  L\_total = L\_contrastive + α × L\_alignment       │

│                                                   │

│  L\_contrastive: 同缺陷类型=正样本, 不同=负样本     │

│  L\_alignment:   同一bug-fix对=正样本 (类CLIP)     │

└──────────────────────────────────────────────────┘

```



\---



\## 关键超参数说明



| 参数 | 默认值 | 说明 | 修改位置 |

|------|--------|------|----------|

| `EMBEDDING\_DIM` | 128 | 最终嵌入维度 | config.py |

| `GNN\_LAYERS` | 3 | GCN层数 | config.py |

| `NUM\_HEADS` | 4 | 交叉注意力头数 | config.py |

| `BATCH\_SIZE` | 16 | 训练批大小 | config.py |

| `LEARNING\_RATE` | 2e-5 | CodeBERT层学习率 | config.py |

| `NUM\_EPOCHS` | 15 | 训练轮次 | config.py |

| `TEMPERATURE` | 0.07 | 对比学习温度 | config.py |

| `ALIGNMENT\_WEIGHT` | 0.5 | 跨模态对齐损失权重 | config.py |

| `TOP\_K` | 5 | 检索Top-K | config.py |

| `MAX\_CODE\_LENGTH` | 256 | Token最大长度 | config.py |

| `MAX\_AST\_NODES` | 128 | AST最大节点数 | config.py |

| `TEST\_RATIO` | 0.2 | 测试集比例 | config.py |

| `RANDOM\_SEED` | 42 | 随机种子 | config.py |



\---



\## 常见问题



\*\*Q: 运行报错 `No module named 'transformers'`？\*\*



A: 安装transformers：`pip install transformers`。如安装失败，程序自动降级为轻量编码器，不影响运行。



\*\*Q: CodeBERT加载失败，显示 `Could not import module 'RobertaModel'`？\*\*



A: 需要安装完整依赖：`pip install transformers torch`。部分环境下transformers与PyTorch版本不兼容，可尝试：`pip install transformers==4.30.2`。降级模式下仍可正常运行实验。



\*\*Q: GPU可用但显示 `CUDA: False`？\*\*



A: 需安装CUDA版PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cu118`（按实际CUDA版本选择）。



\*\*Q: 如何调整训练轮次和批大小？\*\*



A: 修改 `config.py` 中的 `NUM\_EPOCHS` 和 `BATCH\_SIZE`，或在调用时传参：`train(epochs=20)`。



\*\*Q: 如何在自己的数据集上运行？\*\*



A: 将数据整理为 `data/bug\_fix\_pairs.json` 格式，每条记录包含字段：`function\_name`, `buggy\_code`, `fixed\_code`, `defect\_type`, `defect\_type\_id`, `split`。

