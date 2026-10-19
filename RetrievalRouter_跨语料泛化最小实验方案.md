# RetrievalRouter 跨语料库泛化：最小验证实验方案

> **目标**：只训练 **1 个 Router（λ=0.1）**，判断原始 Query-only Routing 从金融语料迁移到未见语料库后是否明显失效，从而决定是否值得继续研究 **Corpus-aware / Capability-Fingerprint Routing**。本轮**不做新模型、不重建检索器索引、不做11轮LODO**。

## 1. 实验划分（1次训练，4组测试）

| 数据集 | 领域 | 本轮用途 | 测试问题 |
|---|---|---|---|
| FinQA、ConvFinQA、VQAonBD | 金融 | **训练源**：各自按80%/10%/10%划分训练/验证/源域测试 | 已见分布上是否正常工作？ |
| TAT-DQA | 金融 | **完全留出**：只测试，不参加训练和模型选择 | 同领域、未见语料库能否泛化？ |
| Wiki-SS | 通用知识 | **完全留出**：只测试，不参加训练和模型选择 | 金融→通用知识能否泛化？ |
| ArxivQA | 科学论文 | **完全留出**：只测试，不参加训练和模型选择 | 金融→科学论文能否泛化？ |

- 三个目标数据集均与训练集**按数据集隔离**；目标测试Query可使用官方原始数据中固定种子42抽取的10%子集，以控制评估工作量。目标数据集余下Query也不得进入训练/验证；后续若需要，可在不调整方案的前提下扩大到全量评估。
- 检查训练与留出数据集底层PDF/文档ID是否重叠，记录去重结果；若重叠严重，不能将相应结果称为严格“未见语料库”。
- 训练集只取三个金融数据集；**验证集也只能取这三个来源**。禁止根据TAT-DQA、Wiki-SS、ArxivQA结果调λ、挑Epoch或改阈值。

## 2. 代码准备与数据划分

```bash
git clone https://github.com/emrekuruu/retrieval-router.git
cd retrieval-router
python -m venv .venv-training
source .venv-training/bin/activate
pip install -r requirements/training_requirements.txt
```

- 原始数据：`train/data/updated_dataset.xlsx`，读取 `DATASET`、`query`、五条可选Pipeline的 `*_ndcg`/`*_latency` 列。
- 新增 `experiments/prepare_xdomain.py`，**按上表指定数据集生成划分**，不要直接使用原 `python -m train.prepare_data`：该脚本默认把全部11个数据集都分进训练集，会造成目标集泄漏。
- 保存源域 `train/val/test` 为官方 `train.train` 可读的Excel格式，并在每个文件中调用原版 `RetrievalRouterObjective.add_targets(df)` 生成Reward列；目标域测试文件也按同样方式补齐Reward，仅供**事后评估**。
- 设置独立的 `RUN_TAG=xc_fin3_l10`，保留原文件，避免覆盖已复现的官方结果。

**特别注意**：官方Router类别次序固定为 `MR, MD, TR, TD, BM25`（对应代码 `arms.py`），计算Oracle和Regret必须按此顺序读取原始列。

## 3. 模型训练（只训练1个）

- 基座：`Qwen/Qwen3-0.6B-Base`，重新初始化LoRA与分类头，**不能从官方已在11个数据集上训练的Router Checkpoint继续微调**。
- 保持原文超参数：`λ=0.1`、LoRA rank=16、max_length=128、2 epochs、batch=16、grad_accumulation=2、AdamW；Seed=42。
- 新增入口 `experiments/run_xdomain.py`：传入自定义的三份数据路径给 `train.train.main(paths=(train_path,val_path,source_test_path))`，取得训练好的 `model, tokenizer`，同一进程中调用 `train.inference.run_inference(...)` 对4组测试数据推理，并导出每条Query的五臂概率与所选Pipeline。
- 这比训练结束后直接执行官方 `python -m train.inference` 更稳妥：**官方默认从Hugging Face加载模型，而不是加载本地新训练的LoRA目录**。
- 预计单张RTX 3090 24GB即可完成；耗时以实际机器为准。无须训练其他λ。

## 4. 评估（只保留研究问题所需指标）

对**源域测试集以及三个留出目标集分别**计算：

1. **Router nDCG@5**：每条Query读取选中Pipeline的真实 `*_ndcg` 再求平均。
2. **五臂Oracle nDCG@5**：每条Query取五条Pipeline的最高nDCG再求平均（仅作可达上界，不可部署）。
3. **Quality Regret**：`max_a nDCG_a(q) - nDCG_chosen(q)` 的均值；重点统计 `Oracle≥0.75 且 chosen=0` 的严重掉零比例。
4. **固定检索器基线**：分别列出BM25、TD、TR、MD、MR的nDCG；注明“测试集最强固定臂”属于**事后比较**，不是可在线选择的方法。
5. **错误类型**：文本↔视觉误路由、Dense↔Rerank误路由，各类数量与累计Quality Regret（切片可能重叠）。

输出一张简洁主表：

| 测试集 | Router nDCG | Oracle nDCG | 平均Regret | 最强固定检索器nDCG | 严重掉零率 |
|---|---:|---:|---:|---:|---:|
| 金融源域测试 | 待填 | 待填 | 待填 | 待填 | 待填 |
| TAT-DQA | 待填 | 待填 | 待填 | 待填 | 待填 |
| Wiki-SS | 待填 | 待填 | 待填 | 待填 | 待填 |
| ArxivQA | 待填 | 待填 | 待填 | 待填 | 待填 |

**避免错误结论**：不同语料库本身难度不同，不要只比较源域与目标域nDCG绝对值；优先比较 **平均Regret**、相对Oracle的差距，以及是否输给简单固定检索器。由于本轮训练样本少于原论文，还不能把劣化**完全归因于缺少Corpus画像**。

## 5. 最小坏例分析及决策

- 对每个目标集挑 **Top-20 Quality Regret**，保存Query、模型五维概率、所选臂、最佳臂、各臂nDCG以及错误类型。
- 判断错误是否集中于某类Corpus：例如在Wiki-SS持续偏选多模态、在ArxivQA持续误判Dense/Rerank；不要仅依据领域名称解释错误原因。
- 比较**目标库中各Retriever的平均表现分布**与源域分布是否变化；这是“语料库检索能力画像”可能有效的直接线索，但不等于因果证明。

**是否继续研究Corpus-aware Routing：**

- **继续**：至少两个未见目标集出现有意义的Regret增长或明显低于固定检索器，并且错误模式与Retriever能力分布变化一致。
- **暂缓**：只在单个目标集变差、没有一致的路由失败规律，或5臂Oracle本身就很低。此时优先补一组反方向迁移验证，而非直接设计新模型。
- 若结果正面，**下一步再用目标语料库少量Query的真实标签模拟“能力画像”做机制上界实验**；这只是验证信息是否有用，不得当作无标签零样本方法。上界有效后才投入无标签合成探针。

## 6. 交付文件

- `xdomain_predictions.csv`：四组测试Query、预测概率、五臂真实nDCG及Regret。
- `xdomain_metrics.md`：上面的主结果表 + 3条主要观察。
- `xdomain_badcases.csv`：每个留出集Top-20错误及错误类别。
- `xdomain_conclusion.md`：**是否出现可复现跨库退化、是否值得进入语料库画像实验**（不预设正结论）。

## 参考代码（已核对）

- [官方仓库与数据](https://github.com/emrekuruu/retrieval-router)
- [train/prepare_data.py：默认逐数据集划分，必须改为跨库隔离](https://github.com/emrekuruu/retrieval-router/blob/master/train/prepare_data.py)
- [train/train.py：`main(paths=...)`支持自定义文件路径](https://github.com/emrekuruu/retrieval-router/blob/master/train/train.py)
- [train/inference.py：`run_inference`可复用，默认CLI依赖HF模型ID](https://github.com/emrekuruu/retrieval-router/blob/master/train/inference.py)
- [arms.py：五臂名称与顺序](https://github.com/emrekuruu/retrieval-router/blob/master/arms.py)
