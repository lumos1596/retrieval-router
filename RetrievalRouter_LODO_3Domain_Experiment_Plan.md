# RetrievalRouter 跨语料泛化（LODO）实验计划

> **本轮目标**：只复现与验证原始 Query-only RetrievalRouter 在**未见过的数据集/语料域**上的泛化表现。**不训练语料画像模型、不做多模态重新索引、不跑完整 11 折。**
>
> 原始代码：https://github.com/emrekuruu/retrieval-router （以实际克隆到的版本为准，记录 commit hash）

## 1. 要回答的研究问题

1. 训练时完全不见某个数据集，原始 Router 的检索质量是否明显下降？
2. 性能下降集中在哪些选择错误：**文本 vs 多模态**、**Dense vs Rerank**、**BM25 被忽视**？
3. 原始模型是否利用了数据集/领域的词汇捷径，而不是可靠地判断检索能力需求？

**限制**：LODO 只能证明跨数据集泛化是否困难，不能单独证明问题根源是“没有感知语料库”，因为 Query 分布也随数据集变化。语料因果验证留到下一轮“同 Query、不同证据载体”配对实验。

## 2. 实验范围及固定配置

原始 11 个数据集，每折**完整留出 1 个数据集**，使用其他 10 个数据集训练一个 Router。本轮做 3 折：

| 实验 | 完全留出的数据集 | 代表性文档类型 | 源训练数据 |
|---|---|---|---|
| Fold 1 | Wiki-SS | 通用知识 / 网页截图 | 其余 10 个数据集 |
| Fold 2 | FinSlides | 财务幻灯片 / 图表 | 其余 10 个数据集 |
| Fold 3 | ArxivQA | 学术论文 / 公式图表 | 其余 10 个数据集 |

- 训练：**Qwen/Qwen3-0.6B-Base + LoRA**，沿用官方 `train/config.py` 的学习率、batch、梯度累积、最大长度、2 epoch、温度 `T=0.1` 等参数。
- 损失：原版 RetrievalRouter soft-target KL，`OBJECTIVE=retrievalrouter`，`LAMBDA=0.1`；不改算法。
- 硬件：先在单张 **RTX 3090 24GB** 跑一折，记录实测速度。此前讨论的“约 1–2 小时/折”仅为粗估，不是官方 3090 基准。
- 主指标：**路由实际选中 arm 的原始 nDCG@5**，并单列 reward、延迟、质量 regret；不要混用。
- 五个可选 arm：**MR、MD、TR、TD、BM**（顺序读取官方 `arms.py`，不得自行重排）。

## 3. 数据准备（最关键：避免跨域泄漏）

官方 `train/data/updated_dataset.xlsx` 已包含每条 Query 在五条检索管线上的 nDCG 与 latency，无需重新跑全部检索器。

1. 克隆官方仓库，安装 `requirements/training_requirements.txt`；记录 Git commit、Python/PyTorch/Transformers 版本和随机种子。
2. 按官方 `train/prepare_data.py` 中的算法，对 **11 个数据集分别独立**重现原始固定种子划分：每域约 80% train、10% val、10% test（`SAMPLE_RATE=0.1`、`RANDOM_SEED=42`；注意整数取整方式）。
3. 对每一折：
   - `train` = 其余 10 域的**原始 train 部分**；
   - `val` = 其余 10 域的**原始 val 部分**（只用于选 checkpoint）；
   - `test_ood` = 留出域的**原始 test 部分**（与官方预测对应，便于逐 Query 对照）；
   - 留出域的 train/val/test **均不得参与本折训练或选模**；留出域非 test 的样本可以保留在单独封存文件中，主评测先不用。
4. 保存三份独立的 `train/val/test` Excel（含原始 Query、`DATASET`、五臂 nDCG/latency、官方 objective 所需的 reward 列）；每折通过唯一 `RUN_TAG` 隔离数据、模型、日志及预测结果。
5. 检查：各集合无相同原始行/Query ID 交叉；训练/验证 `DATASET` 不含留出域；测试只含留出域；统计每折样本量并保存 `split_manifest.json`。对跨域近重复 Query 做简单检查并报告，不直接静默删除样本。

**实现要求**：新增 `experiments/lodo/prepare_lodo.py`，尽量复用官方 `load_and_split_data()`、`calculate_rewards()` / objective 的 `add_targets()`、`save_datasets()`；**不要**对完整 Excel 简单过滤掉留出域后再直接运行原始 `prepare_data.py`，那样得到的 test 会变成其他 10 域，无法完成 OOD 评测。

## 4. 训练与推理

1. 先跑 **Wiki-SS 单折**。可参考官方环境安装与启动方式：

   ```bash
   git clone https://github.com/emrekuruu/retrieval-router.git
   cd retrieval-router
   pip install -r requirements/training_requirements.txt

   # 自行新增的 LODO 数据准备与本地训练/评测驱动脚本（非官方已有命令）
   python experiments/lodo/prepare_lodo.py --heldout wiki-ss --lambda 0.1
   CUDA_VISIBLE_DEVICES=0 python experiments/lodo/run_fold.py --heldout wiki-ss --lambda 0.1
   ```

2. 训练驱动使用官方 `train.train.main(paths=(train_path, val_path, test_path), output_dir=...)`，且在**同一进程**用返回的 `model, tokenizer` 调用官方 `train.inference.run_inference()` 或 `evaluate_model()`；不用上传 Hugging Face 才能评测。
3. 保存逐 Query 五维预测概率、选中 arm、五臂原始 nDCG/latency、被选 arm 原始 nDCG/latency、Oracle 最优 nDCG 和 regret，并保留检查点、训练日志。注意：官方 `evaluate_model()` 的 `predicted_ndcg` 在 RetrievalRouter objective 下实际基于 **reward matrix**，因此最后用原始 `<arm>_ndcg` 列**另行重算真实 nDCG**。
4. Wiki-SS 无异常后，再运行 `finslides` 和 `arxivqa`。如并行多卡，检查官方 `train/train.py` 内的 `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` 硬编码，移除或改为仅由 shell 设置，并检查输出目录隔离。
5. 如果 3090 显存不足，先降低 per-device batch 并增大 gradient accumulation 保持相近有效 batch；变更全部记录在 `run_config.json`，不要悄悄改变训练目标。

## 5. 评测与最小分析

对每个留出域分别输出：

| 对照 | 目的 |
|---|---|
| 原版官方 λ=0.1 Checkpoint 在**相同留出域 test Query** 上的 nDCG@5 | “见过该域训练 Query”参考，不是完全同条件的公平上界 |
| 该域 LODO 模型在相同 test Query 上的 nDCG@5 | 本轮核心测量 |
| 各固定 arm 的 nDCG@5（BM/TD/TR/MD/MR） | 确认 LODO 是否能超过简单固定策略 |
| 每条 Query 取五臂最高 nDCG 的 Oracle | 估计可选择性的上限（不代表可部署方案） |

统计并可视化：

- `nDCG@5`、质量 regret = `max(ndcg_arms) - ndcg_selected`，以及 reward regret 与延迟（分列）。
- **灾难性错路由**：Oracle nDCG≥0.75，但选中 arm nDCG=0 的占比。
- 三类错误切片：`文本↔多模态`、`Dense↔Rerank`、`BM25 本可更优却未选`；同一 Query 可落入多个切片，**不要把比例直接相加**。
- 每折保存 20–30 条高质量 regret 的 Case，记录 Query、五臂分数、官方模型与 LODO 模型选择及概率；涉及“为什么错”时必须拉实际检索页面核实，不能仅凭分数推断 OCR / reranker 原因。
- 对每折 test Query 与原版模型做 **paired bootstrap** 差值置信区间；三个域单独报告，不只汇总均值。特别提示：FinSlides 的测试集比较小，结论更依赖置信区间。

## 6. 结论判定与下一步决策

- **若 3 个域均明显下降**，且比固定强 arm 还差：跨语料泛化确实存在严重问题，优先分析模态偏好漂移。
- **若只有 1–2 个域下降**：按文档结构与 Query 形式做切片，识别出现问题的条件，不笼统宣称整个路由方法无法泛化。
- **若 LODO 下降很小 / 优于强固定基线**：说明原 Router 的跨域能力不弱，不建议直接押注“语料库画像”作为论文主线；进一步做同 Query、不同载体的反事实实验再判断。
- **无论结果如何，LODO 都不是语料感知机制的充分证据**。下一轮只有在存在清晰问题时，才开展：同一 Query 保持信息需求不变、改变证据载体或语料组织，并验证最优 arm 的真实翻转。
- **如果 3 折有稳定现象，再跑剩余 8 折**，并预留与 Query-only Router + 简单语料统计特征基线对比；本轮不训练新方法。

## 7. 最终交付物

```text
experiments/lodo/
├── prepare_lodo.py
├── run_fold.py
├── analyze_lodo.py
└── results/
    ├── wiki-ss/{split_manifest.json,run_config.json,metrics.json,predictions.csv,badcases.csv,train.log}
    ├── finslides/{...}
    ├── arxivqa/{...}
    └── summary.md
```

`summary.md` 只需汇报：**三折核心指标对比表、LODO 与官方同域测试差距、主要错路由类型及具体案例、是否值得进入语料库画像研究**。必要时附测试集结果与置信区间，不需要延迟曲线、复杂调参和新模型消融。

---

**核对依据（官方代码）：**
- [README：数据集、官方基线、训练命令](https://github.com/emrekuruu/retrieval-router)
- [`train/prepare_data.py`：固定分层切分及 reward 构造](https://github.com/emrekuruu/retrieval-router/blob/master/train/prepare_data.py)
- [`train/train.py`：模型训练、返回本地模型](https://github.com/emrekuruu/retrieval-router/blob/master/train/train.py)
- [`train/inference.py`：模型推理与指标注意事项](https://github.com/emrekuruu/retrieval-router/blob/master/train/inference.py)
