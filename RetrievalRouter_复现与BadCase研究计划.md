# RetrievalRouter 复现与 Bad Case 研究计划

**目标**：用官方模型和逐 Query 检索结果，系统定位路由失败的规律，筛选有真实证据、可验证且有论文潜力的改进方向。**本阶段不重新训练模型、不重建全部检索索引，也不复现延迟曲线。**

## 一、复现：先获得可靠的逐 Query 结果

### Step 1. 克隆官方仓库并准备评估环境

```bash
git clone https://github.com/emrekuruu/retrieval-router.git
cd retrieval-router
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements/evaluation_requirements.txt
```

关键文件：
- `evaluation/predictions/retrievalrouter_l10.xlsx`：λ=0.1，主分析对象（8,506 条测试 Query）。
- `evaluation/predictions/retrievalrouter_l00.xlsx`、`retrievalrouter_l70.xlsx`：分析不同训练偏好下相同 Query 的决策差异。
- `evaluation/predictions/baseline.xlsx`：对比已有路由 Baseline。
- `train/data/updated_dataset.xlsx`：包含各检索 Pipeline 的逐 Query 评测数据。
- `arms.py`：**唯一合法的 Pipeline 名称与顺序**，分析脚本直接导入，避免映射错误。

### Step 2. 先用官方预测文件复现聚合检索质量（无需 GPU）

```bash
python -m evaluation.overall_metrics \
  --predictions evaluation/predictions/retrievalrouter_l10.xlsx \
  --output evaluation/metrics_summary.xlsx
```

验收：**nDCG@5 ≈ 0.754739**；测试 Query 计数 **8,506**。这里是重新统计作者已生成的预测结果，**不是重新运行检索器**。随后检查 Excel 的列名、空值、重复样本与数据集分布。

### Step 3. 拉取官方 λ=0.1 Checkpoint（用于独立推理核验；不是分析的前置条件）

- 模型地址：<https://huggingface.co/emrekuruu/RetrievalRouter-lambda-l10>
- 官方全部 Checkpoint：<https://huggingface.co/collections/emrekuruu/retrieval-router>

```bash
# 新建一个推理环境；不要依赖评估环境恰好兼容 Transformers
python -m venv .venv-infer
source .venv-infer/bin/activate
pip install -r requirements/training_requirements.txt
pip install -U huggingface_hub

# 官方 λ=0.1 模型（约 2.4 GB），含自定义模型代码
hf download emrekuruu/RetrievalRouter-lambda-l10 \
  --local-dir checkpoints/l10
```

**仅在需要独立核验输出或预测概率缺失时，才运行推理**。官方 `train/inference.py` 本身会保存 `pred_prob_<arm>` 等概率列；先检查发布的 Excel 是否已经有它们。若已有，则连 Checkpoint 都可以暂不下载。

如需重新推理：

```bash
# 先备份官方预测文件，避免被 inference.py 同名覆盖
mkdir -p analysis/original
cp evaluation/predictions/retrievalrouter_l10.xlsx analysis/original/

# 按官方固定随机种子重新生成同一测试划分
OBJECTIVE=retrievalrouter LAMBDA=0.1 python -m train.prepare_data

# 验证测试 Query 集合与原预测文件一致后，再运行
ROUTER_MODEL_ID=checkpoints/l10 OBJECTIVE=retrievalrouter LAMBDA=0.1 \
  python -m train.inference
```

验收：Query 一一对应；`predicted_pipeline` 大致一致；取得 5 维 `pred_prob_<arm>`（注意模型输出类别顺序来自 `arms.py`，**不能自行改成 BM25 在前**）。独立推理产生的文件仍须保留为单独副本，与官方结果对照。

## 二、建立正确的 Bad Case 判据

对五个可选臂 `A = {BM25, TD, TR, MD, MR}`，使用 Excel 中真实的逐 Query 指标，自己重新计算以下量，不直接把预测文件中名为 `predicted_ndcg` / `regret` 的派生列当作原始 nDCG（在该项目中它们可能对应 reward 矩阵）：

- **检索质量损失**：`quality_regret = max_i(ndcg_i) - ndcg_pred`。
- **与原论文训练目标一致的路由损失**：`r_i = 0.9 * ndcg_i + 0.1 * (1 - latency_i / sum_j latency_j)`；`reward_regret = max_i(r_i) - r_pred`。
- **最优差距**：`margin = 最大 reward - 第二大 reward`，识别近似并列的情况。
- **决策置信度**：`max(pred_prob_*)`；可配合概率熵。统计“高置信度 + 高 reward_regret”的严重错误。
- **五路 Oracle**：只在上述五个可选臂内取最大值；论文包含 TL、ML 的七路 Oracle 是另外一个上界，不能混为一谈。

先按 **高 reward_regret / 高 quality_regret** 找严重错误，再把“全部检索器都差”的样本单独分离。**仅因 `predicted_pipeline != argmax(reward)` 不应判定为严重错误**：接近并列时可能几乎没有实际损失。

## 三、全方位 Bad Case 分析（按顺序执行）

| 分析轴 | 具体统计 / 抽样方法 | 主要要验证的问题 |
|---|---|---|
| **A. 整体错误结构** | reward_regret、quality_regret 分布；各分位点、累计损失贡献；严重错误 Top 10% | 是否存在少量样本贡献了绝大部分性能差距？ |
| **B. 错选到哪里** | `实际选择 × 五路最优选择` 混淆矩阵；同时统计错误次数、平均损失、累计损失 | 最严重错误是否集中在 BM25↔神经检索、TD↔MD、Dense↔Rerank 等具体转换？ |
| **C. 领域与 Query 特征** | 按 `DATASET`、Query 长度、数字/实体/精确匹配、图表指代、比较推理、多个约束等分组；做样本量和损失对比 | Router 是否依赖领域关键词捷径，而非真实检索需求？ |
| **D. 模态判断** | 对比 `{BM25,TD,TR}` 与 `{MD,MR}` 的最优效果；找出偏向文本/视觉但选错的 Query | “需要视觉”的判断是否真由 Query 决定？还是文档内容决定？ |
| **E. 架构判断** | 同 Query 配对计算 `TR−TD`、`MR−MD` 的 nDCG 差；区分提升/无提升/恶化 | Rerank 什么时候有害？什么时候仅加成本没有增益？ |
| **F. 置信度与可辨性** | 置信度–regret 散点图、严重错误率随置信度变化、margin 分层；重点抽“高置信度严重错误” | Router 是不会判断，还是不知道自己判断错了？ |
| **G. 不同 λ 的决策稳定性** | 将 l00/l10/l70 文件按 `(DATASET, Query及必要ID)` 严格对齐，观察同 Query 选臂如何变化；**在同一个固定指标下**比较各版本 | 某些 Query 是否对训练目标特别敏感？不要把目标改变导致的不同选择误记为错误。 |
| **H. 无法靠路由解决的失败** | 计算五臂 `max(ndcg)`；单列全部低分样本；另看高分可得却未选到的样本 | 当前瓶颈究竟是 Router，还是所有 Retriever 都无法找到证据？ |
| **I. 数据/监督噪声** | 检查相同或近似 Query 的标签冲突、并列最优、原始标注和页面关联 | 是否存在信息不足、数据标注或 Reward 定义造成的“伪错误”？ |

**第一轮产出**：`analysis/query_metrics.csv`（逐 Query 分数、选择、概率、regret、margin）、`analysis/slice_summary.csv`（分组统计）、`analysis/confusion.csv`、`analysis/top_badcases.csv`。统计结果必须同时列出**样本数、平均损失、累计损失**，避免凭少数个例推断普遍规律。

## 四、从统计走向机制：人工复核 50–100 条典型样本

从下列群体**分层抽样**，而非只看损失最大的前 100 条：
1. 高置信度、高 regret；
2. 文本/多模态误选；
3. Dense/Rerank 误选，尤其是 Rerank 导致 nDCG 下降；
4. 跨数据集共同出现的相似错误；
5. 多臂效果接近但被判“错误”的对照样本；
6. 五臂全部失败的对照样本。

每个 Case 记录：**原始 Query → 数据集 / 相关文档页 → Router 的五维概率 → 五个 Pipeline 的 nDCG → 选中臂 / 最优臂 → 失败假设 → 支撑证据 → 可验证的干预**。

特别注意：官方 Excel 主要提供**逐 Query 指标**，不足以解释“哪张相关页面为何掉出 Top-K”。若要确认某类原因，需去 Hugging Face 对应数据集查找相关页面，必要时仅对**选定的几十个 Case**重跑目标 Pipeline 并保存 Top-K 页面 / 分数 / 重排前后顺序，而不必重跑全量索引。

## 五、重点验证的研究假设（不是预设创新点）

| 假设 | 最小验证实验 | 什么时候值得继续 |
|---|---|---|
| **H1：Query-only 存在信息论层面的歧义** | 搜索相同/相似 Query 在不同文档上的最优臂冲突；人工核查页面结构；仅在必要时增加文档布局或一次廉价探测信号 | 大量高损失样本无法凭 Query 区分；加入真实文档信号能定向纠错且成本可控 |
| **H2：领域词汇是虚假的模态捷径** | 领域分层 + 视觉关键词匹配分析；必要时做 Query 关键词扰动或留出一个数据集训练/测试 | 错误系统性集中在领域转移，且模型决策对无关词异常敏感 |
| **H3：Dense→Rerank 的增益具有可预测的失败模式** | 找到 `MR<MD` 和 `TR<TD` 的高损失样本，复核其候选页面与重排序 | 能解释某类候选结构导致 Rerank 恶化，并找到可泛化的轻量决策信号 |
| **H4：置信度与真实效用严重脱钩** | 对比高置信度错误、低置信度正确；按置信度或风险排序看实际 regret | 普通 Softmax 置信度不能识别高风险路由，而新的不确定性证据可显著改善 |
| **H5：Soft Label 混淆“几乎无所谓”和“严重选错”** | 统计奖励 margin 与损失贡献、分组的训练目标分布 | 大量可避免损失集中在当前训练目标没有有效强调的样本，并能提出有机制依据的学习策略 |

**优先级**：先看 **模态需求歧义（H1）**、**Rerank 负增益（H3）**、**领域捷径（H2）**，因为论文自身已给出相应现象或局限；其余由数据决定。若只有简单改 Loss、加阈值或多拼接一些特征，不要直接视为顶会级创新。

## 六、研究结论与下一步的门槛

形成一份 `analysis/failure_report.md`，只保留三个最重要的、经过证据核实的失败模式。每个模式必须写清楚：

1. **现象**：错误占比、累计 regret、典型 Query；
2. **原因**：原始 Query / 相关页面 / 各 Pipeline 的证据，排除标签噪声与模型差异混杂；
3. **突破口**：提出能修复该机制的最小实验，与成本相当的简单 Baseline 比较；
4. **新颖性核查**：检索该机制是否已有相关工作，不仅看 RetrievalRouter 原文引用；
5. **泛化验证**：在独立未用于挑选想法的数据上复核，而不是只优化看过的坏例。

**最终交付**：`query_metrics.csv` + `slice_summary.csv` + `top_badcases.csv` + `casebook.md`（人工样本） + `failure_report.md`（Top-3 可研究问题与最小实验）。

**实验纪律**：当前官方测试集只用于**探索和生成假设**。后续确定创新方案后，应另划开发/验证/最终测试集合，或在新的数据集上验证；不要对这 8,506 个已看过的测试 Query 反复调参再用它们证明泛化。跨域留一数据集实验需要**额外训练**，但不属于当前零训练第一阶段。

---

## 参考来源

- 论文：<https://arxiv.org/abs/2608.25625>（方法、数据、实验与 Limitations）
- 官方 GitHub：<https://github.com/emrekuruu/retrieval-router>（README、`arms.py`、`train/inference.py`、`evaluation/overall_metrics.py`、`train/prepare_data.py`）
- 官方 Checkpoints：<https://huggingface.co/collections/emrekuruu/retrieval-router>
