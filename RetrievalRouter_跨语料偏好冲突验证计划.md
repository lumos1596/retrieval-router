# RetrievalRouter：跨语料 Retriever 偏好冲突验证计划（先诊断，暂不训练）

## 1. 核心假设与目标

**待验证假设 H：** 对于具有相似*检索需求特征*的 Query，不同语料库的 Retriever 相对优势可能发生系统性变化。只使用 Query 的原始 RetrievalRouter 容易把源域中的偏好规律误当作普遍规律，导致目标语料库上的路由损失。

本轮只回答两件事：
1. **偏好冲突是否真实、显著且普遍？** 在可比较的 Query 类型内，各语料库上哪些检索器更优，是否出现有意义的优势翻转？
2. **这种冲突与跨库路由失败是否关联？** 相比偏好稳定的 Query 类型，偏好冲突型 Query 是否表现出更高的目标域 Routing Regret？

**本轮不训练模型、不跑新检索器、不做 GRPO / Group DRO。** 直接利用已经计算好的逐 Query 五臂 `nDCG@5` 和 Query-only Router 预测结果。

## 2. 数据准备与隔离

必需字段：`query_id, dataset/corpus_id, query, split, ndcg_BM25, ndcg_TD, ndcg_TR, ndcg_MD, ndcg_MR, predicted_arm`。

- 优先使用原文的完整 11 数据集逐 Query 五臂指标表（如 `train/data/updated_dataset.xlsx`）及当前重训 Router 的预测表。**目前的 `xdomain_badcases.csv` 只有极端样本，不能用于总体偏好比例和统计检验。**
- 第一轮可分析现有源域 FinQA / ConvFinQA / VQAonBD 与目标域 TAT-DQA / Wiki-SS / ArxivQA，**但这三份目标数据已经被多轮查看，属于探索性诊断集，不得冒充最终盲测集。**
- **重要限制：** 三个金融训练语料库可能无法覆盖百科和科学问题，也可能共享底层财报。必须核查文档重叠。如果可行，从余下数据集中加入不同类型的源语料，并预先锁定至少两个尚未用于调参的目标语料库作后续独立验证。
- 先检查各臂字段、Query ID、缺失值、文档重复；不要用 `best_arm` 单一标签替代五维 nDCG（并列最优较多）。

## 3. 步骤 A：定义可比较的 Query 需求类型（不使用检索效果标签）

**为什么不直接用语义 KNN？** 上一轮 Local Probe 已经证明“文本语义接近”不足以预测 Retriever 优势。这里的分组用于诊断*条件偏好变化*，不是直接拿邻居路由。

对每条 Query，按只使用 Query 文本的规则或冻结的统一标注器提取粗粒度特征：

1. **信息操作：** 事实定位 / 比较或关系 / 数字计算 / 开放式描述。
2. **精确约束：** 是否包含年份、数值、明确实体、多个实体或多个约束。
3. **明确视觉指示：** 是否显式提到 table、figure、chart、page、image 等。

先使用低维、可解释分组（例：`数值计算+含年份`、`事实查询+无视觉指示`），避免组合得太细。每组在**至少2个语料库各有 ≥50 条 Query**才进入主要跨库对比；不足的组标记 `insufficient support`，不强行得出偏好冲突结论。随机人工检查每组约20条，验证分类含义一致。

同时记录：目标库 Query 里，有多少属于源域根本没有或极少出现的分组（`out-of-support`）。这部分代表 **Query 分布外推问题**，不能直接当作偏好冲突。

## 4. 步骤 B：直接计算 Retriever 优势是否跨库翻转

对每条 Query，以*同一 Query 上两个检索臂的真实效果差*衡量相对优势（可控制该 Query 的绝对难度）：

\[
\delta_{MR,MD}(q)=nDCG_{MR}(q)-nDCG_{MD}(q)
\]
\[
\delta_{TR,TD}(q)=nDCG_{TR}(q)-nDCG_{TD}(q)
\]
\[
\delta_{V,T}(q)=\max(nDCG_{MR},nDCG_{MD})-\max(nDCG_{TR},nDCG_{TD},nDCG_{BM25})
\]

最后一个量为**事后分析用的模态族潜在优势**，并非推理时已知信息。

对每个 `Query 类型 z × 语料库 C` 计算上述三个 Δ 的均值、样本数和 Query 级 bootstrap 95%置信区间；制作 `类型 × 语料库` 的优势热力图。

**冲突判定建议（诊断阈值，而非统计定理）：** 同一类型内，某两语料库的平均 Δ 一正一负，且两边绝对值都 ≥0.05 nDCG；再检查置信区间与稳健性。对小组、置信区间跨零的情况标记 `uncertain`，不要声称存在真正冲突。主要分析预先确定的3个对比，减少事后挑选“显著”组合。

还需记录每类Query的五臂最高分、并列比例及最佳/次佳差距，以防“最优臂标签切换”其实只是多个Retriever同样好。

## 5. 步骤 C：确认偏好冲突是否对应实际跨库路由错误

对每条目标域 Query，用现有 Router 预测臂计算：

\[
Regret(q)=\max_a nDCG_a(q)-nDCG_{\hat a(q)}(q)
\]

按**仅使用源域数据定义的类别与偏好**，划分目标 Query：

- **未覆盖（OOD）：** Query类型在源域缺少足够样本；
- **覆盖且偏好稳定：** 源域与目标域该类Query的主要Retriever优势方向一致；
- **覆盖但偏好冲突：** 优势方向发生明确翻转；
- **不确定：** 数据不足或多臂近似并列。

注意：这里目标域真实 nDCG 只用于**事后分析分组和判别假设**，绝不能用于训练Router或部署时决定路线。

比较四组的：
1. 平均 Routing Regret、严重掉零率（`oracle ≥ 0.75 & chosen = 0`）；
2. Router 的选臂分布与该组五臂平均收益；
3. 分组后的样本量及数据集构成；
4. 控制 `dataset`、Query长度、约束数量、Oracle质量/可选臂收益差距后的Regret差异（简单分层对比或回归即可）。

最关键的是区分：**是未见Query类型太多（coverage shift），还是已有类型的Retriever优势真的反转（conditional preference shift）？** 不能把二者合并成“偏好冲突”。

## 6. 输出表与最小图表

交付三个文件：

1. `preference_shift_cells.csv`：`query_type, corpus, n, mean_delta_MR_MD, mean_delta_TR_TD, mean_delta_V_T, 95%CI, tie_rate`。
2. `preference_shift_vs_regret.csv`：每组 `n, avg_regret, catastrophic_rate, pred_arm_distribution` 与控制变量分析结果。
3. `preference_shift_report.md`：**一张偏好差值热力图 + 一张四组Regret对比图 + 6–10个真实Query示例**。真实例子必须核对 Query 类型与五臂nDCG，避免只拿极端个例叙事。

## 7. 是否进入新训练方法的决策门槛

- **结果A：同类型跨库偏好反转明显，且冲突组比稳定组有更大Regret。** 支持继续研究 *Preference Stability-Aware Training*：让训练对跨域稳定优势强化学习，对冲突偏好降低过度确信/采用风险感知的效用优化。下一轮再和原文KL、效用回归、均衡采样、Group DRO比较。
- **结果B：主要是新域Query类型在训练中缺席，稳定/冲突组差异不明显。** 核心瓶颈更像是训练分布覆盖不足；优先尝试域多样性和数据组织，不要强行讲“偏好冲突”。
- **结果C：大多Query五臂近似并列或冲突不稳定。** 单纯根据最优臂统计构造冲突是假象；本方向暂停。

**提醒：** 即使发现稳定的条件偏好反转，也只能证明`Query特征无法完全解释Retriever收益`这一观察；对于在输入上不可区分、但最优臂相反的情况，任何纯Query-only训练方法都无法完全消除误差。若希望实现超过这种信息上限的路由，后续必须添加有效的环境/语料信息。

## 8. 最小执行顺序

**第1步（数据清洗与分组）→ 第2步（3个优势差的跨库热力图）→ 第3步（稳定/冲突/OOD组的Regret对照）→ 第4步（决定是否训练）。**

这次实验的成功标准**不是必须证明偏好冲突存在**，而是明确区分“Query覆盖不足”和“条件Retriever偏好发生变化”两类可检验解释。
