# RetrievalRouter 失败模式报告（零训练第一阶段）

- **对象**：官方 λ=0.1 模型，8,506 条测试 Query（`evaluation/predictions/retrievalrouter_l10.xlsx`，与 l00/l70/baseline 逐行对齐）
- **复现验收**：重算 routed nDCG@5 = **0.754739**（与 README 一致）；五臂 Oracle = 0.901424，七臂 Oracle（含 TL/ML，不可路由）= 0.910466
- **指标口径**：所有 regret 均由原始 `<arm>_ndcg` / `<arm>_latency` 重算。官方文件中的 `predicted_ndcg`/`regret`/`true_ndcg_<arm>` 对 retrievalrouter 目标实际是 **reward 矩阵**的派生列（已逐元素验证差为 0，见 `train/inference.py: eval_matrix → REWARD_COLS`），本报告不将其当作 nDCG
- **Reward**：r_i = 0.9·nDCG_i + 0.1·(1 − latency_i / Σ_j latency_j)（五臂内归一化，与 `train/prepare_data.py` 重算误差 <1e-16）
- **臂顺序**（`arms.py`，checkpoint 类别序）：MR, MD, TR, TD, BM（MULTIMODAL_RERANK, MULTIMODAL-SINGLE, TEXT_RERANK, TEXT-SINGLE, BM25）
- **产物**：`query_metrics.csv`（60 列×8506 行）、`slice_summary.csv`、`confusion.csv`、`top_badcases.csv`（Top-200）、`casebook.md`（63 个分层 Case）、`analysis_output.txt`（全量打印）

> **重要的计数纪律**：下列三类集合（rerank 有害、模态误判、灾难性掉零）相互重叠，百分比**不可相加**；它们是同一批严重错误的三种切片视角。总 quality_regret = 1247.7。

---

## 全局画面（轴 A/H/F）

| 事实 | 数值 |
|---|---|
| quality_regret = 0 的 Query | 6,246（73.4%）；reward_regret = 0 为 2,668 |
| Top 10% Query 占累计 quality_regret / reward_regret | **57.7% / 53.9%**；Top 20% 占 87.0% / 82.0% |
| 严重错误（reward_regret ≥ P90=0.493） | 851 条（10%） |
| **灾难性掉零**（五臂 Oracle ≥0.75 但选中臂 nDCG=0） | **519 条（6.1%），贡献 41.6% 的 quality_regret** |
| 高顶但大错（Oracle≥0.75 且 qr≥0.25） | 1,775 条，贡献 **85.4%** 的 quality_regret |
| 五路全失败（Oracle<0.25） | 371 条（4.4%），仅贡献 **0.1%** 的 regret——瓶颈在 Router，不在"无检索器能找到证据" |
| 高置信（≥0.50）严重错误 | 仅 17 条；置信度与 regret 相关系数 ≈ **−0.05**，十个置信度十分位的严重错误率在 8.5–12.8% 间几乎持平 |
| 模型置信度整体偏低 | 中位 0.30、P99 0.45、max 0.80（五类问题随机水平 0.20） |

结论先行：**路由差距高度集中、且绝大多数发生在"明明有臂能答对（Oracle≈1）却选成 0 分"的 Query 上**；模型不是"在两个差不多的答案间犹豫"，而是"平静地（低区分度置信度）选了一个彻底失败的臂"。

---

## 失败模式 1（最高优先级，H3）：MM-Rerank 相对 MM-Dense 的负增益不可从 Query 预测，是最大的单一损失来源

### 现象
- Router 选中 Rerank、而其 Dense 版本 nDCG 高出 >0.02 的 Query 有 **690 条，累计 quality_regret 415.9（33.3%）**，平均每条 0.603；其中 **547 条是 MR 选中而 MD 更好（320.3）**，143 条 TR 选中而 TD 更好（95.6）。344 条进入全局严重错误集合。
- 在 851 条严重错误的"选中×reward最优"表中，**MR↔MD 互错占 436 条（51.2%）**：MR→MD 170 条、MD→MR 115 条（另含 BM/TR 混合）。
- 对偶面：Dense 选中但 Rerank 本可增益 >0.02 的有 782 条（435.6）。说明问题不是"总选错 rerank"，而是**rerank 增益方向这一判断本身不稳定**。
- 数据集集中：tatdqa 265、wiki-ss 128、arxivqa 99（前三位占 71%）。典型形态是金融表格 Query：MR=0 而 MD=1.0——正确页在 Dense Top-5 首位，经 cross-encoder 重排后**直接掉出 Top-5**（见 casebook S3a 全部 8 例，如"What was the net change in cash for Republic Services in 2016…"）。

### 原因（证据强度与待排除混杂）
- **数值证据（已核实）**：rerank 有害选择的平均损失 0.60 远高于全局 0.147；MR↔MD 混淆解释了一半严重错误；损失集中在数字/表格密集域。
- **机制假设（需页面级核实）**：候选页含密集数字、长实体名、渲染表格 OCR 文本时，cross-encoder 的语义匹配把正确的数字证据页误降权。这是**候选集结构决定的失败**，而 Router 只看 Query 文本，信息论上无法预知这次 rerank 会不会把正确页挤出 Top-K。
- **待排除混杂**：① 标签噪声——qrels 仅 5 深，nDCG 呈离散值（0/0.146/0.387/0.5/0.631/1），需检查 MR=0/MD=1 案例中正确页在 MR 排序第 6–20 位的比例（若是稳定第 6–8 位，属于截断位置问题而非语义错杀，干预完全不同）；② 多模态管线本身的索引/OCR 差异而非 rerank 阶段——需对比 MD Top-20 与 MR Top-5 的实际页面列表（casebook S3a 已列 8 个待核案例）。

### 突破口：最小实验
1. 仅对 S3a 分层的 ~20–30 个案例重跑 MD 检索（Top-20）+ MR 重排，保存重排前后排名与页面类型（正文/原生表格/扫描表格/图表），确认"掉位 vs 未召回"。
2. 构造**候选集侧的轻量 rerank 开关**（不重训 Router）：Dense 返回后用 top-20 分差、top5 数字 token 密度、表格/OCR 覆盖率、Query 实义项在候选页的词项覆盖率等特征，训练一个二分类 gate，决定是否调用 rerank。
3. 对照基线（成本相当）：永远 rerank、永不 rerank、按全局固定比例随机 gate、仅用 Query 特征的 gate。**验收必须同时减少 S3a（有害）和 S3b（漏增益）两类**，报告净 nDCG 与延迟。

### 新颖性核查
- "何时该 rerank"本身已有先例：**ARDS（ScienceTePress 2026, paper 148030）**报告有害 rerank 可降 ~25%；UMass 的 **Adaptive Re-Ranking（arXiv:2606.25249）**用轻量分类器在 BM25/dense/rerank 间按成本路由；**Drowning in Documents（arXiv:2411.11767）**记录 cross-encoder 扩大候选数反而伤 Recall。
- 因此"加一个 rerank 开关"不构成新意。**可能的新意空间**：(a) 上述工作均为纯文本域，本文的多模态表格/扫描件候选结构是新场景；(b) 若页面级证据能把失败归约为可预测的"数字证据误降权"机制并给出可泛化的候选集信号（而非再训练一个黑盒分类器），才有论文价值。
- 反向证据：**Beyond the Reranker（arXiv:2606.28367）**发现可实现的 per-query router 打不过固定最优策略——实验设计必须包含足够强的固定基线。

### 泛化验证
当前 8,506 条只用于挑假设。Gate 定型后在**留出 1–2 个数据集**（建议留 tatdqa + arxivqa，二者是损失最集中且风格最不同的域）上评估，并另切开发/验证/测试 Query。

---

## 失败模式 2（高优先级，H2 + H1 载体歧义）：模态需求双向误判，伴随领域/指代词捷径

### 现象
- 按"视觉臂{MD,MR}最佳 vs 文本臂{BM,TD,TR}最佳（nDCG 差 >0.02）"定义，**模态误判 591 条（6.9%），累计 quality_regret 299.8（24.0%）**：
  - 真正需要视觉却选了文本：118 条，平均 qr 0.560（sum 66.1）。集中于 finslides/finreport/convfinqa——扫描或渲染的财务表格页，文本索引里没有证据（如"What is the main goal of the IBM Watson Group?"MR=MD=1 而全部文本臂=0）。
  - **文本足够却过度选视觉：473 条，平均 qr 0.494（sum 233.7），是主方向**。含事实性 wiki 问题（"what is the si unit of temperature"选 MR=0.63 而 BM=1.0）与含 "figure" 指代的 arxivqa 问题（答案在正文/标题，视觉编码器反而失败）。
- **捷径证据**：Query 含 figure/chart/table/graph/page 等词的 826 条（9.7%）被路由到视觉臂的比例 **95.4%（基线 73.1%）**，但其中仅 27.8% 真的视觉臂显著更好，65.3% 是模态并列区，6.9% 文本更好。
- **词法机会损失**：BM25 nDCG≥0.75 但没被选且 qr≥0.5 的有 279 条（sum qr 212.7），wiki-ss 86、tatdqa 73、arxivqa 43。
- 域偏置：reward_regret 占比 tatdqa 34.3%、wiki-ss 21.1%、arxivqa 12.2%；严重错误率 wiki-ss/arxivqa 13.4%、tatdqa 12.2%，而金融表格域 convfinqa/finslides 仅 4% 左右——错误随域显著漂移，论文 Limitations 亦自承跨域泛化有限。

### 原因（证据强度与待排除混杂）
- 已核实：模态误判的损失规模、figure 词的过触发、BM25 在事实型 Query 上被系统性忽略。
- 机制假设分两层：(a) **H2 虚假捷径**——模型用领域词（财务术语）和指代关键词（figure/table）代替真实模态需求判断；(b) **H1 信息论歧义**——同一句 Query 的证据载体（正文文字 vs 页内表格图片）由**文档侧**决定，Query-only 无法区分；5,901 条 Query 处于模态并列区（69%），其中 180 条"顶部并列却选了第三个极差臂"进入严重错误（sum reward_regret 144.8），这是 H1 的强形式证据。
- 待排除：页面级核查 S2a/S2b 共 12 个案例，确认视觉失败页是否确为图片表格（而非 OCR 管线 bug），以及 "figure" 类 Query 的证据是否在 caption/正文。

### 突破口：最小实验
1. **关键词扰动实验（零训练，立即做）**：对含 figure/table 词的 Query 增删该词，观测路由翻转率与 nDCG 变化；对财务 Query 做实体名替换。翻转率高即捷径实锤。
2. **留一数据集（LODO）重训**是论文自己指出的额外训练项，不属于本零训练阶段，但应作为 H2 的决定性验证写入计划。
3. **廉价探测信号**（与 FM1 共用一套 cascade 架构）：先跑最便宜的 BM25/Dense，用返回页的版面类型分布（图片表格占比、OCR 覆盖率）决定是否升级到视觉/rerank 臂——把"Query-only 一次定终身"改成"Query + 一次廉价文档探测"。

### 新颖性核查
- Query 分类路由本身是 RetrievalRouter 的选题；AgentIR（arXiv:2605.25092）已用 BM25 top-k margin 做级联触发，说明"先廉价探测再升级"的范式有先例。
- 新意必须落在**多模态文档（扫描表格/图表）的载体判断机制**上：若关键词扰动 + LODO 证明模型依赖的是可定位、可去除的虚假线索，且一个版面探测特征能定向修复 S2a/S2b 双向错误，则形成"捷径诊断→机制修复"的完整故事；单纯加 OCR 覆盖率特征不够。

### 泛化验证
留出 wiki-ss（文本事实型）和 finslides（视觉扫描型）两个极端域；在新 Query 上同时报告两个方向的错误，避免只修一边。

---

## 失败模式 3（中优先级，H4 + H5）：置信度无风险辨识力，训练目标不区分"无所谓"与"彻底选错"，系统缺少安全网

### 现象
- Softmax 最大概率与 regret 的相关系数 −0.05（近乎无关）；各置信度十分位严重错误率 8.5–12.8% 持平，最高十分位反而最低（6.7%），**不存在用置信度排序识别高风险路由的能力**；高置信严重错误虽只有 17 条（模型全局欠自信，max 仅 0.80），但低置信（<0.30，占半数 Query）中仍有 9.2% 严重错误。
- 519 条"Oracle≥0.75→选中 0 分"的灾难性错误分散在全部五个臂（MR 174、MD 159、BM 95、TR 59、TD 32）和 11 个数据集，是 FM1/FM2 之外的横切视角：**即使选错，系统也没有任何兜底信号**。
- 训练目标侧：76.1% 的 Query nDCG 最优 margin<0.01（34.2% reward margin 同此），argmax 不一致但实际无害（regret<0.01）的"伪错误"占全部臂分歧的 15.2%（889 条）；而 180 条顶部并列、却选到极差第三臂的 Query（sum rr 144.8）与无害并列在当前 soft-target KL（T=0.1）下得到近乎相同的训练信号。
- 附注（延迟侧，不影响 nDCG）：365 条五臂全 0 分 Query 中 276 条被路由给最贵的 MR——λ=0.1 下"廉价失败"也没学会，属于另一个可量化的次要点。

### 原因
- 五分类 softmax 概率在软标签 KL 训练下退化为近似均匀分布（中位 max prob 0.30），既无校准也无排序能力；这不是温度标定能修好的形态。
- soft label 把"两臂并列、选哪个都对"和"最优臂明确、却选了 0 分臂"压成相近的目标分布；大 margin 灾难性样本在 8 万条训练数据中没有被有效强调。

### 突破口：最小实验
1. 先做**免训练诊断**：用 FM1 的候选集信号 + 各检索器返回分数做 risk score（QPP 式），画 risk–coverage 曲线：按风险排序时，拒绝/升级尾部 Query 能否消除远高于均匀比例的 regret。
2. 训练侧最小改动对照：margin 加权/focal 式 KL（加大 margin、大 oracle 样本权重）vs 原 KL；在固定开发集上用 519 条灾难性集合的召回率作为专门指标。
3. 系统侧：与 FM1/FM2 的探测 cascade 合并——低风险单臂直出，高风险先 Dense 探测再决定升级/回退 BM25，与成本相当的"永远跑两臂"基线比较。

### 新颖性核查
- RAG 领域的置信度弃判已有强先例：**The Magnitude Mirage（arXiv:2609.15578）**明确指出绝对分数阈值无效、方差类信号更好；**BalanceRAG（arXiv:2605.20084）**做风险校准级联；SABER 等自感知 RAG 亦成系列。
- 单纯"校准 softmax / 加阈值"无新意。可辩护的切入点是：**多臂路由设定下，风险信号必须来自候选集而非路由器自身 logits**，且把 FM1（rerank 掉位风险）、FM2（载体歧义风险）统一为同一个"先探测、风险触发升级"的决策问题——这是现有单通道弃判文献没有覆盖的。

### 泛化验证
风险阈值只在开发集上标定一次，跨数据集直接迁移测试；报告 risk–coverage 下的 nDCG/延迟曲线而非单点。

---

## H1–H5 假设处置结论

| 假设 | 数据裁决 | 下一步 |
|---|---|---|
| H1 Query-only 信息论歧义 | **部分支持**：强形式仅 180 条（顶部并列却选错）+118 条视觉载体歧义；并列区多数 Query 损失很小 | 页面级核查 S2a 后再定优先级 |
| H2 领域/词汇虚假捷径 | **支持**：figure 词 95.4% 过触发、域间严重率差 3 倍、BM25 机会损失 279 条 | 关键词扰动（零训练）先行，LODO 重训练列入下一阶段 |
| H3 Rerank 负增益有可预测失败模式 | **最强支持**：33.3% 的 quality_regret、51% 的严重错误 | **立即做**：20–30 案例的 Top-K/重排顺序复核 + 候选集 gate |
| H4 置信度与效用脱钩 | **支持但形态是"全局欠自信+无排序力"**，高置信错误仅 17 条 | 转向候选集风险信号，不做 softmax 重标定 |
| H5 Soft label 混淆并列与严重错误 | 支持：15.2% 分歧无害、180 条并列区严重错误信号被稀释 | 作为 FM3 的训练侧对照实验，不单独立项 |
| 噪声 I（标签/重复） | 全量 85,094 条 Query 无一条完全重复；并列是主要噪声源而非标注冲突 | 评估改用期望 regret，argmax accuracy 仅作参考 |

## 实验纪律与下一阶段建议

1. 本阶段全部结论基于 8,506 条测试 Query，**只用于生成假设**；任何 gate/加权/阈值的调参必须切新划分或新数据集。
2. 建议的下一步顺序（成本递增）：① 13 个页面级 Case 复核（casebook 末尾已给清单）→ ② 关键词扰动与 QPP 风险曲线（免训练）→ ③ 候选集 gate 离线实验（不需重训 Router）→ ④ 仅在 ①–③ 为正时，才规划 LODO 重训与 7B 模型可行性评估。
3. Step 3 的 Checkpoint 独立推理核验**本阶段未执行**：发布 Excel 已含完整 `pred_prob_<arm>` 五列（概率和=1、argmax 与 predicted_label 完全一致），按计划"连 Checkpoint 都可以暂不下载"；后续若页面级复核发现概率异常再补做。
