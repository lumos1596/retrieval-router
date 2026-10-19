# RetrievalRouter Bad Case 人工复核 Casebook（λ=0.1，8,506 条测试 Query）

> 生成方式：`analysis/build_casebook.py` 基于 `query_metrics.csv` **分层随机抽样（seed=42）**，共 63 个 Case，覆盖计划第四节的 6 类群体。每个 Case 的五维概率/五臂 nDCG/regret/margin 均直接取自官方预测文件并经脚本重算。
>
> **复核状态**：以下“失败假设”是基于逐 Query 数值证据的机器初判；标注“需要”的 Case 必须按计划第四节去 Hugging Face 对应数据集拉取相关页面、必要时仅对选定 Case 重跑目标 Pipeline 保存 Top-K 页面/重排前后顺序，才能把假设升级为已核实结论。

## S1. 高置信度 + 高 regret（8 例；全量仅 17 例，均在 `top_badcases.csv`）

#### S1-6480 `dude` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: Whats the total of outstanding bonds?
- DATASET: `dude`; qlen=6 words
- probs: MR=0.582 MD=0.196 TR=0.169 TD=0.029 BM=0.024
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=1.000 BM=0.631
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.945
- confidence=0.582, reward margin=0.002; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.58 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.95
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-5822 `arxivqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What is the relation between the parameter 'ω' and the trajectory behavior in figure (a)?
- DATASET: `arxivqa`, features: figure-word, comparison; qlen=15 words
- probs: MR=0.501 MD=0.178 TR=0.172 TD=0.067 BM=0.082
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=0.631 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.941
- confidence=0.501, reward margin=0.328; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.50 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.94
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-2998 `tatdqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What was the percentage change in Net deferred tax assets from 2018 to 2019?
- DATASET: `tatdqa`, features: number, comparison; qlen=14 words
- probs: MR=0.514 MD=0.201 TR=0.195 TD=0.055 BM=0.035
- nDCG@5: MR=0.000 MD=1.000 TR=0.387 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.932
- confidence=0.514, reward margin=0.591; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.51 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.93
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-3931 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **MR**

- **Query**: What are the respective equipment notes payable at 2018 and 2019?
- DATASET: `tatdqa`, features: number; qlen=11 words
- probs: MR=0.256 MD=0.511 TR=0.101 TD=0.046 BM=0.085
- nDCG@5: MR=0.631 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.539
- confidence=0.511, reward margin=0.526; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.51 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.54
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-5868 `arxivqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What variable does 'δ' represent on the y-axis of figures (a) and (c)?
- DATASET: `arxivqa`; qlen=14 words
- probs: MR=0.572 MD=0.093 TR=0.201 TD=0.060 BM=0.074
- nDCG@5: MR=0.000 MD=0.631 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.592
- confidence=0.572, reward margin=0.551; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.57 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.59
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-2196 `tatdqa` — picked **MR**, nDCG-best **TR**, reward-best **TR**

- **Query**: What was the percentage change in interest income for Cornerstone OnDemand Inc. between 2018 and 2019, as reported in the Other Income (Expense) section of the 2019 financial report?
- DATASET: `tatdqa`, features: number, comparison; qlen=29 words
- probs: MR=0.539 MD=0.145 TR=0.181 TD=0.119 BM=0.016
- nDCG@5: MR=0.387 MD=0.000 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.387, 5-arm oracle=1.000, quality_regret=0.613, reward_regret=0.565
- confidence=0.539, reward margin=0.565; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.54 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.56
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-4498 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: In which years is the Other reserves and accruals greater than 5,000 thousand?
- DATASET: `tatdqa`, features: number; qlen=14 words
- probs: MR=0.272 MD=0.520 TR=0.095 TD=0.057 BM=0.056
- nDCG@5: MR=1.000 MD=0.000 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.882
- confidence=0.520, reward margin=0.010; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.52 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.88
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S1-6238 `arxivqa` — picked **MR**, nDCG-best **TR**, reward-best **TR**

- **Query**: Which statement best describes the color mapping in the figures?
- DATASET: `arxivqa`; qlen=10 words
- probs: MR=0.533 MD=0.079 TR=0.264 TD=0.065 BM=0.058
- nDCG@5: MR=0.000 MD=0.000 TR=0.631 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.575
- confidence=0.533, reward margin=0.534; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。
- **数值证据**: 置信度 0.53 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret=0.57
- **可验证干预**: 在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S2a. 真正需要视觉却路由到文本臂（6 例 / 全量 118 例，累计 qr=66.1）

#### S2a-7781 `wiki-ss` — picked **TR**, nDCG-best **MD**, reward-best **MD**

- **Query**: name the city and country in which the taj mahal is located
- DATASET: `wiki-ss`; qlen=12 words
- probs: MR=0.243 MD=0.147 TR=0.261 TD=0.189 BM=0.160
- nDCG@5: MR=0.000 MD=1.000 TR=0.500 TD=0.500 BM=0.000
- picked nDCG=0.500, 5-arm oracle=1.000, quality_regret=0.500, reward_regret=0.509
- confidence=0.261, reward margin=0.451; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 1.00 vs 文本臂最佳 0.50；实际选中 TR=0.50
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2a-7443 `wiki-ss` — picked **TR**, nDCG-best **MR**, reward-best **MR**

- **Query**: when was the first bank of united states formed by the federalists
- DATASET: `wiki-ss`; qlen=12 words
- probs: MR=0.256 MD=0.133 TR=0.321 TD=0.126 BM=0.164
- nDCG@5: MR=1.000 MD=0.631 TR=0.631 TD=0.000 BM=0.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.303
- confidence=0.321, reward margin=0.303; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 1.00 vs 文本臂最佳 0.63；实际选中 TR=0.63
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2a-860 `convfinqa` — picked **BM**, nDCG-best **MR**, reward-best **MR**

- **Query**: What was the increase in the value of free cash flow from $1382 million in 2012 to $2085 million in 2013 for Union Pacific Corporation?
- DATASET: `convfinqa`, features: number, comparison; qlen=25 words
- probs: MR=0.217 MD=0.218 TR=0.174 TD=0.154 BM=0.237
- nDCG@5: MR=1.000 MD=0.000 TR=0.000 TD=0.500 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.856
- confidence=0.237, reward margin=0.413; lambda arms l00/l10/l70 = MR/BM/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 1.00 vs 文本臂最佳 0.50；实际选中 BM=0.00
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2a-5452 `arxivqa` — picked **BM**, nDCG-best **MD**, reward-best **MD**

- **Query**: Which component in the setup is responsible for synchronizing the photon pairs?
- DATASET: `arxivqa`; qlen=12 words
- probs: MR=0.203 MD=0.251 TR=0.142 TD=0.132 BM=0.272
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=0.500 BM=0.431
- picked nDCG=0.431, 5-arm oracle=1.000, quality_regret=0.569, reward_regret=0.476
- confidence=0.272, reward margin=0.420; lambda arms l00/l10/l70 = MR/BM/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 1.00 vs 文本臂最佳 0.50；实际选中 BM=0.43
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2a-8214 `wiki-ss` — picked **BM**, nDCG-best **MR**, reward-best **MR**

- **Query**: when was i dont wan na miss a thing released
- DATASET: `wiki-ss`; qlen=10 words
- probs: MR=0.151 MD=0.162 TR=0.191 TD=0.207 BM=0.288
- nDCG@5: MR=1.000 MD=0.000 TR=0.631 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.867
- confidence=0.288, reward margin=0.335; lambda arms l00/l10/l70 = BM/BM/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 1.00 vs 文本臂最佳 0.63；实际选中 BM=0.00
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2a-7271 `wiki-ss` — picked **TR**, nDCG-best **MD**, reward-best **MD**

- **Query**: when did the french come to north america
- DATASET: `wiki-ss`; qlen=8 words
- probs: MR=0.258 MD=0.122 TR=0.301 TD=0.122 BM=0.198
- nDCG@5: MR=0.000 MD=0.500 TR=0.387 TD=0.387 BM=0.000
- picked nDCG=0.387, 5-arm oracle=0.500, quality_regret=0.113, reward_regret=0.101
- confidence=0.301, reward margin=0.087; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。
- **数值证据**: 视觉臂最佳 0.50 vs 文本臂最佳 0.39；实际选中 TR=0.39
- **可验证干预**: 增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S2b. 文本即可却过度路由到视觉臂（6 例 / 全量 473 例，累计 qr=233.7）

#### S2b-5536 `arxivqa` — picked **MR**, nDCG-best **BM**, reward-best **BM**

- **Query**: Based on the graphs, how does an increase in \( \varepsilon_{ls} \) affect the velocity profile in the x direction?
- DATASET: `arxivqa`, features: comparison; qlen=19 words
- probs: MR=0.319 MD=0.275 TR=0.139 TD=0.115 BM=0.152
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=1.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.941
- confidence=0.319, reward margin=0.905; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 事实性/概念性 Query 被误送入视觉臂；词法匹配本可命中。
- **数值证据**: 文本臂最佳 1.00 vs 视觉臂最佳 0.00；实际选中 MR=0.00
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2b-395 `finqa` — picked **MD**, nDCG-best **TD**, reward-best **TD**

- **Query**: What was the change in asset retirement obligations between 2002 and 2003 for AES Corporation, as reported in their 2003 financial statements?
- DATASET: `finqa`, features: number, comparison; qlen=22 words
- probs: MR=0.216 MD=0.266 TR=0.165 TD=0.229 BM=0.124
- nDCG@5: MR=0.000 MD=0.000 TR=0.431 TD=0.631 BM=0.500
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.568
- confidence=0.266, reward margin=0.108; lambda arms l00/l10/l70 = MR/MD/BM
- **失败假设**: 领域/指代词触发视觉捷径（如出现 figure/财务术语被当作文档视觉需求）；实际证据在正文文本，视觉编码器引入噪声。
- **数值证据**: 文本臂最佳 0.63 vs 视觉臂最佳 0.00；实际选中 MD=0.00
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2b-4670 `tatdqa` — picked **MD**, nDCG-best **TD**, reward-best **TD**

- **Query**: What was FormFactor's net income for the fiscal year ended December 28, 2019, as reflected in the cash flows from operating activities?
- DATASET: `tatdqa`, features: number; qlen=23 words
- probs: MR=0.242 MD=0.285 TR=0.149 TD=0.215 BM=0.110
- nDCG@5: MR=0.631 MD=0.000 TR=0.631 TD=1.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.898
- confidence=0.285, reward margin=0.365; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 领域/指代词触发视觉捷径（如出现 figure/财务术语被当作文档视觉需求）；实际证据在正文文本，视觉编码器引入噪声。
- **数值证据**: 文本臂最佳 1.00 vs 视觉臂最佳 0.63；实际选中 MD=0.00
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2b-5008 `mp-docvqa` — picked **MD**, nDCG-best **BM**, reward-best **BM**

- **Query**: how many no of shares are held in the life insurance corporation of india ?
- DATASET: `mp-docvqa`; qlen=14 words
- probs: MR=0.212 MD=0.264 TR=0.166 TD=0.188 BM=0.171
- nDCG@5: MR=0.631 MD=0.631 TR=0.631 TD=0.000 BM=1.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.347
- confidence=0.264, reward margin=0.347; lambda arms l00/l10/l70 = MD/MD/BM
- **失败假设**: 事实性/概念性 Query 被误送入视觉臂；词法匹配本可命中。
- **数值证据**: 文本臂最佳 1.00 vs 视觉臂最佳 0.63；实际选中 MD=0.63
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2b-3854 `tatdqa` — picked **MR**, nDCG-best **TD**, reward-best **BM**

- **Query**: What was the difference in the net increase (decrease) in cash and cash equivalents between 2017 and 2018 for Adobe Systems Inc.?
- DATASET: `tatdqa`, features: number, comparison; qlen=22 words
- probs: MR=0.387 MD=0.177 TR=0.252 TD=0.131 BM=0.053
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.387 BM=0.387
- picked nDCG=0.000, 5-arm oracle=0.387, quality_regret=0.387, reward_regret=0.387
- confidence=0.387, reward margin=0.020; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 领域/指代词触发视觉捷径（如出现 figure/财务术语被当作文档视觉需求）；实际证据在正文文本，视觉编码器引入噪声。
- **数值证据**: 文本臂最佳 0.39 vs 视觉臂最佳 0.00；实际选中 MR=0.00
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S2b-2112 `vqaonbd` — picked **MD**, nDCG-best **TR**, reward-best **BM**

- **Query**: What is the cumulative value of purchases of available-for-sale investments, sales and maturities of available-for-sale investments, sale of equity method investment, acquisition of Micrel net of cash acquired, purchase of additional controlling interest in ISSC, investments in other assets, proceeds from sale of assets, and capital expenditures that contributed to net cash used in or provided by investing activities for Microchip Technology in 2016?
- DATASET: `vqaonbd`, features: number; qlen=68 words
- probs: MR=0.220 MD=0.267 TR=0.159 TD=0.130 BM=0.224
- nDCG@5: MR=0.500 MD=0.631 TR=1.000 TD=0.431 BM=1.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.334
- confidence=0.267, reward margin=0.038; lambda arms l00/l10/l70 = MD/MD/BM
- **失败假设**: 事实性/概念性 Query 被误送入视觉臂；词法匹配本可命中。
- **数值证据**: 文本臂最佳 1.00 vs 视觉臂最佳 0.63；实际选中 MD=0.63
- **可验证干预**: 做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S3a. 选择 Rerank 但 Rerank 相对 Dense 显著恶化（8 例 / 全量 690 例，累计 qr=415.9）

#### S3a-6829 `sciqag` — picked **TR**, nDCG-best **MD**, reward-best **MD**

- **Query**: How does the aggregation behaviour of ionic liquids affect their applications?
- DATASET: `sciqag`; qlen=11 words
- probs: MR=0.261 MD=0.182 TR=0.308 TD=0.125 BM=0.125
- nDCG@5: MR=0.000 MD=1.000 TR=0.387 TD=0.631 BM=0.631
- picked nDCG=0.387, 5-arm oracle=1.000, quality_regret=0.613, reward_regret=0.570
- confidence=0.308, reward margin=0.318; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: TR=0.39 而 TD=0.63（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-2899 `tatdqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What is the percentage change in Cash and Investments between 2016 and 2017?
- DATASET: `tatdqa`, features: number, comparison; qlen=13 words
- probs: MR=0.441 MD=0.239 TR=0.223 TD=0.047 BM=0.050
- nDCG@5: MR=0.631 MD=1.000 TR=0.631 TD=0.000 BM=0.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.353
- confidence=0.441, reward margin=0.353; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: MR=0.63 而 MD=1.00（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-4445 `tatdqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What is the difference in the total assets between Level 1 and Level 2 as of December 31, 2019?
- DATASET: `tatdqa`, features: number, comparison; qlen=19 words
- probs: MR=0.459 MD=0.288 TR=0.140 TD=0.065 BM=0.048
- nDCG@5: MR=0.000 MD=0.631 TR=0.000 TD=0.500 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.599
- confidence=0.459, reward margin=0.128; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: MR=0.00 而 MD=0.63（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-8381 `wiki-ss` — picked **TR**, nDCG-best **MD**, reward-best **BM**

- **Query**: who created the american image of santa claus
- DATASET: `wiki-ss`, features: figure-word; qlen=8 words
- probs: MR=0.209 MD=0.153 TR=0.233 TD=0.172 BM=0.233
- nDCG@5: MR=0.631 MD=1.000 TR=0.500 TD=1.000 BM=1.000
- picked nDCG=0.500, 5-arm oracle=1.000, quality_regret=0.500, reward_regret=0.491
- confidence=0.233, reward margin=0.004; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: TR=0.50 而 TD=1.00（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-2481 `tatdqa` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What is the average of the research and development expenses for the years 2018 and 2019, as reported in the financial documents of Shopify Inc. for the year 2019?
- DATASET: `tatdqa`, features: number; qlen=29 words
- probs: MR=0.348 MD=0.161 TR=0.253 TD=0.184 BM=0.054
- nDCG@5: MR=0.431 MD=1.000 TR=1.000 TD=1.000 BM=0.500
- picked nDCG=0.431, 5-arm oracle=1.000, quality_regret=0.569, reward_regret=0.549
- confidence=0.348, reward margin=0.023; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: MR=0.43 而 MD=1.00（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-8442 `wiki-ss` — picked **TR**, nDCG-best **MD**, reward-best **BM**

- **Query**: where do the light independent reactions of photosynthesis take place
- DATASET: `wiki-ss`; qlen=10 words
- probs: MR=0.223 MD=0.151 TR=0.242 TD=0.155 BM=0.229
- nDCG@5: MR=0.000 MD=0.631 TR=0.000 TD=0.500 BM=0.631
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.602
- confidence=0.242, reward margin=0.008; lambda arms l00/l10/l70 = TR/TR/BM
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: TR=0.00 而 TD=0.50（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-1599 `vqaonbd` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: What is the median value of deferred income tax assets net of valuation allowance and deferred income tax liabilities for Microsoft in 2013, which contributes to the net deferred income tax assets liabilities?
- DATASET: `vqaonbd`, features: number; qlen=33 words
- probs: MR=0.571 MD=0.115 TR=0.208 TD=0.067 BM=0.039
- nDCG@5: MR=0.631 MD=1.000 TR=1.000 TD=1.000 BM=0.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.357
- confidence=0.571, reward margin=0.016; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: MR=0.63 而 MD=1.00（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3a-4094 `tatdqa` — picked **MR**, nDCG-best **MD**, reward-best **BM**

- **Query**: What was the percentage decrease in Extreme Networks Inc.'s basic net loss per share from $0.41 in 2018 to $0.22 in 2019, as reported in the company's Consolidated Statements of Operations Data for the fiscal year ended June 30, 2019?
- DATASET: `tatdqa`, features: number, comparison; qlen=44 words
- probs: MR=0.339 MD=0.229 TR=0.181 TD=0.161 BM=0.090
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=0.000 BM=1.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.938
- confidence=0.339, reward margin=0.010; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。
- **数值证据**: MR=0.00 而 MD=1.00（重排把正确页挤出 Top-5）
- **可验证干预**: 导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S3b. 选择 Dense 但 Rerank 本可显著增益（8 例 / 全量 782 例，累计 qr=435.6）

#### S3b-6800 `sciqag` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: What was the turn-on field for the β-SiC nanorods?
- DATASET: `sciqag`; qlen=11 words
- probs: MR=0.208 MD=0.222 TR=0.202 TD=0.177 BM=0.191
- nDCG@5: MR=1.000 MD=0.631 TR=1.000 TD=1.000 BM=1.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.354
- confidence=0.222, reward margin=0.016; lambda arms l00/l10/l70 = TD/MD/BM
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.63 而 MR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-4217 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: What was the amount of Restricted cash within other current assets in 2019?
- DATASET: `tatdqa`, features: number; qlen=13 words
- probs: MR=0.329 MD=0.440 TR=0.105 TD=0.069 BM=0.057
- nDCG@5: MR=1.000 MD=0.631 TR=1.000 TD=0.631 BM=0.000
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.318
- confidence=0.440, reward margin=0.007; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.63 而 MR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-3180 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: In which fiscal years did the percentage of net sales from the EMEA region exceed 20% of total net sales for Kemet Corporation?
- DATASET: `tatdqa`, features: number, comparison; qlen=23 words
- probs: MR=0.305 MD=0.334 TR=0.155 TD=0.128 BM=0.077
- nDCG@5: MR=0.631 MD=0.000 TR=0.631 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.547
- confidence=0.334, reward margin=0.019; lambda arms l00/l10/l70 = MR/MD/MD
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.00 而 MR=0.63（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-2330 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: How many years did commitments that were 1 to 3 years exceed $700 million?
- DATASET: `tatdqa`, features: number; qlen=14 words
- probs: MR=0.278 MD=0.383 TR=0.128 TD=0.084 BM=0.126
- nDCG@5: MR=1.000 MD=0.431 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.431, 5-arm oracle=1.000, quality_regret=0.569, reward_regret=0.500
- confidence=0.383, reward margin=0.027; lambda arms l00/l10/l70 = MR/MD/MD
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.43 而 MR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-2544 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **MR**

- **Query**: Where can the discussion of operating income be found?
- DATASET: `tatdqa`; qlen=9 words
- probs: MR=0.300 MD=0.320 TR=0.161 TD=0.094 BM=0.125
- nDCG@5: MR=1.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.874
- confidence=0.320, reward margin=0.857; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.00 而 MR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-7082 `wiki-ss` — picked **TD**, nDCG-best **MR**, reward-best **BM**

- **Query**: who plays carrie on days of our lives
- DATASET: `wiki-ss`; qlen=8 words
- probs: MR=0.202 MD=0.164 TR=0.184 TD=0.285 BM=0.166
- nDCG@5: MR=1.000 MD=0.631 TR=1.000 TD=0.500 BM=1.000
- picked nDCG=0.500, 5-arm oracle=1.000, quality_regret=0.500, reward_regret=0.474
- confidence=0.285, reward margin=0.026; lambda arms l00/l10/l70 = TD/TD/BM
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: TD=0.50 而 TR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-1635 `vqaonbd` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: In W. R. Berkley Corporation's 2013 report, what is the maximum value between the decrease in prior year loss reserves and the increase in prior year earned premiums that contributed to net favorable prior year development for 2012?
- DATASET: `vqaonbd`, features: number, comparison; qlen=39 words
- probs: MR=0.210 MD=0.239 TR=0.161 TD=0.154 BM=0.235
- nDCG@5: MR=1.000 MD=0.631 TR=1.000 TD=0.387 BM=0.500
- picked nDCG=0.631, 5-arm oracle=1.000, quality_regret=0.369, reward_regret=0.310
- confidence=0.239, reward margin=0.008; lambda arms l00/l10/l70 = MR/MD/BM
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.63 而 MR=1.00（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S3b-2858 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **MR**

- **Query**: In which year were the Dividend shares awarded larger?
- DATASET: `tatdqa`; qlen=9 words
- probs: MR=0.358 MD=0.385 TR=0.122 TD=0.074 BM=0.061
- nDCG@5: MR=0.631 MD=0.000 TR=0.000 TD=0.000 BM=0.387
- picked nDCG=0.000, 5-arm oracle=0.631, quality_regret=0.631, reward_regret=0.539
- confidence=0.385, reward margin=0.177; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。
- **数值证据**: MD=0.00 而 MR=0.63（该 Query 上重排本可救回）
- **可验证干预**: 同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S4. 跨数据集复现：每个数据集损失最大的严重错误（11 例）

#### S4-arxivqa `arxivqa` — picked **MR**, nDCG-best **MD**, reward-best **TD**

- **Query**: What inference can be drawn about the function \( v(\lambda) \) at the point \( \lambda^* \) in figure (b)?
- DATASET: `arxivqa`, features: figure-word; qlen=17 words
- probs: MR=0.414 MD=0.222 TR=0.138 TD=0.076 BM=0.150
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=1.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.942
- confidence=0.414, reward margin=0.003; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MR=0.00, best MD=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-convfinqa `convfinqa` — picked **BM**, nDCG-best **MR**, reward-best **MR**

- **Query**: What was the increase in the value of free cash flow from $1382 million in 2012 to $2085 million in 2013 for Union Pacific Corporation?
- DATASET: `convfinqa`, features: number, comparison; qlen=25 words
- probs: MR=0.217 MD=0.218 TR=0.174 TD=0.154 BM=0.237
- nDCG@5: MR=1.000 MD=0.000 TR=0.000 TD=0.500 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.856
- confidence=0.237, reward margin=0.413; lambda arms l00/l10/l70 = MR/BM/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked BM=0.00, best MR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-dude `dude` — picked **MR**, nDCG-best **TR**, reward-best **TR**

- **Query**: What is the name of the person who signs the document on page twenty?
- DATASET: `dude`, features: figure-word; qlen=14 words
- probs: MR=0.419 MD=0.111 TR=0.366 TD=0.051 BM=0.053
- nDCG@5: MR=0.000 MD=0.000 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.950
- confidence=0.419, reward margin=0.880; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MR=0.00, best TR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-finqa `finqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: What proportion of United Parcel Service's total contractual obligations, as disclosed in the 2014 report, is scheduled to be paid in 2015?
- DATASET: `finqa`, features: number; qlen=23 words
- probs: MR=0.296 MD=0.310 TR=0.165 TD=0.173 BM=0.055
- nDCG@5: MR=1.000 MD=0.000 TR=1.000 TD=0.631 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.883
- confidence=0.310, reward margin=0.004; lambda arms l00/l10/l70 = MR/MD/MD
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MD=0.00, best MR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-finreport `finreport` — picked **MR**, nDCG-best **TR**, reward-best **TR**

- **Query**: What was the net cash impact of divestitures on IBM's cash flow in 2007?
- DATASET: `finreport`, features: number; qlen=15 words
- probs: MR=0.302 MD=0.201 TR=0.190 TD=0.194 BM=0.113
- nDCG@5: MR=0.000 MD=0.631 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.888
- confidence=0.302, reward margin=0.302; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MR=0.00, best TR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-finslides `finslides` — picked **TD**, nDCG-best **MR**, reward-best **MR**

- **Query**: What was the expected ROA for IBM's WW (worldwide) retirement plans at the end of 2017?
- DATASET: `finslides`, features: number; qlen=17 words
- probs: MR=0.213 MD=0.204 TR=0.171 TD=0.232 BM=0.181
- nDCG@5: MR=1.000 MD=0.631 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.831
- confidence=0.232, reward margin=0.268; lambda arms l00/l10/l70 = MR/TD/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked TD=0.00, best MR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-mp-docvqa `mp-docvqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: which service is available for transportation ?
- DATASET: `mp-docvqa`; qlen=6 words
- probs: MR=0.271 MD=0.397 TR=0.206 TD=0.064 BM=0.063
- nDCG@5: MR=1.000 MD=0.000 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.892
- confidence=0.397, reward margin=0.011; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MD=0.00, best MR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-sciqag `sciqag` — picked **MR**, nDCG-best **BM**, reward-best **BM**

- **Query**: What is the focus of studies on nitronyl nitroxide radicals?
- DATASET: `sciqag`; qlen=10 words
- probs: MR=0.230 MD=0.207 TR=0.186 TD=0.194 BM=0.184
- nDCG@5: MR=0.000 MD=0.631 TR=0.000 TD=0.000 BM=1.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.926
- confidence=0.230, reward margin=0.349; lambda arms l00/l10/l70 = TR/MR/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MR=0.00, best BM=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-tatdqa `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **MR**

- **Query**: What percentage of the total property and equipment at cost does 'Equipment and software' represent for parkervision in 2019, before accounting for depreciation?
- DATASET: `tatdqa`, features: number, comparison; qlen=23 words
- probs: MR=0.241 MD=0.282 TR=0.176 TD=0.162 BM=0.138
- nDCG@5: MR=1.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.884
- confidence=0.282, reward margin=0.867; lambda arms l00/l10/l70 = MR/MD/MD
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MD=0.00, best MR=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-vqaonbd `vqaonbd` — picked **MR**, nDCG-best **MD**, reward-best **MD**

- **Query**: For the year 2014, what is the cumulative value of operating income, interest expense, and interest income that contributes towards income from operations before income taxes in the fourth quarter?
- DATASET: `vqaonbd`, features: number; qlen=30 words
- probs: MR=0.518 MD=0.304 TR=0.124 TD=0.024 BM=0.030
- nDCG@5: MR=0.000 MD=1.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.925
- confidence=0.518, reward margin=0.890; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked MR=0.00, best MD=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

#### S4-wiki-ss `wiki-ss` — picked **BM**, nDCG-best **TD**, reward-best **TD**

- **Query**: who wrote the song love on a two way street
- DATASET: `wiki-ss`; qlen=10 words
- probs: MR=0.152 MD=0.167 TR=0.164 TD=0.247 BM=0.270
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=1.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=1.000, quality_regret=1.000, reward_regret=0.883
- confidence=0.270, reward margin=0.883; lambda arms l00/l10/l70 = TD/BM/BM
- **失败假设**: 检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。
- **数值证据**: 该数据集中 quality_regret 最大的严重错误：picked BM=0.00, best TD=1.00
- **可验证干预**: 按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。
- 页面级复核: 需要（HF 数据集相关页 Top-K / 重排前后顺序）

## S5. 对照组：多臂近似并列的“伪错误”（8 例 / 全量 889 例）

#### S5-1937 `vqaonbd` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: What is the median value of the following assets for Agilent Technologies as of October 31, 2006: total reportable segments' assets, cash and cash equivalents and short-term investments, prepaid expenses, cost method investments, restricted cash and cash equivalents, long-term and other receivables, and other including valuation allowance?
- DATASET: `vqaonbd`, features: number; qlen=49 words
- probs: MR=0.228 MD=0.269 TR=0.165 TD=0.175 BM=0.163
- nDCG@5: MR=1.000 MD=1.000 TR=1.000 TD=1.000 BM=1.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.006
- confidence=0.269, reward margin=0.006; lambda arms l00/l10/l70 = MR/MD/MD
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0064（margin=0.006），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-3511 `tatdqa` — picked **MR**, nDCG-best **MR**, reward-best **TR**

- **Query**: What is the average of $1,081 million and $1,059 million, which represent the gross profit for the three months ended December 31, 2019, and December 31, 2018, respectively, as reported by STMicroelectronics in their 2019 financial report?
- DATASET: `tatdqa`, features: number; qlen=39 words
- probs: MR=0.248 MD=0.193 TR=0.205 TD=0.140 BM=0.213
- nDCG@5: MR=1.000 MD=0.000 TR=1.000 TD=0.000 BM=0.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.003
- confidence=0.248, reward margin=0.003; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0031（margin=0.003），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-393 `finqa` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: What were the average net sales for Rotary and Mission Systems (RMS) in millions of dollars between 2014 and 2016, considering the significant impact of the Sikorsky acquisition on the 2016 figures?
- DATASET: `finqa`, features: number, comparison; qlen=32 words
- probs: MR=0.207 MD=0.222 TR=0.168 TD=0.196 BM=0.207
- nDCG@5: MR=1.000 MD=1.000 TR=1.000 TD=1.000 BM=1.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.004
- confidence=0.222, reward margin=0.004; lambda arms l00/l10/l70 = TD/MD/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0035（margin=0.004），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-3329 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **TR**

- **Query**: What is the average number of Technicians employed?
- DATASET: `tatdqa`; qlen=8 words
- probs: MR=0.280 MD=0.391 TR=0.202 TD=0.071 BM=0.055
- nDCG@5: MR=1.000 MD=1.000 TR=1.000 TD=0.000 BM=0.387
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.003
- confidence=0.391, reward margin=0.003; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0030（margin=0.003），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-5000 `mp-docvqa` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: Who is a member of editorialboard of five journals in neuroscience and medicine?
- DATASET: `mp-docvqa`; qlen=13 words
- probs: MR=0.185 MD=0.269 TR=0.163 TD=0.148 BM=0.235
- nDCG@5: MR=1.000 MD=1.000 TR=1.000 TD=0.631 BM=1.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.008
- confidence=0.269, reward margin=0.008; lambda arms l00/l10/l70 = MD/MD/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0085（margin=0.008），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-6891 `sciqag` — picked **MD**, nDCG-best **MD**, reward-best **BM**

- **Query**: What are the key requirements that matrixes used in tissue engineering must fulfill?
- DATASET: `sciqag`; qlen=13 words
- probs: MR=0.221 MD=0.244 TR=0.221 TD=0.145 BM=0.168
- nDCG@5: MR=0.000 MD=0.431 TR=0.000 TD=0.387 BM=0.431
- picked nDCG=0.431, 5-arm oracle=0.431, quality_regret=0.000, reward_regret=0.006
- confidence=0.244, reward margin=0.006; lambda arms l00/l10/l70 = MR/MD/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0059（margin=0.006），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-2450 `tatdqa` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: What was the difference in the number of shares Outstanding at 1 April between 2019 and 2018 for the UK SIP and Irish SIP combined?
- DATASET: `tatdqa`, features: number, comparison; qlen=25 words
- probs: MR=0.277 MD=0.289 TR=0.151 TD=0.117 BM=0.166
- nDCG@5: MR=1.000 MD=1.000 TR=1.000 TD=0.500 BM=1.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.006
- confidence=0.289, reward margin=0.006; lambda arms l00/l10/l70 = MD/MD/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0064（margin=0.006），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S5-7519 `wiki-ss` — picked **TD**, nDCG-best **MD**, reward-best **BM**

- **Query**: who played alan michael spaulding on guiding light
- DATASET: `wiki-ss`; qlen=8 words
- probs: MR=0.117 MD=0.185 TR=0.111 TD=0.301 BM=0.286
- nDCG@5: MR=0.000 MD=1.000 TR=0.631 TD=1.000 BM=1.000
- picked nDCG=1.000, 5-arm oracle=1.000, quality_regret=0.000, reward_regret=0.008
- confidence=0.301, reward margin=0.003; lambda arms l00/l10/l70 = TD/TD/BM
- **失败假设**: argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。
- **数值证据**: 选中臂与 reward 最优臂差距仅 0.0085（margin=0.003），属并列最优
- **可验证干预**: 评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

## S6. 对照组：五路全部失败（8 例 / 全量 371 例，仅贡献 0.1% regret）

#### S6-6178 `arxivqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: In figure (f), what feature is highlighted by the red dashed line?
- DATASET: `arxivqa`, features: figure-word; qlen=12 words
- probs: MR=0.541 MD=0.084 TR=0.236 TD=0.058 BM=0.081
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.032
- confidence=0.541, reward margin=0.014; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-2369 `tatdqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: What is the accounts receivable, net increase / (decrease) from 2018 to 2019?
- DATASET: `tatdqa`, features: number, comparison; qlen=12 words
- probs: MR=0.397 MD=0.325 TR=0.165 TD=0.046 BM=0.067
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.053
- confidence=0.397, reward margin=0.008; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-1803 `vqaonbd` — picked **MD**, nDCG-best **MR**, reward-best **BM**

- **Query**: What is the cumulative total of Rents and purchased transportation, Salaries, wages and employee benefits, Fuel and fuel taxes, Depreciation and amortization, Operating supplies and expenses, Insurance and claims, General and administrative expenses net of asset dispositions, Operating taxes and licenses, and Communication and utilities for J.B. Hunt in 2010?
- DATASET: `vqaonbd`, features: number; qlen=51 words
- probs: MR=0.192 MD=0.299 TR=0.152 TD=0.124 BM=0.232
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.010
- confidence=0.299, reward margin=0.010; lambda arms l00/l10/l70 = MD/MD/MD
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-6086 `arxivqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: What is the process being depicted in the transition from (a) to (c) in the figure?
- DATASET: `arxivqa`, features: figure-word; qlen=16 words
- probs: MR=0.549 MD=0.086 TR=0.229 TD=0.070 BM=0.065
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.029
- confidence=0.549, reward margin=0.003; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-2661 `tatdqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: What was the total revenue in 2015?
- DATASET: `tatdqa`, features: number; qlen=7 words
- probs: MR=0.418 MD=0.202 TR=0.188 TD=0.114 BM=0.078
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.034
- confidence=0.418, reward margin=0.012; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-5549 `arxivqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: Based on the figure, which statement is true about the temperature distribution within the ice layer?
- DATASET: `arxivqa`, features: figure-word; qlen=16 words
- probs: MR=0.359 MD=0.240 TR=0.180 TD=0.100 BM=0.121
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.041
- confidence=0.359, reward margin=0.007; lambda arms l00/l10/l70 = MD/MR/BM
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-2947 `tatdqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: What caused the decrease in effective tax rate from 2018 to 2019?
- DATASET: `tatdqa`, features: number, comparison; qlen=12 words
- probs: MR=0.361 MD=0.304 TR=0.152 TD=0.085 BM=0.098
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.032
- confidence=0.361, reward margin=0.002; lambda arms l00/l10/l70 = MR/MR/BM
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）

#### S6-3647 `tatdqa` — picked **MR**, nDCG-best **MR**, reward-best **BM**

- **Query**: How many years did research and development expenses exceed $50 million?
- DATASET: `tatdqa`, features: number; qlen=11 words
- probs: MR=0.487 MD=0.189 TR=0.183 TD=0.082 BM=0.059
- nDCG@5: MR=0.000 MD=0.000 TR=0.000 TD=0.000 BM=0.000
- picked nDCG=0.000, 5-arm oracle=0.000, quality_regret=0.000, reward_regret=0.051
- confidence=0.487, reward margin=0.001; lambda arms l00/l10/l70 = MR/MR/MD
- **失败假设**: 失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。
- **数值证据**: 五臂最佳 nDCG 仅 0.000：无任何可选臂能找到证据（不可路由问题）
- **可验证干预**: 检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。
- 页面级复核: 不需要（逐 Query 指标已足够定性）


---

## 复核工作量建议

- 优先页面级复核顺序：**S3a（4 例）→ S2a（4 例）→ S2b（3 例）→ S1（2 例）**，共 13 例即可初步验证三个主假设；
- 每个页面级 Case 保存：相关页 ID、Dense Top-20、Rerank Top-5 及前后排名、页面类型（正文/表格/扫描图/图表）；
- 全部 63 例的数值字段可由 `query_metrics.csv` 按 query 直接复得，本文件不复制全量列。
