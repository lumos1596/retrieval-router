# Rerank 失败/增益机制核实案例簿（28-Case 重放实验）

- 实验：同一 Dense Top-100 → MaxSim 晚交互 Rerank Top-5 重放（MD→MR：Nomic Dense→ColNomic/ColQwen2.5；TD→TR：Linq→GTE-ModernColBERT）。
- 方法：每个案例同时核对 **官方 nDCG@5 与重放 nDCG@5**；只有**两阶段（或至少 rerank 阶段，单独注明）与官方精确一致**的案例才进入本簿。
- 取证：每页的页面图、文本与 GT 标注来自 HF `emrekuruu/*` 语料（与官方同源、同 revision），导出在 `analysis/replay_data/evidence/<qid>_<arm>/`。
- 规则：nDCG 只说明 GT 页是否在 Top-5；所有"机制"结论均由人工查看页面图/原文独立核实，区分**观察事实**与**推测**。
- 单 GT 页；nDCG 值 0.6309 = GT 排第 2，0.5 = 第 2（多 GT 或 IDCG 口径），0.387 = 第 3。

## 对齐与选案范围说明（重要）

28 个案例 × 最多两臂共 50 次重放中，Dense 臂几乎全部与官方一致（仅 Top-5 边界精确同分处因 Qdrant 存储序不可复现而不同）。但 **Rerank 臂存在一批官方 nDCG 无法用已发布产物复现的案例**（模型权重、数据、软件版本穷尽验证均一致，详见 `rerank_mechanism_report.md` 附录 A），这些案例**不进入**机制结论。本簿 9 个案例全部为官方对齐案例，覆盖：

- 负增益（Rerank 有害）已对齐：**仅 1 例**（2481/MR）；文本臂 B 组 4 例全部落在不可复现/近平局区。
- 正增益（Rerank 有益）已对齐：8 例（MR：2544、4217、5177、697、2013；TR：343、45、7082）。
- 并列控制：735（MR/TR 均 1.0，未展开）。

---

## 负增益侧（A/B：Rerank 把 GT 排下去）

### Case N1 — qid 2481 / tatdqa / MR / A 组（稠密 1.0 → Rerank 0.431，官方=重放）

- Query："What is the average of the research and development expenses for the years 2018 and 2019, as reported in the financial documents of Shopify Inc. for the year 2019?"
- 排名事实：GT 页 Dense 第 1（与第 2 名**精确同分** 0.2896，平局顺序我们与官方恰好一致）→ Rerank 第 4；Rerank 前 3 名分别为"无形资产"附注页、股份支付附注页、提及"qualifying R&D expenditures"的税收抵免页。
- **人工核实（页面图）**：
  - GT 页是 Operating Expenses 章节，含 "Research and development" 行：**2019 = 355,015；2018 = 230,674**（千美元），正是答案 (355,015+230,674)/2 的来源。
  - Rerank 第 1 页是 "10. Intangible Assets" 附注：含 2019/2018 两张数字表，底部摊销费用表里**也有一行 "Research and development"（266 / 60）**——标签词、年份列、数字格式全部命中，但语义是"无形资产摊销费用里的研发部分"，不是研发费用总额。
  - Rerank 第 3 页正文出现 "R&D expenditures … 2019 and 2018" 字样，是纯词面诱饵。
- **观察事实**：三页都同时具备 query 的关键表层线索（"Research and development"、"2018/2019"、数字表）；GT 与诱饵的区别在于**表所属附注语境**（Operating Expenses 正表 vs Intangible Assets 附注），需要跨表头/章节的语义判定。
- **推测机制（label-free 特征一致）**：MaxSim 按 query token 逐块取最大匹配，"Research and development + 年份 + 密集数字"的局部 patch 在诱饵页得分更集中；Dense 全局向量保留了整页"经营费用讨论"的语义。该页 Rerank Top-5 边界相对分差 < 0.5%（近平局区），即晚交互在此不是"自信地选错"，而是"在一堆表层同源页里抖动"。

> 负增益侧可核实样本仅此 1 例（文本臂 B 组 sciqag 6829/6875、wiki 7959/8381 的官方 rerank 值均不可复现，不能用于机制归因）。

---

## 正增益侧（C/D：Rerank 把 GT 排上来）

### Case P1 — qid 2013 / vqaonbd / MR / C 组（稠密 0.631 → Rerank 1.0，双方一致）

- Query："In HCA Healthcare's 2017 report, what is the average of the provision for current year claims and unfavorable development related to prior years' claims for 2016?"
- 事实：GT=`HCA/2017/page_64` Dense 第 2 → Rerank 第 1；Dense 第 1 是 `HCA/2018/page_64`，候选中还有 2012/2014/2015 年同公司同编号页。
- **人工核实（页面图）**：GT 页含 "Professional Liability Claims" 表，2016 列 "Provision for current year claims = 428"、"Unfavorable development related to prior years' claims = 2"，平均 215。2018 版同表列年份错位（2017/2016/2015），不满足"2017 年报告中问 2016"。
- 机制（观察）：**跨年年报同模板近重复页**，版面几乎相同；Rerank 靠年份数字 patch 与报告年份 token 的精确共现把正确年份版本顶到第 1。

### Case P2 — qid 697 / finqa / MR / C 组（稠密 0.631 → Rerank 1.0，双方一致）

- Query 问 Altria 2015→2016 smokeless products 经营利润占比变化。
- 事实：GT=`MO/2017/page_10` Dense 第 2 → Rerank 第 1；Dense 第 1=`MO/2016/page_10`（同一张分部占比表，年份列错位），第 3=`MO/2014/page_11`。
- 机制（观察）：与 P1 同构——**正确年份的同模板表格页**被晚交互修正。TR 臂同案两阶段均为 0.631（文本晚交互未能修正），说明该收益是多模态表格 patch 匹配带来的。

### Case P3 — qid 4217 / tatdqa / MR / C 组（稠密 0.631 → Rerank 1.0，双方一致）

- Query："What was the amount of Restricted cash within other current assets in 2019?"
- 事实：GT（VMware 附注页）Dense 第 2 → Rerank 第 1；Dense 第 1 是 "Inventories" 数字表；第 3/4 名是两张**文本完全相同**的 "restricted cash reconciliation" 页（精确同分）。
- **人工核实（页面图）**：GT 页 Restricted Cash 表中**逐字出现** "Restricted cash within other current assets … 2019 = 35"；第 3/4 名的 reconciliation 页虽然讲 restricted cash，却没有 "within other current assets" 行。
- 机制（观察）：**长页/多表混杂场景下精确行标签匹配**。GT 页前半还有股份表、现金等价物表，全局向量被稀释（Dense 第 2）；MaxSim 对完整行标签逐块匹配，且能把"讲同一主题但缺该行"的近重复页压下去。

### Case P4 — qid 2544 / tatdqa / MR / C 组（Rerank 1.0 双方一致；稠密边界平局，单独注明）

- Query："Where can the discussion of operating income be found?"
- 事实：Rerank 第 1 = GT；GT 在我们的 Dense 中第 5（官方 Dense 为 0.0；GT 落在稠密第 5/6 名边界，属已标记的 tie-zone，两版稠密仅边界序不同）。
- **人工核实（页面图）**：GT 页有独立加粗小标题 **"Operating Income"** 及对应数据表；其余候选均为正文泛泛提及 results of operations 的页面。
- 机制（观察）：**章节标题 patch 精确命中**。注意：此案的收益在官方与重放中都成立（rerank 对齐），但稠密侧未严格对齐，故仅作佐证案例。

### Case P5 — qid 5177 / mp-docvqa / MR / C 组（稠密 0.631 → Rerank 1.0，双方一致）

- Query："What is the old salary/rate of pay of Leanne A Turner?"
- 事实：GT=`lfgn0226_47` Dense 第 2 → Rerank 第 1；候选 5 页全是同一员工的 HR 表单，Dense 第 1=`lfgn0226_35`。
- **人工核实（两张页面图）**：两页均为 SALARY ACTION 表，OLD/NEW 结构完全相同。GT（2007-10-22，Reason SA-OA）OLD=**107,000**；Dense 第 1（2005-02-21，SA-PR）OLD=80,210。qrel 指定 _47 页。
- 机制（观察）：**同实体同模板表单的近重复消歧**——全局向量无法区分两张几乎相同的表单，晚交互对日期/数字/原因码等细粒度 patch 更敏感。为何 qrel 取 2007 页属于数据集标注口径，我们只观察"Rerank 选到了 qrel 页"这一事实。

### Case P6 — qid 343 / finqa / TR / D 组（稠密 0.431 → Rerank 1.0，双方一致）

- Query 问 IPG 2007 年报中 auction rate securities 投资减值百分比。
- 事实：GT=`IPG/2009/page_67` Dense 第 4 → Rerank 第 1；Dense 前 3：IPG/2008 诉讼附注（0.6256）、`MMM/2007` 的 auction rate 页、IPG/2006 MD&A。
- **人工核实（原文）**：GT 页逐句写 "In 2007 we realized an other-than-temporary charge of $5.8 relating to a $12.5 investment in auction rate securities"（减值比例 46.4%）——**公司、年份、主题**三者交集只在此页；Dense 被"年份更近/词面更像"的其他公司页骗走。
- 机制（观察）：文本晚交互对罕见实体短语（auction rate securities + impaired/charge）的逐词共现敏感，能纠正 Dense 的年份/公司混配。

### Case P7 — qid 45 / finreport / TR / D 组（稠密 0.5 → Rerank 1.0，双方一致）

- Query："What benefits does IBM's KONE elevators and escalators technology provide?"
- 事实：GT=`IBM_Annual_Report_2016_page_19` Dense 第 3 → Rerank 第 1；Dense 前两名为 2017 年 Watson 泛案例页。
- **人工核实（原文）**：GT 页逐句写 "KONE enables one billion people to move on more than one million elevators and escalators worldwide … predict potential outages … improve people flow and performance"，正面回答。
- 机制（观察）：**稀有实体词 KONE + elevators/escalators 的全部 query term 同页精确覆盖**；Dense 偏好主题更泛的 IBM/Watson 页面。

### Case P8 — qid 7082 / wiki-ss / TR / D 组（稠密 0.5 → Rerank 1.0，双方一致）

- Query："who plays carrie on days of our lives"。
- 事实：GT=`Austin Reed (Days of Our Lives)` Dense 第 3 → Rerank 第 1；Dense 前两名为演员条目 Kyler Pettis、Marci Miller。
- **人工核实（原文）**：GT 条目多次出现 Carrie Brady/Carrie 与演员关系的描述（"Christie Clark's Carrie"、"Carrie (a lawyer)…"），是 qrel 指向的角色页。
- 机制（观察）：短 query、实体词歧义（carrie 为人名）时，晚交互要求各 query token 在同一条目内都有强匹配，角色条目（含 Days of Our Lives 与 Carrie 双重线索）胜出。

---

## 控制案例（E 组，两阶段都对齐且并列）

- qid 735 / finqa：MD=MR=1.0、TD=TR=1.0，GT 在全部四臂均第 1（Sysco 客户类型页），作为"无歧义强命中"对照：rerank1 token/数字覆盖与 dense1 相同，无重排。
- qid 1962 / vqaonbd MR、7821/8396 wiki TR 同属此类（均双方 1.0）。

## 跨案例观察（仅描述，不做门限宣称）

1. **正增益的 8 个已核实案例共享同一结构**：候选集中存在多个"主题正确、版面/模板高度相似"的页面（跨年年报同表页、同实体多份表单、同主题长短页），GT 的判别线索是**细粒度 token 共现**（行标签、年份、实体专名）。Dense 全局向量对这些近重复页区分度低，MaxSim 区分度高。
2. **唯一已核实负增益案例**中，诱饵页与 GT 同样满足表层线索，差别在表所属的附注语境；MaxSim 的局部匹配无法做跨表头语义消歧。且该案例落在 rerank 近平局区（边界相对分差 <0.5%）。
3. 两个方向都与"近重复候选密度"相关：已对齐案例 Dense Top-10 全部存在近重复/同模板页（label-free `near_dup_in_top10=1`），单凭这一特征无法区分增益方向；方向差异取决于**判别线索是局部行级标签（增益）还是全局语境（减益）**。
4. 样本量：负增益侧仅 1 例可核实，文本臂 B 组无对齐案例——本簿不支持任何"何时该关掉 Rerank"的定量结论。
