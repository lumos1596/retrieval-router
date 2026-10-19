# RetrievalRouter：偏好迁移损失与跨语料路由失败机制验证（下一轮）

> **本轮定位：零训练、零新检索。** 先证明“源域学到的Retriever偏好迁移到目标库会造成真实质量损失”，以及“当前Router是否真的因此选错”。只有两点均有证据，才继续设计Utility Shift-Aware / Robust Training。

## 1. 本轮只回答三个问题

1. **偏好有没有变到足以影响决策？** 同一种Query类型，源库选出的最强臂，放到目标库是否真的不再最优？实际损失多少nDCG？
2. **当前Router是不是沿用了过时偏好？** 在发生迁移损失的Query上，Router是否更倾向于源库强臂，并产生更高Routing Regret？
3. **是不是单纯因为没见过这种Query？** 单独统计源域覆盖不足类别，避免把Query类型缺失误当成Retriever偏好变化。

**大白话：** 我们不能只说“MR比MD领先的幅度变了”。如果MR在两个库里始终最好，Router继续选MR就没有选错。必须证明“照搬旧经验实际会吃亏”，还要证明Router确实照搬了旧经验。

## 2. 数据与隔离规则

- **源域：** FinQA、ConvFinQA、VQAonBD，使用本次Router实际训练用到的Query及五臂nDCG；按真实训练划分提取，不混入目标集。
- **目标域：** TAT-DQA（2,712）、Wiki-SS（1,496）、ArxivQA（903），使用每条Query的五臂nDCG和已训练`λ=0` Router的预测臂；`λ=0.1`仅作为补充核对。
- **必须读取原始逐Query表**（原文五臂指标表 + 此次路由器完整预测表），统一`query_id / DATASET / query / query_type / pred_arm / 五臂ndcg`。检验ID唯一、标签完整、分组对应的split，优先ID关联，不依靠模糊Query匹配。
- 已上传的`preference_shift_cells.csv / conflicts.csv / vs_regret.csv`只有**聚合统计**，`xdomain_badcases.csv`只有筛选的极端样本，**不足以完成下面全部逐Query计算**。从此前实验脚本导出全量逐Query表；无需重跑五个Retriever。
- 使用上轮**已修正**的Query分类器，不因为看到检索结果再次调整；重点人工抽查`computation`、`comparison`、`fact_with_year`等边界样本，每类随机约15–20条。先保持现有类型体系，不做高维聚类。
- 三个目标域已反复用于探索性分析，**不能视为最终盲测**；本轮发现需在以后新的留出语料库上复核。

## 3. 实验A：计算“照搬源域选臂经验”究竟损失多少

### A1. 构造源域的Query类型 → Retriever选择规则

对每个Query类型`t`，计算各臂在**源域训练样本**上的平均nDCG：

\[
\bar r_{S,t,a}=\operatorname{mean}_{q\in S_t}\operatorname{nDCG}_a(q),\quad
 a_S(t)=\arg\max_a\bar r_{S,t,a}.
\]

主报告采用**每个源语料库先求平均、再对3个语料库等权平均**，避免样本量最大的金融库支配结果；同时附上直接按Query混合平均的敏感性对照。某个源库不存在该类Query时，必须如实标明支持不足，不能假装三个库都有证据。

- 若某类Query源域总样本`<50`：标记`uncovered`，不参与偏好迁移主分析。例如`fact_no_visual`源域只有5条。
- 对样本数足够的类型，记录源域前两臂的平均nDCG差；差距`<0.02`或Bootstrap区间含0时标记`weak source preference`，不把随意选出的第一名称为稳定偏好。

### A2. 用源域规则直接评测目标域

对目标库`C`内同类型`t`的Query，仍固定使用`a_S(t)`，计算其真实平均检索质量。再计算目标库中这一类型的**事后最佳固定臂**`a_C(t)`。

\[
\boxed{G(t,C)=\bar r_{C,t,a_C(t)}-\bar r_{C,t,a_S(t)}}
\]

`G`称为**偏好迁移损失**：同类Query若仍按源域选Retriever，相比该目标库同类Query的最佳固定臂，平均多损失多少nDCG。

- **`G≈0`：** 优势幅度即使变化很大，也未必导致源域选臂策略失效。
- **`G>0`且稳定：** 源域类型级决策迁移确实存在可避免的损失。
- `a_C(t)`是利用目标域真实标签事后算出的**诊断上界**，不是实际部署可以拿到的结果。

**防止事后选最佳臂造成乐观偏差：** 每个目标库按Query随机分成两半；仅用第一半决定`a_C(t)`，在另一半评估`G`；互换并在10个固定种子下重复。目标子组过小则不宣称显著改进。输出同时报告全样本描述统计和交叉拟合结果。

### A3. 不只看5臂最优，还看具体架构优势

补充计算各组差值变化（单位nDCG）：

- `MR−MD`：多模态Dense vs 多模态Late Interaction；
- `TR−TD`：文本Dense vs 文本Late Interaction；
- `MR−TR`：视觉与文本Late Interaction；
- `MD−TD`：视觉与文本Dense。

对每组报告源域均值、目标域均值和**均值差的Bootstrap 95%区间**；不要再用“双方都超过±0.05”来替代检验。区分：方向翻转、幅度变化但方向未变、样本不足/优势并列。

**优先核查的前期发现：**
- `visual_explicit`：源域`MR−MD≈+0.015`，ArxivQA约`+0.228`；重点核查这种差值放大是否真的使源域类型级最佳臂失效，**不能预先认定**。
- `comparison`：视觉组−文本组从`+0.078`到Wiki-SS的`−0.049`；检验差值变化，同时用五个具体臂计算真实`G`，避免把组内事后Oracle误认为可用策略。
- `fact_no_visual`：源域仅5条，必须归入覆盖不足，**不能拿它声称偏好冲突**。

## 4. 实验B：验证Router的实际错误是否与迁移损失有关

在目标库内按类型分组，记录：

1. `G(t,C)`，即源域类型规则的迁移损失；
2. Router的平均`Regret= max_a ndcg_a(q)−ndcg_pred(q)`；
3. Router在该类型中选择`a_S(t)`的比例（是否真的沿用源域偏好）；
4. Router、源域类型固定策略、目标域类型最佳固定策略的平均nDCG（第三个只作为后验参考）；
5. 严重掉零率：`max_a ndcg_a ≥0.75 且 ndcg_pred=0`；
6. 若源域偏好与目标域最佳不同，Router在高损失样本中选`a_S(t)`的占比和造成的实际损失。

**核心对照：**
- **有迁移损失且Router沿用源域臂**：最支持“学到旧规律导致错选”的机制。
- **有迁移损失但Router没有沿用源域臂**：不能将错误直接归因于源域偏好。
- **迁移损失≈0但Router仍高Regret**：说明还存在类型内细粒度路由/训练不足等其他问题。
- **源域无支持的OOD类型**：另行统计，不混进上述对照。

**统计要求：** 每个目标数据集分别报告，避免把Wiki-SS整体Regret更高当成因果关系；每个充分支持的类型给出Bootstrap置信区间。以Query类型聚合的组数可能很少，不能根据几个组的相关性强行作因果断言。抽取8–12条高损失真实例子复查五臂值与Router选择。

## 5. 实验C：区分“没见过”和“学错偏好”，并量化解释范围

每条目标Query按源域支持程度与类型级迁移损失分为：

- **C1：未覆盖（coverage OOD）**：源域该类样本`<50`。
- **C2：覆盖且源域偏好可迁移**：源域支持足够、`G`很小。
- **C3：覆盖但源域偏好迁移失败**：源域支持足够、`G`有可靠正值。
- **C4：源域偏好不确定**：源域各臂近似并列或置信区间过宽。

对各组统计：`样本量 / 占目标样本比例 / 平均Router Regret / 占总Regret比例 / 严重掉零率 / 选源域臂比例`。

> **重要：** 统计时C3用目标标签定义，仅用于诊断。`G`反映*类型级源域规则的可迁移程度*，不等于Router实际错误的因果贡献。不能直接把C3的全部Regret宣称由偏好迁移造成。

## 6. 交付物（交给Agent的具体输出）

1. `source_type_arm_profile.csv`：每类型、每源库五臂均值、支持数、源域最优臂、前两臂差值与置信区间。
2. `transfer_loss_by_type.csv`：`target_corpus,query_type,n,source_best_arm,target_best_arm,source_policy_ndcg,target_type_oracle_ndcg,G,CI,coverage_status`。
3. `transfer_loss_vs_router.csv`：各类型目标域的`G,router_ndcg,router_regret,source_arm_choice_rate,source_policy_ndcg,catastrophic_rate`。
4. `transfer_regret_groups.csv`：C1–C4的分数据集与整体统计，包括**各组占总Regret比例**。
5. `transfer_mechanism_report.md`：**2张图**（各类Query的`G`柱图、`G`与Router Regret/选源域臂比例对照图）；**8–12条真实Query**；最后只给出是否继续训练创新的判断及其限制。

## 7. 决策标准（必须如实允许否定假设）

| 观察结果 | 解释 | 下一步 |
|---|---|---|
| 有足够样本支持的类型存在稳定的正`G`，Router又经常沿用源域臂，并在这些Query上发生大损失 | **支持**源域偏好错误迁移这一机制 | 再训练`KL vs 更丰富数据的KL vs Utility/Robust训练`，探索效用偏移感知训练 |
| 大部分类型的`G≈0`，但Router Regret仍高 | 不支持“源域最佳臂迁移错误是主因”；类型内细粒度选择仍是问题 | 转向更细的训练目标/Query表征诊断，不强推效用偏移故事 |
| 大量错误集中在源域样本稀缺的Query类型 | 更像训练覆盖不足 | 先做等规模、多源域训练对照 |
| `G`较大，但Router并不选择源域最优臂 | 说明偏好变化存在，但不能证明Router因此选错 | 检查训练目标与Router预测是否偏向其他臂，避免错误归因 |

**本轮成功标准：** 明确量化源域类型偏好迁移到未见库的实际nDCG代价，以及它与Router真实选臂错误之间的联系；即使假设不成立，也算完成，不以必须发现可发论文的故事为目标。

## 8. 最小执行顺序

`准备全量逐Query数据 → 源域每类型5臂能力表 → 目标域交叉拟合计算G → 对照Router选择/Regret → C1–C4分层 → 报告结论`。

**不做**新训练、不跑全套检索器、不做RL、不做复杂可视化；本轮只验证值得为之设计新训练机制的事实是否存在。
