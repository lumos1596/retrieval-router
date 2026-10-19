# RetrievalRouter 下一轮实验：高价值非MR切换信号诊断

> **本轮任务：零新增模型训练，先确认评估可靠，再检验“Router是否遗漏了稀少但高价值的非MR优势样本”。** 不重跑五个Retriever，不设计新Loss，不开展新一轮Query类型规则分类。
>
> **一句话大白话：** 先查清楚这次考试有没有对错答案；再看看是不是因为MR平时太强，Router学会了“基本都选MR”，却错过少数换成MD/TR/BM25能明显提分的题。

## 1. 实验背景：数据已经表明什么

本轮`train_loss_comparison.csv`报告：三个目标语料库nDCG@5宏平均，B0（原KL）0.5319、B1（Utility Regression）0.6443、B2（Utility+Regret）0.6709、固定MR 0.6723。尤其是B2在ArxivQA中**100%选MR**，nDCG与固定MR完全一样。

因此，B2虽然较本轮B0提高很多，却**尚未证明学会了Query级动态选择**。下一步应该诊断它有没有忽略非MR确实更好的Query。

**两个必须先核对的问题：**

1. 以前λ=0模型在source_test/TAT-DQA/Wiki-SS/ArxivQA的nDCG约为`0.862/0.621/0.666/0.559`，本轮B0却是`0.800/0.505/0.675/0.416`。两轮B0明显不一致，不能直接把全部提升归因于新Loss。
2. `train_loss_badcases.md`第一例写`B0=MD(r=1.000)，B1=MR(r=0.000)`，但紧接的五臂表却写`MD=0，MR=1`，即日志中的`r`与所选臂的真实nDCG对不上。其他案例也有类似冲突。必须先查表列顺序、选臂ID、日志生成与样本对齐。

此外，`train_loss_margin_analysis.csv`中三个分桶数量未覆盖各目标库全部Query：TAT-DQA共2614/2712，Wiki-SS共1468/1496，ArxivQA共890/903。应解释漏掉的98/28/13条，而不是直接忽略。

## 2. 要回答的三个研究问题

**Q1：非MR明显更好的Query，是否在训练集中稀少？**

- 统计源域训练/验证集里，真正“换成其他Retriever可以明显超过MR”的Query数量、比例、所属检索臂、总潜在增益。
- 检查这些样本是否被某类Query或某个语料库垄断。

**Q2：这些Query是否没有被B1/B2学会？**

- 比较B0、B1、B2在上述Query上的选臂、相对MR收益、错误率及实际损失。
- 同时检查模型为换臂付出的负收益，避免只统计“正确切换”。

**Q3：未见语料库上的主要损失，是否来自漏掉高价值切换机会？**

- 如果目标域中这类Query数量不少、潜在收益很大，而B2几乎全选MR，我们就找到了明确的训练瓶颈。
- 如果高价值机会很少，或其可利用收益十分有限，就不应继续专门设计“切换学习”的复杂方法。

**大白话：** MR一直拿高分也没什么问题；真正有问题的是“别的Retriever明明能拿100分、MR只能拿30分，Router却不敢换”。

## 3. 第0步：数据和基线一致性检查【不通过就暂停】

### 3.1 统一逐Query记录

要求Agent导出并校验如下两张表（仅有当前汇总CSV不够）：

**`all_query_rewards.csv`**：

`query_id, dataset, split, query, bm_ndcg, td_ndcg, tr_ndcg, md_ndcg, mr_ndcg`

**`all_router_predictions.csv`**：

`query_id, dataset, split, model, seed, predicted_arm, score_BM, score_TD, score_TR, score_MD, score_MR`

其中B0的`score_*`为路由logits或选择概率，B1/B2为对应的预测效用/打分；**各模型的score含义须另行注明**，不能混用数值做直接校准比较。若数据规模大，可分别保存文件，但必须通过`query_id + dataset + split`严格关联。

### 3.2 必须通过的自动断言

- 五臂列顺序在**训练标签、模型输出、argmax解码、评测、Bad Case生成**所有环节完全相同；特别核对`BM/TD/TR/MD/MR`与模型head index的映射。
- 对每条Query，重新计算`selected_ndcg = reward[predicted_arm]`、`oracle_ndcg = max(five rewards)`、`regret = oracle_ndcg - selected_ndcg`；确保`regret>=0`且和汇总表一致（浮点容差除外）。
- 重建所有B0/B1/B2与固定MR的各库nDCG及选臂比例，必须与`train_loss_comparison.csv`一致；固定MR的选臂比例应为**MR=100%**，不能写成0%。
- 逐条核对Bad Case中“预测臂、该臂r、五臂nDCG”；如果不一致，**重新生成Casebook**，不用旧版本。
- 核实与旧版B0的差异：同一批Query和五臂nDCG、训练数据划分、λ=0、温度、模型初始checkpoint、LoRA与分类头、训练epoch/step、seed、best checkpoint选择、推理argmax是否相同。若是不同seed/训练配置，标明“不构成严格可比”；必要时**仅重跑一次对齐后的B0**。
- 对齐源域训练和验证样本，不允许把目标域qrels用于任何阈值、权重或checkpoint选择。

**交付：** `sanity_report.md`：差异定位、逐条断言、B0对齐后的可用性结论。**未通过前不开展后续机制归因。**

## 4. 第1步：在训练集统计“MR之外的高价值机会”

每条Query具有五臂nDCG。定义：

\[
 r_{\mathrm{alt}}(q)=\max_{a\ne \mathrm{MR}} r_a(q),\qquad
 \Delta(q)=r_{\mathrm{alt}}(q)-r_{\mathrm{MR}}(q).
\]

按`Δ`分桶（对五臂全部比较，不只MD/MR），确保每条Query恰好归入一类：

| 分组 | 判定 | 大白话 |
|---|---|---|
| **High-value switch** | `Δ >= 0.20` | 切换能明显提分，值得重点学 |
| **Moderate switch** | `0.10 <= Δ < 0.20` | 切换有一定收益 |
| **Small positive gain** | `0 < Δ < 0.10` | 其他Retriever略好，但不值得把微小差别当成大错误 |
| **Near tie / slight MR advantage** | `-0.10 < Δ <= 0` | MR不差于其他Retriever，差别很小 |
| **MR clearly better** | `Δ <= -0.10` | 坚持MR通常更合理 |

全部数据只用于**事后诊断**；不能在目标测试库上据此筛训练样本。

主分析只需阈值`0.10`和`0.20`，不要利用目标域尝试一堆阈值挑最优。

在**源域train和validation分别**统计：

1. `High-value switch`数量与比例；所有`Δ>0`比例。
2. `High-value switch`中最佳备选分别是BM、TD、TR、MD的比例；并列最佳允许多臂并列、另报唯一最佳比例。
3. `High-value switch`可贡献的**固定MR潜在提升**：`mean(max(Δ,0))`，也就是允许每条Query从MR切到更好臂时的理论提升上限（不是可实现收益）。
4. 不同源语料库中的分布，检验是否只被少量子数据集贡献；不依赖粗糙的Query类型分类。
5. 选取20条真实Query，包含最大的正Δ、较小正Δ、负Δ与并列，展示五臂实际成绩。

**注意：** `Δ`只表示五臂的真实效用关系，不表示这种差异能否从Query文本被模型预测。训练集里有高价值样本是必要条件，不是充分条件。

## 5. 第2步：在不训练新模型的情况下验证“漏学了高价值切换”

对B0、B1、B2分别计算，仍用**原有Checkpoint**：

### 5.1 切换能力

在`Δ>=0.20`且当前Query含有真正非MR更优臂的样本中：

- `switch_rate = P(predicted_arm != MR)`：Router肯不肯离开MR。
- `beneficial_switch_rate = P(r_predicted > r_MR)`：离开MR之后是否真的选到了能提高分数的臂；必须与单纯切换率分开。
- `captured_gain = mean(max(r_predicted-r_MR,0))`：模型抓住了多少正向切换收益。
- `missed_opportunity = mean(max(r_alt-r_predicted,0))`：距离五臂Oracle还损失多少（可补充按Query求差分）。

在`Δ<=0`的Query中：

- `unnecessary_switch_rate = P(predicted_arm != MR)`：MR不比任何备选差时，模型是否乱切换。
- `harmful_switch_rate = P(r_predicted < r_MR)`：切换是否真的造成负收益。

### 5.2 对固定MR的净收益分解（**最关键**）

\[
 g^+=\frac1N\sum_q\max(r_{\hat a(q)}-r_{MR},0),\qquad
 g^-=\frac1N\sum_q\max(r_{MR}-r_{\hat a(q)},0)
\]

\[
 \boxed{\operatorname{Gain}_{\text{vs MR}}=g^+-g^-}
\]

**大白话：** Router离开MR，有些题能挣分（`g⁺`），有些题反而丢分（`g⁻`）。我们要的是“挣的比丢的多”，不是简单提高换臂次数。

按源域source_test与TAT-DQA/Wiki-SS/ArxivQA分别计算`g+、g−、net`，再对三目标库报告宏平均；同时统计固定MR的Oracle Gap：`mean(max_a r_a-rMR)`。

### 5.3 “模型到底知不知道有机会”的离线判别

如果模型保留完整5维输出，还应在源域validation和目标域做诊断：

- B1/B2预测的`max_{a!=MR} score_a - score_MR`，能否将真实`Δ>=0.2`样本排在更靠前的位置？报告**AUPRC**（以非MR高价值样本为正类）、AUROC和正类基率；不要只报Accuracy。
- 与简单的`永远选MR`、`随机按源域臂频率选`比较；特别报告不同目标库，防止宏平均遮住Wiki-SS。
- 分别给出“正确切换、漏切换、错误切换”真实Query案例，每类至少10条（不足则全部），**自动读取r[predicted_arm]**，不手写得分。

预测打分只能在模型内部按臂比较。若B1/B2输出采用不同变换，需使用各自推理一致的数值定义。

## 6. 第3步：区分三个竞争解释

| 可能机制 | 期待看到的结果 | 对研究方向意味着什么 |
|---|---|---|
| **样本稀缺/默认MR偏置** | 源域High-value switch比例低；B2很少给非MR高分；目标库漏切换严重 | 值得研究稀有高收益偏好样本的训练组织，但先需区分纯类别不平衡Baseline |
| **可学习信号不足** | 源域有很多高价值样本，但B1/B2的预测优势排序AUPRC依然接近随机，尤其跨库变差 | 改Loss或重采样可能作用有限；需要更好的Query表征/输入信息 |
| **已有模型能识别机会，但argmax决策偏MR** | 预测分数对高价值样本的排序有区分力，实际argmax却常选MR | 先做验证集校准或代价敏感决策，不必重训大模型；新论文需超越普通阈值调节 |

**强调：** “高价值样本稀少”与“训练时过度相信MR”不能只凭ArxivQA上100%选MR判断，必须用源域样本分布和逐Query预测证据来支撑。

## 7. Agent交付物

- `sanity_report.md`：模型/数据/五臂列顺序/Bad Case映射的核查，含B0旧版与新版差异；结论：pass/fail。
- `train_switch_opportunities.csv`：源域train/val逐Query的`r_MR, r_best_nonMR, Δ, best_nonMR_arm, bucket`及各模型预测（若存在）。
- `switch_opportunity_summary.csv`：按`dataset,split,bucket`统计数量、比例、备选臂构成和潜在Oracle增益。
- `router_switch_diagnostics.csv`：按`dataset,model,seed,bucket`统计切换率、有益/有害切换率、`g+`,`g-`,net、mean regret。
- `switch_signal_quality.csv`：AUPRC、AUROC、真实正类率等；若缺5维模型打分，应注明未能计算，不能补造结果。
- `switch_badcases.md`：正确切换、漏切换、错误切换各≥10例，逐项列出原Query、真实五臂nDCG、MR基线nDCG、模型选择与真实收益。
- `switch_diagnostic_report.md`：**一页结论表 + 两张图**（源域高价值样本比例；各模型正负切换收益分解），列出支持/不支持的假设。

## 8. 是否值得进入下一轮训练？

**继续做“稀有高价值切换学习”需同时满足：**

1. 源域确实存在少数但不可忽略的高价值非MR样本，且它们在目标域也贡献了可观的固定MR Oracle Gap；
2. B1/B2在这些样本上确实存在显著漏切换，能够解释它们不如固定MR或未能超过固定MR的原因；
3. 至少有一定可学信号（模型预测优势与真实优势存在高于随机的排序能力），否则直接重加权很可能只能让模型更常乱切换。

满足后再设计**Preference Contrast / Counterfactual Utility Training**等真正针对高价值少数样本的新机制，同时加入简单的类别均衡、样本重加权和普通Pairwise Loss作为强基线。

不满足则停止围绕“MR默认偏置”堆Loss，转向训练覆盖、信息缺失或Retriever能力表示研究。

## 9. 最短执行顺序

**核验逐Query数据和B0 → 提取源域高价值非MR样本 → 计算B0/B1/B2切换命中率和净收益 → 在三目标库分析是否稳定 → 给出是否继续新训练的明确结论。**

> **本轮不调用目标域标签选择阈值、模型或checkpoint。** 这些目标库已多轮用于研究假设形成，属于探索性OOD开发集；后续正式论文还需要全新未参与研究决策的盲测语料库。
