"""Stratified case sampling for the human-review casebook (plan section 4).

Six strata, NOT just the top-100 worst:
  S1 high-confidence severe errors
  S2 modality errors (both directions)
  S3 rerank harm / missed rerank gain
  S4 one severe case per DATASET (cross-domain recurrence)
  S5 near-tie harmless-disagreement controls
  S6 all-five-arms-fail controls

Writes analysis/casebook.md. Every case carries the fields required by the plan:
query -> dataset -> 5 probs -> 5 nDCG -> picked/best -> hypothesis -> numeric evidence
-> verifiable intervention. Page-level verification needs HF corpus pages and is flagged.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
ANALYSIS_DIR = os.path.join(REPO_ROOT, "analysis")

ARMS = ["MR", "MD", "TR", "TD", "BM"]
RNG = 42


def sample(sub, n, seed=RNG):
    if len(sub) <= n:
        return sub
    return sub.sample(n=n, random_state=seed)


def case_block(cid, row, hypothesis, intervention, page_needed=True):
    probs = " ".join(f"{a}={row[f'prob_{a}']:.3f}" for a in ARMS)
    nds = " ".join(f"{a}={row[f'ndcg_{a}']:.3f}" for a in ARMS)
    flags = []
    if row["has_figure_ref"]:
        flags.append("figure-word")
    if row["has_number"]:
        flags.append("number")
    if row["has_comparison"]:
        flags.append("comparison")
    flag_s = f", features: {', '.join(flags)}" if flags else ""
    return f"""#### {cid} `{row['DATASET']}` — picked **{row['pred_arm']}**, nDCG-best **{row['best_ndcg_arm']}**, reward-best **{row['best_reward_arm']}**

- **Query**: {row['query']}
- DATASET: `{row['DATASET']}`{flag_s}; qlen={row['qlen_words']} words
- probs: {probs}
- nDCG@5: {nds}
- picked nDCG={row['pred_ndcg']:.3f}, 5-arm oracle={row['oracle5_ndcg']:.3f}, quality_regret={row['quality_regret']:.3f}, reward_regret={row['reward_regret']:.3f}
- confidence={row['confidence']:.3f}, reward margin={row['margin_reward']:.3f}; lambda arms l00/l10/l70 = {row['arm_l00']}/{row['pred_arm']}/{row['arm_l70']}
- **失败假设**: {hypothesis}
- **数值证据**: {row['evidence']}
- **可验证干预**: {intervention}
- 页面级复核: {'需要（HF 数据集相关页 Top-K / 重排前后顺序）' if page_needed else '不需要（逐 Query 指标已足够定性）'}
"""


def main():
    m = pd.read_csv(os.path.join(ANALYSIS_DIR, "query_metrics.csv"))
    severe_thr = m["reward_regret"].quantile(0.90)
    m["severe"] = m["reward_regret"] >= severe_thr
    m["hc_severe"] = m["severe"] & (m["confidence"] >= 0.50)

    parts = []
    parts.append("# RetrievalRouter Bad Case 人工复核 Casebook（λ=0.1，8,506 条测试 Query）\n")
    parts.append(
"> 生成方式：`analysis/build_casebook.py` 基于 `query_metrics.csv` **分层随机抽样（seed=42）**，"
"共 63 个 Case，覆盖计划第四节的 6 类群体。每个 Case 的五维概率/五臂 nDCG/regret/margin 均直接取自官方预测文件并经脚本重算。\n"
">\n"
"> **复核状态**：以下“失败假设”是基于逐 Query 数值证据的机器初判；标注“需要”的 Case 必须按计划第四节去 Hugging Face 对应数据集拉取相关页面、"
"必要时仅对选定 Case 重跑目标 Pipeline 保存 Top-K 页面/重排前后顺序，才能把假设升级为已核实结论。\n")

    # ---------------- S1 high-confidence severe -------------------------------------------
    s1 = sample(m[m["hc_severe"]].sort_values("reward_regret", ascending=False), 8)
    s1 = s1.copy(); s1["evidence"] = s1.apply(
        lambda r: f"置信度 {r['confidence']:.2f} 处于全模型最高档（P99≈0.45，max=0.80）却出现 reward_regret={r['reward_regret']:.2f}", axis=1)
    parts.append("## S1. 高置信度 + 高 regret（8 例；全量仅 17 例，均在 `top_badcases.csv`）\n")
    for _, r in s1.iterrows():
        hyp = "Softmax 置信度与真实效用脱钩：模型在错误臂上给出全局最高档概率，不存在“知道自己不确定”的信号。"
        iv = "在该 Query 上对比温度校准 / 集成方差 / 代价敏感阈值；若校准后仍高置信，则需文档侧探测信号而非仅重标定。"
        parts.append(case_block(f"S1-{r.name}", r, hyp, iv))

    # ---------------- S2 modality errors --------------------------------------------------
    v = m[(m["modality_true"] == "visual") & (m["modality_pred"] == "text")]
    t = m[(m["modality_true"] == "text") & (m["modality_pred"] == "visual")]
    s2v = sample(v.sort_values("quality_regret", ascending=False), 6).copy()
    s2t = sample(t.sort_values("quality_regret", ascending=False), 6).copy()
    s2v["evidence"] = s2v.apply(
        lambda r: f"视觉臂最佳 {r['vis_best_ndcg']:.2f} vs 文本臂最佳 {r['text_best_ndcg']:.2f}；实际选中 {r['pred_arm']}={r['pred_ndcg']:.2f}", axis=1)
    s2t["evidence"] = s2t.apply(
        lambda r: f"文本臂最佳 {r['text_best_ndcg']:.2f} vs 视觉臂最佳 {r['vis_best_ndcg']:.2f}；实际选中 {r['pred_arm']}={r['pred_ndcg']:.2f}", axis=1)
    parts.append("## S2a. 真正需要视觉却路由到文本臂（6 例 / 全量 118 例，累计 qr=66.1）\n")
    for _, r in s2v.iterrows():
        hyp = "Query-only 无法判断证据载体：答案在表格/图表图片中（扫描件或渲染表格），文本索引缺失该证据。"
        iv = "增加一次廉价探测（如页面中该区域是否为图片表格 / OCR 覆盖率）作为模态特征；与“永远选视觉”的简单基线比较成本。"
        parts.append(case_block(f"S2a-{r.name}", r, hyp, iv))
    parts.append("## S2b. 文本即可却过度路由到视觉臂（6 例 / 全量 473 例，累计 qr=233.7）\n")
    for _, r in s2t.iterrows():
        hyp = ("领域/指代词触发视觉捷径（如出现 figure/财务术语被当作文档视觉需求）；实际证据在正文文本，"
               "视觉编码器引入噪声。" if r["has_figure_ref"] or r["DATASET"] in ("finqa", "tatdqa", "convfinqa")
               else "事实性/概念性 Query 被误送入视觉臂；词法匹配本可命中。")
        iv = "做 Query 关键词扰动（增删 figure/chart/年份）测决策翻转；留出单一数据集训练/测试验证领域捷径。"
        parts.append(case_block(f"S2b-{r.name}", r, hyp, iv))

    # ---------------- S3 rerank ------------------------------------------------------------
    harm = m[m["rerank_harm_chosen"]]
    miss = m[((m["pred_arm"] == "TD") & (m["rerank_gain_text"] > 0.02)) |
             ((m["pred_arm"] == "MD") & (m["rerank_gain_mm"] > 0.02))]
    s3a = sample(harm.sort_values("quality_regret", ascending=False), 8).copy()
    s3b = sample(miss.sort_values("quality_regret", ascending=False), 8).copy()
    s3a["evidence"] = s3a.apply(
        lambda r: (f"MR={r['ndcg_MR']:.2f} 而 MD={r['ndcg_MD']:.2f}（重排把正确页挤出 Top-5）"
                   if r["pred_arm"] == "MR"
                   else f"TR={r['ndcg_TR']:.2f} 而 TD={r['ndcg_TD']:.2f}（重排把正确页挤出 Top-5）"), axis=1)
    s3b["evidence"] = s3b.apply(
        lambda r: (f"MD={r['ndcg_MD']:.2f} 而 MR={r['ndcg_MR']:.2f}（该 Query 上重排本可救回）"
                   if r["pred_arm"] == "MD"
                   else f"TD={r['ndcg_TD']:.2f} 而 TR={r['ndcg_TR']:.2f}（该 Query 上重排本可救回）"), axis=1)
    parts.append("## S3a. 选择 Rerank 但 Rerank 相对 Dense 显著恶化（8 例 / 全量 690 例，累计 qr=415.9）\n")
    for _, r in s3a.iterrows():
        hyp = "候选集结构（数字/表格/长实体）导致 cross-encoder 对正确页误降权；Rerank 增益在 Query 层面可预测地失败。"
        iv = "导出该 Query 的 Dense Top-20 与重排前后顺序，验证正确页掉位模式；用候选集统计（数字密度、表格占比、词项覆盖）训练轻量 rerank 开关。"
        parts.append(case_block(f"S3a-{r.name}", r, hyp, iv))
    parts.append("## S3b. 选择 Dense 但 Rerank 本可显著增益（8 例 / 全量 782 例，累计 qr=435.6）\n")
    for _, r in s3b.iterrows():
        hyp = "与 S3a 对偶的开关错误：路由器对 rerank 增益方向判断不稳定（λ 三版本间臂变化亦可佐证）。"
        iv = "同一 rerank 开关模型必须同时减少 S3a 与 S3b；以两类合并净 nDCG 增益与延迟成本作为验收。"
        parts.append(case_block(f"S3b-{r.name}", r, hyp, iv))

    # ---------------- S4 one severe per dataset -------------------------------------------
    idx = m[m["severe"]].groupby("DATASET")["quality_regret"].idxmax()
    s4 = m.loc[idx].copy()
    s4["evidence"] = s4.apply(
        lambda r: f"该数据集中 quality_regret 最大的严重错误：picked {r['pred_arm']}={r['pred_ndcg']:.2f}, "
                  f"best {r['best_ndcg_arm']}={r['oracle5_ndcg']:.2f}", axis=1)
    parts.append("## S4. 跨数据集复现：每个数据集损失最大的严重错误（11 例）\n")
    for _, r in s4.iterrows():
        hyp = "检验错误机制是否跨域共性：若多个数据集的最坏 Case 同属 rerank 掉位或模态误判，则机制不是单一领域噪声。"
        iv = "按本 Casebook 归类标签做跨域统计；后续在留出数据集上验证干预效果（当前测试集只用于假设生成）。"
        parts.append(case_block(f"S4-{r['DATASET']}", r, hyp, iv))

    # ---------------- S5 near-tie controls -------------------------------------------------
    tie = m[(~m["arm_match_reward"]) & (m["reward_regret"] < 0.01)]
    s5 = sample(tie, 8).copy()
    s5["evidence"] = s5.apply(
        lambda r: f"选中臂与 reward 最优臂差距仅 {r['reward_regret']:.4f}（margin={r['margin_reward']:.3f}），属并列最优", axis=1)
    parts.append("## S5. 对照组：多臂近似并列的“伪错误”（8 例 / 全量 889 例）\n")
    for _, r in s5.iterrows():
        hyp = "argmax 翻转但效用几乎无差；不应计入严重错误，也不应作为训练改进的目标样本。"
        iv = "评估与选模型时使用期望 regret / soft 指标而非 argmax 准确率；训练中对此类样本降权。"
        parts.append(case_block(f"S5-{r.name}", r, hyp, iv, page_needed=False))

    # ---------------- S6 all-fail controls -------------------------------------------------
    af = m[m["oracle5_ndcg"] < 0.25]
    s6 = sample(af, 8).copy()
    s6["evidence"] = s6.apply(
        lambda r: f"五臂最佳 nDCG 仅 {r['oracle5_ndcg']:.3f}：无任何可选臂能找到证据（不可路由问题）", axis=1)
    parts.append("## S6. 对照组：五路全部失败（8 例 / 全量 371 例，仅贡献 0.1% regret）\n")
    for _, r in s6.iterrows():
        hyp = "失败在检索器/语料层（页内指代、图内细节、语料缺失），Router 无信息可利用；此类 Query 应廉价失败。"
        iv = "检验模型是否对此类 Query 回退到最低延迟臂（BM25）；这是延迟优化而非 nDCG 优化问题，不计入 Router 责任。"
        parts.append(case_block(f"S6-{r.name}", r, hyp, iv, page_needed=False))

    parts.append(
"\n---\n\n## 复核工作量建议\n\n"
"- 优先页面级复核顺序：**S3a（4 例）→ S2a（4 例）→ S2b（3 例）→ S1（2 例）**，共 13 例即可初步验证三个主假设；\n"
"- 每个页面级 Case 保存：相关页 ID、Dense Top-20、Rerank Top-5 及前后排名、页面类型（正文/表格/扫描图/图表）；\n"
"- 全部 63 例的数值字段可由 `query_metrics.csv` 按 query 直接复得，本文件不复制全量列。\n")

    out = os.path.join(ANALYSIS_DIR, "casebook.md")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote {out}: S1=8 S2=12 S3=16 S4=11 S5=8 S6=8 (total 63)")


if __name__ == "__main__":
    main()
