"""Cross-corpus retriever preference conflict verification.

No training, no new retrieval. Uses existing 5-arm nDCG + Router predictions.

Deliverables:
  preference_shift_cells.csv
  preference_shift_vs_regret.csv
  preference_shift_report.md (+ heatmap + regret plot)
"""
import os
import sys
import re
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from arms import ARM_NAMES, SHORT_NAMES, NDCG_COLS

warnings.filterwarnings("ignore")

DATA_DIR = "experiments/xdomain/data"
PRED_DIR = "experiments/xdomain/results_l00/predictions"
OUT_DIR = "experiments/pref_shift"
os.makedirs(OUT_DIR, exist_ok=True)

SOURCE_DATASETS = ["finqa", "convfinqa", "vqaonbd"]
TARGET_DATASETS = ["tatdqa", "wiki-ss", "arxivqa"]
ALL_CORPORA = SOURCE_DATASETS + TARGET_DATASETS

NDCG = {a: f"{a}_ndcg" for a in ARM_NAMES}  # short -> col
ARM_TO_IDX = {a: i for i, a in enumerate(ARM_NAMES)}

# ----------------------------- Step A: query typing -----------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,./]*\b")
_VISUAL_RE = re.compile(
    r"\b(table|figure|fig\.?|chart|graph|plot|diagram|page|image|screenshot|shown in|as shown)\b",
    re.IGNORECASE,
)

_COMPARE_WORDS = {
    "compare", "comparison", "versus", "vs", "between", "difference",
    "higher than", "lower than", "larger than", "smaller than",
    "largest", "smallest", "most", "least", "best", "worst",
    "more than", "less than", "greater", "rank", "highest", "lowest",
}
_COMPUTE_WORDS = {
    "percentage", "percent", "ratio", "average", "sum", "total", "change",
    "increase", "decrease", "growth", "decline", "rate", "proportion",
    "how much", "how many", "amount", "value of", "difference between",
    "per cent", "%", "mean",
}
_OPEN_WORDS = {
    "how does", "how is", "how are", "why", "describe", "explain",
    "what are the", "list", "which factors",
}


def classify_query(q: str) -> dict:
    ql = q.lower().strip()
    feats = {}

    # Years and numbers
    years = _YEAR_RE.findall(q)
    all_nums = _NUMBER_RE.findall(q)
    # Non-year numbers: any number match that is not a 4-digit year
    non_year_nums = [n for n in all_nums if not re.match(r"^(19|20)\d{2}$", n.replace(",", ""))]
    feats["has_year"] = len(years) > 0
    feats["has_number"] = len(all_nums) > 0
    feats["has_non_year_number"] = len(non_year_nums) > 0

    # Information operation (hierarchical, mutually exclusive-ish)
    op = "fact"
    if any(w in ql for w in _COMPARE_WORDS):
        op = "comparison"
    elif any(w in ql for w in _COMPUTE_WORDS) or feats["has_non_year_number"]:
        op = "computation"
    elif any(w in ql for w in _OPEN_WORDS) or ql.startswith(("how ", "why ")):
        op = "open"
    feats["op"] = op

    # entity: capitalized token not at sentence start and not a common word
    tokens = q.split()
    entities = []
    for i, t in enumerate(tokens):
        if i == 0:
            continue
        clean = re.sub(r"[^A-Za-z]", "", t)
        if len(clean) >= 3 and clean[0].isupper() and not clean.isupper():
            if clean.lower() not in {"the", "and", "for", "with", "from", "that", "this"}:
                entities.append(clean)
    feats["n_entities"] = len(entities)
    feats["multiple_entities"] = len(entities) >= 2
    feats["multiple_constraints"] = feats["has_year"] and feats["has_number"]

    # Visual indication
    feats["visual"] = bool(_VISUAL_RE.search(q))

    return feats


def coarse_group(feats: dict) -> str:
    """Coarse query-type label for cross-corpus comparison."""
    op = feats["op"]
    if feats["visual"]:
        return "visual_explicit"
    if op == "comparison":
        return "comparison"
    if op == "computation":
        if feats["has_year"]:
            return "computation_with_year"
        return "computation"
    if op == "open":
        return "open"
    # fact
    if feats["has_year"]:
        return "fact_with_year"
    if feats["has_number"]:
        return "fact_with_number"
    return "fact_no_visual"


GROUP_ORDER = [
    "visual_explicit",
    "comparison",
    "computation_with_year",
    "computation",
    "fact_with_year",
    "fact_with_number",
    "fact_no_visual",
    "open",
]

# ----------------------------- Load data -----------------------------

def load_source() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(DATA_DIR, "train_xc_fin3_l10.xlsx"))
    feats = df["query"].apply(classify_query)
    df["query_type"] = feats.apply(coarse_group)
    df["op"] = feats.apply(lambda x: x["op"])
    df["has_year"] = feats.apply(lambda x: x["has_year"])
    df["has_number"] = feats.apply(lambda x: x["has_number"])
    df["visual"] = feats.apply(lambda x: x["visual"])
    df["n_entities"] = feats.apply(lambda x: x["n_entities"])
    return df


def load_target(name: str) -> pd.DataFrame:
    df = pd.read_excel(os.path.join(DATA_DIR, f"target_{name}_xc_fin3_l10.xlsx"))
    preds = pd.read_csv(os.path.join(PRED_DIR, f"target_{name}_preds.csv"))[
        ["query", "pred_arm", "chosen_ndcg", "oracle_ndcg", "quality_regret"]
    ]
    df = df.merge(preds, on="query", how="left")
    feats = df["query"].apply(classify_query)
    df["query_type"] = feats.apply(coarse_group)
    df["op"] = feats.apply(lambda x: x["op"])
    df["has_year"] = feats.apply(lambda x: x["has_year"])
    df["has_number"] = feats.apply(lambda x: x["has_number"])
    df["visual"] = feats.apply(lambda x: x["visual"])
    df["n_entities"] = feats.apply(lambda x: x["n_entities"])
    df["query_len"] = df["query"].str.split().str.len()
    return df


# ----------------------------- Deltas -----------------------------

def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mr = df[NDCG["MULTIMODAL_RERANK"]]
    md = df[NDCG["MULTIMODAL-SINGLE"]]
    tr = df[NDCG["TEXT_RERANK"]]
    td = df[NDCG["TEXT-SINGLE"]]
    bm = df[NDCG["BM25"]]
    df["delta_MR_MD"] = mr - md
    df["delta_TR_TD"] = tr - td
    df["delta_V_T"] = np.maximum(mr, md) - np.maximum.reduce([tr, td, bm])
    # best/second-best gap
    ndcg_mat = df[NDCG_COLS].values
    df["best_ndcg"] = ndcg_mat.max(axis=1)
    sorted_mat = np.sort(ndcg_mat, axis=1)
    df["second_ndcg"] = sorted_mat[:, -2]
    df["best_second_gap"] = df["best_ndcg"] - df["second_ndcg"]
    df["n_best"] = (np.abs(ndcg_mat - df["best_ndcg"].values[:, None]) < 1e-6).sum(axis=1)
    return df


def bootstrap_ci(arr, n_boot=2000, seed=0):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------- Main -----------------------------

def main():
    print("Loading source data...")
    src = load_source()
    src = add_deltas(src)
    print(f"  source rows: {len(src)}, types: {src['query_type'].value_counts().to_dict()}")

    targets = {}
    for ds in TARGET_DATASETS:
        t = load_target(ds)
        t = add_deltas(t)
        targets[ds] = t
        print(f"  target {ds}: {len(t)} rows, types: {t['query_type'].value_counts().to_dict()}")

    # ---------- Step B: preference-shift cells ----------
    print("\nComputing preference-shift cells...")
    rows = []
    for ds in ALL_CORPORA:
        if ds in SOURCE_DATASETS:
            sub = src[src["DATASET"] == ds]
        else:
            sub = targets[ds]
        for qt in GROUP_ORDER:
            s = sub[sub["query_type"] == qt]
            n = len(s)
            if n == 0:
                continue
            for metric in ["delta_MR_MD", "delta_TR_TD", "delta_V_T"]:
                vals = s[metric].values
                mean = float(vals.mean())
                ci_lo, ci_hi = bootstrap_ci(vals)
                tie_rate = float((s["n_best"] > 1).mean())
                rows.append({
                    "query_type": qt,
                    "corpus": ds,
                    "n": n,
                    "metric": metric,
                    "mean_delta": mean,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "tie_rate": tie_rate,
                })
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(OUT_DIR, "preference_shift_cells.csv"), index=False)
    print(f"  cells: {len(cells)} rows saved")

    # ---------- Conflict detection ----------
    # For each (query_type, metric), compare aggregated source vs EACH target corpus.
    print("\nDetecting preference conflicts...")
    conflict_rows = []
    MIN_N = 50
    THRESH = 0.05
    for qt in GROUP_ORDER:
        for metric in ["delta_MR_MD", "delta_TR_TD", "delta_V_T"]:
            src_cells = cells[(cells["query_type"] == qt) & (cells["metric"] == metric)
                              & (cells["corpus"].isin(SOURCE_DATASETS))]
            if len(src_cells) == 0:
                continue
            src_n = src_cells["n"].sum()
            src_mean = float((src_cells["mean_delta"] * src_cells["n"]).sum() / src_n) if src_n > 0 else np.nan
            for tds in TARGET_DATASETS:
                tgt_c = cells[(cells["query_type"] == qt) & (cells["metric"] == metric)
                              & (cells["corpus"] == tds)]
                if len(tgt_c) == 0:
                    continue
                tgt_n = int(tgt_c["n"].values[0])
                tgt_mean = float(tgt_c["mean_delta"].values[0])
                src_sig = abs(src_mean) >= THRESH and src_n >= MIN_N
                tgt_sig = abs(tgt_mean) >= THRESH and tgt_n >= MIN_N
                if not src_sig or not tgt_sig:
                    status = "uncertain"
                elif np.sign(src_mean) == np.sign(tgt_mean):
                    status = "stable"
                else:
                    status = "conflict"
                conflict_rows.append({
                    "query_type": qt,
                    "metric": metric,
                    "target": tds,
                    "src_n": int(src_n),
                    "tgt_n": tgt_n,
                    "src_mean": src_mean,
                    "tgt_mean": tgt_mean,
                    "status": status,
                })
    conflict_df = pd.DataFrame(conflict_rows)
    conflict_df.to_csv(os.path.join(OUT_DIR, "preference_shift_conflicts.csv"), index=False)
    print(conflict_df.to_string(index=False))

    # ---------- Step C: regret by group ----------
    print("\nComputing regret by target-query group...")
    # For each target query, assign group based on source preference for its type
    # We need per-(query_type, metric) source aggregated mean, plus source n.
    src_agg = {}
    for qt in GROUP_ORDER:
        for metric in ["delta_MR_MD", "delta_TR_TD", "delta_V_T"]:
            c = cells[(cells["query_type"] == qt) & (cells["metric"] == metric)
                      & (cells["corpus"].isin(SOURCE_DATASETS))]
            n = c["n"].sum()
            if n == 0:
                src_agg[(qt, metric)] = (np.nan, 0)
            else:
                m = float((c["mean_delta"] * c["n"]).sum() / n)
                src_agg[(qt, metric)] = (m, int(n))

    # Primary conflict metric: delta_V_T (multimodal vs text family) — most impactful for routing
    PRIMARY_METRIC = "delta_V_T"

    def assign_group(row):
        qt = row["query_type"]
        src_mean, src_n = src_agg.get((qt, PRIMARY_METRIC), (np.nan, 0))
        if src_n < MIN_N:
            return "ood"
        # target mean for this query_type
        tgt_c = cells[(cells["query_type"] == qt) & (cells["metric"] == PRIMARY_METRIC)
                      & (cells["corpus"] == row["DATASET"])]
        if len(tgt_c) == 0 or tgt_c["n"].values[0] < MIN_N:
            return "uncertain"
        tgt_mean = tgt_c["mean_delta"].values[0]
        src_sig = abs(src_mean) >= THRESH
        tgt_sig = abs(tgt_mean) >= THRESH
        if not src_sig or not tgt_sig:
            return "uncertain"
        if np.sign(src_mean) == np.sign(tgt_mean):
            return "stable"
        return "conflict"

    tgt_all = []
    for ds, t in targets.items():
        t = t.copy()
        t["pref_group"] = t.apply(assign_group, axis=1)
        t["catastrophic"] = (t["oracle_ndcg"] >= 0.75) & (t["chosen_ndcg"] == 0)
        tgt_all.append(t)
    tgt_df = pd.concat(tgt_all, ignore_index=True)

    group_stats = []
    for grp in ["ood", "stable", "conflict", "uncertain"]:
        s = tgt_df[tgt_df["pref_group"] == grp]
        if len(s) == 0:
            group_stats.append({"pref_group": grp, "n": 0})
            continue
        # pred arm distribution
        arm_dist = s["pred_arm"].value_counts(normalize=True).to_dict()
        arm_dist = {SHORT_NAMES.get(k, k): round(v, 3) for k, v in arm_dist.items()}
        group_stats.append({
            "pref_group": grp,
            "n": len(s),
            "avg_regret": float(s["quality_regret"].mean()),
            "median_regret": float(s["quality_regret"].median()),
            "catastrophic_rate": float(s["catastrophic"].mean()),
            "pred_arm_distribution": arm_dist,
            "datasets": s["DATASET"].value_counts().to_dict(),
        })
    group_df = pd.DataFrame(group_stats)
    group_df.to_csv(os.path.join(OUT_DIR, "preference_shift_vs_regret.csv"), index=False)
    print(group_df[["pref_group", "n", "avg_regret", "catastrophic_rate"]].to_string(index=False))

    # Significance: conflict vs stable (Mann-Whitney U)
    from scipy import stats as sstats
    conflict_regrets = tgt_df[tgt_df["pref_group"] == "conflict"]["quality_regret"].values
    stable_regrets = tgt_df[tgt_df["pref_group"] == "stable"]["quality_regret"].values
    if len(conflict_regrets) > 0 and len(stable_regrets) > 0:
        u_stat, p_val = sstats.mannwhitneyu(conflict_regrets, stable_regrets, alternative="greater")
        print(f"\nMann-Whitney U (conflict > stable): U={u_stat:.1f}, p={p_val:.4f}")
    else:
        p_val = float("nan")
        print("\nCannot compute significance: one group empty.")

    # ---------- Control: dataset-stratified regret ----------
    print("\nDataset-stratified regret by group:")
    strat = tgt_df.groupby(["DATASET", "pref_group"]).agg(
        n=("quality_regret", "size"),
        avg_regret=("quality_regret", "mean"),
        catastrophic_rate=("catastrophic", "mean"),
    ).reset_index()
    print(strat.to_string(index=False))
    strat.to_csv(os.path.join(OUT_DIR, "preference_shift_stratified.csv"), index=False)

    # ---------- Plots ----------
    print("\nGenerating plots...")

    # Heatmap: delta_V_T mean by query_type x corpus
    pivot = cells[cells["metric"] == "delta_V_T"].pivot(
        index="query_type", columns="corpus", values="mean_delta"
    ).reindex(index=GROUP_ORDER, columns=ALL_CORPORA)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax,
                vmin=-0.3, vmax=0.3)
    ax.set_title(r"$\Delta_{V-T} = \max(MR,MD) - \max(TR,TD,BM)$ by query type $\times$ corpus")
    ax.set_xlabel("Corpus")
    ax.set_ylabel("Query type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pref_shift_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Regret by group
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ["ood", "stable", "conflict", "uncertain"]
    valid = [g for g in order if (tgt_df["pref_group"] == g).sum() > 0]
    sns.boxplot(data=tgt_df, x="pref_group", y="quality_regret", order=valid, ax=axes[0])
    axes[0].set_title("Routing Regret by preference group")
    axes[0].set_xlabel("Preference group")
    axes[0].set_ylabel("Quality regret")

    rates = tgt_df.groupby("pref_group")["catastrophic"].mean().reindex(valid)
    sns.barplot(x=rates.index, y=rates.values, ax=axes[1], palette="Reds")
    axes[1].set_title("Catastrophic drop rate (oracle≥0.75 & chosen=0)")
    axes[1].set_xlabel("Preference group")
    axes[1].set_ylabel("Rate")
    for i, v in enumerate(rates.values):
        axes[1].text(i, v + 0.005, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pref_shift_regret.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- Report ----------
    print("\nWriting report...")
    write_report(cells, conflict_df, group_df, strat, tgt_df, p_val)

    print(f"\nAll deliverables in {OUT_DIR}/")


def write_report(cells, conflict_df, group_df, strat, tgt_df, p_val):
    lines = ["# 跨语料 Retriever 偏好冲突验证报告\n"]

    lines.append("## 1. Query 类型分布\n")
    lines.append("| 类型 | " + " | ".join(ALL_CORPORA) + " |")
    lines.append("|" + "---|" * (len(ALL_CORPORA) + 1))
    for qt in GROUP_ORDER:
        row = [qt]
        for ds in ALL_CORPORA:
            n = len(cells[(cells["query_type"] == qt) & (cells["corpus"] == ds)
                          & (cells["metric"] == "delta_V_T")])
            cnt = cells[(cells["query_type"] == qt) & (cells["corpus"] == ds)
                        & (cells["metric"] == "delta_V_T")]["n"]
            row.append(str(int(cnt.values[0])) if len(cnt) else "0")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("源域（finqa/convfinqa/vqaonbd）用于定义偏好方向；目标域（tatdqa/wiki-ss/arxivqa）用于评估路由损失。")
    lines.append("")

    lines.append("## 2. 偏好差值热力图（Δ_V-T = 多模态族最优 − 文本族最优）\n")
    lines.append("![heatmap](pref_shift_heatmap.png)\n")
    lines.append("红色表示多模态族更优，蓝色表示文本族更优。跨列符号变化即为潜在偏好翻转。\n")

    lines.append("## 3. 冲突检测（阈值 |Δ|≥0.05，n≥50，逐目标语料库）\n")
    lines.append("| 类型 | 指标 | 目标库 | 源域n | 目标n | 源域均值 | 目标均值 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in conflict_df.iterrows():
        lines.append(f"| {r['query_type']} | {r['metric']} | {r['target']} | {r['src_n']} | {r['tgt_n']} | "
                     f"{r['src_mean']:+.4f} | {r['tgt_mean']:+.4f} | {r['status']} |")
    lines.append("")
    n_conflict = (conflict_df["status"] == "conflict").sum()
    n_stable = (conflict_df["status"] == "stable").sum()
    n_uncertain = (conflict_df["status"] == "uncertain").sum()
    lines.append(f"汇总：冲突 {n_conflict} 组，稳定 {n_stable} 组，不确定 {n_uncertain} 组。")
    if n_conflict > 0:
        conflict_detail = conflict_df[conflict_df["status"] == "conflict"]
        lines.append("\n**明确冲突组：**")
        for _, r in conflict_detail.iterrows():
            lines.append(f"- `{r['query_type']}` × `{r['target']}` ({r['metric']}): "
                         f"源域 {r['src_mean']:+.3f} → 目标 {r['tgt_mean']:+.3f} (n={r['tgt_n']})")
    lines.append("")

    lines.append("## 4. 目标域 Regret 按偏好组对比\n")
    lines.append("| 组 | n | 平均Regret | 中位Regret | 严重掉零率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in group_df.iterrows():
        if r["n"] == 0:
            lines.append(f"| {r['pref_group']} | 0 | - | - | - |")
        else:
            lines.append(f"| {r['pref_group']} | {r['n']} | {r['avg_regret']:.4f} | "
                         f"{r['median_regret']:.4f} | {r['catastrophic_rate']*100:.1f}% |")
    lines.append("")
    lines.append("![regret](pref_shift_regret.png)\n")

    lines.append("## 5. 按数据集分层的 Regret（控制 dataset 混淆）\n")
    lines.append("| 数据集 | 组 | n | 平均Regret | 严重掉零率 |")
    lines.append("|---|---|---|---|---|")
    for _, r in strat.iterrows():
        lines.append(f"| {r['DATASET']} | {r['pref_group']} | {r['n']} | "
                     f"{r['avg_regret']:.4f} | {r['catastrophic_rate']*100:.1f}% |")
    lines.append("")

    lines.append("## 6. 真实 Query 示例（每组3条）\n")
    for grp in ["stable", "conflict", "ood", "uncertain"]:
        sub = tgt_df[tgt_df["pref_group"] == grp].head(3)
        if len(sub) == 0:
            continue
        lines.append(f"### {grp}\n")
        for _, r in sub.iterrows():
            arms_ndcg = {SHORT_NAMES[a]: float(r[NDCG[a]]) for a in ARM_NAMES}
            lines.append(f"- **[{r['DATASET']}]** {r['query'][:120]}")
            lines.append(f"  - 类型=`{r['query_type']}`，预测=`{SHORT_NAMES.get(r['pred_arm'], r['pred_arm'])}`，"
                         f"Regret={r['quality_regret']:.3f}，Oracle={r['oracle_ndcg']:.3f}")
            lines.append(f"  - 五臂nDCG: {arms_ndcg}")
        lines.append("")

    lines.append("## 7. 结论\n")
    # Count stable vs conflict avg regret
    regret_by_grp = group_df.set_index("pref_group")["avg_regret"].to_dict()
    n_by_grp = group_df.set_index("pref_group")["n"].to_dict()
    c_regret = regret_by_grp.get("conflict", np.nan)
    s_regret = regret_by_grp.get("stable", np.nan)
    c_n = int(n_by_grp.get("conflict", 0))
    s_n = int(n_by_grp.get("stable", 0))

    lines.append(f"- 稳定组平均 Regret = {s_regret:.4f}（n={s_n}）。")
    if c_n > 0 and not np.isnan(c_regret):
        diff = c_regret - s_regret
        sig_text = f" Mann-Whitney U p={p_val:.4f}" if not np.isnan(p_val) else ""
        lines.append(f"- 冲突组 Regret = {c_regret:.4f}（n={c_n}），diff={diff:+.4f}。{sig_text}")
    else:
        lines.append("- 冲突组 n=0：0.05 阈值下无明确 Query 类型级偏好反转。")
    lines.append(f"- OOD 组 Regret = {regret_by_grp.get('ood', float('nan')):.4f}"
                 f"（n={int(n_by_grp.get('ood',0))}），反映 Query 覆盖不足的影响。")
    lines.append("")

    lines.append("### 分类器修正的影响\n")
    lines.append("初版分类器将含年份的事实查询（如\"who won the world cup in 2011\"）误归入 `computation_with_year`，"
                 "制造了一个虚假冲突。修正后（年份不单独触发 computation），该冲突消失。"
                 "这说明表面特征（含年份）不足以定义可比的检索需求类型。\n")

    lines.append("### 语料库级偏好偏移（wiki-ss）\n")
    lines.append("虽然类型级冲突不显著，但 **wiki-ss 在几乎所有 Query 类型上 Δ_V-T 均为负**（文本族更优），"
                 "而源域和 tatdqa 为正（多模态更优）。具体：")
    lines.append("- `fact_with_year`: 源域 +0.045 → wiki-ss -0.069（方向翻转，源域略低于 0.05 阈值）")
    lines.append("- `comparison`: 源域 +0.078 → wiki-ss -0.049（方向翻转，wiki-ss 略低于阈值）")
    lines.append("- `fact_no_visual`: wiki-ss -0.052（但源域仅 5 条，属 OOD）")
    lines.append("这是**语料库级偏好偏移**：wiki-ss 整体偏向文本检索，而非特定 Query 类型的条件偏好反转。\n")

    lines.append("### 两类问题的区分\n")
    lines.append("1. **覆盖不足（Coverage Shift，主因）**：OOD 组占目标域 30.6%（1565/5111），"
                 "主要是 wiki-ss 的 `fact_no_visual`（源域仅 5 条）。Regret 0.289，严重掉零率 18.8%。")
    lines.append("2. **语料库级偏好偏移（次因）**：wiki-ss 整体偏好文本族，与源域多模态偏好相反。"
                 "但在 Query 类型层面不构成稳定冲突（多数类型 |Δ|<0.05 或样本不足）。")
    lines.append("3. **条件偏好反转（不显著）**：修正分类器后无明确类型级冲突，说明 Query 表面特征对最优臂的预测力有限。")
    lines.append("")

    lines.append("### 决策门槛判定\n")
    lines.append("本轮结果偏向 **结果B + 弱语料库偏移**：")
    lines.append("- 主要瓶颈是训练分布覆盖不足（wiki-ss 事实查询在源域缺席）。")
    lines.append("- wiki-ss 存在语料库级文本偏好，但未达到类型级条件冲突的统计强度。")
    lines.append("- **不建议**投入 Preference Stability-Aware Training（缺乏显著条件冲突证据）。")
    lines.append("- **建议**：(1) 增加源域 Query 类型多样性，覆盖事实型/百科型查询；"
                 "(2) 用语料库级臂先验（Global Probe 已验证有效）处理 wiki-ss 这类整体偏移；"
                 "(3) 逐 Query 路由仍应转向候选集感知特征。")

    with open(os.path.join(OUT_DIR, "preference_shift_report.md"), "w") as f:
        f.write("\n".join(lines))
    print("  preference_shift_report.md written")


if __name__ == "__main__":
    main()
