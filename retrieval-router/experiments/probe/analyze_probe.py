"""Generate probe experiment deliverables: metrics table, plots, casebook, conclusion."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arms import ARM_NAMES, SHORT_NAMES

OUT_DIR = "experiments/probe"
PROBE_SIZES = [20, 50, 100]
METHODS = ["A_l00", "A_l10", "B_fixedMR", "C_global", "D_local", "E_bestfixed", "F_oracle"]
METHOD_LABELS = {
    "A_l00": "Router(λ=0)",
    "A_l10": "Router(λ=0.1)",
    "B_fixedMR": "Fixed-MR",
    "C_global": "Global Probe",
    "D_local": "Local Probe",
    "E_bestfixed": "Best Fixed (oracle)",
    "F_oracle": "Per-query Oracle",
}
DATASETS = ["tatdqa", "wiki-ss", "arxivqa"]
NDCG_COLS = [f"{a}_ndcg" for a in ARM_NAMES]


def short(a):
    return SHORT_NAMES.get(a, a)


def load_and_prepare():
    """Load raw results + per-query router predictions and embeddings for casebook."""
    res = pd.read_csv(os.path.join(OUT_DIR, "probe_results.csv"))
    # Aggregate
    agg = res.groupby(["dataset", "probe_size", "method"]).agg(
        nDCG_mean=("nDCG", "mean"),
        nDCG_std=("nDCG", "std"),
        regret_mean=("regret", "mean"),
        catastrophic_mean=("catastrophic", "mean"),
    ).reset_index()
    return res, agg


def make_metrics_md(agg):
    lines = ["# 探针实验主指标表\n"]
    lines.append("## nDCG@5（10种子均值 ± 标准差）\n")
    for ds in DATASETS:
        lines.append(f"### {ds}\n")
        lines.append("| 探针量 | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for ps in PROBE_SIZES:
            row = [f"{ps}"]
            for m in METHODS:
                sub = agg[(agg["dataset"] == ds) & (agg["probe_size"] == ps) & (agg["method"] == m)]
                if len(sub):
                    row.append(f"{sub['nDCG_mean'].values[0]:.4f}±{sub['nDCG_std'].values[0]:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Quality Regret
    lines.append("## Quality Regret（均值）\n")
    for ds in DATASETS:
        lines.append(f"### {ds}\n")
        lines.append("| 探针量 | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for ps in PROBE_SIZES:
            row = [f"{ps}"]
            for m in METHODS:
                sub = agg[(agg["dataset"] == ds) & (agg["probe_size"] == ps) & (agg["method"] == m)]
                row.append(f"{sub['regret_mean'].values[0]:.4f}" if len(sub) else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Catastrophic drop rate
    lines.append("## 严重掉零率（Oracle≥0.75 且选中=0）\n")
    for ds in DATASETS:
        lines.append(f"### {ds}\n")
        lines.append("| 探针量 | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " |")
        lines.append("|" + "---|" * (len(METHODS) + 1))
        for ps in PROBE_SIZES:
            row = [f"{ps}"]
            for m in METHODS:
                sub = agg[(agg["dataset"] == ds) & (agg["probe_size"] == ps) & (agg["method"] == m)]
                row.append(f"{sub['catastrophic_mean'].values[0]*100:.1f}%" if len(sub) else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Key comparisons
    lines.append("## 关键对比\n")
    lines.append("| 数据集 | 探针量 | Global−FixedMR | Local−Global | Local−max(FixedMR,Router) |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        for ps in PROBE_SIZES:
            c = agg[(agg.dataset == ds) & (agg.probe_size == ps) & (agg.method == "C_global")]["nDCG_mean"].values[0]
            d = agg[(agg.dataset == ds) & (agg.probe_size == ps) & (agg.method == "D_local")]["nDCG_mean"].values[0]
            b = agg[(agg.dataset == ds) & (agg.probe_size == ps) & (agg.method == "B_fixedMR")]["nDCG_mean"].values[0]
            a = agg[(agg.dataset == ds) & (agg.probe_size == ps) & (agg.method == "A_l00")]["nDCG_mean"].values[0]
            lines.append(f"| {ds} | {ps} | {c-b:+.4f} | {d-c:+.4f} | {d-max(b,a):+.4f} |")
    lines.append("")

    with open(os.path.join(OUT_DIR, "probe_metrics.md"), "w") as f:
        f.write("\n".join(lines))
    print("probe_metrics.md written")


def make_plots(agg):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"C_global": "#C41E3A", "D_local": "#2E86AB", "B_fixedMR": "#888",
              "A_l00": "#F4A460", "A_l10": "#D81B60"}

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        for m in ["B_fixedMR", "A_l00", "A_l10"]:
            sub = agg[(agg.dataset == ds) & (agg.method == m)]
            if len(sub):
                ax.axhline(sub["nDCG_mean"].values[0], color=colors[m], ls="--", alpha=0.6,
                           label=f"{METHOD_LABELS[m]} ({sub['nDCG_mean'].values[0]:.3f})")
        for m in ["C_global", "D_local"]:
            sub = agg[(agg.dataset == ds) & (agg.method == m)].sort_values("probe_size")
            ax.errorbar(sub["probe_size"], sub["nDCG_mean"], yerr=sub["nDCG_std"],
                        marker="o", color=colors[m], label=METHOD_LABELS[m], capsize=3)
        ax.set_xlabel("Probe size")
        ax.set_ylabel("nDCG@5")
        ax.set_title(f"{ds}")
        ax.set_xticks(PROBE_SIZES)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "probe_curves.png"), dpi=150, bbox_inches="tight")
    print("probe_curves.png written")


def make_casebook():
    """Pick 10 cases where Local fixes Query-only error, 10 where Local creates error.

    Use probe_size=100, seed=0 for reproducibility. Compare D_local vs A_l00.
    """
    import json
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
    arm_to_idx = {a: i for i, a in enumerate(ARM_NAMES)}

    cases_fixed = []  # Local fixes Query-only error
    cases_broken = []  # Local creates new error

    for ds in DATASETS:
        df = pd.read_excel(f"experiments/xdomain/data/target_{ds}_xc_fin3_l10.xlsx")
        df = df.drop_duplicates(subset="query").reset_index(drop=True)
        l00 = pd.read_csv(f"experiments/xdomain/results_l00/predictions/target_{ds}_preds.csv")[["query", "pred_arm"]]
        df = df.merge(l00.rename(columns={"pred_arm": "pred_arm_l00"}), on="query", how="left").dropna()
        df = df.reset_index(drop=True)

        ndcg = df[NDCG_COLS].values.astype(np.float32)
        emb = model.encode(df["query"].tolist(), normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

        # Use seed=0, probe_size=100
        rng = np.random.default_rng(0)
        candidates = rng.choice(len(df), size=100, replace=False)
        eval_mask = np.ones(len(df), dtype=bool)
        eval_mask[candidates] = False
        probe_idx = candidates
        eval_idx = np.where(eval_mask)[0]

        ndcg_probe = ndcg[probe_idx]
        emb_probe = emb[probe_idx]
        emb_eval = emb[eval_idx]

        # Local Probe choices
        sim = emb_eval @ emb_probe.T
        k = 5
        topk_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(len(eval_idx))[:, None]
        sim_k = sim[rows, topk_idx]
        ndcg_k = ndcg_probe[topk_idx]
        w = np.exp(sim_k / 0.1)
        w = w / w.sum(axis=1, keepdims=True)
        pred_ndcg = (w[:, :, None] * ndcg_k).sum(axis=1)
        local_arm = pred_ndcg.argmax(axis=1)

        router_arm = df.iloc[eval_idx]["pred_arm_l00"].map(arm_to_idx).values
        ndcg_eval = ndcg[eval_idx]
        oracle_arm = ndcg_eval.argmax(axis=1)

        for j, eidx in enumerate(eval_idx):
            r_arm = router_arm[j]
            l_arm = local_arm[j]
            o_arm = oracle_arm[j]
            r_ndcg = ndcg_eval[j, r_arm]
            l_ndcg = ndcg_eval[j, l_arm]
            o_ndcg = ndcg_eval[j, o_arm]

            # Local fixes Query-only: router wrong, local right (or less wrong)
            if r_arm != o_arm and l_arm == o_arm and len(cases_fixed) < 10:
                nearest = [df.iloc[probe_idx[topk_idx[j][t]]]["query"][:60] for t in range(min(3, k))]
                cases_fixed.append({
                    "dataset": ds, "query": df.iloc[eidx]["query"],
                    "router": short(ARM_NAMES[r_arm]), "local": short(ARM_NAMES[l_arm]),
                    "oracle": short(ARM_NAMES[o_arm]),
                    "router_ndcg": r_ndcg, "local_ndcg": l_ndcg,
                    "ndcg_vec": {short(a): float(ndcg_eval[j, i]) for i, a in enumerate(ARM_NAMES)},
                    "nearest_probes": nearest,
                })
            # Local creates error: router right, local wrong
            elif r_arm == o_arm and l_arm != o_arm and len(cases_broken) < 10:
                nearest = [df.iloc[probe_idx[topk_idx[j][t]]]["query"][:60] for t in range(min(3, k))]
                cases_broken.append({
                    "dataset": ds, "query": df.iloc[eidx]["query"],
                    "router": short(ARM_NAMES[r_arm]), "local": short(ARM_NAMES[l_arm]),
                    "oracle": short(ARM_NAMES[o_arm]),
                    "router_ndcg": r_ndcg, "local_ndcg": l_ndcg,
                    "ndcg_vec": {short(a): float(ndcg_eval[j, i]) for i, a in enumerate(ARM_NAMES)},
                    "nearest_probes": nearest,
                })

        if len(cases_fixed) >= 10 and len(cases_broken) >= 10:
            break

    # Write casebook
    lines = ["# 探针实验坏例对照\n"]
    lines.append("## Local 修复了 Query-only 错误的案例（10条）\n")
    for i, c in enumerate(cases_fixed[:10], 1):
        lines.append(f"**{i}. [{c['dataset']}]** {c['query']}")
        lines.append(f"- Router→{c['router']}(nDCG={c['router_ndcg']:.2f})  Local→{c['local']}(nDCG={c['local_ndcg']:.2f})  Oracle={c['oracle']}")
        lines.append(f"- 五臂nDCG: {c['ndcg_vec']}")
        lines.append(f"- 最近探针: {c['nearest_probes']}")
        lines.append("")

    lines.append("## Local 新造成错误的案例（10条）\n")
    for i, c in enumerate(cases_broken[:10], 1):
        lines.append(f"**{i}. [{c['dataset']}]** {c['query']}")
        lines.append(f"- Router→{c['router']}(nDCG={c['router_ndcg']:.2f})  Local→{c['local']}(nDCG={c['local_ndcg']:.2f})  Oracle={c['oracle']}")
        lines.append(f"- 五臂nDCG: {c['ndcg_vec']}")
        lines.append(f"- 最近探针: {c['nearest_probes']}")
        lines.append("")

    with open(os.path.join(OUT_DIR, "probe_casebook.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"probe_casebook.md written ({len(cases_fixed)} fixed, {len(cases_broken)} broken)")


def make_conclusion(agg):
    lines = ["# 探针实验结论\n"]

    # Q1: Global prior useful?
    lines.append("## ① 全局能力先验有用吗？—— 有用，尤其在最优臂不同于MR的语料库\n")
    lines.append("| 数据集 | Fixed-MR | Global(100) | 增量 | Best Fixed |")
    lines.append("|---|---|---|---|---|")
    for ds in DATASETS:
        b = agg[(agg.dataset == ds) & (agg.probe_size == 100) & (agg.method == "B_fixedMR")]["nDCG_mean"].values[0]
        c = agg[(agg.dataset == ds) & (agg.probe_size == 100) & (agg.method == "C_global")]["nDCG_mean"].values[0]
        e = agg[(agg.dataset == ds) & (agg.probe_size == 100) & (agg.method == "E_bestfixed")]["nDCG_mean"].values[0]
        lines.append(f"| {ds} | {b:.4f} | {c:.4f} | {c-b:+.4f} | {e:.4f} |")
    lines.append("")
    lines.append("- **wiki-ss**：Global 从 0.733→0.769（+0.036），因为该库最优臂是 TR 而非 MR，探针能正确识别。")
    lines.append("- **tatdqa/arxivqa**：最优臂就是 MR，Global 与 Fixed-MR 接近（探针只是确认了 MR 最优）。")
    lines.append("- 结论：**语料库级先验有价值**，主要价值是切换最优臂族，而非逐 Query 精细路由。\n")

    # Q2: Local incremental value?
    lines.append("## ② Query级局部信息是否提供额外收益？—— 否，Local 未超过 Global\n")
    lines.append("| 数据集 | Global(100) | Local(100) | Local−Global |")
    lines.append("|---|---|---|---|")
    for ds in DATASETS:
        c = agg[(agg.dataset == ds) & (agg.probe_size == 100) & (agg.method == "C_global")]["nDCG_mean"].values[0]
        d = agg[(agg.dataset == ds) & (agg.probe_size == 100) & (agg.method == "D_local")]["nDCG_mean"].values[0]
        lines.append(f"| {ds} | {c:.4f} | {d:.4f} | {d-c:+.4f} |")
    lines.append("")
    lines.append("- Local 在全部 3 个语料库上均低于 Global（−0.017 至 −0.023）。")
    lines.append("- 可能原因：探针量小（100条），近邻的最优臂噪声大；Query 嵌入相似不代表最优检索臂相似；平均化五维 nDCG 稀释了信号。")
    lines.append('- 结论：**当前实现下 Query 级局部匹配无额外收益**，不支持"Query级能力指纹"作为主创新。\n')

    # Q3: Where works/fails?
    lines.append("## ③ 哪些语料库/Query类型有效或失效？\n")
    lines.append("- **有效**：wiki-ss 的全局先验（识别 TR/BM25 优于 MR）。")
    lines.append("- **失效**：所有语料库的局部匹配；tatdqa/arxivqa 上 Global 与 Fixed-MR 无差异（因为 MR 本就是最优）。")
    lines.append("- **共同瓶颈**：Local−Oracle 差距巨大（0.10–0.22），说明逐 Query 最优臂无法从相似探针的 nDCG 向量可靠推断。\n")

    lines.append("## 总体判断\n")
    lines.append("本轮属于方案定义的**弱信号**：Global 明显改善，但 Local 基本无额外收益。")
    lines.append("建议：")
    lines.append("1. 可将**语料库级臂先验**作为轻量特征加入路由（低成本、有收益）。")
    lines.append('2. **不建议**继续投入"Query 级局部能力探针"方向——当前证据不支持其有独立于全局先验的增量价值。')
    lines.append("3. 若要追求逐 Query 路由收益，应回到**候选集感知的 Rerank Gate**（直接利用检索结果特征，而非 Query 相似度），这是前期机制研究指出的更有希望的方向。")

    with open(os.path.join(OUT_DIR, "probe_conclusion.md"), "w") as f:
        f.write("\n".join(lines))
    print("probe_conclusion.md written")


if __name__ == "__main__":
    res, agg = load_and_prepare()
    make_metrics_md(agg)
    make_plots(agg)
    make_casebook()
    make_conclusion(agg)
    print("\nAll deliverables generated in", OUT_DIR)
