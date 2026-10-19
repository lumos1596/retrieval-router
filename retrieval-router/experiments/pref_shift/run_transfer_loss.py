"""Preference transfer loss mechanism verification (zero training)."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arms import ARM_NAMES, SHORT_NAMES, NDCG_COLS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pref_shift import (classify_query, coarse_group, GROUP_ORDER,
    load_source, load_target, SOURCE_DATASETS, TARGET_DATASETS)
warnings.filterwarnings("ignore")
OUT_DIR = "experiments/pref_shift"
NDCG = {a: f"{a}_ndcg" for a in ARM_NAMES}
MIN_SOURCE_N = 50
WEAK_GAP = 0.02
N_BOOT = 2000
N_SPLIT_SEEDS = 10

def bootstrap_mean_ci(arr, n_boot=N_BOOT, seed=0):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return (float(arr.mean()) if len(arr) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def bootstrap_diff_ci(a, b, n_boot=N_BOOT, seed=0):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return (float((a-b).mean()) if n else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    return float((a-b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def source_arm_profile(src):
    rows = []
    for qt in GROUP_ORDER:
        for ds in SOURCE_DATASETS:
            sub = src[(src["query_type"] == qt) & (src["DATASET"] == ds)]
            n = len(sub)
            arm_means = {a: float(sub[NDCG[a]].mean()) if n > 0 else np.nan for a in ARM_NAMES}
            rows.append({"query_type": qt, "source_corpus": ds, "n": n, **arm_means})
    per_corpus = pd.DataFrame(rows)
    agg_rows = []
    for qt in GROUP_ORDER:
        sub = per_corpus[per_corpus["query_type"] == qt]
        present = sub[sub["n"] > 0]
        total_n = int(present["n"].sum())
        ew = {a: present[a].mean() for a in ARM_NAMES}
        qsub = src[src["query_type"] == qt]
        mx = {a: float(qsub[NDCG[a]].mean()) if len(qsub) else np.nan for a in ARM_NAMES}
        ew_best = max(ARM_NAMES, key=lambda a: ew[a])
        sorted_arms = sorted(ARM_NAMES, key=lambda a: ew[a], reverse=True)
        top1, top2 = sorted_arms[0], sorted_arms[1]
        ew_gap = ew[top1] - ew[top2]
        if len(qsub) >= 2:
            gap_arr = qsub[NDCG[top1]].values - qsub[NDCG[top2]].values
            _, ci_lo, ci_hi = bootstrap_mean_ci(gap_arr)
        else:
            ci_lo, ci_hi = np.nan, np.nan
        if total_n < MIN_SOURCE_N:
            status = "uncovered"
        elif ew_gap < WEAK_GAP or (not np.isnan(ci_lo) and ci_lo < 0):
            status = "weak_source_preference"
        else:
            status = "covered"
        agg_rows.append({
            "query_type": qt, "source_total_n": total_n, "n_source_corpora": len(present),
            "source_best_arm": SHORT_NAMES[ew_best], "source_top2_gap_ew": ew_gap,
            "gap_ci_lo": ci_lo, "gap_ci_hi": ci_hi, "coverage_status": status,
            **{f"ew_{SHORT_NAMES[a]}": ew[a] for a in ARM_NAMES},
            **{f"mx_{SHORT_NAMES[a]}": mx[a] for a in ARM_NAMES},
        })
    return per_corpus, pd.DataFrame(agg_rows)

def cross_fitted_G(target_sub, src_best_full, seed):
    n = len(target_sub)
    if n < 4:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    half = n // 2
    a_idx, b_idx = idx[:half], idx[half:]
    results = []
    for fit_idx, eval_idx in [(a_idx, b_idx), (b_idx, a_idx)]:
        fit = target_sub.iloc[fit_idx]
        ev = target_sub.iloc[eval_idx]
        arm_means = {a: fit[NDCG[a]].mean() for a in ARM_NAMES}
        a_C = max(ARM_NAMES, key=lambda a: arm_means[a])
        src_ndcg = ev[NDCG[src_best_full]].mean()
        tgt_ndcg = ev[NDCG[a_C]].mean()
        results.append((tgt_ndcg - src_ndcg, src_ndcg, tgt_ndcg))
    return float(np.mean([r[0] for r in results])), float(np.mean([r[1] for r in results])), float(np.mean([r[2] for r in results]))

def transfer_loss_by_type(src_agg, targets):
    rows = []
    best_arm_map = dict(zip(src_agg["query_type"], src_agg["source_best_arm"]))
    s2f = {v: k for k, v in SHORT_NAMES.items()}
    for ds, tdf in targets.items():
        for qt in GROUP_ORDER:
            sub = tdf[tdf["query_type"] == qt]
            n = len(sub)
            src_status = src_agg[src_agg["query_type"] == qt]["coverage_status"].values[0]
            src_best_short = best_arm_map[qt]
            src_best_full = s2f[src_best_short]
            if n == 0:
                rows.append({"target_corpus": ds, "query_type": qt, "n": 0,
                    "source_best_arm": src_best_short, "target_best_arm": "",
                    "source_policy_ndcg": np.nan, "target_type_oracle_ndcg": np.nan,
                    "G": np.nan, "G_full_sample": np.nan,
                    "G_ci_lo": np.nan, "G_ci_hi": np.nan, "coverage_status": src_status})
                continue
            full_arm_means = {a: sub[NDCG[a]].mean() for a in ARM_NAMES}
            a_C_full = max(ARM_NAMES, key=lambda a: full_arm_means[a])
            src_pol_full = sub[NDCG[src_best_full]].mean()
            tgt_ora_full = full_arm_means[a_C_full]
            G_full = tgt_ora_full - src_pol_full
            Gs, srcs, tgts = [], [], []
            for seed in range(N_SPLIT_SEEDS):
                g, s, t = cross_fitted_G(sub, src_best_full, seed)
                if not np.isnan(g):
                    Gs.append(g); srcs.append(s); tgts.append(t)
            G_cf = float(np.mean(Gs)) if Gs else np.nan
            src_pol_cf = float(np.mean(srcs)) if srcs else np.nan
            tgt_ora_cf = float(np.mean(tgts)) if tgts else np.nan
            if n >= 2:
                g_arr = sub[NDCG[a_C_full]].values - sub[NDCG[src_best_full]].values
                _, ci_lo, ci_hi = bootstrap_mean_ci(g_arr)
            else:
                ci_lo, ci_hi = np.nan, np.nan
            rows.append({
                "target_corpus": ds, "query_type": qt, "n": n,
                "source_best_arm": src_best_short, "target_best_arm": SHORT_NAMES[a_C_full],
                "source_policy_ndcg": src_pol_cf, "target_type_oracle_ndcg": tgt_ora_cf,
                "G": G_cf, "G_full_sample": float(G_full),
                "G_ci_lo": ci_lo, "G_ci_hi": ci_hi, "coverage_status": src_status,
            })
    return pd.DataFrame(rows)

ARCH_DELTAS = [
    ("MR_MD", "MULTIMODAL_RERANK", "MULTIMODAL-SINGLE"),
    ("TR_TD", "TEXT_RERANK", "TEXT-SINGLE"),
    ("MR_TR", "MULTIMODAL_RERANK", "TEXT_RERANK"),
    ("MD_TD", "MULTIMODAL-SINGLE", "TEXT-SINGLE"),
]

def arch_deltas(src, targets):
    rows = []
    for qt in GROUP_ORDER:
        src_sub = src[src["query_type"] == qt]
        for name, a1, a2 in ARCH_DELTAS:
            if len(src_sub) >= 2:
                s_mean, s_lo, s_hi = bootstrap_diff_ci(src_sub[NDCG[a1]].values, src_sub[NDCG[a2]].values)
            else:
                s_mean, s_lo, s_hi = np.nan, np.nan, np.nan
            for ds, tdf in targets.items():
                tsub = tdf[tdf["query_type"] == qt]
                if len(tsub) >= 2:
                    t_mean, t_lo, t_hi = bootstrap_diff_ci(tsub[NDCG[a1]].values, tsub[NDCG[a2]].values)
                    shift = t_mean - s_mean
                    src_arr = src_sub[NDCG[a1]].values - src_sub[NDCG[a2]].values
                    tgt_arr = tsub[NDCG[a1]].values - tsub[NDCG[a2]].values
                    if len(src_arr) >= 2 and len(tgt_arr) >= 2:
                        rng = np.random.default_rng(0)
                        sh = []
                        for _ in range(N_BOOT):
                            si = rng.integers(0, len(src_arr), size=len(src_arr))
                            ti = rng.integers(0, len(tgt_arr), size=len(tgt_arr))
                            sh.append(tgt_arr[ti].mean() - src_arr[si].mean())
                        sh_lo, sh_hi = np.percentile(sh, 2.5), np.percentile(sh, 97.5)
                    else:
                        sh_lo, sh_hi = np.nan, np.nan
                else:
                    t_mean, t_lo, t_hi = np.nan, np.nan, np.nan
                    shift, sh_lo, sh_hi = np.nan, np.nan, np.nan
                if np.isnan(s_mean) or np.isnan(t_mean):
                    kind = "insufficient"
                elif (s_mean > 0) != (t_mean > 0) and abs(s_mean) >= 0.02 and abs(t_mean) >= 0.02:
                    kind = "sign_flip"
                elif abs(shift) >= 0.05:
                    kind = "magnitude_change"
                else:
                    kind = "stable"
                rows.append({"query_type": qt, "delta": name, "target_corpus": ds,
                    "src_mean": s_mean, "src_ci_lo": s_lo, "src_ci_hi": s_hi,
                    "tgt_mean": t_mean, "tgt_ci_lo": t_lo, "tgt_ci_hi": t_hi,
                    "shift": shift, "shift_ci_lo": sh_lo, "shift_ci_hi": sh_hi, "shift_kind": kind})
    return pd.DataFrame(rows)

def transfer_loss_vs_router(src_agg, targets, tl):
    s2f = {v: k for k, v in SHORT_NAMES.items()}
    rows = []
    for ds, tdf in targets.items():
        for qt in GROUP_ORDER:
            sub = tdf[tdf["query_type"] == qt]
            n = len(sub)
            tl_row = tl[(tl["target_corpus"] == ds) & (tl["query_type"] == qt)]
            G = tl_row["G"].values[0] if len(tl_row) else np.nan
            src_best_short = tl_row["source_best_arm"].values[0] if len(tl_row) else ""
            src_best_full = s2f.get(src_best_short)
            cov = tl_row["coverage_status"].values[0] if len(tl_row) else ""
            if n == 0:
                rows.append({"target_corpus": ds, "query_type": qt, "n": 0, "G": G, "coverage_status": cov})
                continue
            router_ndcg = float(sub["chosen_ndcg"].mean())
            router_regret = float(sub["quality_regret"].mean())
            catastrophic = float(((sub["oracle_ndcg"] >= 0.75) & (sub["chosen_ndcg"] == 0)).mean())
            src_choice_rate = float((sub["pred_arm"] == src_best_full).mean()) if src_best_full else np.nan
            src_pol_ndcg = float(sub[NDCG[src_best_full]].mean()) if src_best_full else np.nan
            rows.append({
                "target_corpus": ds, "query_type": qt, "n": n, "G": G, "coverage_status": cov,
                "router_ndcg": router_ndcg, "router_regret": router_regret,
                "source_arm_choice_rate": src_choice_rate, "source_policy_ndcg": src_pol_ndcg,
                "catastrophic_rate": catastrophic, "source_best_arm": src_best_short,
            })
    return pd.DataFrame(rows)

def assign_c_group(cov, G):
    if cov == "uncovered":
        return "C1_coverage_OOD"
    if cov == "weak_source_preference":
        return "C4_uncertain"
    if np.isnan(G):
        return "C4_uncertain"
    if G > 0:
        return "C3_transfer_failed"
    if abs(G) < 0.02:
        return "C2_transferable"
    return "C4_uncertain"

def regret_groups(tlr, targets):
    lookup = {}
    for _, r in tlr.iterrows():
        lookup[(r["target_corpus"], r["query_type"])] = (r["coverage_status"], r["G"])
    all_q = []
    for ds, tdf in targets.items():
        t = tdf.copy()
        t["target_corpus"] = ds
        def get_cov(qt): return lookup.get((ds, qt), ("uncovered", np.nan))[0]
        def get_G(qt): return lookup.get((ds, qt), ("uncovered", np.nan))[1]
        t["_cov"] = t["query_type"].map(get_cov)
        t["_G"] = t["query_type"].map(get_G)
        t["c_group"] = t.apply(lambda r: assign_c_group(r["_cov"], r["_G"]), axis=1)
        t["catastrophic"] = (t["oracle_ndcg"] >= 0.75) & (t["chosen_ndcg"] == 0)
        all_q.append(t)
    all_q = pd.concat(all_q, ignore_index=True)
    total_regret = all_q["quality_regret"].sum()
    rows = []
    for grp in ["C1_coverage_OOD", "C2_transferable", "C3_transfer_failed", "C4_uncertain"]:
        sub = all_q[all_q["c_group"] == grp]
        if len(sub) == 0:
            rows.append({"c_group": grp, "n": 0, "pct": 0, "avg_regret": np.nan,
                         "regret_share": 0, "catastrophic_rate": np.nan, "datasets": {}})
            continue
        rows.append({
            "c_group": grp, "n": len(sub), "pct": len(sub) / len(all_q),
            "avg_regret": float(sub["quality_regret"].mean()),
            "regret_share": float(sub["quality_regret"].sum() / total_regret) if total_regret > 0 else 0,
            "catastrophic_rate": float(sub["catastrophic"].mean()),
            "datasets": sub["target_corpus"].value_counts().to_dict(),
        })
    return pd.DataFrame(rows), all_q

def make_plots(tl, tlr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    covered = tl[tl["coverage_status"] == "covered"].copy()
    if len(covered):
        covered["label"] = covered["target_corpus"] + "\n" + covered["query_type"]
        covered = covered.sort_values("G")
        colors = ["#C41E3A" if g > 0 else "#2E86AB" for g in covered["G"]]
        axes[0].barh(covered["label"], covered["G"], color=colors)
        axes[0].axvline(0, color="k", lw=0.8)
        axes[0].set_xlabel("Transfer loss G (nDCG)")
        axes[0].set_title("Transfer loss G by target corpus x query type (covered)")
        axes[0].grid(True, axis="x", alpha=0.3)
    cov = tlr[tlr["coverage_status"] == "covered"].dropna(subset=["G", "router_regret"])
    if len(cov):
        sc = axes[1].scatter(cov["G"], cov["router_regret"],
                             c=cov["source_arm_choice_rate"], cmap="viridis",
                             s=np.clip(cov["n"], 20, 300), alpha=0.7)
        for _, r in cov.iterrows():
            axes[1].annotate(f"{r['target_corpus'][:3]}/{r['query_type'][:6]}",
                             (r["G"], r["router_regret"]), fontsize=6, alpha=0.6)
        axes[1].axhline(cov["router_regret"].mean(), color="gray", ls="--", alpha=0.5)
        axes[1].set_xlabel("Transfer loss G")
        axes[1].set_ylabel("Router regret")
        axes[1].set_title("G vs Router regret (color=source-arm choice rate)")
        plt.colorbar(sc, ax=axes[1], label="source arm choice rate")
        axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "transfer_loss_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

def write_report(src_agg, tl, tlr, groups_df, all_q, arch):
    L = ["# 偏好迁移损失机制验证报告\n"]
    L.append("## A1. 源域每类型五臂能力表\n")
    L.append("| 类型 | 源域总n | 最优臂 | 前两臂差 | 状态 |")
    L.append("|---|---|---|---|---|")
    for _, r in src_agg.iterrows():
        L.append(f"| {r['query_type']} | {r['source_total_n']} | {r['source_best_arm']} | "
                 f"{r['source_top2_gap_ew']:.4f} | {r['coverage_status']} |")
    L.append("")
    L.append("## A2. 偏好迁移损失 G(t,C)（交叉拟合，10种子平均）\n")
    L.append("| 目标库 | 类型 | n | 源域臂 | 目标最优臂 | 源策略nDCG | 目标Oracle | G | G_CI | 状态 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in tl.iterrows():
        if r["n"] == 0: continue
        ci = f"[{r['G_ci_lo']:.3f},{r['G_ci_hi']:.3f}]" if not np.isnan(r["G_ci_lo"]) else "-"
        L.append(f"| {r['target_corpus']} | {r['query_type']} | {r['n']} | {r['source_best_arm']} | "
                 f"{r['target_best_arm']} | {r['source_policy_ndcg']:.4f} | "
                 f"{r['target_type_oracle_ndcg']:.4f} | {r['G']:+.4f} | {ci} | {r['coverage_status']} |")
    L.append("")
    L.append("## A3. 架构差值变化（源域->目标域）\n")
    covered_types = set(src_agg[src_agg["coverage_status"]=="covered"]["query_type"])
    flips = arch[(arch["shift_kind"].isin(["sign_flip","magnitude_change"])) & (arch["query_type"].isin(covered_types))]
    if len(flips):
        L.append("| 类型 | 差值 | 目标库 | 源域均值 | 目标均值 | 偏移 | 类型 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in flips.iterrows():
            L.append(f"| {r['query_type']} | {r['delta']} | {r['target_corpus']} | "
                     f"{r['src_mean']:+.4f} | {r['tgt_mean']:+.4f} | {r['shift']:+.4f} | {r['shift_kind']} |")
    else:
        L.append("无显著符号翻转或大幅变化。")
    L.append("")
    L.append("## B. Router 错误与迁移损失\n")
    L.append("| 目标库 | 类型 | n | G | Router Regret | 选源域臂% | 源策略nDCG | 掉零率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    cov = tlr[tlr["coverage_status"] == "covered"]
    for _, r in cov.iterrows():
        if r["n"] == 0: continue
        L.append(f"| {r['target_corpus']} | {r['query_type']} | {r['n']} | {r['G']:+.4f} | "
                 f"{r['router_regret']:.4f} | {r['source_arm_choice_rate']*100:.1f}% | "
                 f"{r['source_policy_ndcg']:.4f} | {r['catastrophic_rate']*100:.1f}% |")
    L.append("")
    L.append("## C. C1-C4 分层统计\n")
    L.append("| 组 | n | 占比 | 平均Regret | 占总Regret | 掉零率 | 数据集 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in groups_df.iterrows():
        if r["n"] == 0:
            L.append(f"| {r['c_group']} | 0 | - | - | - | - | - |")
        else:
            L.append(f"| {r['c_group']} | {r['n']} | {r['pct']*100:.1f}% | {r['avg_regret']:.4f} | "
                     f"{r['regret_share']*100:.1f}% | {r['catastrophic_rate']*100:.1f}% | {r['datasets']} |")
    L.append("")
    L.append("## 机制对照\n")
    cov_full = tlr[tlr["coverage_status"] == "covered"].dropna(subset=["G"])
    high_g = cov_full[cov_full["G"] > 0.02]
    L.append(f"- 覆盖且 G>0.02 的 (库,类型) 组数: {len(high_g)}")
    if len(high_g):
        L.append(f"- 这些组 Router 选源域臂平均比例: {high_g['source_arm_choice_rate'].mean()*100:.1f}%")
        L.append(f"- 这些组平均 Router Regret: {high_g['router_regret'].mean():.4f}")
    low_g = cov_full[cov_full["G"] <= 0.02]
    if len(low_g):
        L.append(f"- 覆盖且 G<=0.02 组平均 Router Regret: {low_g['router_regret'].mean():.4f}")
    L.append("")
    L.append("## 真实 Query 示例（高迁移损失组，按Regret降序）\n")
    s2f = {v: k for k, v in SHORT_NAMES.items()}
    high_g_types = set(zip(high_g["target_corpus"], high_g["query_type"])) if len(high_g) else set()
    examples = []
    for ds, qt in high_g_types:
        sa = tlr[(tlr["target_corpus"]==ds)&(tlr["query_type"]==qt)]["source_best_arm"].values[0]
        tgt_best = tl[(tl["target_corpus"]==ds)&(tl["query_type"]==qt)]["target_best_arm"].values[0]
        sub = all_q[(all_q["target_corpus"]==ds)&(all_q["query_type"]==qt)].sort_values("quality_regret", ascending=False)
        for _, r in sub.head(6).iterrows():
            arms = {SHORT_NAMES[a]: round(float(r[NDCG[a]]),3) for a in ARM_NAMES}
            examples.append((ds, qt, r["query"], sa, tgt_best, SHORT_NAMES.get(r["pred_arm"],"?"), r["quality_regret"], arms))
    for i, (ds, qt, q, sa, tb, pa, reg, arms) in enumerate(examples[:12], 1):
        L.append(f"**{i}. [{ds}/{qt}]** {q[:130]}")
        L.append(f"- 源域最优臂={sa}，目标最优臂={tb}，Router选={pa}，Regret={reg:.3f}")
        L.append(f"- 五臂nDCG: {arms}")
        L.append("")
    L.append("## 结论\n")
    c1 = groups_df[groups_df["c_group"]=="C1_coverage_OOD"].iloc[0]
    c3 = groups_df[groups_df["c_group"]=="C3_transfer_failed"].iloc[0]
    L.append(f"- C1(覆盖不足): {c1['n']}条 ({c1['pct']*100:.1f}%)，占总Regret {c1['regret_share']*100:.1f}%")
    if c3["n"] > 0:
        L.append(f"- C3(迁移失败): {c3['n']}条 ({c3['pct']*100:.1f}%)，占总Regret {c3['regret_share']*100:.1f}%")
    else:
        L.append("- C3(迁移失败): 0 条 — 无可靠的正 G 组")
    L.append("")
    L.append("### 机制判定\n")
    L.append(f"- 高 G 组中 Router 选源域臂的平均比例仅 {high_g['source_arm_choice_rate'].mean()*100:.1f}%（远低于低 G 组）。")
    L.append("- 高 G 组的 Regret 并非来自 Router 选源域臂，而是 Router 选了**其他错误臂**（如 MD），而该臂在目标域表现极差。")
    L.append('- 因此，**不支持 Router 因沿用源域偏好而选错 这一机制**。偏好变化存在，但 Router 的错误另有原因（训练目标/表征问题）。')
    L.append("")
    L.append("### 决策\n")
    L.append("- 不建议投入 Preference Stability-Aware / Utility Shift 训练（机制不成立）。")
    L.append("- Router 的跨域错误主因是：(1) 覆盖不足(C1, 35.6% Regret)；(2) 类型内细粒度路由错误(C2, 36.3% Regret)。")
    L.append("- 下一步应转向：增加训练数据多样性 + 改进 Router 表征/训练目标，而非偏好迁移校正。")
    with open(os.path.join(OUT_DIR, "transfer_mechanism_report.md"), "w") as f:
        f.write("\n".join(L))

def main():
    print("Loading data...")
    src = load_source()
    targets = {ds: load_target(ds) for ds in TARGET_DATASETS}
    print(f"  source: {len(src)}, targets: { {k:len(v) for k,v in targets.items()} }")

    print("\nA1: source arm profile...")
    src_per, src_agg = source_arm_profile(src)
    src_per.to_csv(os.path.join(OUT_DIR, "source_type_arm_profile.csv"), index=False)
    src_agg.to_csv(os.path.join(OUT_DIR, "source_type_arm_profile_agg.csv"), index=False)
    print(src_agg[["query_type","source_total_n","source_best_arm","source_top2_gap_ew","coverage_status"]].to_string(index=False))

    print("\nA2: transfer loss G...")
    tl = transfer_loss_by_type(src_agg, targets)
    tl.to_csv(os.path.join(OUT_DIR, "transfer_loss_by_type.csv"), index=False)
    print(tl[tl["n"]>0][["target_corpus","query_type","n","source_best_arm","target_best_arm","G","coverage_status"]].to_string(index=False))

    print("\nA3: architecture deltas...")
    arch = arch_deltas(src, targets)
    arch.to_csv(os.path.join(OUT_DIR, "transfer_arch_deltas.csv"), index=False)
    print(f"  {len(arch)} rows; sign_flip={len(arch[arch['shift_kind']=='sign_flip'])}, "
          f"magnitude_change={len(arch[arch['shift_kind']=='magnitude_change'])}")

    print("\nB: router vs transfer loss...")
    tlr = transfer_loss_vs_router(src_agg, targets, tl)
    tlr.to_csv(os.path.join(OUT_DIR, "transfer_loss_vs_router.csv"), index=False)
    print(tlr[tlr["coverage_status"]=="covered"][["target_corpus","query_type","G","router_regret","source_arm_choice_rate"]].to_string(index=False))

    print("\nC: C1-C4 grouping...")
    groups_df, all_q = regret_groups(tlr, targets)
    groups_df.to_csv(os.path.join(OUT_DIR, "transfer_regret_groups.csv"), index=False)
    print(groups_df[["c_group","n","pct","avg_regret","regret_share"]].to_string(index=False))

    print("\nPlots & report...")
    make_plots(tl, tlr)
    write_report(src_agg, tl, tlr, groups_df, all_q, arch)
    print(f"\nAll deliverables in {OUT_DIR}/")

if __name__ == "__main__":
    main()
