"""Axes A-I of the bad-case plan. Consumes analysis/query_metrics.csv and writes:

  analysis/slice_summary.csv  -- per-slice n / mean+cumulative losses / match rates
  analysis/confusion.csv      -- picked arm x five-way best arm (both reward and nDCG best)
  analysis/top_badcases.csv   -- highest-loss queries with all raw evidence columns

Headline statistics for every axis are printed to stdout (used for failure_report.md).
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from arms import ARM_NAMES, SHORT_NAMES  # noqa: E402

ANALYSIS_DIR = os.path.join(REPO_ROOT, "analysis")
ARMS5 = [SHORT_NAMES[a] for a in ARM_NAMES]  # MR MD TR TD BM

SEVERE_Q = 0.90          # top decile of reward_regret = "severe"
HIGH_CONF = 0.50         # softmax max-prob threshold (max observed ~0.80)
TIE_MARGIN = 0.01        # reward margin under which arms are effectively tied
RERANK_DELTA = 0.02      # nDCG gap treated as a real rerank harm/gain


def agg_slice(df, axis, slice_name):
    return {
        "axis": axis,
        "slice": str(slice_name),
        "n": len(df),
        "pct": round(100 * len(df) / 8506, 2),
        "mean_quality_regret": round(df["quality_regret"].mean(), 5),
        "sum_quality_regret": round(df["quality_regret"].sum(), 3),
        "mean_reward_regret": round(df["reward_regret"].mean(), 5),
        "sum_reward_regret": round(df["reward_regret"].sum(), 3),
        "arm_match_reward_rate": round(df["arm_match_reward"].mean(), 4),
        "mean_confidence": round(df["confidence"].mean(), 4),
        "hc_severe_n": int((df["hc_severe"]).sum()),
    }


def main():
    m = pd.read_csv(os.path.join(ANALYSIS_DIR, "query_metrics.csv"))
    severe_thr = m["reward_regret"].quantile(SEVERE_Q)
    m["severe"] = m["reward_regret"] >= severe_thr
    m["hc_severe"] = m["severe"] & (m["confidence"] >= HIGH_CONF)
    m["near_tie"] = m["margin_reward"] < TIE_MARGIN

    rows = []

    # ---------------- A. overall error structure -----------------------------------------
    print("=" * 78)
    print("A. OVERALL ERROR STRUCTURE")
    print("=" * 78)
    qs = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    print("reward_regret quantiles:", dict(zip(qs, np.round(m["reward_regret"].quantile(qs), 4))))
    print("quality_regret quantiles:", dict(zip(qs, np.round(m["quality_regret"].quantile(qs), 4))))
    tot_rr = m["reward_regret"].sum()
    tot_qr = m["quality_regret"].sum()
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = int(np.ceil(len(m) * frac))
        share_r = m["reward_regret"].nlargest(k).sum() / tot_rr
        share_q = m["quality_regret"].nlargest(k).sum() / tot_qr
        print(f"top {int(frac*100):>2}% queries contribute "
              f"{share_r*100:5.1f}% of reward_regret, {share_q*100:5.1f}% of quality_regret")
    print(f"zero reward_regret: {(m['reward_regret']==0).sum()}  "
          f"zero quality_regret: {(m['quality_regret']==0).sum()}")
    print(f"severe threshold (P90 reward_regret) = {severe_thr:.4f}; "
          f"severe n={int(m['severe'].sum())}, high-conf severe n={int(m['hc_severe'].sum())}")
    print(f"near-tie (margin<{TIE_MARGIN}) n={int(m['near_tie'].sum())}, "
          f"their total reward_regret={m.loc[m['near_tie'],'reward_regret'].sum():.1f} "
          f"({m.loc[m['near_tie'],'reward_regret'].sum()/tot_rr*100:.1f}% of all)")
    rows.append(agg_slice(m, "A_overall", "all"))
    for lab, sub in [("severe_top10pct", m[m["severe"]]),
                     ("high_conf_severe", m[m["hc_severe"]]),
                     ("near_tie", m[m["near_tie"]]),
                     ("exact_reward_match", m[m["arm_match_reward"]])]:
        rows.append(agg_slice(m, "A_group", lab))
        rows[-1]["n"] = len(sub)

    # regret deciles
    m["rr_decile"] = pd.qcut(m["reward_regret"], 10, labels=False, duplicates="drop")
    for d, sub in m.groupby("rr_decile"):
        rows.append(agg_slice(sub, "A_reward_regret_decile", int(d)))

    # ---------------- B. confusion: picked x best ----------------------------------------
    print("\n" + "=" * 78)
    print("B. WHERE DO MISROUTES GO (picked x best-reward arm)")
    print("=" * 78)
    conf_rows = []
    for kind, bestcol in [("best_reward", "best_reward_arm"), ("best_ndcg", "best_ndcg_arm")]:
        g = m.groupby(["pred_arm", bestcol])
        for (p, b), sub in g:
            conf_rows.append({
                "target": kind, "pred_arm": p, "best_arm": b, "n": len(sub),
                "mean_quality_regret": round(sub["quality_regret"].mean(), 5),
                "sum_quality_regret": round(sub["quality_regret"].sum(), 3),
                "mean_reward_regret": round(sub["reward_regret"].mean(), 5),
                "sum_reward_regret": round(sub["reward_regret"].sum(), 3),
            })
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(os.path.join(ANALYSIS_DIR, "confusion.csv"), index=False)

    off = conf[(conf["target"] == "best_reward") & (conf["pred_arm"] != conf["best_arm"])]
    print("Top loss-contributing off-diagonal cells (vs reward-best):")
    print(off.sort_values("sum_reward_regret", ascending=False)
          .head(10)[["pred_arm", "best_arm", "n", "mean_reward_regret", "sum_reward_regret",
                     "mean_quality_regret", "sum_quality_regret"]].to_string(index=False))
    print("\nPredicted arm marginal losses:")
    mg = m.groupby("pred_arm").agg(n=("query", "size"),
                                   mean_rr=("reward_regret", "mean"),
                                   sum_rr=("reward_regret", "sum"),
                                   mean_qr=("quality_regret", "mean"),
                                   sum_qr=("quality_regret", "sum")).round(4)
    mg["pct_of_total_rr"] = (100 * mg["sum_rr"] / tot_rr).round(1)
    print(mg)

    # ---------------- C. dataset & query features ----------------------------------------
    print("\n" + "=" * 78)
    print("C. DATASET AND QUERY FEATURES")
    print("=" * 78)
    for ds, sub in m.groupby("DATASET"):
        rows.append(agg_slice(sub, "C_DATASET", ds))
    ds_tbl = m.groupby("DATASET").apply(
        lambda s: pd.Series({
            "n": len(s),
            "mean_rr": s["reward_regret"].mean().round(4),
            "sum_rr": s["reward_regret"].sum().round(1),
            "pct_rr": round(100 * s["reward_regret"].sum() / tot_rr, 1),
            "severe_rate": round(s["severe"].mean(), 3),
            "hc_severe": int(s["hc_severe"].sum()),
        }), include_groups=False)
    print(ds_tbl)

    m["wlen_bin"] = pd.cut(m["qlen_words"], [0, 5, 10, 15, 25, 10000],
                           labels=["1-5", "6-10", "11-15", "16-25", "26+"])
    for b, sub in m.groupby("wlen_bin", observed=True):
        rows.append(agg_slice(sub, "C_qlen_words", str(b)))
    for feat in ["has_number", "has_year", "has_figure_ref", "has_comparison",
                 "has_quoted", "n_clauses"]:
        if feat == "n_clauses":
            bins = pd.cut(m[feat], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
            for b, sub in m.groupby(bins, observed=True):
                rows.append(agg_slice(sub, "C_n_clauses", str(b)))
        else:
            for v in (True, False):
                rows.append(agg_slice(m[m[feat] == v], f"C_{feat}", v))

    # ---------------- D. modality ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("D. MODALITY (text arms {BM,TD,TR} vs visual arms {MD,MR})")
    print("=" * 78)
    for (t, p), sub in m.groupby(["modality_true", "modality_pred"]):
        rows.append(agg_slice(sub, "D_modality_true_x_pred", f"true={t},pred={p}"))
    mod = m.groupby(["modality_true", "modality_pred"]).agg(
        n=("query", "size"), mean_qr=("quality_regret", "mean"),
        sum_qr=("quality_regret", "sum")).round(4)
    print(mod)
    err = m[m["modality_error"]]
    print(f"modality errors: n={len(err)} ({len(err)/len(m)*100:.1f}%), "
          f"sum quality_regret={err['quality_regret'].sum():.1f} "
          f"({err['quality_regret'].sum()/tot_qr*100:.1f}% of all)")
    v_need_text = m[(m["modality_true"] == "visual") & (m["modality_pred"] == "text")]
    t_need_vis = m[(m["modality_true"] == "text") & (m["modality_pred"] == "visual")]
    print(f"  visual needed, text picked: n={len(v_need_text)}, "
          f"mean qr={v_need_text['quality_regret'].mean():.4f}, sum={v_need_text['quality_regret'].sum():.1f}")
    print(f"  text needed, visual picked: n={len(t_need_vis)}, "
          f"mean qr={t_need_vis['quality_regret'].mean():.4f}, sum={t_need_vis['quality_regret'].sum():.1f}")

    # ---------------- E. rerank architecture ----------------------------------------------
    print("\n" + "=" * 78)
    print("E. RERANK DELTAS  (TR-TD, MR-MD on nDCG@5)")
    print("=" * 78)
    for col, axis in [("rerank_gain_text", "E_rerank_TEXT"), ("rerank_gain_mm", "E_rerank_MM")]:
        buckets = pd.cut(m[col], [-10, -RERANK_DELTA, RERANK_DELTA, 10],
                         labels=["harm(<-0.02)", "neutral", "help(>0.02)"])
        for b, sub in m.groupby(buckets, observed=True):
            rows.append(agg_slice(sub, axis, str(b)))
        print(f"{col}: harm n={(m[col] < -RERANK_DELTA).sum()}, "
              f"neutral n={((m[col].abs() <= RERANK_DELTA)).sum()}, "
              f"help n={(m[col] > RERANK_DELTA).sum()}")
    harm = m[m["rerank_harm_chosen"]]
    print(f"rerank CHOSEN and harmful (>0.02 below its dense variant): n={len(harm)}, "
          f"sum quality_regret={harm['quality_regret'].sum():.1f}, "
          f"mean={harm['quality_regret'].mean():.4f}")
    rows.append(agg_slice(harm, "E_rerank_chosen_harmful", "all"))
    rows[-1]["n"] = len(harm)
    # missed rerank gain: dense chosen where its rerank variant is >0.02 better
    miss_text = (m["pred_arm"] == "TD") & (m["rerank_gain_text"] > RERANK_DELTA)
    miss_mm = (m["pred_arm"] == "MD") & (m["rerank_gain_mm"] > RERANK_DELTA)
    miss = m[miss_text | miss_mm]
    print(f"dense chosen but rerank would help >0.02: n={len(miss)}, "
          f"sum quality_regret={miss['quality_regret'].sum():.1f}")
    rows.append(agg_slice(miss, "E_rerank_missed_gain", "all"))
    rows[-1]["n"] = len(miss)

    # ---------------- F. confidence / discernibility --------------------------------------
    print("\n" + "=" * 78)
    print("F. CONFIDENCE vs REGRET")
    print("=" * 78)
    m["conf_decile"] = pd.cut(m["confidence"].rank(method="first"), 10, labels=False)
    for d, sub in m.groupby("conf_decile"):
        rows.append(agg_slice(sub, "F_confidence_decile", int(d)))
    rate = m.groupby("conf_decile").agg(conf_lo=("confidence", "min"), conf_hi=("confidence", "max"),
                                        severe_rate=("severe", "mean"),
                                        mean_rr=("reward_regret", "mean"),
                                        mean_qr=("quality_regret", "mean")).round(4)
    print(rate)
    print(f"corr(confidence, reward_regret) = {m['confidence'].corr(m['reward_regret']):.4f}")
    print(f"corr(confidence, quality_regret) = {m['confidence'].corr(m['quality_regret']):.4f}")
    lc = m[m["confidence"] < 0.30]
    hc = m[m["confidence"] >= HIGH_CONF]
    print(f"low-conf(<0.30): n={len(lc)}, severe rate {lc['severe'].mean():.3f}, "
          f"mean rr {lc['reward_regret'].mean():.4f}")
    print(f"high-conf(>={HIGH_CONF}): n={len(hc)}, severe rate {hc['severe'].mean():.3f}, "
          f"mean rr {hc['reward_regret'].mean():.4f}")
    for lab, lo, hi in [("margin_tied", 0, TIE_MARGIN), ("margin_small", TIE_MARGIN, 0.05),
                        ("margin_medium", 0.05, 0.15), ("margin_large", 0.15, 10)]:
        sub = m[(m["margin_reward"] >= lo) & (m["margin_reward"] < hi)]
        rows.append(agg_slice(sub, "F_reward_margin", lab))

    # ---------------- G. lambda stability --------------------------------------------------
    print("\n" + "=" * 78)
    print("G. DECISION STABILITY ACROSS lambda = 0.0 / 0.1 / 0.7")
    print("=" * 78)
    for k, sub in m.groupby("n_distinct_arms_lambda"):
        rows.append(agg_slice(sub, "G_n_distinct_arms", str(k)))
    print(m["n_distinct_arms_lambda"].value_counts().sort_index())
    print(f"l00 vs l10 switches: {int(m['switches_vs_l00'].sum())}; "
          f"l70 vs l10 switches: {int(m['switches_vs_l70'].sum())}")
    unstable = m[m["n_distinct_arms_lambda"] >= 2]
    print(f"queries changing arm across lambdas: n={len(unstable)} "
          f"({len(unstable)/len(m)*100:.1f}%), mean reward_regret(l10) "
          f"{unstable['reward_regret'].mean():.4f} vs stable {m.loc[m['n_distinct_arms_lambda']==1,'reward_regret'].mean():.4f}")

    # ---------------- H. unroutable failures ----------------------------------------------
    print("\n" + "=" * 78)
    print("H. CEILING: 5-ARM ORACLE nDCG DISTRIBUTION")
    print("=" * 78)
    bins = [-0.01, 0.001, 0.25, 0.5, 0.75, 1.01]
    labels = ["0", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]
    m["ceil_bin"] = pd.cut(m["oracle5_ndcg"], bins, labels=labels)
    for b, sub in m.groupby("ceil_bin", observed=True):
        rows.append(agg_slice(sub, "H_oracle5_ndcg_bin", str(b)))
    ceil = m.groupby("ceil_bin", observed=True).agg(
        n=("query", "size"), mean_oracle=("oracle5_ndcg", "mean"),
        mean_pred=("pred_ndcg", "mean"), mean_qr=("quality_regret", "mean"),
        sum_qr=("quality_regret", "sum")).round(4)
    print(ceil)
    allfail = m[m["oracle5_ndcg"] < 0.25]
    highceil_miss = m[(m["oracle5_ndcg"] >= 0.75) & (m["quality_regret"] >= 0.25)]
    print(f"ALL-FAIL (5-arm oracle <0.25): n={len(allfail)} "
          f"({len(allfail)/len(m)*100:.1f}%), total qr={allfail['quality_regret'].sum():.1f} "
          f"({allfail['quality_regret'].sum()/tot_qr*100:.1f}% of all regret)")
    print(f"HIGH-CEILING BIG MISS (oracle>=0.75 & qr>=0.25): n={len(highceil_miss)}, "
          f"total qr={highceil_miss['quality_regret'].sum():.1f} "
          f"({highceil_miss['quality_regret'].sum()/tot_qr*100:.1f}% of all regret)")
    rows.append(agg_slice(allfail, "H_allfail_oracle_lt_025", "all"))
    rows[-1]["n"] = len(allfail)
    rows.append(agg_slice(highceil_miss, "H_highceiling_bigmiss", "all"))
    rows[-1]["n"] = len(highceil_miss)

    # ---------------- I. supervision noise ------------------------------------------------
    print("\n" + "=" * 78)
    print("I. LABEL / SUPERVISION NOISE")
    print("=" * 78)
    print(f"near-tied optima (reward margin<{TIE_MARGIN}): n={int(m['near_tie'].sum())} "
          f"({m['near_tie'].mean()*100:.1f}%)")
    ndcg_tie = m[m["margin_ndcg"] < TIE_MARGIN]
    print(f"near-tied nDCG optima (margin<{TIE_MARGIN}): n={len(ndcg_tie)} ({len(ndcg_tie)/len(m)*100:.1f}%)")
    # A near tie among the TOP arms is only harmless if the picked arm is one of them.
    # near-tie + LARGE regret => router did not even reach the tied pair: genuinely severe.
    pseudo = m[(~m["arm_match_reward"]) & (m["reward_regret"] < TIE_MARGIN)]
    tie_wild = m[m["near_tie"] & m["severe"]]
    print(f"harmless disagreements (regret<{TIE_MARGIN}): n={len(pseudo)} "
          f"({len(pseudo)/(~m['arm_match_reward']).sum()*100:.1f}% of arm disagreements)")
    print(f"NEAR-TIE TOP BUT ROUTER WILDLY WRONG (margin<{TIE_MARGIN} & severe): n={len(tie_wild)}, "
          f"sum reward_regret={tie_wild['reward_regret'].sum():.1f}")
    rows.append(agg_slice(tie_wild, "I_neartie_but_severe", "all"))
    rows[-1]["n"] = len(tie_wild)
    full = pd.read_excel(os.path.join(REPO_ROOT, "train", "data", "updated_dataset.xlsx"))
    N = full[[f"{a}_ndcg" for a in ARM_NAMES]].to_numpy(float)
    full["best_arm_ndcg"] = [ARMS5[i] for i in N.argmax(1)]
    dup = full[full.duplicated("query", keep=False)].sort_values("query")
    print(f"full dataset: {len(full)} rows, {full['query'].nunique()} unique queries, "
          f"{len(dup)} rows sharing an exact-duplicate query string")
    if len(dup):
        n_conf = sum(g["best_arm_ndcg"].nunique() > 1 for _, g in dup.groupby("query"))
        n_ds = sum(g["DATASET"].nunique() > 1 for _, g in dup.groupby("query"))
        print(f"  duplicate-query groups with conflicting nDCG-best arm: {n_conf}")
        print(f"  duplicate-query groups spanning >1 DATASET: {n_ds}")
    rows.append(agg_slice(pseudo, "I_harmless_disagreement_regret_lt_001", "all"))
    rows[-1]["n"] = len(pseudo)

    # ---------------- supplementary headline stats ---------------------------------------
    print("\n" + "=" * 78)
    print("SUPPLEMENT: CATASTROPHIC MISSES / KEYWORD SHORTCUT / BM25 OPPORTUNITY")
    print("=" * 78)
    cat = m[(m["oracle5_ndcg"] >= 0.75) & (m["pred_ndcg"] == 0)]
    print(f"catastrophic (oracle>=0.75 but picked ndcg=0): n={len(cat)} "
          f"({len(cat)/len(m)*100:.1f}%), sum qr={cat['quality_regret'].sum():.1f} "
          f"({cat['quality_regret'].sum()/tot_qr*100:.1f}% of all regret)")
    print("  catastrophic by picked arm:", dict(cat["pred_arm"].value_counts()))
    print("  catastrophic by DATASET:", dict(cat["DATASET"].value_counts()))
    rows.append(agg_slice(cat, "X_catastrophic_oracle_hi_picked0", "all"))
    rows[-1]["n"] = len(cat)

    fig = m[m["has_figure_ref"]]
    fig_vis = fig["modality_pred"].eq("visual").mean()
    all_vis = m["modality_pred"].eq("visual").mean()
    print(f"figure/chart/table-word queries: n={len(fig)} ({len(fig)/len(m)*100:.1f}%); "
          f"routed to visual arms {fig_vis*100:.1f}% vs baseline {all_vis*100:.1f}%")
    print(f"  among them truly visual-needed: {(fig['modality_true']=='visual').mean()*100:.1f}%, "
          f"text-needed: {(fig['modality_true']=='text').mean()*100:.1f}%, "
          f"tie: {(fig['modality_true']=='tie').mean()*100:.1f}%")
    fig_err = fig[fig["modality_error"]]
    print(f"  figure-word modality errors: n={len(fig_err)}, sum qr={fig_err['quality_regret'].sum():.1f}")
    rows.append(agg_slice(fig_err, "X_figure_word_modality_error", "all"))
    rows[-1]["n"] = len(fig_err)

    bm_miss = m[(m["ndcg_BM"] >= 0.75) & (m["pred_arm"] != "BM") & (m["quality_regret"] >= 0.5)]
    print(f"BM25 was strong (>=0.75), not picked, qr>=0.5: n={len(bm_miss)}, "
          f"sum qr={bm_miss['quality_regret'].sum():.1f}; DATASET "
          f"{dict(bm_miss['DATASET'].value_counts().head(5))}")
    rows.append(agg_slice(bm_miss, "X_bm25_strong_missed", "all"))
    rows[-1]["n"] = len(bm_miss)

    # severe-error picked->best pairs (drives the top-badcase reading)
    sv = m[m["severe"]]
    print("\nsevere (top-decile reward_regret) picked x reward-best:")
    print(pd.crosstab(sv["pred_arm"], sv["best_reward_arm"]).reindex(
        index=ARMS5, columns=ARMS5, fill_value=0).fillna(0).astype(int))
    print("severe by DATASET:", dict(sv["DATASET"].value_counts()))

    # ---------------- save ----------------------------------------------------------------
    sl = pd.DataFrame(rows)
    sl = sl[["axis", "slice", "n", "pct", "mean_quality_regret", "sum_quality_regret",
             "mean_reward_regret", "sum_reward_regret", "arm_match_reward_rate",
             "mean_confidence", "hc_severe_n"]]
    sl.to_csv(os.path.join(ANALYSIS_DIR, "slice_summary.csv"), index=False)

    keep = ["query", "DATASET", "pred_arm", "best_reward_arm", "best_ndcg_arm",
            "reward_regret", "quality_regret", "margin_reward", "confidence",
            "prob_MR", "prob_MD", "prob_TR", "prob_TD", "prob_BM",
            "ndcg_MR", "ndcg_MD", "ndcg_TR", "ndcg_TD", "ndcg_BM",
            "oracle5_ndcg", "pred_ndcg", "modality_true", "modality_pred",
            "rerank_gain_text", "rerank_gain_mm", "qlen_words", "has_number",
            "has_figure_ref", "has_comparison", "n_distinct_arms_lambda",
            "arm_l00", "arm_l70", "hc_severe", "near_tie", "rerank_harm_chosen"]
    top = m.sort_values("reward_regret", ascending=False).head(200)[keep]
    top.to_csv(os.path.join(ANALYSIS_DIR, "top_badcases.csv"), index=False)
    print("\nWrote analysis/slice_summary.csv, confusion.csv, top_badcases.csv")


if __name__ == "__main__":
    main()
