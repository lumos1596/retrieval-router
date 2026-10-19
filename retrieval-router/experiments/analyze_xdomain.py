"""Cross-domain error analysis for RetrievalRouter.

Loads per-query predictions from experiments/xdomain/results/predictions/ and
produces:
  - error type breakdown (text<->visual, dense<->rerank) per set
  - router arm-selection distribution vs oracle arm distribution
  - Top-20 quality-regret bad cases per target set
  - confusion matrix (chosen x best) per set
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arms import ARM_NAMES, SHORT_NAMES

PREDS_DIR = os.path.join("experiments", "xdomain", "results", "predictions")
OUT_DIR = os.path.join("experiments", "xdomain", "results")

SETS = ["source_test", "target_tatdqa", "target_wiki-ss", "target_arxivqa"]

# Arm groupings
VISUAL = {"MULTIMODAL_RERANK", "MULTIMODAL-SINGLE"}
TEXT = {"TEXT_RERANK", "TEXT-SINGLE", "BM25"}
RERANK = {"MULTIMODAL_RERANK", "TEXT_RERANK"}
DENSE = {"MULTIMODAL-SINGLE", "TEXT-SINGLE", "BM25"}


def short(arm):
    return SHORT_NAMES.get(arm, arm)


def load_preds(set_name):
    return pd.read_csv(os.path.join(PREDS_DIR, f"{set_name}_preds.csv"))


def error_types(df):
    """Return dict of error-type counts and cumulative quality_regret."""
    chosen = df["pred_arm"]
    best = df[f"{ARM_NAMES[0]}_ndcg"]  # placeholder
    # determine best arm per row from raw ndcg columns
    ndcg_cols = [f"{a}_ndcg" for a in ARM_NAMES]
    best_idx = df[ndcg_cols].values.argmax(axis=1)
    best_arm = pd.Series([ARM_NAMES[i] for i in best_idx], index=df.index)

    out = {}

    # text <-> visual
    chosen_visual = chosen.isin(VISUAL)
    best_visual = best_arm.isin(VISUAL)
    # chose text but best was visual
    txt_to_vis = (~chosen_visual) & best_visual
    vis_to_txt = chosen_visual & (~best_visual)
    out["text_chosen_visual_best"] = {
        "count": int(txt_to_vis.sum()),
        "frac": float(txt_to_vis.mean()),
        "cum_regret": float(df.loc[txt_to_vis, "quality_regret"].sum()),
    }
    out["visual_chosen_text_best"] = {
        "count": int(vis_to_txt.sum()),
        "frac": float(vis_to_txt.mean()),
        "cum_regret": float(df.loc[vis_to_txt, "quality_regret"].sum()),
    }

    # dense <-> rerank (only among same-modality pairs; BM25 excluded as lexical)
    chosen_rerank = chosen.isin(RERANK)
    best_rerank = best_arm.isin(RERANK)
    dense_to_rerank = (~chosen_rerank) & chosen.isin(DENSE) & best_rerank
    rerank_to_dense = chosen_rerank & best_arm.isin(DENSE) & (~best_rerank)
    out["dense_chosen_rerank_best"] = {
        "count": int(dense_to_rerank.sum()),
        "frac": float(dense_to_rerank.mean()),
        "cum_regret": float(df.loc[dense_to_rerank, "quality_regret"].sum()),
    }
    out["rerank_chosen_dense_best"] = {
        "count": int(rerank_to_dense.sum()),
        "frac": float(rerank_to_dense.mean()),
        "cum_regret": float(df.loc[rerank_to_dense, "quality_regret"].sum()),
    }

    # catastrophic: oracle >= 0.75 but chosen ndcg == 0
    cat = (df["oracle_ndcg"] >= 0.75) & (df["chosen_ndcg"] == 0)
    out["catastrophic"] = {
        "count": int(cat.sum()),
        "frac": float(cat.mean()),
        "cum_regret": float(df.loc[cat, "quality_regret"].sum()),
    }

    return out


def arm_distribution(df):
    """Router's chosen-arm distribution vs oracle (best-arm) distribution."""
    ndcg_cols = [f"{a}_ndcg" for a in ARM_NAMES]
    best_idx = df[ndcg_cols].values.argmax(axis=1)
    best_arm = pd.Series([ARM_NAMES[i] for i in best_idx])

    chosen_dist = df["pred_arm"].value_counts(normalize=True).to_dict()
    best_dist = best_arm.value_counts(normalize=True).to_dict()

    dist = {}
    for a in ARM_NAMES:
        dist[short(a)] = {
            "chosen_frac": round(chosen_dist.get(a, 0.0), 4),
            "oracle_frac": round(best_dist.get(a, 0.0), 4),
        }
    return dist


def confusion(df):
    """chosen x best arm count matrix."""
    ndcg_cols = [f"{a}_ndcg" for a in ARM_NAMES]
    best_idx = df[ndcg_cols].values.argmax(axis=1)
    best_arm = pd.Series([ARM_NAMES[i] for i in best_idx], index=df.index)

    mat = pd.crosstab(df["pred_arm"], best_arm)
    # reindex to ARM_NAMES order
    mat = mat.reindex(index=ARM_NAMES, columns=ARM_NAMES, fill_value=0)
    mat.index = [short(a) for a in mat.index]
    mat.columns = [short(a) for a in mat.columns]
    return mat


def top_badcases(df, set_name, n=20):
    cols = ["DATASET", "query", "pred_arm", "chosen_ndcg", "oracle_ndcg",
            "quality_regret"] + [f"{a}_ndcg" for a in ARM_NAMES] + \
           [f"prob_{a}" for a in ARM_NAMES]
    sub = df.sort_values("quality_regret", ascending=False).head(n)[cols].copy()
    # add best arm
    ndcg_cols = [f"{a}_ndcg" for a in ARM_NAMES]
    best_idx = sub[ndcg_cols].values.argmax(axis=1)
    sub["best_arm"] = [short(ARM_NAMES[i]) for i in best_idx]
    sub["pred_arm"] = sub["pred_arm"].map(short)
    return sub


def main():
    summary = {}
    for s in SETS:
        df = load_preds(s)
        summary[s] = {
            "n": int(len(df)),
            "router_ndcg": round(float(df["chosen_ndcg"].mean()), 4),
            "oracle_ndcg": round(float(df["oracle_ndcg"].mean()), 4),
            "quality_regret_mean": round(float(df["quality_regret"].mean()), 4),
            "error_types": error_types(df),
            "arm_distribution": arm_distribution(df),
        }
        # save confusion matrix
        cm = confusion(df)
        cm.to_csv(os.path.join(OUT_DIR, f"confusion_{s}.csv"))
        # save top bad cases
        bc = top_badcases(df, s, 20)
        bc.to_csv(os.path.join(OUT_DIR, f"badcases_{s}.csv"), index=False)

    with open(os.path.join(OUT_DIR, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Print human-readable report ---
    print("=" * 90)
    print("CROSS-DOMAIN ERROR ANALYSIS")
    print("=" * 90)

    print("\n## 1. Arm selection: Router vs Oracle (fraction)")
    for s in SETS:
        print(f"\n  [{s}]")
        d = summary[s]["arm_distribution"]
        print(f"  {'arm':>5} {'chosen':>8} {'oracle':>8} {'delta':>8}")
        for a in [short(x) for x in ARM_NAMES]:
            ch = d[a]["chosen_frac"]
            orc = d[a]["oracle_frac"]
            print(f"  {a:>5} {ch:>8.3f} {orc:>8.3f} {ch-orc:>+8.3f}")

    print("\n## 2. Error types (count / frac / cum_regret)")
    for s in SETS:
        print(f"\n  [{s}]  n={summary[s]['n']}")
        et = summary[s]["error_types"]
        for k, v in et.items():
            print(f"  {k:>30s}: {v['count']:>5d}  ({v['frac']*100:>5.1f}%)  cum_regret={v['cum_regret']:.1f}")

    print("\n## 3. Main metrics recap")
    print(f"  {'set':<18} {'Router':>8} {'Oracle':>8} {'QRegret':>9} {'Catast%':>9}")
    for s in SETS:
        m = summary[s]
        cat = m["error_types"]["catastrophic"]["frac"] * 100
        print(f"  {s:<18} {m['router_ndcg']:>8.4f} {m['oracle_ndcg']:>8.4f} "
              f"{m['quality_regret_mean']:>9.4f} {cat:>8.1f}%")

    print(f"\nAnalysis saved to {OUT_DIR}/analysis.json")
    print(f"Confusion matrices: {OUT_DIR}/confusion_*.csv")
    print(f"Bad cases: {OUT_DIR}/badcases_*.csv")


if __name__ == "__main__":
    main()
