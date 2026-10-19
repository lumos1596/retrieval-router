"""Compute reasoning router metrics and generate deliverables."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from arms import ARM_NAMES, NDCG_COLS, SHORT_NAMES

OUT = "experiments/reasoning"
MODELS = ["P0", "P1", "P2"]
MR_IDX = ARM_NAMES.index("MULTIMODAL_RERANK")
ARM_SHORT = {a: SHORT_NAMES[a] for a in ARM_NAMES}

def load_all():
    frames = []
    for ds in ["val", "tatdqa", "wiki-ss", "arxivqa"]:
        path = os.path.join(OUT, f"reasoning_results_{ds}.json")
        if os.path.exists(path):
            df = pd.read_json(path)
            df["dataset"] = ds
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

def ndcg_for_arm(row, arm_col):
    arm = row[arm_col]
    if pd.isna(arm) or arm is None:
        return np.nan
    return row[f"ndcg_{arm}"]

def bucket(row):
    mr = row["mr_ndcg"]
    others = [row[f"ndcg_{a}"] for a in ARM_NAMES if a != "MULTIMODAL_RERANK"]
    best_other = max(others)
    d = best_other - mr
    if d > 0.1: return "NonMR-advantage"
    if d < -0.1: return "MR-advantage"
    return "Near-tie"

def main():
    df = load_all()
    print(f"Loaded {len(df)} rows, datasets: {df['dataset'].unique()}")

    # Compute chosen ndcg per model
    for m in MODELS:
        df[f"{m}_chosen"] = df.apply(lambda r: ndcg_for_arm(r, f"{m}_arm"), axis=1)

    # Drop parse failures for metric computation (per plan, don't default to MR)
    df_clean = df.copy()
    for m in MODELS:
        df_clean = df_clean[df_clean[f"{m}_ok"] == True]
    print(f"After dropping parse failures: {len(df_clean)} rows")

    rows = []
    for ds in df_clean["dataset"].unique():
        sub = df_clean[df_clean["dataset"] == ds]
        for m in MODELS + ["FixedMR", "Oracle"]:
            if m == "FixedMR":
                chosen = sub["mr_ndcg"].values
            elif m == "Oracle":
                chosen = sub["oracle_ndcg"].values
            else:
                chosen = sub[f"{m}_chosen"].values
            mr = sub["mr_ndcg"].values
            oracle = sub["oracle_ndcg"].values
            net = float(np.mean(chosen - mr))
            g_plus = float(np.mean(np.maximum(chosen - mr, 0)))
            g_minus = float(np.mean(np.maximum(mr - chosen, 0)))
            rows.append({
                "dataset": ds, "model": m, "n": len(sub),
                "ndcg": float(np.mean(chosen)),
                "regret": float(np.mean(oracle - chosen)),
                "catastrophic_rate": float(np.mean((oracle >= 0.75) & (chosen == 0))),
                "net_vs_mr": net, "g_plus": g_plus, "g_minus": g_minus,
                "mr_sel_rate": float(np.mean(sub[f"{m}_arm"] == "MULTIMODAL_RERANK")) if m in MODELS else 1.0,
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUT, "reasoning_metrics_table.csv"), index=False)

    # Arm distribution
    arm_rows = []
    for ds in df_clean["dataset"].unique():
        sub = df_clean[df_clean["dataset"] == ds]
        for m in MODELS:
            counts = sub[f"{m}_arm"].value_counts(normalize=True)
            row = {"dataset": ds, "model": m}
            for a in ARM_NAMES:
                row[ARM_SHORT[a]] = float(counts.get(a, 0))
            arm_rows.append(row)
    arm_dist = pd.DataFrame(arm_rows)
    arm_dist.to_csv(os.path.join(OUT, "reasoning_arm_distribution.csv"), index=False)

    # Stratified by bucket
    df_clean["bucket"] = df_clean.apply(bucket, axis=1)
    strat_rows = []
    for ds in df_clean["dataset"].unique():
        for bk in ["NonMR-advantage", "MR-advantage", "Near-tie"]:
            sub = df_clean[(df_clean["dataset"] == ds) & (df_clean["bucket"] == bk)]
            if len(sub) == 0: continue
            for m in MODELS:
                chosen = sub[f"{m}_chosen"].values
                mr = sub["mr_ndcg"].values
                strat_rows.append({
                    "dataset": ds, "bucket": bk, "n": len(sub), "model": m,
                    "nonMR_sel_rate": float(np.mean(sub[f"{m}_arm"] != "MULTIMODAL_RERANK")),
                    "ndcg": float(np.mean(chosen)),
                    "net_vs_mr": float(np.mean(chosen - mr)),
                })
    strat = pd.DataFrame(strat_rows)
    strat.to_csv(os.path.join(OUT, "reasoning_stratified.csv"), index=False)

    # Bootstrap paired CIs for P2 vs P0/P1
    print("\nBootstrap paired CIs (P2 - P0, P2 - P1)...")
    bs_rows = []
    for ds in df_clean["dataset"].unique():
        sub = df_clean[df_clean["dataset"] == ds]
        for baseline in ["P0", "P1"]:
            diff = sub["P2_chosen"].values - sub[f"{baseline}_chosen"].values
            means = []
            rng = np.random.RandomState(42)
            for _ in range(1000):
                idx = rng.choice(len(diff), len(diff), replace=True)
                means.append(diff[idx].mean())
            lo, hi = np.percentile(means, [2.5, 97.5])
            bs_rows.append({"dataset": ds, "comparison": f"P2-{baseline}",
                "mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)})
    bs = pd.DataFrame(bs_rows)
    bs.to_csv(os.path.join(OUT, "reasoning_bootstrap_ci.csv"), index=False)

    # Cost
    cost_rows = []
    for ds in df_clean["dataset"].unique():
        sub = df_clean[df_clean["dataset"] == ds]
        for m in MODELS:
            cost_rows.append({"dataset": ds, "model": m,
                "avg_tokens": float(sub[f"{m}_tokens"].mean()),
                "avg_output_chars": float(sub[f"{m}_resp"].str.len().mean())})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(os.path.join(OUT, "reasoning_cost.csv"), index=False)

    print("\n=== MAIN METRICS ===")
    print(metrics[["dataset","model","ndcg","regret","catastrophic_rate","net_vs_mr"]].to_string(index=False))
    print("\n=== BOOTSTRAP CI ===")
    print(bs.to_string(index=False))
    print("\nSaved all metric tables.")

if __name__ == "__main__":
    main()
