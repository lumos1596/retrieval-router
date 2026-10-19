"""Local capability probe experiment (zero-training).

For each unseen target corpus, sample a small probe set (20/50/100 queries) whose
true 5-arm nDCG is known. Test whether this local capability information helps
routing decisions on the remaining evaluation queries, beyond:
  A. Query-only Router (lambda=0 and lambda=0.1)
  B. Fixed-MR
  C. Global Probe (corpus-level best arm from probe means)
  D. Local Probe (kNN-weighted nDCG prediction, argmax)
  E. Best Fixed (oracle: best arm on eval set, not deployable)
  F. Per-query Oracle (not deployable)

Outputs:
  experiments/probe/probe_results.csv
  experiments/probe/probe_metrics.md
  experiments/probe/probe_casebook.md
  experiments/probe/probe_conclusion.md
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from arms import ARM_NAMES, SHORT_NAMES

# Arm order: MR, MD, TR, TD, BM
NDCG_COLS = [f"{a}_ndcg" for a in ARM_NAMES]

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
K = 5
TAU = 0.1
PROBE_SIZES = [20, 50, 100]
N_SEEDS = 10
NESTED_CANDIDATES = 100  # sample 100, use first 20/50/100

DATASETS = {
    "tatdqa": "experiments/xdomain/data/target_tatdqa_xc_fin3_l10.xlsx",
    "wiki-ss": "experiments/xdomain/data/target_wiki-ss_xc_fin3_l10.xlsx",
    "arxivqa": "experiments/xdomain/data/target_arxivqa_xc_fin3_l10.xlsx",
}

ROUTER_PREDS = {
    "l00": {  # lambda=0
        "tatdqa": "experiments/xdomain/results_l00/predictions/target_tatdqa_preds.csv",
        "wiki-ss": "experiments/xdomain/results_l00/predictions/target_wiki-ss_preds.csv",
        "arxivqa": "experiments/xdomain/results_l00/predictions/target_arxivqa_preds.csv",
    },
    "l10": {  # lambda=0.1
        "tatdqa": "experiments/xdomain/results/predictions/target_tatdqa_preds.csv",
        "wiki-ss": "experiments/xdomain/results/predictions/target_wiki-ss_preds.csv",
        "arxivqa": "experiments/xdomain/results/predictions/target_arxivqa_preds.csv",
    },
}

OUT_DIR = "experiments/probe"


def short(arm):
    return SHORT_NAMES.get(arm, arm)


def load_dataset(name):
    """Load target dataset with 5-arm nDCG and attach router predictions."""
    df = pd.read_excel(DATASETS[name])
    # Deduplicate by query
    df = df.drop_duplicates(subset="query").reset_index(drop=True)

    # Attach lambda=0 router predictions
    l00 = pd.read_csv(ROUTER_PREDS["l00"][name])[["query", "pred_arm"]].rename(
        columns={"pred_arm": "pred_arm_l00"}
    )
    l10 = pd.read_csv(ROUTER_PREDS["l10"][name])[["query", "pred_arm"]].rename(
        columns={"pred_arm": "pred_arm_l10"}
    )
    df = df.merge(l00, on="query", how="left")
    df = df.merge(l10, on="query", how="left")

    # Some queries might not have router predictions (rare), drop them
    df = df.dropna(subset=["pred_arm_l00", "pred_arm_l10"]).reset_index(drop=True)
    return df


def encode_queries(queries, model=None, device="cuda"):
    """Encode queries with bge-small-en-v1.5, L2-normalize."""
    from sentence_transformers import SentenceTransformer
    if model is None:
        model = SentenceTransformer(EMBED_MODEL, device=device)
    emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True, batch_size=256)
    return emb.astype(np.float32)


def make_probe_splits(n_total, seed, nested=NESTED_CANDIDATES):
    """Return dict: probe_size -> (probe_indices, eval_indices).

    Nested: sample `nested` candidates with this seed; probe set of size s uses
    the first s candidates. Eval = everything not in the largest probe set (100).
    """
    rng = np.random.default_rng(seed)
    candidates = rng.choice(n_total, size=min(nested, n_total), replace=False)
    eval_mask = np.ones(n_total, dtype=bool)
    eval_mask[candidates] = False
    eval_idx = np.where(eval_mask)[0]

    splits = {}
    for s in PROBE_SIZES:
        s = min(s, len(candidates))
        probe_idx = candidates[:s]
        splits[s] = (probe_idx, eval_idx)
    return splits


def global_probe_arm(ndcg_probe):
    """Pick the arm with highest mean nDCG in the probe set."""
    return int(ndcg_probe.mean(axis=0).argmax())


def local_probe_arms(emb_eval, emb_probe, ndcg_probe, k=K, tau=TAU):
    """For each eval query, weighted-average nDCG over k nearest probes, argmax.

    emb_eval: (Ne, D), emb_probe: (Np, D), ndcg_probe: (Np, 5)
    Returns array of chosen arm indices (Ne,).
    """
    # Cosine similarity (embeddings are L2-normalized)
    sim = emb_eval @ emb_probe.T  # (Ne, Np)
    Ne = sim.shape[0]
    # Top-k nearest probes per query
    if k >= sim.shape[1]:
        topk_idx = np.broadcast_to(np.arange(sim.shape[1]), (Ne, sim.shape[1]))
    else:
        topk_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    # Gather similarities and nDCG
    rows = np.arange(Ne)[:, None]
    sim_k = sim[rows, topk_idx]  # (Ne, k)
    ndcg_k = ndcg_probe[topk_idx]  # (Ne, k, 5)
    # Softmax weights over k probes
    w = np.exp(sim_k / tau)
    w = w / w.sum(axis=1, keepdims=True)  # (Ne, k)
    # Weighted average nDCG
    pred_ndcg = (w[:, :, None] * ndcg_k).sum(axis=1)  # (Ne, 5)
    return pred_ndcg.argmax(axis=1)


def evaluate_choice(chosen_arms, ndcg_eval):
    """Compute nDCG, regret, catastrophic drop rate for chosen arms."""
    chosen_ndcg = ndcg_eval[np.arange(len(chosen_arms)), chosen_arms]
    oracle_ndcg = ndcg_eval.max(axis=1)
    regret = oracle_ndcg - chosen_ndcg
    catastrophic = (oracle_ndcg >= 0.75) & (chosen_ndcg == 0)
    return {
        "nDCG": float(chosen_ndcg.mean()),
        "regret": float(regret.mean()),
        "catastrophic": float(catastrophic.mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load datasets ---
    datasets = {}
    all_queries = {}
    for name in DATASETS:
        df = load_dataset(name)
        datasets[name] = df
        all_queries[name] = df["query"].tolist()
        print(f"{name}: {len(df)} queries")

    # --- Encode queries (once per dataset) ---
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=device)
    embeddings = {}
    for name in all_queries:
        print(f"Encoding {name}...")
        embeddings[name] = encode_queries(all_queries[name], model=model, device=device)

    # --- Run experiment ---
    results = []  # per (dataset, seed, probe_size, method) row

    for ds_name, df in datasets.items():
        ndcg = df[NDCG_COLS].values.astype(np.float32)  # (N, 5)
        emb = embeddings[ds_name]
        n = len(df)

        # Arm index for router predictions
        arm_to_idx = {a: i for i, a in enumerate(ARM_NAMES)}
        router_l00 = df["pred_arm_l00"].map(arm_to_idx).values
        router_l10 = df["pred_arm_l10"].map(arm_to_idx).values
        fixed_mr = np.full(n, arm_to_idx["MULTIMODAL_RERANK"])

        for seed in range(N_SEEDS):
            splits = make_probe_splits(n, seed)

            for psize, (probe_idx, eval_idx) in splits.items():
                ndcg_eval = ndcg[eval_idx]
                ndcg_probe = ndcg[probe_idx]
                emb_eval = emb[eval_idx]
                emb_probe = emb[probe_idx]

                # A. Query-only Router (lambda=0)
                m = evaluate_choice(router_l00[eval_idx], ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "A_l00", **m})
                # A. Query-only Router (lambda=0.1)
                m = evaluate_choice(router_l10[eval_idx], ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "A_l10", **m})
                # B. Fixed-MR
                m = evaluate_choice(fixed_mr[eval_idx], ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "B_fixedMR", **m})
                # C. Global Probe
                g_arm = global_probe_arm(ndcg_probe)
                chosen = np.full(len(eval_idx), g_arm)
                m = evaluate_choice(chosen, ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "C_global", **m})
                # D. Local Probe
                chosen = local_probe_arms(emb_eval, emb_probe, ndcg_probe)
                m = evaluate_choice(chosen, ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "D_local", **m})
                # E. Best Fixed (oracle on eval set)
                best_fixed_arm = int(ndcg_eval.mean(axis=0).argmax())
                chosen = np.full(len(eval_idx), best_fixed_arm)
                m = evaluate_choice(chosen, ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "E_bestfixed", **m})
                # F. Per-query Oracle
                chosen = ndcg_eval.argmax(axis=1)
                m = evaluate_choice(chosen, ndcg_eval)
                results.append({"dataset": ds_name, "seed": seed, "probe_size": psize,
                                "method": "F_oracle", **m})

        print(f"  {ds_name}: done")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "probe_results.csv"), index=False)
    print(f"\nResults saved to {OUT_DIR}/probe_results.csv")
    print(f"Total rows: {len(res_df)}")

    # --- Aggregate metrics (mean +/- std over seeds) ---
    agg = res_df.groupby(["dataset", "probe_size", "method"]).agg(
        nDCG_mean=("nDCG", "mean"),
        nDCG_std=("nDCG", "std"),
        regret_mean=("regret", "mean"),
        catastrophic_mean=("catastrophic", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "probe_aggregated.csv"), index=False)

    # Print summary
    print("\n=== nDCG summary (mean over seeds) ===")
    for ds in datasets:
        print(f"\n[{ds}]")
        sub = agg[agg["dataset"] == ds]
        for ps in PROBE_SIZES:
            row = sub[sub["probe_size"] == ps]
            line = f"  psize={ps:3d}: "
            for m in ["A_l00", "A_l10", "B_fixedMR", "C_global", "D_local", "E_bestfixed", "F_oracle"]:
                v = row[row["method"] == m]["nDCG_mean"]
                line += f"{m[:8]}={v.values[0]:.4f}  " if len(v) else f"{m[:8]}=  N/A  "
            print(line)


if __name__ == "__main__":
    main()
