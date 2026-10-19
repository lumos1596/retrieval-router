"""Step 2 of the bad-case plan: rebuild every per-query quantity from RAW columns.

The released predictions file contains derived columns named ``predicted_ndcg`` /
``regret`` / ``true_ndcg_<arm>`` -- for the retrievalrouter objective these are computed on
the REWARD matrix (see train/inference.py: eval_matrix -> REWARD_COLS), so they must not be
treated as raw nDCG. This script recomputes everything from ``<arm>_ndcg`` / ``<arm>_latency``
and the official reward r_i = (1-lambda)*ndcg_i + lambda*(1 - latency_i / sum_j latency_j).

Outputs ``analysis/query_metrics.csv`` with one row per test query.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from arms import ARM_NAMES, SHORT_NAMES, NON_ARM_PIPELINES  # noqa: E402

LAMBDA = 0.1
PRED_PATH = os.path.join(REPO_ROOT, "evaluation", "predictions", "retrievalrouter_l10.xlsx")
L00_PATH = os.path.join(REPO_ROOT, "evaluation", "predictions", "retrievalrouter_l00.xlsx")
L70_PATH = os.path.join(REPO_ROOT, "evaluation", "predictions", "retrievalrouter_l70.xlsx")
OUT_PATH = os.path.join(REPO_ROOT, "analysis", "query_metrics.csv")

SHORT = {a: SHORT_NAMES[a] for a in ARM_NAMES}

# --- query feature regexes (axis C) ---------------------------------------------------
RE_FIGURE = re.compile(r"\b(figure|fig\.?|chart|graph|table|plot|diagram|image|picture|photo|page|slide)\b", re.I)
RE_COMPARE = re.compile(
    r"\b(more than|less than|most|least|between|versus|vs\.?|compared?|comparison|difference|"
    r"higher|lower|better|worse|largest|smallest|increase|decrease|percent|percentage)\b", re.I)
RE_NUMBER = re.compile(r"\d")
RE_EXACT = re.compile(r'["“”]')
RE_ENTITY_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def query_features(q: str) -> dict:
    q = str(q)
    words = re.findall(r"\w+", q)
    return {
        "qlen_chars": len(q),
        "qlen_words": len(words),
        "has_number": bool(RE_NUMBER.search(q)),
        "has_year": bool(RE_ENTITY_YEAR.search(q)),
        "has_figure_ref": bool(RE_FIGURE.search(q)),
        "has_comparison": bool(RE_COMPARE.search(q)),
        "has_quoted": bool(RE_EXACT.search(q)),
        "n_clauses": q.count(",") + len(re.findall(r"\band\b|\bthat\b|\bwhich\b|\bwho\b|\bwhere\b|\bwhen\b", q, re.I)),
    }


def main():
    df = pd.read_excel(PRED_PATH)
    n = len(df)

    ndcg = df[[f"{a}_ndcg" for a in ARM_NAMES]].to_numpy(dtype=float)          # (n,5) MR,MD,TR,TD,BM
    lat = df[[f"{a}_latency" for a in ARM_NAMES]].to_numpy(dtype=float)
    probs = df[[f"pred_prob_{a}" for a in ARM_NAMES]].to_numpy(dtype=float)
    reward = (1 - LAMBDA) * ndcg + LAMBDA * (1 - lat / (lat.sum(axis=1, keepdims=True) + 1e-9))

    pred_idx = df["predicted_label"].to_numpy(dtype=int)
    rows = np.arange(n)

    # ---- routed outcomes ---------------------------------------------------------------
    pred_ndcg = ndcg[rows, pred_idx]
    pred_reward = reward[rows, pred_idx]
    oracle5_ndcg = ndcg.max(axis=1)
    oracle5_reward = reward.max(axis=1)
    best_ndcg_idx = ndcg.argmax(axis=1)
    best_reward_idx = reward.argmax(axis=1)

    order_ndcg = np.sort(ndcg, axis=1)
    order_rew = np.sort(reward, axis=1)
    margin_ndcg = order_ndcg[:, -1] - order_ndcg[:, -2]
    margin_reward = order_rew[:, -1] - order_rew[:, -2]

    # seven-arm oracle (TL/ML measured, not routable) -- reference upper bound only
    seven = np.column_stack([ndcg, df[[f"{p}_ndcg" for p in NON_ARM_PIPELINES]].to_numpy(dtype=float)])
    oracle7_ndcg = seven.max(axis=1)

    confidence = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / np.log(5)  # normalised to [0,1]

    out = pd.DataFrame({
        "query": df["query"],
        "DATASET": df["DATASET"],
        "pred_arm": [SHORT[a] for a in df["predicted_pipeline"]],
        "best_reward_arm": [SHORT[ARM_NAMES[i]] for i in best_reward_idx],
        "best_ndcg_arm": [SHORT[ARM_NAMES[i]] for i in best_ndcg_idx],
        "pred_ndcg": pred_ndcg,
        "oracle5_ndcg": oracle5_ndcg,
        "oracle7_ndcg": oracle7_ndcg,
        "quality_regret": oracle5_ndcg - pred_ndcg,
        "pred_reward": pred_reward,
        "oracle5_reward": oracle5_reward,
        "reward_regret": oracle5_reward - pred_reward,
        "margin_reward": margin_reward,
        "margin_ndcg": margin_ndcg,
        "confidence": confidence,
        "prob_entropy_norm": entropy,
        "arm_match_reward": pred_idx == best_reward_idx,
        "arm_match_ndcg": pred_idx == best_ndcg_idx,
    })

    for j, a in enumerate(ARM_NAMES):
        out[f"ndcg_{SHORT[a]}"] = ndcg[:, j]
        out[f"lat_{SHORT[a]}"] = lat[:, j]
        out[f"reward_{SHORT[a]}"] = reward[:, j]
        out[f"prob_{SHORT[a]}"] = probs[:, j]

    # ---- modality (axis D) --------------------------------------------------------------
    text_idx = [ARM_NAMES.index(a) for a in ("BM25", "TEXT-SINGLE", "TEXT_RERANK")]
    vis_idx = [ARM_NAMES.index(a) for a in ("MULTIMODAL-SINGLE", "MULTIMODAL_RERANK")]
    text_best = ndcg[:, text_idx].max(axis=1)
    vis_best = ndcg[:, vis_idx].max(axis=1)
    MOD_DELTA = 0.02  # nDCG gap needed to call a query genuinely modality-demanding
    modality_true = np.where(vis_best - text_best > MOD_DELTA, "visual",
                    np.where(text_best - vis_best > MOD_DELTA, "text", "tie"))
    modality_pred = np.where(np.isin(pred_idx, vis_idx), "visual", "text")
    out["text_best_ndcg"] = text_best
    out["vis_best_ndcg"] = vis_best
    out["modality_true"] = modality_true
    out["modality_pred"] = modality_pred
    out["modality_error"] = (modality_true != "tie") & (modality_true != modality_pred)

    # ---- rerank deltas (axis E) ----------------------------------------------------------
    td, tr = ndcg[:, ARM_NAMES.index("TEXT-SINGLE")], ndcg[:, ARM_NAMES.index("TEXT_RERANK")]
    md, mr = ndcg[:, ARM_NAMES.index("MULTIMODAL-SINGLE")], ndcg[:, ARM_NAMES.index("MULTIMODAL_RERANK")]
    out["rerank_gain_text"] = tr - td
    out["rerank_gain_mm"] = mr - md
    pred_name = df["predicted_pipeline"].to_numpy()
    out["picked_rerank"] = np.isin(pred_name, ["TEXT_RERANK", "MULTIMODAL_RERANK"])
    # rerank chosen but its dense variant would have been strictly better (>0.02 nDCG)
    rerank_harm = np.where(
        pred_name == "TEXT_RERANK", (td - tr) > MOD_DELTA,
        np.where(pred_name == "MULTIMODAL_RERANK", (md - mr) > MOD_DELTA, False))
    out["rerank_harm_chosen"] = rerank_harm

    # ---- query features (axis C) ---------------------------------------------------------
    feats = pd.DataFrame([query_features(q) for q in df["query"]])
    out = pd.concat([out, feats], axis=1)

    # ---- lambda stability (axis G) -------------------------------------------------------
    d00 = pd.read_excel(L00_PATH)
    d70 = pd.read_excel(L70_PATH)
    assert (d00["query"].to_numpy() == df["query"].to_numpy()).all()
    assert (d70["query"].to_numpy() == df["query"].to_numpy()).all()
    out["arm_l00"] = [SHORT[a] for a in d00["predicted_pipeline"]]
    out["arm_l70"] = [SHORT[a] for a in d70["predicted_pipeline"]]
    stacked = np.column_stack([out["arm_l00"], out["pred_arm"], out["arm_l70"]])
    out["n_distinct_arms_lambda"] = [len(set(row)) for row in stacked]
    out["switches_vs_l00"] = out["arm_l00"] != out["pred_arm"]
    out["switches_vs_l70"] = out["arm_l70"] != out["pred_arm"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}: {out.shape[0]} rows x {out.shape[1]} cols")
    print(f"mean routed nDCG@5          = {out['pred_ndcg'].mean():.6f}")
    print(f"mean 5-arm oracle nDCG@5    = {out['oracle5_ndcg'].mean():.6f}")
    print(f"mean 7-arm oracle nDCG@5    = {out['oracle7_ndcg'].mean():.6f}")
    print(f"total quality_regret        = {out['quality_regret'].sum():.4f}")
    print(f"total reward_regret         = {out['reward_regret'].sum():.4f}")


if __name__ == "__main__":
    main()
