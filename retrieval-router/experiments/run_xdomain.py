"""Train ONE RetrievalRouter (lambda=0.1) on 3 financial datasets, then evaluate
on the in-domain source test and three held-out target corpora.

Usage:
    cd retrieval-router
    CUDA_VISIBLE_DEVICES=1 OBJECTIVE=retrievalrouter LAMBDA=0.1 \\
        /path/to/rr-train/bin/python experiments/run_xdomain.py

Outputs:
    experiments/xdomain/results/
        checkpoint/                    LoRA adapter + classifier
        predictions/<set>_preds.csv    per-query probs, chosen arm, true nDCG, regret
        metrics.json                   aggregate metrics per set
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from train.config import (
    DATA_PATH, OUTPUT_DIR, RUN_TAG, ARM_NAMES, NDCG_COLS, LATENCY_COLS,
    LAMBDA, device,
)
from train.train import main as train_main
from train.inference import run_inference

RUN_TAG = "xc_fin3_l10"
DATA_DIR = os.path.join("experiments", "xdomain", "data")
RESULTS_DIR = os.path.join("experiments", "xdomain", "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoint")
PREDS_DIR = os.path.join(RESULTS_DIR, "predictions")

TRAIN_PATH = os.path.join(DATA_DIR, f"train_{RUN_TAG}.xlsx")
VAL_PATH = os.path.join(DATA_DIR, f"val_{RUN_TAG}.xlsx")
SOURCE_TEST_PATH = os.path.join(DATA_DIR, f"source_test_{RUN_TAG}.xlsx")

TARGET_SETS = {
    "source_test": SOURCE_TEST_PATH,
    "target_tatdqa": os.path.join(DATA_DIR, f"target_tatdqa_{RUN_TAG}.xlsx"),
    "target_wiki-ss": os.path.join(DATA_DIR, f"target_wiki-ss_{RUN_TAG}.xlsx"),
    "target_arxivqa": os.path.join(DATA_DIR, f"target_arxivqa_{RUN_TAG}.xlsx"),
}


def evaluate_set(model, tokenizer, set_name, test_path):
    """Run inference, attach probs + chosen arm + true nDCG + regret, return metrics dict."""
    df = pd.read_excel(test_path)
    probs = run_inference(model, tokenizer, df)

    pred_label = probs.argmax(axis=1)
    pred_arm = [ARM_NAMES[i] for i in pred_label]

    # Raw nDCG for each arm (true measured quality)
    ndcg_mat = df[NDCG_COLS].values  # columns follow ARM_NAMES order
    chosen_ndcg = ndcg_mat[np.arange(len(df)), pred_label]
    oracle_ndcg = ndcg_mat.max(axis=1)
    quality_regret = oracle_ndcg - chosen_ndcg

    # Reward-based regret (consistent with training objective)
    from train.prepare_data import calculate_rewards
    df_reward = df.copy()
    calculate_rewards(df_reward)
    reward_mat = df_reward[[f"{a}_reward" for a in ARM_NAMES]].values
    chosen_reward = reward_mat[np.arange(len(df)), pred_label]
    oracle_reward = reward_mat.max(axis=1)
    reward_regret = oracle_reward - chosen_reward

    # Fixed-arm nDCG
    fixed = {ARM_NAMES[i]: float(ndcg_mat[:, i].mean()) for i in range(len(ARM_NAMES))}

    out = df.copy()
    for i, a in enumerate(ARM_NAMES):
        out[f"prob_{a}"] = probs[:, i]
    out["pred_label"] = pred_label
    out["pred_arm"] = pred_arm
    out["chosen_ndcg"] = chosen_ndcg
    out["oracle_ndcg"] = oracle_ndcg
    out["quality_regret"] = quality_regret
    out["chosen_reward"] = chosen_reward
    out["reward_regret"] = reward_regret

    os.makedirs(PREDS_DIR, exist_ok=True)
    out_path = os.path.join(PREDS_DIR, f"{set_name}_preds.csv")
    out.to_csv(out_path, index=False)

    metrics = {
        "set": set_name,
        "n_queries": int(len(df)),
        "router_ndcg": float(chosen_ndcg.mean()),
        "oracle_ndcg": float(oracle_ndcg.mean()),
        "quality_regret_mean": float(quality_regret.mean()),
        "reward_regret_mean": float(reward_regret.mean()),
        "catastrophic_drop": float((quality_regret >= 0.75).mean()),
        "fixed_arms": fixed,
        "best_fixed_arm": max(fixed, key=fixed.get),
        "best_fixed_ndcg": max(fixed.values()),
    }
    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"Device: {device}, visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Train: {TRAIN_PATH}  Val: {VAL_PATH}")

    # Train on source (3 financial datasets), returns best-epoch model + tokenizer
    model, tokenizer = train_main(
        paths=(TRAIN_PATH, VAL_PATH, SOURCE_TEST_PATH),
        output_dir=CKPT_DIR,
    )
    model.eval()

    # Evaluate on all 4 sets
    all_metrics = {}
    for set_name, test_path in TARGET_SETS.items():
        print(f"\n{'='*80}\nEvaluating: {set_name}\n{'='*80}")
        m = evaluate_set(model, tokenizer, set_name, test_path)
        all_metrics[set_name] = m
        print(json.dumps(m, indent=2, ensure_ascii=False))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    header = f"{'Set':<18} {'Router':>8} {'Oracle':>8} {'QRegret':>9} {'BestFixed':>10} {'BF_ndcg':>8}"
    print(header)
    print("-" * len(header))
    for s, m in all_metrics.items():
        print(f"{s:<18} {m['router_ndcg']:>8.4f} {m['oracle_ndcg']:>8.4f} "
              f"{m['quality_regret_mean']:>9.4f} {m['best_fixed_arm']:>10} {m['best_fixed_ndcg']:>8.4f}")

    print(f"\nMetrics saved to {RESULTS_DIR}/metrics.json")
    print(f"Predictions saved to {PREDS_DIR}/")


if __name__ == "__main__":
    main()
