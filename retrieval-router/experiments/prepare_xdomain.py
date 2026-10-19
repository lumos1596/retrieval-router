"""Cross-domain data preparation for RetrievalRouter generalization experiment.

Splits:
  Source (financial): finqa, convfinqa, vqaonbd -> train/val/source_test (80/10/10)
  Target (held out):  tatdqa, wiki-ss, arxivqa  -> test only (official 10% test subset)

Rewards are attached via RetrievalRouterObjective.add_targets so the saved Excel
files are directly readable by train.train.main(paths=...).

Run:
    cd retrieval-router
    /path/to/rr-train/bin/python experiments/prepare_xdomain.py
"""
import os
import sys

# Make `train` importable when run as a script from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from train.config import DATA_PATH, SAMPLE_RATE, RANDOM_SEED, LAMBDA, REWARD_COLS
from train.prepare_data import calculate_rewards
from arms import ARM_NAMES

RUN_TAG = "xc_fin3_l10"
OUT_DIR = os.path.join("experiments", "xdomain", "data")

SOURCE_DATASETS = ["finqa", "convfinqa", "vqaonbd"]
TARGET_DATASETS = ["tatdqa", "wiki-ss", "arxivqa"]


def split_dataset(dataset_df, sample_rate=SAMPLE_RATE, seed=RANDOM_SEED):
    """Reproduce the official per-dataset 80/10/10 split (test first, then val)."""
    total = len(dataset_df)
    n_test = int(total * sample_rate)
    n_val = int(total * sample_rate)

    test_df = dataset_df.sample(n=n_test, random_state=seed)
    remaining = dataset_df.drop(test_df.index)
    val_df = remaining.sample(n=n_val, random_state=seed)
    train_df = remaining.drop(val_df.index)
    return train_df, val_df, test_df


def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH).dropna()
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "query"}, inplace=True)
    print(f"Total rows: {len(df)}")

    # --- Source datasets: full 80/10/10 split ---
    train_parts, val_parts, src_test_parts = [], [], []
    for ds in SOURCE_DATASETS:
        sub = df[df["DATASET"] == ds]
        tr, va, te = split_dataset(sub)
        train_parts.append(tr)
        val_parts.append(va)
        src_test_parts.append(te)
        print(f"  source {ds}: train={len(tr)} val={len(va)} test={len(te)}")

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    source_test_df = pd.concat(src_test_parts, ignore_index=True)

    # --- Target datasets: official 10% test subset only ---
    target_tests = {}
    for ds in TARGET_DATASETS:
        sub = df[df["DATASET"] == ds]
        _, _, te = split_dataset(sub)
        target_tests[ds] = te
        print(f"  target {ds}: test={len(te)} (of {len(sub)})")

    # --- Attach rewards (shared, lambda from config / env) ---
    print(f"\nAttaching rewards (lambda={LAMBDA})...")
    for name, d in [("train", train_df), ("val", val_df),
                    ("source_test", source_test_df)] + \
                   [(f"target_{ds}", target_tests[ds]) for ds in TARGET_DATASETS]:
        calculate_rewards(d)
        # Sanity: reward columns present
        missing = [c for c in REWARD_COLS if c not in d.columns]
        assert not missing, f"{name}: missing reward cols {missing}"

    # --- Save ---
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {
        "train": os.path.join(OUT_DIR, f"train_{RUN_TAG}.xlsx"),
        "val": os.path.join(OUT_DIR, f"val_{RUN_TAG}.xlsx"),
        "source_test": os.path.join(OUT_DIR, f"source_test_{RUN_TAG}.xlsx"),
    }
    for ds in TARGET_DATASETS:
        paths[f"target_{ds}"] = os.path.join(OUT_DIR, f"target_{ds}_{RUN_TAG}.xlsx")

    train_df.to_excel(paths["train"], index=False)
    val_df.to_excel(paths["val"], index=False)
    source_test_df.to_excel(paths["source_test"], index=False)
    for ds in TARGET_DATASETS:
        target_tests[ds].to_excel(paths[f"target_{ds}"], index=False)

    print("\nSaved:")
    for k, p in paths.items():
        print(f"  {k:20s} -> {p}")

    # Overlap check: no query id shared across train/val and any target test
    train_ids = set(train_df["query"])
    for ds in TARGET_DATASETS:
        overlap = train_ids & set(target_tests[ds]["query"])
        print(f"  overlap train vs target_{ds}: {len(overlap)}")

    # Also report per-arm mean nDCG on each target set (oracle + fixed arms)
    print("\nPer-set mean nDCG by arm (raw <arm>_ndcg):")
    ndcg_cols = [f"{a}_ndcg" for a in ARM_NAMES]
    for name, d in [("source_test", source_test_df)] + \
                   [(f"target_{ds}", target_tests[ds]) for ds in TARGET_DATASETS]:
        means = d[ndcg_cols].mean().round(4).to_dict()
        print(f"  {name:20s}: {means}")


if __name__ == "__main__":
    main()
