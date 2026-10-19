"""Select the 28 verification cases (plan section 3).

Groups (same Top-100 candidate set is later replayed per modality):
  A (8): MD - MR > .02  (multimodal rerank HURTS), prefer router picked MR
  B (4): TD - TR > .02  (text rerank hurts),          prefer router picked TR
  C (8): MR - MD > .02  (multimodal rerank HELPS),    prefer router picked MD
  D (4): TR - TD > .02  (text rerank helps),          prefer router picked TD
  E (4): |MR-MD| < .01 AND |TR-TD| < .01 (tie control)

Forced inclusions from casebook round 1 (qid = row index in query_metrics.csv,
equal to row order of evaluation/predictions/retrievalrouter_l10.xlsx):
  S3a: 4094 4445 2899 2481 6829 8381 ; S3b: 2544 3180 4217 2330 7082
They are assigned to A-D by their actual delta conditions. Remaining slots are
filled with seed=20261016, preferring the router-picked arm of interest and
balancing DATASET coverage. Writes analysis/cases_28.csv.
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(REPO_ROOT, "analysis")
SEED = 20261016
DELTA = 0.02
TIE = 0.01

FORCED = {
    "A": [2899, 4445, 2481, 4094],   # picked MR, MD-MR > .02 (verified in casebook S3a)
    "B": [6829, 8381],               # picked TR, TD-TR > .02 (S3a text-side cases)
    "C": [4217, 3180, 2330, 2544],   # picked MD, MR-MD > .02 (S3b mm cases)
    "D": [7082],                     # picked TD, TR-TD > .02 (S3b text-side case)
}
TARGET_N = {"A": 8, "B": 4, "C": 8, "D": 4, "E": 4}
PREF_PICKED = {"A": "MR", "B": "TR", "C": "MD", "D": "TD"}


def main():
    m = pd.read_csv(os.path.join(AN, "query_metrics.csv"))
    m["qid"] = np.arange(len(m))

    pools = {
        "A": m[m["ndcg_MD"] - m["ndcg_MR"] > DELTA],
        "B": m[m["ndcg_TD"] - m["ndcg_TR"] > DELTA],
        "C": m[m["ndcg_MR"] - m["ndcg_MD"] > DELTA],
        "D": m[m["ndcg_TR"] - m["ndcg_TD"] > DELTA],
        "E": m[(m["ndcg_MR"] - m["ndcg_MD"]).abs() < TIE]
            .pipe(lambda d: d[(d["ndcg_TR"] - d["ndcg_TD"]).abs() < TIE]),
    }

    # verify forced ids actually satisfy their group condition
    for g, ids in FORCED.items():
        pool_ids = set(pools[g]["qid"])
        for i in ids:
            assert i in pool_ids, f"forced qid {i} fails group {g} condition"
    print("pool sizes:", {g: len(d) for g, d in pools.items()})

    rng = np.random.default_rng(SEED)
    chosen = {g: [] for g in "ABCDE"}
    used = set()
    selected_rows = []
    for g, ids in FORCED.items():
        chosen[g] = list(ids)
        used.update(ids)
        for i in ids:
            selected_rows.append(m.loc[m["qid"] == i].iloc[0])

    cols = ["qid", "DATASET", "query", "pred_arm",
            "ndcg_MR", "ndcg_MD", "ndcg_TR", "ndcg_TD", "ndcg_BM",
            "quality_regret", "confidence"]

    def pick_with_diversity(pool, k, prefer_arm):
        """Greedy fill: preferred-picked rows first; draw dataset-diverse samples."""
        out = []
        pref = pool[pool["pred_arm"] == prefer_arm]
        rest = pool[pool["pred_arm"] != prefer_arm]

        def draw(cand):
            cand = cand[~cand["qid"].isin(used)]
            if cand.empty:
                return None
            # weight inversely to how many picks each DATASET already has
            from collections import Counter
            ds_counts = Counter(r["DATASET"] for r in selected_rows)
            w = cand["DATASET"].map(lambda d: 1.0 / (1 + ds_counts.get(d, 0))).to_numpy()
            w = w / w.sum()
            return cand.iloc[rng.choice(len(cand), p=w)]

        for _ in range(k):
            row = draw(pref)
            if row is None:
                row = draw(rest)
            if row is None:
                break
            out.append(int(row["qid"]))
            used.add(int(row["qid"]))
            selected_rows.append(row)
        return out

    for g in "ABCD":
        need = TARGET_N[g] - len(chosen[g])
        ids = pick_with_diversity(pools[g], need, PREF_PICKED[g])
        chosen[g] += ids

    # E: tie controls, no arm preference, dataset-diverse; avoid rows already used
    ids = pick_with_diversity(pools["E"], TARGET_N["E"], "__none__")
    chosen["E"] = ids

    for g in "ABCDE":
        assert len(chosen[g]) == TARGET_N[g], f"group {g} only has {len(chosen[g])}"

    frames = []
    for g in "ABCDE":
        sub = m[m["qid"].isin(chosen[g])][cols].copy()
        sub["group"] = g
        sub["forced"] = sub["qid"].isin([i for ids in FORCED.values() for i in ids])
        sub["d_mm"] = sub["ndcg_MR"] - sub["ndcg_MD"]
        sub["d_text"] = sub["ndcg_TR"] - sub["ndcg_TD"]
        frames.append(sub)
    out = pd.concat(frames).sort_values(["group", "qid"])
    out.to_csv(os.path.join(AN, "cases_28.csv"), index=False)

    print(f"\nWrote {os.path.join(AN, 'cases_28.csv')}")
    print(out.groupby("group").agg(n=("qid", "size"),
                                   datasets=("DATASET", lambda s: dict(s.value_counts())),
                                   picked=("pred_arm", lambda s: dict(s.value_counts()))))
    print("\nqids by group:", {g: chosen[g] for g in "ABCDE"})


if __name__ == "__main__":
    main()
