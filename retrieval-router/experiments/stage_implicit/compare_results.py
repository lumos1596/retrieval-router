"""Compare M0/M1/M2/M2-parallel results and produce stage_training_comparison.csv."""
import os, json
import numpy as np
import pandas as pd

OUT = "experiments/stage_implicit"

def load(name):
    p = f"{OUT}/{name}_results.json"
    if os.path.exists(p):
        return json.load(open(p))
    return None

models = {
    "M0": load("stage0_M0"),
    "M1": load("stage1_M1"),
    "M2": load("stage1_M2"),
}

rows = []
for name, res in models.items():
    if res is None:
        print(f"{name}: not ready")
        continue
    o = res["overall"]
    rows.append({"model": name, "nDCG@5": o["nDCG@5"], "MRR": o.get("MRR",""),
                 "oracle": o["oracle"], "fixed_MR": o["fixed_MR"],
                 "regret": o["regret"], "net_vs_MR": o["net_vs_MR"]})
    for ds, r in res["per_dataset"].items():
        rows.append({"model": name, "dataset": ds, "nDCG@5": r["nDCG@5"],
                     "regret": r["regret"], "net_vs_MR": r["net_vs_MR"]})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/stage_training_comparison.csv", index=False)
print(df.to_string(index=False))
