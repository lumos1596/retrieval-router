"""Training objective comparison: B0 KL vs B1 Utility Regression vs B2 Utility+Regret."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from tqdm import tqdm
from train.config import (MODEL_NAME, MAX_LENGTH, ARM_NAMES, NDCG_COLS, device,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES, DTYPE)
from train.train import main as train_main
from train.model import QwenForSoftClassification
from train.objectives import get_objective

DATA_DIR = "experiments/xdomain/data"
RUN_TAG = "xc_fin3_l10"
TRAIN_PATH = os.path.join(DATA_DIR, f"train_{RUN_TAG}.xlsx")
VAL_PATH = os.path.join(DATA_DIR, f"val_{RUN_TAG}.xlsx")
SOURCE_TEST_PATH = os.path.join(DATA_DIR, f"source_test_{RUN_TAG}.xlsx")
OUT_DIR = "experiments/train_loss"
os.makedirs(OUT_DIR, exist_ok=True)
TARGET_SETS = {
    "source_test": SOURCE_TEST_PATH,
    "tatdqa": os.path.join(DATA_DIR, f"target_tatdqa_{RUN_TAG}.xlsx"),
    "wiki-ss": os.path.join(DATA_DIR, f"target_wiki-ss_{RUN_TAG}.xlsx"),
    "arxivqa": os.path.join(DATA_DIR, f"target_arxivqa_{RUN_TAG}.xlsx"),
}
B0_CKPT = "experiments/xdomain/results_l00/checkpoint/lora_adapter"
SHORT = {"MULTIMODAL_RERANK":"MR","MULTIMODAL-SINGLE":"MD","TEXT_RERANK":"TR","TEXT-SINGLE":"TD","BM25":"BM"}

def load_trained_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    base = PeftModel.from_pretrained(base, adapter_path)
    model = QwenForSoftClassification(base_model=base, num_labels=len(ARM_NAMES),
        hidden_size=base.config.hidden_size, dropout=0.1).to(device)
    clf_path = os.path.join(adapter_path, "classifier.pt")
    if os.path.exists(clf_path):
        model.classifier.load_state_dict(torch.load(clf_path, map_location=device))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def predict(model, tokenizer, df, activation="softmax", batch_size=32):
    ds = Dataset.from_dict({"text": df["query"].tolist()})
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH),
                batched=True, desc="Tokenizing")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    all_logits = []
    for batch in tqdm(loader, desc="Inference"):
        out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        all_logits.append(out["logits"].cpu())
    logits = torch.cat(all_logits, dim=0)
    if activation == "sigmoid":
        return torch.sigmoid(logits).numpy()
    return torch.softmax(logits, dim=-1).numpy()

def evaluate_set(model, tokenizer, df, activation):
    preds = predict(model, tokenizer, df, activation)
    ndcg_mat = df[NDCG_COLS].values.astype(float)
    pred_label = preds.argmax(axis=1)
    chosen_ndcg = ndcg_mat[np.arange(len(df)), pred_label]
    oracle_ndcg = ndcg_mat.max(axis=1)
    regret = oracle_ndcg - chosen_ndcg
    catastrophic = (oracle_ndcg >= 0.75) & (chosen_ndcg == 0)
    mr_ndcg = ndcg_mat[:, 0]
    res = pd.DataFrame({"query": df["query"].values})
    for i, a in enumerate(ARM_NAMES):
        res[f"pred_{SHORT[a]}"] = preds[:, i]
    res["pred_arm"] = [ARM_NAMES[i] for i in pred_label]
    res["pred_arm_short"] = [SHORT[ARM_NAMES[i]] for i in pred_label]
    res["chosen_ndcg"] = chosen_ndcg
    res["oracle_ndcg"] = oracle_ndcg
    res["regret"] = regret
    res["catastrophic"] = catastrophic
    res["mr_ndcg"] = mr_ndcg
    metrics = {"n": len(df), "ndcg": float(chosen_ndcg.mean()), "regret": float(regret.mean()),
        "catastrophic_rate": float(catastrophic.mean()), "mr_ndcg": float(mr_ndcg.mean()),
        "gain_vs_mr": float(chosen_ndcg.mean() - mr_ndcg.mean())}
    counts = pd.Series(pred_label).value_counts().to_dict()
    for i, a in enumerate(ARM_NAMES):
        metrics[f"sel_{SHORT[a]}"] = counts.get(i, 0) / len(df)
    return res, metrics

def fixed_mr_eval(df):
    ndcg_mat = df[NDCG_COLS].values.astype(float)
    chosen = ndcg_mat[:, 0]
    oracle = ndcg_mat.max(axis=1)
    regret = oracle - chosen
    catastrophic = (oracle >= 0.75) & (chosen == 0)
    return {"n": len(df), "ndcg": float(chosen.mean()), "regret": float(regret.mean()),
        "catastrophic_rate": float(catastrophic.mean()), "mr_ndcg": float(chosen.mean()), "gain_vs_mr": 0.0}

def main():
    print(f"Device: {device}")
    models = {}
    for obj_name, tag in [("utility_regression", "B1"), ("utility_regret", "B2")]:
        print(f"\n{'='*80}\nTraining {tag} ({obj_name})\n{'='*80}")
        os.environ["OBJECTIVE"] = obj_name
        os.environ["LAMBDA"] = "0"
        ckpt_dir = os.path.join(OUT_DIR, f"checkpoint_{tag}")
        objective = get_objective(obj_name)
        model, tokenizer = train_main(objective=objective,
            paths=(TRAIN_PATH, VAL_PATH, SOURCE_TEST_PATH), output_dir=ckpt_dir)
        model.eval()
        models[tag] = (model, tokenizer, getattr(objective, "inference_activation", "softmax"))

    print(f"\n{'='*80}\nLoading B0 (KL, lambda=0)\n{'='*80}")
    b0_model, b0_tokenizer = load_trained_model(B0_CKPT)
    models["B0"] = (b0_model, b0_tokenizer, "softmax")

    dfs = {name: pd.read_excel(path) for name, path in TARGET_SETS.items()}
    all_metrics = []
    all_per_query = {}
    for set_name, df in dfs.items():
        print(f"\nEvaluating on {set_name} (n={len(df)})")
        fm = fixed_mr_eval(df)
        fm.update({"dataset": set_name, "model": "FixedMR", "seed": 0})
        all_metrics.append(fm)
        for tag, (model, tok, act) in models.items():
            res, m = evaluate_set(model, tok, df, act)
            m.update({"dataset": set_name, "model": tag, "seed": 0})
            all_metrics.append(m)
            all_per_query[(set_name, tag)] = res

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "train_loss_comparison.csv"), index=False)
    print("\n=== Main results ===")
    print(metrics_df[["dataset","model","ndcg","regret","catastrophic_rate","gain_vs_mr"]].to_string(index=False))

    arm_rows = []
    for m in all_metrics:
        row = {"dataset": m["dataset"], "model": m["model"]}
        for a in ["MR","MD","TR","TD","BM"]:
            row[a] = m.get(f"sel_{a}", 0)
        arm_rows.append(row)
    pd.DataFrame(arm_rows).to_csv(os.path.join(OUT_DIR, "train_loss_arm_selection.csv"), index=False)

    margin_rows = []
    for set_name in ["tatdqa","wiki-ss","arxivqa"]:
        df = dfs[set_name]
        mr = df[NDCG_COLS[0]].values
        md = df[NDCG_COLS[1]].values
        diff = mr - md
        for tag in ["B0","B1","B2"]:
            res = all_per_query[(set_name, tag)]
            for lo, hi, label in [(-10,-0.2,"MD_better"),(-0.05,0.05,"tie"),(0.2,10,"MR_better")]:
                mask = (diff >= lo) & (diff < hi)
                if mask.sum() == 0: continue
                sub = res[mask]
                if label == "MR_better": wrong = (sub["pred_arm_short"] == "MD").mean()
                elif label == "MD_better": wrong = (sub["pred_arm_short"] == "MR").mean()
                else: wrong = np.nan
                margin_rows.append({"dataset": set_name, "model": tag, "bucket": label,
                    "n": int(mask.sum()), "avg_regret": float(sub["regret"].mean()),
                    "wrong_choice_rate": float(wrong) if not np.isnan(wrong) else np.nan})
    pd.DataFrame(margin_rows).to_csv(os.path.join(OUT_DIR, "train_loss_margin_analysis.csv"), index=False)

    print("\nGenerating badcases...")
    set_name = "wiki-ss"
    b0 = all_per_query[(set_name, "B0")]
    b1 = all_per_query[(set_name, "B1")]
    b2 = all_per_query[(set_name, "B2")]
    df = dfs[set_name]
    diff1 = b1["regret"].values - b0["regret"].values
    improved_idx = np.argsort(diff1)[:10]
    worsened_idx = np.argsort(-diff1)[:10]
    def arm_vec(i):
        return {SHORT[a]: round(float(df[NDCG_COLS[j]].iloc[i]), 3) for j, a in enumerate(ARM_NAMES)}
    lines = ["# Training objective badcases (Wiki-SS)\n"]
    lines.append("## B1 improved vs B0 (top 10 by regret drop)\n")
    for rank, i in enumerate(improved_idx, 1):
        lines.append(f"**{rank}.** {df['query'].iloc[i][:140]}")
        lines.append(f"- B0={b0['pred_arm_short'].iloc[i]}(r={b0['regret'].iloc[i]:.3f})  "
            f"B1={b1['pred_arm_short'].iloc[i]}(r={b1['regret'].iloc[i]:.3f})  "
            f"B2={b2['pred_arm_short'].iloc[i]}(r={b2['regret'].iloc[i]:.3f})")
        lines.append(f"- nDCG: {arm_vec(i)}")
        lines.append("")
    lines.append("## B1 worsened vs B0 (top 10 by regret increase)\n")
    for rank, i in enumerate(worsened_idx, 1):
        lines.append(f"**{rank}.** {df['query'].iloc[i][:140]}")
        lines.append(f"- B0={b0['pred_arm_short'].iloc[i]}(r={b0['regret'].iloc[i]:.3f})  "
            f"B1={b1['pred_arm_short'].iloc[i]}(r={b1['regret'].iloc[i]:.3f})  "
            f"B2={b2['pred_arm_short'].iloc[i]}(r={b2['regret'].iloc[i]:.3f})")
        lines.append(f"- nDCG: {arm_vec(i)}")
        lines.append("")
    with open(os.path.join(OUT_DIR, "train_loss_badcases.md"), "w") as f:
        f.write("\n".join(lines))

    lines = ["# Training objective comparison report\n"]
    lines.append("## Main results\n")
    lines.append("| Dataset | Model | nDCG | Regret | Catastrophic | vs FixedMR |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in metrics_df.iterrows():
        lines.append(f"| {r['dataset']} | {r['model']} | {r['ndcg']:.4f} | {r['regret']:.4f} | "
            f"{r['catastrophic_rate']*100:.1f}% | {r['gain_vs_mr']:+.4f} |")
    lines.append("")
    lines.append("## Arm selection distribution\n")
    lines.append("| Dataset | Model | MR | MD | TR | TD | BM |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in arm_rows:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['MR']*100:.1f}% | {r['MD']*100:.1f}% | "
            f"{r['TR']*100:.1f}% | {r['TD']*100:.1f}% | {r['BM']*100:.1f}% |")
    lines.append("")
    lines.append("## Conclusion\n")
    tgt = metrics_df[metrics_df["dataset"].isin(["tatdqa","wiki-ss","arxivqa"])]
    macro = tgt.groupby("model")["ndcg"].mean()
    lines.append(f"- 3-target macro nDCG: B0={macro.get('B0',0):.4f}, B1={macro.get('B1',0):.4f}, "
        f"B2={macro.get('B2',0):.4f}, FixedMR={macro.get('FixedMR',0):.4f}")
    lines.append("- See badcases and margin_analysis for details.")
    with open(os.path.join(OUT_DIR, "train_loss_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nAll deliverables saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
