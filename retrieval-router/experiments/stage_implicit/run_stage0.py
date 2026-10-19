"""Stage 0: Reproduce paper baseline (M0 = original soft-label KL) on 11 datasets.

- Per-dataset 80/10/10 split (same as train/prepare_data.py)
- Qwen3-0.6B + LoRA(r=16), lambda=0.1, soft-label KL
- Evaluate pooled nDCG@5 on test set
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from tqdm import tqdm
from train.config import (MODEL_NAME, MAX_LENGTH, ARM_NAMES, NDCG_COLS, LATENCY_COLS,
    REWARD_COLS, LAMBDA, TEMPERATURE, device, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    TARGET_MODULES, DTYPE, RANDOM_SEED)
from train.model import QwenForSoftClassification
from train.objectives import get_objective

OUT = "experiments/stage_implicit"
os.makedirs(OUT, exist_ok=True)

def make_split():
    df = pd.read_excel("train/data/updated_dataset.xlsx").dropna()
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "query"}, inplace=True)
    train_dfs, val_dfs, test_dfs = [], [], []
    for ds in df["DATASET"].unique():
        sub = df[df["DATASET"] == ds]
        n = len(sub)
        n_test = int(n * 0.1)
        n_val = int(n * 0.1)
        test = sub.sample(n=n_test, random_state=RANDOM_SEED)
        rem = sub.drop(test.index)
        val = rem.sample(n=n_val, random_state=RANDOM_SEED)
        train = rem.drop(val.index)
        train_dfs.append(train); val_dfs.append(val); test_dfs.append(test)
    tr = pd.concat(train_dfs, ignore_index=True)
    va = pd.concat(val_dfs, ignore_index=True)
    te = pd.concat(test_dfs, ignore_index=True)
    print(f"Split: train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te

def build_model(num_labels):
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION))
    model = QwenForSoftClassification(base_model=base, num_labels=num_labels,
        hidden_size=base.config.hidden_size, dropout=0.1).to(device)
    return model

@torch.no_grad()
def predict(model, tok, df, bs=32):
    ds = Dataset.from_dict({"text": df["query"].tolist()})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH),
                batched=True, desc="tok")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=DataCollatorWithPadding(tok))
    all_logits = []
    for batch in loader:
        out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        all_logits.append(out["logits"].cpu())
    return torch.softmax(torch.cat(all_logits, 0), dim=-1).numpy()

def evaluate(preds, df):
    ndcg = df[NDCG_COLS].values.astype(float)
    pred_arm = preds.argmax(1)
    chosen = ndcg[np.arange(len(df)), pred_arm]
    oracle = ndcg.max(1)
    mr = ndcg[:, 0]
    return {
        "nDCG@5": float(chosen.mean()),
        "oracle": float(oracle.mean()),
        "fixed_MR": float(mr.mean()),
        "regret": float((oracle - chosen).mean()),
        "net_vs_MR": float((chosen - mr).mean()),
    }

def main():
    tr, va, te = make_split()
    tr.to_excel(f"{OUT}/train_split.xlsx", index=False)
    va.to_excel(f"{OUT}/val_split.xlsx", index=False)
    te.to_excel(f"{OUT}/test_split.xlsx", index=False)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    obj = get_objective("retrievalrouter")
    tr_lab = obj.add_targets(tr)
    va_lab = obj.add_targets(va)

    model = build_model(len(ARM_NAMES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Tokenize
    def tok_df(df):
        ds = Dataset.from_dict({"text": df["query"].tolist()})
        ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
        ds = ds.add_column("labels", obj.targets(df))
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        return ds
    train_ds = tok_df(tr_lab)
    collator = DataCollatorWithPadding(tok)
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collator)

    best_val = -1
    for epoch in range(2):
        model.train()
        losses = []
        for batch in tqdm(loader, desc=f"epoch {epoch}"):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            loss = obj.loss(out["logits"], batch["labels"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # val
        model.eval()
        vp = predict(model, tok, va_lab)
        vr = evaluate(vp, va_lab)
        print(f"epoch {epoch}: train_loss={np.mean(losses):.4f} val_nDCG={vr['nDCG@5']:.4f}")
        if vr["nDCG@5"] > best_val:
            best_val = vr["nDCG@5"]
            os.makedirs(f"{OUT}/M0_ckpt", exist_ok=True)
            model.base_model.save_pretrained(f"{OUT}/M0_ckpt/lora_adapter")
            torch.save(model.classifier.state_dict(), f"{OUT}/M0_ckpt/lora_adapter/classifier.pt")
            tok.save_pretrained(f"{OUT}/M0_ckpt/lora_adapter")

    # Load best and test
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION))
    base = PeftModel.from_pretrained(base, f"{OUT}/M0_ckpt/lora_adapter")
    model = QwenForSoftClassification(base_model=base, num_labels=len(ARM_NAMES),
        hidden_size=base.config.hidden_size, dropout=0.1).to(device)
    model.classifier.load_state_dict(torch.load(f"{OUT}/M0_ckpt/lora_adapter/classifier.pt", map_location=device))
    model.eval()

    te_lab = obj.add_targets(te)
    tp = predict(model, tok, te_lab)
    tr_overall = evaluate(tp, te_lab)
    print(f"\n=== TEST (pooled 11 datasets) ===")
    print(json.dumps(tr_overall, indent=2))

    # Per-dataset
    per_ds = {}
    for ds in te["DATASET"].unique():
        mask = te["DATASET"] == ds
        per_ds[ds] = evaluate(tp[mask.values], te[mask])
        print(f"  {ds}: nDCG={per_ds[ds]['nDCG@5']:.4f}")

    with open(f"{OUT}/stage0_M0_results.json", "w") as f:
        json.dump({"overall": tr_overall, "per_dataset": per_ds}, f, indent=2)
    print(f"\nSaved to {OUT}/stage0_M0_results.json")

if __name__ == "__main__":
    main()
