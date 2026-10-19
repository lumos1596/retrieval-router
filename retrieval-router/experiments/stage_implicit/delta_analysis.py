"""Delta bucket analysis: how M0/M1/M2 perform by rerank gain sign/magnitude."""
import os, json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from train.config import (MODEL_NAME, MAX_LENGTH, ARM_NAMES, NDCG_COLS, DTYPE,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES)
from stage_models import M1Model, M2Model, make_backbone, BM_I, TD_I, TR_I, MD_I, MR_I

OUT = "experiments/stage_implicit"

def predict_util(model, tok, queries, device, bs=32):
    from torch.utils.data import DataLoader
    from datasets import Dataset
    from transformers import DataCollatorWithPadding
    ds = Dataset.from_dict({"text": queries})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
    ds.set_format(type="torch", columns=["input_ids","attention_mask"])
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=DataCollatorWithPadding(tok)):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            out.append(o["util_pred"].cpu())
    return torch.cat(out, 0).numpy()

def load_model(name, device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    backbone = make_backbone(base, base.config.hidden_size).to(device)
    if name == "M1":
        model = M1Model(backbone, base.config.hidden_size).to(device)
    else:
        model = M2Model(backbone, base.config.hidden_size).to(device)
    model.load_state_dict(torch.load(f"{OUT}/{name}_ckpt/model.pt", map_location=device))
    model.eval()
    return model, tok

def main():
    te = pd.read_excel(f"{OUT}/test_split.xlsx")
    ndcg = te[NDCG_COLS].values.astype(float)
    delta_T = ndcg[:, TR_I] - ndcg[:, TD_I]
    delta_M = ndcg[:, MR_I] - ndcg[:, MD_I]
    mr = ndcg[:, MR_I]

    results = {}
    for name in ["M1", "M2"]:
        device = "cuda:0"
        model, tok = load_model(name, device)
        pred = predict_util(model, tok, te["query"].tolist(), device)
        pred_arm = pred.argmax(1)
        chosen = ndcg[np.arange(len(te)), pred_arm]
        del model
        torch.cuda.empty_cache()

        buckets = {
            "Δ_M>0.1 (MM rerank helps)": delta_M > 0.1,
            "Δ_M<-0.1 (MM rerank harms)": delta_M < -0.1,
            "Δ_T>0.1 (text rerank helps)": delta_T > 0.1,
            "Δ_T<-0.1 (text rerank harms)": delta_T < -0.1,
            "|Δ_M|<=0.05 (MM tie)": np.abs(delta_M) <= 0.05,
        }
        for bname, mask in buckets.items():
            if mask.sum() == 0: continue
            r = {
                "model": name, "bucket": bname, "n": int(mask.sum()),
                "nDCG": float(chosen[mask].mean()),
                "fixed_MR": float(mr[mask].mean()),
                "net_vs_MR": float((chosen[mask] - mr[mask]).mean()),
                "oracle": float(ndcg[mask].max(1).mean()),
                "pct_choose_MR": float((pred_arm[mask] == MR_I).mean()),
            }
            results.setdefault(bname, []).append(r)

    rows = [r for v in results.values() for r in v]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/stage_delta_analysis.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
