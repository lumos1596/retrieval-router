"""Train M3 (M2 + process supervision), M3-flat, M3-shuffle on 11 datasets.

Uses full training set for utility loss + 960 labeled queries for process supervision.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from tqdm import tqdm
from train.config import (MODEL_NAME, MAX_LENGTH, ARM_NAMES, NDCG_COLS, LATENCY_COLS,
    DTYPE, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES)
from stage_models import make_backbone, compute_reward
from stage_m3 import M3Model, M3FlatModel

OUT = "experiments/stage_implicit"
PROC_LABELS = ["recall_hit_M","recall_hit_T","rerank_rescue_M","rerank_harm_M",
               "rerank_rescue_T","rerank_harm_T"]

def load_process_labels():
    df = pd.read_csv(f"{OUT}/retrieval_process_labels.csv")
    labels = {}
    for _, r in df.iterrows():
        labels[r["query"]] = [float(r[c]) for c in PROC_LABELS]
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["M3","M3-flat","M3-shuffle"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.5)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    tr = pd.read_excel(f"{OUT}/train_split.xlsx")
    va = pd.read_excel(f"{OUT}/val_split.xlsx")
    te = pd.read_excel(f"{OUT}/test_split.xlsx")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Process labels
    proc_labels = load_process_labels()
    if args.model == "M3-shuffle":
        vals = list(proc_labels.values())
        np.random.RandomState(42).shuffle(vals)
        proc_labels = {k: v for k, v in zip(proc_labels.keys(), vals)}

    # Build training labels: proc labels where available, else zeros + mask
    ndcg_tr = tr[NDCG_COLS].values.astype(np.float32)
    lat_tr = tr[LATENCY_COLS].values.astype(np.float32)
    reward_tr = compute_reward(ndcg_tr, lat_tr)
    proc_tr = np.zeros((len(tr), 6), dtype=np.float32)
    proc_mask = np.zeros(len(tr), dtype=np.float32)
    for i, q in enumerate(tr["query"]):
        if q in proc_labels:
            proc_tr[i] = proc_labels[q]
            proc_mask[i] = 1.0
    print(f"Labeled queries in train: {proc_mask.sum()}/{len(tr)}")

    lat_va = va[LATENCY_COLS].values.astype(np.float32)

    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    backbone = make_backbone(base, base.config.hidden_size).to(device)
    if args.model in ("M3", "M3-shuffle"):
        model = M3Model(backbone, base.config.hidden_size, gamma=args.gamma).to(device)
    else:
        model = M3FlatModel(backbone, base.config.hidden_size, gamma=args.gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    ds = Dataset.from_dict({"text": tr["query"].tolist()})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
    ds = ds.add_column("ndcg", ndcg_tr.tolist()).add_column("reward", reward_tr.tolist())
    ds = ds.add_column("proc", proc_tr.tolist()).add_column("proc_mask", proc_mask.tolist())
    ds.set_format(type="torch", columns=["input_ids","attention_mask","ndcg","reward","proc","proc_mask"])
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=DataCollatorWithPadding(tok))
    GRAD_ACCUM = 2
    total_steps = len(loader) * 2 // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    @torch.no_grad()
    def predict(df):
        d = Dataset.from_dict({"text": df["query"].tolist()})
        d = d.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
        d.set_format(type="torch", columns=["input_ids","attention_mask"])
        out = []
        for b in DataLoader(d, batch_size=32, shuffle=False, collate_fn=DataCollatorWithPadding(tok)):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            out.append(o["util_pred"].cpu())
        return torch.cat(out, 0).numpy()

    def evaluate(pred, df):
        ndcg = df[NDCG_COLS].values.astype(float)
        chosen = ndcg[np.arange(len(df)), pred.argmax(1)]
        return {"nDCG@5": float(chosen.mean()), "oracle": float(ndcg.max(1).mean()),
                "fixed_MR": float(ndcg[:,0].mean()), "regret": float((ndcg.max(1)-chosen).mean()),
                "net_vs_MR": float((chosen-ndcg[:,0]).mean())}

    best = -1; ckpt = f"{OUT}/{args.model}_ckpt"
    for ep in range(2):
        model.train(); losses=[]; opt.zero_grad()
        for i, b in enumerate(tqdm(loader, desc=f"{args.model} ep{ep}")):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            ndcg_b = b["ndcg"].to(device)
            reward_b = b["reward"].to(device)
            proc_b = b["proc"].to(device)
            mask_b = b["proc_mask"].to(device)
            loss = model.loss(o["base_pred"], o["delta_pred"], o["util_pred"],
                              o["recall_logits"], o["rerank_logits"], reward_b, ndcg_b, proc_b) / GRAD_ACCUM
            loss.backward()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item() * GRAD_ACCUM)
        vp = predict(va); vr = evaluate(vp, va)
        print(f"  {args.model} ep{ep}: loss={np.mean(losses):.4f} val_nDCG={vr['nDCG@5']:.4f}")
        if vr["nDCG@5"] > best:
            best = vr["nDCG@5"]
            os.makedirs(ckpt, exist_ok=True)
            model.backbone.base_model.save_pretrained(f"{ckpt}/lora_adapter")
            torch.save(model.state_dict(), f"{ckpt}/model.pt")
            tok.save_pretrained(f"{ckpt}/lora_adapter")

    # Test
    base2 = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base2 = get_peft_model(base2, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    backbone2 = make_backbone(base2, base2.config.hidden_size).to(device)
    if args.model in ("M3", "M3-shuffle"):
        model = M3Model(backbone2, base2.config.hidden_size, gamma=args.gamma).to(device)
    else:
        model = M3FlatModel(backbone2, base2.config.hidden_size, gamma=args.gamma).to(device)
    model.load_state_dict(torch.load(f"{ckpt}/model.pt", map_location=device))
    model.eval()
    tp = predict(te); overall = evaluate(tp, te)
    per_ds = {ds: evaluate(tp[te["DATASET"].values==ds], te[te["DATASET"]==ds]) for ds in te["DATASET"].unique()}
    print(f"\n=== {args.model} TEST ===")
    print(json.dumps(overall, indent=2))
    with open(f"{OUT}/stage2_{args.model}_results.json","w") as f:
        json.dump({"overall": overall, "per_dataset": per_ds, "gamma": args.gamma}, f, indent=2)

if __name__ == "__main__":
    main()
