"""Stage 1: Train M1 (KL+utility regression) and M2 (stage-latent) on 11 datasets.

Usage: python run_stage1.py --model M1|M2 --gpu <gpu_id>
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from tqdm import tqdm
from train.config import (MODEL_NAME, MAX_LENGTH, ARM_NAMES, NDCG_COLS, LATENCY_COLS,
    LAMBDA, TEMPERATURE, DTYPE, RANDOM_SEED, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES)
from stage_models import (M1Model, M2Model, make_backbone, compute_reward,
    BM_I, TD_I, TR_I, MD_I, MR_I)

OUT = "experiments/stage_implicit"

def load_split():
    tr = pd.read_excel(f"{OUT}/train_split.xlsx")
    va = pd.read_excel(f"{OUT}/val_split.xlsx")
    te = pd.read_excel(f"{OUT}/test_split.xlsx")
    return tr, va, te

def build_backbone(device):
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION))
    hidden = base.config.hidden_size
    return make_backbone(base, hidden).to(device), hidden

@torch.no_grad()
def predict(model, tok, df, device, bs=32):
    ds = Dataset.from_dict({"text": df["query"].tolist()})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=DataCollatorWithPadding(tok))
    all_util = []
    for batch in loader:
        out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        all_util.append(out["util_pred"].cpu())
    return torch.cat(all_util, 0).numpy()

def evaluate(util_pred, df, latencies):
    ndcg = df[NDCG_COLS].values.astype(float)
    pred_arm = util_pred.argmax(1)
    chosen = ndcg[np.arange(len(df)), pred_arm]
    oracle = ndcg.max(1)
    mr = ndcg[:, MR_I]
    return {"nDCG@5": float(chosen.mean()), "oracle": float(oracle.mean()),
            "fixed_MR": float(mr.mean()), "regret": float((oracle-chosen).mean()),
            "net_vs_MR": float((chosen-mr).mean())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["M1","M2"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    print(f"Training {args.model} on {device}")

    tr, va, te = load_split()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Compute rewards and labels
    ndcg_tr = tr[NDCG_COLS].values.astype(np.float32)
    lat_tr = tr[LATENCY_COLS].values.astype(np.float32)
    reward_tr = compute_reward(ndcg_tr, lat_tr)
    ndcg_va = va[NDCG_COLS].values.astype(np.float32)
    lat_va = va[LATENCY_COLS].values.astype(np.float32)

    backbone, hidden = build_backbone(device)
    if args.model == "M1":
        model = M1Model(backbone, hidden, alpha=args.alpha).to(device)
    else:
        model = M2Model(backbone, hidden, alpha=args.alpha, beta=args.beta).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def make_loader(df, ndcg, reward, lat):
        ds = Dataset.from_dict({"text": df["query"].tolist()})
        ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
        ds = ds.add_column("ndcg", ndcg.tolist())
        ds = ds.add_column("reward", reward.tolist())
        ds = ds.add_column("latency", lat.tolist())
        ds.set_format(type="torch", columns=["input_ids","attention_mask","ndcg","reward","latency"])
        return DataLoader(ds, batch_size=16, shuffle=True, collate_fn=DataCollatorWithPadding(tok))

    train_loader = make_loader(tr, ndcg_tr, reward_tr, lat_tr)
    GRAD_ACCUM = 2
    total_steps = len(train_loader) * 2 // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_val = -1
    ckpt_dir = f"{OUT}/{args.model}_ckpt"
    for epoch in range(2):
        model.train()
        losses = []
        opt.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"{args.model} ep{epoch}")):
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            ndcg_b = batch["ndcg"].to(device)
            reward_b = batch["reward"].to(device)
            if args.model == "M1":
                loss = model.loss(out["util_pred"], reward_b, ndcg_b) / GRAD_ACCUM
            else:
                loss = model.loss(out["base_pred"], out["delta_pred"], out["util_pred"],
                                  reward_b, ndcg_b, batch["latency"].to(device)) / GRAD_ACCUM
            loss.backward()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item() * GRAD_ACCUM)
        model.eval()
        vp = predict(model, tok, va, device)
        vr = evaluate(vp, va, lat_va)
        print(f"  {args.model} ep{epoch}: loss={np.mean(losses):.4f} val_nDCG={vr['nDCG@5']:.4f}")
        if vr["nDCG@5"] > best_val:
            best_val = vr["nDCG@5"]
            os.makedirs(ckpt_dir, exist_ok=True)
            model.backbone.base_model.save_pretrained(f"{ckpt_dir}/lora_adapter")
            torch.save(model.state_dict(), f"{ckpt_dir}/model.pt")
            tok.save_pretrained(f"{ckpt_dir}/lora_adapter")

    # Test
    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION))
    backbone2 = make_backbone(base, base.config.hidden_size).to(device)
    if args.model == "M1":
        model = M1Model(backbone2, base.config.hidden_size, alpha=args.alpha).to(device)
    else:
        model = M2Model(backbone2, base.config.hidden_size, alpha=args.alpha, beta=args.beta).to(device)
    model.load_state_dict(torch.load(f"{ckpt_dir}/model.pt", map_location=device))
    model.eval()

    lat_te = te[LATENCY_COLS].values.astype(np.float32)
    tp = predict(model, tok, te, device)
    overall = evaluate(tp, te, lat_te)
    per_ds = {}
    for ds in te["DATASET"].unique():
        mask = te["DATASET"].values == ds
        per_ds[ds] = evaluate(tp[mask], te[mask], lat_te[mask])
    print(f"\n=== {args.model} TEST ===")
    print(json.dumps(overall, indent=2))
    for ds, r in per_ds.items():
        print(f"  {ds}: {r['nDCG@5']:.4f}")
    with open(f"{OUT}/stage1_{args.model}_results.json", "w") as f:
        json.dump({"overall": overall, "per_dataset": per_ds,
                   "alpha": args.alpha, "beta": args.beta}, f, indent=2)

if __name__ == "__main__":
    main()
