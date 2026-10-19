"""M2 ablation: parallel heads (no stage1->stage2 dependency). Same supervision."""
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
    LAMBDA, TEMPERATURE, DTYPE, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES)
from stage_models import make_backbone, compute_reward
from stage_ablation import M2ParallelModel

OUT = "experiments/stage_implicit"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)

    tr = pd.read_excel(f"{OUT}/train_split.xlsx")
    va = pd.read_excel(f"{OUT}/val_split.xlsx")
    te = pd.read_excel(f"{OUT}/test_split.xlsx")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    ndcg_tr = tr[NDCG_COLS].values.astype(np.float32)
    lat_tr = tr[LATENCY_COLS].values.astype(np.float32)
    reward_tr = compute_reward(ndcg_tr, lat_tr)
    lat_va = va[LATENCY_COLS].values.astype(np.float32)

    base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    backbone = make_backbone(base, base.config.hidden_size).to(device)
    model = M2ParallelModel(backbone, base.config.hidden_size, alpha=args.alpha, beta=args.beta).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    ds = Dataset.from_dict({"text": tr["query"].tolist()})
    ds = ds.map(lambda x: tok(x["text"], truncation=True, padding=False, max_length=MAX_LENGTH), batched=True)
    ds = ds.add_column("ndcg", ndcg_tr.tolist()).add_column("reward", reward_tr.tolist()).add_column("latency", lat_tr.tolist())
    ds.set_format(type="torch", columns=["input_ids","attention_mask","ndcg","reward","latency"])
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

    best = -1; ckpt = f"{OUT}/M2par_ckpt"
    for ep in range(2):
        model.train(); losses=[]; opt.zero_grad()
        for i, b in enumerate(tqdm(loader, desc=f"M2par ep{ep}")):
            o = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = model.loss(o["base_pred"], o["delta_pred"], o["util_pred"],
                              b["reward"].to(device), b["ndcg"].to(device), b["latency"].to(device)) / GRAD_ACCUM
            loss.backward()
            if (i+1) % GRAD_ACCUM == 0:
                opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item() * GRAD_ACCUM)
        vp = predict(va); vr = evaluate(vp, va)
        print(f"  M2par ep{ep}: loss={np.mean(losses):.4f} val_nDCG={vr['nDCG@5']:.4f}")
        if vr["nDCG@5"] > best:
            best = vr["nDCG@5"]
            os.makedirs(ckpt, exist_ok=True)
            model.backbone.base_model.save_pretrained(f"{ckpt}/lora_adapter")
            torch.save(model.state_dict(), f"{ckpt}/model.pt")
            tok.save_pretrained(f"{ckpt}/lora_adapter")

    base2 = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(device)
    base2 = get_peft_model(base2, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    backbone2 = make_backbone(base2, base2.config.hidden_size).to(device)
    model = M2ParallelModel(backbone2, base2.config.hidden_size, alpha=args.alpha, beta=args.beta).to(device)
    model.load_state_dict(torch.load(f"{ckpt}/model.pt", map_location=device))
    model.eval()
    tp = predict(te); overall = evaluate(tp, te)
    per_ds = {ds: evaluate(tp[te["DATASET"].values==ds], te[te["DATASET"]==ds]) for ds in te["DATASET"].unique()}
    print(f"\n=== M2-parallel TEST ===")
    print(json.dumps(overall, indent=2))
    with open(f"{OUT}/stage1_M2parallel_results.json","w") as f:
        json.dump({"overall": overall, "per_dataset": per_ds}, f, indent=2)

if __name__ == "__main__":
    main()
