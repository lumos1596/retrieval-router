"""Test ColNomic LoRA over the ORIGINAL base (Qwen2.5-VL-7B-Instruct, as the
adapter config said before 2025-04-15 and as the release results.json commit
records) against the current default colqwen2.5-7B-base."""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from replay_lib import DATA_ROOT, EMB_ROOT, RUNS_ROOT, maxsim_scores, ndcg_at_k, order_by_scores

ADAPTER = "nomic-ai/colnomic-embed-multimodal-7b"
OLD_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    d = json.load(open(os.path.join(RUNS_ROOT, f"{args.qid}_MR.json")))
    ds = d["dataset"]
    cands = [x["filename"] for x in sorted(d["candidate_table"], key=lambda r: r["dense_rank"])]
    pages = pd.read_parquet(os.path.join(DATA_ROOT, ds, "pages.parquet"))
    pidx = {fn: i for i, fn in enumerate(pages["filename"])}
    gt = set(d["gt_rerank_ranks"])

    device = torch.device(f"cuda:{args.gpu}")
    from peft import PeftModel
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    base = ColQwen2_5.from_pretrained(OLD_BASE, torch_dtype=torch.bfloat16,
                                      device_map=device).eval()
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    processor = ColQwen2_5_Processor.from_pretrained(ADAPTER)

    def doc_v(fn):
        img = Image.open(io.BytesIO(pages["img_bytes"].iloc[pidx[fn]])).convert("RGB")
        with torch.inference_mode():
            return model(**processor.process_images([img]).to(device)).float().cpu().numpy()[0]

    with torch.inference_mode():
        qv = model(**processor.process_queries([d["query"]]).to(device)).float().cpu().numpy()[0]
    dv = [doc_v(c) for c in cands]
    sc = maxsim_scores(qv, dv)
    order = order_by_scores(np.asarray(sc), cands)
    ranked = [cands[i] for i in order]
    print(f"qid {args.qid} OLD-BASE rerank nDCG@5 = {ndcg_at_k(ranked, gt):.4f} "
          f"(official {d['official_rerank_ndcg5']}, current-base {d['rerank_ndcg5']:.4f})")
    for r, i in enumerate(order[:7], 1):
        fn = cands[i]
        print(f" {r:2d} {sc[i]:.3f} {'GT' if fn in gt else '  '} {fn[:50]}")


if __name__ == "__main__":
    main()
