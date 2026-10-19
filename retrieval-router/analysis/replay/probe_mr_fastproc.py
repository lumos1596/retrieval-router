"""Probe: ColNomic with the FAST image processor (use_fast=True) to test
whether preprocessing explains the official-rerank divergence."""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from replay_lib import DATA_ROOT, RUNS_ROOT, maxsim_scores, ndcg_at_k, order_by_scores

MR_MODEL = "nomic-ai/colnomic-embed-multimodal-7b"


def run(qid, gpu, fast):
    d = json.load(open(os.path.join(RUNS_ROOT, f"{qid}_MR.json")))
    ds = d["dataset"]
    cands = [x["filename"] for x in sorted(d["candidate_table"], key=lambda r: r["dense_rank"])]
    pages = pd.read_parquet(os.path.join(DATA_ROOT, ds, "pages.parquet"))
    pidx = {fn: i for i, fn in enumerate(pages["filename"])}
    gt = set(d["gt_rerank_ranks"])
    device = torch.device(f"cuda:{gpu}")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(MR_MODEL, torch_dtype=torch.bfloat16,
                                       device_map=device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MR_MODEL, use_fast=fast)

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
    print(f"fast={fast}: nDCG={ndcg_at_k(ranked, gt):.4f} official={d['official_rerank_ndcg5']:.4f}")
    for r, i in enumerate(order[:6], 1):
        fn = cands[i]
        print(f" {r:2d} {sc[i]:.3f} {'GT' if fn in gt else '  '} {fn[:46]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    run(a.qid, a.gpu, a.fast)
