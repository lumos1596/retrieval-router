"""Encode the union of dense Top-100 candidate pages with late-interaction models.

  MR : nomic-ai/colnomic-embed-multimodal-7b (ColQwen2.5), content = page image
  TR : lightonai/GTE-ModernColBERT-v1 via pylate (document_length=8192), content = text

Faithfulness: the official indexer/embedder encoded each document on its own
(process_images([image]) / encode(document, is_query=False)), so we do the same
batch-size-1 and keep the returned token vectors verbatim (the model already
L2-normalises every token vector; no padding rows are stored).

Candidates = union of filenames in runs/<qid>_<dense arm>.json.
Output: analysis/replay_data/emb/<dataset>/<arm>_multi.npz  (object array of (T,D) float32)
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

from replay_lib import DATA_ROOT, EMB_ROOT, RUNS_ROOT

MR_MODEL = "nomic-ai/colnomic-embed-multimodal-7b"
TR_MODEL = "lightonai/GTE-ModernColBERT-v1"
DENSE_OF = {"MR": "MD", "TR": "TD"}


def needed_candidates(dataset, dense_arm):
    want = set()
    for p in glob.glob(os.path.join(RUNS_ROOT, f"*_{dense_arm}.json")):
        rec = json.load(open(p))
        if rec["dataset"] == dataset:
            want.update(x["filename"] for x in rec["top100"])
    return want


def load_existing(path):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z["filenames"]), list(z["vectors"])
    return [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["MR", "TR"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--all-pages", action="store_true",
                    help="encode the FULL corpus (diagnostic), not only dense-Top100 union")
    ap.add_argument("--shard", default="0/1", help="i/n shard of the page list")
    args = ap.parse_args()

    dense_arm = DENSE_OF[args.arm]
    si, sn = map(int, args.shard.split("/"))
    if args.all_pages:
        pp = pd.read_parquet(os.path.join(DATA_ROOT, args.dataset, "pages.parquet"))
        want = set(pp["filename"].iloc[si::sn])
    else:
        want = needed_candidates(args.dataset, dense_arm)
    tag = f"_shard{si}of{sn}" if args.all_pages and sn > 1 else ("_full" if args.all_pages else "")
    out_path = os.path.join(EMB_ROOT, args.dataset, f"{args.arm}_multi{tag}.npz")
    fns, vecs = load_existing(out_path)
    todo = sorted(want - set(fns))
    print(f"{args.dataset} {args.arm}: candidates {len(want)}, have {len(fns)}, todo {len(todo)}")
    if not todo:
        return

    pages = pd.read_parquet(os.path.join(DATA_ROOT, args.dataset, "pages.parquet"))
    page_idx = {fn: i for i, fn in enumerate(pages["filename"])}

    device = torch.device(f"cuda:{args.gpu}")
    if args.arm == "MR":
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        model = ColQwen2_5.from_pretrained(MR_MODEL, torch_dtype=torch.bfloat16,
                                           device_map=device).eval()
        processor = ColQwen2_5_Processor.from_pretrained(MR_MODEL)

        def embed_one(fn):
            img = Image.open(io.BytesIO(pages["img_bytes"].iloc[page_idx[fn]])).convert("RGB")
            with torch.inference_mode():
                inp = processor.process_images([img]).to(device)
                return model(**inp).float().cpu().numpy()[0]
    else:
        from pylate import models
        model = models.ColBERT(model_name_or_path=TR_MODEL, document_length=8192).to(device)

        def embed_one(fn):
            t = pages["text"].iloc[page_idx[fn]]
            if not isinstance(t, str) or not t.strip():
                t = " "
            return np.asarray(model.encode(t, is_query=False), dtype=np.float32)

    for n, fn in enumerate(todo):
        v = embed_one(fn)
        vecs.append(v.astype(np.float32))
        fns.append(fn)
        if n % 10 == 0:
            print(f"  {args.arm} {n+1}/{len(todo)} tokens={v.shape[0]}", flush=True)
            np.savez(out_path, vectors=np.array(vecs, dtype=object), filenames=np.array(fns))
    np.savez(out_path, vectors=np.array(vecs, dtype=object), filenames=np.array(fns))
    print(f"saved {out_path}: {len(fns)} pages")


if __name__ == "__main__":
    main()
