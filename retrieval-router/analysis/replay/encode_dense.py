"""Encode the FULL corpus of a dataset with the official stage-1 (dense) models.

  MD : nomic-ai/nomic-embed-multimodal-7b (BiQwen2.5, last-token pooled, L2 normed)
       content = rendered page image  (matches data_collection/pipelines/indexing.py)
  TD : Linq-AI-Research/Linq-Embed-Mistral via SentenceTransformer
       content = extracted page `text`; query-side prompt is added only at query time

Output: analysis/replay_data/emb/<dataset>/<arm>_dense.npz
          vectors float32 (N x D), filenames aligned by row
"""
import argparse
import io
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "corpus")
EMB_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "emb")

MD_MODEL = "nomic-ai/nomic-embed-multimodal-7b"
TD_MODEL = "Linq-AI-Research/Linq-Embed-Mistral"


def load_pages(dataset):
    df = pd.read_parquet(os.path.join(DATA_ROOT, dataset, "pages.parquet"))
    return df


def encode_md(pages, gpu, batch_size=4):
    from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
    device = torch.device(f"cuda:{gpu}")
    model = BiQwen2_5.from_pretrained(MD_MODEL, torch_dtype=torch.bfloat16,
                                      device_map=device).eval()
    processor = BiQwen2_5_Processor.from_pretrained(MD_MODEL)
    vecs = []
    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in pages["img_bytes"]]
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size]
        with torch.inference_mode():
            inp = processor.process_images(batch).to(device)
            out = model(**inp).float().cpu().numpy()  # (B, D), already L2 normed
        if out.ndim != 2:
            raise RuntimeError(f"expected pooled (B,D), got {out.shape}")
        vecs.append(out)
        if (i // batch_size) % 20 == 0:
            print(f"  MD {i+len(batch)}/{len(imgs)}", flush=True)
    return np.concatenate(vecs, 0).astype(np.float32)


def encode_td(pages, gpu, batch_size=16):
    from sentence_transformers import SentenceTransformer
    device = f"cuda:{gpu}"
    # 7B model: fp32 weights need ~28 GiB and OOM on a 24 GiB card; bf16 matches
    # the model's release dtype and the official GPU inference setup.
    model = SentenceTransformer(TD_MODEL, device=device,
                                model_kwargs={"torch_dtype": torch.bfloat16})
    texts = [t if isinstance(t, str) and t.strip() else " " for t in pages["text"]]
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=False)
    return vecs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["MD", "TD"], required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    out_dir = os.path.join(EMB_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.arm}_dense.npz")
    if os.path.exists(out_path):
        existing = np.load(out_path, allow_pickle=True)
        print(f"{out_path} exists: {existing['vectors'].shape}; remove to rebuild")
        return

    pages = load_pages(args.dataset)
    print(f"{args.dataset}: {len(pages)} pages -> {args.arm} on gpu{args.gpu}")
    if args.arm == "MD":
        vecs = encode_md(pages, args.gpu, args.batch_size or 4)
    else:
        vecs = encode_td(pages, args.gpu, args.batch_size or 16)
    np.savez(out_path, vectors=vecs, filenames=np.array(pages["filename"].tolist()))
    print(f"saved {out_path} {vecs.shape}")


if __name__ == "__main__":
    main()
