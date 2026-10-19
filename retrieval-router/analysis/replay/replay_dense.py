"""Replay dense stage over the FULL corpus and write per-case Top-100 runs.

  --arm MD  uses BiQwen2_5 (query processed with process_queries)
  --arm TD  uses Linq-Embed-Mistral (official query prompt)

Writes analysis/replay_data/runs/<qid>_<arm>.json with the Top-100 (filename,score),
dense nDCG@5, ground-truth page ranks and the official xlsx nDCG for comparison.
"""
import argparse
import json
import os

import numpy as np
import torch

from replay_lib import (RUNS_ROOT, cosine_topk, gt_buckets, load_cases, load_qrels,
                        ndcg_at_k)

LINQ_PROMPT = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
MD_MODEL = "nomic-ai/nomic-embed-multimodal-7b"
TD_MODEL = "Linq-AI-Research/Linq-Embed-Mistral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["MD", "TD"], required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RUNS_ROOT, exist_ok=True)

    cases = load_cases()
    device = torch.device(f"cuda:{args.gpu}")

    if args.arm == "MD":
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        model = BiQwen2_5.from_pretrained(MD_MODEL, torch_dtype=torch.bfloat16,
                                          device_map=device).eval()
        processor = BiQwen2_5_Processor.from_pretrained(MD_MODEL)

        def qvec(q):
            with torch.inference_mode():
                inp = processor.process_queries([q]).to(device)
                return model(**inp).float().cpu().numpy()[0]
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(TD_MODEL, device=device,
                                    model_kwargs={"torch_dtype": torch.bfloat16})

        def qvec(q):
            return model.encode(q, prompt=LINQ_PROMPT, convert_to_numpy=True)

    for dataset in args.datasets:
        sub = cases[cases["DATASET"] == dataset]
        if sub.empty:
            continue
        z = np.load(os.path.join(os.path.dirname(RUNS_ROOT), "emb", dataset,
                                 f"{args.arm}_dense.npz"), allow_pickle=True)
        vecs, filenames = z["vectors"], list(z["filenames"])
        qrels = load_qrels(dataset)
        print(f"{dataset} {args.arm}: corpus {vecs.shape}, {len(sub)} cases")

        for _, c in sub.iterrows():
            qid = int(c["qid"])
            out_path = os.path.join(RUNS_ROOT, f"{qid}_{args.arm}.json")
            if os.path.exists(out_path):
                continue
            qv = qvec(c["query"])
            top_fns, top_scores = cosine_topk(qv, vecs, filenames, 100)
            gt = set(qrels[c["query"]])
            ranked5 = top_fns[:5]
            rec = {
                "qid": qid, "dataset": dataset, "arm": args.arm, "stage": "dense",
                "query": c["query"], "group": c["group"],
                "top100": [{"rank": i + 1, "filename": fn, "score": float(s)}
                           for i, (fn, s) in enumerate(zip(top_fns, top_scores))],
                "gt_pages": gt_buckets(top_fns, gt),
                "replay_ndcg5": ndcg_at_k(top_fns, gt),
                "official_ndcg5": float(c[f"ndcg_{args.arm}"]),
                "n_gt_pages": len(gt),
            }
            with open(out_path, "w") as f:
                json.dump(rec, f)
            print(f"  qid {qid} [{c['group']}] replay={rec['replay_ndcg5']:.4f} "
                  f"official={rec['official_ndcg5']:.4f} "
                  f"{'MATCH' if abs(rec['replay_ndcg5']-rec['official_ndcg5'])<1e-9 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
