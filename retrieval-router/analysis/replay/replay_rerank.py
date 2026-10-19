"""Replay the late-interaction rerank on EACH case's dense Top-100 (same candidates).

  MR reranks the MD Top-100 (ColNomic MaxSim); TR reranks the TD Top-100.
Writes analysis/replay_data/runs/<qid>_<arm>.json with the full 100-row table
(dense rank/score + rerank rank/score), GT page positions in both stages,
rerank nDCG@5 and the official xlsx value for alignment verification.
"""
import argparse
import json
import os

import numpy as np
import torch

from replay_lib import (EMB_ROOT, RUNS_ROOT, gt_buckets, load_qrels, maxsim_scores,
                        ndcg_at_k, order_by_scores)

MR_MODEL = "nomic-ai/colnomic-embed-multimodal-7b"
TR_MODEL = "lightonai/GTE-ModernColBERT-v1"
DENSE_OF = {"MR": "MD", "TR": "TD"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["MR", "TR"], required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dense_arm = DENSE_OF[args.arm]
    device = torch.device(f"cuda:{args.gpu}")

    if args.arm == "MR":
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        model = ColQwen2_5.from_pretrained(MR_MODEL, torch_dtype=torch.bfloat16,
                                           device_map=device).eval()
        processor = ColQwen2_5_Processor.from_pretrained(MR_MODEL)

        def qtok(q):
            with torch.inference_mode():
                inp = processor.process_queries([q]).to(device)
                return model(**inp).float().cpu().numpy()[0]
    else:
        from pylate import models
        model = models.ColBERT(model_name_or_path=TR_MODEL, document_length=8192).to(device)

        def qtok(q):
            # pylate returns a single (T,D) ndarray when given a single string
            return np.asarray(model.encode(q, is_query=True), dtype=np.float32)

    for dataset in args.datasets:
        qrels = load_qrels(dataset)
        z = np.load(os.path.join(EMB_ROOT, dataset, f"{args.arm}_multi.npz"), allow_pickle=True)
        mv = {fn: v for fn, v in zip(z["filenames"], z["vectors"])}

        dense_files = sorted(
            p for p in [os.path.join(RUNS_ROOT, f) for f in os.listdir(RUNS_ROOT)]
            if p.endswith(f"_{dense_arm}.json") and json.load(open(p))["dataset"] == dataset)
        print(f"{dataset} {args.arm}: {len(dense_files)} cases, {len(mv)} candidate vectors")

        for dp in dense_files:
            d = json.load(open(dp))
            qid = d["qid"]
            out_path = os.path.join(RUNS_ROOT, f"{qid}_{args.arm}.json")
            if os.path.exists(out_path):
                continue
            cands = [x["filename"] for x in d["top100"]]
            missing = [c for c in cands if c not in mv]
            if missing:
                print(f"  qid {qid}: {len(missing)} candidates not encoded, skip")
                continue
            qv = qtok(d["query"])
            scores = maxsim_scores(qv, [mv[c] for c in cands])
            order = order_by_scores(np.asarray(scores), cands).tolist()
            rr_rank = {cands[i]: r + 1 for r, i in enumerate(order)}
            rr_score = {cands[i]: float(scores[i]) for i in range(len(cands))}
            ranked_rr = [cands[i] for i in order]
            gt = set(qrels[d["query"]])

            rows = []
            for x in d["top100"]:
                fn = x["filename"]
                rows.append({"filename": fn, "dense_rank": x["rank"],
                             "dense_score": x["score"],
                             "rerank_rank": rr_rank[fn], "rerank_score": rr_score[fn]})
            rec = {
                "qid": qid, "dataset": dataset, "arm": args.arm, "stage": "rerank",
                "query": d["query"], "group": d["group"],
                "candidate_table": rows,
                "gt_dense_ranks": d["gt_pages"],
                "gt_rerank_ranks": gt_buckets(ranked_rr, gt),
                "dense_ndcg5": d["replay_ndcg5"],
                "rerank_ndcg5": ndcg_at_k(ranked_rr, gt),
                "official_dense_ndcg5": d["official_ndcg5"],
                "official_rerank_ndcg5": float(_official(qid, args.arm)),
                "n_gt_pages": d["n_gt_pages"],
            }
            with open(out_path, "w") as f:
                json.dump(rec, f)
            print(f"  qid {qid} [{d['group']}] dense={rec['dense_ndcg5']:.4f} "
                  f"(off {rec['official_dense_ndcg5']:.4f}) | "
                  f"rerank={rec['rerank_ndcg5']:.4f} (off {rec['official_rerank_ndcg5']:.4f}) "
                  f"{'MATCH' if abs(rec['rerank_ndcg5']-rec['official_rerank_ndcg5'])<1e-9 else 'MISMATCH'}")


def _official(qid, arm):
    import pandas as pd
    from replay_lib import REPO_ROOT
    c = pd.read_csv(os.path.join(REPO_ROOT, "analysis", "cases_28.csv"))
    return float(c.loc[c["qid"] == qid, f"ndcg_{arm}"].iloc[0])


if __name__ == "__main__":
    main()
