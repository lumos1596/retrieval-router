"""Diagnostic: rerank case queries over the FULL corpus multivec store
(merged MR_multi_shard*.npz) to separate candidate-tail effects from
model-output drift vs the official two-stage nDCG."""
import argparse
import glob
import json
import os

import numpy as np
import torch

from replay_lib import EMB_ROOT, RUNS_ROOT, load_qrels, maxsim_scores, ndcg_at_k, order_by_scores

MR_MODEL = "nomic-ai/colnomic-embed-multimodal-7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    mv = {}
    for p in glob.glob(os.path.join(EMB_ROOT, args.dataset, "MR_multi_shard*of*.npz")):
        z = np.load(p, allow_pickle=True)
        for fn, v in zip(z["filenames"], z["vectors"]):
            mv[fn] = v
    print(f"full store: {len(mv)} pages")

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(MR_MODEL, torch_dtype=torch.bfloat16,
                                       device_map=device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MR_MODEL)
    qrels = load_qrels(args.dataset)

    runs = sorted(glob.glob(os.path.join(RUNS_ROOT, "*_MR.json")))
    for rp in runs:
        d = json.load(open(rp))
        if d["dataset"] != args.dataset:
            continue
        fns = sorted(mv)
        with torch.inference_mode():
            qv = model(**processor.process_queries([d["query"]]).to(device)).float().cpu().numpy()[0]
        sc = maxsim_scores(qv, [mv[f] for f in fns])
        order = order_by_scores(np.asarray(sc), fns)
        ranked = [fns[i] for i in order]
        gt = set(qrels[d["query"]])
        n5 = ndcg_at_k(ranked, gt)
        gt_rank_full = next(i + 1 for i, f in enumerate(ranked) if f in gt)
        # rerun restricted to our dense100 for comparison
        c100 = [x["filename"] for x in sorted(d["candidate_table"], key=lambda r: r["dense_rank"])]
        sub = maxsim_scores(qv, [mv[f] for f in c100])
        o2 = order_by_scores(np.asarray(sub), c100)
        gt_rank_100 = next(i + 1 for i in o2 if c100[i] in gt)
        # highest page outside our dense100
        outside = [(fns[i], sc[i]) for i in order if fns[i] not in set(c100)]
        top_out = outside[0]
        in100_sc = {x["filename"]: x["rerank_score"] for x in d["candidate_table"]}
        print(f"qid {d['qid']:5d} [{d['group']}] official MR={d['official_rerank_ndcg5']:.4f} "
              f"| in100 replay={d['rerank_ndcg5']:.4f} (GT r{gt_rank_100}) "
              f"| FULL={n5:.4f} (GT r{gt_rank_full}) "
              f"| best-outside: {top_out[1]:.3f} vs in100-best {max(in100_sc.values()):.3f}")


if __name__ == "__main__":
    main()
