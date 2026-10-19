"""Compare numpy MaxSim rerank against a real local Qdrant MAX_SIM collection
on one case's 100 candidates. Settles whether score semantics explain the
official-rerank mismatch."""
import argparse
import json
import os
import uuid

import numpy as np
import torch
from qdrant_client import QdrantClient, models

from replay_lib import EMB_ROOT, RUNS_ROOT, maxsim_scores, order_by_scores

MR_MODEL = "nomic-ai/colnomic-embed-multimodal-7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    d = json.load(open(os.path.join(RUNS_ROOT, f"{args.qid}_MR.json")))
    ds = d["dataset"]
    z = np.load(os.path.join(EMB_ROOT, ds, "MR_multi.npz"), allow_pickle=True)
    mv = {fn: v for fn, v in zip(z["filenames"], z["vectors"])}
    cands = [x["filename"] for x in sorted(d["candidate_table"], key=lambda r: r["dense_rank"])]

    device = torch.device(f"cuda:{args.gpu}")
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    model = ColQwen2_5.from_pretrained(MR_MODEL, torch_dtype=torch.bfloat16,
                                       device_map=device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MR_MODEL)
    with torch.inference_mode():
        qv = model(**processor.process_queries([d["query"]]).to(device)).float().cpu().numpy()[0]
    print("query tokens:", qv.shape, "token norms ~", np.linalg.norm(qv, axis=1).mean())

    client = QdrantClient(":memory:")
    client.recreate_collection(
        "c",
        vectors_config=models.VectorParams(
            size=int(mv[cands[0]].shape[1]), distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM)),
    )
    pts = []
    for fn in cands:
        pts.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, fn)),
                                      vector=np.asarray(mv[fn], dtype=np.float32).tolist(),
                                      payload={"filename": fn}))
    client.upsert("c", pts)

    hits = client.query_points(
        "c", query=np.asarray(qv, dtype=np.float32).tolist(),
        limit=100, search_params=models.SearchParams(exact=True)).points
    q_order = [h.payload["filename"] for h in hits]
    q_scores = [h.score for h in hits]

    ns = maxsim_scores(qv, [mv[c] for c in cands])
    n_order = [cands[i] for i in order_by_scores(np.asarray(ns), cands)]

    print("\nrank qdrant_score  numpy_score  same?  GT  filename")
    gt = set(d["gt_rerank_ranks"])
    for i in range(10):
        fn = q_order[i]
        j = n_order.index(fn)
        print(f"{i+1:3d} {q_scores[i]:9.4f} {ns[cands.index(fn)]:9.4f} "
              f"{'EQ' if q_order[i]==n_order[i] else 'XX':>2} "
              f"{'GT' if fn in gt else '  '} d{next(x['dense_rank'] for x in d['candidate_table'] if x['filename']==fn):>3} {fn[:46]}")
    print("\nnumpy top5:", [f[:20] for f in n_order[:5]])
    print("qdrant GT ranks:", [i+1 for i,f in enumerate(q_order) if f in gt],
          "numpy GT ranks:", [i+1 for i,f in enumerate(n_order) if f in gt])


if __name__ == "__main__":
    main()
