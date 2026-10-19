"""Export page images + text for manual verification of representative cases.

For each <qid, arm> pair saves under replay_data/evidence/<qid>_<arm>/:
  query.txt
  dense_top5/  and  rerank_top5/   page images (rankN__<filename>.png)
  gt/          every judged-relevant page that lands in the top100
  pages.txt    rank, filename, dense/rerank score, GT flag, first 400 chars text
Nothing here feeds any metric; it only supports the human-verified casebook.
"""
import argparse
import json
import os
import re

import pandas as pd

from replay_lib import DATA_ROOT, RUNS_ROOT

EVID = os.path.join(os.path.dirname(RUNS_ROOT), "evidence")


def safe(fn):
    return re.sub(r"[^A-Za-z0-9._-]", "_", fn)


def export(qid, arm, pages, save_img):
    path = os.path.join(RUNS_ROOT, f"{qid}_{arm}.json")
    rec = json.load(open(path))
    dataset = rec["dataset"]
    out = os.path.join(EVID, f"{qid}_{arm}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "query.txt"), "w") as f:
        f.write(f"qid={qid} dataset={dataset} arm={arm} group={rec['group']}\n"
                f"replay dense nDCG@5={rec['dense_ndcg5']} (off {rec['official_dense_ndcg5']})\n"
                f"replay rerank nDCG@5={rec['rerank_ndcg5']} (off {rec['official_rerank_ndcg5']})\n\n"
                + rec["query"] + "\n")

    tab = pd.DataFrame(rec["candidate_table"])
    gt_dense = {fn for fn, r in rec["gt_dense_ranks"].items() if r}
    pidx = {fn: i for i, fn in enumerate(pages["filename"])}

    def dump_subdir(view, col, subdir, topn):
        d = os.path.join(out, subdir)
        os.makedirs(d, exist_ok=True)
        for _, r in view.head(topn).iterrows():
            fn = r["filename"]
            stem = f"{int(r[col]):03d}__{'GT' if fn in gt_dense else 'xx'}__{safe(fn)}"
            if save_img:
                b = pages["img_bytes"].iloc[pidx[fn]]
                if b is not None and (isinstance(b, bytes) and len(b) > 30):
                    with open(os.path.join(d, stem + ".png"), "wb") as f:
                        f.write(b)

    dump_subdir(tab.sort_values("dense_rank"), "dense_rank", "dense_top5", 5)
    dump_subdir(tab.sort_values("rerank_rank"), "rerank_rank", "rerank_top5", 5)

    # GT pages present in candidate set (up to 8)
    gtd = os.path.join(out, "gt")
    os.makedirs(gtd, exist_ok=True)
    gt_rows = tab[tab["filename"].isin(gt_dense)].sort_values("dense_rank").head(8)
    with open(os.path.join(out, "pages.txt"), "w") as f:
        f.write("rank_d rank_r | dense_sc rr_sc GT | filename | text_head\n")
        for _, r in tab.sort_values("dense_rank").head(20).iterrows():
            fn = r["filename"]
            txt = str(pages["text"].iloc[pidx[fn]]).replace("\n", " ")[:400]
            f.write(f"{int(r['dense_rank']):>3} {int(r['rerank_rank']):>4} | "
                    f"{r['dense_score']:.4f} {r['rerank_score']:.4f} "
                    f"{'GT' if fn in gt_dense else '  '} | {fn} | {txt}\n")
    if save_img:
        for _, r in gt_rows.iterrows():
            fn = r["filename"]
            b = pages["img_bytes"].iloc[pidx[fn]]
            if isinstance(b, bytes) and len(b) > 30:
                with open(os.path.join(gtd, f"d{int(r['dense_rank'])}__{safe(fn)}.png"), "wb") as f:
                    f.write(b)
    print(f"evidence -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True, help="qid:MR qid:TR ...")
    args = ap.parse_args()
    cache = {}
    for t in args.targets:
        qid, arm = t.split(":")
        qid = int(qid)
        rec = json.load(open(os.path.join(RUNS_ROOT, f"{qid}_{arm}.json")))
        ds = rec["dataset"]
        if ds not in cache:
            cache[ds] = pd.read_parquet(os.path.join(DATA_ROOT, ds, "pages.parquet"))
        export(qid, arm, cache[ds], save_img=(arm == "MR"))


if __name__ == "__main__":
    main()
