"""Aggregate the 28 replay runs into rerank_case_details.csv (plan section 6).

Per case we record:
  * official vs replay nDCG@5 for dense & rerank (alignment gate);
  * every GT page's rank in dense vs rerank (1-5 / 6-20 / 21-100 / missing) and
    the plan-section-4 primary error type;
  * candidate-set features available at inference time WITHOUT labels:
    dense margins, query content-token coverage, digit/year density,
    near-duplicate candidate count, rerank margins;
  * the page that overtook (or was overtook by) each GT page and text snippets
    for the manual casebook.

Features use only query text + candidate pages (never qrels) so they could be
computed online in a candidate-aware gate.
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from replay_lib import DATA_ROOT, REPO_ROOT, RUNS_ROOT, load_cases, load_qrels

STOP = set("the a an of in on at to for and or with from by as is are was were be been "
           "what which who whom whose when where why how how's did do does done this that "
           "these those it its their his her our your my we you they i he she them us not "
           "no yes can could would should will shall may might must than then into per via "
           "between over under about across after before during each all any both more most "
           "other some such only own same so if because while company s t inc ltd".split())
TOK = re.compile(r"[a-z0-9]+")
YEAR = re.compile(r"(19|20)\d{2}")
PCT = re.compile(r"\d[\d,\.]*\s*(?:%|percent|million|billion|thousand|m|bn)?")


def tokens(s):
    return [t for t in TOK.findall(str(s).lower()) if len(t) > 1 and t not in STOP]


def digit_density(s):
    s = str(s)
    toks = s.split()
    if not toks:
        return 0.0
    return sum(any(c.isdigit() for c in t) for t in toks) / len(toks)


def table_score(s):
    """Crude page-type score: markdown table / pipe density."""
    s = str(s)
    lines = s.splitlines() or [s]
    return sum("|" in ln for ln in lines) / max(len(lines), 1)


def features_for_page(query_toks, years, numbers, text):
    toks = set(tokens(text))
    if not query_toks:
        cov = 0.0
    else:
        cov = sum(t in toks for t in query_toks) / len(query_toks)
    y_hit = sum(y in str(text) for y in years) / max(len(years), 1)
    n_hit = sum(n in str(text) for n in numbers) / max(len(numbers), 1)
    return {
        "token_coverage": round(cov, 3),
        "year_coverage": round(y_hit, 3),
        "number_coverage": round(n_hit, 3),
        "digit_density": round(digit_density(text), 3),
        "table_line_ratio": round(table_score(text), 3),
        "text_len_chars": len(str(text)),
    }


def bucket(rank):
    if rank is None:
        return "missing_top100"
    if rank <= 5:
        return "1-5"
    if rank <= 20:
        return "6-20"
    return "21-100"


def classify(gt_dense, gt_rr):
    """Plan section 4 five-way primary label (per query, using the best GT page)."""
    d_best = min([r for r in gt_dense if r is not None], default=None)
    r_best = min([r for r in gt_rr if r is not None], default=None)
    if d_best is None and r_best is None:
        return "4_recall_eval_anomaly"
    if d_best is not None and d_best <= 5 and (r_best is None or r_best > 5):
        return "1_rerank_demotes"
    if (d_best is None or d_best > 5) and r_best is not None and r_best <= 5:
        return "2_rerank_promotes"
    if d_best is not None and r_best is not None and d_best <= 5 and r_best <= 5 and d_best != r_best:
        return "3_rank_shift_inside5"
    n_gt = len(gt_dense)
    if n_gt > 1 and any((a is None) != (b is None) or (a and b and a <= 5) != (b and b <= 5)
                        for a, b in zip(gt_dense, gt_rr)):
        return "5_complex_multi_gt"
    return "3_rank_shift_inside5" if d_best != r_best else "no_change"


def main():
    cases = load_cases().set_index("qid")
    rows = []
    for _, c in cases.iterrows():
        qid = int(c.name)
        dataset = c["DATASET"]
        qrels = load_qrels(dataset)
        pages = pd.read_parquet(os.path.join(DATA_ROOT, dataset, "pages.parquet"))
        pidx = {fn: i for i, fn in enumerate(pages["filename"])}
        gt = set(qrels[c["query"]])

        q_toks = tokens(c["query"])
        years = YEAR.findall(c["query"])  # tuples; use full matches below
        years = [m.group(0) for m in YEAR.finditer(c["query"])]
        numbers = [m.group(0).replace(",", "").strip()
                   for m in re.finditer(r"\d[\d,]*\.?\d*", c["query"])]

        row = {"qid": qid, "dataset": dataset, "group": c["group"], "query": c["query"],
               "pred_arm": c["pred_arm"],
               "off_ndcg_MR": c["ndcg_MR"], "off_ndcg_MD": c["ndcg_MD"],
               "off_ndcg_TR": c["ndcg_TR"], "off_ndcg_TD": c["ndcg_TD"],
               "n_gt_pages": len(gt)}

        for arm, dense in [("MR", "MD"), ("TR", "TD")]:
            path = os.path.join(RUNS_ROOT, f"{qid}_{arm}.json")
            if not os.path.exists(path):
                continue
            rec = json.load(open(path))
            tab = pd.DataFrame(rec["candidate_table"])
            d_sorted = tab.sort_values("dense_rank")
            r_sorted = tab.sort_values("rerank_rank")

            row[f"{arm}_dense_ndcg_replay"] = rec["dense_ndcg5"]
            row[f"{arm}_rerank_ndcg_replay"] = rec["rerank_ndcg5"]
            row[f"{arm}_dense_ndcg_official"] = rec["official_dense_ndcg5"]
            row[f"{arm}_rerank_ndcg_official"] = rec["official_rerank_ndcg5"]
            row[f"{arm}_dense_aligned"] = abs(rec["dense_ndcg5"] - rec["official_dense_ndcg5"]) < 1e-9
            row[f"{arm}_rerank_aligned"] = abs(rec["rerank_ndcg5"] - rec["official_rerank_ndcg5"]) < 1e-9

            gd = [rec["gt_dense_ranks"].get(fn) for fn in sorted(gt)]
            gr = [rec["gt_rerank_ranks"].get(fn) for fn in sorted(gt)]
            row[f"{arm}_gt_dense_buckets"] = ";".join(f"{fn[-8:]}:{bucket(r)}" for fn, r in zip(sorted(gt), gd))
            row[f"{arm}_gt_rerank_buckets"] = ";".join(f"{fn[-8:]}:{bucket(r)}" for fn, r in zip(sorted(gt), gr))
            row[f"{arm}_error_type"] = classify(gd, gr)
            row[f"{arm}_best_gt_dense_rank"] = min([r for r in gd if r], default=999)
            row[f"{arm}_best_gt_rerank_rank"] = min([r for r in gr if r], default=999)

            # candidate-set features (label-free), evaluated on dense top10 / top5
            top10 = d_sorted.head(10).reset_index(drop=True)
            top5 = d_sorted.head(5).reset_index(drop=True)
            ds = top10["dense_score"].to_numpy()
            rs_all = tab["rerank_score"].to_numpy()
            row[f"{arm}_dense_margin12"] = round(float(ds[0] - ds[1]), 4) if len(ds) > 1 else 0
            row[f"{arm}_dense_gap56"] = round(float(d_sorted["dense_score"].iloc[4] -
                                                    d_sorted["dense_score"].iloc[5]), 4)
            r5 = tab.nlargest(6, "rerank_score")["rerank_score"].to_numpy()
            row[f"{arm}_rerank_margin12"] = round(float(r5[0] - r5[1]), 4)
            row[f"{arm}_rerank_gap56"] = round(float(r5[4] - r5[5]), 4)
            row[f"{arm}_near_dup_in_top10"] = int((ds[None, :] - ds[:, None] < 0.005).sum() - len(ds) > 0)

            # tie-zone flags around the Top-5 boundary: exact ties are ordered by
            # Qdrant point/storage order which our replay cannot reproduce, and
            # sub-0.5% margins can flip under bf16 numeric drift; such cases are
            # reproduction-boundary, NOT evidence for/against the mechanism.
            def boundary_flags(view, score_col):
                window = view.head(7).reset_index(drop=True)
                sc = window[score_col].to_numpy()
                scale = max(abs(sc).max(), 1e-9)
                exact = bool((abs(sc[:, None] - sc[None, :]) < 1e-7).sum() > len(sc))
                # near-tie = relative margin < 0.5 % (score scales differ across arms:
                # cosine dense ~0-1, MaxSim sums ~10-40, so an absolute cutoff is biased).
                near = bool((abs(sc[:, None] - sc[None, :]) / scale < 0.005).sum() > len(sc))
                gap = round(float(view[score_col].iloc[4] - view[score_col].iloc[5]), 5)
                return exact, near, gap
            d_exact, d_near, d_gap = boundary_flags(d_sorted, "dense_score")
            r_exact, r_near, r_gap = boundary_flags(r_sorted, "rerank_score")
            row[f"{arm}_dense_exact_tie_near5"] = d_exact
            row[f"{arm}_dense_neartie_near5"] = d_near
            row[f"{arm}_rerank_exact_tie_near5"] = r_exact
            row[f"{arm}_rerank_neartie_near5"] = r_near
            row[f"{arm}_dense_rel_gap56"] = round(abs(d_gap) / max(abs(d_sorted["dense_score"].iloc[4]), 1e-9), 4)
            row[f"{arm}_rerank_rel_gap56"] = round(abs(r_gap) / max(abs(r_sorted["rerank_score"].iloc[4]), 1e-9), 4)

            # GT margin: relative score gap between the best GT page and the best
            # non-GT page in OUR replay. A mismatch with a large GT margin is a
            # decisive disagreement (cannot be explained by bf16/tie jitter);
            # a mismatch inside a ~1 % cluster is a boundary-jitter case.
            gt_sc = tab.loc[tab["filename"].isin(gt), "rerank_score"]
            non_gt = tab.loc[~tab["filename"].isin(gt), "rerank_score"]
            if len(gt_sc) and len(non_gt):
                best_gt, best_other = float(gt_sc.max()), float(non_gt.max())
                row[f"{arm}_gt_rerank_rel_margin"] = round(
                    abs(best_gt - best_other) / max(abs(best_gt), abs(best_other), 1e-9), 4)
            else:
                row[f"{arm}_gt_rerank_rel_margin"] = np.nan

            # mismatch provenance (reproduction audit, plan section 4):
            #  exact/near tie at the boundary -> Qdrant order / bf16 drift;
            #  large-margin rerank mismatch with aligned dense -> unrecoverable
            #  upstream late-interaction collection state (see report appendix).
            def mismatch_kind(aligned, exact, near):
                if aligned:
                    return "aligned"
                if exact:
                    return "exact_tie_boundary"
                if near:
                    return "near_tie_boundary"
                return "upstream_rerank_deviation"
            row[f"{arm}_dense_mismatch_kind"] = mismatch_kind(
                row[f"{arm}_dense_aligned"], d_exact, d_near)
            if row[f"{arm}_rerank_aligned"]:
                r_kind = "aligned"
            elif r_exact:
                r_kind = "exact_tie_boundary"
            elif r_near and row[f"{arm}_gt_rerank_rel_margin"] < 0.01:
                r_kind = "near_tie_cluster"          # dense score cluster + GT inside ~1%
            elif r_near:
                r_kind = "boundary_jitter_decisive_gt"  # tight 5/6 cluster but GT decisively placed
            else:
                r_kind = "upstream_rerank_deviation"
            row[f"{arm}_rerank_mismatch_kind"] = r_kind

            # feature of the page ranked #1 by dense and by rerank, and of best GT page
            def page_feats(fn):
                return features_for_page(q_toks, years, numbers, pages["text"].iloc[pidx[fn]])
            d1 = top5.iloc[0]["filename"]; r1 = r_sorted.iloc[0]["filename"]
            gt_best_fn = sorted(gt, key=lambda f: rec["gt_dense_ranks"].get(f) or 999)[0]
            for tag, fn in [("dense1", d1), ("rerank1", r1), ("gt_best", gt_best_fn)]:
                for k, v in page_feats(fn).items():
                    row[f"{arm}_{tag}_{k}"] = v

            # which non-GT page overtook GT, and how far GT moved
            rr1_is_gt = r1 in gt
            row[f"{arm}_rerank1_is_gt"] = rr1_is_gt
            overtaker = None
            best_gt_rr = min([r for r in gr if r], default=None)
            if not rr1_is_gt:
                overtaker = r1
            row[f"{arm}_overtaker_fn"] = overtaker
            if overtaker:
                row[f"{arm}_overtaker_snippet"] = str(pages["text"].iloc[pidx[overtaker]])[:240].replace("\n", " ")
                row[f"{arm}_gt_best_snippet"] = str(pages["text"].iloc[pidx[gt_best_fn]])[:240].replace("\n", " ")

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(REPO_ROOT, "analysis", "rerank_case_details.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(out)} rows x {len(out.columns)} cols")

    # alignment summary
    for arm in ["MR", "TR"]:
        cols = [f"{arm}_dense_aligned", f"{arm}_rerank_aligned"]
        for cc in cols:
            if cc in out:
                print(cc, out[cc].value_counts(dropna=False).to_dict())
        if f"{arm}_error_type" in out:
            print(arm, "error types:", out[f"{arm}_error_type"].value_counts().to_dict())
        for cc in (f"{arm}_dense_mismatch_kind", f"{arm}_rerank_mismatch_kind"):
            if cc in out:
                print(cc, out[cc].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
