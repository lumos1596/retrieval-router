"""Shared helpers for rerank replay."""
import json
import os
import uuid

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "corpus")
EMB_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "emb")
RUNS_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "runs")


def load_cases():
    return pd.read_csv(os.path.join(REPO_ROOT, "analysis", "cases_28.csv"))


def load_qrels(dataset):
    with open(os.path.join(DATA_ROOT, dataset, "qrels.json")) as f:
        return json.load(f)


def page_uuid(fn):
    """Official point id: uuid5(NAMESPACE_DNS, filename) (base_database.py)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, fn))


def cosine_topk(query_vec, doc_vecs, filenames, top_k):
    """Exact cosine ranking (Qdrant exact_search=True), uuid5 tie-break."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    scores = d @ q
    order = order_by_scores(scores, filenames)[:top_k]
    return [filenames[r] for r in order], scores[order].astype(float)


def order_by_scores(scores, filenames):
    """Descending order with Qdrant point-id tie-break (uuid5 of filename)."""
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    final, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[j] == s[i]:
            j += 1
        grp = list(order[i:j])
        if len(grp) > 1:
            grp.sort(key=lambda r: int(page_uuid(filenames[r]).replace("-", ""), 16), reverse=True)
        final.extend(grp)
        i = j
    return np.array(final)


def maxsim_scores(query_tok, doc_toks):
    """ColBERT/ColQwen late interaction: sum over query tokens of max dot product.
    Vectors are L2-normalised by the models, so dot == cosine (Qdrant MAX_SIM).
    query_tok: (Tq, D); doc_toks: list of (Td, D)."""
    out = np.empty(len(doc_toks), dtype=np.float64)
    for i, p in enumerate(doc_toks):
        sim = query_tok @ p.T
        out[i] = sim.max(axis=1).sum()
    return out


def rerank_topk(query_tok, doc_toks, filenames, top_k):
    scores = maxsim_scores(query_tok, doc_toks)
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    final, i = [], 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s_sorted[j] == s_sorted[i]:
            j += 1
        grp = list(order[i:j])
        if len(grp) > 1:
            grp.sort(key=lambda r: int(page_uuid(filenames[r]).replace("-", ""), 16), reverse=True)
        final.extend(grp)
        i = j
    order = np.array(final[:top_k])
    return [filenames[r] for r in order], scores[order].astype(float)


def ndcg_at_k(ranked_filenames, gt_set, k=5):
    """Binary-relevance nDCG@k, pytrec_eval convention."""
    dcg = sum(1.0 / np.log2(i + 2) for i, fn in enumerate(ranked_filenames[:k]) if fn in gt_set)
    ideal = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def gt_buckets(ranked_100, gt_set):
    """Relevant page positions: rank 1-5 / 6-20 / 21-100 / null(not in dense top100)."""
    rank_of = {fn: i + 1 for i, fn in enumerate(ranked_100)}
    return {fn: rank_of.get(fn) for fn in sorted(gt_set)}
