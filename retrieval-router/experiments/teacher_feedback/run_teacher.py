"""Teacher feedback: T0 query-only, T1 outcome-aware, T2 evidence-grounded.
Uses generated Top-5 retrieval evidence (MD/TD dense + MR/TR rerank with GT flags)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arms import ARM_NAMES, NDCG_COLS, SHORT_NAMES

MODEL_PATH = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
OUT = "experiments/teacher_feedback"
os.makedirs(OUT, exist_ok=True)
EVIDENCE_JSON = os.path.join(OUT, "evidence_generated.json")

DESC = """Five retrieval methods:
- BM (BM25): lexical keyword matching. Strong for exact entities, rare terms, precise strings.
- TD (Text-Dense): single-vector dense text retrieval. Strong for semantic similarity/paraphrase.
- TR (Text-Rerank): text-dense recall + token-level late-interaction rerank. Strong for fine-grained multi-condition text.
- MD (MM-Dense): single-vector dense over multimodal page representations (text+image regions).
- MR (MM-Rerank): multimodal-dense recall + visual late-interaction rerank. Strongest for figures/tables/layout."""

SYS = "You analyze why a retrieval method works or fails on a query. Distinguish observed facts from hypotheses. Do not invent evidence you were not given."

ARM_SHORT = {a: SHORT_NAMES[a] for a in ARM_NAMES}
MR = "MULTIMODAL_RERANK"

def load_evidence():
    """Load generated evidence (Top-5 rankings with GT flags)."""
    ev = {}
    if os.path.exists(EVIDENCE_JSON):
        data = json.load(open(EVIDENCE_JSON))
        for r in data:
            q = r["query"].strip()
            arms = {}
            # MD dense
            if "MD_dense_top5" in r:
                arms["MULTIMODAL-SINGLE"] = {
                    "dense_top5": [(x["filename"], i+1, x["score"], x["gt"]) for i,x in enumerate(r["MD_dense_top5"])],
                    "dense_ndcg5": r.get("MD_dense_ndcg5"),
                }
            # TD dense
            if "TD_dense_top5" in r:
                arms["TEXT-SINGLE"] = {
                    "dense_top5": [(x["filename"], i+1, x["score"], x["gt"]) for i,x in enumerate(r["TD_dense_top5"])],
                    "dense_ndcg5": r.get("TD_dense_ndcg5"),
                }
            # MR rerank (uses MD dense recall)
            if "MR_rerank_top5" in r:
                mr_dense = arms.get("MULTIMODAL-SINGLE", {}).get("dense_top5", [])
                arms[MR] = {
                    "dense_top5": mr_dense,
                    "rerank_top5": [(x["filename"], i+1, x["score"], x["gt"]) for i,x in enumerate(r["MR_rerank_top5"])],
                    "dense_ndcg5": r.get("MD_dense_ndcg5"),
                    "rerank_ndcg5": r.get("MR_rerank_ndcg5"),
                }
            else:
                # MR = MD dense + rerank; if no rerank, use MD dense as MR's evidence
                if "MULTIMODAL-SINGLE" in arms:
                    arms[MR] = {
                        "dense_top5": arms["MULTIMODAL-SINGLE"]["dense_top5"],
                        "dense_ndcg5": r.get("MD_dense_ndcg5"),
                        "note": "rerank not available; showing dense recall only",
                    }
            # TR rerank
            if "TR_rerank_top5" in r:
                td_dense = arms.get("TEXT-SINGLE", {}).get("dense_top5", [])
                arms["TEXT_RERANK"] = {
                    "dense_top5": td_dense,
                    "rerank_top5": [(x["filename"], i+1, x["score"], x["gt"]) for i,x in enumerate(r["TR_rerank_top5"])],
                    "dense_ndcg5": r.get("TD_dense_ndcg5"),
                    "rerank_ndcg5": r.get("TR_rerank_ndcg5"),
                }
            ev[q] = arms
    print(f"Loaded evidence for {len(ev)} queries")
    return ev

def build_prompt(query, ndcg_vals, best_nonmr_arm, evidence, mode):
    mr_ndcg = ndcg_vals[MR]
    other_ndcg = ndcg_vals[best_nonmr_arm]
    ndcg_str = "\n".join(f"- {ARM_SHORT[a]}: {ndcg_vals[a]:.3f}" for a in ARM_NAMES)
    if mode == "T0":
        user = (f"{DESC}\n\nQuery: {query}\n\n"
                "Output exactly these 4 short fields (under 150 words total):\n"
                "Query signals: <entities, conditions, computation/visual needs>\n"
                "Observed retrieval evidence: unavailable\n"
                f"Arm comparison: compare MR and {ARM_SHORT[best_nonmr_arm]} from query signals only; distinguish facts from hypotheses.\n"
                "Uncertainty: <is evidence sufficient?>")
    elif mode == "T1":
        user = (f"{DESC}\n\nQuery: {query}\n\n"
                f"Observed nDCG@5 on this query:\n{ndcg_str}\n\n"
                "Output exactly these 4 short fields (under 150 words total):\n"
                "Query signals: <entities, conditions, computation/visual needs>\n"
                "Observed retrieval evidence: nDCG scores only; no page/ranking details.\n"
                f"Arm comparison: MR(nDCG={mr_ndcg:.3f}) vs {ARM_SHORT[best_nonmr_arm]}(nDCG={other_ndcg:.3f}). State the gap as fact; explain as hypotheses. Do not claim to know why.\n"
                "Uncertainty: <can you explain the gap from query+nDCG alone?>")
    else:
        if evidence:
            mr_ev = evidence.get(MR, {})
            cmp_ev = evidence.get(best_nonmr_arm, {})
            def fmt(top5, label):
                lines = [f"  {label} Top-5 (GT=relevant page, xx=not relevant):"]
                for fn, rank, score, is_gt in top5:
                    tag = "GT" if is_gt else "xx"
                    lines.append(f"    {rank}. [{tag}] {fn[:55]}")
                return "\n".join(lines)
            parts = []
            if mr_ev:
                if mr_ev.get("rerank_top5"):
                    parts.append(f"MR final rerank nDCG@5={mr_ev.get('rerank_ndcg5')}")
                    parts.append(fmt(mr_ev["rerank_top5"], "MR rerank"))
                elif mr_ev.get("dense_top5"):
                    parts.append(f"MR dense recall nDCG@5={mr_ev.get('dense_ndcg5')} (rerank unavailable)")
                    parts.append(fmt(mr_ev["dense_top5"], "MR dense recall"))
            if cmp_ev:
                if cmp_ev.get("rerank_top5"):
                    parts.append(f"{ARM_SHORT[best_nonmr_arm]} final rerank nDCG@5={cmp_ev.get('rerank_ndcg5')}")
                    parts.append(fmt(cmp_ev["rerank_top5"], f"{ARM_SHORT[best_nonmr_arm]} rerank"))
                elif cmp_ev.get("dense_top5"):
                    parts.append(f"{ARM_SHORT[best_nonmr_arm]} dense nDCG@5={cmp_ev.get('dense_ndcg5')}")
                    parts.append(fmt(cmp_ev["dense_top5"], f"{ARM_SHORT[best_nonmr_arm]} dense"))
            ev_text = "\n".join(parts) if parts else "unavailable"
        else:
            ev_text = "unavailable"
        user = (f"{DESC}\n\nQuery: {query}\n\n"
                f"Observed nDCG@5:\n{ndcg_str}\n\n"
                f"Observed retrieval ranking evidence:\n{ev_text}\n\n"
                "Output exactly these 4 short fields (under 150 words total):\n"
                "Query signals: <entities, conditions, computation/visual needs>\n"
                "Observed retrieval evidence: <cite 1-2 specific ranks/GT flags; do NOT relist all pages; or 'unavailable'>\n"
                f"Arm comparison: MR(nDCG={mr_ndcg:.3f}) vs {ARM_SHORT[best_nonmr_arm]}(nDCG={other_ndcg:.3f}). Use only provided ranking evidence; distinguish observed facts from hypotheses.\n"
                "Uncertainty: <does the evidence explain the difference?>")
    return [{"role":"system","content":SYS},{"role":"user","content":user}]

@torch.no_grad()
def batch_gen(model, tok, prompts, max_new=200, bs=8):
    results = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1500).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            results.append(tok.decode(gen, skip_special_tokens=True))
    return results

def main():
    torch.cuda.set_device(0)
    print(f"Loading {MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0")
    model.eval()

    full = pd.read_excel("train/data/updated_dataset.xlsx")
    src = full[full["DATASET"].isin(["finqa","convfinqa","vqaonbd"])].copy()
    src["query_norm"] = src["query"].str.strip()

    evidence = load_evidence()
    ev_queries = set(evidence.keys())

    mr_idx = ARM_NAMES.index(MR)
    ndcg_mat = src[NDCG_COLS].values.astype(float)
    src["mr_ndcg"] = ndcg_mat[:, mr_idx]
    non_mr = np.delete(ndcg_mat, mr_idx, axis=1)
    non_mr_arms = [a for i,a in enumerate(ARM_NAMES) if i != mr_idx]
    src["best_nonmr_ndcg"] = non_mr.max(1)
    src["best_nonmr_arm"] = [non_mr_arms[i] for i in non_mr.argmax(1)]
    src["delta"] = src["best_nonmr_ndcg"] - src["mr_ndcg"]

    # Use the 120 queries from evidence_generated as the sample
    # Match back to src to get nDCG values
    ev_in_src = src[src["query_norm"].isin(ev_queries)].copy()
    print(f"Evidence queries matched in source: {len(ev_in_src)}")

    # Stratify
    mr_win = ev_in_src[ev_in_src["delta"] <= -0.15]
    nonmr_win = ev_in_src[ev_in_src["delta"] >= 0.15]
    tie = ev_in_src[ev_in_src["delta"].abs() <= 0.05]
    print(f"Strata: MR_win={len(mr_win)}, nonMR_win={len(nonmr_win)}, tie={len(tie)}")

    rows = []
    for idx, r in ev_in_src.iterrows():
        ndcg_vals = {a: r[f"{a}_ndcg"] for a in ARM_NAMES}
        q = r["query_norm"]
        ev = evidence.get(q)
        best_arm = r["best_nonmr_arm"]
        has_ev = ev is not None and (MR in ev or best_arm in ev)
        # Determine stratum
        d = r["delta"]
        stratum = "MR_win" if d <= -0.15 else ("nonMR_win" if d >= 0.15 else "tie")
        rows.append({"query_id": len(rows), "query": q, "dataset": r["DATASET"], "stratum": stratum,
            "best_nonmr_arm": best_arm, "mr_ndcg": r["mr_ndcg"], "best_nonmr_ndcg": r["best_nonmr_ndcg"],
            "delta": r["delta"], "has_evidence": has_ev,
            "evidence_arms": list(ev.keys()) if ev else [],
            "ndcg_vals": ndcg_vals, "evidence": ev})

    print(f"Total cases: {len(rows)}, with evidence: {sum(r['has_evidence'] for r in rows)}")

    for mode in ["T0","T1","T2"]:
        prompts = [build_prompt(r["query"], r["ndcg_vals"], r["best_nonmr_arm"],
                                r["evidence"] if mode=="T2" else None, mode) for r in rows]
        max_new = 350 if mode == "T2" else 200
        print(f"\nGenerating {mode} for {len(prompts)} queries (max_new={max_new})...")
        t0 = time.time()
        resps = batch_gen(model, tok, prompts, max_new=max_new, bs=8)
        print(f"  done in {time.time()-t0:.1f}s")
        for i, r in enumerate(rows):
            r[f"{mode}_resp"] = resps[i]
            r[f"{mode}_tokens"] = len(resps[i].split())

    out_rows = []
    for r in rows:
        out_rows.append({"query_id": r["query_id"], "dataset": r["dataset"], "query": r["query"],
            "stratum": r["stratum"], "best_nonmr_arm": r["best_nonmr_arm"],
            "mr_ndcg": r["mr_ndcg"], "best_nonmr_ndcg": r["best_nonmr_ndcg"], "delta": r["delta"],
            "has_evidence": r["has_evidence"], "evidence_arms": ",".join(r["evidence_arms"]),
            "T0_resp": r["T0_resp"], "T1_resp": r["T1_resp"], "T2_resp": r["T2_resp"],
            "T0_tokens": r["T0_tokens"], "T1_tokens": r["T1_tokens"], "T2_tokens": r["T2_tokens"],
            "model": "Qwen2.5-3B-Instruct"})
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT, "teacher_feedback_cases.csv"), index=False)
    print(f"\nSaved {len(out_rows)} cases")

if __name__ == "__main__":
    main()
