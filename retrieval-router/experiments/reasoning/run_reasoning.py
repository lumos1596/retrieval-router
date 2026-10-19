"""Reasoning router experiment: P0 direct, P1 free CoT, P2 capability reasoning."""
import os, sys, re, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from arms import ARM_NAMES, NDCG_COLS

MODEL_PATH = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
DATA_DIR = "experiments/xdomain/data"
RUN_TAG = "xc_fin3_l10"
OUT = "experiments/reasoning"
os.makedirs(OUT, exist_ok=True)

DESC = """You will choose the best retrieval method for a user question from five options:

- BM (BM25): Lexical keyword matching retrieval. Strong for exact entity names, rare terms, and precise string matches; weak for semantic paraphrase.
- TD (Text-Dense): Single-vector dense retrieval over text. Strong for semantic similarity and paraphrased questions; weaker on exact keyword and fine-grained multi-condition matching.
- TR (Text-Rerank): Text-dense recall followed by token-level late-interaction reranking. Strong for fine-grained multi-condition and relevance discrimination within text; does not use visual information.
- MD (MM-Dense): Single-vector dense retrieval over multimodal page representations (text + image regions). Strong when a page's visual layout or image content carries the answer; pages may also contain figure captions as text.
- MR (MM-Rerank): Multimodal-dense recall followed by visual late-interaction reranking. Strongest for queries that require reading figures, tables, or layout; text pages with captions are also handled, but the visual rerank step adds cost.

Note: Text-only pages may include figure captions. Do not assume a non-MR retriever cannot find an answer that happens to be near a figure."""

SYS = "You are a retrieval router that selects the best retrieval method for a question."

def build_prompt(query, mode):
    if mode == "P0":
        user = f"{DESC}\n\nQuestion: {query}\n\nOutput exactly one line in the form:\nFINAL=<BM|TD|TR|MD|MR>"
    elif mode == "P1":
        user = f"{DESC}\n\nQuestion: {query}\n\nBriefly reason in 1-3 sentences about what kind of retrieval this question needs, then choose. End your answer with exactly one line:\nFINAL=<BM|TD|TR|MD|MR>"
    else:
        user = (f"{DESC}\n\nQuestion: {query}\n\n"
                "Analyse by filling the five fields. Keep under 150 words total.\n"
                "Task: <pick relevant labels from: fact lookup, numerical computation, constrained comparison, explicit visual request, other>\n"
                "Constraints: <exact entities, years, numbers, relations, multi-condition needs>\n"
                "Matching Need: <pick the most relevant: lexical exact match, semantic recall, token-level fine-grained match, page visual representation>\n"
                "Comparison: <name at least two competing retrievers and state their capability difference for this query; do NOT claim to know where evidence lives>\n"
                "FINAL=<BM|TD|TR|MD|MR>")
    return [{"role":"system","content":SYS},{"role":"user","content":user}]

SHORT_TO_FULL = {"BM":"BM25","TD":"TEXT-SINGLE","TR":"TEXT_RERANK","MD":"MULTIMODAL-SINGLE","MR":"MULTIMODAL_RERANK"}
def parse_arm(text):
    matches = list(re.finditer(r"FINAL\s*[=:]\s*(BM|TD|TR|MD|MR)", text, re.I))
    return SHORT_TO_FULL[matches[-1].group(1).upper()] if matches else None

@torch.no_grad()
def batch_generate(model, tok, queries, mode, max_new, batch_size=16):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        texts = [tok.apply_chat_template(build_prompt(q, mode), tokenize=False, add_generation_prompt=True) for q in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=0.0,
                             top_p=1.0, pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            resp = tok.decode(gen, skip_special_tokens=True)
            results.append(resp)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["val","tatdqa","wiki-ss","arxivqa"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu", type=str, default="0")
    args = ap.parse_args()

    torch.cuda.set_device(int(args.gpu))
    print(f"Loading {MODEL_PATH} on cuda:{args.gpu}")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(f"cuda:{args.gpu}")
    model.eval()

    all_rows = []
    for ds in args.datasets:
        if ds == "val":
            path = os.path.join(DATA_DIR, f"val_{RUN_TAG}.xlsx")
        else:
            path = os.path.join(DATA_DIR, f"target_{ds}_{RUN_TAG}.xlsx")
        df = pd.read_excel(path)
        if args.limit:
            df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)
        queries = df["query"].tolist()
        ndcg_mat = df[NDCG_COLS].values.astype(float)
        print(f"\n=== {ds}: n={len(df)} ===")

        for mode, max_new in [("P0", 32), ("P1", 256), ("P2", 400)]:
            t0 = time.time()
            resps = batch_generate(model, tok, queries, mode, max_new, args.batch_size)
            dt = time.time() - t0
            arms = [parse_arm(r) for r in resps]
            parse_ok = [a is not None for a in arms]
            print(f"  {mode}: {dt:.1f}s ({dt/len(queries):.2f}s/q), parse_ok={sum(parse_ok)}/{len(queries)}")
            for i, (arm, resp) in enumerate(zip(arms, resps)):
                row = all_rows[i] if i < len(all_rows) else None
                if row is None:
                    row = {"query_id": i, "dataset": ds, "query": queries[i][:300]}
                    for k, a in enumerate(ARM_NAMES):
                        row[f"ndcg_{a}"] = ndcg_mat[i, k]
                    row["mr_ndcg"] = ndcg_mat[i, ARM_NAMES.index("MULTIMODAL_RERANK")]
                    row["oracle_ndcg"] = ndcg_mat[i].max()
                    all_rows.append(row)
                row[f"{mode}_arm"] = arm
                row[f"{mode}_ok"] = arm is not None
                if arm is not None:
                    row[f"{mode}_ndcg"] = ndcg_mat[i, ARM_NAMES.index(arm)]
                else:
                    row[f"{mode}_ndcg"] = np.nan
                row[f"{mode}_tokens"] = len(resp.split())
                row[f"{mode}_resp"] = resp[:500]

    out_df = pd.DataFrame(all_rows)
    ds_tag = "_".join(args.datasets)
    out_path = os.path.join(OUT, f"reasoning_results_{ds_tag}.json")
    out_df.to_json(out_path, orient="records", force_ascii=False)
    out_df.to_csv(os.path.join(OUT, f"reasoning_results_{ds_tag}.csv"), index=False)
    print(f"\nSaved {len(out_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
