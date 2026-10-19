import sys, os, re, time
sys.path.insert(0, '.')
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct"
print("Loading", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

DESC = """You will choose the best retrieval method for a user question from five options:

- BM (BM25): Lexical keyword matching retrieval. Strong for exact entity names, rare terms, and precise string matches; weak for semantic paraphrase.
- TD (Text-Dense): Single-vector dense retrieval over text. Strong for semantic similarity and paraphrased questions; weaker on exact keyword and fine-grained multi-condition matching.
- TR (Text-Rerank): Text-dense recall followed by token-level late-interaction reranking. Strong for fine-grained multi-condition and relevance discrimination within text; does not use visual information.
- MD (MM-Dense): Single-vector dense retrieval over multimodal page representations (text + image regions). Strong when a page's visual layout or image content carries the answer; pages may also contain figure captions as text.
- MR (MM-Rerank): Multimodal-dense recall followed by visual late-interaction reranking. Strongest for queries that require reading figures, tables, or layout; text pages with captions are also handled, but the visual rerank step adds cost.

Note: Text-only pages may include figure captions. Do not assume a non-MR retriever cannot find an answer that happens to be near a figure."""

SYS = "You are a retrieval router that selects the best retrieval method for a question."

def gen(query, mode):
    if mode == "P0":
        user = f"{DESC}\n\nQuestion: {query}\n\nOutput exactly one line in the form:\nFINAL=<BM|TD|TR|MD|MR>"
    elif mode == "P1":
        user = f"{DESC}\n\nQuestion: {query}\n\nBriefly reason in 1-3 sentences about what kind of retrieval this question needs, then choose. End your answer with exactly one line:\nFINAL=<BM|TD|TR|MD|MR>"
    else:
        user = f"{DESC}\n\nQuestion: {query}\n\nAnalyse the question by filling in the five fields below. Keep it concise (under 150 words total).\n\nTask: <fact lookup / numerical computation / constrained comparison / explicit visual request / other>\nConstraints: <exact entities, years, numbers, relations, multi-condition>\nMatching Need: <lexical exact match / semantic recall / token-level fine-grained match / page visual representation>\nComparison: <name at least two competing retrievers and state their capability difference for this query; do NOT claim to know where evidence lives>\nFINAL=<BM|TD|TR|MD|MR>"
    msgs = [{"role":"system","content":SYS},{"role":"user","content":user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=300, do_sample=False, temperature=0.0, top_p=1.0, pad_token_id=tok.eos_token_id)
    gen_ids = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True)

val = pd.read_excel("experiments/xdomain/data/val_xc_fin3_l10.xlsx")
queries = val["query"].tolist()[:5]
for q in queries:
    print("="*80)
    print("Q:", q[:120])
    for mode in ["P0","P1","P2"]:
        t0 = time.time()
        resp = gen(q, mode)
        dt = time.time()-t0
        m = re.search(r"FINAL\s*=\s*(BM|TD|TR|MD|MR)", resp, re.I)
        arm = m.group(1).upper() if m else "PARSE_FAIL"
        print(f"\n--- {mode} ({dt:.1f}s) -> {arm} ---")
        print(resp[:400])
