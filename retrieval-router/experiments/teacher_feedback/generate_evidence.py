"""Generate Top-5 retrieval ranking evidence for source-domain queries."""
import os, sys, json, time, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from arms import ARM_NAMES, NDCG_COLS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB_ROOT = os.path.join(REPO, "analysis", "replay_data", "emb")
CORPUS = os.path.join(REPO, "analysis", "replay_data", "corpus")
MR = "MULTIMODAL_RERANK"
OUT_JSON = os.path.join(os.path.dirname(__file__), "evidence_generated.json")

def ndcg_at_k(ranked, gt_set, k=5):
    dcg = sum(1.0/np.log2(i+2) for i,fn in enumerate(ranked[:k]) if fn in gt_set)
    ideal = min(len(gt_set), k)
    idcg = sum(1.0/np.log2(i+2) for i in range(ideal))
    return dcg/idcg if idcg > 0 else 0.0

def cosine_topk(qv, dv, fns, k):
    q = qv/(np.linalg.norm(qv)+1e-12)
    d = dv/(np.linalg.norm(dv, axis=1, keepdims=True)+1e-12)
    scores = d @ q
    order = np.argsort(-scores, kind="stable")[:k]
    return [(fns[i], float(scores[i])) for i in order]

def load_emb(ds, name):
    z = np.load(os.path.join(EMB_ROOT, ds, f"{name}.npz"), allow_pickle=True)
    return z["filenames"], z["vectors"]

def build_sample():
    qrels = {}
    for ds in ["finqa","vqaonbd"]:
        qrels.update(json.load(open(os.path.join(CORPUS, ds, "qrels.json"))))
    df = pd.read_excel(os.path.join(REPO, "train","data","updated_dataset.xlsx"))
    src = df[df["DATASET"].isin(["finqa","vqaonbd"])].copy()
    src["qnorm"] = src["query"].str.strip()
    src = src[src["qnorm"].isin(qrels.keys())].copy()
    mr_idx = ARM_NAMES.index(MR)
    ndcg_mat = src[NDCG_COLS].values.astype(float)
    src["mr_ndcg"] = ndcg_mat[:, mr_idx]
    non_mr = np.delete(ndcg_mat, mr_idx, axis=1)
    non_arms = [a for i,a in enumerate(ARM_NAMES) if i != mr_idx]
    src["best_nonmr_ndcg"] = non_mr.max(1)
    src["best_nonmr_arm"] = [non_arms[i] for i in non_mr.argmax(1)]
    src["delta"] = src["best_nonmr_ndcg"] - src["mr_ndcg"]
    samples = []
    for label, cond in [("MR_win", src["delta"] <= -0.15),
                         ("nonMR_win", src["delta"] >= 0.15),
                         ("tie", src["delta"].abs() <= 0.05)]:
        pool = src[cond]
        for ds in ["finqa","vqaonbd"]:
            sub = pool[pool["DATASET"]==ds]
            n = min(20, len(sub))
            samples.append(sub.sample(n=n, random_state=42))
    sample_df = pd.concat(samples, ignore_index=True)
    results = []
    for _, row in sample_df.iterrows():
        q = row["qnorm"]; ds = row["DATASET"]; gt = set(qrels[q])
        results.append({"query": q, "dataset": ds, "gt_pages": list(gt),
            "mr_ndcg": float(row["mr_ndcg"]), "best_nonmr_arm": row["best_nonmr_arm"],
            "best_nonmr_ndcg": float(row["best_nonmr_ndcg"]), "delta": float(row["delta"]),
            "stratum": label})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all","md","td","mr","tr"])
    args = ap.parse_args()
    device = "cuda:0"
    torch.cuda.set_device(0)

    print("Loading embeddings...")
    md_fns={}; md_vecs={}; td_fns={}; td_vecs={}
    mr_fns={}; mr_vecs={}; tr_fns={}; tr_vecs={}
    for ds in ["finqa","vqaonbd"]:
        f,v = load_emb(ds,"MD_dense"); md_fns[ds]=f; md_vecs[ds]=v.astype(np.float32)
        try:
            f,v = load_emb(ds,"TD_dense"); td_fns[ds]=f; td_vecs[ds]=v.astype(np.float32)
        except: pass
        try:
            f,v = load_emb(ds,"MR_multi"); mr_fns[ds]=set(f); mr_vecs[ds]={fn:vec for fn,vec in zip(f,v)}
        except: pass
        try:
            f,v = load_emb(ds,"TR_multi"); tr_fns[ds]=set(f); tr_vecs[ds]={fn:vec for fn,vec in zip(f,v)}
        except: pass

    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else build_sample()
    print(f"Working on {len(results)} queries")
    queries = [r["query"] for r in results]

    def maxsim(qv, dv):
        out = np.empty(len(dv))
        for i,p in enumerate(dv):
            out[i] = (qv @ p.T).max(axis=1).sum()
        return out

    need = lambda p: args.phase in ("all", p)
    save = lambda: json.dump(results, open(OUT_JSON,"w"), indent=2)

    if need("md") and not any("MD_dense_top5" in r for r in results):
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        print("Loading MD model...")
        model = BiQwen2_5.from_pretrained("nomic-ai/nomic-embed-multimodal-7b",
            torch_dtype=torch.bfloat16, device_map=device).eval()
        proc = BiQwen2_5_Processor.from_pretrained("nomic-ai/nomic-embed-multimodal-7b")
        qvs = []
        with torch.inference_mode():
            for q in queries:
                inp = proc.process_queries([q]).to(device)
                qvs.append(model(**inp).float().cpu().numpy()[0])
        del model, proc; gc.collect(); torch.cuda.empty_cache()
        for i,r in enumerate(results):
            ds=r["dataset"]; gt=set(r["gt_pages"])
            t5 = cosine_topk(qvs[i], md_vecs[ds], md_fns[ds], 5)
            r["MD_dense_top5"]=[{"filename":fn,"score":s,"gt":fn in gt} for fn,s in t5]
            r["MD_dense_ndcg5"]=ndcg_at_k([fn for fn,_ in t5], gt)
        save(); print(f"  MD done ({len(results)})")

    if need("td") and not any("TD_dense_top5" in r for r in results if r["dataset"] in td_vecs):
        from sentence_transformers import SentenceTransformer
        ti = [i for i,r in enumerate(results) if r["dataset"] in td_vecs]
        if ti:
            print("Loading TD model...")
            model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral",
                model_kwargs={"torch_dtype": torch.bfloat16}).to(device)
            prompt = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
            qvs = model.encode([queries[i] for i in ti], prompt=prompt)
            del model; gc.collect(); torch.cuda.empty_cache()
            for j,i in enumerate(ti):
                r=results[i]; ds=r["dataset"]; gt=set(r["gt_pages"])
                t5 = cosine_topk(qvs[j], td_vecs[ds], td_fns[ds], 5)
                r["TD_dense_top5"]=[{"filename":fn,"score":s,"gt":fn in gt} for fn,s in t5]
                r["TD_dense_ndcg5"]=ndcg_at_k([fn for fn,_ in t5], gt)
            save(); print(f"  TD done ({len(ti)})")

    if need("mr") and not any("MR_rerank_top5" in r for r in results):
        mi = [i for i,r in enumerate(results)
              if "MD_dense_top5" in r
              and sum(1 for x in r["MD_dense_top5"] if x["filename"] in mr_fns.get(r["dataset"],set())) >= 2]
        if mi:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            print(f"Loading MR model for {len(mi)} queries...")
            model = ColQwen2_5.from_pretrained("nomic-ai/colnomic-embed-multimodal-7b",
                torch_dtype=torch.bfloat16, device_map=device).eval()
            proc = ColQwen2_5_Processor.from_pretrained("nomic-ai/colnomic-embed-multimodal-7b")
            with torch.inference_mode():
                for i in mi:
                    r=results[i]; ds=r["dataset"]; gt=set(r["gt_pages"])
                    cands=[x["filename"] for x in r["MD_dense_top5"] if x["filename"] in mr_fns[ds]]
                    inp=proc.process_queries([r["query"]]).to(device)
                    qv=model(**inp).float().cpu().numpy()[0]
                    dv=[mr_vecs[ds][fn] for fn in cands]
                    sc=maxsim(qv,dv); od=np.argsort(-sc)
                    t5=[(cands[k],float(sc[k])) for k in od][:5]
                    r["MR_rerank_top5"]=[{"filename":fn,"score":s,"gt":fn in gt} for fn,s in t5]
                    r["MR_rerank_ndcg5"]=ndcg_at_k([fn for fn,_ in t5], gt)
            del model,proc; gc.collect(); torch.cuda.empty_cache()
            save(); print(f"  MR done ({len(mi)})")

    if need("tr") and not any("TR_rerank_top5" in r for r in results):
        tri = [i for i,r in enumerate(results)
               if r["dataset"] in tr_fns and "TD_dense_top5" in r
               and sum(1 for x in r["TD_dense_top5"] if x["filename"] in tr_fns.get(r["dataset"],set())) >= 2]
        if tri:
            from pylate import models
            print(f"Loading TR model for {len(tri)} queries...")
            model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1", document_length=8192).to(device)
            for i in tri:
                r=results[i]; ds=r["dataset"]; gt=set(r["gt_pages"])
                cands=[x["filename"] for x in r["TD_dense_top5"] if x["filename"] in tr_fns[ds]]
                qv=np.asarray(model.encode(r["query"],is_query=True),dtype=np.float32)
                dv=[tr_vecs[ds][fn] for fn in cands]
                sc=maxsim(qv,dv); od=np.argsort(-sc)
                t5=[(cands[k],float(sc[k])) for k in od][:5]
                r["TR_rerank_top5"]=[{"filename":fn,"score":s,"gt":fn in gt} for fn,s in t5]
                r["TR_rerank_ndcg5"]=ndcg_at_k([fn for fn,_ in t5], gt)
            del model; gc.collect(); torch.cuda.empty_cache()
            save(); print(f"  TR done ({len(tri)})")

    for r in results:
        r["arms_with_evidence"] = [k.replace("_dense_top5","").replace("_rerank_top5","")
                                   for k in r if k.endswith("_top5")]
    save()
    arms_count = {}
    for r in results:
        for a in r["arms_with_evidence"]:
            arms_count[a] = arms_count.get(a,0)+1
    print(f"\nSaved {len(results)} cases. Evidence by arm: {arms_count}")

if __name__ == "__main__":
    main()
