"""Stage 2 data collection: dense Top-100 + rerank + GT labels.

For each query:
- MD/TD dense Top-100 with GT flags
- recall_hit_M / recall_hit_T: any GT in dense Top-100
- rerank_rescue_M / rerank_rescue_T: GT in Top100 but not dense Top5, in rerank Top5
- rerank_harm_M / rerank_harm_T: GT in dense Top5 but not rerank Top5
"""
import os, sys, json, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from train.config import ARM_NAMES, NDCG_COLS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMB = os.path.join(REPO, "analysis", "replay_data", "emb")
CORPUS = os.path.join(REPO, "analysis", "replay_data", "corpus")
OUT = os.path.join(os.path.dirname(__file__), "process_labels")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["finqa","vqaonbd","tatdqa","wiki-ss","arxivqa","mp-docvqa","sciqag","finreport"]
ARM_IDX = {a:i for i,a in enumerate(ARM_NAMES)}
MR_I=ARM_IDX["MULTIMODAL_RERANK"]; MD_I=ARM_IDX["MULTIMODAL-SINGLE"]
TR_I=ARM_IDX["TEXT_RERANK"]; TD_I=ARM_IDX["TEXT-SINGLE"]; BM_I=ARM_IDX["BM25"]

def ndcg_k(ranked, gt, k=5):
    dcg = sum(1.0/np.log2(i+2) for i,f in enumerate(ranked[:k]) if f in gt)
    ideal = min(len(gt), k)
    idcg = sum(1.0/np.log2(i+2) for i in range(ideal))
    return dcg/idcg if idcg>0 else 0.0

def cosine_topk(qv, dv, fns, k):
    q = qv/(np.linalg.norm(qv)+1e-12)
    d = dv/(np.linalg.norm(dv,axis=1,keepdims=True)+1e-12)
    sc = d@q
    od = np.argsort(-sc, kind="stable")[:k]
    return [(fns[i], float(sc[i])) for i in od]

def load_emb(ds, name):
    z = np.load(os.path.join(EMB, ds, f"{name}.npz"), allow_pickle=True)
    return z["filenames"], z["vectors"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all","md","td","mr","tr","label"])
    args = ap.parse_args()
    device = "cuda:0"; torch.cuda.set_device(0)

    # Load qrels
    qrels = {}
    for ds in DATASETS:
        p = os.path.join(CORPUS, ds, "qrels.json")
        if os.path.exists(p):
            for q, gt in json.load(open(p)).items():
                qrels[q] = gt
    print(f"qrels: {len(qrels)} queries")

    # Load training data, match qrels
    df = pd.read_excel("train/data/updated_dataset.xlsx")
    df["qnorm"] = df["query"].str.strip()
    src = df[df["qnorm"].isin(qrels.keys()) & df["DATASET"].isin(DATASETS)].copy()
    print(f"Matched queries in training data: {len(src)}")

    # Stratify: MR win / nonMR win, rerank pos/neg
    ndcg_mat = src[NDCG_COLS].values.astype(float)
    src["mr"] = ndcg_mat[:, MR_I]
    src["md"] = ndcg_mat[:, MD_I]
    src["td"] = ndcg_mat[:, TD_I]
    src["tr"] = ndcg_mat[:, TR_I]
    src["delta_M"] = src["mr"] - src["md"]
    src["delta_T"] = src["tr"] - src["td"]
    src["nonmr_best"] = np.delete(ndcg_mat, MR_I, 1).max(1)
    src["delta"] = src["nonmr_best"] - src["mr"]

    # Sample ~100 per dataset, stratified by delta_M sign
    samples = []
    for ds in DATASETS:
        sub = src[src["DATASET"]==ds]
        if len(sub) == 0: continue
        for cond in [sub["delta_M"]>=0.1, sub["delta_M"]<=-0.1, sub["delta_M"].abs()<0.1]:
            pool = sub[cond]
            n = min(40, len(pool))
            if n > 0:
                samples.append(pool.sample(n=n, random_state=42))
    sample_df = pd.concat(samples, ignore_index=True)
    print(f"Sampled {len(sample_df)} queries")

    records = []
    for _, row in sample_df.iterrows():
        q = row["qnorm"]; ds = row["DATASET"]; gt = set(qrels[q])
        records.append({"query": q, "dataset": ds, "gt_pages": list(gt),
            "mr_ndcg": float(row["mr"]), "md_ndcg": float(row["md"]),
            "td_ndcg": float(row["td"]), "tr_ndcg": float(row["tr"]),
            "delta_M": float(row["delta_M"]), "delta_T": float(row["delta_T"]),
            "stratum": "MR_win" if row["delta"]<=-0.1 else ("nonMR_win" if row["delta"]>=0.1 else "tie")})

    out_json = os.path.join(OUT, "process_evidence.json")
    if os.path.exists(out_json):
        results = json.load(open(out_json))
    else:
        results = records
    queries = [r["query"] for r in results]
    print(f"Working on {len(results)} queries")

    # Load embeddings
    md_fns={}; md_vecs={}; td_fns={}; td_vecs={}
    mr_fns={}; mr_vecs={}; tr_fns={}; tr_vecs={}
    for ds in DATASETS:
        try:
            f,v = load_emb(ds,"MD_dense"); md_fns[ds]=f; md_vecs[ds]=v.astype(np.float32)
        except: pass
        try:
            f,v = load_emb(ds,"TD_dense"); td_fns[ds]=f; td_vecs[ds]=v.astype(np.float32)
        except: pass
        try:
            f,v = load_emb(ds,"MR_multi"); mr_fns[ds]=set(f); mr_vecs[ds]={fn:vec for fn,vec in zip(f,v)}
        except: pass
        try:
            f,v = load_emb(ds,"TR_multi"); tr_fns[ds]=set(f); tr_vecs[ds]={fn:vec for fn,vec in zip(f,v)}
        except: pass

    need = lambda p: args.phase in ("all", p)
    save = lambda: json.dump(results, open(out_json,"w"))

    def maxsim(qv, dv):
        out = np.empty(len(dv))
        for i,p in enumerate(dv):
            out[i] = (qv @ p.T).max(axis=1).sum()
        return out

    # Phase MD dense Top-100
    if need("md") and not any("MD_top100" in r for r in results):
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
            ds=r["dataset"]
            if ds not in md_vecs: continue
            gt=set(r["gt_pages"])
            t100 = cosine_topk(qvs[i], md_vecs[ds], md_fns[ds], 100)
            r["MD_top100"] = [{"filename":fn,"gt":fn in gt} for fn,_ in t100]
            r["MD_dense_ndcg5"] = ndcg_k([fn for fn,_ in t100], gt)
            r["recall_hit_M"] = any(fn in gt for fn,_ in t100)
        save(); print("  MD done")

    # Phase TD dense Top-100
    if need("td") and not any("TD_top100" in r for r in results if r["dataset"] in td_vecs):
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
                t100 = cosine_topk(qvs[j], td_vecs[ds], td_fns[ds], 100)
                r["TD_top100"] = [{"filename":fn,"gt":fn in gt} for fn,_ in t100]
                r["TD_dense_ndcg5"] = ndcg_k([fn for fn,_ in t100], gt)
                r["recall_hit_T"] = any(fn in gt for fn,_ in t100)
            save(); print("  TD done")

    # Phase MR rerank
    if need("mr") and not any("MR_rerank_top5" in r for r in results):
        mi = [i for i,r in enumerate(results)
              if "MD_top100" in r and r["dataset"] in mr_fns]
        if mi:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
            print(f"Loading MR model for {len(mi)} queries...")
            model = ColQwen2_5.from_pretrained("nomic-ai/colnomic-embed-multimodal-7b",
                torch_dtype=torch.bfloat16, device_map=device).eval()
            proc = ColQwen2_5_Processor.from_pretrained("nomic-ai/colnomic-embed-multimodal-7b")
            with torch.inference_mode():
                for i in mi:
                    r=results[i]; ds=r["dataset"]; gt=set(r["gt_pages"])
                    cands = [x["filename"] for x in r["MD_top100"] if x["filename"] in mr_fns[ds]]
                    if len(cands) < 5: continue
                    inp = proc.process_queries([r["query"]]).to(device)
                    qv = model(**inp).float().cpu().numpy()[0]
                    dv = [mr_vecs[ds][fn] for fn in cands]
                    sc = maxsim(qv, dv); od = np.argsort(-sc)
                    t5 = [cands[k] for k in od[:5]]
                    r["MR_rerank_top5"] = [{"filename":fn,"gt":fn in gt} for fn in t5]
                    r["MR_rerank_ndcg5"] = ndcg_k(t5, gt)
                    md_t5 = [x["filename"] for x in r["MD_top100"][:5]]
                    gt_in_top100 = [x["filename"] for x in r["MD_top100"] if x["gt"]]
                    gt_not_in_dense5 = [g for g in gt_in_top100 if g not in md_t5]
                    r["rerank_rescue_M"] = any(g in t5 for g in gt_not_in_dense5) if gt_not_in_dense5 else False
                    gt_in_dense5 = [g for g in gt_in_top100 if g in md_t5]
                    r["rerank_harm_M"] = any(g not in t5 for g in gt_in_dense5) if gt_in_dense5 else False
            del model, proc; gc.collect(); torch.cuda.empty_cache()
            save(); print("  MR done")

    # Phase TR rerank
    if need("tr") and not any("TR_rerank_top5" in r for r in results):
        tri = [i for i,r in enumerate(results)
               if "TD_top100" in r and r["dataset"] in tr_fns]
        if tri:
            from pylate import models
            print(f"Loading TR model for {len(tri)} queries...")
            model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1", document_length=8192).to(device)
            for i in tri:
                r=results[i]; ds=r["dataset"]; gt=set(r["gt_pages"])
                cands = [x["filename"] for x in r["TD_top100"] if x["filename"] in tr_fns[ds]]
                if len(cands) < 5: continue
                qv = np.asarray(model.encode(r["query"], is_query=True), dtype=np.float32)
                dv = [tr_vecs[ds][fn] for fn in cands]
                sc = maxsim(qv, dv); od = np.argsort(-sc)
                t5 = [cands[k] for k in od[:5]]
                r["TR_rerank_top5"] = [{"filename":fn,"gt":fn in gt} for fn in t5]
                r["TR_rerank_ndcg5"] = ndcg_k(t5, gt)
                td_t5 = [x["filename"] for x in r["TD_top100"][:5]]
                gt_in_top100 = [x["filename"] for x in r["TD_top100"] if x["gt"]]
                gt_not_in_dense5 = [g for g in gt_in_top100 if g not in td_t5]
                r["rerank_rescue_T"] = any(g in t5 for g in gt_not_in_dense5) if gt_not_in_dense5 else False
                gt_in_dense5 = [g for g in gt_in_top100 if g in td_t5]
                r["rerank_harm_T"] = any(g not in t5 for g in gt_in_dense5) if gt_in_dense5 else False
            del model; gc.collect(); torch.cuda.empty_cache()
            save(); print("  TR done")

    # Phase: compute final labels
    if need("label"):
        for r in results:
            r["recall_hit_M"] = r.get("recall_hit_M", False)
            r["recall_hit_T"] = r.get("recall_hit_T", False)
            r["rerank_rescue_M"] = r.get("rerank_rescue_M", False)
            r["rerank_harm_M"] = r.get("rerank_harm_M", False)
            r["rerank_rescue_T"] = r.get("rerank_rescue_T", False)
            r["rerank_harm_T"] = r.get("rerank_harm_T", False)
        save()

    # Summary
    n = len(results)
    has_md = sum("MD_top100" in r for r in results)
    has_td = sum("TD_top100" in r for r in results)
    has_mr = sum("MR_rerank_top5" in r for r in results)
    has_tr = sum("TR_rerank_top5" in r for r in results)
    print(f"\nTotal: {n}")
    print(f"MD Top100: {has_md}, TD Top100: {has_td}, MR rerank: {has_mr}, TR rerank: {has_tr}")
    print(f"recall_hit_M: {sum(r.get('recall_hit_M',False) for r in results)}")
    print(f"recall_hit_T: {sum(r.get('recall_hit_T',False) for r in results)}")
    print(f"rerank_rescue_M: {sum(r.get('rerank_rescue_M',False) for r in results)}")
    print(f"rerank_harm_M: {sum(r.get('rerank_harm_M',False) for r in results)}")

if __name__ == "__main__":
    main()
