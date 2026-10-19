"""Stage 1 of the replay: materialise, per HF dataset,

  replay_data/corpus/<DATASET>/pages.parquet   unique page units: filename, text, img_bytes
  replay_data/corpus/<DATASET>/qrels.json      query -> [relevant filenames] (dedup, binary)
  replay_data/corpus/<DATASET>/meta.json       counts and consistency checks

The HF rows are (query, page) relevance annotations: one row per QA pair, and
image_filename uniquely identifies one rendered page image (verified constant
image bytes + extracted text per filename). We reproduce the official indexer
semantics (data_collection/pipelines/indexing.py: one point per unique
image_filename; content = image for MM arms, text for TEXT arms).
"""
import glob
import io
import json
import os

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(REPO_ROOT, "analysis", "replay_data", "corpus")

# DATASET label in query_metrics.csv -> HF repo
HF_DATASETS = {
    "tatdqa": "emrekuruu/TATDQA",
    "arxivqa": "emrekuruu/ArxivQA",
    "wiki-ss": "emrekuruu/Wiki-ss",
    "sciqag": "emrekuruu/SciQAG",
    "mp-docvqa": "emrekuruu/MP-DocVQA",
    "finqa": "emrekuruu/FinQA",
    "finreport": "emrekuruu/FinReport",
    "vqaonbd": "emrekuruu/VQAonBD",
}


def main():
    cases = pd.read_csv(os.path.join(REPO_ROOT, "analysis", "cases_28.csv"))
    needed = set(cases["DATASET"])

    for label, repo in HF_DATASETS.items():
        if label not in needed:
            continue
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        pages_path = os.path.join(out_dir, "pages.parquet")
        qrels_path = os.path.join(out_dir, "qrels.json")

        print(f"\n=== {label} <- {repo} ===")
        local = snapshot_download(repo, repo_type="dataset", allow_patterns=["*.parquet"])
        frames = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(local, "**", "*.parquet"), recursive=True))]
        df = pd.concat(frames, ignore_index=True)
        print(f"rows: {len(df)}, columns: {list(df.columns)}")

        # Official indexer (indexing.py) embeds the FIRST row of each filename.
        # Many datasets also carry extra annotation rows whose image/text are
        # None (questions about other pages of the same document): those are not
        # real page content, so for content we take the first NON-NONE row.
        # Consistency counts below only compare rows that both carry content.
        img_key, txt_key = {}, {}
        none_rows = 0
        inconsistent_img = inconsistent_txt = 0
        for row in df.itertuples(index=False):
            fn = row.image_filename
            img = row.image
            raw = img["bytes"] if isinstance(img, dict) else None
            txt = row.text
            if raw is None and txt is None:
                none_rows += 1
            if fn not in img_key:
                img_key[fn] = raw
                txt_key[fn] = txt
            else:
                if raw is not None:
                    if img_key[fn] is not None and img_key[fn] != raw:
                        inconsistent_img += 1
                    elif img_key[fn] is None:
                        img_key[fn] = raw  # first row was an annotation-only row
                if txt is not None:
                    if txt_key[fn] is not None and txt_key[fn] != txt:
                        inconsistent_txt += 1
                    elif txt_key[fn] is None:
                        txt_key[fn] = txt

        pages = pd.DataFrame(
            [{"filename": fn, "text": txt_key[fn], "img_bytes": img_key[fn]}
             for fn in sorted(img_key)])
        pages.to_parquet(pages_path, index=False, compression="zstd")

        qrels = {}
        dup_qf = 0
        seen = set()
        for row in df.itertuples(index=False):
            q, fn = row.query, row.image_filename
            if q is None:
                continue
            if (q, fn) in seen:
                dup_qf += 1
                continue
            seen.add((q, fn))
            qrels.setdefault(q, []).append(fn)
        with open(qrels_path, "w") as f:
            json.dump(qrels, f)

        case_queries = set(cases.loc[cases["DATASET"] == label, "query"])
        missing = [q for q in case_queries if q not in qrels]
        pages_per_q = pd.Series({q: len(v) for q, v in qrels.items()})
        meta = {
            "hf_repo": repo,
            "n_rows": int(len(df)),
            "n_unique_pages": int(len(pages)),
            "n_queries": int(len(qrels)),
            "duplicate_query_filename_rows": int(dup_qf),
            "annotation_only_none_rows": int(none_rows),
            "real_conflicting_image_per_filename": int(inconsistent_img),
            "real_conflicting_text_per_filename": int(inconsistent_txt),
            "max_pages_per_query": int(pages_per_q.max()),
            "multi_page_queries": int((pages_per_q > 1).sum()),
            "n_case_queries": int(len(case_queries)),
            "case_queries_missing_in_hf": missing,
            "img_none_pages": int(pages["img_bytes"].isna().sum()),
            "txt_none_pages": int(pages["text"].isna().sum()),
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(json.dumps(meta, indent=2))
        assert not missing, f"selected queries not found in {label}!"


if __name__ == "__main__":
    main()
