"""Audit teacher feedback: fact-checkable, unsupported, distinguishing, uncertainty."""
import os, sys, re
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from arms import ARM_NAMES, SHORT_NAMES

CASES = "experiments/teacher_feedback/teacher_feedback_cases.csv"
OUT = "experiments/teacher_feedback"

FIELDS = ["Query signals", "Observed retrieval evidence", "Arm comparison", "Uncertainty"]

def parse_fields(text):
    """Extract the 4 fields from model response."""
    result = {}
    text = str(text)
    # Try label: value or label- value patterns
    for field in FIELDS:
        # Match "Field:" or "Field -" at start of line
        pat = rf'{re.escape(field)}\s*[:\-]\s*(.+?)(?=\n\s*(?:Query signals|Observed retrieval evidence|Arm comparison|Uncertainty)\s*[:\-]|$)'
        m = re.search(pat, text, re.DOTALL)
        if m:
            result[field] = m.group(1).strip()
        else:
            result[field] = ""
    return result

def audit_case(row, mode):
    """Return audit dict for one case in one mode."""
    fields = parse_fields(row[f"{mode}_resp"])
    ev_text = fields.get("Observed retrieval evidence", "")
    comp = fields.get("Arm comparison", "")
    unc = fields.get("Uncertainty", "")
    signals = fields.get("Query signals", "")

    # 1. Fact-checkable: cites specific nDCG/rank/GT that match provided info
    has_ndcg = bool(re.search(r'\d\.\d{2,}', ev_text + comp))  # has decimal numbers
    has_rank = bool(re.search(r'\b(rank|top-?5|top 5|position)\b', ev_text, re.I)) or bool(re.search(r'\b(1\.|2\.|3\.|4\.|5\.)', ev_text))
    has_gt = bool(re.search(r'\b(GT|ground.?truth|relevant|hit)\b', ev_text, re.I))
    has_unavailable = bool(re.search(r'unavailable', ev_text, re.I))
    fact_checkable = (has_ndcg or has_rank or has_gt) and not has_unavailable

    # 2. Unsupported assertion: causal claims with "because/since/due to" not backed by evidence
    causal_words = re.findall(r'\b(because|since|due to|therefore|thus|hence|as a result)\b', comp, re.I)
    # If mode is T0 (no evidence given), any causal claim is unsupported
    # If mode is T1 (only nDCG), causal claims about WHY are unsupported
    # If mode is T2 with evidence, causal claims referencing specific evidence may be supported
    unsupported = 0
    if mode == "T0":
        unsupported = len(causal_words)
    elif mode == "T1":
        # In T1, only nDCG is given. Causal claims about document content/ranking are unsupported.
        unsupported = len(causal_words)
    else:  # T2
        # Check if causal claims reference specific evidence (rank/GT)
        for cw in causal_words:
            # find context around causal word
            idx = comp.lower().find(cw.lower())
            ctx = comp[max(0,idx-50):idx+80]
            if not (re.search(r'\b(rank|GT|top.?5|page|ndcg|score)\b', ctx, re.I)):
                unsupported += 1

    # 3. Distinguishing explanation: compares MR vs other arm with capability-specific reasoning
    mentions_mr = bool(re.search(r'\bMR\b', comp))
    mentions_other = bool(re.search(r'\b(MD|TD|TR|BM)\b', comp))
    mentions_capability = bool(re.search(r'\b(visual|figure|table|layout|lexical|keyword|semantic|paraphrase|dense|rerank|multimodal|text)\b', comp, re.I))
    distinguishing = mentions_mr and mentions_other and mentions_capability

    # 4. Reasonable uncertainty
    mentions_uncertain = bool(re.search(r'\b(uncertain|not enough|insufficient|cannot|can\'t|unclear|limited|unavailable|hypothesi|speculat)\b', unc, re.I))
    reasonable_uncertainty = mentions_uncertain

    return {
        "query_id": row["query_id"],
        "mode": mode,
        "stratum": row["stratum"],
        "has_evidence": row["has_evidence"],
        "fact_checkable": int(fact_checkable),
        "n_unsupported": unsupported,
        "has_unsupported": int(unsupported > 0),
        "distinguishing": int(distinguishing),
        "reasonable_uncertainty": int(reasonable_uncertainty),
        "fields_parsed": int(all(v for v in fields.values())),
    }

def main():
    df = pd.read_csv(CASES)
    print(f"Loaded {len(df)} cases")

    # Sample 60 for audit (20 per stratum), fixed seed
    rng = np.random.RandomState(123)
    audit_ids = []
    for s in ["MR_win","nonMR_win","tie"]:
        pool = df[df["stratum"]==s]
        audit_ids.extend(pool.sample(n=min(20,len(pool)), random_state=123)["query_id"].tolist())
    audit_df = df[df["query_id"].isin(audit_ids)].copy()
    print(f"Audit sample: {len(audit_df)} cases")

    # Audit all modes
    audit_rows = []
    for _, row in audit_df.iterrows():
        for mode in ["T0","T1","T2"]:
            audit_rows.append(audit_case(row, mode))
    audit = pd.DataFrame(audit_rows)

    # Aggregate by mode
    agg = audit.groupby("mode").agg(
        n=("fact_checkable","count"),
        fact_checkable=("fact_checkable","mean"),
        unsupported_rate=("has_unsupported","mean"),
        avg_unsupported=("n_unsupported","mean"),
        distinguishing=("distinguishing","mean"),
        reasonable_uncertainty=("reasonable_uncertainty","mean"),
        fields_parsed=("fields_parsed","mean"),
    ).round(3)
    print("\n=== Audit by mode (all 60) ===")
    print(agg.to_string())

    # For T2, separate evidence vs no-evidence
    t2 = audit[audit["mode"]=="T2"]
    t2_agg = t2.groupby("has_evidence").agg(
        n=("fact_checkable","count"),
        fact_checkable=("fact_checkable","mean"),
        unsupported_rate=("has_unsupported","mean"),
        distinguishing=("distinguishing","mean"),
        reasonable_uncertainty=("reasonable_uncertainty","mean"),
    ).round(3)
    print("\n=== T2: with vs without ranking evidence ===")
    print(t2_agg.to_string())

    # By stratum
    strat = audit.groupby(["mode","stratum"]).agg(
        fact_checkable=("fact_checkable","mean"),
        unsupported_rate=("has_unsupported","mean"),
        distinguishing=("distinguishing","mean"),
    ).round(3)
    print("\n=== By mode x stratum ===")
    print(strat.to_string())

    audit.to_csv(os.path.join(OUT, "teacher_reasoning_audit.csv"), index=False)
    print(f"\nSaved audit to {OUT}/teacher_reasoning_audit.csv")
    return df, audit

if __name__ == "__main__":
    main()
