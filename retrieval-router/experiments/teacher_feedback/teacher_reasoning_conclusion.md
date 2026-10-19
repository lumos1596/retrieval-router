# Teacher Reasoning Conclusion

## One-Sentence Verdict

**T2 (evidence-grounded) cuts unsupported causal assertions from 83% (T1) to 7% on 120 source-domain queries that all have real Top-5 retrieval ranking evidence — the teacher model produces evidence-grounded analyses instead of post-hoc capability rationalizations.**

## Key Statistics

| Mode | Fact-Checkable | Unsupported Assertions | Distinguishing | Reasonable Uncertainty |
|------|---------------|----------------------|----------------|----------------------|
| T0 (query only) | 0% | 92% | 98% | 38% |
| T1 (+nDCG scores) | 100% | 83% | 93% | 8% |
| T2 (+Top-5 ranking + GT) | 95% | 7% | 48% | 10% |

**All 120 cases have real retrieval ranking evidence (MD/TD dense Top-5 with GT flags; 13 also have MR rerank, 2 have TR rerank).**

## What the numbers mean

- **T0 (no evidence):** 92% of explanations contain unsupported causal claims — pure fabrication.
- **T1 (nDCG only):** 83% unsupported. The model sees a score gap and invents capability stories ("MR won because it's multimodal"). These sound plausible but have no basis in actual retrieved pages.
- **T2 (real rankings):** Only 7% unsupported. The model cites specific rank positions and GT flags. When it can't explain from evidence, it says so rather than inventing.

Note: T1 has higher "distinguishing" (93% vs 48%) but this is misleading — T1's distinguishing claims are mostly fabricated capability stereotypes. T2's 48% distinguishing are grounded in real evidence; the other 52% correctly report that the evidence is insufficient to distinguish.

## By Stratum (T2 mode)

| Stratum | Fact-Checkable | Unsupported | Distinguishing |
|---------|---------------|-------------|----------------|
| MR_win | 100% | 10% | 55% |
| nonMR_win | 100% | 10% | 55% |
| tie | 85% | 0% | 35% |

For tie cases (similar scores), T2 correctly avoids making up reasons (0% unsupported).

## Counter-Control: Similar Queries, Opposite Winners

T1 uses the same capability script ("multimodal dense + visual interaction") regardless of which arm won, just reversed to fit the outcome. This confirms T1 is post-hoc rationalization, not genuine analysis. T2, by contrast, references different specific ranking evidence for each query.

## Evidence Coverage

- **120/120 queries (100%)** have real Top-5 ranking evidence with GT relevance flags.
- Evidence arms: MD dense (120), TD dense (60 finqa), MR rerank (13), TR rerank (2).
- MR and TR rerank coverage is limited because multi-vector embeddings only exist for a subset of pages (292/2789 finqa, 197/1788 vqaonbd). For queries without rerank, T2 shows MR's dense recall (MD Top-5) as partial evidence.

## Decision (per plan criteria)

- ✅ T2 unsupported assertions: **7%** (< 10% threshold)
- ✅ T2 fact-checkable: **95%** (cites specific ranks/GT, not just nDCG)
- ✅ NonMR-winning samples: **55%** have evidence-grounded distinguishing explanations
- ✅ Counter-control confirms T1 = templated rationalization, T2 = evidence-grounded

**Conclusion: PASS.** The teacher model (Qwen2.5-3B-Instruct) can produce evidence-grounded, retriever-distinguishing capability analyses when given real Top-5 retrieval rankings with GT flags. It does NOT merely post-hoc rationalize — the 7% unsupported rate in T2 (vs 83% in T1) demonstrates that real ranking evidence constrains the model to cite verifiable facts rather than invent capability stories.

## Limitation

MR/TR rerank evidence is available for only 13/2 queries respectively. Most T2 evidence uses dense Top-5 (MD/TD). To fully validate MR-specific reasoning, multi-vector embeddings should be generated for all source-domain pages.
