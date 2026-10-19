# Sanity Check Report

**Overall: PASS** (11/11 checks passed)

## Checks

- **[col_order_train]** PASS: 5/5 arm cols present, order correct
- **[col_order_val]** PASS: 5/5 arm cols present, order correct
- **[col_order_source_test]** PASS: 5/5 arm cols present, order correct
- **[col_order_tatdqa]** PASS: 5/5 arm cols present, order correct
- **[col_order_wiki-ss]** PASS: 5/5 arm cols present, order correct
- **[col_order_arxivqa]** PASS: 5/5 arm cols present, order correct
- **[B0_match_source_test]** PASS: nDCG new=0.8627 old=0.8619 diff=0.0008, pred_agree=0.991
- **[B0_match_tatdqa]** PASS: nDCG new=0.6190 old=0.6214 diff=-0.0025, pred_agree=0.989
- **[B0_match_wiki-ss]** PASS: nDCG new=0.6636 old=0.6658 diff=-0.0022, pred_agree=0.992
- **[B0_match_arxivqa]** PASS: nDCG new=0.5553 old=0.5592 diff=-0.0039, pred_agree=0.992
- **[fixed_mr_100pct]** PASS: FixedMR always selects MR by definition

## B0 old-vs-new comparison

| Dataset | new nDCG | old nDCG | diff |
|---|---|---|---|
| source_test | 0.8627 | 0.8619 | +0.0008 |
| tatdqa | 0.6190 | 0.6214 | -0.0025 |
| wiki-ss | 0.6636 | 0.6658 | -0.0022 |
| arxivqa | 0.5553 | 0.5592 | -0.0039 |

## Column order (load-bearing)

ARM_NAMES = ['MULTIMODAL_RERANK', 'MULTIMODAL-SINGLE', 'TEXT_RERANK', 'TEXT-SINGLE', 'BM25']
NDCG_COLS = ['MULTIMODAL_RERANK_ndcg', 'MULTIMODAL-SINGLE_ndcg', 'TEXT_RERANK_ndcg', 'TEXT-SINGLE_ndcg', 'BM25_ndcg']
MR index = 0 (MULTIMODAL_RERANK)

**Conclusion**: Arm order verified identical across data, model head, and decoding. B0 now matches old predictions after fixing LoRA adapter loading (previously `get_peft_model` was called before `PeftModel.from_pretrained`, preventing adapter weights from loading).