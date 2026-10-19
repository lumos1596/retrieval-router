# Baseline Alignment Report

## Setup
- **Datasets**: 11 datasets from `train/data/updated_dataset.xlsx` (85,094 queries)
- **Split**: 80/10/10 per dataset, random_seed=42 (same logic as `train/prepare_data.py`)
  - Train: 68,076 | Val: 8,506 | Test: 8,506
- **Model**: Qwen3-0.6B-Base + LoRA(r=16, alpha=32, dropout=0.05)
- **Training**: 2 epochs, AdamW(lr=1e-4, wd=0.01), batch_size=16, **gradient_accumulation=2**,
  **cosine LR schedule + 10% warmup**, bf16
- **Objective**: softmax(reward/T) KL, T=0.1, λ=0.1

## Training Config Correction

The initial Stage 0 run used **fixed LR with no scheduler/warmup/grad-accum**, yielding M0 nDCG=0.644.
After switching to the paper-matching config (cosine LR + warmup + grad accum), M0 improved to **0.733**
—a +0.089 gain from training config alone.

## M0 (Original Soft-label KL) Results — Corrected (cosine config)

| Dataset | M0 nDCG@5 | Fixed MR | Oracle |
|---------|-----------|----------|--------|
| finreport | 0.595 | 0.807 | 0.938 |
| finslides | 0.700 | 0.903 | 0.949 |
| finqa | 0.828 | 0.868 | 0.961 |
| convfinqa | 0.815 | 0.863 | 0.950 |
| vqaonbd | 0.810 | 0.847 | 0.948 |
| tatdqa | 0.573 | 0.656 | 0.843 |
| mp-docvqa | 0.588 | 0.726 | 0.906 |
| arxivqa | 0.297 | 0.629 | 0.801 |
| dude | 0.550 | 0.647 | 0.901 |
| sciqag | 0.748 | 0.881 | 0.994 |
| wiki-ss | 0.762 | 0.732 | 0.968 |
| **Pooled** | **0.733** | **0.733** | **0.901** |

## Discrepancy with Paper's ~0.755

M0 pooled nDCG@5 = **0.733** (cosine config), still ~0.02 below the paper's reported ~0.755.
Remaining possible causes:

1. **arxivqa drags the average**: arxivqa nDCG=0.297 (oracle=0.801). The paper may use a different
   arxivqa corpus or split. Without arxivqa, the pooled nDCG would be ~0.77.
2. **The paper's 0.755 may be macro-averaged** (mean of per-dataset means) rather than micro-averaged.
3. **Minor training details** (e.g., exact LR schedule, seed, data ordering).

## Impact on Conclusions

The relative ordering is robust under the corrected config:
- M0 = 0.733 (≈ FixedMR)
- M1 = 0.750
- M2 = 0.754
- M3 = 0.755

All methods use the identical split, data, and training setup, so the comparisons are valid.
