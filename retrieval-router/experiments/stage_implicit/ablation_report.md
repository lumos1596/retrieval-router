# Ablation Report: Stage-Dependency, Label Decomposition & Process Supervision

## Important: Training Config Correction

Initial results used a **fixed LR (no scheduler, no warmup, no grad accumulation)** which
under-trained all models. All results below use the **correct config matching the original paper**:
cosine LR schedule + 10% warmup + gradient_accumulation=2 (effective batch 32).

M0 corrected: **0.644 → 0.733** (+0.089 from config alone).

## Stage 1 Results (cosine config)

| Model | Description | Pooled nDCG@5 | net_vs_MR |
|-------|-------------|---------------|-----------|
| M0 | Original KL (5-class) | 0.733 | +0.000 |
| M1 | KL + 5-arm utility regression | 0.750 | +0.017 |
| M2 | Stage-latent (stage1→stage2) | 0.754 | +0.021 |
| M2-parallel | Same labels, parallel heads (no dependency) | 0.753 | +0.021 |

## Stage 2 Results (cosine config) — Process Supervision

| Model | Description | Pooled nDCG@5 | net_vs_MR |
|-------|-------------|---------------|-----------|
| M2 | (baseline, no process labels) | 0.754 | +0.021 |
| M3 | M2 + process labels (recall→rescue/harm serial) | 0.755 | +0.022 |
| M3-flat | M2 + process labels (parallel heads) | 0.754 | +0.021 |
| M3-shuffle | M3 with shuffled process labels | 0.755 | +0.022 |

Process labels: recall_hit_M/T, rerank_rescue_M/T, rerank_harm_M/T (960 labeled queries).

## Key Findings

### 1. Stage Dependency Does NOT Add Value
M2-parallel (0.753) ≈ M2 (0.754). The serial structure (stage1→stage2) provides no benefit.

### 2. Label Decomposition DOES Add Value
M2 (0.754) > M1 (0.750) > M0 (0.733). The gain comes from splitting 5 absolute scores into
3 base scores (BM/TD/MD) + 2 small deltas (Δ_T, Δ_M). Smaller-range regression targets are easier to fit.

### 3. Process Supervision Does NOT Add Value
- M3 (0.755) ≈ M2 (0.754): adding recall/rescue/harm labels gives no measurable gain.
- M3-shuffle (0.755) ≈ M3 (0.755): **shuffled labels work just as well**, proving the auxiliary
  loss is only a regularizer, not genuine process knowledge.
- M3-flat (0.754) ≈ M3 (0.755): serial process heads (recall→rescue/harm) offer no advantage
  over parallel heads.

## Delta Bucket Analysis (M1 vs M2, cosine config)

| Bucket | n | M1 nDCG | M1 %MR | M2 nDCG | M2 %MR |
|--------|---|---------|--------|---------|--------|
| Δ_M > 0.1 (MR should win) | 2625 | 0.796 | 64.9% | 0.791 | 60.3% |
| Δ_M < -0.1 (MR should lose) | 1862 | 0.517 | 52.8% | 0.542 | 46.7% |
| Δ_T > 0.1 (TR should win) | 2946 | 0.821 | 62.4% | 0.826 | 56.9% |
| Δ_T < -0.1 (TR should lose) | 1654 | 0.669 | 50.4% | 0.678 | 42.4% |
| \|Δ_M\| ≤ 0.05 (tie) | 3972 | 0.832 | 57.9% | 0.831 | 50.2% |

### Mechanism (cosine config)
With proper training, **both M1 and M2 pick MR frequently (50-65%)**. The difference is subtle:
- **M1**: More aggressive on MR-win buckets (64.9% vs 60.3%) but also more reckless on MR-lose buckets (52.8% vs 46.7%).
- **M2**: Slightly more balanced — picks MR a bit less when it loses, at the cost of picking it slightly less when it wins.
- **Net effect**: M2's better calibration on MR-lose buckets (+0.025) outweighs its slight underperformance on MR-win buckets (-0.005).

## Overall Conclusion

Under the correct training config:

1. **Training config matters more than architecture**: M0 went from 0.644 to 0.733 just by
   switching to cosine LR + warmup + grad accumulation.
2. **Label decomposition (base + delta) is the only architectural change that helps**: M2 > M1 > M0.
3. **Stage dependency is a red herring**: M2 ≈ M2-parallel.
4. **Process supervision adds nothing**: M3 ≈ M3-flat ≈ M3-shuffle ≈ M2. The retrieval process
   labels (recall/rescue/harm) provide no information beyond what the nDCG decomposition already captures.

**The original "staged implicit reasoning" hypothesis is not supported.** The gains come from
numerically easier regression targets (base + delta), not from any latent reasoning or process knowledge.
