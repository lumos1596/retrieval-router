# Reasoning Router Metrics Report

**Model**: Qwen2.5-3B-Instruct (local), temperature=0. P0/P1/P2 share identical 5-arm descriptions.

## 1. Main nDCG (with B2 and FixedMR)

| Dataset | P0 | P1 | P2 | B2(0.6B) | FixedMR | Oracle |
|---|---|---|---|---|---|---|
| val | 0.7450 | 0.6334 | 0.6368 | 0.8490 | 0.8361 | 0.9455 |
| tatdqa | 0.3802 | 0.3651 | 0.3603 | 0.6482 | 0.6558 | 0.8426 |
| wiki-ss | 0.7073 | 0.7169 | 0.7068 | 0.7368 | 0.7315 | 0.9677 |
| arxivqa | 0.3636 | 0.4262 | 0.4110 | 0.6295 | 0.6269 | 0.7980 |

## 2. Regret, catastrophic rate, net vs MR

| Dataset | Model | Regret | Catastrophic | net_vs_MR |
|---|---|---|---|---|
| tatdqa | P0 | 0.4624 | 31.5% | -0.2756 |
| tatdqa | P1 | 0.4775 | 32.1% | -0.2907 |
| tatdqa | P2 | 0.4823 | 32.7% | -0.2955 |
| tatdqa | FixedMR | 0.1868 | 8.7% | +0.0000 |
| wiki-ss | P0 | 0.2603 | 16.4% | -0.0241 |
| wiki-ss | P1 | 0.2508 | 15.7% | -0.0146 |
| wiki-ss | P2 | 0.2609 | 15.9% | -0.0247 |
| wiki-ss | FixedMR | 0.2362 | 14.6% | +0.0000 |
| arxivqa | P0 | 0.4344 | 32.7% | -0.2633 |
| arxivqa | P1 | 0.3718 | 27.7% | -0.2007 |
| arxivqa | P2 | 0.3870 | 29.3% | -0.2159 |
| arxivqa | FixedMR | 0.1711 | 10.4% | +0.0000 |

## 3. MR selection rate

| Dataset | P0 MR | P1 MR | P2 MR | P2 top arm |
|---|---|---|---|---|
| tatdqa | 0.1% | 0.4% | 1.0% | BM (59%) |
| wiki-ss | 0.2% | 3.0% | 3.8% | TD (52%) |
| arxivqa | 1.3% | 23.1% | 17.9% | MD (39%) |

## 4. Stratified (NonMR-advantage: best_nonMR-MR>0.1)

| Dataset | Model | nonMR_sel | nDCG | net_vs_MR |
|---|---|---|---|---|
| tatdqa | P0 | 99.9% | 0.4352 | +0.1742 |
| tatdqa | P1 | 99.5% | 0.4007 | +0.1397 |
| tatdqa | P2 | 99.2% | 0.3922 | +0.1312 |
| wiki-ss | P0 | 99.8% | 0.7836 | +0.4997 |
| wiki-ss | P1 | 96.6% | 0.7431 | +0.4592 |
| wiki-ss | P2 | 95.6% | 0.7275 | +0.4436 |
| arxivqa | P0 | 99.1% | 0.5512 | +0.3767 |
| arxivqa | P1 | 79.2% | 0.4456 | +0.2710 |
| arxivqa | P2 | 86.0% | 0.4592 | +0.2846 |

### MR-advantage (MR-best_nonMR>0.1)

| Dataset | Model | nonMR_sel | nDCG | net_vs_MR |
|---|---|---|---|---|
| tatdqa | P0 | 99.5% | 0.0732 | -0.7898 |
| tatdqa | P1 | 99.7% | 0.0794 | -0.7836 |
| tatdqa | P2 | 98.2% | 0.0842 | -0.7788 |
| wiki-ss | P0 | 100.0% | 0.2245 | -0.6639 |
| wiki-ss | P1 | 93.8% | 0.2785 | -0.6099 |
| wiki-ss | P2 | 96.9% | 0.2538 | -0.6346 |
| arxivqa | P0 | 98.4% | 0.0875 | -0.7972 |
| arxivqa | P1 | 69.6% | 0.3282 | -0.5564 |
| arxivqa | P2 | 79.3% | 0.2412 | -0.6434 |

## 5. Paired bootstrap 95% CI (P2 - baseline)

| Dataset | P2-P0 [CI] | P2-P1 [CI] |
|---|---|---|
| tatdqa | -0.0199 [-0.0325,-0.0062] | -0.0048 [-0.0160,+0.0061] |
| wiki-ss | -0.0006 [-0.0155,+0.0140] | -0.0101 [-0.0257,+0.0063] |
| arxivqa | +0.0474 [+0.0208,+0.0726] | -0.0152 [-0.0470,+0.0161] |

## 6. Cost (avg output tokens)

| Dataset | P0 | P1 | P2 | P2/P0 |
|---|---|---|---|---|
| tatdqa | 1 | 53 | 63 | 63x |
| wiki-ss | 1 | 54 | 47 | 47x |
| arxivqa | 1 | 49 | 51 | 51x |
