# Training objective comparison report

## Main results

| Dataset | Model | nDCG | Regret | Catastrophic | vs FixedMR |
|---|---|---|---|---|---|
| source_test | FixedMR | 0.8565 | 0.0958 | 1.6% | +0.0000 |
| source_test | B1 | 0.8615 | 0.0908 | 1.4% | +0.0050 |
| source_test | B2 | 0.8645 | 0.0878 | 1.1% | +0.0080 |
| source_test | B0 | 0.8003 | 0.1521 | 4.7% | -0.0563 |
| tatdqa | FixedMR | 0.6558 | 0.1867 | 8.7% | +0.0000 |
| tatdqa | B1 | 0.6462 | 0.1962 | 9.5% | -0.0095 |
| tatdqa | B2 | 0.6483 | 0.1942 | 9.3% | -0.0075 |
| tatdqa | B0 | 0.5050 | 0.3374 | 20.8% | -0.1507 |
| wiki-ss | FixedMR | 0.7316 | 0.2360 | 14.6% | +0.0000 |
| wiki-ss | B1 | 0.7147 | 0.2530 | 15.8% | -0.0170 |
| wiki-ss | B2 | 0.7349 | 0.2328 | 13.4% | +0.0033 |
| wiki-ss | B0 | 0.6748 | 0.2929 | 19.1% | -0.0568 |
| arxivqa | FixedMR | 0.6295 | 0.1718 | 10.5% | +0.0000 |
| arxivqa | B1 | 0.5721 | 0.2292 | 14.7% | -0.0575 |
| arxivqa | B2 | 0.6295 | 0.1718 | 10.5% | +0.0000 |
| arxivqa | B0 | 0.4158 | 0.3855 | 27.6% | -0.2137 |

## Arm selection distribution

| Dataset | Model | MR | MD | TR | TD | BM |
|---|---|---|---|---|---|---|
| source_test | FixedMR | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| source_test | B1 | 86.4% | 12.3% | 1.1% | 0.2% | 0.1% |
| source_test | B2 | 80.0% | 19.0% | 0.7% | 0.2% | 0.1% |
| source_test | B0 | 43.1% | 0.1% | 52.4% | 0.0% | 4.4% |
| tatdqa | FixedMR | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| tatdqa | B1 | 88.2% | 10.6% | 1.0% | 0.1% | 0.1% |
| tatdqa | B2 | 91.3% | 7.8% | 0.6% | 0.3% | 0.0% |
| tatdqa | B0 | 36.1% | 0.0% | 56.9% | 0.1% | 6.9% |
| wiki-ss | FixedMR | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| wiki-ss | B1 | 44.4% | 31.2% | 7.6% | 0.3% | 16.6% |
| wiki-ss | B2 | 63.0% | 6.0% | 8.8% | 2.1% | 20.0% |
| wiki-ss | B0 | 5.2% | 65.8% | 8.0% | 0.5% | 20.5% |
| arxivqa | FixedMR | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| arxivqa | B1 | 73.4% | 16.3% | 6.2% | 0.0% | 4.1% |
| arxivqa | B2 | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| arxivqa | B0 | 25.5% | 0.9% | 39.5% | 17.5% | 16.6% |

## Conclusion

- 3-target macro nDCG: B0=0.5319, B1=0.6443, B2=0.6709, FixedMR=0.6723
- See badcases and margin_analysis for details.