# 偏好迁移损失机制验证报告

## A1. 源域每类型五臂能力表

| 类型 | 源域总n | 最优臂 | 前两臂差 | 状态 |
|---|---|---|---|---|
| visual_explicit | 695 | MR | 0.0173 | weak_source_preference |
| comparison | 2157 | MR | 0.0463 | covered |
| computation_with_year | 11645 | MR | 0.0221 | covered |
| computation | 145 | MR | 0.0260 | weak_source_preference |
| fact_with_year | 888 | MR | 0.0242 | covered |
| fact_with_number | 0 | MR | nan | uncovered |
| fact_no_visual | 5 | BM | 0.0615 | uncovered |
| open | 14 | MD | 0.0083 | uncovered |

## A2. 偏好迁移损失 G(t,C)（交叉拟合，10种子平均）

| 目标库 | 类型 | n | 源域臂 | 目标最优臂 | 源策略nDCG | 目标Oracle | G | G_CI | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| tatdqa | visual_explicit | 114 | MR | MR | 0.6719 | 0.6609 | -0.0110 | [0.000,0.000] | weak_source_preference |
| tatdqa | comparison | 504 | MR | MR | 0.6727 | 0.6562 | -0.0165 | [0.000,0.000] | covered |
| tatdqa | computation_with_year | 1347 | MR | MR | 0.6634 | 0.6634 | +0.0000 | [0.000,0.000] | covered |
| tatdqa | computation | 286 | MR | MR | 0.6814 | 0.6814 | +0.0000 | [0.000,0.000] | weak_source_preference |
| tatdqa | fact_with_year | 202 | MR | MR | 0.5854 | 0.5854 | +0.0000 | [0.000,0.000] | covered |
| tatdqa | fact_no_visual | 169 | BM | MR | 0.3857 | 0.5950 | +0.2093 | [0.153,0.309] | uncovered |
| tatdqa | open | 90 | MD | MR | 0.5349 | 0.5525 | +0.0176 | [-0.070,0.157] | uncovered |
| wiki-ss | visual_explicit | 9 | MR | BM | 0.6347 | 0.7462 | +0.1114 | [-0.094,0.515] | weak_source_preference |
| wiki-ss | comparison | 64 | MR | MR | 0.7849 | 0.7543 | -0.0306 | [0.000,0.000] | covered |
| wiki-ss | computation_with_year | 6 | MR | TD | 0.2500 | 0.7833 | +0.5333 | [0.417,1.000] | covered |
| wiki-ss | computation | 178 | MR | TR | 0.7691 | 0.8182 | +0.0492 | [0.010,0.087] | weak_source_preference |
| wiki-ss | fact_with_year | 48 | MR | TR | 0.7167 | 0.7934 | +0.0767 | [0.006,0.151] | covered |
| wiki-ss | fact_no_visual | 1174 | BM | TR | 0.7546 | 0.7618 | +0.0072 | [-0.016,0.041] | uncovered |
| wiki-ss | open | 17 | MD | MD | 0.7772 | 0.6547 | -0.1225 | [0.000,0.000] | uncovered |
| arxivqa | visual_explicit | 591 | MR | MR | 0.6119 | 0.6119 | +0.0000 | [0.000,0.000] | weak_source_preference |
| arxivqa | comparison | 112 | MR | MR | 0.6677 | 0.6677 | +0.0000 | [0.000,0.000] | covered |
| arxivqa | computation_with_year | 1 | MR | MD | nan | nan | +nan | - | covered |
| arxivqa | computation | 84 | MR | MR | 0.6573 | 0.6483 | -0.0090 | [0.000,0.000] | weak_source_preference |
| arxivqa | fact_no_visual | 113 | BM | MR | 0.3008 | 0.6618 | +0.3610 | [0.251,0.474] | uncovered |
| arxivqa | open | 2 | MD | MR | nan | nan | +nan | [0.000,0.000] | uncovered |

## A3. 架构差值变化（源域->目标域）

| 类型 | 差值 | 目标库 | 源域均值 | 目标均值 | 偏移 | 类型 |
|---|---|---|---|---|---|---|
| comparison | MR_MD | wiki-ss | +0.0494 | +0.1541 | +0.1047 | magnitude_change |
| comparison | MR_MD | arxivqa | +0.0494 | +0.1449 | +0.0955 | magnitude_change |
| comparison | TR_TD | arxivqa | +0.1330 | +0.0641 | -0.0689 | magnitude_change |
| comparison | MR_TR | tatdqa | +0.1219 | +0.2124 | +0.0905 | magnitude_change |
| comparison | MR_TR | wiki-ss | +0.1219 | +0.0116 | -0.1103 | magnitude_change |
| comparison | MR_TR | arxivqa | +0.1219 | +0.2614 | +0.1395 | magnitude_change |
| comparison | MD_TD | tatdqa | +0.2055 | +0.3413 | +0.1358 | magnitude_change |
| comparison | MD_TD | wiki-ss | +0.2055 | +0.0140 | -0.1915 | magnitude_change |
| computation_with_year | MR_MD | wiki-ss | +0.0253 | -0.5833 | -0.6086 | sign_flip |
| computation_with_year | TR_TD | wiki-ss | +0.1249 | -0.5000 | -0.6249 | sign_flip |
| computation_with_year | MR_TR | tatdqa | +0.0927 | +0.1917 | +0.0990 | magnitude_change |
| computation_with_year | MR_TR | wiki-ss | +0.0927 | -0.2500 | -0.3427 | sign_flip |
| computation_with_year | MD_TD | tatdqa | +0.1923 | +0.2519 | +0.0595 | magnitude_change |
| computation_with_year | MD_TD | wiki-ss | +0.1923 | -0.1667 | -0.3590 | sign_flip |
| fact_with_year | MR_MD | tatdqa | +0.0203 | +0.0791 | +0.0588 | magnitude_change |
| fact_with_year | MR_MD | wiki-ss | +0.0203 | +0.1825 | +0.1622 | magnitude_change |
| fact_with_year | TR_TD | wiki-ss | +0.1133 | +0.1840 | +0.0707 | magnitude_change |
| fact_with_year | MR_TR | tatdqa | +0.0923 | +0.2602 | +0.1680 | magnitude_change |
| fact_with_year | MR_TR | wiki-ss | +0.0923 | -0.0767 | -0.1690 | sign_flip |
| fact_with_year | MD_TD | tatdqa | +0.1852 | +0.2491 | +0.0639 | magnitude_change |
| fact_with_year | MD_TD | wiki-ss | +0.1852 | -0.0752 | -0.2604 | sign_flip |

## B. Router 错误与迁移损失

| 目标库 | 类型 | n | G | Router Regret | 选源域臂% | 源策略nDCG | 掉零率 |
|---|---|---|---|---|---|---|---|
| tatdqa | comparison | 504 | -0.0165 | 0.2206 | 73.2% | 0.6727 | 11.5% |
| tatdqa | computation_with_year | 1347 | +0.0000 | 0.2040 | 59.8% | 0.6634 | 10.5% |
| tatdqa | fact_with_year | 202 | +0.0000 | 0.2475 | 54.5% | 0.5854 | 10.9% |
| wiki-ss | comparison | 64 | -0.0306 | 0.2977 | 12.5% | 0.7849 | 20.3% |
| wiki-ss | computation_with_year | 6 | +0.5333 | 0.1667 | 0.0% | 0.2500 | 16.7% |
| wiki-ss | fact_with_year | 48 | +0.0767 | 0.3983 | 6.2% | 0.7167 | 27.1% |
| arxivqa | comparison | 112 | +0.0000 | 0.2249 | 63.4% | 0.6677 | 13.4% |
| arxivqa | computation_with_year | 1 | +nan | 1.0000 | 100.0% | 0.0000 | 100.0% |

## C. C1-C4 分层统计

| 组 | n | 占比 | 平均Regret | 占总Regret | 掉零率 | 数据集 |
|---|---|---|---|---|---|---|
| C1_coverage_OOD | 1565 | 30.6% | 0.2889 | 35.6% | 18.8% | {'wiki-ss': 1191, 'tatdqa': 259, 'arxivqa': 115} |
| C2_transferable | 2165 | 42.4% | 0.2130 | 36.3% | 10.9% | {'tatdqa': 2053, 'arxivqa': 112} |
| C3_transfer_failed | 54 | 1.1% | 0.3725 | 1.6% | 25.9% | {'wiki-ss': 54} |
| C4_uncertain | 1327 | 26.0% | 0.2534 | 26.5% | 16.5% | {'arxivqa': 676, 'tatdqa': 400, 'wiki-ss': 251} |

## 机制对照

- 覆盖且 G>0.02 的 (库,类型) 组数: 2
- 这些组 Router 选源域臂平均比例: 3.1%
- 这些组平均 Router Regret: 0.2825
- 覆盖且 G<=0.02 组平均 Router Regret: 0.2389

## 真实 Query 示例（高迁移损失组，按Regret降序）

**1. [wiki-ss/fact_with_year]** dancing with the stars athletes 2018 guest judges
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.431, 'BM': 0.0}

**2. [wiki-ss/fact_with_year]** who played mike tv in willy wonka 2005
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 0.631, 'MD': 0.0, 'TR': 1.0, 'TD': 1.0, 'BM': 0.631}

**3. [wiki-ss/fact_with_year]** who did the patriots play in the super bowl 2004
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.0}

**4. [wiki-ss/fact_with_year]** who won the 2017 national championship in basketball
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.0, 'BM': 0.0}

**5. [wiki-ss/fact_with_year]** where are the chargers playing home games in 2017
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 0.5, 'BM': 0.631}

**6. [wiki-ss/fact_with_year]** who voted for sarah to win survivor 2017
- 源域最优臂=MR，目标最优臂=TR，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 1.0, 'BM': 0.0}

**7. [wiki-ss/computation_with_year]** who sang you 're a mean one mr. grinch in the iconic 1966 animated tv show
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=1.000
- 五臂nDCG: {'MR': 1.0, 'MD': 0.0, 'TR': 1.0, 'TD': 1.0, 'BM': 0.631}

**8. [wiki-ss/computation_with_year]** in summer paralympics games 2016 held in rio de generio gold medal in men 's high jump was won by
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=0.000
- 五臂nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 1.0}

**9. [wiki-ss/computation_with_year]** total score of david warner in ipl 2017
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=0.000
- 五臂nDCG: {'MR': 0.5, 'MD': 1.0, 'TR': 1.0, 'TD': 1.0, 'BM': 1.0}

**10. [wiki-ss/computation_with_year]** how many korean athletes in winter olympics 2018
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=0.000
- 五臂nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 0.5}

**11. [wiki-ss/computation_with_year]** who won the 1959 fa cup final beating luton town 2-1
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=0.000
- 五臂nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 1.0, 'TD': 1.0, 'BM': 1.0}

**12. [wiki-ss/computation_with_year]** how many british troops were rescued in the 1940 dunkirk evacuation
- 源域最优臂=MR，目标最优臂=TD，Router选=MD，Regret=0.000
- 五臂nDCG: {'MR': 0.0, 'MD': 1.0, 'TR': 0.0, 'TD': 1.0, 'BM': 1.0}

## 结论

- C1(覆盖不足): 1565条 (30.6%)，占总Regret 35.6%
- C3(迁移失败): 54条 (1.1%)，占总Regret 1.6%

### 机制判定

- 高 G 组中 Router 选源域臂的平均比例仅 3.1%（远低于低 G 组）。
- 高 G 组的 Regret 并非来自 Router 选源域臂，而是 Router 选了**其他错误臂**（如 MD），而该臂在目标域表现极差。
- 因此，**不支持 Router 因沿用源域偏好而选错 这一机制**。偏好变化存在，但 Router 的错误另有原因（训练目标/表征问题）。

### 决策

- 不建议投入 Preference Stability-Aware / Utility Shift 训练（机制不成立）。
- Router 的跨域错误主因是：(1) 覆盖不足(C1, 35.6% Regret)；(2) 类型内细粒度路由错误(C2, 36.3% Regret)。
- 下一步应转向：增加训练数据多样性 + 改进 Router 表征/训练目标，而非偏好迁移校正。