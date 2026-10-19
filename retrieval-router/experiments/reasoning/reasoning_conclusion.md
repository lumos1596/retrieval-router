# Reasoning Router Conclusion

## One-sentence answer

**零样本检索能力推理（3B）没有帮助。** P2（结构化能力推理）未能稳定超过 P0（直接选择），且所有推理模型都远不如 FixedMR 和 B2。问题不在推理深度，而在 3B 模型从 Query 文本无法判断哪个检索器实际更优。

## Facts (data)

1. **三目标宏平均 nDCG**: P0=0.484, P1=0.503, P2=0.493, FixedMR=0.672, B2=0.672
2. **P2 未稳定超过 P0**: tatdqa P2<P0 (-0.020, 显著), wiki-ss P2≈P0, arxivqa P2>P0 (+0.047, 显著)
3. **P2 未超过 P1**: 所有库 P2-P1 的 95% CI 均包含 0
4. **反 MR 偏置严重**: P0/P1/P2 的 MR 选择率分别 <1% / <4% / <4%，而 FixedMR 是最强基线
5. **MR-advantage 桶灾难**: 当 MR 真正更优时，P2 仍有 79-98% 概率选非 MR，净损失 -0.55 到 -0.78
6. **NonMR-advantage 桶**: P2 切换率 >95%，但 P0 的实际 nDCG 反而最高（说明推理没帮上忙，直接选更好）
7. **灾难性掉零率**: P2 在 tatdqa 32.7%、arxivqa 29.3%，远高于 FixedMR 的 8.7%/10.4%
8. **成本**: P2 平均输出 ~50 tokens，是 P0（1 token）的 50 倍，却没有带来收益

## Inference (speculation)

- 3B 模型从 Prompt 中的五臂描述形成了"MR 是昂贵的最后手段"的先验，导致几乎不选 MR。这与 B2（过度选 MR）是镜像问题。
- 结构化推理步骤（Task/Constraints/Matching Need/Comparison）没有提供超出直接判断的有效信号，因为 Query 文本本身不包含"哪个检索器能找到答案"的信息——这本质上需要检索反馈。
- arxivqa 上 P2 唯一显著优于 P0，可能因为 arxivqa 含大量"figure"类 Query，推理步骤让模型更倾向选 MD/MR（视觉检索），碰巧有效。

## Decision

按计划决策表第 4 行：**P2 不能超过 P0 或 FixedMR，且高价值切换没有改善 → 当前零样本能力推理假设不成立（或 3B 能力不足）→ 暂停推理蒸馏，不投入完整 SFT/RL。**

**不建议**进入训练阶段。若未来重试，需先解决：(1) 如何给模型提供检索反馈而非纯 Query 推理；(2) 如何校准 MR 先验偏置；(3) 更大的模型或监督微调。
