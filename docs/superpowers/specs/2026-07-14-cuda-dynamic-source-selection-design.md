# CUDA Dynamic Source Selection Design

## 目标

在现有 CUDA PatchMatch 中引入 NCC likelihood 驱动的 dense source-selection probability 更新，使 source 采样从静态 SfM 评分升级为随当前深度/法线和光度质量迭代变化。

## 与 COLMAP 的关系

- 使用 COLMAP 3.11 的截断高斯 NCC likelihood，`ncc_sigma` 默认 0.6。
- COLMAP 在方向 sweep 中使用前向/后向 HMM message；COLITE 当前以 checkerboard/red-black 为主，第一版使用逐像素 Bayesian posterior 与 previous-probability EMA。
- 初始光度 prior 为 0.5；若调用方提供 dense probability map，则从该图开始。
- 动态光度概率与现有 source view 静态评分相乘后用于采样，避免两类 prior 相互覆盖。
- 当前光度代价范围为 `0.5 * (1 - NCC)`，进入 likelihood 前乘 2 转换为 COLMAP 的 `1 - NCC` 范围。

## 更新规则

```text
emission = truncated_gaussian_likelihood(2 * photometric_cost)
posterior = prior * emission / (prior * emission + (1 - prior) * 0.5)
updated = previous_weight * prior + (1 - previous_weight) * posterior
```

最终概率限制在可配置的 `[min_probability, max_probability]`，防止单轮错误后 source 永久失活。由于当前 checkerboard 调度没有 COLMAP 的固定数量 CDF 抽样，动态概率采用确定性 floor gate：只有降到 floor 的 source 被淘汰，其余 source 继续参加 best-K 聚合，避免独立 Bernoulli 造成有效 source 数不足。

## 约束与诊断

- 功能默认关闭，通过 PatchMatch 选项和 `recon_mvs` 命令行显式启用。
- 动态概率 buffer 在 CUDA workspace 中复用。
- 每轮记录概率更新数和平均概率，结束时记录每个 source 的平均概率。
- 核心代码使用中文注释。

## 验证

1. NCC likelihood 随 cost 单调下降。
2. 相同初始 prior 下，匹配 source 的最终平均概率高于光度不一致 source。
3. 固定输入和种子时概率统计可复现。
4. 完整 CPU/CUDA PatchMatch 回归通过。
5. building-18 分别验证动态概率单独启用及与 bilateral NCC 组合启用。
