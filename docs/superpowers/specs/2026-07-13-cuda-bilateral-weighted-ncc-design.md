# CUDA Bilateral Weighted NCC Design

## 目标

将 COLITE CPU/CUDA PatchMatch 的普通 ZNCC 替换为参考 COLMAP 3.11 的 bilateral weighted NCC，降低深度不连续边缘处跨表面像素对光度代价的污染。

## 行为约束

- 权重由参考窗口中心灰度、参考采样灰度和二维空间距离共同决定。
- 同一个权重用于参考均值、源图均值、参考方差、源图方差和互协方差。
- 双边加权通过 `PatchMatchCostOptions::bilateral_weighting` 显式启用；当前默认关闭以保持既有小窗口流程质量。
- `sigma_spatial <= 0` 时使用 `max(1, window_radius)`，与 COLMAP 的默认语义一致。
- 启用时默认 `sigma_color = 0.2`，输入灰度范围保持 `[0, 1]`。
- 保留 COLITE 当前 `0.5 * (1 - NCC)` 代价尺度，避免现有过滤阈值整体漂移。
- CPU 是数值参考；CUDA 使用相同公式并允许浮点精度误差。
- 核心实现新增中文注释。

## 验证

1. 合成强边缘窗口中，小 `sigma_color` 的代价应显著低于近似无权重的代价。
2. 固定深度/法线和单源图时，CUDA 中心像素代价应与 CPU bilateral weighted NCC 接近。
3. 全部 CPU/CUDA PatchMatchBackend 测试通过。
4. building-18 使用与上一里程碑相同参数运行，比较一致像素、融合点、过滤像素和 CUDA 阶段耗时。

## 后续边界

本阶段不引入 COLMAP 的 NCC likelihood/HMM source selection 更新，也不改变过滤阈值的成本尺度；这些作为后续独立里程碑处理。
