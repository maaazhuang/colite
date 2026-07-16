# IncrementalTriangulator 与 COLMAP 3.11 对齐报告（2026-07-15）

## 结论

本轮对齐了 IncrementalTriangulator 的全局 complete/merge 调度、递归 track completion、track-length 加权 merge、纯双视图轨迹过滤，以及 BA 前后的执行时机。

building-18 上，sequential 的 3D 点数从 15,764 降到 9,547，平均 track length 从 3.2518 提升到 4.0560；COLMAP 3.11 为 9,148 点和 4.3428。模型仍保持 18/18 注册，最大相机旋转误差为 0.2147°，没有用牺牲位姿精度换取结构指标。

## 实现变化

- 为已验证匹配构建 observation-centric correspondence adjacency，不再只按当前图像逐 pair 做全局 completion。
- `CompleteAndMergeAllTracks` 从每条已有 track 的全部观测出发，按 `complete_max_transitivity=5` 递归传播。
- complete、continue 和 merge 使用独立的 4 px 重投影阈值。
- merge 使用两个点的 track length 加权三维坐标，并要求合并后两个 track 的所有观测均通过 merge 阈值；成功后继续递归合并。
- merge 与 COLMAP 一样保留同一图像中的不同 feature observation，由后续 BA 和 observation filter 判断其几何有效性。
- 全局 BA 前、全局 BA 后各执行一次 complete/merge；BA 后阶段不依赖 observation filter 开关。
- `ignore_two_view_tracks` 默认设为 `true`，纯 degree-1/degree-1 的孤立双视图对应不再生成大量短 track。单元测试可显式关闭该选项以验证纯双视图三角化。
- sparse `summary.txt` 新增全局补全、合并、BA 前后误差和过滤数量诊断。

## building-18 结构 A/B

候选模型复用同一批已缓存特征和几何验证结果，仅重跑 Mapper。`before` 是本轮对齐前版本，`after` 是最终 v13。

| 模型 | 注册图像 | 3D 点 | observations | 平均 track | track=2 | track>=4 | 平均重投影误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLMAP 3.11 | 18 | 9,148 | 39,728 | 4.3428 | 6.59% | 52.97% | 0.7907 px |
| sequential before | 18 | 15,764 | 51,262 | 3.2518 | 49.89% | - | 0.6183 px |
| sequential after | 18 | 9,547 | 38,723 | 4.0560 | 16.98% | 46.94% | 0.7467 px |
| vocab before | 18 | 15,894 | 51,528 | 3.2420 | 50.08% | - | 0.6282 px |
| vocab after | 18 | 9,632 | 38,958 | 4.0446 | 18.10% | 46.66% | 0.7527 px |

最终 sequential 相比 COLMAP 的点数多 4.36%，平均 track length 低 6.60%；vocab 点数多 5.29%，平均 track length 低 6.87%。对齐前约一半点是双视图短 track，对齐后下降到 17%～18%，这是本轮结构差距收敛的主要来源。

全局阶段诊断：

| 指标 | sequential | vocab |
|---|---:|---:|
| completed observations | 174 | 113 |
| merged points | 30 | 20 |
| merged observations | 182 | 108 |
| BA 初始/最终平均误差 | 0.7508 / 0.7458 px | 0.7570 / 0.7514 px |

## 相机位姿精度

两个模型均使用共同图像的相机中心估计 Umeyama Sim(3)，对齐到 COLMAP 坐标系后比较。

| 指标 | sequential after | vocab after |
|---|---:|---:|
| 平均/最大旋转误差 | 0.1310° / 0.2147° | 0.0745° / 0.1309° |
| 平均/最大中心误差 | 0.00564 / 0.01070 | 0.00460 / 0.01124 |
| 平均/最大中心误差占最大基线 | 0.0447% / 0.0849% | 0.0365% / 0.0892% |

## 验证

- Incremental 定向测试：25/25 通过。
- 完整 WSL CTest：283/283 通过。
- 24 项需要运行时 CUDA 设备的 PatchMatch 测试按既有条件跳过。
- building-18 sequential/vocab：均为 18/18 注册。

## Point-centric multi-view Create 与 angular Continue（v14）

在 complete/merge v13 基础上继续完成：

- `TriangulateRegisteredImage` 改为逐 reference observation 处理，不再逐 image pair 创建两观测点。
- 通过 correspondence adjacency 收集 `max_transitivity` 层对应，默认与 COLMAP 一致为 1。
- Create 从所有尚未三角化的 registered observations 中生成两视图假设，按 2° angular error 统计内点，再用全部内点进行多视图线性重估。
- 鲁棒估计留下至少 3 个离群观测时，递归创建剩余 track，避免一个 correspondence component 中的多个真实点被直接丢弃。
- Continue 在所有已三角化候选 3D 点中计算 reference observation 的 angular error，选择误差最小且不超过 2° 的候选。
- Continue 先于 Create 执行，避免先创建重复短 track、再依赖 merge 修复。
- 最小三角化角默认从 0.5° 对齐到 COLMAP 的 1.5°。

v14 building-18 结果：

| 模型 | 注册 | 3D 点 | observations | 平均 track | track=2 | track>=4 | 平均重投影误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLMAP 3.11 | 18/18 | 9,148 | 39,728 | 4.3428 | 6.59% | 52.97% | 0.7907 px |
| sequential v13 | 18/18 | 9,547 | 38,723 | 4.0560 | 16.98% | 46.94% | 0.7467 px |
| sequential v14 | 18/18 | 9,057 | 39,202 | 4.3284 | 7.07% | 53.40% | 0.8655 px |
| vocab v14 | 18/18 | 9,120 | 39,432 | 4.3237 | 7.86% | 52.84% | 0.8682 px |

sequential v14 相比 COLMAP：点数少 0.99%，observations 少 1.32%，平均 track length 低 0.33%，双视图点比例高 0.48 个百分点。vocab v14 的点数少 0.31%，平均 track length 低 0.44%。稀疏模型的核心结构指标已经基本对齐。

Sim(3) 位姿比较：

| 指标 | sequential v14 | vocab v14 |
|---|---:|---:|
| 平均/最大旋转误差 | 0.0388° / 0.0872° | 0.0567° / 0.0934° |
| 平均/最大中心误差占最大基线 | 0.0162% / 0.0528% | 0.0135% / 0.0327% |

位姿精度比 v13 进一步改善。当前代价是平均重投影误差比 COLMAP 高约 9.5%，并且在相同缓存输入下，sequential Mapper 时间从约 6.2 秒增加到 10.7 秒。这与角度阈值允许更宽的像素残差，以及当前确定性两视图假设枚举尚未使用 COLMAP 的 LORANSAC 自适应采样有关。

## 剩余结构差距与后续方向

Create/Continue 的主数据流已对齐，剩余差距集中在估计器和大规模效率：

1. COLMAP 使用通用 LORANSAC、组合采样和局部估计器；COLITE 当前使用最多 10,000 个确定性两视图假设和两轮多视图线性重估。
2. COLMAP 使用持久 correspondence graph、modified-point 和 merge-trial 缓存；COLITE 当前每次 Mapper 调用构建 adjacency，大数据集效率仍可优化。
3. 当前 angular Create 使重投影误差略高于 COLMAP。后续应增加与 COLMAP 等价的局部非线性 point refinement 或复用项目内统一 RANSAC 框架，而不是收紧 2° 默认阈值破坏语义对齐。
4. 仍需在宽基线、重复纹理、无序图像和数百张图像数据集验证鲁棒性及内存增长。

下一轮建议优先优化鲁棒多视图估计器和 correspondence adjacency 缓存，并把重投影误差降低到 COLMAP 的 ±5% 范围。
