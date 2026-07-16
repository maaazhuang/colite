# MVS 五项优先优化与 COLMAP 4.0 对比（2026-07-16）

本轮按 Fusion、Simplifier、Texturer、PatchMatch、Delaunay 的顺序完成优化。测试使用
`data/building-18` 的既有 COLMAP/colite 稠密结果，以及
`scripts/create_synthetic_mvs_fixture.py` 生成的已知真值球面场景。

## 结果摘要

| 项目 | 优化前 | 优化后 | COLMAP 4.0 / 目标 |
|---|---:|---:|---:|
| Fusion 墙钟时间，8 线程 | 8.76 s | **2.44 s** | `< 3 s` |
| Simplifier 候选点到输入最大距离 | 0.472881 | **0.025608** | 0.026227 |
| Texturer 有效面 / 1000 面 | 428 | **429** | 511（CGAL） |
| Texturer 有效面 Jaccard | 81.27% | **83.95%** | reference |
| PatchMatch 覆盖率 | 55.593% | **62.485%** | 61.3% 目标；COLMAP 68.062% |
| Delaunay building-18 顶点 / 面 | 9734 / 19462 | **10270 / 20498** | 10270 / 20496 |
| Delaunay building-18 面积 | 3.916 | **1.601199** | 1.601156 |
| Delaunay building-18 非流形边 | 24 | **2** | 2 |

## 1. Fusion seed-track 并行化

`src/colite/mvs/patch_match.cpp` 使用 `core/parallel_for` 按图像行带并行处理 seed，
以原子 CAS 认领访问像素。点、track、可见性、诊断矩阵和误差样本均在线程局部累积，
最后统一合并，避免共享容器锁竞争。

building-18 的相同 PatchMatch 输入上，8 线程两次输出分别为 60087 和 60153 点，
点数波动约 0.11%；第二次墙钟时间 2.44 s，达到 3 秒以内目标。并行认领顺序会使边界
seed 的归属存在轻微非确定性，不影响几何阈值语义。

## 2. Simplifier QEM 外推修复

QEM 的无约束最优点现在投影并限制到待折叠边段 `t in [0, 1]`，端点和中点仍参与候选。
building-18 Poisson 网格简化后，最大候选到输入距离由 0.472881 降到 0.025608；P95 为
0.005132，与 COLMAP 的 0.005160 对齐。新增
`MeshSimplificationTest.CollapsePositionsStayInsideInputBounds` 回归外推问题。

## 3. Texturer BVH ray occlusion

以三角形 AABB BVH 和 Moller-Trumbore 线段求交替代低分辨率 inverse-depth 栅格遮挡。
每个面测试三个顶点，排除当前面及共享目标顶点的相邻面；view score 改为与 COLMAP 一致的
投影面积。building-18 的 1000 面公共网格上，colite 的 429 个有效面全部属于 COLMAP
的 511 个有效面，消除了原来的 7 个 colite-only 假阳性，Jaccard 从 81.27% 提升到
83.95%，墙钟时间 0.79 s。

当前 BVH 判定会查找任意更近遮挡，比 COLMAP/CGAL 的“第一个返回交点若为当前面则可见”
更保守，因此仍少 82 个面。这是剩余 recall 差距的主要来源；若继续追求逐面一致，可增加
兼容模式模拟 CGAL `any_intersection` 的首交点语义，但几何正确性会弱于当前实现。

## 4. PatchMatch 覆盖率

保持 HMM/CDF15 参数，启用两层 coarse-to-fine：

```bash
colite recon_mvs ... --run_patch_match \
  --patch_match_directional_hmm \
  --patch_match_pyramid_levels 2
```

覆盖率由 55.593% 提升到 62.485%，为 COLMAP 68.062% 的 91.81%，超过 61.3% 门槛。
中位相对深度误差为 2.047%，P95 为 6.071%；墙钟时间由 41.98 s 增至 52.34 s。
因此默认值暂保留 1 层，质量优先流程明确使用 2 层。

## 5. Delaunay graph-cut 与长面清理

对齐 COLMAP 4.0 的三处关键语义：

1. 形状正则项使用有符号 facet-cell cosine，不再错误取绝对值；
2. 点后方 cell 的 facet 求交不再跳过 infinite facet；
3. 子采样使用 seed 0 的 COLMAP Fisher-Yates 顺序，而不是实现相关的 `std::shuffle`。

building-18 实际点云上，修复后顶点完全对齐，面数仅差 2，面积相对差约 0.0027%，边界边
和非流形边与 COLMAP 完全相同。默认的长面规则已确认与 COLMAP 一致：
`factor * percentile(max_face_edge)`。

严格单位球是共球退化输入，Delaunay 四面体剖分会被微小数值和插入顺序主导，不适合作为
跨程序逐面真值。新增 `--radial_jitter 0.015` 生成确定性非共球场景；在该场景的
`quality_regularization=0, distance_sigma_factor=2` 对比中，colite/COLMAP 为
771/775 面，面积 40.0384/40.1225，所有输出顶点一致。

## 复现与产物

主要结果位于 `build-wsl-cuda/full-mvs-ab-20260716/`：

- `colite-fusion-parallel8-r2.time`
- `colite-simplifier-segment.time`
- `colite-texturer-bvh-v2.time`
- `patch-match-hmm-pyr2-256-comparison.json`
- `delaunay-real-aligned-comparison.json`
- `delaunay-truth-jitter/comparison-float.json`

网格比较脚本现在除顶点最近邻和拓扑外，还报告基于坐标规范化的三角面 Jaccard。单位球
诊断可使用 `scripts/evaluate_sphere_mesh.py`。

最终 WSL 增量编译通过；完整 CTest 为 **285/285 通过**，1 个无 CUDA 环境专用测试跳过。
