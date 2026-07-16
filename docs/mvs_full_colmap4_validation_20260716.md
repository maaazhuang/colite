# colite 与 COLMAP 4.0 全 MVS 单步骤 A/B（2026-07-16）

## 结论

本轮依次验证 `image_undistorter`、`patch_match_stereo`、`stereo_fusion`、
`poisson_mesher`、`delaunay_mesher`、`mesh_simplifier` 和 `mesh_texturer`。

- **Image undistorter 的输出相机已经完全对齐。** 18 张图共同像素区域 100%，整体
  PSNR 36.76dB；剩余差异主要是插值/JPEG 编码。
- **PatchMatch 的保留区域精度达到 COLMAP 水平，但覆盖率和效率仍有差距。** 最终
  median/P95 为 2.05%/6.04%，COLMAP 为 2.02%/7.03%；有效覆盖率为 55.59%，
  COLMAP 为 68.06%。
- **Fusion 的几何覆盖已经接近。** colite 输出 58,206 点，达到 COLMAP 65,647 点的
  88.7%，双向最近邻 median 均约 0.0035；但单线程 track 主循环明显更慢。
- **Poisson 已基本数值等价。** 相同输入下顶点/面数和拓扑统计一致，P95 顶点差仅
  6.1e-6，colite 墙钟时间约为 COLMAP 的 20.9%。
- **Delaunay 仍存在结构差异。** 两边均使用输入融合点的子集，colite 对融合云的覆盖
  略好，但表面积、边流形性和最终表面范围不同，不能视为已对齐。
- **Simplifier 的主体误差接近，但存在严重局部外推点。** P95 与 COLMAP 几乎一致，
  但最大误差 0.473 对 0.0262，需要约束 QEM collapse position。
- **Texturer 的投影/视图选择主体已接近，差距集中在遮挡判定。** 无遮挡控制组为
  676/677 有效纹理面；默认遮挡下为 428/511。

## 环境与公平性

| 项目 | 配置 |
|---|---|
| 数据 | `data/building-18`，18 张图，共用已注册位姿 |
| GPU | RTX 4070 SUPER，CUDA 13.2 |
| colite | `build-wsl-mvs-cuda-20260716` |
| COLMAP CUDA | `build-wsl-colmap4-cuda132-reference`，4.0.0 |
| COLMAP CGAL | `build-wsl-colmap4-reference/install`，4.0.0 CPU + CGAL |
| PM 尺寸 | 最长边 256，8 sources，3 iterations |
| 去畸变尺寸 | 最长边 800 |

CUDA 参考构建未启用 CGAL，因此 Delaunay 和带真实遮挡检测的 Texturer 使用另一个
隔离的 COLMAP 4.0+CGAL CPU 构建。计时均包含进程启动，且为单次热文件缓存结果；它们
适合定位数量级瓶颈，不应当作稳定 microbenchmark。Poisson、Delaunay、Simplifier、
Texturer 均使用相同输入，以隔离单模块差异。

## 1. Image undistorter

两边使用相同 OPENCV 相机、`blank_pixels=0`、`min_scale=0.2`、`max_scale=2` 和
`max_image_size=800`。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间（本轮热缓存） | 5.09s | 3.29s |
| 输出尺寸 | 800x579 | 800x579 |
| `fx, fy` | 593.981873, 593.638048 | 完全相同 |
| `cx, cy` | 400, 289.5 | 完全相同 |
| 共同像素区域 | 100% | reference |
| RMSE / MAE | 3.704 / 2.640 | reference |
| PSNR | 36.756dB | reference |

本轮 COLMAP 使用多线程且文件已在系统缓存中，因此快于 colite；此前冷启动测量为
colite 5.43s、COLMAP 22.98s。应增加三次 warm-up 后重复计时，当前只能确认数值路径
已经对齐，不能下稳定的速度结论。

## 2. PatchMatch stereo

colite 使用 bilateral NCC、directional HMM、CDF15、3 轮光度和 3 轮几何 refinement；
COLMAP 使用其原生 source message/CDF 和 geometric pass。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 完整墙钟 | 41.98s | 34.32s |
| 有效深度覆盖率 | 55.593% | 68.062% |
| 共同有效覆盖率 | 51.036% | — |
| 稀疏点有效样本 | 30,725 | 35,098 |
| 稀疏点误差 median | 2.047% | 2.016% |
| 稀疏点误差 P90 | **4.791%** | 5.305% |
| 稀疏点误差 P95 | **6.037%** | 7.029% |
| 共同深度差 median / P95 | 0.719% / 4.582% | reference |

保留下来的深度精度已经达到参考水平，但覆盖率只达到 COLMAP 的 81.7%，且墙钟慢
22.3%。下一步重点不是继续收紧过滤，而是把 HMM/CDF 内联到逐方向 sweep，并补齐
candidate-dependent incident-angle 与 resolution prior，减少光度长尾导致的召回损失。

## 3. Stereo fusion

双方使用最少 2 个观测、2px reprojection、1% depth、10° normal。colite 输入其 HMM
几何图，COLMAP 输入其 geometric maps，因此本项同时包含上游覆盖差异；当前还不能把
点数差全部归因于 Fusion 本身。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | 8.76s | 0.90s |
| 点数 | 58,206 | 65,647 |
| 点数比例 | 88.7% | 100% |
| colite→COLMAP median / P95 | 0.003745 / 0.013072 | — |
| COLMAP→colite median / P95 | 0.003437 / 0.011789 | — |

旧冷缓存 COLMAP 结果约 2.95s，即使使用该值，colite 仍约慢 3 倍。质量差距已经大幅
缩小，当前明确瓶颈是 seed-track 主循环仍为单线程；像素缓存的 `parallel_for` 只覆盖
预计算，不能替代 COLMAP 的分行 track 并行。

## 4. Poisson mesher

输入为同一份 58,206 点 colite Fusion PLY，参数为 `depth=8`、`point_weight=1`、
`color=1`、`trim=0`。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | **7.85s** | 37.51s |
| 顶点 / 三角形 | 17,769 / 35,386 | 完全相同 |
| edges / boundary | 53,114 / 70 | 完全相同 |
| Euler characteristic | 41 | 41 |
| 面积 | 9.506725 | 9.506735 |
| 双向顶点距离 median / P95 | 4.08e-6 / 6.11e-6 | reference |

colite 快 4.78 倍，输出几何可视为等价。唯一可见问题是 ASCII PLY 精度导致 6 个
数值面积低于 1e-15 的退化三角形，而 COLMAP binary PLY 为 0；建议提升 ASCII 输出
精度或增加 binary PLY。

## 5. Delaunay mesher

双方使用同一融合云、同一 `points_visibility`、相机和图像，
`quality_regularization=1`、`distance_sigma_factor=1`。

| 指标 | colite | COLMAP 4.0+CGAL |
|---|---:|---:|
| 墙钟时间 | 2.89s | **1.71s** |
| 顶点 / 三角形 | 9,734 / 19,462 | 10,270 / 20,496 |
| boundary / non-manifold edges | 73 / 24 | 42 / 2 |
| 面积 | 3.916 | 1.601 |
| colite→COLMAP vertex P95 | 0.013895 | reference |
| COLMAP→colite vertex P95 | 0.009818 | reference |
| 融合云→mesh vertices P95 | **0.013413** | 0.016186 |

两边输出顶点都来自公共融合云；colite 对输入云的空间覆盖略好，但非流形边更多，且
表面积为 COLMAP 的 2.45 倍。由于没有真实表面 GT，不能仅以 COLMAP 较小的表面范围
判定 colite 是离群；可以确定的是 graph-cut unary/smoothness 和 outlier-face 清理仍未
结构对齐。建议增加合成已知 mesh 的 precision/recall，而不是只比较两份输出 Chamfer。

## 6. Mesh simplifier

输入为同一份 COLMAP Poisson 网格，目标比例 0.5（35,386 → 17,693 faces）。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | **0.27s** | 0.35s |
| 顶点 / 三角形 | 8,916 / 17,692 | 8,920 / 17,693 |
| 对输入 candidate→input P95 | 0.005193 | 0.005160 |
| 对输入 input→candidate P95 | 0.012611 | 0.012588 |
| 最大 candidate→input | **0.472881** | 0.026227 |
| non-manifold edges | 0 | 4 |

主体分布和效率已接近，colite 还保持了更好的边流形性；但最大误差暴露一个 P0 问题：
QEM 最优点被外推到局部表面之外，输出 `z=5.305`，而输入最大 `z=4.832`。建议将 collapse
候选限制在线段/局部 one-ring 包围盒内，并把最大 Hausdorff 近似加入回归门槛。

## 7. Mesh texturer

使用同一个 536 顶点、1,000 面 building-18 网格和相同 18 张 800x579 去畸变图。
COLMAP 使用 CGAL 遮挡检测；colite 使用 512 分辨率深度缓冲遮挡检测。

| 指标 | colite | COLMAP 4.0+CGAL |
|---|---:|---:|
| 墙钟时间 | **0.72s** | 1.33s |
| 有效 UV 面 | 428 / 1000 | 511 / 1000 |
| 有效面集合 intersection / union | 421 / 518 | — |
| 有效面 Jaccard | 81.27% | reference |
| colite-only / COLMAP-only | 7 / 90 | — |
| atlas | 2048x1513 | 2048x1801 |

无 CGAL/无遮挡控制组中，colite 将 occlusion tolerance 放宽到 1.0 后得到 676 面，
COLMAP 跳过遮挡得到 677 面，说明投影、基础 view eligibility 和 atlas packing 已经很接近。
默认结果主要是 colite 深度缓冲产生了 90 个额外 false rejection；把 tolerance 从 1% 增至
5% 只把 428 提高到 434，说明问题不是简单阈值，而是低分辨率栅格可见性、薄三角形和
深度采样语义。建议下一步优先实现分层/保守 rasterization 或 CPU BVH ray fallback。

纹理图本身因 atlas packing 不同不能逐像素直接比较；“非黑像素比例”也不公平，因为
colite 会对几乎整个裁剪 atlas 扩散/inpaint，而 COLMAP 只填充 patch 周边。因此本轮把
有效 UV 面集合与 atlas 面积作为稳定指标，没有用 PNG PSNR 伪装纹理精度。

## 优化优先级

### P0

1. Fusion seed-track 分行并行，并使用线程局部 points/tracks/diagnostics；目标先从 8.76s
   降到 3s 以内。
2. Simplifier 约束 QEM collapse position，消除 0.473 的局部外推点。
3. Texturer 修复 raster visibility false rejection；以 CGAL 有效面 recall 为门槛。

### P1

1. PatchMatch 将 HMM/CDF 融入方向 sweep，并补 incident/resolution prior，使覆盖率从
   55.6% 提升到至少 COLMAP 的 90% 相对水平（61.3%）。
2. Delaunay 对齐 ray unary、outside/boundary 权重、graph-cut regularization 和长面清理；
   加入已知真值合成场景。
3. Poisson 增加 binary PLY 或提高 ASCII 精度，消除写出阶段的微小退化面。

## 产物与复现工具

全部产物位于：

```text
build-wsl-cuda/full-mvs-ab-20260716/
```

机器可读结果包括 `undistortion-comparison.json`、`fusion-comparison.json`、
`poisson-real-comparison.json`、`delaunay-real-comparison.json`、
`simplifier-real-comparison.json` 和 `texturer-cgal-comparison.json`。

本轮新增两个可复用比较工具：

```bash
python3 scripts/compare_mesh_outputs.py \
  --candidate <candidate.ply> --reference <reference.ply> --output report.json

python3 scripts/compare_textured_outputs.py \
  --colite_dir <obj-mtl-png-dir> --colmap_dir <ply-png-dir> \
  --output report.json
```

`compare_mesh_outputs.py` 报告顶点最近邻、面积、边界边、非流形边、退化面和 Euler
characteristic；`compare_textured_outputs.py` 报告有效 UV 面集合、Jaccard 和 atlas 指标。
