# House 场景 Colite / COLMAP 4.0 A/B 测试记录

测试日期：2026-07-17。代码固定版本：`b6bd068`（`feat: fix colite sfm mvs reconstruction pipeline`）。测试数据 `data/house` 未提交到 Git；所有大体积结果保存在被忽略的 `build-wsl-cuda` 目录中。

## 测试条件

- WSL：Ubuntu-22.04；GPU：NVIDIA GeForce RTX 4070 SUPER；CUDA：13.2。
- House：36 张 JPEG，原始尺寸 4592×3056。
- 相机：单相机 `PINHOLE`，`f=5500, cx=2296, cy=1528`。
- SfM：4096 特征/图，Colite 使用 CUDA 优先但本次 SfM 使用 `cvsift + FLANN`；COLMAP 使用 SIFT、exhaustive matching、guided matching，CPU 线程 4，GPU 关闭。
- MVS：最长边 800（两端相同）、窗口半径 5、5 次迭代、几何一致性最少 2 个源视图、融合最少 5 个观测、重投影阈值 2 px、相对深度阈值 0.01、法线阈值 10°。
- Colite 正式 PatchMatch：CUDA、GPU 0、单 CUDA 工作区（`num_threads=1`）、best-K 源选择（`source_samples=0`）。COLMAP 4.0：CUDA、GPU 0、4 个线程、`num_samples=15`。这是当前能稳定完成的共同配置；Colite 的实验性 CDF 采样在该场景会阻塞，已单独保留诊断日志。
- Poisson 为保证两端都有有效网格，正式结果使用 `depth=13, point_weight=1, color=1, trim=0`。Colite 的 `trim=10` 在此点云密度下输出 0 顶点，空结果保留为 `meshed-poisson-trim10-empty.ply`。

## SfM 结果

| 项目 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 完整 SfM 墙钟时间 | 82.61 s | 352.19 s（特征 165.18 + 匹配 45.72 + Mapper 141.29） |
| 注册图像 | 36/36 | 36/36 |
| 稀疏点数 | 3,404 | 11,765 |
| 平均重投影误差 | 1.29593 px | 1.11545 px |
| 最大重投影误差 | 4.92457 px | 3.95530 px |

`compare_colmap` 的 Sim(3) 对齐报告：共同图像 36、共同图像对 164；相机中心误差均值/中位数/最大值为 1.302/1.329/2.519（按参考基线归一化均值 0.1658、最大 0.3207）；两端点 ID 不同，因此按点 ID 的共同点数为 0。报告文件位于 `metrics/sfm/`。

## MVS 阶段耗时

| 阶段 | Colite CUDA | COLMAP 4.0 CUDA |
|---|---:|---:|
| image_undistorter（800） | 20.64 s | 24.01 s |
| patch_match_stereo（800，5 iter） | 423.44 s | 373.21 s |
| stereo_fusion | 22.81 s | 14.77 s |
| poisson_mesher（trim=0） | 31.62 s | 134.23 s |
| mesh_simplifier（50% 面数） | 5.79 s | 4.03 s |
| mesh_texturer | 12.16 s | 79.76 s |
| delaunay_mesher | 9.42 s | 当前 COLMAP 构建无 CGAL，命令不可用 |

按 Poisson 主路径（不计 Delaunay），Colite MVS 为 516.46 s，COLMAP 为 630.01 s；加上 SfM 后分别约 599.07 s 和 982.20 s。由于 Colite 正式 PatchMatch 为单工作区稳定配置，不能把该总耗时视为完全公平的线程数对比。

## 精度、覆盖率和拓扑

- 视差有效覆盖率（36 张平均）：Colite 31.47%，COLMAP 41.46%；共同有效像素平均 18.12%。共同有效像素的原始相对深度差中位数 62.90%。两套 SfM 的尺度不同，以公共像素的稳健尺度因子 `Colite→COLMAP = 0.613885` 对齐后，中位数/P90/P95 相对误差为 11.06%/23.48%/25.71%（2,776,339 个像素）。
- 融合点数：Colite 99,537，COLMAP 106,895。点云和网格的绝对坐标距离必须先做 Sim(3) 对齐，不能直接用原始坐标差作为精度结论。
- Poisson：Colite 466,981 顶点、932,252 三角形；COLMAP 477,202 顶点、953,288 三角形。边界边分别 176/170，非流形边均为 0，退化三角形分别 110/14。
- 简化网格：Colite 233,883 顶点、465,820 三角形；COLMAP 238,823 顶点、476,644 三角形。Colite 边界边 1,000，COLMAP 146；COLMAP 简化结果仍有 25 条非流形边。
- 纹理：Colite 287,911/465,820 面有有效 UV（61.81%），图集 8192×3352，非黑像素 99.98%；COLMAP 445,282/476,644（93.42%），图集 16384×12610，非黑像素 45.92%。有效纹理面集合 Jaccard 为 0.5803。纹理对比是当前最明显的结构差距。
- Colite Delaunay 输出 17,253 顶点、34,395 三角形；COLMAP 端因构建时未启用 CGAL 未生成对应结果。

定量 JSON 位于 `build-wsl-cuda/house-ab-20260717/metrics/`：`fused.json`、`poisson.json`、`simplified.json`、`textured.json`。这些脚本对网格使用最多 20,000 点采样近邻；未进行 Sim(3) 几何对齐。

## 结果文件

根目录：`build-wsl-cuda/house-ab-20260717/`

- Colite SfM：`colite/sfm/`（`cameras.txt`、`images.txt`、`points3D.txt`、`points3D.ply`）。
- Colite CUDA MVS：`colite-cuda/mvs/`；800 像素去畸变图在 `undistorted/`，视差/法线在 `stereo/`，融合点云在 `dense/fused.ply`，网格和纹理在 `surface/`。
- COLMAP 稀疏模型：`colmap/sparse/0/`（原生 BIN）；文本转换副本：`colmap/mvs/undistorted/sparse-txt/`。
- COLMAP 4.0 MVS：`colmap/mvs/undistorted/`（800 像素视差/法线及配置），融合点云 `colmap/mvs/dense/fused.ply`，网格和纹理 `colmap/mvs/surface/`。
- 早期 1600 像素 COLMAP 视差完整副本保存在 `colmap/mvs/undistorted/stereo-1600/`，可用于分辨率敏感性测试。
- 每个阶段的完整命令输出和 `/usr/bin/time` 文件均保存在测试根目录，例如 `colite-cuda-patch-match.log/.time`、`colmap-patch-match-800.log/.time`。

## 主要结论

Colite 的 SfM 已能注册全部 36 张图，但点数和重投影误差仍落后 COLMAP；MVS 融合点数接近，Poisson、简化和纹理化速度明显更快。当前最大精度/效果差距在 PatchMatch 覆盖率（31.47% 对 41.46%）和 Texturer 有效面覆盖率（61.81% 对 93.42%）。后续优先级应为：稳定并行 CUDA 工作区、提高可用深度覆盖率、修复纹理视图选择/保守栅格化造成的未纹理面，再做 Sim(3) 对齐后的点云/网格几何评估。
