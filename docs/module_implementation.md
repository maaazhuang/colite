# Colite 模块实现说明

本文面向继续开发算法的工程师，说明当前实现、关键语义、代码入口和已知边界。接口与默认值最终以源码和测试为准。

## Core 与并行框架

- `include/colite/core`、`src/colite/core` 定义相机、图像、点、track、pose、重建状态和通用几何。
- `core/parallel` 是持久工作线程池和 `parallel_for_` 实现。CPU SIFT、匹配、Fusion、纹理等优先复用该框架，避免每轮创建线程。
- `Reconstruction` 负责双向 observation/track 一致性；修改观测必须走其接口，不能只改 `Point3D.track` 或 `Image.points2d` 单侧。

### AVX2 与 Eigen ABI

AVX2/FMA 只能加到不跨库边界传递 Eigen 对象的 SIFT 内核源文件，不能加到整个 `colite_core`。如果库内部按 AVX 对齐构造 `FeatureSet`，而测试或调用方按非 AVX ABI 析构，Eigen 的对齐分配/释放约定可能不一致，最终把偏移后的地址传给 `free`。当前 CMake 仅对 `sift_opencv.cpp` 和 `sift_vlfeat.c` 添加 `-mavx2 -mfma`，不要扩大作用域。

## SIFT

统一选择：`auto`、`cvsift`、`vlsift`、`cuda_opencv`、`cuda_vlfeat`。`auto` 检测到可用 GPU 时优先 CUDA OpenCV 模式，否则使用 CPU OpenCV 模式。

| 后端 | 输入/语义 | 特点 |
|---|---|---|
| `cvsift` | OpenCV-like，灰度 `[0,255]` | 持久线程池、AVX2/FMA、全局 response top-K |
| `vlsift` | VLFeat/COLMAP-like，灰度 `[0,1]` | 4 octave、COLMAP threshold、最多双方向、descriptor bin 转换 |
| `cuda_opencv` | CUDA OpenCV 语义 | 与 CPU OpenCV 点集目标重叠率 99.9% 以上 |
| `cuda_vlfeat` | CUDA VLFeat 语义 | `+0.5` 像素中心和 descriptor 顺序对齐 CPU VLFeat |

关键实现：

- `src/colite/features/sift_extractor.cpp`：统一入口、RootSIFT、后端选择。
- `src/colite/features/backends/sift_opencv.*`：CPU OpenCV 语义。
- `src/colite/features/backends/sift_vlfeat.*`：CPU VLFeat 语义。
- `src/colite/features/backends/cuda_sift/`：两种 CUDA traits 模式。

OpenCV-like 高斯边界使用周期折叠的 reflect/`REFLECT_101` 映射，支持卷积核宽于极小金字塔层；相关越界回归测试不可删除。VLFeat 的 `max_num_features` 按基础 keypoint/DoG level 保留，随后可生成多个方向，因此最终 descriptor 数可能超过上限；A/B 时不能把它等同于严格输出行数。

## 匹配与两视图几何

- BF：在 COLMAP 风格 uint8/512 描述子空间执行 dot-product、max distance、ratio 和 cross-check。
- FLANN：`flann::L2<uint8_t>`，KDTree 和 search 参数向 COLMAP CPU 路径靠齐。
- pair 模式：exhaustive、sequential、spatial、vocabulary tree。
- spatial 必须有所有图像的位置先验，缺失时直接报错，不静默退化。
- vocabulary tree 支持离线持久层次树；A/B 时必须把 builder 与 matcher 分开计时。
- 两视图验证内部估计 E/F/H，PoseLib 可用时使用 five-point；支持平面、全景、多模型和退化分类。
- guided matching 与 weak-pair rematch 属于二阶段匹配，统计语义与 COLMAP SQLite 的 `two_view_geometries.rows` 不完全相同。

主要入口位于 `src/colite/matching` 和 `src/colite/controllers/*matcher*`。

## Incremental SfM

实现位于 `src/colite/sfm`：

- `incremental_initializer`：初始对选择和相对位姿。
- `image_registrar`：2D-3D 注册；PoseLib 可用时 P3P，否则 DLT PnP。
- `incremental_triangulator`：Continue、point-centric multi-view Create、complete/merge。
- `local_bundle_adjuster` / `global_bundle_adjuster`：Ceres BA、参数 mask 和 gauge fixing。
- `observation_manager` / `observation_filter`：track 冲突、合并、坏点和坏观测清理。

当前三角化语义：

1. Continue 先检查已三角化候选，以最小 angular error 选择可见点。
2. Create 从 correspondence adjacency 收集多视图观测，生成两视图假设并用 angular error 判内点，再多视图重估。
3. 全局 complete/merge 从 track 全部观测递归传播，merge 使用 track length 加权坐标并验证合并后的全部观测。
4. 全局 BA 前后均执行 complete/merge 和过滤；纯孤立双视图 track 默认忽略。

在 `building-18` 上 v14 的点数、观测、track 长度和位姿已接近 COLMAP 3.11。剩余差距主要是鲁棒多视图估计器、重投影误差尾部和大规模 correspondence 缓存效率。

## 图像去畸变

实现位于 `src/colite/mvs/image_undistorter.cpp` 和 MVS 控制器。每个输出 PINHOLE ray 通过原相机投影反查源图，支持已实现投影的畸变、鱼眼和薄棱镜模型。输出模型同步更新内参、图像尺寸和稀疏观测。

`blank_pixels`、scale 范围和 max image size 控制视场及尺寸。`resolution_level=1` 表示宽高各缩小一级；若去畸变阶段已经缩放，后续 workspace 不重复缩放。

## PatchMatch Stereo

CPU/CUDA 共享 `PatchMatchView`、深度/法线/代价图、source selection 和诊断结构。CUDA 实现位于 `src/colite/mvs/patch_match_cuda.cu`，CPU/调度位于 `patch_match.cpp`。

长期保留的 COLMAP 对齐能力：

- 共享稀疏点、尺度和覆盖率 source-view scoring；
- bilateral weighted NCC；
- 平面感知传播、随机法线、深度/法线坐标下降扰动；
- forward/backward、four-direction、red-black checkerboard 调度；
- dynamic source-selection probability；
- directional HMM forward-backward message；
- CDF source sampling（常用 15）与 best-K；
- candidate-dependent 几何 pre-cost gate、遮挡/深度冲突过滤；
- photometric pass、geometric refinement 和 consistency graph；
- coarse-to-fine 金字塔、binary dense maps；
- layered texture cache、workspace pool、pinned async H2D/D2H、packed output readback。

默认策略为 `resolution_level=1`、3 次迭代、checkerboard 每迭代一轮红黑更新。质量优先可启用两层金字塔、directional HMM 和 CDF15；详细原子诊断默认关闭，因为会显著影响 CUDA 热路径。

历史 `docs/superpowers` 计划中的 source scoring、几何代价、传播、稳定初始化、binary maps、并行缓存、质量诊断、法线 refinement 和 CUDA 对齐任务均已落地，其长期行为已归纳在本节；原计划文件不再保留。

## Stereo Fusion

Fusion 读取持久化 dense maps 和 consistency graph，从 seed 像素沿互相一致的投影构造 track。位置、法线和 RGB 使用稳健聚合；输出 `fused.ply`、`points_visibility.bin` 和诊断 TSV。

seed-track 使用 `core/parallel_for` 按图像行带并行，原子 CAS 认领像素，线程局部收集点、track、可见性和误差样本后合并。并发认领可能让少量边界 seed 归属不同，因此点数允许极小非确定性，但几何阈值不变。

MVS workspace 同时缓存灰度图和 RGB 图。PatchMatch 只看灰度；Fusion 对 RGB 双线性采样。仅构造灰度 `MvsImage` 的旧测试/调用方继续回退为三通道同灰度。

## Poisson Mesher

- 使用仓库内 PoissonRecon，默认随项目启用。
- 对齐 COLMAP 4.0 的 `depth`、`point_weight`、`color`、`trim`、线程和 shallow-tree `fullDepth` 行为。
- 输入必须包含有向点；`color=1` 时保留稠密点云颜色。
- 相同输入和参数下与 COLMAP 4.0 的拓扑、顶点和面积已达到近数值等价。

## Delaunay Mesher

使用 CGAL 3D Delaunay 四面体剖分、相机到点的可见射线、Boost Boykov-Kolmogorov s-t min-cut 和定向边界面提取。已对齐：

- source/sink 与 oriented facet 权重；
- 有符号 facet-cell cosine；
- infinite facet 穿越；
- 固定 seed 的 Fisher-Yates 子采样；
- percentile × factor 长面过滤。

严格共球数据会对插入顺序和浮点扰动敏感。合成真值比较应添加确定性 radial jitter，并同时报告顶点、面集合 Jaccard、面积、边界/非流形边和真值 precision/recall。

## Mesh Simplifier

自研 QEM edge collapse，检查 link condition、退化面、翻面、边界和法线变化。无约束 QEM 最优点会投影到待折叠边段 `t∈[0,1]`，避免局部外推尖点。支持相对/绝对目标、最大误差和边界策略。

## Mesh Texturer

当前流程：标定视图评分 → 三角形 AABB BVH ray occlusion → 邻域 label smoothing → 同图相邻面 patch → atlas packing → UV/颜色烘焙 → 可选 inpaint 和跨视图颜色校正。

BVH 线段求交排除当前面及共享目标顶点邻面。当前“任意更近遮挡即拒绝”比 COLMAP/CGAL 首交点语义更保守，假阳性较少但有效面召回偏低，是主要剩余差距。大网格应先简化并逐级增加 `visibility_resolution` 和 atlas 上限。

## I/O 与颜色

- 稀疏 `points3D.txt`/PLY 写真实 `Point3D.color`，不再固定白色。
- 稠密 `fused.ply` 包含坐标、法线和 RGB；Delaunay 可见性另存 binary。
- COLMAP text/binary 读取保留 RGB；去畸变稀疏模型也保留颜色。
- OpenMVG 需要 `ComputeSfM_DataColor` 后处理输出 `colorized.ply`；其代表视图策略与 Colite/COLMAP 的多观测均值不同。

## 已知差距

1. Texturer 保守遮挡导致有效面覆盖低于 COLMAP。
2. House SfM 在不同前端配置下点数和误差尾部仍可能与 COLMAP 有明显差异。
3. PatchMatch 全分辨率、多并发 workspace 的显存/调度仍可优化。
4. Delaunay 需要更多非退化真值场景验证，而不只是与 COLMAP 输出互比。
5. ASCII PLY 的精度、体积和退化三角形舍入问题尚可通过统一 binary 输出改善。
