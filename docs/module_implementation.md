# Colite 模块实现说明

本文面向继续开发算法的工程师，说明当前实现、关键语义、代码入口和已知边界。接口与默认值最终以源码和测试为准。

## Core 与并行框架

- `include/colite/core`、`src/colite/core` 定义相机、图像、点、track、pose、重建状态和通用几何。
- `core/parallel` 是持久工作线程池和 `parallel_for_` 实现。CPU SIFT、匹配、Fusion、纹理等优先复用该框架，避免每轮创建线程。
- `Reconstruction` 负责双向 observation/track 一致性；修改观测必须走其接口，不能只改 `Point3D.track` 或 `Image.points2d` 单侧。

### AVX2 与 Eigen ABI

AVX2/FMA 只能加到不跨库边界传递 Eigen 对象的局部数值内核，不能加到整个 `colite_core`。如果库内部按 AVX 对齐构造 `FeatureSet`，而测试或调用方按非 AVX ABI 析构，Eigen 的对齐分配/释放约定可能不一致，最终把偏移后的地址传给 `free`。当前 CMake 仅对 `sift_opencv.cpp` 和 `sift_vlfeat.c` 添加整文件 `-mavx2 -mfma`；descriptor matcher 只给内部 uint8 点积函数增加 `target("avx2")`，不得扩大到公共 Eigen ABI。

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
- CUDA BF：每张图量化后只上传一次。特征数满足 INT8 GEMM 对齐时，将 uint8 描述子无损偏置到 int8，使用 cuBLAS 一次矩阵乘同时生成双向点积，并用描述子和补偿恢复原始精确整数点积；不满足对齐要求时回退 warp 级 `__dp4a`。点积矩阵、双向输出和 cuBLAS handle 由进程级 matcher workspace 按容量增长并持续复用，避免每个 image pair 重复 `cudaMalloc/cudaFree`。两条路径均复用相同的角距离、ratio 和 cross-check 判定；`COLITE_ENABLE_CUDA_MATCHER` 控制编译，CLI 为 `--matcher cuda`。
- FLANN：`flann::L2<uint8_t>`，KDTree 和 search 参数向 COLMAP CPU 路径靠齐。
- pair 模式：exhaustive、sequential、spatial、vocabulary tree。
- spatial 必须有所有图像的位置先验，缺失时直接报错，不静默退化。
- vocabulary tree 支持离线持久层次树；A/B 时必须把 builder 与 matcher 分开计时。
- 两视图验证内部估计 E/F/H，PoseLib 可用时使用 five-point；支持平面、全景、多模型和退化分类。批量验证按 image pair 使用持久线程池并行计算，最后按 pair 顺序写回，避免 630 对全串行。
- `recon_sfm` 的 guided matching 先用初始 F 建立极线走廊，再在走廊内重新搜索描述子，而不是只过滤首轮匹配；这可以恢复被极线外假近邻压掉的真实对应。CPU 后端预计算查询极线项并遍历 16 px 网格；CUDA 后端在同一 kernel 中融合 Fundamental/Sampson 门控、精确 uint8 描述子点积和双向前两名归约，不生成或上传 CPU 候选 CSR。
- E/F/H 使用自适应 LO-RANSAC；局部优化只对支持度已经优于当前最优值的假设执行，避免为每个随机假设重复求解和评分大内点集。
- exhaustive/weak-pair 批处理会为每张图预量化一次描述子；guided 重搜索同时复用每图 16 px 空间网格。CPU BF 的 uint8/512 点积使用局部 AVX2 target，不改变跨库 Eigen ABI；CUDA BF 复用每图显存缓冲。`recon_sfm --matcher cuda` 按 image pair 流水执行初始 GPU 匹配、指定数量的 CPU 几何 worker 和 guided GPU 匹配，并在末尾按 pair 顺序发布结果；这样保留确定性写回，同时消除三个全局阶段之间的 barrier。summary 中 `frontend_pipeline_seconds` 是端到端墙钟；三个阶段字段在 CUDA 流水线下是累计工作量，不能直接相加。CUDA SIFT 提取期间线程池临时为 1，结束后恢复调用方指定线程数，避免后续匹配意外单线程。

主要入口位于 `src/colite/matching` 和 `src/colite/controllers/*matcher*`。

## Incremental SfM

默认焦距初始化与 COLMAP 4.0 对齐：用户显式值优先；EXIF `FocalLengthIn35mmFilm` 使用图像对角线与 43.27 mm 全画幅对角线换算；其次使用 `FocalLength` 与焦平面分辨率；最后才使用 `1.2 × max(width,height)`。前两类 EXIF 值和用户值标为 prior，尺寸回退不是 prior。默认开启焦距 BA；主点和额外畸变参数仍默认固定。

实现位于 `src/colite/sfm`：

- `incremental_initializer`：初始对选择和相对位姿。
- `image_registrar`：2D-3D 注册；PoseLib 可用时 P3P，否则 DLT PnP。
- `incremental_triangulator`：Continue、point-centric multi-view Create、complete/merge；一次 Mapper 运行只构建一次只读 correspondence adjacency，所有局部/全局三角化复用。
- `local_bundle_adjuster` / `global_bundle_adjuster`：Ceres BA、参数 mask 和 gauge fixing；Ceres 线程数继承 SfM 控制器的统一线程设置，不再固定为 1。
- `observation_manager` / `observation_filter`：track 冲突、合并、坏点和坏观测清理。

当前三角化语义：

1. Continue 先检查已三角化候选，以最小 angular error 选择可见点。
2. Create 从 correspondence adjacency 收集多视图观测，生成两视图假设并用 angular error 判内点，再多视图重估。
3. 全局 complete/merge 从 track 全部观测递归传播，merge 使用 track length 加权坐标并验证合并后的全部观测。
4. 全部图像注册后先执行全局 complete/merge 和一次全模型 retriangulate，再进入全局 BA；BA 后再次 complete/merge 和过滤。纯孤立双视图 track 默认忽略。

局部 BA 对共享 Camera 有完整覆盖保护：只有局部问题包含该 Camera 的全部已注册图像时，才允许修改其内参；否则内参保持常量，由全局 BA 使用完整观测统一优化。局部 BA 后的观测过滤也只作用于本次调整图像所观测的点，不再每注册一张图就重复过滤整个模型。这两条规则避免局部焦距变化使窗口外观测突然超阈值并被批量删除。

稀疏点 RGB 仍按注册图像 ID 顺序累加全部有效观测。图像解码与双线性采样使用持久线程池并行，每图结果按原顺序归并，因此颜色求和语义不变；并行解码会以更高峰值内存换取墙钟时间。

在 `building-18` 上 v14 的点数、观测、track 长度和位姿已接近 COLMAP 3.11。剩余差距主要是鲁棒多视图估计器、重投影误差尾部和大规模 correspondence 缓存效率。

## 图像去畸变

实现位于 `src/colite/mvs/image_undistorter.cpp` 和 MVS 控制器。每个输出 PINHOLE ray 通过原相机投影反查源图，支持已实现投影的畸变、鱼眼和薄棱镜模型。输出模型同步更新内参、图像尺寸和稀疏观测。

`blank_pixels`、scale 范围和 max image size 控制视场及尺寸。`resolution_level=1` 表示宽高各缩小一级；若去畸变阶段已经缩放，后续 workspace 不重复缩放。

## PatchMatch Stereo

CPU/CUDA 共享 `PatchMatchView`、深度/法线/代价图、source selection 和诊断结构。CUDA 实现位于 `src/colite/mvs/patch_match_cuda.cu`，CPU/调度位于 `patch_match.cpp`。

CUDA 的平面 patch 评价使用 FP32 bilateral ZNCC。每个
reference candidate/source 只组合一次平面单应矩阵，窗口内以行列增量更新齐次
投影坐标，不再为每个 sample 重做世界坐标、外参和内参投影。RTX 4070 SUPER
一类消费卡的 FP64 吞吐远低于 FP32，因此热路径不得恢复 double 累加；15×15
窗口已有 CPU/CUDA 数值一致回归测试。

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

生产默认不再用固定 `(0,0,-1)` 法线和 5° 小邻域启动。CUDA 初始化在每个
reference 像素的相机朝向半球上均匀采样法线，初始法线扰动为 90°，优化扰动从
约 1 rad（57.2958°）开始并按迭代自适应收缩；这与 COLMAP 的大范围随机初始化
语义一致，避免屋顶、地面等弱纹理平面被错误法线局部最优锁定。兼容性参数
`--patch_match_fixed_initial_normals` 可恢复固定法线，两个角度参数也可显式覆盖，
但只应用于诊断/A/B，不应作为质量默认。

大图几何细化不会再保留“全部旧 normal 快照 + 全部新 normal 结果”两套
12-byte/pixel 常驻数组。每个 reference 的旧 normal 所有权随 refinement job
移动，任务完成即释放；全局只保留所有 source 都需要的旧 depth snapshot。
CUDA 输入图像、source 图、source depth 和初值 depth/normal 直接打包到 pinned
staging，不再先构造同尺寸 `std::vector<float>` 再复制；关闭
`--patch_match_diagnostics` 时，动态 source probability 在设备端初始化，且不再
为了摘要回读整张概率图。算法候选、随机种子、迭代次数和过滤阈值均不变。

默认策略为 `resolution_level=1`、3 次迭代、checkerboard 每迭代一轮红黑更新。checkerboard 会先按邻居已有代价预选两个平面再执行昂贵的光度评价；House 的 4 轮质量配置相对四邻居全评估将墙钟降低 41.5%，覆盖降低 3.33 个百分点，深度 P95 基本不变。动态 source selection 现在默认同时使用 directional HMM 和 CDF sampling，以保持弱纹理区域的源选择连续性。

几何细化与公开的光度金字塔参数分开：即使 `--patch_match_pyramid_levels 1`，几何 pass 内部也至少使用 3 层 coarse-to-fine，并执行 1 个继承光度结果的同步 pass；该 pass 的内层迭代数仍由 `--patch_match_geometric_refinement_iterations` 控制。reference depth/normal 使用光度结果作为初始假设，所有视图的光度 depth 作为不可变 source snapshot。第二个外层同步 pass 在 2296 px 下增加约 103 s 墙钟，只把严格覆盖从 65.84% 提高到 77.37%；单 pass 已高于相同数据上 COLMAP 的历史严格覆盖 61.96%，因此默认不重复。这样保留单层光度阶段的参数/耗时口径，同时避免红黑传播在大面积弱纹理表面停滞。几何代价按 Colite 的 `0.5*(1-ZNCC)` 量纲缩放，源深度无效时使用截断最大惩罚；关闭遮挡/冲突硬门控时，候选直接进入 COLMAP 风格软重投影代价。CUDA 光度分支只评价真实平面单应，不再额外取 ±7.5° 旋转补丁最小值，也不叠加固定 source/incident cost 惩罚。详细原子诊断默认关闭，因为会显著影响 CUDA 热路径。

最终几何一致性图直接构造 CSR offsets/IDs，native binary v2 按连续数组批量写出；构建和第二遍读取过滤均按 `num_threads` 有界并行。结果数组在构图阶段只读，过滤阶段每个任务只修改自己的 view，因此不会改变几何判定和输出顺序。binary dense-map 模式同时覆盖 depth、normal 和 cost，不再把 cost 以十进制文本写入大文件。

历史 `docs/superpowers` 计划中的 source scoring、几何代价、传播、稳定初始化、binary maps、并行缓存、质量诊断、法线 refinement 和 CUDA 对齐任务均已落地，其长期行为已归纳在本节；原计划文件不再保留。

## Stereo Fusion

Fusion 读取持久化 dense maps 和 consistency graph，从 seed 像素沿一致投影构造
track。生产入口按 reference 顺序处理，已完成的早期 reference 不再作为 source
重复遍历；默认不再次做成本很高的 mutual graph 反向查找。source pixel 搜索半径
默认为 0，与 COLMAP 的最近投影像素语义一致。需要兼容低精度旧深度图时仍可显式
增大半径。

持久化入口使用由 `image_cache_size` 控制的线程安全 LRU view cache；缓存项同时
包含缩放后的 RGB 图像、depth、normal、cost（需要时）和 consistency graph。
全局 LRU 锁只保护命中、插入和淘汰，图像解码及 dense map/graph 文件读取在每视图
锁下于全局锁外完成，避免 8 个 worker 被慢 I/O 串行化。淘汰时由正在使用该视图的
`shared_ptr` 保证生命周期。Fusion 不再为全部有效像素预计算 double XYZ、法线和
颜色，只在当前 track 中按需保留这些数据。

PatchMatch 与 Fusion 在控制器中是两个明确的资源生命周期：PatchMatch 先完成最终 dense map 和 native CSR consistency graph 写出并退出作用域，随后 Fusion 才从持久化结果按缓存上限加载。几何一致性图按视图计算、压紧和写出，第二个流式读取 pass 再应用过滤，避免同时保留未过滤深度副本、全部 append-friendly graph 和 COLMAP/native 两套图。默认只写 native graph；reader 同时兼容旧 native v1、当前 CSR v2 和 COLMAP 格式。

自动 PatchMatch 深度范围与 COLMAP 对齐：每视图对正稀疏深度排序，取 1%–99% 分位数，并分别乘以 0.75/1.25 扩展。不得改回原始最小/最大值；单个稀疏远端异常点会显著扩张 CUDA 逆深度搜索范围，使低视差远平面假设通过几何一致性并在 Fusion 中形成远端薄片。

Fusion 图遍历只用当前观测寻找下一候选，但重投影、双线性深度和相对深度误差
始终相对原始 seed 观测计算，与 COLMAP 的 `fused_ref_point` 约束一致。不得把
校验基准改为每一跳的 current point：逐跳都合法的小位移会累积成大空间漂移，
合并相邻但不共面的表面，最终表现为平面网格坑洼和异常长 track。

seed-track 使用 `core/parallel_for` 按图像行带并行，原子 CAS 认领像素，线程局部收集点、track、可见性和误差样本后合并。投影到 source 像素后先检查全局 visited，已认领像素不再重复做反投影、法线和双向深度验证；PINHOLE 路径预计算内外参和相机中心，并以 cosine 阈值代替热路径 `acos`。并发认领可能让少量边界 seed 归属不同，因此点数允许极小非确定性，但几何阈值不变。

多视图一致的错误远端表面不能靠最小 track observation 数过滤。`--dense_cloud_spatial_bounds_quantile <q>` 为显式鲁棒范围保护：在通过 track 条件的点上按 XYZ 各自裁掉上下 `q` 空间尾部，输出 `fused_clean.ply` 和按原因分类的删除统计；默认 `q=0` 完全关闭。House 使用 `q=0.001` 删除 0.418% 点后，包围盒对角线从约 100.8 恢复到 15.43，接近 COLMAP 的 17.02。该开关不应对包含合法稀疏远景的场景盲目启用。

MVS workspace 可分别控制灰度图和 RGB 图加载。PatchMatch 使用灰度；持久化 Fusion 只解码 RGB，并在没有 RGB 的旧测试/调用方中回退为三通道同灰度。

持久化 Stereo Fusion 的一致性图使用 CSR 连续表示：构建期仍可逐像素追加，
落盘读取时一次批量读取 binary words 并直接生成 offsets/IDs，避免为每个空像素
构造 `std::vector`。depth/normal binary 也直接读入最终数组。融合核心按 worker
复用 track/median 临时缓冲，每像素缓存使用 double XYZ、float normal 和 uint8
RGB。几何过滤后的持久化融合不再重复加载 text cost map。图像对拒绝计数和误差
分布由 `--fusion_diagnostics` 显式开启；点云、可见性、track 质量和
`fusion_timing.txt` 始终输出。

大图 Fusion 热路径继续使用以下等价优化：CSR offsets 在内存中保持 uint32；
worker 使用按 view 下标的弱引用索引和 4 项强引用热集，无锁复用其他 worker
仍持有的视图，同时不固定驻留全部 dense maps；ImageId 到 view index 使用稠密
查找表；同一投影位置的双线性深度只采样一次。track 的 9 个中位数使用与 COLMAP 百分位实现一致
的 `nth_element`，source 法线的世界变换结果在几何检查和融合点构造之间复用。
持久化 Fusion 不超过 64 个视图时，以 uint64 mask 暂存可见性并直接写出相同的
COLMAP binary 格式，避免每个融合点一次小 vector 分配。默认 180° view gate
不会过滤观测；关闭 `--fusion_diagnostics` 时不再为全部观测计算仅供诊断的视角，
`fusion_summary.txt` 以 `num_view_angle_tracks 0` 明确标记未采集。

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

当前流程：标定视图评分 → 三角形 AABB BVH ray occlusion → 邻域 label smoothing → 同图相邻面 patch → atlas packing → 全局相机曝光偏移 + 局部 seam 顶点偏移场 → UV/颜色烘焙 → 可选 inpaint。

BVH 线段求交排除当前面及共享目标顶点邻面。大网格应先简化，并根据纹理细节
逐级增加 atlas 上限。

BVH 遮挡判定使用 `occlusion_relative_tolerance` 作为目标端相对容差：默认
1% 范围内的近共面交点按重建表面噪声处理，不再把屋顶、地面上的轻微层叠误判
为遮挡；明显位于目标前方的表面仍拒绝。严格选择后，仅当至少两个边邻面已有
纹理时，未选择面才会从邻面候选视图中选择一个仍满足完整投影和正面约束的视图。

最终仍没有可靠相机视图的面不再把三个 UV 压到统一灰色 texel。Texturer 为每个
此类面分配独立小 atlas patch，并对 Poisson/QEM 保留的 RGB 顶点色做重心插值。
该 fallback 不改变网格拓扑、不穿透遮挡，也不会伪造相机纹理；其目标是消除
退化 UV 形成的三角形/异形灰色孔洞。

跨视图颜色校正分为两级。第一级求每个相机的常量 RGB 偏移，校正整体曝光；
第二级只在不同视图 patch 的共享边顶点建立带正则的最小二乘残差约束，并在
三角形内部按重心坐标插值局部偏移，从而处理同一对图像在平面不同位置出现的
空间变化色差。House 的 seam 采样中，平均/中位 BGR L2 色差分别下降
23.8%/35.1%，峰值内存基本不变。

该实现依据 Waechter 2014 的全局 seam leveling 能量独立编写，并以
BSD-3-Clause 的 COLMAP 行为作 A/B。OpenMVS 仅在公开 CLI/论文层面对
global/local seam leveling、共面 virtual faces、孔洞闭合和 mesh refinement
进行功能调研；不得复制或改写其 AGPL-3.0 核心源码进入 Colite。

## I/O 与颜色

- 稀疏 `points3D.txt`/PLY 写真实 `Point3D.color`，不再固定白色。
- 稠密 `fused.ply` 使用 binary little-endian PLY，float XYZ/normal 和 uchar RGB；Delaunay 可见性另存 binary。
- COLMAP text/binary 读取保留 RGB；去畸变稀疏模型也保留颜色。
- OpenMVG 需要 `ComputeSfM_DataColor` 后处理输出 `colorized.ply`；其代表视图策略与 Colite/COLMAP 的多观测均值不同。

## 已知差距

1. Texturer 的 RGB 顶点色 fallback 能消除退化 UV 孔洞，但细节低于真实相机
   投影；全局曝光 + 局部 seam 顶点场已降低跨视图色差，仍未实现局部窄带
   Poisson seam blending 和共面区域级视图选择。
2. House SfM 的共享 Camera 局部 BA/全模型过滤错误和 guided matching 缺口均已修复；相同 CUDA VLFeat 8192 配置最终为 48,794 点、147,407 observations。与 COLMAP 的剩余差异主要是 descriptor 数、SIFT 实现和相机模型，而不是 GPS。
3. House GUI High 等价 2400×1597、20 sources、radius 5 配置中，平面单应
   + FP32 ZNCC 将 PatchMatch 从 3537.21 s 降至 772.17 s，比同输入 COLMAP
   1504.87 s 快 48.7%。峰值 RSS 8,968,060 KiB，仍高于 COLMAP，后续重点是
   CUDA workspace 和 host dense-map 生命周期，而非继续牺牲数值覆盖。
4. 同一配置的持久化 Fusion 从 1145.55 s 降至 36.47 s；36 图 cache 容量设为
   36 时仅比 COLMAP 30.79 s 慢 18.4%，输出 2,529,490 个 RGB+normal 点。
   cache 32 会因最后 4 个视图反复淘汰而退化到 126.33 s，因此内存允许时应令
   `image_cache_size` 覆盖活跃视图；默认仍保持有界，避免大数据集无界驻留。
5. Delaunay 需要更多非退化真值场景验证，而不只是与 COLMAP 输出互比。
6. 历史 ASCII double PLY 已替换为 binary little-endian float PLY；外部脚本仍需
   同时支持两种格式以读取旧测试产物。
