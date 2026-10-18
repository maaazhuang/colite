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
- CUDA BF：每张图量化后只上传一次。特征数满足 INT8 GEMM 对齐时，将 uint8 描述子无损偏置到 int8，使用 cuBLAS 一次矩阵乘同时生成双向点积，并用描述子和补偿恢复原始精确整数点积；不满足对齐要求时回退 warp 级 `__dp4a`。点积矩阵、双向输出和 cuBLAS handle 由进程级 matcher workspace 按容量增长并持续复用，避免每个 image pair 重复 `cudaMalloc/cudaFree`。两条路径均复用相同的角距离、ratio 和 cross-check 判定；`COLITE_ENABLE_CUDA_MATCHER` 控制编译，CLI 为 `--FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1`。
- FLANN：`flann::L2<uint8_t>`，KDTree 和 search 参数向 COLMAP CPU 路径靠齐。
- pair 模式：exhaustive、sequential、spatial、vocabulary tree。
- spatial 必须有所有图像的位置先验，缺失时直接报错，不静默退化。
- vocabulary tree 支持离线持久层次树；A/B 时必须把 builder 与 matcher 分开计时。
- 两视图验证内部估计 E/F/H，PoseLib 可用时使用 five-point；支持平面、全景、多模型和退化分类。批量验证按 image pair 使用持久线程池并行计算，最后按 pair 顺序写回，避免 630 对全串行。
- `recon_sfm` 的 guided matching 先用初始 F 建立极线走廊，再在走廊内重新搜索描述子，而不是只过滤首轮匹配；这可以恢复被极线外假近邻压掉的真实对应。CPU 后端预计算查询极线项并遍历 16 px 网格；CUDA 后端在同一 kernel 中融合 Fundamental/Sampson 门控、精确 uint8 描述子点积和双向前两名归约，不生成或上传 CPU 候选 CSR。
- E/F/H 使用自适应 LO-RANSAC；局部优化只对支持度已经优于当前最优值的假设执行，避免为每个随机假设重复求解和评分大内点集。
- exhaustive/weak-pair 批处理会为每张图预量化一次描述子；guided 重搜索同时复用每图 16 px 空间网格。CPU BF 的 uint8/512 点积使用局部 AVX2 target，不改变跨库 Eigen ABI；CUDA BF 复用每图显存缓冲。`recon_sfm --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1` 按 image pair 流水执行初始 GPU 匹配、指定数量的 CPU 几何 worker 和 guided GPU 匹配，并在末尾按 pair 顺序发布结果；这样保留确定性写回，同时消除三个全局阶段之间的 barrier。summary 中 `frontend_pipeline_seconds` 是端到端墙钟；三个阶段字段在 CUDA 流水线下是累计工作量，不能直接相加。CUDA SIFT 提取期间线程池临时为 1，结束后恢复调用方指定线程数，避免后续匹配意外单线程。

主要入口位于 `src/colite/matching` 和 `src/colite/controllers/*matcher*`。

学习前端通过可选 ONNX Runtime 提供 ALIKED N16Rot/N32、ALIKED cosine BF
以及 SIFT/ALIKED LightGlue。官方模型 basename 会自动匹配内置 SHA-256 清单；
自定义显式路径保持离线可用。EXIF Orientation 1–8 在统一图像读取层应用。

全局 SfM 首版位于 `src/colite/sfm/global_mapper.cpp`：verified view graph
依次经过约束筛选、旋转平均、带边尺度的相机中心求解、track 三角化/合并和
全局 BA，并由 `recon_sfm --Colite.mapper global` 调用。该实现已经形成可运行垂直链路，
旋转异常边剔除后会重新求解，定位只消费最终有效边，并通过稀疏 QR 检查秩；
定位过滤每轮重新检查锚点连通分量，不输出无约束相机。固定旋转 BA、重三角化
和观测过滤已接通，但焦距 calibration 与 track 驱动定位仍需增强，不能视为
与完整 GLOMAP 数值行为等价。

层次 SfM 位于 `src/colite/sfm/hierarchical_mapper.cpp`：verified view graph 先按
目标规模、最大规模和重叠图像数分区，每个 cluster 在显式 image-ID 约束下执行
局部增量重建；局部模型按注册规模排序，通过 verified 3D-3D 对应估计鲁棒 Sim(3)
并合并。暂时无法合并的模型会在其他模型并入后重试，以覆盖链式重叠；合并结果最终
执行一次全局 BA。`recon_sfm --Colite.mapper hierarchical` 与
`recon_auto --sfm.Colite.mapper hierarchical` 已接入同一输出协议。局部任务已经支持有界并行，
通过 `--Colite.hierarchical_num_workers` 配置；大规模调度、模型图联合优化和系统 A/B
仍是后续增强项。

structure-less 注册是增量 mapper 的显式回退：PnP 失败后，从至少两个已注册
邻居的 verified relative pose 估计旋转并交会世界坐标平移射线。双视图平移仅作为
方向，距离由已注册邻居的基线决定；中心误差阈值按该基线归一化。共线射线不唯一，
因此明确失败。候选需通过至少两个邻居的正深度、重投影及视差验证，才允许注册
并创建 track。默认关闭。该回退仍不是完整 generalized relative pose 求解器。
`transitive_matcher` 只生成新候选 pair，匹配证据仍来自真实描述子匹配
和 two-view geometry verification。

## Incremental SfM

### Guided/verified 匹配图一致性

`matching/match_graph_consistency` 在 guided matching 和两视图几何验证完成后统一规范
verified correspondences。图像对内按跨图支持度、三视图闭环、Sampson 误差和稳定索引顺序
执行一对一选择；同时统计跨图连通分量中“同一图像出现多个特征”的冲突规模。

跨图冲突不能在相机位姿未知时直接按连通分量删边：House exhaustive 图中少量错误桥接会把大量
真实轨迹连成大组件，硬裁剪会显著损失有效观测。因此 verified 证据保持完整，冲突延后到
pose-aware 阶段处理：

- PnP 注册一张新图时，同一 3D 点只链接该图中重投影误差最小的 inlier 特征；
- track completion 不允许向同一 3D 点加入同一图像的第二个观测；
- merge 遇到同图冲突时，以合并后 3D 位置的重投影误差选出唯一观测，再完成轨迹合并。

`recon_sfm/summary.txt` 会记录 verified 一致性处理前后内点数、pair 冲突数、冲突组件数和冲突
观测数，便于持续回归。

图像导入递归枚举受支持图像并保存相对根目录的通用路径。默认 Auto 分组对齐
COLMAP ImageReader：EXIF Make、Model、焦距标识及宽高完整一致时共享 Camera；
缺字段的图像使用唯一分组。另提供 Single、Per-folder、Per-image，旧
Shared/Individual 名称作为兼容别名。所有组均支持混合目录和重复文件名；
真正共享内参的组要求图像尺寸一致。

焦距初始化中用户显式值优先；EXIF `FocalLengthIn35mmFilm` 使用本图对角线与
43.27 mm 全画幅对角线换算；其次使用 `FocalLength` 与焦平面分辨率；最后使用
可配置的 `default_focal_length_factor × max(width,height)`。组内 prior 取
中位数，并拒绝相对跨度超过 5% 的冲突。该策略在
`controllers/sfm_camera_setup` 中统一实现，完整 `recon_sfm` 和分阶段
`feature_extractor → matcher → mapper` 共用。默认模型改为 SIMPLE_RADIAL；
默认优化焦距和畸变/额外参数，主点固定。

Core 当前登记并完整支持 12 个模型：

`SIMPLE_PINHOLE`、`PINHOLE`、`SIMPLE_RADIAL`、`RADIAL`、`OPENCV`、
`OPENCV_FISHEYE`、`FULL_OPENCV`、`FOV`、`SIMPLE_RADIAL_FISHEYE`、
`RADIAL_FISHEYE`、`THIN_PRISM_FISHEYE`、`RAD_TAN_THIN_PRISM_FISHEYE`。
投影、反投影、BA、去畸变和 text/binary 读写共用同一参数登记表。反投影对
畸变平面使用带数值 Jacobian、信赖半径和回溯的二维求解，再将鱼眼角度平面
转回透视 ray；失败返回明确状态，不输出 NaN。合法性检查覆盖参数个数、有限值、
正焦距、FOV 范围及可配置的异常焦距/畸变阈值。暂不登记 COLMAP 4.0 新增的
4 个模型。

实现位于 `src/colite/sfm`：

- `incremental_initializer`：初始对选择和相对位姿。
- `image_registrar`：2D-3D 注册；PoseLib 可用时 P3P，否则 DLT PnP。
- `incremental_triangulator`：Continue、point-centric multi-view Create、complete/merge；一次 Mapper 运行只构建一次只读 correspondence adjacency，所有局部/全局三角化复用。
- `local_bundle_adjuster` / `global_bundle_adjuster`：Ceres BA、参数 mask 和 gauge fixing；Ceres 线程数继承 SfM 控制器的统一线程设置，不再固定为 1。
- `observation_manager` / `observation_filter`：track 冲突、合并、坏点和坏观测清理。
- `reconstruction_manager`：已有模型续建、verified 连通分量枚举、多模型生命周期和模型间
  Sim(3) 合并。

多模型与断点续建语义：

1. 输入模型支持 COLMAP text 和 binary 格式。恢复时以图像名而不是历史 ID 关联当前图像清单，因此可在
   原数据集加入新图或改变枚举顺序；当前图像路径和位置先验保留，历史优化内参、位姿、点及
   track 恢复。
2. 相机模型和图像尺寸必须与当前 schema 兼容；若当前策略要求多个图像共享同一 Camera，而
   输入模型为它们提供了不同内参，则明确报错，不静默选择最后一组参数。
3. 恢复前校验已注册图像的特征数量、坐标和索引顺序与当前 MatchGraph 一致；只有历史
   `POINTS2D` 为空的已注册图像可从当前特征安全补齐。改变 SIFT 后端、参数或特征排序后必须
   重新建模，不能复用旧 track 索引。
4. seed 模型先继续执行 2D-3D 注册。其余图像按 verified 匹配图的连通分量、由大到小重复
   初始化；初始化器和注册候选均受当前分量 image-ID 集约束，避免跨分量污染。
5. 模型合并只使用 verified 匹配连接的已三角化观测建立一对一 3D-3D 对应，经过确定性鲁棒
   Sim(3) 假设、尺度相关残差门限和全内点重估后，才迁移相机、点与观测。合并后重新执行
   complete/merge、补注册、过滤和全局 BA。
6. `sparse/` 或 `recon_sfm` 输出根目录保存最大模型；`models/<index>/` 保存全部模型。
   `multiple_models=0` 只保留一个初始化机会，`merge_models=0` 保留可对齐模型为独立结果。

incremental、global 与 hierarchical 三种 Mapper 均已形成可运行链路。后两者仍需在
数百至数千图数据集上验证退化行为、内存和 wall time；hierarchical 后续重点是并行局部任务
与模型图联合优化，global 后续重点是更完整的 view-graph calibration、固定旋转的位置/结构
优化和分阶段重三角化。

当前三角化语义：

1. Continue 先检查已三角化候选，以最小 angular error 选择可见点。
2. Create 从 correspondence adjacency 收集多视图观测，生成两视图假设并用 angular error 判内点，再多视图重估。
3. 全局 complete/merge 从 track 全部观测递归传播，merge 使用 track length 加权坐标并验证合并后的全部观测。
4. 全部图像注册后先执行全局 complete/merge 和一次全模型 retriangulate，再进入全局 BA；BA 后再次 complete/merge 和过滤。纯孤立双视图 track 默认忽略。

局部 BA 对共享 Camera 有完整覆盖保护：只有局部问题包含该 Camera 的全部已注册图像时，才允许修改其内参；否则内参保持常量，由全局 BA 使用完整观测统一优化。局部 BA 后的观测过滤也只作用于本次调整图像所观测的点，不再每注册一张图就重复过滤整个模型。这两条规则避免局部焦距变化使窗口外观测突然超阈值并被批量删除。

稀疏点 RGB 仍按注册图像 ID 顺序累加全部有效观测。图像解码与双线性采样使用持久线程池并行，每图结果按原顺序归并，因此颜色求和语义不变；并行解码会以更高峰值内存换取墙钟时间。

在 `building-18` 上 v14 的点数、观测、track 长度和位姿已接近 COLMAP 3.11。剩余差距主要是鲁棒多视图估计器、重投影误差尾部和大规模 correspondence 缓存效率。

## 图像去畸变

实现位于 `src/colite/mvs/image_undistorter.cpp` 和 MVS 控制器。输出 PINHOLE
像素先构造 ray，再通过原相机投影反查源图；稀疏观测去畸变复用 Core 的统一
稳健反投影，支持上述全部12种模型。输出模型同步更新内参、图像尺寸和稀疏观测。

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

动态 source CDF 默认将 NCC-HMM 概率与 COLMAP 式几何先验相乘：小于 1°
的三角化角连续降权，源相机入射角使用 `sigma=0.9` 的概率，并按当前平面单应
映射后的 patch 面积比抑制严重上/下采样源图。该先验只改变 15 次 CDF 抽样权重，
不放宽最终 1 px/1%/法向阈值。House 800 px A/B 中，PatchMatch 墙钟
98.05→98.23 秒，Fusion 点数 287,344→303,978（+5.79%），平均光度代价
0.2635→0.2249。通用 source 列表仍默认使用 weighted 模式，以免极小基线数据被
1°门限清空；High/COLMAP 对齐可显式使用 `--source_selection_mode colmap`。

默认策略为 `resolution_level=1`、3 次迭代、checkerboard 每迭代一轮红黑更新。checkerboard 会先按邻居已有代价预选两个平面再执行昂贵的光度评价；House 的 4 轮质量配置相对四邻居全评估将墙钟降低 41.5%，覆盖降低 3.33 个百分点，深度 P95 基本不变。动态 source selection 现在默认同时使用 directional HMM 和 CDF sampling，以保持弱纹理区域的源选择连续性。

几何细化与公开的光度金字塔参数分开：即使 `--patch_match_pyramid_levels 1`，几何 pass 内部也至少使用 3 层 coarse-to-fine，并执行 1 个继承光度结果的同步 pass；该 pass 的内层迭代数仍由 `--patch_match_geometric_refinement_iterations` 控制。reference depth/normal 使用光度结果作为初始假设，所有视图的光度 depth 作为不可变 source snapshot。第二个外层同步 pass 在 2296 px 下增加约 103 s 墙钟，只把严格覆盖从 65.84% 提高到 77.37%；单 pass 已高于相同数据上 COLMAP 的历史严格覆盖 61.96%，因此默认不重复。这样保留单层光度阶段的参数/耗时口径，同时避免红黑传播在大面积弱纹理表面停滞。几何代价按 Colite 的 `0.5*(1-ZNCC)` 量纲缩放，源深度无效时使用截断最大惩罚；关闭遮挡/冲突硬门控时，候选直接进入 COLMAP 风格软重投影代价。CUDA 光度分支只评价真实平面单应，不再额外取 ±7.5° 旋转补丁最小值，也不叠加固定 source/incident cost 惩罚。详细原子诊断默认关闭，因为会显著影响 CUDA 热路径。

几何 refinement 默认将光度 source 排名与几何共识解耦：光度部分继续使用动态
selection/best-sample 聚合；几何部分对所有有效 source-depth 的 forward-backward
重投影误差排序，取较低一半均值，并以原权重加入候选代价。CUDA 几何阶段使用正无穷
作为无效哨兵，避免合法的“光度 + 几何”组合代价大于 1 时被误删。House 2296px
相对旧组合点数增加 2.10%，24-NN median/P95 降低 21.52%/10.71%，wall 仅增加
0.26%；可用 `--patch_match_legacy_geometric_aggregation` 恢复旧 source 混排语义。

最终几何一致性图直接构造 CSR offsets/IDs，native binary v2 按连续数组批量写出；构建和第二遍读取过滤均按 `num_threads` 有界并行。结果数组在构图阶段只读，过滤阶段每个任务只修改自己的 view，因此不会改变几何判定和输出顺序。binary dense-map 模式同时覆盖 depth、normal 和 cost，不再把 cost 以十进制文本写入大文件。

历史 `docs/superpowers` 计划中的 source scoring、几何代价、传播、稳定初始化、binary maps、并行缓存、质量诊断、法线 refinement 和 CUDA 对齐任务均已落地，其长期行为已归纳在本节；原计划文件不再保留。

## Stereo Fusion

内存式 `recon_mvs --fuse_stereo` 不再把 PatchMatch 的 depth/normal/cost 先复制到
控制器向量、再由 Fusion API 二次复制；maps 所有权直接移交给共享 view data，最终
dense-map 输出继续复用同一份存储。CUDA geometric refinement 通过不可变 depth
指针引用全局 snapshot，避免每 job 深拷贝 source maps。大于等于 200 万像素时只
保留一个 refinement workspace；refinement 后立即释放全部 device/pinned workspace，
再以最多 2 worker 构建 consistency graph。House 2296px 全流程峰值由 7.09 GiB
降到 5.53 GiB，Fusion core 保持约 29.4 s。

## Incremental Mapper 与 BA 热路径

全局 `CompleteAndMergeTracks` 使用 dirty-set fixed point：第一轮检查全部三维点，后续
只重新检查本轮确实增加观测或发生合并的目标点。Mapper 对每张新注册图完成局部三角化
后只执行一次 completion，不再在每个相邻图对之后重复全图 completion。该策略保留
complete/merge 的收敛语义，同时显著降低 House 的全局 completion 时间。

Local/Global BA 对默认 SIMPLE_PINHOLE、PINHOLE 和 SIMPLE_RADIAL 相机路径使用解析
投影 Jacobian 与相机参数 Jacobian；姿态部分使用包含单位化链式导数的 xyzw 四元数解析
Jacobian。其余九种已登记相机模型保留数值差分回退，因此模型覆盖能力不受影响。
House 回归中 Mapper 为 62.20 s、46,192 点、145,710 观测、平均重投影 1.13196 px，
相对 COLMAP 4.0 的 54.55 s 为 1.14 倍。

全局 track completion 额外处理 verified 组件中的跨中间节点证据。候选阶段按未关联
Point2D 并行执行最多四跳只读 BFS，每个目标保留最多四个最低重投影候选；随后按误差和
观测 ID 确定性排序，在单线程中执行一图一观测检查和 Reconstruction 修改。这样不会在
并行阶段产生数据竞争，也不会因逐 Point3D 重复遍历大组件而拖慢 Mapper。House 得到
146,531 观测、mean track 3.17579、重复观测 0，Mapper 61.92 s。

后续隔离实验把 COLMAP 4.0 数据库 verified 图直接交给 Colite Mapper；Colite 得到
40,495 点、154,880 观测、mean track 3.824670、同图重复观测 0，高于 COLMAP 在
同一图上的 3.738058。Colite 自有图 55.6% 的边无精确三视图闭环，而 COLMAP 为
26.8%，因此剩余长轨迹差距定位到前端图拓扑。此前生产 completion 只在第一次全局
complete/merge 对四跳无安全候选的目标继续第五跳；该基线 House 为 146,640 观测、
mean 3.179876、Mapper 61.653 s。当前首轮允许继续到第六跳，BA/过滤后的重复轮次仍
固定四跳，避免深搜弱连通组件造成耗时回退。

### Guided matching 比率语义

standalone `sequential/exhaustive/spatial/vocab_tree matcher` 支持 `--guided_matching`、`--guided_match_ratio` 和 `--guided_min_inlier_gain`。guided ratio 独立进入 geometry cache hash，不再被 `nearby_match_ratio` 隐式抬高；standalone 默认 0.8，用于对齐 COLMAP High 的同描述子语义。

`recon_sfm` 同样开放 `--guided_match_ratio`，但生产默认暂为 0.95。原因是当前冻结 P1 后，CUDA VLFeat 严格输出 8192 特征/图，0.8 会把 House observations 从 146,640 降到 79,351。调用方可显式用 0.8 做同描述子对照。全局跨图冲突不能在未知位姿时硬拆；House A/B 会把 observations 降到 123,012，因此继续由 PnP、complete/merge 和重投影过滤做位姿感知消歧。

### 最终位姿双描述子轨迹补全

全局 BA 后执行一次 pose-guided completion。对每个尚未观测某注册图像的 Point3D，将其投影到该图像，只搜索 2 px 邻域内的未分配 SIFT 特征；目标描述子必须与轨迹中至少两条已有描述子一致，第二高归一化余弦相似度对应的角距离不得超过 0.301 rad。候选按第二/第一描述子相似度、重投影误差、轨迹长度和稳定 ID 排序，串行提交时再次检查一图一观测约束。新增观测随后进入全局过滤和最终 BA。

该路径不创建或删除 Point3D，不依赖 verified 巨型冲突组件的深层遍历，也不改变 SIFT 上限与多方向语义。House 配合首轮自适应六跳后得到 152,013 observations、mean track 3.300469、重复观测 0，平均重投影误差 1.118657 px；pose-guided 阶段新增 5,196 observations、耗时 0.105 s，完整 Mapper 为 61.989 s。稀疏阶段缓存版本为 `recon_sfm_resume_multimodel_v6_pose_guided_tracks`，旧 checkpoint 会自动重算 Mapper。

## Stereo Fusion 持久化入口

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

BVH 主视图选择将目标端相对容差限制为 0.5%，避免把 1% 范围内所有近端交点
无条件当作表面噪声。配置的完整 `occlusion_relative_tolerance` 只用于受近共面
双邻面支持的孔洞恢复；明显位于目标前方的表面仍拒绝。

最终仍没有可靠相机视图的面默认映射到中性 atlas texel，不再逐面建立 RGB
顶点色 patch，以免遮挡拒绝面形成明显三角纹理块。兼容旧输出时可显式启用
`use_vertex_color_fallback`。

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

1. Texturer 已实现近共面小视图孤岛合并和受拓扑支持的孔洞恢复；全局曝光 +
   局部 seam 顶点场已降低跨视图色差，仍未实现局部窄带 Poisson seam blending。
2. House SfM 的共享 Camera 局部 BA/全模型过滤错误和 guided matching 缺口均已修复。EXIF GPS 现可转换为局部 ENU，并可选用于模型 Sim(3) 对齐和最终全局 BA；无 GPS 的 House 长轨迹差异仍主要来自 verified 匹配图闭环拓扑，而不是位置先验。
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
### 纹理视图连续性与遮挡策略

纹理主视图选择使用 BVH 射线遮挡、投影面积和法向夹角。为避免带噪平面被 1% 端点容差整体误判为可见，主选择的端点噪声带上限为 0.005；配置的完整 `occlusion_relative_tolerance` 仅用于后续孔洞恢复。恢复面必须同时满足：同一相邻视图至少两个边邻面支持、相邻法向夹角不超过约 20°、投影有效，并通过带配置容差的射线遮挡检查。

多数标签平滑后，小于 `view_selection_min_component_size`（默认 64 面）的近共面视图孤岛会在整块候选视图均有效且总投影面积至少保留 50% 时合并。这减少屋顶、地面等平面上的单三角 atlas chart，同时不会跨越屋脊等真实折线。

未选中视图的面默认使用中性 atlas 坐标，不再逐三角创建顶点色 fallback patch；后者容易把遮挡拒绝面显示为明显的三角纹理碎片。如需兼容旧行为，可显式设置 `use_vertex_color_fallback=1`。

## Fusion 候选遍历与法线方向约定

几何过滤已经把 consistency graph 支持不足的像素同步置为无效深度、无效法线和无效代价。持久化 Fusion 不再只沿逐像素 graph 二次限制 track，而是按 MVS source scoring 得到的有序重叠视图列表重新投影，并执行 seed-relative 深度、重投影和法线阈值检查。这与 COLMAP 4.0 默认 Fusion 的重叠图像遍历语义一致；native graph 仍由 PatchMatch 保存并用于几何过滤，核心 Fusion API 也保留显式 graph 模式供测试和特殊调用方使用。

PatchMatch 与 Fusion 的候选数量不再共用一个上限。Fusion 默认独立按共享稀疏点
数排序并检查最多 50 个重叠视图（COLMAP `check_num_images=50` 语义），PatchMatch
继续使用自己的较小 source 集。House High 的 COLMAP maps 回归中，20-view
Colite Fusion 为 2,591,903 点；改为独立 50-view 后为 2,666,592 点，与 COLMAP
自身 2,667,152 仅差 560 点（0.02%）。

COLMAP 默认 photometric PatchMatch 不一定写 consistency graph。读取不到图时必须保留真正的空 `ConsistencyGraph` 并使用 source-view fallback；不能创建“有尺寸但零连接”的图，否则每个有效像素都会被误判为 graph 支持不足。

平面单应代价对法线 `n` 与 `-n` 不敏感，因此 PatchMatch 优化本身不能稳定法线符号。所有 PatchMatch 最终法线在几何过滤、Fusion 和落盘前统一朝向参考相机；持久化 Fusion 加载旧图或外部图时也执行同一幂等规范化。House High 受控实验中，这一修复把 Colite 图经 Colite Fusion 的点数从 2,298,163 提高到 2,386,766；同一 Colite 图经 COLMAP Fusion 为 2,395,492，差距降至 0.36%。

binary dense-map reader 同时接受 Colite 自有格式和 COLMAP 的 `width&height&channels&` 格式。COLMAP normal map 的 payload 是完整 x/y/z 通道平面，读取时必须转换为 Colite 的逐像素 xyz 交错布局；不能把 payload 直接解释为 `Eigen::Vector3f[]`。
