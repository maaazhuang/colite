# Colite 命令行参考

本文对应 Colite 0.1.0。参数解析器和当前构建的帮助是最终依据：

```bash
cd /mnt/d/Code/CXX/colite
COLITE=./build-wsl-colite-cuda/src/colite/exe/colite
$COLITE --help
$COLITE <command> --help
```

统一使用 `colite <command>`。少数命令同时生成独立可执行文件，但参数相同。所有示例输出到新的 `output/<功能>_<场景>_<日期>/`，不要写入构建目录。

## 命令总览

| 命令 | 输入 | 输出/职责 |
|---|---|---|
| `feature_extractor` | 原图 | SfM workspace、特征缓存 |
| `exhaustive_matcher` | SfM workspace | 全 pair 匹配和几何 |
| `sequential_matcher` | SfM workspace | 时序邻近匹配和几何 |
| `spatial_matcher` | 带位置先验 workspace | 空间邻近匹配和几何 |
| `vocab_tree_builder` | 特征 workspace | 持久层次词树 |
| `vocab_tree_matcher` | workspace、可选词树 | 检索候选匹配和几何 |
| `transitive_matcher` | 已有 verified workspace | 沿已验证图生成受控新 pair，再执行真实匹配和几何验证 |
| `mapper` | 已匹配 workspace | 稀疏模型和带 RGB PLY |
| `compare_colmap` | 两个 COLMAP 模型 | Sim(3) 位姿/结构差异报告 |
| `model_converter` | COLMAP TXT/BIN 模型 | 保留模型字段，转换为 TXT、BIN 或两种格式 |
| `model_analyzer` | COLMAP 稀疏模型 | 相机、图像、点、观测、track、误差和边界统计 |
| `model_transformer` | 稀疏模型、3×4/4×4 矩阵 | 应用 Sim(3) 并写兼容模型 |
| `model_aligner` | 稀疏模型、按图像名 XYZ | 鲁棒位置对齐 |
| `model_cropper` | 稀疏模型、三维边界 | 裁剪点及其反向观测 |
| `model_splitter` | 稀疏模型 | 按 track 连通分量输出多个模型 |
| `bundle_adjuster` | 稀疏模型 | 独立全局 BA，可选位置先验 |
| `point_triangulator` | workspace、已有模型 | 独立重三角化、complete/merge |
| `image_registrator` | workspace、已有模型 | 独立注册指定或全部未注册图像 |
| `recon_sfm` | 原图 | 完整稀疏重建 |
| `image_undistorter` | 原图、稀疏模型 | 去畸变 MVS workspace |
| `patch_match_stereo` | 去畸变 workspace | 深度、法线、代价、一致性图 |
| `stereo_fusion` | dense maps | 带 RGB/法线融合点云、可见性 |
| `poisson_mesher` | 有向点云 | Poisson PLY 网格 |
| `delaunay_mesher` | 点云、可见性、模型 | Delaunay PLY 网格 |
| `mesh_simplifier` | 三角 PLY | 简化 PLY |
| `mesh_texturer` | 网格、图像、模型 | OBJ、MTL、PNG |
| `recon_mvs` | 稀疏模型、图像 | 兼容组合 MVS 入口 |
| `recon_auto` | 原图 | SfM 到纹理网格完整编排 |

## Stage 通用参数

`image_undistorter` 到 `mesh_texturer` 的独立 stage，以及 `recon_auto`，支持：

- `--resume`：manifest 输入/参数指纹一致且所有输出存在时跳过。
- `--force`：忽略 manifest 强制执行；与 `--resume` 互斥。
- `--dry_run`：只打印执行计划。

manifest 位于输出根的 `.colite/stages/<stage>.json`。

## 全局日志参数

所有独立程序和 `colite <command>` 统一支持 `--log_dir <dir>`、
`--log_level INFO|WARNING|ERROR|FATAL`、`--log_to_stderr <0|1>` 与
`--v <n>`。参数会在子命令解析前移除，因此可放在命令参数任意位置。未指定
`log_dir` 时默认写 stderr，便于测试目录重定向保留；指定目录后默认写 glog
滚动文件。INFO 记录阶段开始、缓存命中、核心数量/后端、耗时和结束状态，
逐图细节只在 `--v 1` 或更高等级输出。

## SfM 参数契约（COLMAP 4.0）

SfM 同义旧参数和别名已经移除，不提供兼容层。标准参数直接使用 COLMAP 4.0 名称；只有存储方式、实现后端或其他特有功能使用 `Colite.*`。`--name value` 与 `--name=value` 都可用，重复参数、无效数值、冲突相机分组和不适用的特征/匹配器族参数会在执行前报错。

相同名称应有相同含义、单位、布尔语义和枚举。当前只接受实际接入的选项；下文列出的算法/默认值差距仍需处理，不能把现阶段命令当成完整 COLMAP CLI 的替代品。目标是继续消除差距，而不是为旧设计永久保留差异。

### 特征提取

```bash
$COLITE feature_extractor --image_path data/building-18 \
  --Colite.workspace_path "$OUT/workspace" \
  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
  --FeatureExtraction.type SIFT --FeatureExtraction.num_threads 8 \
  --SiftExtraction.max_num_features 8192 --Colite.feature_extractor cuda_vlfeat
```

| 参数 | 当前行为 / 默认值 |
|---|---|
| `ImageReader.camera_model`、`camera_params` | SIMPLE_RADIAL；显式 CSV 参数按 COLMAP 顺序。初始焦距由完整 camera_params 表达，旧 focal_length 已移除 |
| `ImageReader.default_focal_length_factor` | 1.2；缺少 EXIF 焦距时使用 |
| `ImageReader.single_camera`、`single_camera_per_folder`、`single_camera_per_image` | 最多一个 true；全 false 时自动按 EXIF 相机身份分组。旧 camera_mode 已移除 |
| `FeatureExtraction.type` | SIFT / ALIKED_N16ROT / ALIKED_N32 |
| `FeatureExtraction.num_threads` | -1；提取 CPU 线程预算 |
| `SiftExtraction.max_num_features` | 8192；只接受 SIFT。CPU VLFeat 按基础点整层保留；GPU 与 COLMAP 的 -tc2 一样在方向展开后再次按整层限额，边界层可使数量超过预算 |
| `AlikedExtraction.max_num_features`、`min_score` | 2048 / 0.2；只接受 ALIKED |
| `AlikedExtraction.n16rot_model_path`、`n32_model_path` | 本地 ONNX 路径，必须对应选定模型；没有自动下载默认 URI 的行为 |
| `Colite.feature_extractor` | auto / vlsift / cvsift / cuda_opencv / cuda_vlfeat；auto 统一选择 CUDA VLFeat 或 CPU VLFeat |
| `FeatureExtraction.max_image_size` | 非正数按类型自动选择：SIFT 3200、ALIKED 1600；缩放提取后关键点回映原图，Camera 尺寸保持原始值 |
| `FeatureExtraction.use_gpu`、`gpu_index` | CPU/GPU 选择；单设备整数，-1 自动选设备 0。多 GPU CSV 暂不支持 |
| `SiftExtraction.first_octave`、`num_octaves`、`octave_resolution` | -1 / 4 / 3；first_octave 当前支持 -1 或 0 |
| `SiftExtraction.peak_threshold`、`edge_threshold`、`max_num_orientations`、`normalization` | 0.02/3 / 10 / 2 / L1_ROOT；可选 L2，方向上限 1..4；显式几何控制要求 VLFeat 后端 |
| `Colite.learned_cuda_memory_limit_mb` | 8192；ONNX CUDA arena 预算 |

非正 max_image_size 不表示关闭缩放；要保留原分辨率，指定不小于图像最长边的正尺寸。CLI 的 max_num_features 必须为正数，低层提取器测试 API 的 0=不限额不作为公开参数行为。

### 匹配与几何验证

```bash
$COLITE exhaustive_matcher --Colite.workspace_path "$OUT/workspace" \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 --FeatureMatching.guided_matching 1 \
  --TwoViewGeometry.max_error 4
$COLITE sequential_matcher --Colite.workspace_path "$OUT/workspace" \
  --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1
```

匹配策略由命令选定；冗余 matching_type 已移除。给 exhaustive_matcher 传 SequentialMatching 参数等无效组合会报错。

| 参数 | 当前行为 / 默认值 |
|---|---|
| `FeatureMatching.type` | SIFT_BRUTEFORCE / SIFT_LIGHTGLUE / ALIKED_BRUTEFORCE / ALIKED_LIGHTGLUE；旧 matcher 已移除 |
| `SiftMatching.max_ratio`、`max_distance`、`cross_check` | 0.8 / 0.7 / true，只用于 SIFT BF 族 |
| `SiftMatching.lightglue_model_path`、`lightglue_min_score` | SIFT LightGlue 模型路径 / 0.1 |
| `AlikedMatching.brute_force_min_cossim`、`brute_force_max_ratio`、`brute_force_cross_check` | 0.85 / 1.0 / true，ALIKED BF；本地实现不需要 ONNX bruteforce_model_path |
| `AlikedMatching.lightglue_model_path`、`lightglue_min_score` | ALIKED LightGlue 模型路径 / 0.1 |
| `FeatureMatching.guided_matching` | matcher 与 recon_sfm 统一默认 false；当前重搜索仅支持 SIFT BF |
| `TwoViewGeometry.max_error`、`min_num_inliers`、`confidence`、`max_num_trials` | 4 px / 15 / 0.999 / 10000，实际控制几何估计 |
| `SequentialMatching.overlap`、`quadratic_overlap` | 10 / true；仅 sequential_matcher |
| `SpatialMatching.max_num_neighbors`、`max_distance` | 50 / 100；距离必须 >0，未实现 min_num_neighbors 补足规则 |
| `VocabTreeMatching.num_images` | 100；仅 vocab_tree_matcher |
| `FeatureMatching.use_gpu`、`gpu_index`、`num_threads` | SIFT 默认用可用 GPU；单设备索引 -1 自动选 0；线程预算默认 -1 自动。ALIKED BF 当前仅 CPU，显式 use_gpu=1 会拒绝；LightGlue 接到 ONNX provider |
| `Colite.guided_match_ratio`、`guided_min_inlier_gain` | 默认继承 SiftMatching.max_ratio，并沿用其 cross_check；显式 ratio 可覆盖。独立 matcher 内点增量门槛 10 |
| `Colite.vocab_tree_path` | COLITE 格式词树；没有使用 VocabTreeMatching.vocab_tree_path 假装可读取 COLMAP 词树 |

ONNX 设备与匹配并行预算使用标准 FeatureExtraction.* / FeatureMatching.*。特有 arena / I/O binding 仍使用 `Colite.learned_cuda_memory_limit_mb` / `Colite.learned_use_io_binding`。旧 Colite.matcher_backend、Colite.aliked_max_image_size、Colite.learned_device/gpu_index/matcher_num_threads 已移除，没有别名。

Colite 自建词树、词树保护顺序边和 transitive matcher 的控制也使用 `Colite.*`，以各命令 help 为准。它们不属于 COLMAP 同名文件/算法兼容承诺。

### Mapper 与组合流程

```bash
$COLITE mapper --Colite.workspace_path "$OUT/workspace" \
  --Mapper.multiple_models 1 --Mapper.max_num_models 50 \
  --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 --Colite.output_format both

$COLITE recon_sfm --image_path data/building-18 --output_path "$OUT/sfm" \
  --ImageReader.single_camera 1 --FeatureExtraction.type SIFT \
  --SiftExtraction.max_num_features 8192 --Colite.feature_extractor cuda_vlfeat \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1 \
  --FeatureMatching.guided_matching 1
```

Mapper / recon_sfm 共用已实现的标准参数：

- 多模型：`Mapper.multiple_models`、`max_num_models`、`min_model_size`。
- 初始化：`Mapper.init_min_num_inliers`、`init_max_error`、`init_min_tri_angle`、`init_num_trials`。
- 注册：`Mapper.max_reg_trials`、`abs_pose_max_error`、`abs_pose_min_num_inliers`、`abs_pose_min_inlier_ratio`、`structure_less_registration_fallback`。
- 过滤：`Mapper.filter_max_reproj_error`、`filter_min_tri_angle`。
- BA：`Mapper.ba_refine_focal_length`、`ba_refine_principal_point`、`ba_refine_extra_params`、`ba_local_num_images`、`ba_local_max_num_iterations`、`ba_global_max_num_iterations`。旧 ba_refine_params 逐名称选择入口已删除。
- 三角化：`Mapper.tri_max_transitivity`、`tri_create_max_angle_error`、`tri_continue_max_angle_error`、`tri_merge_max_reproj_error`、`tri_complete_max_reproj_error`、`tri_complete_max_transitivity`、`tri_min_angle`、`tri_ignore_two_view_tracks`。

`input_path` 用于接续已有模型。`Colite.mapper` 选择 incremental / global / hierarchical；global 当前只接入 Mapper.tri_* 和 Mapper.ba_global_max_num_iterations，其他新增 Mapper 参数明确拒绝。

仍有意义的独立功能包括 `Colite.merge_models`、`Colite.registration_retry_error_multiplier` / `registration_retry_max_error`、`Colite.hierarchical_*`、`Colite.nearby_match_*`、`Colite.use_position_priors` / `position_prior_*`。标准 max_reg_trials、init_num_trials、structure_less_registration_fallback 的旧扁平名称不再接受。

### 默认行为与剩余边界（2026-10-18）

Mapper / recon_sfm 已共用 COLMAP 默认配置：初始化 100 内点 / 16° / 200 次候选预算，abs pose 12 px / 30 内点 / 0.25 比率，过滤 1.5°，最小模型 10 张（受总图像数限制），structure-less 回退默认 true。初始化失败后按两轮“内点减半 → 角度减半”重试；默认 PnP 重试不再隐式放宽用户的误差阈值。

这并不表示完整增量调度等价：init_num_trials 的候选/重建尝试组织、structure-less 求解器、周期全局 BA 和收敛策略仍有实现差异。BA 增长控制、全部 SIFT 仿射/DSP/upright/遮罩选项、统一 random_seed、多 GPU CSV 等未接入项继续明确报错。ALIKED BF 尚无 GPU 实现，不能以接受 use_gpu=1 代替实际加速。

数据库适配按本轮要求暂缓，继续使用文件 workspace。优化与验证记录见 [参数对齐与前端优化](sfm_alignment_20261018.md)。

### TXT / BIN 与模型维护

Colite 文件工作区使用 `Colite.workspace_path`；它不是 COLMAP SQLite database_path，不能通过改名宣称等价。模型文件保持 COLMAP world-to-camera、WXYZ、little-endian 与完整 POINT2D_IDX/track。

特征目前可通过 `scripts/convert_colmap_database_to_colite_workspace.py --database <db> --image_path <images> --workspace <new-workspace> --include_descriptors` 单向导入 SIFT、ALIKED N16Rot/N32；源 SIFT 若采用 L2，额外传 `--SiftExtraction.normalization L2`，默认声明 COLMAP 的 L1_ROOT。随后可用对应 FeatureMatching.type 执行 Colite matcher/mapper。该工具保留关键点 xy/顺序与描述子值，写 feature v2；不保留完整仿射形状、raw matches、F/E/H、rig/frame 与 pose prior，也没有反向数据库导出。直接复用 verified 相对几何要求源数据库已保存有效相对位姿。详见 [当前互通审查与路线](colmap4_interop_audit_20261013.md)。

mapper / recon_sfm 用 `Colite.output_format txt|bin|both`，默认 both。模型维护命令用 `Colite.input_format auto|txt|bin` 选择输入（auto 优先完整 BIN），以及 `Colite.output_format`；model_converter 使用标准 `output_type TXT|BIN`，另支持扩展 BOTH。模型转换不优化，并保留 ID、参数、位姿、观测、RGB、ERROR 和 track。

```bash
$COLITE model_converter --input_path colmap/sparse/0 --output_path converted --output_type BOTH
colmap model_converter --input_path converted --output_path colmap_text --output_type TXT
$COLITE bundle_adjuster --input_path converted --output_path ba \
  --BundleAdjustment.refine_focal_length 1 --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 1 --BundleAdjustmentCeres.max_num_iterations 50
$COLITE point_triangulator --Colite.workspace_path workspace --input_path converted --output_path triangulated \
  --Mapper.tri_min_angle 1.5
$COLITE image_registrator --Colite.workspace_path workspace --input_path converted --output_path registered \
  --Mapper.abs_pose_max_error 12 --Mapper.abs_pose_min_num_inliers 30
```

旧 BundleAdjustment.max_num_iterations 与 max_reprojection_error 已删除。point_triangulator 只接受实际使用的 Mapper.tri_*，image_registrator 只接受实际使用的 Mapper.abs_pose_*。

TXT/BIN 写出仍为三文件模型，不保留 COLMAP 4 rig/frame 元数据；写出时清理另一格式或旧 rig/frame，避免自动读取陈旧数据。SfM 改变几何后重算 ERROR；纯模型转换保留 ERROR。工作区特征/匹配缓存与 SQLite 保持各自格式。

### 2026-10-08 参数清理验证

最终 CUDA 构建的 CTest：404 项通过、6 项跳过、0 失败。跳过项为五项需要额外 ONNX 模型/运行环境的测试，以及与当前 CUDA 构建条件不适用的一项无 CUDA 失败路径测试。

`scripts/check_colmap_model_interop.py` 使用官方 COLMAP 4.0 验证 TXT/BIN 双向转换：12 种相机模型的构造样本，以及 House 的 36 张图像、40,034 个点，均保留相机参数、ID、位姿、二维观测、RGB、ERROR 和 track。报告位于 `output/sfm_contract_20261008/interop/report.json`；此验证范围为三文件稀疏模型。

Building 的新参数 GPU 全流程回归使用显式 `cuda_vlfeat` / `cuda` 与 guided matching，注册 18/18 张图像，输出 15,727 个点。报告位于 `output/sfm_contract_20261008/building_gpu_final/report.json`；COLMAP 对照复用了原有运行，不是重新计时。这是参数迁移与重建覆盖回归，不是公开参考集精度评估。真实参考集与评估要求见 [SfM 真实数据与参考质量评估](sfm_reference_datasets.md)。

## MVS 单步骤

### 图像去畸变

```bash
ROOT=output/mvs_single_building18_YYYYMMDD
$COLITE image_undistorter \
  --image_path data/building-18 \
  --input_path output/recon_sfm_building18_YYYYMMDD \
  --output_path "$ROOT" \
  --undistort_max_image_size 1600
```

必填 `image_path`、`input_path`（可用 `model_path`）和 `output_path`。主要默认：`blank_pixels=0`、`min_scale=0.2`、`max_scale=2`、`max_image_size=-1`。

### PatchMatch Stereo

```bash
$COLITE patch_match_stereo \
  --model_path "$ROOT/undistorted/sparse" \
  --image_path "$ROOT/undistorted/images" \
  --output_path "$ROOT" \
  --patch_match_backend cuda \
  --patch_match_iterations 3 \
  --patch_match_pyramid_levels 2 \
  --patch_match_directional_hmm \
  --patch_match_source_samples 15 \
  --dense_map_format binary
```

也可使用 `--workspace_path <undistorted-dir>`，读取其 `sparse/`、`images/` 并输出到该目录。

重要参数：

| 类别 | 参数与默认 |
|---|---|
| 后端 | `patch_match_backend=auto`，`gpu_index=-1`，`num_threads=1` |
| 尺寸 | `resolution_level=1`，`max_image_size=-1`，`image_cache_size=16` |
| 迭代 | `iterations=3`，`pyramid_levels=1`，`window_radius=2`，`random_seed=42` |
| source | `max_num_source_images=8`，`min_num_shared_points=1`；`source_selection_mode=weighted` 保持通用小基线兼容，`colmap` 使用 75% 分位三角化角门限和共享点排序 |
| 聚合 | `min_num_valid_sources=1`，`max_num_aggregated_costs=3` |
| 初始法线 | 默认相机朝向半球随机初始化；`initial_normal_hemisphere_degrees=90`，`normal_perturbation_degrees=57.2958`；`fixed_initial_normals` 仅用于兼容/A-B |
| 调度 | `propagation_schedule=checkerboard`，每迭代默认一轮红黑 |
| 光度 | bilateral NCC 可选，`sigma_color=0.2`；dynamic selection 可选，`ncc_sigma=0.6` |
| HMM/CDF | 动态 source 默认启用 directional HMM；`directional_hmm` 也会自动启用动态 source；`source_samples=0` 为 best-K，15 为 COLMAP 风格 CDF；CDF 默认叠加三角化角、入射角和局部单应分辨率先验，可用 `patch_match_no_colmap_source_priors` 关闭 |
| 几何 | refinement iterations/weight/depth error；内部至少 3 层、1 个继承光度结果的同步 pass；默认使用独立的稳健跨视图几何聚合，`patch_match_legacy_geometric_aggregation` 可恢复旧的逐 source 混排；最终 reprojection 默认 1 px |
| 输出 | `dense_map_format=text|binary` 同时控制 depth/normal/cost，可选 photometric filter 和 depth-normal refinement |

Fusion 的候选源图数量与 PatchMatch 解耦：`fusion_max_num_source_images=50`
默认按共享稀疏点数排序检查最多 50 张重叠图；调整
`max_num_source_images` 不会再意外限制 Fusion。

`--patch_match_diagnostics` 会启用详细 CUDA 原子统计，正式性能运行默认不要开启。质量优先推荐两层金字塔 + HMM + CDF15；快速预览可使用 `resolution_level=2` 和 2 iterations。

### Stereo Fusion

```bash
$COLITE stereo_fusion \
  --model_path "$ROOT/undistorted/sparse" \
  --image_path "$ROOT/undistorted/images" \
  --output_path "$ROOT" \
  --fusion_min_num_observations 2
```

读取已有 maps，不重跑 PatchMatch。主要默认：

- observations：min 1、max 100000、traversal depth 100；
- reprojection 2 px、relative depth 0.01、normal 10°、view angle 180°；
- source pixel search radius 0（与 COLMAP 的最近投影像素语义一致；仅在明确需要
  邻域补偿时手动增大）；
- 图传播候选发现使用 current point，但重投影/深度一致性始终相对 seed point，
  防止逐跳空间漂移；
- `filter_geometric_min_num_consistent_sources` 控制一致性来源下限。

输出 binary little-endian `dense/fused.ply`（float 坐标/法线、uchar RGB）、可见性和 track 质量统计。
默认不采集高开销的成对拒绝计数和误差样本；需要调参数据时增加
`--fusion_diagnostics`，此时额外写出逐 track、image-pair 和误差分布明细。独立持久化融合始终写出
`dense/fusion_timing.txt`，细分 RGB、depth/normal、一致性图、核心融合和写出
耗时。

## 表面单步骤

### Poisson

```bash
$COLITE poisson_mesher \
  --input_path "$ROOT/dense/fused.ply" \
  --output_path "$ROOT/surface/meshed-poisson.ply" \
  --PoissonMeshing.depth 10 \
  --PoissonMeshing.point_weight 1 \
  --PoissonMeshing.color 1 \
  --PoissonMeshing.trim 10 \
  --PoissonMeshing.num_threads -1
```

默认 `depth=13`、`point_weight=1`、`color=1`、`trim=10`。稀疏点云上 trim 过高可能得到空网格，A/B 必须记录 trim。

### Delaunay

```bash
$COLITE delaunay_mesher \
  --input_path "$ROOT/dense" \
  --model_path "$ROOT/undistorted/sparse" \
  --output_path "$ROOT/surface/meshed-delaunay.ply"
```

输入目录需包含 `fused.ply`、`points_visibility.bin`（兼容 `fused.ply.vis`）。`--DelaunayMeshing.` 参数包括 `max_proj_dist=20`、`max_depth_dist=0.05`、`visibility_sigma=3`、`distance_sigma_factor=1`、`quality_regularization=1`、`max_side_length_factor=25`、`max_side_length_percentile=95`、`num_threads`。

### 简化

```bash
$COLITE mesh_simplifier \
  --input_path "$ROOT/surface/meshed-poisson.ply" \
  --output_path "$ROOT/surface/mesh-simplified.ply" \
  --MeshSimplification.target_ratio 0.5
```

参数：`target_ratio=0.5`、`target_num_triangles=0`、`max_error=0`、`preserve_boundary=1`、`max_normal_change=60`。

### 纹理

```bash
$COLITE mesh_texturer \
  --input_path "$ROOT/surface/mesh-simplified.ply" \
  --model_path "$ROOT/undistorted/sparse" \
  --image_path "$ROOT/undistorted/images" \
  --output_path "$ROOT/surface/textured"
```

重要 `--MeshTextureMapping.` 默认：atlas 4096、max atlas 32768、padding 2、inpaint 5、scale 1、min cosine 0.1、min visible vertices 3、label smoothing 3、`view_selection_min_component_size=64`、visibility resolution 1024、occlusion tolerance 0.01、`use_vertex_color_fallback=0`、color correction 1、regularization 0.1。遮挡容差只在受拓扑支持的孔洞恢复中完整使用；主视图选择最多使用 0.005 的端点噪声带。

`occlusion_relative_tolerance=0.01` 只允许受近共面双邻面支持的孔洞恢复在
目标表面 1% 范围内忽略端点噪声；不建议在没有 A/B 的情况下继续放宽。没有
可靠相机视图的面默认映射到中性 texel，OBJ 仍保留全部三角形；如需旧版逐面
RGB 顶点色 patch，可显式设置 `--MeshTextureMapping.use_vertex_color_fallback 1`。

大网格先用 `target_num_triangles=500`、`visibility_resolution=512` 验证，再逐级提高。

## 组合 MVS

`recon_mvs` 是兼容组合入口：

```bash
$COLITE recon_mvs \
  --model_path output/recon_sfm_building18_YYYYMMDD \
  --image_path data/building-18 \
  --output_path output/recon_mvs_building18_YYYYMMDD \
  --image_undistorter --run_patch_match --fuse_stereo \
  --poisson_mesher --mesh_simplifier --mesh_texturer \
  --patch_match_backend cuda --dense_map_format binary
```

阶段开关还包括 `initialize_patch_match`、`check_geometric_consistency`、`filter_dense_cloud`、`mesh_dense_cloud`、`delaunay_mesher`。`dense_cloud_spatial_bounds_quantile` 可显式裁掉 XYZ 各轴上下空间尾部并写入 `fused_clean.ply`，默认 0 表示关闭；House 离群保护可使用 0.001。算法参数使用小写兼容形式，如 `poisson_depth`、`mesh_simplification_ratio`、`texture_atlas_size`；用 `recon_mvs --help` 查看完整默认。

## 完整自动流程

```bash
$COLITE recon_auto \
  --image_path data/building-18 \
  --output_path output/recon_auto_building18_YYYYMMDD \
  --sfm.SiftExtraction.max_num_features 8192 \
  --mvs.patch_match_backend cuda \
  --mvs.patch_match_iterations 3 \
  --mvs.dense_map_format binary \
  --mesher poisson --simplify 1 --texture 1
```

- `start_from/stop_after`：`sfm|undistort|stereo|fusion|mesh|simplify|texture`。
- `mesher`：`poisson|delaunay|none`，默认 poisson。
- `simplify`、`texture` 默认 1。
- `--sfm.<canonical-option>` 转发到 `recon_sfm`，包括 ImageReader.*、FeatureExtraction.*、Mapper.* 和 Colite.*；dry_run 也先校验。旧 --sfm.camera_mode 等名称不再接受。
- `--mvs.<option>` 转发到去畸变、PatchMatch 和 Fusion。
- 表面参数保持 `--PoissonMeshing.*`、`--DelaunayMeshing.*`、`--MeshSimplification.*`、`--MeshTextureMapping.*`。

断点示例：

```bash
$COLITE recon_auto --image_path data/building-18 --output_path "$OUT" --resume
$COLITE recon_auto --image_path data/building-18 --output_path "$OUT" \
  --start_from stereo --stop_after fusion
```

从中间阶段开始前，上游约定产物必须已经存在于相同输出结构中。

## COLMAP 模型比较

比较器支持 TXT/BIN 和混合格式；默认自动识别。也可以在同一模型目录中直接
比较两种保存格式：

```bash
$COLITE compare_colmap --reference_path converted --candidate_path converted \
  --reference_format txt --candidate_format bin --output_path format_comparison
```

```bash
$COLITE compare_colmap \
  --reference_path output/reference_colmap/sparse/0 \
  --candidate_path output/recon_sfm_building18_YYYYMMDD \
  --output_path output/compare_sfm_building18_YYYYMMDD
```

输出 `summary.txt`、`image_diff.tsv`、`pose_diff.tsv`、`point_diff.tsv` 和 `pair_diff.tsv`。位姿先按共同相机中心估计 Sim(3)；独立模型点 ID 不可视为对应关系。

可用标准库 Python 脚本执行真实双向互操作验证，覆盖全部 12 种相机模型，并可
额外传入已有重建结果：

```bash
python3 scripts/check_colmap_model_interop.py \
  --colite "$COLITE" --colmap /path/to/colmap \
  --output_path output/model_interop --input_path sparse/0
```

脚本独立解码 TXT/BIN，逐字段检查 `COLITE → COLMAP → COLITE`，保存模型、
进程日志和 `report.json`。本地 COLMAP 4.0 的 TXT 读取器会截断含空格的图像名，
并将点 ID 按有符号 64 位解析；这些特殊数据跨工具交换时应选择 BIN。
