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

## SfM 单步骤

### 特征提取

```bash
OUT=output/sfm_single_building18_YYYYMMDD
$COLITE feature_extractor \
  --image_path data/building-18 \
  --workspace_path "$OUT/workspace" \
  --feature_extractor auto \
  --max_num_features 8192 \
  --feature_num_threads -1
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--image_path` | 必填 | 图像目录 |
| `--workspace_path` | 必填 | 文件 workspace |
| `--focal_length` | EXIF/尺寸先验 | 显式初始焦距像素；是否由 BA 继续优化由 `--ba_refine_focal_length` 控制 |
| `--camera_model` | SIMPLE_RADIAL | 现有12种 COLMAP 兼容模型之一 |
| `--camera_params` | 空 | 按模型参数顺序给出完整 CSV；参数数量必须精确匹配 |
| `--camera_mode` | auto | `auto|single|per_folder|per_image`；兼容别名 `shared=single`、`individual=per_image` |
| `--default_focal_length_factor` | 1.2 | EXIF 不足时的 `factor × max(width,height)` |
| `--max_num_features` | 4096 | OpenCV/CUDA 后端为最终输出上限；CPU VLFeat 按基础 keypoint/DoG level 限制，最终多方向点可超过 |
| `--feature_num_threads` | -1 | CPU 线程数；-1 为逻辑核 |
| `--feature_extractor` | auto | `auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat` |

`auto` 有可用 CUDA 时选择 `cuda_opencv`，否则 `cvsift`。`cudasift`、`cuda_sift`、`cuda` 是兼容别名。CUDA 模式使用单个外层图像 worker，避免同一 GPU 被多任务争用。

### Matcher

```bash
$COLITE exhaustive_matcher --workspace_path "$OUT/workspace" --matcher flann
$COLITE sequential_matcher --workspace_path "$OUT/workspace" \
  --overlap 5 --matcher flann --max_ratio 0.8
```

共同参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--workspace_path` | 必填 | 已提取特征的 workspace |
| `--matching_type` | 由命令决定 | `exhaustive|sequential|spatial|vocab_tree` |
| `--matcher` | flann | `bf|flann|cuda|aliked_bf|sift_lightglue|aliked_lightglue`；`cuda` 是与 CPU BF 口径一致的精确 GPU 暴力匹配 |
| `--max_ratio` | 0.8 | Lowe ratio |
| `--guided_matching` | 0 | 几何验证后按 F 极线约束重搜描述子 |
| `--guided_match_ratio` | 0.8 | standalone guided Lowe ratio；COLMAP High 同描述子语义 |
| `--guided_min_inlier_gain` | 10 | 替换图像对几何所需的最小内点增量 |
| `--overlap` | 5 | sequential 后向邻居数 |
| `--spatial_max_num_neighbors` | 10 | 空间邻居数 |
| `--spatial_max_distance` | ≤0 不限 | 位置先验距离阈值 |

spatial 要求 `images.tsv` 中所有图像都有位置先验，缺失会报错。

Vocabulary tree：

```bash
$COLITE vocab_tree_builder \
  --workspace_path "$OUT/workspace" \
  --output_path "$OUT/vocab/tree.txt" \
  --num_visual_words 1024 --branching 16 \
  --num_iterations 6 --max_num_features 5000

$COLITE vocab_tree_matcher \
  --workspace_path "$OUT/workspace" \
  --vocab_tree_path "$OUT/vocab/tree.txt" \
  --vocab_tree_num_images 50 \
  --vocab_tree_sequential_overlap 2 --matcher flann
```

`vocab_tree_num_visual_words=1024`、`branching=16`、`num_iterations=6`、每图检索描述子上限 5000、检索候选图 50、时序保护 overlap 2。纯 matcher A/B 应复用已构建词树并单独记录 builder 时间。

### Mapper

```bash
$COLITE mapper --workspace_path "$OUT/workspace"
```

默认会重建 verified 匹配图中的多个可初始化分量，并尝试合并可用 verified 3D-3D
对应对齐的模型：

```bash
$COLITE mapper \
  --workspace_path "$OUT/workspace" \
  --multiple_models 1 --merge_models 1 \
  --max_num_models 50 --min_model_size 2
```

从已有 COLMAP text model 继续注册：

```bash
$COLITE mapper \
  --workspace_path "$OUT/workspace" \
  --input_model_path "$OUT/workspace/sparse/models/0"
```

恢复入口按图像名重映射历史模型 ID，允许当前 workspace 增加新图，但要求相机模型、尺寸和
共享内参关系兼容；已有图的特征数量、坐标和索引顺序也必须与断点模型一致，改变特征配置后
会拒绝恢复，以免旧 track 错接到新特征。`--input_model_path` 会绕过 sparse 结果缓存并实际执行续建。输出位于
workspace 的 `sparse/`：根目录是最大主模型，`models/<index>/` 是全部模型，
`models.txt` 记录索引；每个模型均含 COLMAP text model 和带 RGB 的 `points3D.ply`。
算法缓存命中时会直接返回；修改配置或做计时时使用新 workspace。

## 完整 SfM

```bash
$COLITE recon_sfm \
  --image_path data/building-18 \
  --output_path output/recon_sfm_building18_YYYYMMDD \
  --feature_extractor auto --matcher flann \
  --max_num_features 8192 --feature_num_threads -1
```

主要参数：

- `--camera_model` 默认 `SIMPLE_RADIAL`；`--camera_params` 可按该模型登记顺序
  提供完整参数。未提供时由焦距先验、图像中心和零畸变初始化。
- `--camera_mode auto|single|per_folder|per_image`，默认 `auto`，对齐 COLMAP
  GUI “Individual images”的 ImageReader AUTO 语义：只有 Make、Model、焦距
  标识及尺寸完整一致的图像共享 Camera；元数据不完整时每图独立，避免误共享。
  `single` 要求尺寸一致；`per_folder` 按相对父目录共享；`per_image` 强制每图
  独立。旧名称 `shared`、`individual` 分别保留为 `single`、`per_image` 别名。
  各组多个焦距 prior 使用中位数，相对跨度超过 5% 时明确报错。
- `--focal_length`；省略时每张图依次使用 EXIF 35mm 等效焦距（按 43.27 mm
  全画幅对角线换算）、EXIF 焦距毫米值与焦平面分辨率，最后回退
  `default_focal_length_factor × max(width,height)`。EXIF/用户值标记为焦距
  先验，尺寸回退不标记为先验。
- `--nearby_match_window` 默认 0，控制首轮匹配中相邻图像对的 ratio 放宽；`--nearby_match_ratio` 默认 0.8。首轮始终遍历全部图像对。几何验证后会对所有具有可用初始 F 的 pair 执行极线约束描述子重搜索，不受 nearby window 限制。`--guided_match_ratio` 独立控制该阶段，`recon_sfm` 默认 0.95；使用同一套 COLMAP 描述子做 High 对照时显式设为 0.8。
- `--max_registration_trials_per_image`、`--max_initialization_trials` 默认 3。
- `--registration_retry_error_multiplier` 默认 1.5。
- `--registration_retry_max_error` 默认 8 px。
- `--ba_refine_focal_length`、`--ba_refine_extra_params` 默认 1，与 COLMAP
  增量 Mapper 一致；`--ba_refine_principal_point` 默认 0。
- `--ba_refine_params <fx,fy,k1,...>` 按名称指定内参优化。
- `--use_position_priors 1` 从 JPEG EXIF GPS 读取 WGS84 坐标，以全组中位坐标
  为原点转换为米制 ENU；每个稀疏模型先用至少 3 个已注册相机中心执行 Sim(3)
  对齐，再在最终全局 BA 中加入鲁棒位置残差。默认关闭以保持无 GPS 数据的旧
  行为。`--position_prior_stddev` 默认 5 m，
  `--position_prior_loss_scale` 默认 2 m，
  `--position_prior_alignment_max_error` 默认 20 m。
- `--input_model_path <dir>` 从已有 COLMAP text model 继续注册；完整 `recon_sfm`
  会重新计算前端特征/匹配，若只需复用已有前端结果，优先使用分阶段 `mapper`。
- `--multiple_models 0|1`、`--merge_models 0|1` 默认均为 1；
  `--max_num_models` 默认 50，`--min_model_size` 默认 2。`recon_auto` 可用
  `--sfm.input_model_path`、`--sfm.multiple_models` 等同名转发参数。

`recon_sfm` 输出 cameras/images/points3D text、带 RGB PLY、pair/registration
diagnostics 和 summary。最大模型位于输出根目录，全部模型位于
`models/<index>/`。`summary.txt` 记录相机策略、相机数量、具有焦距先验的
图像数、初始焦距范围、实际 SIFT/matcher 后端、CUDA 可用性、CPU 线程数、

新增前端和 mapper 参数：`--feature_type sift|aliked_n16rot|aliked_n32`、
`--aliked_model_path`、`--learned_matcher_model_path`、
`--mapper incremental|global|hierarchical`。全局模式从 verified view graph 完成旋转平均、
相机中心估计、track 三角化和全局 BA；首版不接受 `--input_model_path`。
层次模式通过 `--hierarchical_target_cluster_size`、
`--hierarchical_max_cluster_size` 和 `--hierarchical_overlap` 控制重叠分区，
逐簇执行局部增量重建，重试合并具有公共 verified track 的局部模型，最后执行全局 BA；
该模式同样不接受 `--input_model_path`。
增量模式可用 `--structureless_registration 1` 显式启用 PnP 失败后的多邻居
相对位姿回退。`recon_auto` 中对应写法为 `--sfm.mapper global` 和
`--sfm.mapper hierarchical`，以及 `--sfm.structureless_registration 1`。
特征/匹配/内点数量，以及各流程耗时。当前 CUDA 后端加速首轮全局 BF；极线
走廊 guided 重搜索仍走 CPU。`recon_auto` 通过 `--sfm.<option>` 转发上述
相机模型、参数和分组策略。

## 稀疏模型维护与独立 SfM 内核

所有模型命令读取 `--input_format auto|txt|bin`，写出
`--output_format txt|bin|both`（默认 both），保持 COLMAP 模型兼容。

```bash
$COLITE model_analyzer --input_path sparse/0
$COLITE model_transformer --input_path sparse/0 --output_path transformed \
  --transform_path sim3.txt
$COLITE model_aligner --input_path sparse/0 --output_path aligned \
  --ref_images_path positions.txt --max_error 20
$COLITE model_cropper --input_path sparse/0 --output_path cropped \
  --min "-10,-10,-2" --max "10,10,20"
$COLITE model_splitter --input_path sparse/0 --output_path split
$COLITE bundle_adjuster --input_path sparse/0 --output_path ba
$COLITE point_triangulator --workspace_path workspace \
  --input_path sparse/0 --output_path triangulated
$COLITE image_registrator --workspace_path workspace \
  --input_path sparse/0 --output_path registered
```

`positions.txt` 每行格式为 `IMAGE_NAME X Y Z`。`point_triangulator` 是显式的
重三角化/track complete/merge 工具，输出后建议继续执行独立 BA 和稳健过滤；
`image_registrator` 可用 `--image_id` 只注册一张图，否则按 ID 尝试全部未注册图。

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
  --sfm.max_num_features 8192 \
  --mvs.patch_match_backend cuda \
  --mvs.patch_match_iterations 3 \
  --mvs.dense_map_format binary \
  --mesher poisson --simplify 1 --texture 1
```

- `start_from/stop_after`：`sfm|undistort|stereo|fusion|mesh|simplify|texture`。
- `mesher`：`poisson|delaunay|none`，默认 poisson。
- `simplify`、`texture` 默认 1。
- `--sfm.<option>` 转发到 `recon_sfm`。
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

```bash
$COLITE compare_colmap \
  --reference_path output/reference_colmap/sparse/0 \
  --candidate_path output/recon_sfm_building18_YYYYMMDD \
  --output_path output/compare_sfm_building18_YYYYMMDD
```

输出 `summary.txt`、`image_diff.tsv`、`pose_diff.tsv`、`point_diff.tsv` 和 `pair_diff.tsv`。位姿先按共同相机中心估计 Sim(3)；独立模型点 ID 不可视为对应关系。
