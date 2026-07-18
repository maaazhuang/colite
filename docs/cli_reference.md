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
| `mapper` | 已匹配 workspace | 稀疏模型和带 RGB PLY |
| `compare_colmap` | 两个 COLMAP 模型 | Sim(3) 位姿/结构差异报告 |
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
| `--focal_length` | EXIF/尺寸先验 | 固定初始焦距像素 |
| `--max_num_features` | 4096 | 基础 keypoint 上限；VLFeat 最终方向点可超过 |
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
| `--matcher` | flann | `bf|flann` |
| `--max_ratio` | 0.8 | Lowe ratio |
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

输出位于 workspace 的 `sparse/`，包括 COLMAP text model、带 RGB 的 `points3D.ply`、summary 和注册诊断。算法缓存命中时会直接返回；修改配置或做计时时使用新 workspace。

## 完整 SfM

```bash
$COLITE recon_sfm \
  --image_path data/building-18 \
  --output_path output/recon_sfm_building18_YYYYMMDD \
  --feature_extractor auto --matcher flann \
  --max_num_features 8192 --feature_num_threads -1
```

主要参数：

- `--focal_length`；省略时优先 EXIF，否则 `1.2 × max(width,height)`。
- `--nearby_match_window` 默认 2，0 禁用；`--nearby_match_ratio` 默认 0.9。
- `--max_registration_trials_per_image`、`--max_initialization_trials` 默认 3。
- `--registration_retry_error_multiplier` 默认 1.5。
- `--registration_retry_max_error` 默认 8 px。
- `--ba_refine_focal_length`、`--ba_refine_principal_point`、`--ba_refine_extra_params` 默认 0。
- `--ba_refine_params <fx,fy,k1,...>` 按名称指定内参优化。

`recon_sfm` 输出 cameras/images/points3D text、带 RGB PLY、pair/registration diagnostics 和 summary。

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
| source | `max_num_source_images=8`，`min_num_shared_points=1` |
| 聚合 | `min_num_valid_sources=1`，`max_num_aggregated_costs=3` |
| 调度 | `propagation_schedule=checkerboard`，每迭代默认一轮红黑 |
| 光度 | bilateral NCC 可选，`sigma_color=0.2`；dynamic selection 可选，`ncc_sigma=0.6` |
| HMM/CDF | `directional_hmm` 自动启用动态 source；`source_samples=0` 为 best-K，15 为 COLMAP 风格 CDF |
| 几何 | refinement iterations/weight/depth error；最终 reprojection 默认 1 px |
| 输出 | `dense_map_format=text|binary`，可选 photometric filter 和 depth-normal refinement |

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
- source pixel search radius 1；
- `filter_geometric_min_num_consistent_sources` 控制一致性来源下限。

输出 `dense/fused.ply`（坐标、法线、RGB）、可见性和 Fusion 诊断。

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

重要 `--MeshTextureMapping.` 默认：atlas 4096、max atlas 32768、padding 2、inpaint 5、scale 1、min cosine 0.1、min visible vertices 3、label smoothing 3、visibility resolution 1024、occlusion tolerance 0.01、color correction 1、regularization 0.1。

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

阶段开关还包括 `initialize_patch_match`、`check_geometric_consistency`、`filter_dense_cloud`、`mesh_dense_cloud`、`delaunay_mesher`。算法参数使用小写兼容形式，如 `poisson_depth`、`mesh_simplification_ratio`、`texture_atlas_size`；用 `recon_mvs --help` 查看完整默认。

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
