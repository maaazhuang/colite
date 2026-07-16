# COLITE 命令行运行与参数参考

本文对应 `colite 0.1.0`。以下命令以 WSL Release 构建为例：

```bash
cd /mnt/d/Code/CXX/colite
COLITE=./build-wsl-release/src/colite/exe/colite
$COLITE --help
$COLITE <command> --help
```

推荐统一使用 `colite <command>`。`feature_extractor`、各 matcher、`mapper`、
`compare_colmap`、`recon_sfm` 和 `recon_mvs` 也有独立可执行文件，但参数相同。

## 命令总览

| 命令 | 输入 | 主要输出 |
|---|---|---|
| `feature_extractor` | 原始图像目录 | COLITE SfM workspace、特征文件 |
| `exhaustive_matcher` | SfM workspace | 全连接候选匹配与两视图几何 |
| `sequential_matcher` | SfM workspace | 相邻图像匹配与两视图几何 |
| `spatial_matcher` | SfM workspace | 空间邻域匹配与两视图几何 |
| `vocab_tree_matcher` | SfM workspace | 视觉词检索匹配与两视图几何 |
| `vocab_tree_builder` | SfM workspace | 构建可复用的层次视觉词树 |
| `mapper` | 已匹配的 SfM workspace | `sparse/{cameras,images,points3D}.txt` |
| `compare_colmap` | 两个 COLMAP text model | 固定格式差异报告 |
| `recon_sfm` | 原始图像 | 完整稀疏模型 |
| `image_undistorter` | 图像和稀疏模型 | `undistorted/{images,sparse}` |
| `patch_match_stereo` | 去畸变图像和模型 | depth/normal/cost/consistency maps |
| `stereo_fusion` | 已有 dense maps | `dense/fused.ply`、可见性记录 |
| `poisson_mesher` | 有向融合点云 | Poisson PLY 网格 |
| `delaunay_mesher` | 融合点云、可见性、稀疏模型 | Delaunay PLY 网格 |
| `mesh_simplifier` | 三角 PLY 网格 | 简化 PLY 网格 |
| `mesh_texturer` | 网格、图像、稀疏模型 | OBJ、MTL、PNG |
| `recon_mvs` | 稀疏模型和图像 | 可组合的完整 MVS 结果 |
| `recon_auto` | 原始图像 | SfM 到纹理网格的完整结果 |

## 通用 Stage 参数

以下参数适用于 `image_undistorter`、`patch_match_stereo`、`stereo_fusion`、
`poisson_mesher`、`delaunay_mesher`、`mesh_simplifier`、`mesh_texturer`，也适用于
`recon_auto` 的编排：

- `--resume`：manifest 指纹与所有输出均匹配时跳过。
- `--force`：忽略已有 manifest，强制执行；不能与 `--resume` 同时使用。
- `--dry_run`：只打印 `PLAN <stage>`，不运行算法。

manifest 默认写入输出根目录的 `.colite/stages/<stage>.json`，包含输入指纹、
参数指纹、状态、耗时和声明的输出。

## SfM 单步骤

### feature_extractor

```bash
$COLITE feature_extractor \
  --image_path data/building-18 \
  --workspace_path output/sfm-workspace \
  --feature_extractor auto \
  --max_num_features 4096 \
  --feature_num_threads -1
```

- `--image_path <dir>`、`--workspace_path <dir>`：必填。
- `--focal_length <px>`：固定初始焦距；省略时优先读取 EXIF，否则使用图像尺寸先验。
- `--max_num_features <n>`：每图最大特征数，默认 `4096`。
- `--feature_num_threads <n>`：CPU 线程数，默认 `-1`。
- `--feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>`：默认 `auto`；有可用 GPU 时选择 `cuda_opencv`，否则回退到 `cvsift`。`cudasift`、`cuda_sift` 和 `cuda` 保留为 `cuda_opencv` 的兼容别名。

### matcher 系列

```bash
$COLITE sequential_matcher --workspace_path output/sfm-workspace \
  --overlap 5 --matcher flann --max_ratio 0.8
```

四个 matcher 共用参数：

- `--workspace_path <dir>`：必填。
- `--matching_type <exhaustive|sequential|spatial|vocab_tree>`：覆盖命令默认策略。
- `--matcher <bf|flann>`：描述子匹配器，默认 `flann`。
- `--max_ratio <r>`：Lowe ratio，默认 `0.8`。
- `--overlap <n>`：sequential 后向相邻数，默认 `5`。
- `--spatial_max_num_neighbors <n>`：空间邻居数，默认 `10`。
- `--spatial_max_distance <d>`：空间距离上限，`<=0` 表示不限制。
- `--vocab_tree_num_images <n>`：每图检索候选数，默认 `50`。
- `--vocab_tree_num_visual_words <n>`：层次树叶子视觉词数，默认 `1024`。
- `--vocab_tree_branching <n>`：层次树分支因子，默认 `16`。
- `--vocab_tree_num_iterations <n>`：每个树节点的 k-means 迭代数，默认 `6`。
- `--vocab_tree_max_num_features <n>`：每图用于检索的描述子上限，默认 `5000`。
- `--vocab_tree_path <file>`：读取 `vocab_tree_builder` 生成的持久化树；兼容旧版 v1 平面词表。省略时在内存中构建层次树。
- `--vocab_tree_sequential_overlap <n>`：向检索结果合并后续 `n` 帧，默认 `2`，设为 `0` 禁用。

推荐先离线构建并复用词树，避免把建树耗时计入每次匹配：

```bash
$COLITE vocab_tree_builder \
  --workspace_path output/sfm-workspace \
  --output_path output/vocab/tree-v2.txt \
  --num_visual_words 1024 \
  --branching 16 \
  --num_iterations 6 \
  --max_num_features 5000

$COLITE vocab_tree_matcher \
  --workspace_path output/sfm-workspace \
  --vocab_tree_path output/vocab/tree-v2.txt \
  --vocab_tree_num_images 50 \
  --vocab_tree_sequential_overlap 2 \
  --matcher flann
```

matcher 标准输出及 `matches/summary.txt` 会记录 retrieval pair 数、时序保护边召回率与新增边数、候选图连通分量，以及最小/平均/最大度。A/B 测试应将 `vocab_tree_builder` 与 matcher 分别计时；比较纯检索/匹配时，两侧都使用已构建好的词树。

`spatial_matcher` 要求 workspace 的 `images.tsv` 中每张图都有位置先验；缺少任意一张图的
位置会直接报错，避免悄悄退化成文件名邻接匹配。无 GPS/位姿先验的数据应使用
`sequential_matcher`、`exhaustive_matcher` 或 `vocab_tree_matcher`。

### mapper

```bash
$COLITE mapper --workspace_path output/sfm-workspace
```

目前只要求 `--workspace_path`。输出位于 workspace 的 sparse 模型目录，并打印注册图像数、
缓存命中状态和三维点数。

### recon_sfm

```bash
$COLITE recon_sfm --image_path data/building-18 --output_path output/sfm \
  --feature_extractor auto --matcher bf --max_num_features 4096
```

主要参数：

- `--focal_length <px>`：初始焦距；省略时优先 EXIF，否则 `1.2 * max(width,height)`。
- `--max_num_features <n>`：默认 `4096`。
- `--feature_num_threads <n>`：默认 `-1`。
- `--feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>`：默认 `auto`，选择规则同单步 `feature_extractor`。
- `--matcher <bf|flann>`：默认 `bf`。
- `--nearby_match_window <n>`：文件序相邻重匹配窗口，默认 `2`，`0` 禁用。
- `--nearby_match_ratio <r>`：相邻重匹配 ratio，默认 `0.9`。
- `--max_registration_trials_per_image <n>`：默认 `3`。
- `--max_initialization_trials <n>`：默认 `3`。
- `--registration_retry_error_multiplier <r>`：默认 `1.5`。
- `--registration_retry_max_error <px>`：默认 `8`。
- `--ba_refine_focal_length <0|1>`、`--ba_refine_principal_point <0|1>`、
  `--ba_refine_extra_params <0|1>`：BA 内参优化开关，默认均为 `0`。
- `--ba_refine_params <fx,fy,k1,...>`：按名称指定 BA 内参。

## MVS 单步骤

### image_undistorter

```bash
$COLITE image_undistorter --image_path data/building-18 \
  --input_path output/sfm --output_path output/mvs \
  --undistort_max_image_size 1600
```

必填：`--image_path`、`--input_path`（也可用 `--model_path`）、`--output_path`。
去畸变参数：`--undistort_blank_pixels`（默认 `0`）、`--undistort_min_scale`
（默认 `0.2`）、`--undistort_max_scale`（默认 `2`）、
`--undistort_max_image_size`（默认 `-1`）。

### patch_match_stereo

```bash
$COLITE patch_match_stereo \
  --model_path output/mvs/undistorted/sparse \
  --image_path output/mvs/undistorted/images --output_path output/mvs \
  --patch_match_backend cpu --patch_match_iterations 3 \
  --patch_match_pyramid_levels 1 --max_image_size 1200
```

也可用 `--workspace_path <dir>` 代替三个路径，此时读取 `<dir>/sparse` 和
`<dir>/images`，输出到 `<dir>`。常用参数：

- `--patch_match_backend <auto|cpu|cuda>`，默认 `auto`。
- `--patch_match_gpu_index <n>`，默认 `-1`。
- `--patch_match_iterations <n>`，默认 `3`。
- `--patch_match_pyramid_levels <n>`，默认 `1`。
- `--patch_match_window_radius <n>`，默认 `2`。
- `--patch_match_bilateral_ncc`、`--patch_match_sigma_spatial <s>`、
  `--patch_match_sigma_color <s>`（默认 `0.2`）。
- `--patch_match_dynamic_source_selection`、`--patch_match_ncc_sigma <s>`（默认 `0.6`）。
- `--patch_match_directional_hmm` 在 CUDA 上启用 source probability 的行/列
  forward-backward 消息；公式对齐 COLMAP 4.0，并自动启用 dynamic source selection。
- `--patch_match_perturbation_samples <n>`，默认 `1`。
- `--patch_match_source_samples <n>`，默认 `0` 使用 best-K；设为 `15` 启用
  COLMAP 风格逐像素 source CDF 抽样。建议质量档与
  `--patch_match_directional_hmm` 一起使用；为保持既有输出兼容，两项仍不默认开启。
- `--patch_match_min_num_valid_sources <n>`，默认 `1`。
- `--patch_match_max_num_aggregated_costs <n>`，默认 `3`，`0` 使用全部。
- `--patch_match_propagation_schedule <forward_backward|four_direction|checkerboard>`，
  默认 `checkerboard`。
- `--patch_match_geometric_refinement_iterations <n>`、
  `--patch_match_geometric_consistency_weight <w>`、
  `--patch_match_geometric_max_relative_depth_error <r>`。
- `--geometric_max_reprojection_error <px>` 默认 `1`，用于最终 consistency
  mask 的前后向重投影过滤。
- `--geometric_max_relative_depth_error <r>` 默认 `0`；`0` 表示仅使用
  COLMAP 兼容的重投影判据，正值额外启用相对深度约束。
- `--refine_normals_from_depth`、`--filter_photometric_max_cost <c>`。
- `--dense_map_format <text|binary>`，默认 `text`。
- `--max_num_source_images <n>`（默认 `8`）、`--min_num_shared_points <n>`（默认 `1`）。
- `--num_threads <n>`（默认 `1`）、`--image_cache_size <n>`（默认 `16`）、
  `--max_image_size <n>`、`--random_seed <n>`（默认 `42`）。

### stereo_fusion

```bash
$COLITE stereo_fusion \
  --model_path output/mvs/undistorted/sparse \
  --image_path output/mvs/undistorted/images --output_path output/mvs \
  --fusion_min_num_observations 2
```

该命令读取已有 dense maps，不重跑 PatchMatch。参数：

- `--fusion_min_num_observations <n>`：默认 `1`。
- `--fusion_max_num_observations <n>`：默认 `100000`。
- `--fusion_max_traversal_depth <n>`：默认 `100`。
- `--fusion_max_reprojection_error <px>`：默认 `2`。
- `--fusion_max_relative_depth_error <r>`：默认 `0.01`。
- `--fusion_max_normal_error <deg>`：默认 `10`。
- `--fusion_max_view_angle <deg>`：默认 `180`。
- `--fusion_source_pixel_search_radius <n>`：默认 `1`。
- `--filter_geometric_min_num_consistent_sources <n>`：一致性来源下限。

### surface 命令

```bash
$COLITE poisson_mesher --input_path output/mvs/dense/fused.ply \
  --output_path output/surface/meshed-poisson.ply \
  --PoissonMeshing.depth 10 --PoissonMeshing.point_weight 1 \
  --PoissonMeshing.color 1 --PoissonMeshing.trim 10 \
  --PoissonMeshing.num_threads -1

$COLITE delaunay_mesher --input_path output/mvs/dense \
  --model_path output/mvs/undistorted/sparse \
  --output_path output/surface/meshed-delaunay.ply

$COLITE mesh_simplifier --input_path output/surface/meshed-poisson.ply \
  --output_path output/surface/mesh-simplified.ply \
  --MeshSimplification.target_ratio 0.5

$COLITE mesh_texturer --input_path output/surface/mesh-simplified.ply \
  --model_path output/mvs/undistorted/sparse \
  --image_path output/mvs/undistorted/images \
  --output_path output/surface/textured
```

Poisson 参数：`depth` 默认 `13`、`point_weight` 默认 `1`、`color` 默认 `1`、
`trim` 默认 `10`、`num_threads` 默认后端配置。

Delaunay 参数前缀为 `--DelaunayMeshing.`：`max_proj_dist=20`、
`max_depth_dist=0.05`、`visibility_sigma=3`、`distance_sigma_factor=1`、
`quality_regularization=1`、`max_side_length_factor=25`、
`max_side_length_percentile=95`、`num_threads`。输入目录需包含 `fused.ply` 和
`points_visibility.bin`；兼容 `fused.ply.vis`。

简化参数前缀为 `--MeshSimplification.`：`target_ratio=0.5`、
`target_num_triangles=0`、`max_error=0`、`preserve_boundary=1`、
`max_normal_change=60`。

纹理参数前缀为 `--MeshTextureMapping.`：`atlas_size=4096`、
`max_atlas_size=32768`、`atlas_patch_padding=2`、`inpaint_radius=5`、
`texture_scale_factor=1`、`min_cos_normal_angle=0.1`、
`min_visible_vertices=3`、`view_selection_smoothing_iterations=3`、
`visibility_resolution=1024`、`occlusion_relative_tolerance=0.01`、
`apply_color_correction=1`、`color_correction_regularization=0.1`、`num_threads`。

当前 CPU 纹理可见性计算随三角形数和图像分辨率增长较快。快速验收或普通 CPU 环境建议先用
`--MeshSimplification.target_num_triangles 500`，并设置
`--MeshTextureMapping.visibility_resolution 512`；生产参数应根据模型规模逐级提高。

## recon_mvs

`recon_mvs` 是兼容旧用法的组合入口。必填 `--model_path` 和 `--output_path`；
执行需要图像的阶段时还需 `--image_path`。阶段开关为：

- `--image_undistorter`
- `--initialize_patch_match`
- `--run_patch_match`
- `--check_geometric_consistency`
- `--fuse_stereo`
- `--filter_dense_cloud`
- `--mesh_dense_cloud`
- `--poisson_mesher` / `--delaunay_mesher`
- `--mesh_simplifier`
- `--mesh_texturer`

其余参数与上面的独立 MVS/surface 参数语义相同，但使用小写形式，例如
`--poisson_depth`、`--mesh_simplification_ratio`、`--texture_atlas_size`。
查看当前构建的完整默认值：

```bash
$COLITE recon_mvs --help
```

## recon_auto

默认完成原始图像到纹理网格：

```bash
$COLITE recon_auto --image_path data/building-18 --output_path output/auto
```

- `--start_from <sfm|undistort|stereo|fusion|mesh|simplify|texture>`。
- `--stop_after <sfm|undistort|stereo|fusion|mesh|simplify|texture>`，默认 `texture`。
- `--mesher <poisson|delaunay|none>`，默认 `poisson`。
- `--simplify <0|1>`，默认 `1`。
- `--texture <0|1>`，默认 `1`。
- `--sfm.<option> <value>`：传给 `recon_sfm`，例如 `--sfm.max_num_features 4096`。
- `--mvs.<option> <value>`：传给 MVS stages，例如 `--mvs.max_image_size 1200`。
- Poisson、Delaunay、简化、纹理参数直接使用上述带大写前缀的独立命令形式。

示例：

```bash
$COLITE recon_auto --image_path data/building-18 --output_path output/auto \
  --sfm.max_num_features 4096 --mvs.patch_match_backend cpu \
  --mvs.patch_match_iterations 2 --mvs.max_image_size 1200 \
  --PoissonMeshing.depth 8 --MeshTextureMapping.atlas_size 2048 --resume
```

## compare_colmap

```bash
$COLITE compare_colmap --reference_path colmap/sparse/0 \
  --candidate_path output/sfm --output_path output/compare
```

只接受以上三个必填路径，输出 image、pose、point、pair 和 summary 差异报告。位姿比较会先按共同图像相机中心执行 Sim(3) 对齐，再报告旋转和相机中心误差；独立重建的 `point3D_id` 不具有跨模型语义。Mapper 的 building-18 对齐方法和结果见 [mapper_colmap311_validation_20260715.md](mapper_colmap311_validation_20260715.md)。

## building-18 可复现测试顺序

测试输出统一放在 `build-wsl-release/data-pipeline-test`。必须先通过单步骤，再运行
组合入口：

1. `feature_extractor`。
2. `sequential_matcher`，并分别验证 exhaustive/spatial/vocab-tree 命令。
3. `mapper`。
4. `image_undistorter`。
5. `patch_match_stereo`。
6. `stereo_fusion`。
7. `poisson_mesher`、`delaunay_mesher`、`mesh_simplifier`、`mesh_texturer`。
8. `recon_sfm`。
9. `recon_mvs`。
10. `recon_auto`。

实际测试参数和结果记录在本文末尾的“验证记录”中。

## 验证记录

测试日期：2026-07-14。环境：WSL Ubuntu-22.04、Release 构建、CPU 后端；输入为
`data/building-18` 的 18 张 3072x2304 JPEG。输出保存在
`build-wsl-release/data-pipeline-test`，未修改原始测试图像。

为缩短回归测试时间，特征上限设为 2048，去畸变最大边长设为 800，PatchMatch 使用 1 次迭代、
1 层金字塔、2 个线程和 binary dense map。该参数组合用于验证命令和数据衔接，不代表推荐的
高质量重建参数。

### 单步骤结果

| 步骤 | 关键参数 | 实测结果 |
|---|---|---|
| `feature_extractor` | `cvsift`，2048 特征，1 线程 | 18/18 特征文件，317.1 s |
| `sequential_matcher` | overlap=5，FLANN，ratio=0.8 | 75 对候选，72 对通过，13819 inliers |
| `exhaustive_matcher` | FLANN，ratio=0.8 | 153 对候选，94 对通过，14619 inliers |
| `spatial_matcher` | 10 邻居；测试副本注入线性位置先验 | 105 对候选，84 对通过，14232 inliers |
| `vocab_tree_matcher` | 10 邻居，256 visual words | 102 对候选，68 对通过，13244 inliers |
| `mapper` | sequential workspace | 注册 5 张图，1204 个稀疏点 |
| `image_undistorter` | max image size=800 | 5/5 图像，3298 个稀疏深度样本 |
| `patch_match_stereo` | CPU，iterations=1，radius=1 | 5/5 深度图，383495 个几何一致像素，85.7 s |
| `stereo_fusion` | min observations=2 | 78518 点、232944 observations |
| `poisson_mesher` | depth=7，trim=0 | 12787 顶点、25316 面 |
| `delaunay_mesher` | 默认距离阈值 | 4360 顶点、8631 面 |
| `mesh_simplifier` | target ratio=0.5 | 6407 顶点、12651 面 |
| `mesh_texturer` | 先简化到约 500 面，atlas/visibility=512 | OBJ 314 顶点、461 面，MTL/PNG 齐全 |

`spatial_matcher` 的位置仅注入到测试输出副本
`single/sfm-workspace-spatial/images.tsv`，用于验证命令路径；它不是由 building-18 图像恢复出的 GPS。

真实图像还暴露并修复了 `cvsift` 在最小高斯金字塔层上“卷积核半径大于图像尺寸”时的
reflect border 越界。修复后 1 线程和 4 线程特征提取均通过，并新增
`CvSiftHandlesKernelWiderThanSmallestPyramidLevel` 回归测试。

### 组合流程结果

| 流程 | 输出 | 实测结果 |
|---|---|---|
| `recon_sfm` | `composed/sfm` | 注册 4/18 张，1362 稀疏点，平均重投影误差 0.333 px，124.7 s |
| `recon_mvs` | `composed/mvs` | 4/4 深度图，15730 融合点、51910 observations，45.5 s |
| `recon_auto` | `auto` | SfM→MVS→Poisson→简化→纹理全部完成，184.8 s |

`recon_auto` 最终产物包括 53800 点的 `mvs/dense/fused.ply`、43128 面的
`surface/meshed-poisson.ply`、400 面的 `surface/mesh-simplified.ply`，以及
`surface/textured/{mesh.obj,mesh.mtl,texture.png}`。所有 MVS、网格和纹理阶段均生成成功的
`.colite/stages/*.json` manifest。

### 已知性能限制

直接对 12651 面网格、原分辨率可见性配置执行当前 CPU `mesh_texturer`，超过约 19 分钟仍未
完成，测试被终止且没有产生输出；同一真实模型先简化到 461 面并使用
`visibility_resolution=512` 后可在 1 秒内完成。因此本文的“通过”表示功能与产物契约正确，
大网格纹理的加速仍是后续性能优化项。
