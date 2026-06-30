# COLITE MVS 开发状态

更新时间：2026-06-19

本文用于承接 SfM 阶段之后的 MVS 开发上下文。当前路线保持“保守 COLMAP 化”：先建立稀疏模型输入、图像 workspace、source view 选择、CPU PatchMatch、几何一致性和 stereo fusion 的主流程，再逐步补齐性能、去重和更严格的 COLMAP 细节。

## 当前实现范围

- 已新增 `colite::mvs` 模块和 `recon_mvs` 命令行。
- 支持读取 COLMAP text model：`cameras.txt`、`images.txt`、`points3D.txt`。
- 支持按共享 SfM 点数选择每张参考图像的 source images。
- 支持为每张注册图像生成稀疏深度样本和深度范围。
- 支持 MVS image workspace/cache：
  - 从 `image_path / image.name` 加载图像。
  - 转为 `CV_32FC1` 灰度图，像素范围 `[0, 1]`。
  - 支持 `--max_image_size` 缩放和 LRU cache。
- 已实现 dense map 基础结构：
  - `DenseDepthMap`
  - `NormalMap`
  - `DenseCostMap`
  - ASCII 调试读写。
- 已实现 CPU PatchMatch 基础链路：
  - 随机 depth/normal 初始化。
  - 灰度双线性采样。
  - ZNCC photometric cost。
  - depth hypothesis 投影。
  - depth+normal 局部平面 patch 重投影。
  - 多源视图 top-k cost 聚合。
  - best cost map。
  - forward/backward sweep 传播。
  - 随机扰动和多轮 PatchMatch 控制器。
- 已实现结果统计与过滤：
  - photometric cost 统计和阈值过滤。
  - geometric consistency graph。
  - 几何一致性过滤。
- 已实现 COLMAP 化的 stereo fusion 第一阶段：
  - `FusedPoint` 保存世界坐标、世界法线、灰度转 RGB 颜色和来源像素。
  - `FuseStereoDepthMaps` 沿 consistency graph 查找互相一致的跨视图像素 track。
  - 使用 visited bitmap 避免同一 depth 像素被重复融合。
  - 对同一 track 内的坐标、法线和颜色做平均，只输出一个 fused point。
  - 支持最少融合观测数、最大 track 观测数、最大遍历深度、重投影阈值和相对深度阈值。
  - `recon_mvs --fuse_stereo` 输出 `dense/fused.ply`。
  - `summary.txt` 写入 `fused_points`、`fusion_tracks`、`fused_observations`、`fusion_visited_pixels`、`fusion_rejected_tracks`。

## 主要代码位置

- MVS pipeline：`include/colite/mvs/mvs_pipeline.h`，`src/colite/mvs/mvs_pipeline.cpp`
- dense map：`include/colite/mvs/dense_map.h`，`src/colite/mvs/dense_map.cpp`
- image workspace/cache：`include/colite/mvs/image_workspace.h`，`src/colite/mvs/image_workspace.cpp`
- CPU PatchMatch / consistency / fusion：`include/colite/mvs/patch_match.h`，`src/colite/mvs/patch_match.cpp`
- MVS 命令行：`app/recon_mvs`
- 测试：
  - `tests/mvs/dense_map_test.cpp`
  - `tests/mvs/image_workspace_test.cpp`
  - `tests/mvs/mvs_pipeline_test.cpp`
  - `tests/mvs/patch_match_test.cpp`
  - `tests/integration/recon_mvs_app_test.cpp`

## recon_mvs 常用命令

只生成 MVS workspace 诊断：

```bash
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --output_path build-wsl/mvs-building-18/colite-sparse-workspace
```

运行 CPU PatchMatch + 几何一致性 + stereo fusion：

```bash
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl/mvs-building-18/colite-stereo-fusion-48 \
  --run_patch_match \
  --max_image_size 48 \
  --patch_match_iterations 1 \
  --patch_match_window_radius 1 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --filter_geometric_min_num_consistent_sources 1
```

主要参数：

```bash
--max_num_source_images <n>   每张参考图最多保留的源图数量，默认 5
--min_num_shared_points <n>   源图至少需要共享的稀疏点数量，默认 1
--image_path <dir>            输入图像目录
--initialize_patch_match      只输出 PatchMatch 初始化 depth/normal map
--run_patch_match             运行 CPU PatchMatch 并输出 depth/normal/cost map
--patch_match_iterations <n>  PatchMatch 迭代轮数，默认 3
--patch_match_window_radius <n> ZNCC patch 半径，默认 2
--filter_photometric_max_cost <c> 过滤 cost 大于 c 的像素，默认关闭
--check_geometric_consistency 生成 consistency graph 和几何诊断
--filter_geometric_min_num_consistent_sources <n> 过滤一致源图少于 n 的像素
--fuse_stereo                 融合过滤后的 depth maps，输出 dense/fused.ply
--fusion_min_num_observations <n> 一个 fused track 至少需要的像素观测数，默认 1
--fusion_max_num_observations <n> 一个 fused track 最多使用的像素观测数，默认 100000
--fusion_max_traversal_depth <n> consistency graph 最大遍历深度，默认 100
--fusion_max_reprojection_error <px> fusion 投影到源图最近像素的误差阈值，默认 2
--fusion_max_relative_depth_error <r> fusion 源图相对深度误差阈值，默认 0.01
--geometric_max_reprojection_error <px> 几何一致性重投影阈值，默认 2
--geometric_max_relative_depth_error <r> 几何一致性相对深度阈值，默认 0.01
--image_cache_size <n>        图像 LRU cache 容量，默认 16
--max_image_size <n>          MVS 图像最长边缩放限制
--random_seed <n>             PatchMatch 随机种子，默认 42
```

## 当前输出

- `summary.txt`
- `source_images.tsv`
- `sparse_depth_samples.tsv`
- `stereo/depth_maps/*.txt`
- `stereo/normal_maps/*.txt`
- `stereo/cost_maps/*.txt`
- `stereo/consistency_graphs/*.txt`
- `stereo/patch_match.tsv`
- `dense/fused.ply`

## 当前 stereo fusion 说明

当前 fusion 以“稳定可诊断 + COLMAP 风格去重”为目标：

- 从一个有效 seed 像素出发，沿 consistency graph 中记录的 source image id 投影到源图。
- 源图候选像素需要满足 depth/normal/cost 有效、相对深度误差合格、重投影到最近像素误差合格。
- 默认要求源图像素的 consistency graph 中也包含当前参考图，形成互相一致约束。
- 同一个 track 中的观测会平均坐标、法线和颜色。
- 融合成功后将 track 内所有像素标记为 visited，后续不会重复生成点。

暂未实现的 COLMAP 化细节：

- view angle / normal angle 更严格过滤。
- consistency graph 二进制格式和更完整的轨迹诊断输出。
- 二进制 depth/normal/consistency graph 格式。
- 并行化和内存访问优化。

## 后续优先事项

1. 增加 fusion 阶段的法线夹角和视角夹角过滤。
2. 输出 fusion track 诊断文件，便于和 COLMAP `stereo_fusion` 对比。
3. 与 COLMAP `patch_match_stereo` / `stereo_fusion` 在 building-18 上做阶段性对比。
4. 优化 CPU PatchMatch 性能：图像级并行、patch cost 缓存、减少文本输出开销。
5. 开始 meshing 前的 dense point cloud 清理和统计。

## 验证命令

构建：

```bash
cmake --build --preset wsl-debug
```

MVS 相关测试：

```bash
ctest --test-dir build-wsl -R 'Mvs|PatchMatch|StereoFusion|ReconMvs' --output-on-failure
```

全量测试：

```bash
ctest --test-dir build-wsl --output-on-failure
```

## 2026-06-20 Stereo Fusion 更新

本轮补齐了更接近 COLMAP `stereo_fusion` 的角度过滤和融合轨迹诊断：

- `StereoFusionOptions` 新增：
  - `max_normal_error_degrees`，默认 `10`，用于限制 seed 法线与 source 法线的夹角。
  - `max_view_angle_degrees`，默认 `180`，用于限制 seed 相机视线与 source 相机视线之间的夹角；该项是 COLITE 当前显式诊断阈值，便于调试不同视图基线。
- `StereoFusionSummary` 新增：
  - `num_rejected_normal_observations`
  - `num_rejected_view_observations`
- `StereoFusionResult` 新增 `tracks`，每个成功 fused point 对应一条 `StereoFusionTrack`，记录 seed 图像、seed 像素、观测数、唯一图像数、最大遍历深度、最大 normal angle 和最大 view angle。
- `recon_mvs` 新增命令行参数：
  - `--fusion_max_normal_error <deg>`，默认 `10`
  - `--fusion_max_view_angle <deg>`，默认 `180`
- `summary.txt` 新增：
  - `fusion_rejected_normal_observations`
  - `fusion_rejected_view_observations`
- `dense/fusion_tracks.tsv` 新增融合轨迹诊断输出，表头为：

```text
point_idx	seed_image_id	seed_row	seed_col	num_observations	num_unique_images	max_traversal_depth	max_normal_angle_degrees	max_view_angle_degrees
```

当前 fusion 路径仍保持保守实现：沿 consistency graph 做 BFS，候选 source 像素需要通过重投影、相对深度、互相一致性、normal angle 和 view angle 检查后才会加入 track。

## 2026-06-22 Stereo Fusion 诊断更新

本轮继续补充 fusion 可观测性，不改变融合结果本身：

- `StereoFusionResult` 新增 `pair_diagnostics`，记录 BFS 遍历过程中每条 reference/source 图像边的候选和拒绝情况。
- 新增 `dense/fusion_image_pairs.tsv`，字段为：

```text
reference_image_id	source_image_id	num_candidates	num_accepted	num_rejected_normal	num_rejected_view	num_rejected_other
```

- 新增 `dense/fusion_track_lengths.tsv`，用于查看 fused track 长度分布，字段为：

```text
num_observations	num_tracks
```

说明：

- `fusion_tracks.tsv` 的 `seed_image_id` 表示生成该 fused point 的原始 seed 像素所属图像。
- `fusion_image_pairs.tsv` 的 `reference_image_id` 表示 BFS 当前边的起点图像，不一定等于原始 seed 图像；这样更适合诊断 consistency graph 遍历过程中哪些 image pair 贡献有效观测、哪些 image pair 被 normal/view/其他条件拒绝。

## 2026-06-22 PatchMatch / Fusion 诊断结论与优化

基于 building-18、`max_image_size=48`、`patch_match_iterations=1` 的诊断：

- 默认 5 个 source view 时：
  - `fused_points=4704`
  - `fused_observations=5613`
  - fusion pair 候选 `7492`，接受 `909`
  - `other` 拒绝主要来自：
    - `num_rejected_invalid=3138`
    - `num_rejected_mutual_consistency=1457`
    - `num_rejected_duplicate=968`
    - `num_rejected_depth=548`
- `num_rejected_visited=159` 很小，说明当前瓶颈不是 visited 去重过强。
- `num_rejected_projection=0`，说明相机投影/视野覆盖不是主要问题。
- 最大问题是 source view 覆盖偏窄，以及过滤后的 source 像素有效性/互相一致性不足。

本轮保守优化：

- `MvsPipelineOptions::max_num_source_images` 和 `recon_mvs --max_num_source_images` 默认值从 `5` 提升到 `8`。
- 新增 PatchMatch 多源聚合命令行参数：
  - `--patch_match_min_num_valid_sources <n>`，默认 `1`
  - `--patch_match_max_num_aggregated_costs <n>`，默认 `3`，设为 `0` 表示使用全部有效 source cost

对比结论：

- `--max_num_source_images 8` 在 building-18 smoke 中提升明显：
  - `fused_points: 4704 -> 6030`
  - `fused_observations: 5613 -> 7058`
  - `geometric_consistent_pixels: 5613 -> 7058`
- `--patch_match_min_num_valid_sources 2` 更偏质量保守：
  - 点数略降 `4704 -> 4680`
  - 单观测 track 略减少，多观测 track 略增加
  - depth / normal 拒绝略降，但 invalid 略升
- `--patch_match_min_num_valid_sources 2 --patch_match_max_num_aggregated_costs 0` 在该低分辨率 smoke 下过严：
  - `fused_points=4566`
  - `geometric_consistent_pixels=5451`

当前推荐 smoke 参数：

```bash
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl/mvs-building-18/colite-stereo-fusion-src8-48 \
  --max_num_source_images 8 \
  --run_patch_match \
  --max_image_size 48 \
  --patch_match_iterations 1 \
  --patch_match_window_radius 1 \
  --patch_match_min_num_valid_sources 1 \
  --patch_match_max_num_aggregated_costs 3 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --fusion_max_normal_error 10 \
  --fusion_max_view_angle 180 \
  --filter_geometric_min_num_consistent_sources 1
```

PatchMatch 当前进度：

- 已实现：
  - depth/normal 初始化
  - 灰度图 workspace/cache
  - ZNCC patch cost
  - plane hypothesis 重投影
  - 多源 top-k cost 聚合
  - best cost map
  - forward/backward sweep 传播
  - random perturbation
  - photometric stats/filter
  - geometric consistency graph/filter
  - stereo fusion 与 PLY 输出
  - fusion track / pair diagnostics
- 仍缺：
  - COLMAP 风格 source view 选择评分：三角夹角、尺度、视角覆盖，而不是只按共享稀疏点
  - 几何一致性参与 PatchMatch 迭代中的 cost，而不是只作为后处理 filter
  - 更完整的传播方向、棋盘/红黑调度、随机采样策略
  - 自适应深度范围与更稳定的法线采样
  - depth/normal/consistency graph 二进制格式
  - 多线程并行和缓存优化
  - 最终 meshing 前的 dense cloud 清理

## 2026-06-23 Source View Scoring 更新

本轮开始执行 `docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md` 的 Milestone 1，先实现 COLMAP 风格 source view 评分的第一版。

实现内容：

- `SourceImageScore` 新增：
  - `mean_triangulation_angle_degrees`
  - `score`
- `MvsPipelineOptions` 新增：
  - `min_triangulation_angle_degrees`，默认 `1`
  - `max_triangulation_angle_degrees`，默认 `30`
  - `triangulation_angle_weight`，默认 `1`
  - `shared_points_weight`，默认 `1`
- `SelectSourceImages` 不再只按共享稀疏点数量排序，而是：
  - 先收集 target/source 共享 3D 点数量。
  - 对每个共享点计算 target/source 两个相机中心到该 3D 点的三角夹角。
  - 只把 `[min_triangulation_angle_degrees, max_triangulation_angle_degrees]` 范围内的角度纳入均值。
  - 使用 `score = triangulation_angle_weight * mean_angle + shared_points_weight * num_shared_points` 排序。
  - score 相同则按共享点数、image id 排序。
- `recon_mvs` 新增参数：
  - `--source_min_triangulation_angle <deg>`
  - `--source_max_triangulation_angle <deg>`
  - `--source_triangulation_angle_weight <w>`
  - `--source_shared_points_weight <w>`
- `source_images.tsv` 新增列：

```text
mean_triangulation_angle_degrees	score
```

调试建议：

- 若 source view 覆盖不足，优先保持 `--max_num_source_images 8` 或更高，并查看 `source_images.tsv` 中低 score 的图像是否仍进入候选。
- 若相邻图像过近导致 baseline 太小，可适当提高 `--source_min_triangulation_angle`。
- 若过度偏好大 baseline 导致匹配/重投影不稳，可降低 `--source_triangulation_angle_weight` 或降低 `--source_max_triangulation_angle`。
- building-18 smoke 中，`source_shared_points_weight=1` 比 `0.01` 更稳；当前默认让共享点数主导，三角夹角主要用于改善同等共享点附近的排序。

## 2026-06-23 PatchMatch 几何一致性代价更新

本轮完成 `docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md` 的 Milestone 2：把几何一致性从纯后处理推进到 PatchMatch cost/refinement pass 中。

实现内容：

- `PatchMatchCostOptions` 新增：
  - `geometric_consistency_weight`，默认 `0`，保持原纯光度行为。
  - `geometric_max_relative_depth_error`，默认 `0.01`，用于归一化 source depth 相对误差。
- `ComputePlaneHypothesisCost` 支持可选 `DenseDepthMap* source_depth_map`。若存在 source depth，则把当前候选深度投影到 source 图，双线性采样 source depth，并把相对深度误差作为软惩罚加入 cost。
- `ComputeMultiSourcePlaneHypothesisCost`、`ComputeBestCostMap`、传播 sweep、随机扰动均支持可选 source depth maps。
- 新增 `RunPatchMatchStereoForImageWithSourceDepths`，用于第二轮几何 refinement pass。
- `recon_mvs` 新增参数：
  - `--patch_match_geometric_refinement_iterations <n>`
  - `--patch_match_geometric_consistency_weight <w>`
  - `--patch_match_geometric_max_relative_depth_error <r>`

当前推荐：

- 默认仍关闭几何 refinement。
- building-18 低分辨率 smoke 显示 `weight=0.05` 比 `0.2` 更稳，建议后续先用保守权重。

building-18 `max_image_size=32` 对比：

```text
baseline:
photometric_filtered_pixels 4182
geometric_consistent_pixels 2889
fused_points 2440
fused_observations 2889

geometric refinement, weight=0.2:
photometric_filtered_pixels 7369
geometric_consistent_pixels 2743
fused_points 2082
fused_observations 2743

geometric refinement, weight=0.05:
photometric_filtered_pixels 4547
geometric_consistent_pixels 3447
fused_points 2629
fused_observations 3447
```

注意：

- `max_image_size=48`、`weight=0.2` 在当前单线程 CPU 实现下超过 5 分钟 smoke timeout，说明几何 cost 接入后性能压力明显，后续 Milestone 6 的多线程与缓存优化会很关键。
- 当前 refinement 是“先跑一轮普通 PatchMatch 得到所有 source depth，再跑一轮带 source-depth soft penalty 的 PatchMatch”。它还不是 COLMAP 完整的迭代式几何一致性调度，但已经形成可测试、可调参的闭环。

## 2026-06-23 PatchMatch 传播调度更新

本轮完成 `docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md` 的 Milestone 3：补充更完整的 PatchMatch 传播调度。

实现内容：

- 新增 `PatchMatchPropagationSchedule`：
  - `kForwardBackward`，默认行为，保持原有 forward/backward sweep。
  - `kFourDirection`，每轮执行 forward、backward、top-down、bottom-up 四个方向。
  - `kCheckerboard`，每轮执行 4 组红黑传播，也就是 8 个 half-pass。
- `PatchMatchSweepDirection` 新增：
  - `kTopDown`
  - `kBottomUp`
- 新增 `RunPatchMatchCheckerboardPropagation`，只更新指定 `(row + col) % 2` 奇偶格，候选来自四邻域。
- `PatchMatchStereoSummary` 新增：
  - `num_top_down_sweeps`
  - `num_bottom_up_sweeps`
  - `num_checkerboard_passes`
- `recon_mvs` 新增参数：
  - `--patch_match_propagation_schedule forward_backward|four_direction|checkerboard`
- `stereo/patch_match.tsv` 新增列：
  - `forward_sweeps`
  - `backward_sweeps`
  - `top_down_sweeps`
  - `bottom_up_sweeps`
  - `checkerboard_passes`

building-18 `max_image_size=32`、`patch_match_iterations=1` 对比：

```text
baseline forward_backward:
photometric_filtered_pixels 4182
geometric_consistent_pixels 2889
fused_points 2440
fused_observations 2889

four_direction:
photometric_filtered_pixels 4168
geometric_consistent_pixels 3014
fused_points 2544
fused_observations 3014

checkerboard, 4 red/black rounds:
photometric_filtered_pixels 4184
geometric_consistent_pixels 2099
fused_points 1899
fused_observations 2099
```

当前推荐：

- 默认仍使用 `forward_backward`，保持稳定。
- `four_direction` 当前有小幅正向收益，可以作为优先实验选项。
- `checkerboard` 当前已可用但不推荐作为质量路径；它需要后续配合更完整的随机采样、局部候选策略和并行化再评估。

## 2026-06-23 PatchMatch 初始化与扰动稳定性更新

本轮完成 `docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md` 的 Milestone 4：稳定深度范围、初始法线和扰动调度。

实现内容：

- `PatchMatchInitializationOptions` 新增稀疏深度分位数范围：
  - `sparse_depth_min_percentile = 0.01`
  - `sparse_depth_max_percentile = 0.99`
  - `depth_range_padding = 0.0`
- 稀疏深度初始化现在会先对有效深度排序，再按分位数取 `[depth_min, depth_max]`，可显式剔除极端离群深度。
- `PatchMatchInitializationOptions` 新增随机初始法线：
  - `randomize_initial_normals = false`
  - `initial_normal_hemisphere_degrees = 60`
- 随机初始法线围绕 `default_normal` 的球冠采样，固定 `random_seed` 下可复现，且保持单位长度。
- `PatchMatchStereoSummary` 新增扰动诊断：
  - `update_ratios`
  - `depth_radii`
  - `normal_radii_degrees`
- `PatchMatchStereoOptions` 新增：
  - `adaptive_perturbation_min_update_ratio = 0.01`
- 每轮 PatchMatch 会记录传播+扰动候选的更新率；若低于阈值，下一轮 depth/normal 扰动半径额外收缩一次。

验证结果：

```text
ctest --test-dir build-wsl --output-on-failure
120/120 passed
```

building-18 `max_image_size=32`、默认 `forward_backward` 对比：

```text
photometric_filtered_pixels 4182
geometric_consistent_pixels 2889
fused_points 2440
fused_observations 2889
```

说明：

- 该 smoke 与 Milestone 3 的默认基线一致，说明 Milestone 4 默认配置没有扰动现有主流程。
- `recon_mvs` 目前仍使用 SfM/MVS summary 中的显式 depth min/max 调用 PatchMatch；稀疏深度分位数逻辑主要影响 `InitializePatchMatchMaps(image, sparse_depth_map, options)` 这一路库接口。
- 随机初始法线默认关闭。后续若要在 building-18 中评估，需要给 `recon_mvs` 增加对应 CLI，或在更完整的 PatchMatch 参数配置里统一暴露。

## 2026-06-25 Milestone 8 Regression Matrix

Milestone 8 added a clean experiment matrix in `docs/mvs_experiments.md`.
Current building-18 smoke case: `colite-ms8-smoke-48`.

Key metrics:

```text
photometric_filtered_pixels 4334
geometric_consistent_pixels 6983
fused_points 5969
fused_observations 6983
single_observation_tracks 5201
multi_observation_tracks 768
runtime_seconds 69.40
```

Verification command for milestone completion remains:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && cmake --build --preset wsl-debug && ctest --test-dir build-wsl --output-on-failure"
```

## 2026-06-30 PatchMatch CUDA backend boundary

本轮先完成 GPU PatchMatch 的工程边界，而不是直接替换 CPU
PatchMatch 算法：

- `PatchMatchStereoOptions` 新增 `backend` 和 `cuda_gpu_index`。
- `PatchMatchBackend` 支持 `auto`、`cpu`、`cuda`；`auto` 是默认值，后续
  CUDA backend 可用时优先走 CUDA，否则回退 CPU。
- `recon_mvs` 新增：
  - `--patch_match_backend <auto|cpu|cuda>`
  - `--patch_match_gpu_index <n>`
- `PatchMatchStereoSummary` 新增 `requested_backend` 和 `actual_backend`。
- `stereo/patch_match.tsv` 新增 `requested_backend`、`actual_backend` 诊断列。
- CMake 新增 `COLITE_ENABLE_CUDA_MVS` 选项，并加入 CUDA PatchMatch stub
  入口，后续真实 CUDA kernel 会从该入口替换。

当前状态：

- `--patch_match_backend auto` 在当前构建中回退 CPU。
- `--patch_match_backend cpu` 强制 CPU。
- `--patch_match_backend cuda` 在 CUDA kernel 未实现前会明确失败。
- WSL 中已确认 CUDA 编译器路径：
  `/usr/local/cuda-13.2/bin/nvcc`，版本 `13.2.51`。

验证：

```text
cmake --build build-wsl --target colite_core_tests -j2
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
25/25 passed
```

## 2026-06-30 First runnable CUDA PatchMatch backend

本轮将 CUDA PatchMatch 从 stub 推进到可运行的第一版 GPU backend：

- CUDA build 下使用 `src/colite/mvs/patch_match_cuda.cu`。
- 非 CUDA build 下仍使用 `src/colite/mvs/patch_match_cuda_stub.cpp`。
- GPU backend 已实现：
  - reference/source 图像上传；
  - packed source image cache；
  - optional source depth map cache；
  - reference-to-source 相对位姿上传；
  - depth/normal/cost device map；
  - 随机 depth 初始化；
  - 默认 normal 初始化；
  - ZNCC photometric cost；
  - plane reprojection；
  - top-k multi-source cost aggregation；
  - source-depth geometric penalty；
  - propagation kernel；
  - random depth perturbation kernel。
- `COLITE_ENABLE_CUDA_MVS=ON` 时，`--patch_match_backend auto` 会优先尝试
  CUDA。CUDA 当前不支持的请求会在 auto 模式回退 CPU；强制
  `--patch_match_backend cuda` 时会返回明确错误。

当前限制：

- 这是第一版可运行 CUDA backend，不是完整 COLMAP CUDA PatchMatch。
- 当前只支持 single-scale；`patch_match_pyramid_levels > 1` 会回退 CPU。
- GPU perturbation 当前主要扰动 depth，normal perturbation 仍需补齐。
- 尚未实现 COLMAP 的 selection probability、rotation sweep、incident angle
  weighting 和完整 GPU 几何过滤。

验证：

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest' --output-on-failure
12/12 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R 'ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
14/14 passed

ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
26/26 passed
```

## 2026-06-30 CUDA layered texture image cache

本轮继续向 COLMAP CUDA PatchMatch 靠拢，将 CUDA backend 的图像访问从
packed linear buffer 升级为 layered CUDA texture：

- `patch_match_cuda.cu` 内部新增 `FloatLayeredTexture` RAII 封装。
- reference image、source image cache、optional source depth map cache 现在
  上传到 CUDA layered array。
- cost / propagation / perturbation kernels 通过 `cudaTextureObject_t` 采样。
- texture 使用 linear filtering、border addressing、unnormalized coordinates。
- CUDA 侧旧的手写图像双线性采样路径已移除。

当前仍未完成的 COLMAP CUDA 细节：

- pose 仍通过 device struct array 传入，尚未改成 COLMAP 风格 pose texture。
- 暂不支持 CUDA multi-scale pyramid。
- selection probability、rotation sweep scheduling、normal perturbation 和更完整
  geometric filtering 仍待实现。

验证：

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
26/26 passed

ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
26/26 passed
```

## 2026-06-30 CUDA pose texture 与几何代价组织

本轮继续按 COLMAP CUDA PatchMatch 的方向推进，把 CUDA backend 中的 source
pose 从 device struct array 改为 texture-backed pose cache：

- `patch_match_cuda.cu` 新增 pose layered texture，每个 source view 打包 43
  个 float 参数。
- 打包内容包括 `K`、reference-to-source `R/T`、source camera center、`P`
  以及 `invP`。
- CUDA kernel 现在通过 `cudaTextureObject_t` 读取 source intrinsics /
  transform；source metadata struct 只保留图像尺寸。
- 几何一致性代价从“直接比较 source camera z 深度”改为更接近 COLMAP 的
  forward/backward depth check：
  - reference hypothesis 通过 source `P` 投影到 source depth map；
  - 采样 source depth；
  - 通过 source `invP` 回投到 reference camera；
  - 比较回投后的 reference depth 与当前 hypothesis depth。

当前意义：

- CUDA PatchMatch 的数据组织更接近 COLMAP 的 pose/source texture cache。
- 后续加入 rotated sweeps、selection probability、source view 几何权重时，
  不需要再把 pose 组织重写一遍。

仍未完成：

- pose texture 目前只保存基础方向，尚未打包 rotated sweep 变体。
- selection probability、incident/view-angle weighting、完整几何过滤仍待实现。
- CUDA backend 仍只支持 single-scale；multi-scale pyramid 在 `auto` 模式下继续回退 CPU。

验证：

```text
CUDA:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
26/26 passed

CPU:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
26/26 passed
```
