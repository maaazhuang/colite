# COLITE MVS 开发状态

## 2026-07-14 CUDA NCC likelihood 动态 source probability

已实现 CUDA dense source-selection probability 的迭代更新闭环：

- 使用 COLMAP 3.11 的截断高斯 NCC likelihood，默认 `ncc_sigma=0.6`。
- 将 COLITE `0.5 * (1 - NCC)` 乘 2 后输入 COLMAP likelihood。
- 每轮根据当前 depth/normal 的逐 source 代价更新 Bayesian posterior，并与上一轮概率做 EMA。
- 未提供外部概率图时从 0.5 光度先验开始；动态光度概率与静态 source/geometry prior 相乘。
- CUDA workspace 复用 dense probability buffer，仅在显式启用时分配、上传和最终回传。
- checkerboard 路径使用确定性 floor gate，避免独立 Bernoulli 造成有效 source 数不足。
- `patch_match_initial.tsv` 和 `patch_match.tsv` 输出更新数量、每轮均值及最终各 source 均值。

命令行新增：

- `--patch_match_dynamic_source_selection`
- `--patch_match_ncc_sigma`
- `--patch_match_previous_selection_probability_weight`
- `--patch_match_min_selection_probability`
- `--patch_match_max_selection_probability`

building-18 双迭代 64px 回归中，静态与动态 floor gate 均为 272 个融合点；动态路径完成 442368 次概率更新，最终 source 均值约为 0.35 到 0.44，CUDA 总时间增加约 5%。动态与 bilateral NCC 组合为 219 点，主要回退仍来自当前小窗口 bilateral cost。

尚未补齐 COLMAP 的方向 HMM forward/backward message 和固定数量 CDF source sampling；这是下一阶段最重要的 source-selection 差距。

## 2026-07-13 CPU/CUDA bilateral weighted NCC

已参考 COLMAP 3.11 实现 CPU/CUDA 一致的 bilateral weighted NCC：

- 权重由参考中心灰度、参考采样灰度和空间距离计算。
- 同一权重用于参考/源图均值、方差和互协方差。
- 保留 COLITE 的 `0.5 * (1 - NCC)` 代价尺度，避免过滤阈值整体迁移。
- `recon_mvs` 新增 `--patch_match_bilateral_ncc`、`--patch_match_sigma_spatial` 和 `--patch_match_sigma_color`。
- `patch_match.tsv` 新增 `bilateral_ncc`、`sigma_spatial` 和 `sigma_color`，记录实际实验配置。

当前策略为显式启用、默认关闭。building-18 的 64px 快速回归表明，现有 radius=1 简化流程中直接启用 COLMAP 默认权重会把融合点从 268 降到 216；默认关闭后恢复为 268。双边代价的边缘抑制测试和 CPU/CUDA 数值一致性测试均通过，功能可用于后续 COLMAP 化组合实验。

下一阶段目标：补齐 NCC likelihood 和动态 source-selection probability 更新，使 bilateral NCC 不再孤立工作，再以更接近 COLMAP 的窗口和迭代参数重新评估默认启用。

## 2026-07-13 CUDA PatchMatch COLMAP 化候选搜索第一阶段

本阶段优先修正 CUDA 假设生成和传播语义，使其更接近 COLMAP 3.11 PatchMatch：

- CUDA 初始化支持确定性随机法线，使用已有随机种子和半球角度配置。
- 邻域传播不再直接复制邻居深度，而是将邻居深度/法线解释为平面，并求该平面与当前像素射线的交点深度。
- 每个随机样本分别评估“深度+法线”“仅法线”“仅深度”三个扰动候选，并复用同一套几何 pre-cost gate 和多源代价聚合。
- summary/TSV 新增 `plane_propagation_candidates` 和 `perturbation_candidate_evaluations`，用于区分传播与扰动搜索规模。

验证结果：

```text
CUDA PatchMatchBackendTest: 26 passed, 1 skipped
CPU  PatchMatchBackendTest: 27 passed

building-18, max_image_size=64, 18 images:
baseline geometric_consistent_pixels/fused_points: 216/216
current  geometric_consistent_pixels/fused_points: 268/268
relative fused-point change: +24.1%

plane_propagation_candidates: 868608
perturbation_candidate_evaluations: 165888
CUDA refinement total: 169.779 ms -> 109.978 ms
```

当前判断：质量指标没有回退，且小分辨率单次运行显示耗时下降；耗时结论仍需在更高分辨率和多次运行下验证。下一阶段是 bilateral weighted NCC，之后再继续动态 source selection probability 和更完整的重投影几何代价。

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
  - `kForwardBackward`，保留的兼容调度，保持原有 forward/backward sweep。
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

- 默认已切换为 `checkerboard`；`forward_backward` 保留为兼容和对照调度。
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

building-18 `max_image_size=32`、旧默认 `forward_backward` 对比：

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

## 2026-06-30 CUDA PatchMatch 细节补齐：sweep / selection / angle / pyramid

本轮继续补齐 CUDA PatchMatch 的 COLMAP 化细节：

- pose texture 从 43 个参数扩展为 47 个参数，并使用 3 个 layered sweep
  variant。
- 每个 source view 现在在 pose texture 中额外保存：
  - source geometry weight；
  - source selection probability；
  - rotated patch sweep 的 `cos/sin`。
- CUDA cost evaluation 会对每个 source 的 3 个 rotated sweep variant 取最优
  cost。
- 多源聚合新增：
  - per-pixel/source deterministic selection；
  - source geometry weight；
  - 基于 hypothesis normal 与 source camera center 的 incident-angle penalty。
- CUDA backend 已支持 multi-scale pyramid：
  - 每层缩放 reference/source 图像；
  - 每层缩放相机内参；
  - source depth map 随层缩放；
  - coarse depth/normal 通过最近邻上采样作为下一层初值。
- CUDA single-scale 结果现在会在开启 `--refine_normals_from_depth` 时调用
  现有 normal refinement，并写回 summary 统计，避免默认 `auto` backend 下
  `recon_mvs` 诊断缺失。

当前边界：

- rotated sweep 目前是 patch 坐标旋转变体，不是完整 COLMAP sweep scheduler。
- selection probability 目前是 source-level 概率并按 pixel/source 确定性采样，
  还不是 dense probability map。
- angle weighting 仍较保守，完整 normal/view-angle 过滤主要仍在 fusion 阶段。
- pyramid 层间 coarse map 通过 host memory 传递，后续可继续优化为 GPU cache。

验证：

```text
CUDA:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
27/27 passed

CPU:
本轮 CPU 回归命令因 WSL 执行触发环境用量限制未运行；CUDA 构建与 CUDA MVS 回归已通过。
```

## 2026-07-01 CUDA normal perturbation 与法线扰动诊断

本轮继续完善 CUDA PatchMatch 的随机扰动阶段：

- GPU 端新增 `float3` normal 运算 helper。
- `PerturbationKernel` 现在会在当前 normal 周围做球冠采样，生成
  `(depth, normal)` 联合候选，而不是只扰动 depth。
- 接受候选时会同时写回 depth map 和 normal map。
- CUDA summary 现在会记录 normal perturbation 的：
  - total normal angle；
  - max normal angle；
  - mean normal angle。
- 新增 CUDA 回归测试：初始 depth 固定、depth perturbation radius 为 0，
  确认 CUDA 路径可以通过 normal hypothesis 产生扰动更新。

验证：

```text
CUDA:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
28/28 passed

CPU:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
28/28 passed
```

## 2026-07-01 CUDA per-iteration diagnostics 与 source selection 统计

本轮给 CUDA PatchMatch 补充可调参诊断：

- `PatchMatchStereoSummary` 新增 source selection 总量：
  - tests；
  - acceptances；
  - rejections。
- 新增 per-level / per-iteration 诊断向量：
  - pyramid level；
  - iteration index；
  - candidate evaluations；
  - propagation updates；
  - perturbation updates；
  - source selection tests / acceptances / rejections。
- CUDA kernel 在 source selection gate 处统计每次 source 被考虑、接受或拒绝。
- CUDA 单尺度执行每轮 reset/copy counter，写入真实 per-iteration 诊断。
- CUDA pyramid 执行会把每层 iteration 记录标记为真实 coarse-to-fine level。
- `stereo/patch_match.tsv` 新增 aggregate selection counter 和 per-iteration CSV
  向量列，便于后续用脚本分析每轮行为。

验证：

```text
cmake --build build-wsl-cuda-mvs-tests --target colite_core colite_recon_mvs_exe -j2
passed

cmake --build build-wsl --target colite_core colite_recon_mvs_exe -j2
passed

cmake --build build-wsl-cuda-mvs-tests --target tests/CMakeFiles/colite_core_tests.dir/mvs/patch_match_test.cpp.o -j2
passed
```

说明：

- 完整 `colite_core_tests` 当前被 unrelated SfM 测试编译错误阻塞，错误来自
  `incremental_mapper_test.cpp` 对 `IncrementalMappingOptions` 中不存在字段的引用；
  本轮未修改该部分。

## 2026-07-03 CUDA dense selection probability map 与 red-black checkerboard

本轮继续完善 CUDA PatchMatch：

- `CudaPatchMatchRequest` 新增可选 dense selection probability maps。
- dense selection map 定义为 reference-view 像素网格，每个 source 一层。
- CUDA 后端会校验尺寸、clamp 到 `[0, 1]`，并上传为 layered texture。
- source selection gate 优先采样 dense probability texture；未提供时回退到
  pose texture 中的 source-level probability。
- CUDA pyramid 每层会同步缩放 dense selection maps。
- 新增显式 red-black `CheckerboardPropagationKernel`，只更新指定 parity 像素，
  并从四邻域传播候选。
- CUDA `kCheckerboard` 现在使用与 CPU 一致的 8-pass parity 序列：
  `0,1,1,0,0,1,1,0`。

验证：

```text
CUDA:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
31/31 passed

CPU:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
31/31 passed
```

## 2026-07-03 CUDA 几何候选过滤

本轮继续加强 CUDA PatchMatch 的几何一致性：

- 新增 GPU 端 `PassesGeometricCandidateFilter`。
- 候选 depth 在进入完整代价聚合前，会先执行 source depth forward/backward
  consistency 检查。
- 过滤逻辑复用已有几何一致性定义：
  - reference hypothesis 投影到 source view；
  - 采样 source depth map；
  - 将 source depth 回投影到 reference camera；
  - 比较回投影 reference depth 与当前候选 depth 的相对误差。
- propagation 候选和 perturbation 候选都会先经过该几何过滤。
- CUDA summary 沿用已有 geometric candidate accept/reject 字段记录诊断；
  当前这些字段统计的是 propagation + perturbation 两类候选。
- 新增 CUDA 回归测试：
  `PatchMatchBackendTest.CudaBackendRejectsGeometricallyInconsistentCandidates`。

当前边界：

- 该过滤目前是深度相对误差 gate，还没有显式建模遮挡排序和 depth conflict。
- 字段名仍带有 perturbation 历史命名，后续可单独重命名或增加通用字段。
- 这次硬候选过滤先接入 CUDA 路径，CPU 路径暂未同步改成完全一致的 pre-cost
  gate。

验证：

```text
CUDA:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
32/32 passed

CPU:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
32/32 passed
```

## 2026-07-03 CPU/CUDA 几何 pre-cost gate 同步与遮挡/冲突过滤

本轮继续完善 PatchMatch 几何一致性：

- CPU 与 CUDA 现在使用一致的候选级 pre-cost gate。
- CPU 的 source-depth consistency 改为与 CUDA 一致的 forward/backward
  reference-depth 检查，不再只比较 source camera depth。
- gate 对每个 source 做三类判断：
  - `consistent`：回投影 reference depth 与候选 depth 的相对误差在阈值内；
  - `occluded`：source depth 回投影后明显比候选更近，说明候选被遮挡；
  - `conflicting`：source depth 回投影后明显比候选更远，说明存在深度冲突。
- propagation、checkerboard propagation、perturbation 三类 CPU 候选路径都在
  cost aggregation 前执行该 gate。
- CUDA propagation 与 perturbation 使用同样规则。
- 默认拒绝策略：
  - 一致 source 数少于 `min_num_valid_sources` 时拒绝；
  - 存在明确 occluded source 时拒绝；
  - conflicting source 数量不小于 consistent source 数量时拒绝。
- `PatchMatchCostOptions` 新增：
  - `geometric_reject_occluded_candidates`；
  - `geometric_reject_depth_conflicts`。

说明：

- 旧 summary 字段 `num_perturbation_geometric_candidate_*` 为兼容输出暂不重命名；
  当前语义已经扩展为 propagation + perturbation 的总几何候选 gate 计数。
- 这仍不是完整 COLMAP visibility graph / 全局遮挡排序，但已能在候选进入 NCC
  和多源 cost 聚合前过滤明显遮挡与竞争深度。

验证：

```text
新增遮挡/冲突测试:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'RejectsGeometricallyOccludedCandidateBeforeCostAggregation|RejectsCandidateWhenDepthConflictsCompeteWithConsistentSources|CudaBackendRejectsOccludedCandidateDespiteOneConsistentSource|CudaBackendRejectsDepthConflictDespiteOneConsistentSource' --output-on-failure
4/4 passed

CUDA/CPU PatchMatch 定向回归:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchPropagationTest|CpuPatchMatchPerturbationTest|CpuPatchMatchStereoTest|CpuPatchMatchPhotometricCostTest' --output-on-failure
38/38 passed

ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchPropagationTest|CpuPatchMatchPerturbationTest|CpuPatchMatchStereoTest|CpuPatchMatchPhotometricCostTest' --output-on-failure
38/38 passed
```

## 2026-07-03 COLMAP-style MVS visibility graph

本轮先实现 MVS 阶段的 COLMAP visibility/consistency graph 对齐，不进入
graph-cut meshing 的 free-space visibility。

新增内容：

- `ConsistencyGraph` 新增 COLMAP sparse layout 转换：
  - 内部仍保持 image_id 语义；
  - 写 COLMAP 文件时通过 `image_ids_by_index` 转成 MVS image index；
  - sparse data 格式为 `[col, row, N, image_idx_1, ...]`，只写有一致源图的像素。
- 新增 COLMAP 兼容 consistency graph IO：
  - `WriteConsistencyGraphColmapBinary`
  - `ReadConsistencyGraphColmapBinary`
  - 文件头为 `width&height&1&`，随后写 little-endian `int` data。
- `StereoFusionResult` 新增 `points_visibility`：
  - 每个 fused point 对应一个 image-index 列表；
  - image index 是当前 MVS 输入图像顺序，不是 COLMAP sparse model 的 image_id；
  - track 中重复 image index 会去重并排序。
- 新增 COLMAP 兼容 points visibility IO：
  - `WritePointsVisibilityColmapBinary`
  - `ReadPointsVisibilityColmapBinary`
  - 格式为 `num_points:uint64`，随后每个点写 `num_visible:uint32` 和
    `image_idx:uint32[]`。
- `recon_mvs` 新增输出：
  - `stereo/consistency_graphs_colmap/<image_name>.bin`
  - `dense/points_visibility.bin`

验证：

```text
ctest --test-dir build-wsl -R 'RoundTripsConsistencyGraphAsColmapBinary|RoundTripsColmapPointsVisibilityBinary|MergesReciprocalConsistentPixelsIntoSingleTrack' --output-on-failure
3/3 passed

ctest --test-dir build-wsl -R 'ReconMvsExeTest.(WritesBinaryDenseMapsWhenRequested|WritesStereoFusionPlyWhenRequested)' --output-on-failure
2/2 passed
```

## 2026-07-03 CUDA PatchMatch profiling baseline

本轮开始按性能优化顺序推进，先建立 CUDA PatchMatch 的 profiling 基线。

新增内容：
- `PatchMatchStereoSummary` 新增 CUDA 阶段耗时字段：
  - `cuda_upload_time_ms`
  - `cuda_initialization_time_ms`
  - `cuda_evaluate_cost_time_ms`
  - `cuda_propagation_time_ms`
  - `cuda_perturbation_time_ms`
  - `cuda_download_time_ms`
  - `cuda_total_time_ms`
- 单尺度 CUDA PatchMatch 会分别统计：
  - host 侧打包、texture/buffer 创建和 H2D 上传；
  - hypothesis 初始化 kernel；
  - 初始 cost evaluation kernel；
  - 每轮 propagation kernel 组；
  - 每轮 perturbation kernel；
  - D2H 下载；
  - CUDA backend 总 wall time。
- CUDA pyramid 会累加各 level 的耗时。
- `stereo/patch_match.tsv` 增加上述 profiling 列，CPU backend 对应值保持 0。

说明：
- 当前 profiling 为基线版本，会引入 CUDA event 同步；后续优化 host-device 往返和 kernel fusion 时，应以这些字段判断瓶颈移动方向。
- 下一步优先处理 buffer/texture 复用，减少每张图和每个 pyramid level 的重复分配、重复上传。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReportsProfilingTimes --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-04 CUDA PatchMatch async/pinned host staging

本轮继续减少 host-device 往返和同步开销，把最终 packed output 与 iteration counters 的 D2H
下载切换到 pinned host staging + `cudaMemcpyAsync`。

新增内容：
- `CudaPatchMatchWorkspace` 现在懒创建 CUDA stream：
  - stream 在 `cudaSetDevice` 后首次使用时创建；
  - 避免外部提前创建 workspace 时绑定到错误 GPU。
- workspace 新增 pinned host staging buffers：
  - `host_iteration_counters`
  - `host_output_pixels`
- `DeviceBuffer` 新增 `CopyToHostAsync`。
- iteration counters：
  - device 端保存 snapshots；
  - D2H 使用 pinned host buffer + `cudaMemcpyAsync`。
- packed output：
  - `PackOutputMapsKernel` 在 workspace stream 上运行；
  - packed output D2H 使用 pinned host buffer + `cudaMemcpyAsync`。
- CUDA kernel、event timing、pack kernel 和 async D2H 统一进入 workspace stream。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_async_counter_readbacks`
  - `cuda_async_output_readbacks`
  - `cuda_pinned_host_staging_bytes`
  - `cuda_stream_kernel_launches`

当前边界：
- 目前仍在每张 reference image 完成后 `cudaStreamSynchronize`，因为 host 需要立即构造 `DenseDepthMap`、
  `NormalMap` 和 `DenseCostMap`。
- 下一步如果要继续减少阻塞，需要引入双缓冲或 workspace pool，让下一张图的 upload/kernel 与上一张图的 D2H
  有机会重叠。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncReadbackStaging --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
19/19 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-03 CUDA PatchMatch shared reference tile

本轮继续推进性能优化顺序中的 shared memory 项，采用低风险版本：

新增内容：
- `EvaluateCostKernel` 使用动态 shared memory 缓存当前 CUDA block 覆盖的 reference image 中心像素。
- patch cost 中如果 reference patch 采样点落在当前 block 的线性像素范围内，则直接读取 shared tile；否则回退到原有 texture 采样。
- source image、source depth、pose texture、几何一致性和 propagation/perturbation 逻辑暂不改变。
- `PatchMatchStereoSummary` 新增：
  - `cuda_shared_reference_tile_pixels`
- `stereo/patch_match.tsv` 新增同名列，用于确认 CUDA 初始化 cost 阶段是否启用了 shared reference tile。

说明：
- 当前 shared tile 只覆盖 initial cost evaluation，不覆盖 propagation/perturbation 的候选重评分。
- 这是保守实现，目标是先减少 reference texture 的重复读取；更大收益需要后续把 patch window 的 halo 区域也纳入 tile，或改成二维 block 结构。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReportsSharedReferenceTileUsage --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
15/15 passed
```

## 2026-07-04 CUDA PatchMatch workspace/cache 复用

本轮继续推进性能优化中的 buffer/texture 复用。

新增内容：
- `CudaPatchMatchWorkspace` opaque handle：
  - `MakeCudaPatchMatchWorkspace`
  - `DestroyCudaPatchMatchWorkspace`
  - `CudaPatchMatchRequest::workspace`
- CUDA backend 支持从外部传入 workspace；未传入时自动创建临时 workspace，保持旧接口行为。
- workspace 持有并复用：
  - reference image texture；
  - source images layered texture；
  - pose texture；
  - source depth maps texture；
  - selection probability texture；
  - source image info buffer；
  - depth / normal / cost map buffer；
  - device counters buffer。
- `DeviceBuffer` 改为 capacity-based resize，尺寸不超过既有容量时不重新 `cudaMalloc`。
- `FloatLayeredTexture` 在 width/height/depth/filter 一致时复用既有 CUDA array 和 texture object，只重新上传数据。
- CUDA pyramid 以及 `recon_mvs` 多图 PatchMatch 批处理会复用同一个 workspace。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_workspace_texture_allocations`
  - `cuda_workspace_texture_reuses`
  - `cuda_workspace_buffer_allocations`
  - `cuda_workspace_buffer_reuses`

调度说明：
- `recon_mvs` 在 CUDA/auto backend 下使用共享 CUDA workspace。
- 为避免多个 CPU 线程同时写同一个 CUDA workspace，当前 CUDA/auto PatchMatch 批处理采用顺序调度；CPU backend 仍保留原多线程调度。
- 这是保守实现，先保证生命周期拉长和可诊断，后续再做 stream/pinned memory/多 workspace 并发。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReusesWorkspaceResources --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
16/16 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch default checkerboard propagation

本轮继续追踪上一轮 normal materialization 优化后 building-18 曾出现的
`fused_points 156 -> 155` watch item。

诊断结论：
- 少掉的点定位为 `image_id=14` / `P1180179.JPG` 的像素 `(row=5, col=52)`。
- ms47 中该像素 depth=7.79879427、cost=0.0737642869，并在 consistency graph 中有
  `5 52 1 15`。
- ms49 中该像素最终 depth=0，consistency graph 少掉同一条记录。
- image 14 的 photometric 统计一致：
  - `valid_cost_pixels_after_filter=1343`
  - `photometric_filtered_pixels=1729`
- 差异只出现在 geometric consistency：`9 -> 8`。
- 同一当前代码重复跑默认 `forward_backward` 得到 `fused_points=156`，说明 ms49 的少 1
  点不是固定退化，而是旧 CUDA 并行传播对边缘像素的时序/数值敏感。

变更：
- `recon_mvs` 默认 propagation schedule 从 `forward_backward` 改为 `checkerboard`。
- 底层 `PatchMatchStereoOptions` 默认 propagation schedule 同步改为 `kCheckerboard`，避免库 API
  和 CLI 默认行为不一致。
- 命令行仍保留显式选择：
  - `--patch_match_propagation_schedule forward_backward`
  - `--patch_match_propagation_schedule four_direction`
  - `--patch_match_propagation_schedule checkerboard`
- help 文本同步为 `default checkerboard`。

building-18 CUDA profiling：
```text
ms50 current default before changing CLI default:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms50
  fused_points=156
  geometric_consistent_pixels=156

ms51 explicit checkerboard:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms51-checkerboard
  fused_points=216
  geometric_consistent_pixels=216

ms52 default checkerboard:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms52-default-checkerboard
  fused_points=216
  geometric_consistent_pixels=216
```

当前说明：
- checkerboard/red-black 更接近 COLMAP CUDA PatchMatch 的传播组织方式。
- 在 building-18 64px 小测试上，新默认显著提高几何一致像素和 fused points。
- 旧 `forward_backward` 仍可显式选择，用于对比或调试。

验证：
```text
./build-wsl-cuda-mvs-tests/tests/colite_core_tests --gtest_filter='ReconMvsExeTest.PrintsHelp'
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
23/23 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

./build-wsl-cuda-mvs-tests/tests/colite_core_tests --gtest_filter='PatchMatchBackendTest.DefaultsToCheckerboardPropagation'
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

本轮未复跑：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
reason: WSL escalation request rejected by approval system.
```

## 2026-07-08 CUDA PatchMatch materialization breakdown

This milestone targets the post-D2H result materialization path. After explicit
CUDA warmup, the remaining per-image wrapper overhead was dominated by turning
the packed CUDA output into COLITE dense maps and related CPU-side post steps.

Implemented:

- Removed the extra `PinnedHostBuffer<DeviceOutputPixel>::ToVector()` copy for
  packed output materialization. Dense maps now read directly from the pinned
  output staging buffer after stream synchronization.
- Added materialization sub-timers to both `stereo/patch_match_initial.tsv` and
  `stereo/patch_match.tsv`:
  - `cuda_output_to_vector_time_ms`
  - `cuda_dense_map_fill_time_ms`
  - `cuda_normal_refinement_time_ms`
  - `cuda_valid_cost_count_time_ms`

building-18 CUDA check, `max_image_size=64`, 18 images, `--num_threads 2`:

```text
after removing output ToVector copy:
output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms46

initial cuda_result_materialization_time_ms=86.848577 ms
initial cuda_output_to_vector_time_ms=0.000408 ms
initial cuda_dense_map_fill_time_ms=56.471037 ms
initial cuda_normal_refinement_time_ms=29.741048 ms
initial cuda_valid_cost_count_time_ms=0.554789 ms

final cuda_result_materialization_time_ms=84.490515 ms
final cuda_output_to_vector_time_ms=0.000396 ms
final cuda_dense_map_fill_time_ms=56.587363 ms
final cuda_normal_refinement_time_ms=27.263273 ms
final cuda_valid_cost_count_time_ms=0.559550 ms
fused_points=156
```

Interpretation:

- The removed vector copy is no longer measurable, but it was not the dominant
  cost at this image size.
- The remaining materialization cost is mainly dense map fill plus CPU normal
  refinement.
- Next optimization target: make dense map materialization cheaper by adding
  bulk/mutable dense-map storage access, or move normal refinement to CUDA /
  make it optional in the CUDA postprocess path when a later refinement stage
  will recompute normals.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch explicit warmup

This milestone separates one-time CUDA runtime/device initialization from
per-image PatchMatch timing. The previous device-count cache removed repeated
runtime queries, but the first CUDA runtime initialization still appeared inside
the first image's validation timing.

Implemented:

- Added `WarmupCudaPatchMatch(cuda_gpu_index, unavailable_reason)`.
- CUDA builds use cached device-count validation, optional `cudaSetDevice`, and
  `cudaFree(nullptr)` to trigger runtime/context initialization.
- Stub/non-CUDA builds return `0.0` with the existing unavailable reason.
- `recon_mvs` runs warmup before the initial PatchMatch batch when the requested
  backend is `cuda` or `auto`.
- `summary.txt` now writes `cuda_warmup_time_ms`.
- Forced CUDA (`--patch_match_backend cuda`) reports warmup failure as an error;
  auto mode keeps the existing CPU fallback behavior.

building-18 CUDA check, `max_image_size=64`, 18 images, `--num_threads 2`:

```text
before warmup: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms43
after warmup : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms44

ms43 initial_wall=447.090526 ms
ms43 initial cuda_request_validation_time_ms=403.682959 ms
ms43 initial wrapper_overhead_sum=687.553451 ms

ms44 cuda_warmup_time_ms=345.080609 ms
ms44 initial_wall=90.707164 ms
ms44 initial cuda_request_validation_time_ms=0.017520 ms
ms44 initial wrapper_overhead_sum=1.201502 ms
ms44 fused_points=156
```

Interpretation:

- The large first-use CUDA runtime initialization is now explicit
  `cuda_warmup_time_ms`, not hidden in per-image PatchMatch validation.
- Initial per-image wrapper overhead is now near zero at this test scale.
- Remaining visible per-image costs are CUDA upload/compute/download plus CPU
  result materialization.
- Next target: reduce or defer result materialization cost, which still sums to
  roughly 80-90 ms for 18 images at `max_image_size=64`.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch initial TSV and device-count cache

This milestone separates initial-stage CUDA diagnostics from the final
post-refinement `patch_match.tsv`. The final TSV is still useful for final map
quality, but it cannot explain initial PatchMatch wall time once geometric
refinement overwrites each image result.

Implemented:

- Added `stereo/patch_match_initial.tsv`, written immediately after the initial
  PatchMatch batch and before geometric refinement.
- Added CUDA wrapper timing fields to both initial and final TSV output:
  - `cuda_wrapper_total_time_ms`
  - `cuda_wrapper_overhead_time_ms` in the initial TSV
  - `cuda_request_validation_time_ms`
  - `cuda_pyramid_resize_time_ms`
  - `cuda_host_pack_time_ms`
  - `cuda_single_scale_dispatch_time_ms`
  - `cuda_result_materialization_time_ms`
- Added process-local CUDA device-count caching with diagnostics:
  - `cuda_device_count_queries`
  - `cuda_device_count_cache_hits`

building-18 CUDA check, `max_image_size=64`, 18 images, `--num_threads 2`:

```text
before cache: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms42
after cache : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms43

ms42 initial_wall=453.418734 ms
ms42 initial cuda_request_validation_time_ms=441.954736 ms

ms43 initial_wall=447.090526 ms
ms43 initial cuda_device_count_queries=1
ms43 initial cuda_device_count_cache_hits=17
ms43 initial cuda_request_validation_time_ms=403.682959 ms

ms43 final/refinement cuda_device_count_queries=0
ms43 final/refinement cuda_device_count_cache_hits=18
ms43 fused_points=155
```

Interpretation:

- The per-image device-count query has been removed; the process now performs
  one runtime query and then uses cache hits.
- Most remaining validation cost is the first CUDA runtime/device initialization
  in the process, not repeated per-image validation.
- Next optimization target: explicit CUDA warmup/pre-initialization before the
  initial PatchMatch batch, with a separate `cuda_warmup_time_ms` summary field,
  so the one-time runtime initialization is visible and no longer contaminates
  per-image PatchMatch scheduling diagnostics.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch initial-stage timing breakdown

This milestone adds finer-grained diagnostics for the initial PatchMatch batch.
The previous overlap summary showed that initial wall time was still much larger
than the summed CUDA time. The new fields split each initial job into controller
source setup, the full `RunPatchMatchStereo...` call, wrapper overhead outside
the CUDA backend summary, and postprocess/stat filtering.

New `summary.txt` fields:

- `patch_match_initial_source_setup_sum_time_ms`
- `patch_match_initial_stereo_run_sum_time_ms`
- `patch_match_initial_stereo_wrapper_overhead_sum_time_ms`
- `patch_match_initial_postprocess_sum_time_ms`

building-18 CUDA check, `max_image_size=64`, 18 images, `--num_threads 2`:

```text
output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms40

initial_wall=427.307613 ms
initial_cuda_sum=172.494390 ms
source_setup_sum=0.174237 ms
stereo_run_sum=849.639651 ms
stereo_wrapper_overhead_sum=677.145261 ms
postprocess_sum=0.725358 ms
refinement_wall=61.334458 ms
refinement_overlap_ratio=0.479375
fused_points=155
```

Interpretation:

- Controller-level source setup and postprocess are not the current bottleneck.
- Most remaining initial-stage cost is inside the full PatchMatch stereo call
  but outside the CUDA backend `cuda_total_time_ms` summary.
- Next profiling step should split initial/refinement CUDA TSV output or add
  CUDA wrapper sub-timers around request validation, pyramid/resize preparation,
  host packing, texture upload preparation, and result materialization.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-04 CUDA PatchMatch counter readback / sync 收敛

本轮继续推进“减少 host-device 往返”。

新增内容：
- CUDA iteration counters 不再每轮 `cudaDeviceSynchronize + D2H CopyScalarToHost`。
- 新增 device 端 counter buffer：
  - 每轮迭代结束后通过 `StoreIterationCountersKernel` 把当前 counters 保存到 `iteration_counters` buffer；
  - 全部迭代结束后只执行一次 `CopyToHost`，一次性生成 per-iteration diagnostics。
- counter reset 从 host-to-device scalar copy 改为 `ResetCountersKernel`，避免每轮上传一个 `DeviceCounters{}`。
- propagation / perturbation 的 CUDA event timing 改为循环内只 record stop event，统一在循环后读取 elapsed time，减少 profiling 导致的逐轮同步。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_iteration_counter_snapshots`
  - `cuda_iteration_counter_readbacks`

当前效果：
- 对 `num_iterations=N` 的 CUDA PatchMatch：
  - device counter snapshots = `N`
  - host counter readbacks = `1`（当 `N > 0`）
  - 不再有每轮显式 `cudaDeviceSynchronize()`。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendBatchesIterationCounterReadbacks --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
17/17 passed
```

## 2026-07-04 CUDA PatchMatch packed output map readback

本轮继续减少最终输出阶段的 host-device 往返。

新增内容：
- 新增 `DeviceOutputPixel`，将每个像素的：
  - depth
  - normal x/y/z
  - cost
  打包为一个 device buffer 元素。
- 新增 `PackOutputMapsKernel`：
  - 在 GPU 端把独立的 depth / normal / cost buffer 打包到 `output_pixels`；
  - host 端只执行一次 `CopyToHost`，替代原来的 depth、normal、cost 三次 D2H。
- `CudaPatchMatchWorkspace` 新增可复用 `output_pixels` buffer。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_packed_output_pixels`
  - `cuda_output_map_readbacks`

当前效果：
- CUDA PatchMatch 最终 map 输出 readback 从 3 次降为 1 次。
- `cuda_output_map_readbacks` 在正常非空图像上应为 `1`。
- `cuda_packed_output_pixels` 应等于当前 reference image 的像素数；多尺度 pyramid summary 会累加各 level 的 packed 像素数。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendPacksOutputMapsForSingleReadback --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
18/18 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-04 CUDA PatchMatch workspace pool / 双缓冲

本轮继续减少跨图像 PatchMatch 的等待时间，目标是让上一张图的 D2H
回传与下一张图的 upload/kernel 有机会重叠。

新增内容：
- `recon_mvs` 在 CUDA/auto backend 下创建 `CudaPatchMatchWorkspace` 池：
  - 当前池大小为 `min(num_threads, num_images, 2)`；
  - 每个 workspace 拥有独立 CUDA stream、texture/cache、device buffer 和 pinned host staging；
  - 初始 PatchMatch 批处理按 slot 轮转调度，只有对应 slot 的前一个任务结束后才复用该 workspace。
- 初始 PatchMatch 在 pool 非空时使用最多两个并发 future：
  - slot 0 可执行 image N 的 upload/kernel/D2H；
  - slot 1 可同时执行 image N+1 的 upload/kernel/D2H；
  - 因为每个 slot 有独立 stream/pinned buffer，上一张图异步 D2H 与下一张图 kernel/upload 具备重叠条件。
- 几何 refinement 当前仍保持顺序执行，但会复用 workspace pool 并按 round-robin 记录 slot。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_workspace_pool_size`
  - `cuda_workspace_pool_slot`

调度说明：
- CPU backend 不使用 CUDA workspace pool，仍走原多线程 CPU 调度。
- CUDA/auto backend 下当前 pool 上限保守限制为 2，先实现双缓冲；后续可在确认显存和多 stream 收益后放宽。
- 该实现主要优化跨 reference image 的流水线重叠，不改变单张图内部 kernel 算法。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
19/19 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch output normal materialization

本轮继续压缩 CUDA PatchMatch 的 host-side materialization 开销，重点处理上一轮
`cuda_dense_map_fill_time_ms` 中仍保留的 normal 写回归一化。

新增内容：
- CUDA initial map 上传时会先归一化 normal，避免外部非单位 initial normal 破坏设备端假设不变量。
- `PackOutputMapsKernel` 在设备端归一化输出 normal，host materialization 只做 finite/非零检查和直接拷贝。
- `PatchMatchStereoSummary` 与 `patch_match*.tsv` 新增诊断：
  - `cuda_host_initial_normal_upload_normalizations`
  - `cuda_host_output_normalizations`
- 新增 CUDA 单元测试，验证非单位 initial normal 上传后输出仍是单位 normal，并且 output host 归一化次数为 0。

building-18 CUDA profiling：
```text
baseline: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms47
current : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms49

ms47 final cuda_result_materialization_time_ms sum=82.700163 ms
ms49 final cuda_result_materialization_time_ms sum=61.074301 ms

ms47 final cuda_dense_map_fill_time_ms sum=55.202973 ms
ms49 final cuda_dense_map_fill_time_ms sum=34.254693 ms

ms49 final cuda_host_output_normalizations sum=0

ms47 fused_points=156
ms49 fused_points=155
```

当前说明：
- 性能收益明显，host dense map fill 约减少 21ms/18 images。
- output normal 归一化已从 host 移到 CUDA pack kernel；这比完全跳过归一化更接近旧语义。
- building-18 小 demo 的 fused points 少 1 个，需要后续结合更大分辨率和多次运行判断是数值扰动还是阈值边缘点。
- 后续若要完全保持旧 CPU/Eigen normal 归一化数值，应考虑在 host 端保留一个可选严格路径，或放宽 fusion/normal 阈值敏感性诊断。

验证：
```text
./build-wsl-cuda-mvs-tests/tests/colite_core_tests --gtest_filter='PatchMatchBackendTest.CudaBackendNormalizesInitialNormalsBeforeUpload'
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
22/22 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed / no work to do

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch dense map bulk materialization

本轮继续压缩 CUDA PatchMatch 的 host-side materialization 开销，针对上一轮 profile 中
`cuda_dense_map_fill_time_ms` 占比较高的问题做保守优化。

新增内容：
- `DenseDepthMap`、`NormalMap`、`DenseCostMap` 新增 `MutableData()`，用于受控批量写入内部连续存储。
- CUDA PatchMatch 下载 packed output 后，直接按线性索引写入 depth/normal/cost 底层数组。
- 保留原有 normal 有效性语义：非 finite 或零向量仍写为零向量；有效 normal 仍保持单位化。
- 新增 DenseMap 单元测试，覆盖三类 map 的 mutable data 写入可被 `Get`、`IsValid`、`ValidPixelCount` 正确观察。

building-18 CUDA profiling：
```text
baseline: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms46
current : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms47

ms46 initial cuda_dense_map_fill_time_ms sum=56.471037 ms
ms47 initial cuda_dense_map_fill_time_ms sum=55.341985 ms

ms46 final   cuda_dense_map_fill_time_ms sum=56.587363 ms
ms47 final   cuda_dense_map_fill_time_ms sum=55.202973 ms

ms46 final   cuda_result_materialization_time_ms sum=84.490515 ms
ms47 final   cuda_result_materialization_time_ms sum=82.700163 ms

ms46 fused_points=156
ms47 fused_points=156
```

当前说明：
- 这次优化主要去掉逐像素 setter 的边界检查和函数调用，收益稳定但不大。
- 当前 materialization 的主要剩余开销转向 normal 写回归一化、normal refinement 和 valid cost 统计。
- 后续如果要继续压缩该阶段，应优先考虑：
  - 证明 CUDA 输出 normal 的单位向量不变量后，减少 host 端重复归一化；
  - 将 normal refinement 进一步迁移到 CUDA；
  - 或者只在最终输出阶段做 normal refinement，避免 initial/refinement 重复计算。

验证：
```text
./build-wsl-cuda-mvs-tests/tests/colite_core_tests --gtest_filter='DenseDepthMapTest.MutableDataUpdatesDepthValues:NormalMapTest.MutableDataUpdatesNormalValues:DenseCostMapTest.MutableDataUpdatesCostValues'
3/3 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch deferred initial photometric stats

This milestone trims CPU-side work around the CUDA PatchMatch scheduling path.
When geometric refinement is enabled, the initial PatchMatch photometric stats
are no longer computed immediately because the refinement pass recomputes the
final `stats_before_filter`, photometric filter result, and
`stats_after_photometric_filter` for every image.

Implemented:

- `PatchMatchWorkspaceResult` records whether initial photometric stats were
  deferred.
- `PatchMatchSchedulingSummary` records
  `patch_match_initial_deferred_photometric_stats_images`.
- `summary.txt` now writes
  `patch_match_initial_deferred_photometric_stats_images`.
- The final TSV/statistics path is unchanged: after geometric refinement, every
  PatchMatch result still receives final photometric stats and filtering.

building-18 CUDA check, `max_image_size=64`, 18 images:

```text
previous thread1: initial_wall=560.480585 ms, initial_cuda_sum=153.011242 ms, fused_points=155
current  thread1: initial_wall=476.695071 ms, initial_cuda_sum=151.284996 ms, deferred=18, fused_points=156

previous thread2: initial_wall=432.277589 ms, initial_cuda_sum=171.995448 ms, refine_overlap_ratio=0.485008, fused_points=156
current  thread2: initial_wall=388.363980 ms, initial_cuda_sum=165.981279 ms, deferred=18, refine_overlap_ratio=0.479275, fused_points=155
```

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-07 CUDA PatchMatch pinned async H2D upload staging

本轮继续推进 host-device 往返优化，把上一轮只覆盖 D2H 的 pinned/async
机制扩展到一部分 H2D 上传。

新增内容：
- `DeviceBuffer` 新增 `CopyFromHostAsync`，支持在 workspace stream 上执行 H2D。
- `CudaPatchMatchWorkspace` 新增可复用 pinned upload staging：
  - `host_depth_upload`
  - `host_normal_upload`
- 当 CUDA PatchMatch 接收 `initial_maps` 时：
  - depth map 先复制到 pinned host buffer；
  - normal map 先复制到 pinned host buffer；
  - depth/normal 再通过 `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)` 上传；
  - 后续 cost / propagation / perturbation kernel 位于同一 stream，会自然等待上传完成。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_async_host_uploads`
  - `cuda_pinned_host_upload_bytes`

当前范围：
- 这轮先覆盖 initial maps 的线性 depth/normal buffer 上传，主要服务 CUDA pyramid
  coarse-to-fine、几何 refinement 和后续重用初始假设的路径。
- reference/source image、pose texture、source depth texture、selection probability
  texture 在后续 2026-07-07 texture upload milestone 中已继续接入 async array copy。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncUploadStagingForInitialMaps --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
20/20 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R PatchMatchBackendTest --output-on-failure
19/19 passed, CUDA-only tests skipped

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-07 CUDA PatchMatch pinned async texture upload staging

本轮继续推进 H2D 上传优化，把 texture-backed 输入也接入 workspace stream。

新增内容：
- `FloatLayeredTexture` 新增 `CreateAsync`：
  - texture array / texture object 生命周期仍由 workspace 管理；
  - host-to-array 上传从 `cudaMemcpy3D` 切换为 `cudaMemcpy3DAsync`；
  - copy 操作进入当前 `CudaPatchMatchWorkspace` 的 stream。
- `CudaPatchMatchWorkspace` 新增每类 texture 独立 pinned staging：
  - `host_reference_texture_upload`
  - `host_source_images_texture_upload`
  - `host_poses_texture_upload`
  - `host_source_depth_maps_texture_upload`
  - `host_selection_probability_texture_upload`
- CUDA PatchMatch 单尺度执行中，以下输入会先写入 pinned staging，再异步上传到 CUDA layered array：
  - reference image texture；
  - source images layered texture；
  - pose texture；
  - source depth maps texture（启用几何一致性且传入 source depth 时）；
  - dense selection probability texture（传入 dense probability maps 时）。
- 每类 texture 使用独立 staging buffer，避免一个 async copy 尚未完成时 host buffer 被下一份 texture 数据覆盖。
- `PatchMatchStereoSummary` 和 `stereo/patch_match.tsv` 新增：
  - `cuda_async_texture_uploads`
  - `cuda_pinned_texture_upload_bytes`

当前效果：
- CUDA PatchMatch 的主要 H2D 输入已经都进入 workspace stream：
  - texture-backed image / pose / source-depth / selection-probability 输入；
  - initial depth/normal linear buffer 输入；
  - D2H output/counter readback。
- 与 workspace pool / 双缓冲结合后，上一张图的 D2H 与下一张图的 texture/linear H2D + kernel 更有机会重叠。

当前限制：
- CUDA array 分配和 texture object 创建仍会发生在 host 侧，尚未进一步池化到更细粒度的 shape cache。
- pinned staging 目前按每个 workspace slot 保留容量，显存/主机 pinned memory 压力需要在真实场景中继续 profiling。
- 后续 2026-07-07 overlap diagnostics milestone 已加入 summary 级 end-to-end
  overlap 估计指标。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncTextureUploadStaging --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R PatchMatchBackendTest --output-on-failure
20/20 passed, CUDA-only tests skipped

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-07 CUDA PatchMatch workspace pool overlap diagnostics

本轮不改变 CUDA kernel 行为，补齐 workspace pool / 双缓冲的端到端可观测性。

新增内容：
- `recon_mvs` 新增 `PatchMatchSchedulingSummary`，记录初始 PatchMatch 批处理的调度效果：
  - `patch_match_initial_wall_time_ms`
  - `patch_match_initial_sum_cuda_time_ms`
  - `patch_match_initial_overlap_saved_time_ms`
  - `patch_match_initial_overlap_ratio`
- 指标含义：
  - wall time：初始 PatchMatch 批处理从开始调度到全部图像完成的实际墙钟时间；
  - sum CUDA time：所有实际使用 CUDA backend 的 reference image 的 `cuda_total_time_ms` 之和；
  - overlap saved time：`max(0, sum_cuda_time - wall_time)`；
  - overlap ratio：`overlap_saved_time / sum_cuda_time`，无 CUDA 时间时为 0。
- 指标写入 `summary.txt`，用于观察 workspace pool 下跨图 upload/kernel/D2H 是否真的产生重叠收益。

当前说明：
- 这是 conservative estimate：单图 `cuda_total_time_ms` 和批处理 wall time 不是完全同一计时域，但足够用于判断是否存在明显跨图重叠。
- 后续 2026-07-08 refinement pool milestone 已将几何 refinement 接入 workspace pool
  并加入独立 overlap 指标。
- CPU / CUDA unavailable 路径会输出 0 或接近 0 的 CUDA sum/overlap 指标，不影响流程。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed
```

## 2026-07-08 CUDA PatchMatch geometric refinement workspace pool

本轮继续推进 workspace pool 优化，把 geometric refinement 从顺序执行改为可并发调度。

新增内容：
- 新增 `PatchMatchRefinementJob` / `PatchMatchRefinementResult`，将每张图 refinement
  所需的 reference/source view、image、source depth map 快照成独立任务。
- geometric refinement 现在使用与初始 PatchMatch 类似的调度：
  - CUDA/auto backend 下使用 `CudaPatchMatchWorkspace` pool；
  - pool slot 内复用 stream、texture/buffer、pinned upload/download staging；
  - CPU backend 下继续按 `--num_threads` 并发。
- refinement 使用初始 depth map 快照作为 source depth 输入，避免旧顺序实现中后处理图像读到前面已 refined depth map 的顺序依赖。
- `summary.txt` 新增 refinement 阶段 overlap 诊断：
  - `patch_match_refinement_wall_time_ms`
  - `patch_match_refinement_sum_cuda_time_ms`
  - `patch_match_refinement_overlap_saved_time_ms`
  - `patch_match_refinement_overlap_ratio`

当前说明：
- refinement overlap 指标和 initial overlap 指标使用同一套 conservative estimate。
- refinement 仍需要等初始 PatchMatch 全部完成后再开始；还不是 initial/refinement 跨阶段流水线。
- source depth map 快照会增加一些内存占用，但消除了顺序依赖，便于后续进一步并发。

验证：
```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
1/1 passed
```
## 2026-07-13 CUDA PatchMatch device-side adaptive perturbation radius

本轮继续缩小 CUDA PatchMatch 与 CPU/COLMAP 化调度的差距：CUDA 端不再由 host
固定执行 `perturbation_shrink`，而是在 GPU 上根据每轮 `DeviceCounters` 计算
`update_ratio`，并按 `adaptive_perturbation_min_update_ratio` 决定是否执行双倍
shrink。

实现内容：
- 新增 device-side `perturbation_radius_state`，保存下一轮 depth/normal 扰动半径。
- 每轮 perturbation 前通过 kernel 记录实际使用的 `depth_radii` 和
  `normal_radii_degrees`。
- 每轮 perturbation 后通过 kernel 根据同轮 counters 更新下一轮半径。
- counters 与每轮半径仍在迭代结束后一次性 async D2H 回传，避免恢复逐轮
  host-device 同步。

意义：
- CUDA summary 中的 `update_ratios`、`depth_radii`、`normal_radii_degrees`
  与 CPU 路径语义对齐。
- 保留前一阶段的 batched counter readback / pinned async readback 优化。

验证：
```text
cmake --build build-wsl-cuda-mvs-tests --target colite_core_tests -j2
passed

ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest\.(CudaBackendAppliesAdaptivePerturbationShrinkOnDevice|CudaBackendBatchesIterationCounterReadbacks|CudaBackendUsesPinnedAsyncReadbackStaging)' --output-on-failure
3/3 passed
```
## COLMAP 4.0 核心 A/B

2026-07-16 在同一去畸变工作区、RTX 4070 SUPER 和 CUDA 13.2 上完成了
image undistorter、PatchMatch、geometric consistency/filter 与 stereo fusion 对比。
结论、完整参数和后续 P0/P1 顺序见
[mvs_colmap4_core_validation_20260716.md](mvs_colmap4_core_validation_20260716.md)。

P0/P1 优化复测已追加到上述报告：几何有效覆盖率由 1.08% 提升到 54.15%，Fusion
由 780 点提升到 32,886 点；去畸变输出相机与 COLMAP 完全一致，PSNR 提升到
36.76dB。纯光度阶段已接近 COLMAP 耗时，但 P95 长尾仍是下一优先项。
