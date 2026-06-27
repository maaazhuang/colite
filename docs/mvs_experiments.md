# COLITE MVS 实验矩阵

更新日期：2026-06-25

本文记录 MVS 阶段的可复现实验指标。当前主数据集为 `data/building-18`，输入稀疏模型为 `build-wsl/compare-building-18/controlled-colite-4000-pinhole`。

## 指标说明

- `case`：实验输出目录名，位于 `build-wsl/mvs-building-18/` 下。
- `max_num_source_images`：每张 target image 最多保留的 source view 数。
- `patch_match_iterations`：CPU PatchMatch 迭代轮数。
- `patch_match_min_num_valid_sources`：PatchMatch cost 至少需要多少个有效 source view。
- `patch_match_max_num_aggregated_costs`：聚合代价时保留的 best source cost 数，`0` 表示使用全部。
- `photometric_filtered_pixels`：photometric cost 阈值过滤掉的像素数。
- `geometric_consistent_pixels`：几何一致性图中保留的一致像素数。
- `fused_points`：stereo fusion 输出的点数。
- `fused_observations`：fusion track 中的总观测数。
- `single_observation_tracks`：只有一个观测的 fusion track 数。
- `multi_observation_tracks`：至少两个观测的 fusion track 数。
- `runtime_seconds`：使用 `/usr/bin/time` 记录的 WSL wall-clock 时间，仅用于同机同参数趋势比较。

## building-18 当前回归矩阵

| case | max_num_source_images | patch_match_iterations | patch_match_min_num_valid_sources | patch_match_max_num_aggregated_costs | photometric_filtered_pixels | geometric_consistent_pixels | fused_points | fused_observations | single_observation_tracks | multi_observation_tracks | runtime_seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `colite-ms8-smoke-48` | 8 | 1 | 1 | 3 | 4334 | 6983 | 5969 | 6983 | 5201 | 768 | 69.40 |

### 本次 smoke 参数

```bash
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl/mvs-building-18/colite-ms8-smoke-48 \
  --run_patch_match \
  --max_image_size 48 \
  --patch_match_iterations 1 \
  --patch_match_window_radius 1 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --fusion_max_normal_error 10 \
  --fusion_max_view_angle 180 \
  --filter_geometric_min_num_consistent_sources 1
```

### Track 观测数直方图

来自 `dense/fusion_track_lengths.tsv`：

| num_observations | num_tracks |
| ---: | ---: |
| 1 | 5201 |
| 2 | 613 |
| 3 | 115 |
| 4 | 26 |
| 5 | 4 |
| 6 | 3 |
| 7 | 2 |
| 8 | 1 |
| 9 | 1 |
| 10 | 2 |
| 18 | 1 |

由此得到：

```text
single_observation_tracks = 5201
multi_observation_tracks = 5969 - 5201 = 768
```

## 历史对照

这些行来自已有输出目录，部分旧实验没有记录 runtime，因此不进入“当前回归矩阵”的主表。

| case | max_image_size | photometric_filtered_pixels | geometric_consistent_pixels | fused_points | fused_observations | single_observation_tracks | multi_observation_tracks | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `colite-source-scoring-final-48` | 48 | 4334 | 6983 | 5969 | 6983 | 5201 | 768 | Milestone 1 后的 48 尺寸质量基线 |
| `colite-clean-mesh-32-ms7` | 32 | 4182 | 2889 | 2440 | 2889 | 2083 | 357 | Milestone 7 dense clean/mesh smoke |

## Milestone 8 验收记录

计划中的 48 尺寸 smoke 期望：

```text
fused_points >= 5400
fused_observations >= 6300
```

本次记录：

```text
fused_points 5969
fused_observations 6983
```

结果满足当前 smoke 阈值。后续质量型里程碑不强制要求点数增加，应继续关注：

- multi-observation track ratio
- fusion rejected 原因分布
- pair-level geometry/depth 失败原因
- dense clean 后的保留率
- mesh 面片数量、长边比例和法线一致性

## 后续维护约定

每次完成 MVS 相关里程碑时，至少追加一行实验记录，并保留：

- 完整 CLI 参数
- `summary.txt`
- `dense/fusion_track_lengths.tsv`
- 如果涉及 dense clean/mesh，记录 `dense/fused_clean_stats.txt` 和 `dense/mesh.ply` 的 vertex/face 头信息
- `ctest` 结果和 smoke runtime

## Milestone 9 Diagnostics

Milestone 9 adds automatic quality summaries beside the existing raw TSV/PLY outputs:

```text
dense/fusion_summary.txt
dense/mesh_stats.txt
```

`dense/fusion_summary.txt` records track-count, observation-count, multi-view ratio, and max normal/view angle metrics.

`dense/mesh_stats.txt` records vertex count, triangle count, edge-length statistics, and triangle-area statistics.

These files remove manual post-processing from the next PatchMatch/Fusion optimization pass.
### Milestone 9 building-18 smoke

Case: `colite-ms9-diagnostics-32`

```text
photometric_filtered_pixels 2491
geometric_consistent_pixels 2437
fused_points 2152
fused_observations 2437
num_single_observation_tracks 1926
num_multi_observation_tracks 226
multi_observation_track_ratio 0.1050185873605948
mesh_vertices 2152
mesh_triangles 47045
mesh_mean_edge_length 0.30100532545381642
mesh_max_edge_length 0.49998993362140515
mesh_mean_triangle_area 0.031532898460199174
```

This run used `max_image_size=32`, one PatchMatch iteration, `fusion_max_normal_error=10`, dense cloud filtering with minimum one observation, and the local-neighbor mesh prototype.
## Milestone 10 Fusion Track Ratio Optimization

This pass targeted multi-observation track ratio without increasing `fusion_min_num_observations`.

Root cause evidence from `colite-ms10-fusion-radius0-32`:

```text
num_candidates 3435
num_accepted 285
num_rejected_invalid 1561
num_rejected_mutual_consistency 647
num_rejected_depth 353
num_single_observation_tracks 1926
num_multi_observation_tracks 226
multi_observation_track_ratio 0.1050185873605948
```

Implemented `--fusion_source_pixel_search_radius`, default `1`, so fusion can search nearby source pixels while preserving depth, mutual consistency, normal, and view angle checks.

Optimized case `colite-ms10-fusion-radius1-32-fixed`:

```text
num_candidates 3435
num_accepted 438
num_rejected_invalid 471
num_rejected_mutual_consistency 377
num_rejected_depth 1052
num_rejected_normal 427
num_single_observation_tracks 1771
num_multi_observation_tracks 228
multi_observation_track_ratio 0.11405702851425713
fused_points 1999
fused_observations 2437
mesh_triangles 38593
```

Result: fused observations stayed at `2437`, while fused tracks dropped from `2152` to `1999`. This indicates more observations are being merged into shared tracks rather than discarded. Radius `2` gave almost no extra benefit over radius `1`, so radius `1` remains the conservative default.

## Milestone 11 Fusion Depth/Normal Diagnostics

This pass continued optimizing `multi_observation_track_ratio` on top of the
Milestone 10 nearby-source-pixel fusion change.

Implemented source-depth validation alignment:

- Fusion now first validates source depth at the real projected source
  coordinate with bilinear depth sampling.
- If bilinear sampling is unavailable at image borders or incomplete depth
  neighborhoods, fusion falls back to the discrete candidate pixel depth.
- Regression test: `CpuStereoFusionTest.UsesBilinearSourceDepthForFusionValidation`.

Verified before the later normal-propagation edit:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 135
```

building-18 baseline after bilinear depth validation:

```text
case colite-ms11-fusion-bilinear-depth-32
num_tracks 1999
num_single_observation_tracks 1771
num_multi_observation_tracks 228
num_observations 2437
multi_observation_track_ratio 0.11405702851425713
num_accepted 438
num_rejected_depth 1052
num_rejected_mutual_consistency 377
num_rejected_invalid 471
num_rejected_normal 427
```

Result: bilinear fusion-depth validation is a correctness alignment with the
geometric consistency graph, but it did not change building-18 metrics at
`max_image_size=32`.

Threshold sweep for diagnosis:

```text
case colite-ms11-sweep-depth003-32
fusion_max_relative_depth_error 0.03
fusion_max_normal_error 10
num_tracks 1979
num_single_observation_tracks 1747
num_multi_observation_tracks 232
multi_observation_track_ratio 0.11723092470944922
num_accepted 458
num_rejected_depth 602
num_rejected_normal 509

case colite-ms11-sweep-normal30-32
fusion_max_relative_depth_error 0.01
fusion_max_normal_error 30
num_tracks 1846
num_single_observation_tracks 1613
num_multi_observation_tracks 233
multi_observation_track_ratio 0.12621885157096424
num_accepted 591
num_rejected_depth 1052
num_rejected_normal 0

case colite-ms11-sweep-depth003-normal30-32
fusion_max_relative_depth_error 0.03
fusion_max_normal_error 30
num_tracks 1802
num_single_observation_tracks 1578
num_multi_observation_tracks 224
multi_observation_track_ratio 0.12430632630410655
num_accepted 635
num_rejected_depth 602
num_rejected_normal 0
```

Diagnosis:

- Loosening depth alone gives a small improvement.
- Loosening normal gives the larger improvement, but simply relaxing the global
  threshold is too blunt.
- A conservative follow-up code change was started: while expanding a fusion
  track, normal filtering now compares the source normal against both the seed
  normal and the current traversal reference normal, taking the smaller angle.
  This is intended to preserve the strict `10 deg` threshold while allowing
  locally continuous curved surfaces to propagate through the track.
- Regression test added: `CpuStereoFusionTest.PropagatesTrackAcrossLocallyConsistentNormals`.

Validation after the normal-propagation edit:

```text
cmake --build build-wsl --target colite_core_tests -j2
ctest --test-dir build-wsl -R CpuStereoFusionTest --output-on-failure
ctest --test-dir build-wsl --output-on-failure
```

Result:

```text
CpuStereoFusionTest: 8/8 passed
Full ctest: 136/136 passed
```

building-18 smoke after local-normal propagation:

```text
case colite-ms11-normal-propagation-32
num_tracks 1999
num_single_observation_tracks 1771
num_multi_observation_tracks 228
num_observations 2437
multi_observation_track_ratio 0.11405702851425713
num_accepted 438
num_rejected_depth 1052
num_rejected_mutual_consistency 377
num_rejected_invalid 471
num_rejected_normal 427
```

Result: the local-normal propagation rule is covered and safe, but it does not
move the building-18 32px metric. The current normal rejects are therefore
mostly direct one-hop source normal disagreements, not long traversal drift.
The next optimization should target direct source normal/depth quality: normal
map refinement/smoothing, source normal sampling, or PatchMatch normal
initialization/refinement rather than traversal logic.

## Milestone 12 Depth-Based Normal Refinement

Implemented an optional PatchMatch normal refinement pass:

- API: `RefineNormalMapFromDepth`.
- CLI: `--refine_normals_from_depth`.
- CLI threshold: `--normal_refinement_max_relative_depth_difference <r>`.
- Summary outputs:
  - `summary.txt`: `normal_refined_pixels`
  - `stereo/patch_match.tsv`: `normal_refinement_pixels`,
    `normal_refined_pixels`, `normal_refinement_skipped_pixels`

The pass estimates a camera-space normal from left/right/up/down depth
neighbors. It only updates a pixel when the four-neighborhood is valid and each
neighbor depth is within the configured relative difference from the center
depth. The refined normal is flipped to the same hemisphere as the existing
normal before writing.

Validation:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 140
```

building-18 32px comparison, baseline from `colite-ms11-normal-propagation-32`:

```text
baseline
num_tracks 1999
num_single_observation_tracks 1771
num_multi_observation_tracks 228
num_observations 2437
mean_observations 1.2191095547773887
multi_observation_track_ratio 0.11405702851425713
num_accepted 438
num_rejected_depth 1052
num_rejected_mutual_consistency 377
num_rejected_normal 427
```

Strict normal refinement, `max_relative_depth_difference=0.005`:

```text
case colite-ms12-normal-refine-depth0005-32
normal_refined_pixels 1678
num_tracks 1971
num_single_observation_tracks 1766
num_multi_observation_tracks 205
num_observations 2437
mean_observations 1.2364282090309489
multi_observation_track_ratio 0.10400811770674784
num_accepted 466
num_rejected_depth 946
num_rejected_mutual_consistency 465
num_rejected_normal 382
```

Threshold sweep:

```text
max_relative_depth_difference=0.01
normal_refined_pixels 1747
num_tracks 1973
num_multi_observation_tracks 208
mean_observations 1.2351748606183477
multi_observation_track_ratio 0.10542321338063862
num_accepted 464
num_rejected_normal 381

max_relative_depth_difference=0.05
normal_refined_pixels 2752
num_tracks 2080
num_multi_observation_tracks 171
multi_observation_track_ratio 0.082211538461538461
num_accepted 357
num_rejected_normal 650

max_relative_depth_difference=0.20
normal_refined_pixels 5666
num_tracks 2152
num_multi_observation_tracks 138
multi_observation_track_ratio 0.064126394052044608
num_accepted 250
num_rejected_normal 869
```

Conclusion:

- Depth-derived normals help only when gated very conservatively.
- `0.005` is the current default because it increases accepted fusion
  observations (`438 -> 466`), lowers normal rejects (`427 -> 382`), and
  increases mean observations per fused track (`1.219 -> 1.236`).
- The simple ratio `multi_observation_track_ratio` decreases because accepted
  observations concentrate into fewer, longer tracks. Future diagnostics should
  track both `multi_observation_track_ratio` and `mean_observations`.
- Wider thresholds are harmful on the current 32px building-18 smoke because
  PatchMatch depth is too noisy for unconstrained local normal estimation.

## Milestone 13 Plane-Aware PatchMatch Propagation

Changed PatchMatch spatial propagation from "copy neighbor depth" to
"propagate neighbor plane hypothesis." For a neighbor hypothesis
`(neighbor_depth, neighbor_normal)`, the neighbor pixel is first unprojected to
a 3D point on that plane. The propagated candidate depth at the current pixel is
then computed by intersecting the current camera ray with the neighbor plane.
This matches the plane-hypothesis interpretation used by COLMAP-style
PatchMatch more closely than copying the scalar neighbor depth.

Regression test:

```text
CpuPatchMatchPropagationTest.PropagatesNeighborPlaneDepthToCurrentRay
```

Validation:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 141
```

building-18 32px, one PatchMatch iteration, with strict normal refinement
(`0.005`):

```text
case colite-ms13-plane-propagation-normal-refine-32
num_tracks 1971
num_single_observation_tracks 1766
num_multi_observation_tracks 205
num_observations 2437
mean_observations 1.2364282090309489
multi_observation_track_ratio 0.10400811770674784
normal_refined_pixels 1678
num_accepted 466
num_rejected_depth 946
num_rejected_mutual_consistency 465
num_rejected_normal 382
```

This is identical to `colite-ms12-normal-refine-depth0005-32`, so the
correctness fix does not move the 1-iteration 32px smoke. This likely means the
current low-iteration run either keeps mostly fronto-parallel normals or does
not give plane propagation enough opportunity to change accepted hypotheses.

3-iteration exploratory smoke with the same options:

```text
case colite-ms13-plane-propagation-normal-refine-iter3-32
photometric_filtered_pixels 2329
geometric_consistent_pixels 3182
geometric_filtered_pixels 10642
fused_points 2541
fused_observations 3182
num_tracks 2541
num_single_observation_tracks 2290
num_multi_observation_tracks 251
mean_observations 1.252262888626525
multi_observation_track_ratio 0.098780007870916967
normal_refined_pixels 666
num_accepted 641
num_rejected_depth 1886
num_rejected_mutual_consistency 683
num_rejected_normal 608
```

The 3-iteration run substantially increases geometric-consistent pixels and
fused observations (`2437 -> 3182`). Because the iteration count changed, this
is not an isolated proof that plane propagation alone improves building-18, but
it does show that PatchMatch depth quality is still iteration/refinement
limited.

Next useful depth-quality direction:

- Add a dedicated depth-refinement diagnostic for per-pixel depth change and
  source-depth reprojection error distribution.
- Add a plane-aware random refinement step that perturbs depth along the
  current plane/ray and normal jointly, instead of only using scalar depth
  perturbation.
- Compare 1/2/3 iterations with and without geometric consistency weight to
  find a better runtime-quality default.

## Milestone 14 Multi-Candidate PatchMatch Perturbation Diagnostics

Implemented configurable multi-candidate random plane perturbation:

- `PatchMatchPerturbationOptions::num_samples_per_pixel` controls how many
  random depth/normal candidates are evaluated for each valid pixel.
- `RunPatchMatchRandomPerturbation` now keeps the best improving candidate and
  applies at most one update per pixel.
- Perturbation diagnostics track accepted depth update magnitude and normal
  angle change:
  - `total_abs_depth_update`
  - `mean_abs_depth_update`
  - `max_abs_depth_update`
  - `mean_normal_angle_degrees`
  - `max_normal_angle_degrees`
- `recon_mvs` exposes the setting as
  `--patch_match_perturbation_samples <n>`.
- `stereo/patch_match.tsv` writes the new perturbation diagnostics per image.

Regression tests:

```text
CpuPatchMatchPerturbationTest.RandomPerturbationImprovesCurrentCost
CpuPatchMatchPerturbationTest.EvaluatesMultiplePlaneCandidatesAndReportsUpdateMagnitude
ReconMvsAppTest.PrintsHelp
ReconMvsAppTest.RunsPatchMatchStereoWhenRequested
```

Validation:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 142
```

building-18 32px, one PatchMatch iteration, strict normal refinement
(`0.005`), `--patch_match_perturbation_samples 3`:

```text
case colite-ms14-perturb-samples3-32
num_tracks 1961
num_single_observation_tracks 1765
num_multi_observation_tracks 196
num_observations 2385
mean_observations 1.2162162162162162
multi_observation_track_ratio 0.099949005609382968
photometric_filtered_pixels 2423
normal_refined_pixels 1306
geometric_consistent_pixels 2385
geometric_filtered_pixels 11439
fused_points 1961
fused_observations 2385
num_accepted 424
num_rejected_depth 1023
num_rejected_mutual_consistency 467
num_rejected_normal 328
perturbation_mean_abs_depth_update_avg 3.34111380838889
perturbation_mean_abs_depth_update_max 13.6555714
perturbation_max_abs_depth_update_max 51.0192223
perturbation_mean_normal_angle_degrees_avg 3.98577418777778
perturbation_max_normal_angle_degrees_max 4.99999897
```

Compared with `colite-ms13-plane-propagation-normal-refine-32`:

```text
fused_observations 2437 -> 2385
num_multi_observation_tracks 205 -> 196
mean_observations 1.2364282090309489 -> 1.2162162162162162
multi_observation_track_ratio 0.10400811770674784 -> 0.099949005609382968
num_accepted 466 -> 424
num_rejected_depth 946 -> 1023
num_rejected_normal 382 -> 328
```

Conclusion:

- Multi-candidate perturbation is useful infrastructure because it exposes how
  aggressively PatchMatch is moving depth and normal hypotheses.
- On the current 32px one-iteration building-18 smoke, increasing samples from
  the default `1` to `3` is not beneficial: depth rejects increase and fused
  observations decrease.
- Keep the default at `1`. Treat `--patch_match_perturbation_samples` as an
  experimental tuning knob until depth-range initialization and geometric
  consistency cost are stronger.
- The next quality step should constrain perturbation candidates by source-view
  reprojection/geometric consistency instead of only choosing the lowest
  photometric cost among more random samples.

## Milestone 15 Source-Depth-Gated Perturbation Candidates

Implemented a COLMAP-style geometric gate inside PatchMatch random
perturbation. The gate is intentionally conservative:

- It is active only when source depth maps are available and
  `geometric_consistency_weight > 0`.
- Each perturbation candidate must be consistent with at least
  `min_num_valid_sources` source depth maps.
- Consistency uses the same relative depth threshold as the geometric cost:
  `geometric_max_relative_depth_error`.
- Rejected candidates do not enter photometric/geometric cost ranking.
- `stereo/patch_match.tsv` now writes:
  - `perturbation_geometric_candidate_rejections`
  - `perturbation_geometric_candidate_acceptances`

Regression test:

```text
CpuPatchMatchPerturbationTest.RejectsCandidatesWithoutSourceDepthConsistencyDuringRefinement
```

Target validation:

```text
ctest --test-dir build-wsl -R 'CpuPatchMatchPerturbationTest|ReconMvsAppTest.RunsPatchMatchStereoWhenRequested' --output-on-failure
100% tests passed, 0 tests failed out of 4
```

building-18 32px, one photometric PatchMatch iteration plus one geometric
refinement iteration:

```text
case colite-ms15-geo-perturb-gate-samples3-32-r2
patch_match_perturbation_samples 3
patch_match_geometric_refinement_iterations 1
patch_match_geometric_consistency_weight 0.05
patch_match_geometric_max_relative_depth_error 0.01
normal_refinement_max_relative_depth_difference 0.005

num_tracks 2228
num_single_observation_tracks 1922
num_multi_observation_tracks 306
num_observations 2981
mean_observations 1.3379712746858168
multi_observation_track_ratio 0.13734290843806105
photometric_filtered_pixels 2925
normal_refined_pixels 1706
geometric_consistent_pixels 2981
geometric_filtered_pixels 10843
fused_points 2228
fused_observations 2981
num_accepted 753
num_rejected_depth 1145
num_rejected_mutual_consistency 654
num_rejected_normal 625
perturbation_geometric_candidate_rejections 33547
perturbation_geometric_candidate_acceptances 1790
perturbation_mean_abs_depth_update_avg 2.98812115394444
perturbation_max_abs_depth_update_max 39.4942372
```

Compared with the previous pure multi-candidate perturbation smoke
(`colite-ms14-perturb-samples3-32`):

```text
fused_observations 2385 -> 2981
num_multi_observation_tracks 196 -> 306
mean_observations 1.2162162162162162 -> 1.3379712746858168
multi_observation_track_ratio 0.099949005609382968 -> 0.13734290843806105
num_accepted 424 -> 753
perturbation_max_abs_depth_update_max 51.0192223 -> 39.4942372
```

Conclusion:

- Source-depth gating makes multi-candidate perturbation useful: it filters out
  most random depth/normal candidates before cost ranking and improves the
  multi-observation track ratio on the 32px building-18 smoke.
- The current run still has many normal rejects (`625`) and depth rejects
  (`1145`), so the next depth-quality step should tune the geometric refinement
  schedule and source-depth threshold rather than simply increasing the number
  of random candidates.
- The historical `colite-pm-geom-refine-w005-32` result still has more fused
  observations (`3447`) with different settings, so a controlled sweep of
  `geometric_consistency_weight`, `geometric_max_relative_depth_error`, and
  normal refinement should be run before changing defaults.

## Milestone 16 MVS Geometric Refinement Sweep Tool

Added a small dependency-free Python sweep helper:

```text
scripts/mvs_sweep.py
```

Supported modes:

- `summarize`: read existing `recon_mvs` output directories and write a unified
  TSV summary.
- `run`: execute a small grid of geometric refinement parameters, write
  `sweep_case.json` into each output directory, and then write `summary.tsv`.

The summary includes:

- high-level `summary.txt` metrics
- dense fusion track metrics from `dense/fusion_summary.txt`
- pair-level reject sums from `dense/fusion_image_pairs.tsv`
- perturbation gate accept/reject and update diagnostics from
  `stereo/patch_match.tsv`

Regression test:

```text
MvsSweepScriptTest
```

Validation:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 144
```

### building-18 32px Sweep

Common options:

```text
patch_match_iterations 1
patch_match_window_radius 1
patch_match_perturbation_samples 3
patch_match_geometric_refinement_iterations 1
filter_photometric_max_cost 0.3
fusion_min_num_observations 1
fusion_max_normal_error 10
fusion_source_pixel_search_radius 1
filter_geometric_min_num_consistent_sources 1
```

Depth-error sweep at `weight=0.05`, `normal_depth_threshold=0.005`:

```text
case ms16-w0p05-gd0p005-nd0p005
fused_observations 2818
num_multi_observation_tracks 283
mean_observations 1.3425440686040973
multi_observation_track_ratio 0.13482610767031919
num_accepted 719
num_rejected_depth 1003
num_rejected_mutual_consistency 523
num_rejected_normal 604
perturbation_gate_accept 1001

case ms16-w0p05-gd0p01-nd0p005
fused_observations 2981
num_multi_observation_tracks 306
mean_observations 1.3379712746858168
multi_observation_track_ratio 0.13734290843806105
num_accepted 753
num_rejected_depth 1145
num_rejected_mutual_consistency 654
num_rejected_normal 625
perturbation_gate_accept 1790

case ms16-w0p05-gd0p02-nd0p005
fused_observations 3140
num_multi_observation_tracks 342
mean_observations 1.3569576490924806
multi_observation_track_ratio 0.14779602420051857
num_accepted 826
num_rejected_depth 1212
num_rejected_mutual_consistency 734
num_rejected_normal 641
perturbation_gate_accept 3108
```

Weight sweep at `geometric_depth_error=0.02`,
`normal_depth_threshold=0.005`:

```text
case ms16-w0p02-gd0p02-nd0p005
fused_observations 2890
num_multi_observation_tracks 289
mean_observations 1.2924865831842576
multi_observation_track_ratio 0.12924865831842575
photometric_filtered_pixels 2650
num_accepted 654

case ms16-w0p05-gd0p02-nd0p005
fused_observations 3140
num_multi_observation_tracks 342
mean_observations 1.3569576490924806
multi_observation_track_ratio 0.14779602420051857
photometric_filtered_pixels 2905
num_accepted 826

case ms16-w0p1-gd0p02-nd0p005
fused_observations 3364
num_multi_observation_tracks 334
mean_observations 1.4550173010380623
multi_observation_track_ratio 0.1444636678200692
photometric_filtered_pixels 3493
num_accepted 1052
```

Normal-refinement threshold sweep at `weight=0.05`,
`geometric_depth_error=0.02`:

```text
case ms16-w0p05-gd0p02-nd0p0025
fused_observations 3140
num_tracks 2315
num_multi_observation_tracks 342
mean_observations 1.3563714902807775
multi_observation_track_ratio 0.14773218142548597
num_rejected_normal 643

case ms16-w0p05-gd0p02-nd0p005
fused_observations 3140
num_tracks 2314
num_multi_observation_tracks 342
mean_observations 1.3569576490924806
multi_observation_track_ratio 0.14779602420051857
num_rejected_normal 641

case ms16-w0p05-gd0p02-nd0p01
fused_observations 3140
num_tracks 2305
num_multi_observation_tracks 346
mean_observations 1.3622559652928417
multi_observation_track_ratio 0.15010845986984817
num_rejected_normal 607
```

Single follow-up check:

```text
case ms16-w0p1-gd0p02-nd0p01
fused_observations 3364
num_tracks 2306
num_multi_observation_tracks 335
mean_observations 1.4588031222896791
multi_observation_track_ratio 0.14527320034692107
photometric_filtered_pixels 3493
num_accepted 1058
num_rejected_depth 1199
num_rejected_mutual_consistency 808
num_rejected_normal 712
```

Conclusion:

- For the current "increase multi-observation track ratio" objective, the best
  32px building-18 point from this sweep is:

```text
patch_match_geometric_consistency_weight 0.05
patch_match_geometric_max_relative_depth_error 0.02
normal_refinement_max_relative_depth_difference 0.01
```

- This improves Milestone 15 from `multi_observation_track_ratio=0.1373` to
  `0.1501` and increases fused observations from `2981` to `3140`.
- If the objective shifts to total fused observations / mean observations,
  `weight=0.10`, `gd=0.02`, `nd=0.01` is stronger (`3364` observations,
  mean observations `1.4588`) but it also increases photometric filtering and
  normal/mutual rejects. Treat it as a denser but less conservative option.
- Do not change defaults yet based only on 32px. The next validation should run
  the two candidate settings at 64px or 128px and compare against the historical
  `colite-pm-geom-refine-w005-32` and COLMAP outputs.

## Milestone 17 Multi-Scale PatchMatch Pyramid

Implemented the first coarse-to-fine PatchMatch path:

- `PatchMatchStereoOptions::pyramid_levels` controls the number of levels.
- `PatchMatchStereoSummary::num_pyramid_levels` records the effective level
  count.
- `pyramid_levels=1` keeps the previous single-scale behavior.
- For `pyramid_levels>1`, the implementation builds an image pyramid from
  coarse to fine, scales the per-level camera intrinsics, runs PatchMatch at the
  coarsest level, and upsamples depth/normal maps into the next finer level.
- Source depth maps used by geometric refinement are resized per pyramid level,
  so the second-pass source-depth consistency path remains active.
- `recon_mvs` exposes:

```text
--patch_match_pyramid_levels <n>
```

- `stereo/patch_match.tsv` now includes `pyramid_levels`.

Regression tests:

```text
CpuPatchMatchStereoTest.PyramidInitializationUpsamplesCoarseDepthToFineLevel
ReconMvsAppTest.PrintsHelp
ReconMvsAppTest.RunsPatchMatchStereoWhenRequested
```

Target validation:

```text
ctest --test-dir build-wsl -R 'CpuPatchMatchStereoTest|ReconMvsAppTest.(PrintsHelp|RunsPatchMatchStereoWhenRequested)|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 10
```

building-18 32px smoke, using the conservative Milestone 16 parameter point and
two pyramid levels:

```text
case colite-ms17-pyramid2-w005-gd002-nd001-32
patch_match_pyramid_levels 2
patch_match_iterations 1
patch_match_perturbation_samples 3
patch_match_geometric_refinement_iterations 1
patch_match_geometric_consistency_weight 0.05
patch_match_geometric_max_relative_depth_error 0.02
normal_refinement_max_relative_depth_difference 0.01

num_tracks 2150
num_single_observation_tracks 1868
num_multi_observation_tracks 282
num_observations 3034
mean_observations 1.4111627906976745
multi_observation_track_ratio 0.13116279069767442
photometric_filtered_pixels 3131
normal_refined_pixels 2321
geometric_consistent_pixels 3034
geometric_filtered_pixels 10790
fused_points 2150
fused_observations 3034
num_accepted 884
num_rejected_depth 1161
num_rejected_mutual_consistency 686
num_rejected_normal 688
perturbation_gate_accept 3566
perturbation_gate_reject 39187
perturbation_mean_abs_depth_update_avg 1.63774108311111
perturbation_max_abs_depth_update_max 29.3537235
```

Compared with the single-scale Milestone 16 conservative point
(`ms16-w0p05-gd0p02-nd0p01`):

```text
fused_observations 3140 -> 3034
num_multi_observation_tracks 346 -> 282
mean_observations 1.3622559652928417 -> 1.4111627906976745
multi_observation_track_ratio 0.15010845986984817 -> 0.13116279069767442
num_accepted 835 -> 884
num_rejected_depth 1212 -> 1161
num_rejected_mutual_consistency 734 -> 686
num_rejected_normal 607 -> 688
perturbation_mean_abs_depth_update_avg 2.74740101827778 -> 1.63774108311111
perturbation_max_abs_depth_update_max 39.5037231 -> 29.3537235
```

Conclusion:

- The coarse-to-fine path is functional and produces less aggressive depth
  perturbation updates.
- On the very small 32px building-18 smoke, the current two-level pyramid is
  more conservative: mean observations improve, but total fused observations
  and multi-observation track ratio decrease.
- Keep `--patch_match_pyramid_levels` default at `1` for now.
- The next useful validation is to run the same setting at 64px/128px. A
  pyramid usually has more room to help at higher resolution; at 32px the
  coarse level is extremely small and may oversmooth useful local detail.
- Follow-up tuning should consider per-level iteration counts or disabling
  normal refinement on the coarse level before changing defaults.

### Milestone 18: 64px/128px pyramid candidate validation

Goal: validate whether the Milestone 17 coarse-to-fine PatchMatch path becomes
useful at higher image sizes. The 32px smoke test was intentionally tiny and
showed that a two-level pyramid can be too conservative when the coarse level
has very little image structure.

Common 64px sweep settings:

```text
dataset building-18
model build-wsl/compare-building-18/controlled-colite-4000-pinhole
max_image_size 64
patch_match_iterations 1
patch_match_window_radius 1
patch_match_perturbation_samples 3
patch_match_geometric_refinement_iterations 1
patch_match_geometric_max_relative_depth_error 0.02
normal_refinement_max_relative_depth_difference 0.01
photometric_max_cost 0.3
fusion_min_observations 1
fusion_max_normal_angle_degrees 10
fusion_max_view_angle_degrees 180
fusion_source_pixel_radius 1
geometric_filter_min_consistent_sources 1
num_threads 2
```

64px results:

```text
case                                      obs    tracks  multi  mean_obs  multi_ratio  photo  pair_acc  rej_depth  rej_mutual  rej_normal  perturb_acc
ms18-64-pyr1-w0p05-gd0p02-nd0p01         18446  11289   2039   1.6340    0.1806       8643   7157      5419       5203        5444        12137
ms18-64-pyr1-w0p1-gd0p02-nd0p01          19298  11287   2124   1.7098    0.1882       11153  8011      5419       5736        6006        12044
ms18-64-pyr2-w0p05-gd0p02-nd0p01         19872  11569   2127   1.7177    0.1839       9635   8303      5677       6425        6783        14824
ms18-64-pyr2-w0p1-gd0p02-nd0p01          20520  11513   1980   1.7823    0.1720       11897  9007      5504       6942        7526        14587
```

64px observations:

- `pyramid_levels=2` improves fused observations for both tested weights.
- `w=0.05`: `18446 -> 19872`, up `1426` observations, about `7.7%`.
- `w=0.10`: `19298 -> 20520`, up `1222` observations, about `6.3%`.
- The best dense observation count is currently
  `pyramid_levels=2, geometric_consistency_weight=0.10`.
- Multi-observation ratio is more mixed: it improves for `w=0.05`, but drops
  for `w=0.10`. The latter appears to create more total observations and
  tracks, but also more single-observation tracks.

The initial full 128px four-case sweep, with cloud cleaning and meshing enabled,
was too slow for an interactive validation run and was stopped after the first
case exceeded the time budget. For 128px, the validation was changed to a
lighter direct `recon_mvs` run: PatchMatch, photometric filtering, geometric
filtering, and stereo fusion only; dense cloud cleaning and meshing were
disabled.

128px light validation settings:

```text
max_image_size 128
patch_match_iterations 1
patch_match_window_radius 1
patch_match_perturbation_samples 3
patch_match_geometric_refinement_iterations 1
patch_match_geometric_consistency_weight 0.10
patch_match_geometric_max_relative_depth_error 0.02
normal_refinement_max_relative_depth_difference 0.01
photometric_max_cost 0.3
fusion_min_observations 1
fusion_max_normal_angle_degrees 10
fusion_max_view_angle_degrees 180
fusion_source_pixel_radius 1
geometric_filter_min_consistent_sources 1
num_threads 4
```

128px light results:

```text
case                                           pyramid  obs     tracks  multi  mean_obs  multi_ratio  photo  geom_consistent  geom_filtered  pair_acc  rej_depth  rej_mutual  rej_normal  perturb_acc
colite-ms18-128-pyr1-w01-gd002-nd001-light    1        99592   49307   9795   2.0198    0.1987       43761  99592            121592         50285     20075      40242       54469       45955
colite-ms18-128-pyr2-w01-gd002-nd001-light    2        110255  54358   9484   2.0283    0.1745       46629  110255           110929         55897     21411      50618       71953       54047
```

128px observations:

- `pyramid_levels=2` improves fused observations from `99592` to `110255`,
  up `10663`, about `10.7%`.
- Fused tracks increase from `49307` to `54358`, up `5051`, about `10.2%`.
- Geometric filtered pixels decrease from `121592` to `110929`, matching the
  increase in accepted/consistent pixels.
- Photometric filtered pixels increase from `43761` to `46629`, so the pyramid
  run is denser but still admits more weak photometric pixels.
- Multi-observation ratio drops from `0.1987` to `0.1745`; the next tuning
  pass should prefer improving multi-view support instead of only increasing
  raw fused observation count.

Conclusion:

- The 64px and 128px checks validate that the pyramid implementation is useful
  once the image is large enough. This closes the main concern from the 32px
  smoke test.
- Keep the default `--patch_match_pyramid_levels=1` until validation covers
  more datasets and runtime profiles.
- For building-18 at 64px/128px, the current experimental recommendation is:

```text
--patch_match_pyramid_levels 2
--patch_match_geometric_consistency_weight 0.10
--patch_match_geometric_max_relative_depth_error 0.02
--normal_refinement_max_relative_depth_difference 0.01
```

- The next quality-oriented work should target multi-observation support:
  per-level source-view pruning, stricter final fusion support, and less
  aggressive acceptance of pyramid-created single-observation tracks.

### Milestone 19: fusion error distribution diagnostics

Goal: make stereo fusion threshold tuning less blind. Pair-level counters show
how many candidates are rejected, but they do not show how close the rejected
or accepted candidates are to the current thresholds. This milestone records
the actual error distributions used by fusion.

New output:

```text
dense/fusion_error_distribution.tsv
```

TSV columns:

```text
metric  record  label  lower  upper  count  value
```

Recorded metrics:

- `relative_depth_error`: fusion relative depth error.
- `reprojection_error_pixels`: discrete source-pixel reprojection error.
- `normal_angle_degrees`: normal consistency angle.
- `view_angle_degrees`: seed/source view angle.

For each metric, the file writes:

- `summary/count`
- percentiles: `p50`, `p75`, `p90`, `p95`, `p99`
- fixed histogram bins around practical thresholds:
  - depth: `0.0025`, `0.005`, `0.01`, `0.02`, `0.05`, `0.1`, `0.2`
  - reprojection: `0.25`, `0.5`, `1`, `2`, `3`, `5`, `10` pixels
  - angles: `1`, `2`, `5`, `10`, `20`, `45`, `90`, `180` degrees

The sweep helper now also copies the most useful percentiles into
`summary.tsv`:

```text
fusion_relative_depth_error_p50/p95/p99
fusion_reprojection_error_pixels_p50/p95/p99
fusion_normal_angle_degrees_p50/p95/p99
fusion_view_angle_degrees_p50/p95/p99
```

Intended use:

- If `p95` is far below the current threshold but many candidates are still
  rejected by the same reason, the issue is likely earlier gating or support,
  not that threshold.
- If `p95` or `p99` clusters just above the threshold, a small threshold sweep
  is meaningful.
- If view/normal percentiles grow while multi-observation ratio drops, the
  fusion is probably accepting broader but weaker support; prefer source-view
  or per-level pruning before simply relaxing thresholds.

Verification:

```text
ctest --test-dir build-wsl -R 'CpuStereoFusionTest|ReconMvsAppTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 22
```

### Milestone 20: source-view scale and coverage scoring

Goal: make source view selection less dependent on shared sparse point count
alone. COLMAP-style MVS source selection should prefer views that have useful
baseline, comparable projection scale, and broad target-image support.

New source-view score components:

- `mean_scale_ratio`: for each shared sparse point, estimate image footprint as
  `mean_focal_length / depth` in the target and source image, then use
  `min(scale_target, scale_source) / max(scale_target, scale_source)`. The value
  is in `[0, 1]`; higher means the two views see the point at a more similar
  image scale.
- `coverage_score`: project shared sparse points on the target image into a
  regular grid and compute occupied cells / total cells. The default grid is
  `4x4`, so the value is also in `[0, 1]`; higher means the source view covers
  the target image more broadly instead of only a small local cluster.

Updated score:

```text
score =
  triangulation_angle_weight * mean_triangulation_angle_degrees +
  shared_points_weight * num_shared_points +
  scale_consistency_weight * mean_scale_ratio +
  coverage_weight * coverage_score
```

New defaults:

```text
source_scale_consistency_weight 2.0
source_coverage_weight 5.0
source_coverage_grid_size 4
```

New CLI options:

```text
--source_scale_consistency_weight <w>
--source_coverage_weight <w>
--source_coverage_grid_size <n>
```

`source_images.tsv` now includes:

```text
mean_scale_ratio
coverage_score
```

Verification:

```text
ctest --test-dir build-wsl -R 'MvsPipelineTest|CpuPatchMatch|CpuStereoFusionTest|ReconMvsAppTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 68
```
