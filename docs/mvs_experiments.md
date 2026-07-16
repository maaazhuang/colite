# COLITE MVS 实验矩阵

### Milestone 27: CUDA NCC likelihood dynamic source probability

Date: 2026-07-14

Controlled building-18 setup: `max_image_size=64`, two PatchMatch iterations,
one perturbation sample, one geometric refinement iteration.

```text
case                                      fused_points  CUDA total sum (ms)
static source probability                         272             132.765
dynamic probability, deterministic floor gate    272             139.459
dynamic probability + bilateral NCC              219                 n/a
```

Dynamic diagnostics:

```text
probability updates: 442368
source selection tests: 12712
source selection acceptances: 12712
source selection rejections: 0
final per-source mean probability: approximately 0.35 to 0.44
```

Outputs:

```text
static:    build-wsl/mvs-building-18/cuda-static-selection-i2-ms62-64
dynamic:   build-wsl/mvs-building-18/cuda-dynamic-gate-i2-ms66-64
combined:  build-wsl/mvs-building-18/cuda-dynamic-gate-bilateral-i2-ms67-64
```

An intermediate independent-Bernoulli experiment produced only 34 fused
points because candidates frequently had fewer than the required number of
sources. The retained implementation therefore uses a deterministic floor
gate until fixed-count CDF sampling is available.

Conclusion: the likelihood/posterior update loop is functional and neutral on
the current demo, but does not yet reproduce COLMAP's directional HMM or
fixed-count source sampling. Bilateral NCC remains opt-in.

### Milestone 26: opt-in bilateral weighted NCC

Date: 2026-07-13

Implementation:

- CPU and CUDA share the COLMAP-style reference-driven bilateral weight.
- The existing COLITE `0.5 * (1 - NCC)` scale is preserved.
- The feature is exposed by `--patch_match_bilateral_ncc` and is disabled by default.
- Effective mode and sigma values are exported to `stereo/patch_match.tsv`.

building-18 controlled results (`max_image_size=64`):

```text
case                                      radius  sigma_color  fused_points
previous unweighted baseline                  1       n/a           268
bilateral                                     1       0.2           216
bilateral                                     1       0.3           230
bilateral                                     1       0.5           238
near-unweighted                               2       1000          186
bilateral                                     2       0.2           129
new default, bilateral disabled               1       0.2           268
```

Outputs:

```text
opt-in bilateral: build-wsl/mvs-building-18/cuda-bilateral-ncc-ms56-64
default guarded:  build-wsl/mvs-building-18/cuda-bilateral-optin-ms61-64
```

Conclusion: bilateral weighting is numerically correct and suppresses synthetic cross-edge outliers, but enabling it alone regresses the current low-resolution, small-window pipeline. It therefore remains opt-in until NCC likelihood and dynamic source selection are integrated and evaluated together.

### Milestone 25: CUDA plane propagation and decoupled perturbation candidates

Date: 2026-07-13

Goal: align the CUDA hypothesis search semantics with COLMAP before replacing
the current photometric cost.

Implemented:

- deterministic randomized initial normals on CUDA;
- plane-induced neighbor propagation to the current reference ray;
- joint, normal-only, and depth-only perturbation candidates;
- separate propagation and perturbation candidate counters in TSV diagnostics.

Controlled building-18 result (`max_image_size=64`, one PatchMatch iteration,
one perturbation sample, one geometric refinement iteration):

```text
baseline output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms52-default-checkerboard
current output:  build-wsl/mvs-building-18/cuda-colmap-align-ms55-64

metric                              baseline       current
geometric_consistent_pixels             216           268
fused_points                            216           268
photometric_filtered_pixels           32124         26477
CUDA refinement total (ms)          169.779       109.978
CUDA propagation (ms)                64.468        16.618
CUDA perturbation (ms)                3.025         2.626

current plane propagation candidates: 868608
current perturbation candidates:       165888
```

Interpretation:

- The fused-point count improved by 24.1%, so the first alignment milestone
  closes rather than regresses the building-18 quality proxy.
- Timing is a single low-resolution observation and must not be treated as a
  stable speedup until repeated at 128 px and a practical resolution.
- The next implementation target is bilateral weighted NCC, followed by
  dynamic selection probability and stronger reprojection geometry scoring.

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

### Milestone 21: PatchMatch backend selection and CUDA entry point

Goal: add a COLMAP-style backend boundary before porting PatchMatch kernels to
CUDA. The CPU implementation remains the fallback path, while `auto` is now the
default user-facing backend and will prefer CUDA once the real CUDA backend is
compiled and available.

Implemented:

- Added `PatchMatchBackend { auto, cpu, cuda }` to `PatchMatchStereoOptions`.
- Added `--patch_match_backend <auto|cpu|cuda>` and
  `--patch_match_gpu_index <n>` to `recon_mvs`.
- Added `PatchMatchStereoSummary::requested_backend` and `actual_backend`.
- Added `stereo/patch_match.tsv` columns:
  `requested_backend` and `actual_backend`.
- Added optional `COLITE_ENABLE_CUDA_MVS` CMake switch and a CUDA PatchMatch
  stub entry point. The stub intentionally reports unavailable until real CUDA
  kernels are implemented.
- Updated `scripts/mvs_sweep.py` to summarize requested/actual PatchMatch
  backends.

CUDA environment check:

```text
/usr/local/cuda-13.2/bin/nvcc -V
Cuda compilation tools, release 13.2, V13.2.51
```

Current limitation:

- This milestone does not yet implement the CUDA PatchMatch kernels. Forcing
  `--patch_match_backend cuda` currently fails with a clear runtime error.
- `--patch_match_backend auto` falls back to CPU in the current build.

Verification:

```text
cmake --build build-wsl --target colite_core_tests -j2
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 25
```

### Milestone 22: first runnable CUDA PatchMatch backend

Goal: replace the CUDA PatchMatch stub with a runnable GPU backend while
keeping the CPU backend as the correctness baseline.

Implemented:

- `src/colite/mvs/patch_match_cuda.cu` now provides the compiled CUDA backend
  when `COLITE_ENABLE_CUDA_MVS=ON`.
- CMake selects `patch_match_cuda.cu` for CUDA builds and keeps
  `patch_match_cuda_stub.cpp` for non-CUDA builds.
- The CUDA backend now uploads:
  - reference image,
  - packed source image cache,
  - optional packed source depth maps,
  - relative reference-to-source camera transforms,
  - depth, normal, cost, and summary counter buffers.
- Implemented GPU kernels for:
  - random depth hypothesis initialization,
  - default normal initialization,
  - ZNCC photometric cost,
  - plane reprojection into source views,
  - top-k multi-source cost aggregation,
  - optional source-depth geometric penalty,
  - forward/backward/four-direction propagation,
  - random depth perturbation.
- `auto` now uses CUDA in CUDA-enabled builds when the request is supported;
  unsupported CUDA cases, such as multi-scale pyramid execution, still fall
  back to CPU unless the user explicitly forces `cuda`.

Current limitations:

- The GPU path is a first runnable backend, not a full COLMAP CUDA port.
- It supports single-scale execution only. `pyramid_levels > 1` falls back to
  CPU in `auto` mode.
- Normal perturbation is still minimal; the first GPU perturbation pass changes
  depth hypotheses but keeps the current normal.
- COLMAP selection probability maps, rotated sweep scheduling, incident-angle
  weighting, and mature geometric filtering are not implemented yet.

Verification:

```text
CUDA build:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest' --output-on-failure
100% tests passed, 0 tests failed out of 12

CUDA recon_mvs integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 14

CPU build:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 26
```

### Milestone 40: CUDA output normal materialization

Goal: remove host-side output normal normalization from dense map
materialization while keeping CUDA output normals valid and normalized.

Implemented:

- Added `cuda_host_initial_normal_upload_normalizations` and
  `cuda_host_output_normalizations` diagnostics to `PatchMatchStereoSummary`.
- Exported both diagnostics in `patch_match_initial.tsv` and
  `patch_match.tsv`.
- Normalized initial-map normals before asynchronous CUDA upload, so external
  non-unit initial maps do not break the device-side hypothesis invariant.
- Normalized packed output normals in `PackOutputMapsKernel`, then let host
  materialization copy finite non-zero normals directly into `NormalMap`.
- Added a CUDA regression test with non-unit initial normals.

building-18 CUDA profiling:

```text
baseline: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms47
current : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms49

initial result_materialization_sum:
  ms47 = 86.179596 ms
  ms49 = 64.711323 ms

initial dense_map_fill_sum:
  ms47 = 55.341985 ms
  ms49 = 34.116230 ms

final result_materialization_sum:
  ms47 = 82.700163 ms
  ms49 = 61.074301 ms

final dense_map_fill_sum:
  ms47 = 55.202973 ms
  ms49 = 34.254693 ms

final cuda_total_time_sum:
  ms47 = 117.598514 ms
  ms49 = 93.137280 ms

fused_points:
  ms47 = 156
  ms49 = 155
```

Conclusion:

- Moving output normal normalization out of host materialization removes the
  largest remaining dense-map-fill cost at this 64px test size.
- The current implementation keeps normalization in CUDA pack output instead of
  dropping it entirely, which is a conservative compromise between performance
  and prior output semantics.
- The one-point fused cloud difference on building-18 should be treated as a
  watch item. It may be a threshold-edge numeric difference, but larger tests
  should verify that the optimization does not systematically reduce geometric
  consistency or fusion density.

Verification:

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

### Milestone 41: Default CUDA propagation schedule moves to checkerboard

Goal: close the stability/quality gap between COLITE CUDA PatchMatch and
COLMAP-style red-black propagation after observing a one-point fused cloud
difference in the output-normal materialization milestone.

Investigation:

- The ms47 -> ms49 one-point difference was isolated to:
  - image id 14,
  - `P1180179.JPG`,
  - pixel `(row=5, col=52)`.
- In ms47, the pixel had:
  - `depth=7.79879427`,
  - `cost=0.0737642869`,
  - consistency graph entry `5 52 1 15`.
- In ms49, the same pixel was removed by geometric consistency.
- Photometric filtering for image 14 was unchanged:
  - `valid_cost_pixels_after_filter=1343`,
  - `photometric_filtered_pixels=1729`.
- Re-running the current code with the old default `forward_backward` produced
  `fused_points=156` again, so the ms49 one-point drop is a schedule/timing
  sensitive edge case rather than a fixed normal-materialization quality
  regression.

Implemented:

- Changed `ReconMvsControllerOptions::patch_match_propagation_schedule` default
  from `forward_backward` to `checkerboard`.
- Changed `PatchMatchStereoOptions::propagation_schedule` default from
  `kForwardBackward` to `kCheckerboard`, so library API and CLI defaults are
  aligned.
- Updated `recon_mvs --help` to report `default checkerboard`.
- Kept explicit CLI overrides for all schedules.
- Added a help regression test for the new user-facing default.
- Added a PatchMatch options regression test for the lower-level default.

building-18 CUDA profiling:

```text
ms50 current code, old implicit forward_backward:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms50
  geometric_consistent_pixels=156
  fused_points=156

ms51 explicit checkerboard:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms51-checkerboard
  geometric_consistent_pixels=216
  fused_points=216

ms52 default checkerboard:
  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms52-default-checkerboard
  geometric_consistent_pixels=216
  fused_points=216
```

Conclusion:

- Checkerboard/red-black propagation is a better default for the CUDA path and
  more aligned with COLMAP's PatchMatch structure.
- The previous forward/backward CUDA kernel remains useful as an experiment but
  is more sensitive because it reads and writes neighboring hypotheses in the
  same global map during a propagation pass.
- On building-18 at 64px, the new default increases fused points from 156 to
  216 under otherwise identical parameters.

Verification:

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

Not rerun in this pass:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
reason: WSL escalation request rejected by approval system.
```

### Milestone 39: CUDA dense map bulk materialization

Goal: reduce the host-side dense map fill cost after CUDA output readback by
writing directly into map storage instead of using per-pixel setters.

Implemented:

- Added mutable storage accessors:
  - `DenseDepthMap::MutableData()`
  - `NormalMap::MutableData()`
  - `DenseCostMap::MutableData()`
- Added unit tests proving mutable writes are visible through the public map
  query APIs.
- Changed CUDA output materialization to fill depth, normal, and cost maps by
  linear index.
- Kept normal validity behavior unchanged: invalid output normals become zero;
  valid normals remain normalized.

building-18 CUDA profiling:

```text
baseline: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms46
current : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms47

initial dense_map_fill_sum:
  ms46 = 56.471037 ms
  ms47 = 55.341985 ms

final dense_map_fill_sum:
  ms46 = 56.587363 ms
  ms47 = 55.202973 ms

final result_materialization_sum:
  ms46 = 84.490515 ms
  ms47 = 82.700163 ms

fused_points:
  ms46 = 156
  ms47 = 156
```

Conclusion:

- The bulk write path removes some setter overhead without changing output
  scale, but the measured gain is modest at 64px.
- Remaining materialization time is dominated by normal handling and downstream
  normal refinement rather than vector copying or setter dispatch.
- Next optimization target should be a stronger CUDA normal-output invariant
  plus either GPU-side normal refinement or avoiding duplicate initial/final
  normal refinement work.

Verification:

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

### Milestone 32: CUDA workspace pool and double buffering

Goal: reduce inter-image host/device idle time by allowing one CUDA PatchMatch
workspace to download results while another workspace uploads and runs kernels
for the next reference image.

Implemented:

- Added controller-level `CudaPatchMatchWorkspace` pooling for CUDA/auto
  `recon_mvs` runs.
- The pool is conservatively capped at two slots:
  `min(num_threads, num_images, 2)`.
- Each slot owns its own CUDA stream, cached textures, reusable device buffers,
  and pinned host staging buffers.
- Initial PatchMatch dispatch now uses at most one active future per workspace
  slot. A slot is reused only after its previous future has completed.
- Geometric refinement remains sequential, but now also uses the pool by
  round-robin slot selection.
- `PatchMatchStereoSummary` and `stereo/patch_match.tsv` now report:
  - `cuda_workspace_pool_size`,
  - `cuda_workspace_pool_slot`.

Why this matters:

- This moves the CUDA path from a single shared workspace toward a pipeline
  where D2H readback for image N can overlap upload/kernel work for image N+1.
- It preserves the previous cache-reuse work while avoiding concurrent writes
  to the same workspace.

Still not complete:

- The pool size is intentionally capped at two until real scene profiling shows
  that more concurrent streams help more than they increase memory pressure.
- Geometric refinement is not yet fully pipelined.
- Per-slot timing is reported per image through the existing PatchMatch TSV,
  but there is no end-to-end overlap efficiency metric yet.

Verification:

```text
CUDA build:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 19, 1 skipped

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CPU build:
cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 33: CUDA pinned async H2D upload staging for initial maps

Goal: continue reducing host-device transfer stalls by moving reusable
linear-buffer uploads onto pinned host staging and the workspace stream.

Implemented:

- Added `DeviceBuffer::CopyFromHostAsync`.
- Added pinned upload staging buffers to `CudaPatchMatchWorkspace`:
  - `host_depth_upload`,
  - `host_normal_upload`.
- When CUDA PatchMatch receives `initial_maps`, depth and normal maps are now
  copied into pinned host buffers and uploaded with
  `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)`.
- The subsequent CUDA kernels run on the same workspace stream, so they are
  ordered after the async uploads without a global device synchronize.
- Added diagnostics:
  - `cuda_async_host_uploads`,
  - `cuda_pinned_host_upload_bytes`.
- Added the same fields to `stereo/patch_match.tsv`.

Why this matters:

- CUDA pyramid execution and geometric refinement commonly enter PatchMatch
  with prior depth/normal hypotheses. Those paths no longer use synchronous
  depth/normal H2D copies.
- This fits the existing workspace-pool direction: each slot now owns both
  pinned upload staging and pinned download staging.

Current limitation:

- This milestone covers only linear depth/normal uploads for `initial_maps`.
  Texture-backed inputs are handled by the following texture-upload milestone.

Verification:

```text
CUDA target test:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncUploadStagingForInitialMaps --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA backend regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 20, 1 skipped

CUDA recon_mvs integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CPU build and backend regression:
cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 19, CUDA-only tests skipped

CPU recon_mvs integration:
ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 34: CUDA pinned async texture upload staging

Goal: extend the previous pinned/async H2D work from linear depth/normal
buffers to the texture-backed inputs that dominate CUDA PatchMatch setup.

Implemented:

- Added `FloatLayeredTexture::CreateAsync`.
- Replaced synchronous `cudaMemcpy3D` texture uploads in CUDA PatchMatch with
  `cudaMemcpy3DAsync(..., workspace_stream)`.
- Added per-texture pinned staging buffers to `CudaPatchMatchWorkspace`:
  - `host_reference_texture_upload`,
  - `host_source_images_texture_upload`,
  - `host_poses_texture_upload`,
  - `host_source_depth_maps_texture_upload`,
  - `host_selection_probability_texture_upload`.
- Each texture input uses a distinct staging buffer so queued async copies do
  not race with host-side overwrites from later texture packing.
- The following inputs now upload through pinned staging and the workspace
  stream:
  - reference image texture,
  - source image layered texture,
  - pose texture,
  - optional source depth maps texture,
  - optional dense selection probability texture.
- Added diagnostics:
  - `cuda_async_texture_uploads`,
  - `cuda_pinned_texture_upload_bytes`.
- Added the same fields to `stereo/patch_match.tsv`.

Why this matters:

- The CUDA workspace slot now owns pinned staging for both upload and download
  sides of PatchMatch.
- Together with workspace pooling, this gives the runtime a cleaner chance to
  overlap image N downloads with image N+1 texture uploads and kernels.

Current limitation:

- CUDA array allocation and texture object creation still happen on the host
  setup path. The async work covers the host-to-array copy, not allocation.
- Pinned texture staging is capacity-based per workspace slot; real-scene
  profiling should watch host pinned memory pressure before increasing pool
  size beyond two.
- The following workspace-pool diagnostics milestone adds summary-level
  overlap-efficiency estimates.

Verification:

```text
CUDA target test:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncTextureUploadStaging --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA backend regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 21, 1 skipped

CUDA recon_mvs integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CPU build and backend regression:
cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 20, CUDA-only tests skipped

CPU recon_mvs integration:
ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 35: CUDA workspace-pool overlap diagnostics

Goal: make the workspace-pool / double-buffering optimization measurable at the
batch level, instead of inferring overlap only from per-image timings.

Implemented:

- Added `PatchMatchSchedulingSummary` in `recon_mvs`.
- The initial PatchMatch batch now records:
  - `patch_match_initial_wall_time_ms`,
  - `patch_match_initial_sum_cuda_time_ms`,
  - `patch_match_initial_overlap_saved_time_ms`,
  - `patch_match_initial_overlap_ratio`.
- The values are written to `summary.txt`.
- The overlap estimate is conservative:
  - wall time is measured around the full initial PatchMatch dispatch/collect
    phase,
  - summed CUDA time adds each CUDA image's `cuda_total_time_ms`,
  - saved time is `max(0, summed_cuda_time - wall_time)`,
  - ratio is saved time divided by summed CUDA time.

Why this matters:

- Upload/download/kernel work has been moved onto per-workspace streams and
  pinned staging. This milestone makes it possible to tell whether the
  two-slot workspace pool is actually reducing batch wall time.
- The metric is especially useful when comparing `--num_threads 1` and
  `--num_threads 2` with the CUDA backend.

Current limitation:

- The following refinement-pool milestone adds a separate overlap phase for
  geometric refinement.
- `cuda_total_time_ms` and controller wall time are not perfectly identical
  timing domains, so this should be treated as a diagnostic estimate, not a
  benchmark-grade profiler result.

Verification:

```text
CUDA two-thread integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA backend regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 21, 1 skipped

CUDA recon_mvs single-thread integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CPU build:
cmake --build build-wsl --target colite_core_tests -j2
passed

CPU recon_mvs integration:
ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 36: CUDA workspace-pool geometric refinement

Goal: remove the sequential geometric-refinement bottleneck now that CUDA
PatchMatch inputs and outputs are stream-aware and workspace-slot scoped.

Implemented:

- Added `PatchMatchRefinementJob` and `PatchMatchRefinementResult`.
- Geometric refinement now builds immutable per-image jobs before dispatch:
  - reference view/image,
  - source views/images,
  - source depth map snapshots,
  - per-image depth range.
- The CUDA/auto path dispatches refinement jobs through the same
  `CudaPatchMatchWorkspace` pool used by the initial PatchMatch pass.
- The CPU path can also dispatch refinement through `--num_threads`.
- Source depth maps for refinement are taken from the initial PatchMatch
  snapshot, removing the previous order-dependent behavior where later images
  could read already-refined source depth maps.
- Added refinement-stage scheduling diagnostics to `summary.txt`:
  - `patch_match_refinement_wall_time_ms`,
  - `patch_match_refinement_sum_cuda_time_ms`,
  - `patch_match_refinement_overlap_saved_time_ms`,
  - `patch_match_refinement_overlap_ratio`.

Why this matters:

- Geometric refinement now benefits from the same stream/pinned-staging
  pipeline as the initial pass.
- The snapshot input model is easier to reason about and prepares the code for
  future multi-round geometric consistency refinement.

Current limitation:

- Initial PatchMatch and geometric refinement are still separate phases.
  Refinement starts only after all initial depth maps are complete.
- The source-depth snapshot increases temporary memory use; larger scenes
  should be profiled before increasing the workspace pool beyond two slots.

Verification:

```text
CUDA refinement integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA backend regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 21, 1 skipped

CUDA two-thread integration:
ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

CPU build:
cmake --build build-wsl --target colite_core_tests -j2
passed

CPU recon_mvs integration:
ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWithTwoThreadsWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 25: CUDA PatchMatch profiling baseline

Goal: establish measurable CUDA PatchMatch performance diagnostics before
changing memory reuse, host-device transfer behavior, or kernel organization.

Implemented:

- Added CUDA timing fields to `PatchMatchStereoSummary`:
  `cuda_upload_time_ms`, `cuda_initialization_time_ms`,
  `cuda_evaluate_cost_time_ms`, `cuda_propagation_time_ms`,
  `cuda_perturbation_time_ms`, `cuda_download_time_ms`, and
  `cuda_total_time_ms`.
- Single-scale CUDA PatchMatch now records:
  - host packing, layered texture creation, device buffer allocation, and H2D
    upload wall time,
  - initialization kernel time,
  - initial cost evaluation kernel time,
  - propagation kernel group time across iterations,
  - perturbation kernel time across iterations,
  - D2H download wall time,
  - total CUDA backend wall time.
- CUDA pyramid execution accumulates per-level timing into the final summary.
- `stereo/patch_match.tsv` now writes the same profiling fields for every image.
  CPU backend rows keep these values at zero.

Why this matters:

- The next optimization stages need a stable baseline. These fields separate
  upload/download overhead from kernel execution, so buffer/texture reuse and
  host-device roundtrip changes can be measured directly.
- The propagation and perturbation split makes it possible to decide whether
  checkerboard/red-black scheduling, random perturbation, or cost aggregation
  dominates runtime.

Current limitation:

- CUDA event timing intentionally synchronizes after measured kernel groups.
  This is acceptable for diagnostics now, but should become optional if profiling
  overhead becomes visible in full-scale dense reconstruction.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReportsProfilingTimes --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 30: CUDA async D2H with pinned host staging

Goal: move CUDA PatchMatch readbacks toward stream-aware asynchronous transfers
without changing the public MVS output model.

Implemented:

- Added pinned host staging buffers to `CudaPatchMatchWorkspace`:
  - `host_iteration_counters`,
  - `host_output_pixels`.
- Added `DeviceBuffer::CopyToHostAsync`.
- Changed iteration-counter D2H from synchronous copy to
  `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, workspace_stream)` into pinned
  host memory.
- Changed packed output D2H from synchronous copy to async copy into pinned host
  memory.
- Added a lazy-created CUDA stream to `CudaPatchMatchWorkspace`.
  The stream is created only after device selection, so externally created
  workspaces do not bind early to the wrong GPU.
- Moved CUDA PatchMatch kernels, propagation/perturbation events, output pack,
  and async D2H onto the workspace stream.
- Added diagnostics:
  `cuda_async_counter_readbacks`, `cuda_async_output_readbacks`,
  `cuda_pinned_host_staging_bytes`, and `cuda_stream_kernel_launches`.

Why this matters:

- `cudaMemcpyAsync` only gives useful overlap with page-locked host memory.
  This milestone establishes the staging buffers and stream ownership needed for
  real overlapping transfers in later work.
- The implementation still synchronizes before constructing CPU-side dense maps,
  but it removes default-stream assumptions and makes the data path ready for
  double-buffering or a workspace pool.

Current limitation:

- The controller still consumes each result immediately, so a final
  `cudaStreamSynchronize` remains necessary before filling `DenseDepthMap`,
  `NormalMap`, and `DenseCostMap`.
- Uploads and texture creation are still mostly synchronous. Future work should
  target pinned upload staging and overlapping the next image's upload with the
  previous image's output transfer.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendUsesPinnedAsyncReadbackStaging --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 19

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 26: CUDA shared-memory reference tile

Goal: start CUDA PatchMatch shared-memory optimization without changing
photometric/geometric behavior.

Implemented:

- `EvaluateCostKernel` now allocates a dynamic shared-memory reference tile.
- Each CUDA block caches the reference image center pixels covered by that
  block.
- Reference patch samples that fall inside the cached linear block range read
  from shared memory; samples outside that range keep using the original
  reference texture path.
- Source image sampling, source depth sampling, pose texture access,
  propagation, checkerboard propagation, and perturbation remain unchanged.
- Added `cuda_shared_reference_tile_pixels` to `PatchMatchStereoSummary` and
  `stereo/patch_match.tsv`.

Why this matters:

- The initial cost evaluation repeatedly reads the same reference pixels across
  source views and pose variants. This milestone removes part of that repeated
  reference texture traffic while preserving the existing texture fallback for
  block boundaries.
- The implementation is intentionally conservative. It validates the shared
  memory path before moving to a larger 2D tile with halo pixels.

Current limitation:

- The tile is one-dimensional and covers only pixels owned by the current block,
  so patch halo pixels near block boundaries still fall back to texture reads.
- Propagation and perturbation still use the old reference texture sampling path.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReportsSharedReferenceTileUsage --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 15
```

### Milestone 27: CUDA PatchMatch workspace/cache reuse

Goal: reduce repeated CUDA allocation and texture-object setup across pyramid
levels and across reference images in the `recon_mvs` PatchMatch batch.

Implemented:

- Added an opaque `CudaPatchMatchWorkspace` handle with:
  - `MakeCudaPatchMatchWorkspace`,
  - `DestroyCudaPatchMatchWorkspace`,
  - `CudaPatchMatchRequest::workspace`.
- The CUDA backend now uses a caller-provided workspace when available and
  creates a temporary workspace otherwise.
- The workspace owns reusable CUDA resources:
  - reference image layered texture,
  - source images layered texture,
  - pose texture,
  - optional source depth maps texture,
  - optional dense source-selection probability texture,
  - source image metadata buffer,
  - depth, normal, and cost buffers,
  - device counter buffer.
- `DeviceBuffer` now keeps capacity and only calls `cudaMalloc` when the new
  requested size exceeds current capacity.
- `FloatLayeredTexture` now reuses its CUDA array and texture object when
  width, height, depth, and filter mode match; it only re-uploads data.
- CUDA pyramid execution reuses one workspace across levels.
- `recon_mvs` creates one CUDA workspace for the PatchMatch batch and reuses it
  across initial PatchMatch images and geometric refinement passes.
- Added diagnostics:
  `cuda_workspace_texture_allocations`, `cuda_workspace_texture_reuses`,
  `cuda_workspace_buffer_allocations`, and `cuda_workspace_buffer_reuses`.

Why this matters:

- Dense reconstruction runs PatchMatch repeatedly over similarly sized images.
  Before this milestone, every reference image and every pyramid level created
  fresh CUDA textures and buffers. This made upload/setup time harder to
  separate from actual kernel cost and introduced avoidable allocation churn.
- The new diagnostics make reuse visible in `stereo/patch_match.tsv`, so later
  host-device and kernel-fusion work can be measured against allocation churn.

Current limitation:

- `recon_mvs` uses sequential PatchMatch scheduling when a shared CUDA workspace
  is active. This avoids concurrent writes to the same workspace. Future work can
  add one workspace per CPU worker, CUDA streams, or a small workspace pool.
- Textures are reused only when dimensions and filter mode match. Different
  pyramid levels usually allocate separate shapes as expected; same-size images
  benefit most.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendReusesWorkspaceResources --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 16

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 28: CUDA iteration counter batching and sync reduction

Goal: reduce per-iteration host-device traffic in CUDA PatchMatch while keeping
the existing per-iteration diagnostics.

Implemented:

- Added a reusable device-side `iteration_counters` buffer to
  `CudaPatchMatchWorkspace`.
- Added `ResetCountersKernel` so each iteration resets counters on device rather
  than uploading a host-side zeroed `DeviceCounters`.
- Added `StoreIterationCountersKernel` so each iteration snapshots counters on
  device after propagation and perturbation.
- Replaced per-iteration `cudaDeviceSynchronize()` plus scalar D2H counter copy
  with one final synchronization and one batched counter `CopyToHost`.
- Changed propagation and perturbation profiling events so the loop records
  event stops immediately but reads elapsed time only after the final sync.
- Added diagnostics:
  `cuda_iteration_counter_snapshots` and
  `cuda_iteration_counter_readbacks`.

Why this matters:

- PatchMatch iteration diagnostics previously forced a host-device roundtrip per
  iteration. That made profiling more intrusive and serialized the CUDA stream
  more than necessary.
- The new path preserves the same per-iteration vectors while reducing counter
  readback count from `num_iterations` to `1`.

Current limitation:

- Depth, normal, and cost maps are still downloaded as three separate buffers.
  A later kernel-fusion or packed-output step can reduce those final downloads.
- Kernel launches are still host-driven per propagation/perturbation step. This
  milestone removes the most obvious diagnostics readback loop, not the full
  launch overhead.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendBatchesIterationCounterReadbacks --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 17
```

### Milestone 29: CUDA packed output map readback

Goal: reduce final CUDA PatchMatch map downloads from three host-device
readbacks to one.

Implemented:

- Added `DeviceOutputPixel` containing depth, normal x/y/z, and cost.
- Added a reusable `output_pixels` buffer to `CudaPatchMatchWorkspace`.
- Added `PackOutputMapsKernel` to pack depth, normal, and cost device buffers
  into `output_pixels` before host download.
- Replaced three final D2H copies:
  - depth map,
  - normal map,
  - cost map,
  with one packed `CopyToHost`.
- Added diagnostics:
  `cuda_packed_output_pixels` and `cuda_output_map_readbacks`.

Why this matters:

- The previous output path performed three separate synchronous downloads after
  PatchMatch. Packing output on GPU reduces host-device roundtrips and gives a
  cleaner boundary for later async/pinned-memory work.
- This also keeps the CPU map construction code simple: host-side reconstruction
  now reads one contiguous output vector.

Current limitation:

- The packed buffer is still copied with synchronous `cudaMemcpy`.
  The next natural step is pinned host memory plus `cudaMemcpyAsync`, or a
  stream-aware workspace so output copy can overlap with setup for the next
  reference image.

Verification:

```text
ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest.CudaBackendPacksOutputMapsForSingleReadback --output-on-failure
100% tests passed, 0 tests failed out of 1

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
100% tests passed, 0 tests failed out of 18

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
100% tests passed, 0 tests failed out of 1
```

### Milestone 30: CPU/CUDA geometric pre-cost gate with occlusion and depth conflicts

Goal: make the geometric candidate gate stronger and consistent across CPU and
CUDA PatchMatch propagation / perturbation candidates.

Implemented:

- CPU source-depth consistency now follows the CUDA/COLMAP-like
  forward/backward reference-depth check:
  1. project the reference hypothesis to the source view,
  2. sample the source depth map,
  3. back-project the sampled source depth to the reference camera,
  4. compare returned reference depth against the candidate depth.
- The geometric candidate gate classifies each source as:
  - `consistent`: absolute returned-depth error is within threshold,
  - `occluded`: returned depth is closer than the candidate depth,
  - `conflicting`: returned depth is farther than the candidate depth.
- CPU propagation, CPU checkerboard propagation, and CPU perturbation now use
  the same pre-cost gate before multi-source cost aggregation.
- CUDA propagation and CUDA perturbation now use the same classification rule.
- The default rule is conservative:
  - reject if fewer than `min_num_valid_sources` are consistent,
  - reject if any source clearly occludes the candidate,
  - reject if the number of conflicting sources is greater than or equal to the
    number of consistent sources.
- `PatchMatchCostOptions` now exposes:
  - `geometric_reject_occluded_candidates`,
  - `geometric_reject_depth_conflicts`.
- Added regression tests for:
  - CPU propagation rejecting an occluded candidate before cost aggregation,
  - CPU propagation rejecting competing depth conflicts,
  - CUDA rejecting an occluded candidate despite one consistent source,
  - CUDA rejecting a farther depth conflict despite one consistent source.

Notes:

- The existing `num_perturbation_geometric_candidate_*` summary fields are kept
  for compatibility, but now represent the total geometric candidate gate
  counters for propagation + perturbation.
- This still does not implement a full COLMAP visibility graph or full
  occlusion ordering across all depth hypotheses; it is a local candidate gate
  that prevents obviously inconsistent hypotheses from entering the cost stage.

Verification:

```text
New occlusion/conflict tests:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'RejectsGeometricallyOccludedCandidateBeforeCostAggregation|RejectsCandidateWhenDepthConflictsCompeteWithConsistentSources|CudaBackendRejectsOccludedCandidateDespiteOneConsistentSource|CudaBackendRejectsDepthConflictDespiteOneConsistentSource' --output-on-failure
100% tests passed, 0 tests failed out of 4

PatchMatch targeted regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchPropagationTest|CpuPatchMatchPerturbationTest|CpuPatchMatchStereoTest|CpuPatchMatchPhotometricCostTest' --output-on-failure
100% tests passed, 0 tests failed out of 38

CPU targeted regression:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchPropagationTest|CpuPatchMatchPerturbationTest|CpuPatchMatchStereoTest|CpuPatchMatchPhotometricCostTest' --output-on-failure
100% tests passed, 0 tests failed out of 38
```

### Milestone 31: COLMAP-style consistency graph and fused point visibility

Goal: implement the MVS visibility pieces that COLMAP exposes around
`patch_match_stereo` and `stereo_fusion`, while keeping COLITE's existing text
and binary diagnostics stable.

Implemented:

- `ConsistencyGraph` can now convert to/from COLMAP's sparse consistency graph
  data layout: `[col, row, N, image_idx_1, ...]` for every pixel with at least
  one consistent source image.
- Added COLMAP-compatible consistency graph binary IO:
  - header: `width&height&1&`,
  - payload: little-endian `int` sparse graph data.
- The internal `ConsistencyGraph` continues to store `ImageId`; COLMAP graph IO
  maps between `ImageId` and MVS image index through `image_ids_by_index`.
- `StereoFusionResult` now records `points_visibility`, one sorted unique
  image-index list per fused dense point.
- Added COLMAP-compatible `points_visibility.bin` IO:
  - `num_points:uint64`,
  - for each point: `num_visible_images:uint32` followed by
    `image_idx:uint32[]`.
- `recon_mvs` now writes:
  - `stereo/consistency_graphs_colmap/<image_name>.bin`,
  - `dense/points_visibility.bin`.

Why this matters:

- COLITE fused dense points now carry the same kind of per-point visibility
  information as COLMAP stereo fusion.
- The consistency graph can be inspected or compared in a COLMAP-like binary
  layout without replacing the existing COLITE graph files.
- Downstream meshing can consume fused point visibility explicitly instead of
  relying only on local-neighbor geometry.

Still not complete:

- This is not yet COLMAP's Delaunay/free-space graph-cut meshing visibility.
- The image index order is COLITE's current MVS result order; it is stable for a
  run and documented, but downstream consumers must use the same order.

Verification:

```text
Core visibility tests:
ctest --test-dir build-wsl -R 'RoundTripsConsistencyGraphAsColmapBinary|RoundTripsColmapPointsVisibilityBinary|MergesReciprocalConsistentPixelsIntoSingleTrack' --output-on-failure
100% tests passed, 0 tests failed out of 3

recon_mvs output tests:
ctest --test-dir build-wsl -R 'ReconMvsExeTest.(WritesBinaryDenseMapsWhenRequested|WritesStereoFusionPlyWhenRequested)' --output-on-failure
100% tests passed, 0 tests failed out of 2
```

### Milestone 29: CUDA geometric candidate filtering before cost aggregation

Goal: strengthen geometric consistency in CUDA PatchMatch by rejecting
obviously inconsistent propagation and perturbation hypotheses before the full
photometric/geometric cost aggregation step.

Implemented:

- Added a device-side `PassesGeometricCandidateFilter` helper.
- The filter reuses the existing COLMAP-like forward/backward source-depth
  check:
  1. project the reference hypothesis into each source view,
  2. sample the source depth map,
  3. back-project the source depth to the reference camera,
  4. compare returned reference depth with the current hypothesis depth.
- Propagation candidates now pass this geometric candidate filter before
  `ComputeAggregatedCost`.
- Perturbation candidates now pass the same filter before
  `ComputeAggregatedCost`.
- CUDA summary records candidate accept/reject counters through the existing
  geometric candidate diagnostic fields. These fields now include both
  propagation and perturbation candidates.
- Added a CUDA regression test with intentionally inconsistent source depths to
  verify that the backend rejects inconsistent candidates before accepting any
  update.

Why this matters:

- Geometric consistency is no longer only a soft term inside the aggregated
  cost for CUDA refinement. It now also acts as a hard candidate gate, closer to
  COLMAP's repeated geometric consistency refinement direction.
- The filter reduces wasted NCC/cost work on candidates that are already
  inconsistent with available source depth maps.
- This creates a clearer hook for later occlusion/depth-conflict handling.

Still not complete:

- The current gate is a relative-depth consistency filter; it does not yet model
  occlusion ordering or competing-depth conflicts explicitly.
- The diagnostic field names still mention perturbation, but the counters now
  cover both propagation and perturbation candidates.
- CPU PatchMatch already has geometric refinement support, but this exact hard
  pre-cost candidate gate is currently implemented on the CUDA path only.

Verification:

```text
CUDA targeted regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest.CudaBackendRejectsGeometricallyInconsistentCandidates' --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA MVS regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 32

CPU MVS regression:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 32
```

### Milestone 25: CUDA sweep variants, source selection, angle weighting, and pyramid execution

Goal: continue the CUDA PatchMatch COLMAP-ization by filling in the next
geometry/cost organization pieces and removing the single-scale CUDA boundary.

Implemented:

- Pose texture now stores 47 parameters per source and uses 3 layered sweep
  variants:
  - base projection pose,
  - negative patch sweep rotation,
  - positive patch sweep rotation.
- Each pose texture entry now includes:
  - source geometry weight,
  - deterministic source selection probability,
  - patch sweep rotation `cos/sin`.
- CUDA cost evaluation now computes the best cost across rotated patch sweep
  variants for each source view.
- Multi-source aggregation now applies:
  - source-level deterministic selection probability,
  - source geometry weighting,
  - candidate incident-angle penalty based on the hypothesis normal and source
    camera center.
- CUDA PatchMatch now supports multi-scale pyramid execution:
  - each pyramid level builds resized reference/source images,
  - camera intrinsics are scaled per level,
  - source depth maps are resized when geometric consistency is enabled,
  - the previous coarse level depth/normal maps are upsampled as the next
    level's initial hypotheses.
- CUDA single-scale results now run the existing normal-map refinement pass
  when `normal_refinement.enabled` is set, preserving `recon_mvs` summary
  diagnostics under the default `auto` backend.

Still not complete:

- The rotated sweep implementation is a compact patch-coordinate rotation, not
  the full COLMAP selection-probability and sweep scheduler stack.
- Selection probability is source-level and deterministic per pixel/source; it
  is not yet a learned or dense probability map.
- Angle weighting is conservative and currently uses source geometry plus
  incident-angle penalty. Full view-angle / normal-angle filtering remains
  downstream in fusion and future CUDA filtering work.
- The pyramid currently transfers coarse maps through host memory between
  levels. This is simpler and testable, but not yet an optimized all-GPU cache.

Verification:

```text
CUDA build:
cmake --build build-wsl-cuda-mvs-tests --target colite_core_tests -j2
completed successfully

CUDA pyramid test:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest.CudaBackendRunsPyramidStereoWhenCompiled' --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA MVS regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 27
```

CPU regression note:

```text
The CPU build regression command was not run in this turn because WSL command
execution was rejected by the environment usage limit after CUDA verification.
```

### Milestone 26: CUDA normal perturbation and normal-angle diagnostics

Goal: close the remaining gap in the CUDA random perturbation pass by sampling
normal hypotheses on the GPU instead of perturbing depth only.

Implemented:

- Added device-side vector helpers for normalized `float3` arithmetic.
- Added CUDA normal cone sampling around the current hypothesis normal.
- `PerturbationKernel` now evaluates candidate `(depth, normal)` hypotheses.
- Accepted perturbation updates now write both depth and normal maps.
- CUDA perturbation counters now record:
  - total normal-angle update,
  - max normal-angle update,
  - mean normal-angle update.
- Added a CUDA regression test that starts from explicit initial maps with
  zero depth perturbation radius, forcing the perturbation pass to improve via
  normal hypotheses.

Verification:

```text
CUDA normal perturbation test:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest.CudaBackendReportsNormalPerturbationUpdates' --output-on-failure
100% tests passed, 0 tests failed out of 1

CUDA MVS regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 28

CPU MVS regression:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 28
```

### Milestone 27: CUDA per-iteration diagnostics and source selection statistics

Goal: make CUDA PatchMatch tuning less opaque by reporting per-level,
per-iteration counters and source selection probability statistics.

Implemented:

- `PatchMatchStereoSummary` now records CUDA source-selection totals:
  - source selection tests,
  - source selection acceptances,
  - source selection rejections.
- Added per-iteration diagnostic vectors:
  - pyramid level index,
  - iteration index,
  - candidate evaluations,
  - propagation updates,
  - perturbation updates,
  - source selection tests / acceptances / rejections.
- CUDA cost aggregation now counts source-selection decisions at the point
  where each source is actually considered for a candidate hypothesis.
- CUDA single-scale execution now resets/copies counters per iteration and
  fills update ratio, depth radius, normal radius, and the new diagnostic
  vectors.
- CUDA pyramid execution rewrites the per-level iteration diagnostics with the
  real coarse-to-fine level index before accumulating summaries.
- `stereo/patch_match.tsv` now writes the aggregate source selection counters
  and per-iteration diagnostic vectors as comma-separated values.

Verification:

```text
CUDA core/recon_mvs build:
cmake --build build-wsl-cuda-mvs-tests --target colite_core colite_recon_mvs_exe -j2
completed successfully

CPU core/recon_mvs build:
cmake --build build-wsl --target colite_core colite_recon_mvs_exe -j2
completed successfully

PatchMatch test object build:
cmake --build build-wsl-cuda-mvs-tests --target tests/CMakeFiles/colite_core_tests.dir/mvs/patch_match_test.cpp.o -j2
completed successfully
```

Full test note:

```text
The full colite_core_tests target is currently blocked by unrelated SfM test
compile errors in incremental_mapper_test.cpp against IncrementalMappingOptions.
The CUDA diagnostics implementation and recon_mvs target compile successfully.
```

### Milestone 28: dense CUDA selection probability maps and explicit red-black propagation

Goal: replace the source-level selection gate with an optional dense
reference-view probability texture, and make CUDA checkerboard propagation use
explicit red/black parity updates.

Implemented:

- `CudaPatchMatchRequest` now accepts optional per-source dense selection
  probability maps on the reference image grid.
- The CUDA backend validates, clamps, packs, and uploads those maps as a
  layered texture.
- Source selection now samples the dense probability texture when present, and
  falls back to the source-level pose texture probability otherwise.
- Pyramid execution resizes dense selection probability maps per level.
- Added `CheckerboardPropagationKernel`, which updates only one red/black
  parity and evaluates four-neighbor propagation candidates.
- CUDA `kCheckerboard` now runs the same 8-pass parity sequence used by the CPU
  implementation: `0,1,1,0,0,1,1,0`.

Verification:

```text
CUDA dense selection / red-black tests:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest.CudaBackendUsesDenseSelectionProbabilityMap|PatchMatchBackendTest.CudaBackendRunsExplicitRedBlackCheckerboard' --output-on-failure
100% tests passed, 0 tests failed out of 2

CUDA MVS regression:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 31

CPU MVS regression:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 31
```

### Milestone 23: CUDA layered texture image cache

Goal: move the first CUDA backend closer to COLMAP's image access structure by
using CUDA texture objects for reference images, source images, and optional
source depth maps.

Implemented:

- Added an internal `FloatLayeredTexture` RAII wrapper in
  `patch_match_cuda.cu`.
- Reference image, packed source images, and optional source depth maps are now
  uploaded to CUDA layered arrays.
- Cost, propagation, and perturbation kernels now sample through
  `cudaTextureObject_t` with linear filtering and border addressing.
- The old manual bilinear sampling path for CUDA image/depth buffers was
  removed.

Why this matters:

- This matches COLMAP's `CudaArrayLayeredTexture` direction more closely.
- Texture sampling centralizes interpolation and prepares the backend for
  COLMAP-style rotated sweeps, source-depth consistency, and texture-backed
  pose/source data in later milestones.

Still not complete:

- Pose data is still passed through a device struct array, not a pose texture.
- The CUDA backend still supports only single-scale execution.
- Selection probability, rotation sweep scheduling, and full normal
  perturbation are still pending.

Verification:

```text
CUDA build:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 26

CPU build:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 26
```

### Milestone 37: Defer redundant initial photometric stats

Goal: reduce CPU-side overhead in the CUDA PatchMatch scheduling path without
changing final MVS outputs.

Implemented:

- When `--patch_match_geometric_refinement_iterations > 0` and geometric cost is
  enabled, initial PatchMatch skips the immediate `ComputePatchMatchPhotometricStats`
  scans.
- The refinement pass still recomputes final photometric stats and applies the
  photometric filter for every result, so `stereo/patch_match.tsv` remains based
  on post-refinement maps.
- `summary.txt` now includes
  `patch_match_initial_deferred_photometric_stats_images`.

building-18 CUDA profiling:

```text
baseline output: build-wsl/mvs-building-18/cuda-overlap-thread1-64-ms37c
current  output: build-wsl/mvs-building-18/cuda-overlap-thread1-64-ms38

thread1 baseline: initial_wall=560.480585 ms, initial_cuda_sum=153.011242 ms, fused_points=155
thread1 current : initial_wall=476.695071 ms, initial_cuda_sum=151.284996 ms, deferred=18, fused_points=156

baseline output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms37c
current  output: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms38

thread2 baseline: initial_wall=432.277589 ms, initial_cuda_sum=171.995448 ms, refine_overlap_ratio=0.485008, fused_points=156
thread2 current : initial_wall=388.363980 ms, initial_cuda_sum=165.981279 ms, deferred=18, refine_overlap_ratio=0.479275, fused_points=155
```

Verification:

```text
cmake --build build-wsl-cuda-mvs-tests --target colite_core_tests -j2
passed

ctest --test-dir build-wsl-cuda-mvs-tests -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed

ctest --test-dir build-wsl-cuda-mvs-tests -R PatchMatchBackendTest --output-on-failure
21/21 passed, 1 skipped

cmake --build build-wsl --target colite_core_tests -j2
passed

ctest --test-dir build-wsl -R ReconMvsExeTest.RunsPatchMatchStereoWhenRequested --output-on-failure
1/1 passed
```

### Milestone 41: CUDA result materialization breakdown

Goal: reduce and explain `cuda_result_materialization_time_ms` after explicit
warmup removed first-use CUDA runtime overhead from per-image timings.

Implemented:

- Removed the packed-output `ToVector()` copy. CUDA result materialization now
  reads directly from `PinnedHostBuffer<DeviceOutputPixel>` after the stream has
  synchronized.
- Added sub-timers:
  - `cuda_output_to_vector_time_ms`
  - `cuda_dense_map_fill_time_ms`
  - `cuda_normal_refinement_time_ms`
  - `cuda_valid_cost_count_time_ms`

building-18 CUDA profiling:

```text
before sub-timers: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms45
after sub-timers : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms46

ms45 initial:
cuda_result_materialization_time_ms=88.445675 ms
cuda_output_to_vector_time_ms=0.000402 ms
cuda_dense_map_fill_time_ms=58.126664 ms

ms46 initial:
cuda_result_materialization_time_ms=86.848577 ms
cuda_output_to_vector_time_ms=0.000408 ms
cuda_dense_map_fill_time_ms=56.471037 ms
cuda_normal_refinement_time_ms=29.741048 ms
cuda_valid_cost_count_time_ms=0.554789 ms

ms46 final/refinement:
cuda_result_materialization_time_ms=84.490515 ms
cuda_output_to_vector_time_ms=0.000396 ms
cuda_dense_map_fill_time_ms=56.587363 ms
cuda_normal_refinement_time_ms=27.263273 ms
cuda_valid_cost_count_time_ms=0.559550 ms
fused_points=156
```

Conclusion:

- The extra packed-output vector copy has been removed and is now effectively
  zero in diagnostics.
- The remaining CPU-side materialization is dominated by dense map fill and CPU
  normal refinement, not by D2H staging conversion.
- Next candidates:
  - add mutable/bulk dense-map storage APIs to avoid repeated `Set` /
    `SetNormalized` overhead,
  - move normal refinement to CUDA,
  - skip initial normal refinement when geometric refinement will run and
    recompute/refilter the maps.

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

### Milestone 40: Explicit CUDA warmup

Goal: remove first-use CUDA runtime initialization from per-image PatchMatch
diagnostics.

Implemented:

- Added `WarmupCudaPatchMatch`.
- Warmup validates cached device count, applies the requested GPU index when
  needed, and calls `cudaFree(nullptr)` to initialize the CUDA runtime/context.
- `summary.txt` now includes `cuda_warmup_time_ms`.
- `recon_mvs` runs warmup before initial PatchMatch for `cuda` and `auto`
  backends. Forced CUDA propagates warmup failure; auto mode keeps CPU fallback.

building-18 CUDA profiling:

```text
before warmup: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms43
after warmup : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms44

ms43:
initial_wall=447.090526 ms
initial_cuda_sum=200.449025 ms
stereo_run_sum=888.002476 ms
wrapper_overhead_sum=687.553451 ms
initial cuda_request_validation_time_ms=403.682959 ms
fused_points=155

ms44:
cuda_warmup_time_ms=345.080609 ms
initial_wall=90.707164 ms
initial_cuda_sum=173.398549 ms
stereo_run_sum=174.600051 ms
wrapper_overhead_sum=1.201502 ms
initial cuda_request_validation_time_ms=0.017520 ms
initial cuda_device_count_queries=0
initial cuda_device_count_cache_hits=18
fused_points=156
```

Conclusion:

- The remaining large validation cost was the CUDA runtime's first-use
  initialization, not PatchMatch-specific work.
- After explicit warmup, initial per-image wrapper overhead is effectively
  removed from the scheduling diagnostics.
- The next performance target is result materialization:
  `cuda_result_materialization_time_ms` still sums to roughly 80-90 ms for this
  18-image, 64px test. Candidate optimizations are direct dense-map
  materialization from pinned output, reduced CPU copies, and optional deferred
  materialization when only diagnostics or fusion need a subset.

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

### Milestone 39: Initial CUDA wrapper diagnostics and device-count cache

Goal: explain the remaining initial PatchMatch wrapper overhead and remove
repeated CUDA device-count queries from per-image execution.

Implemented:

- Added CUDA wrapper sub-timers to `PatchMatchStereoSummary` and TSV output:
  - `cuda_wrapper_total_time_ms`
  - `cuda_request_validation_time_ms`
  - `cuda_pyramid_resize_time_ms`
  - `cuda_host_pack_time_ms`
  - `cuda_single_scale_dispatch_time_ms`
  - `cuda_result_materialization_time_ms`
- Added `stereo/patch_match_initial.tsv`, written before geometric refinement
  mutates/replaces the initial results.
- Added a process-local CUDA device-count cache based on `std::call_once`.
- Added device-count diagnostics:
  - `cuda_device_count_queries`
  - `cuda_device_count_cache_hits`

building-18 CUDA profiling:

```text
before cache: build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms42
after cache : build-wsl/mvs-building-18/cuda-overlap-thread2-64-ms43

ms42 summary:
initial_wall=453.418734 ms
initial_cuda_sum=183.762562 ms
stereo_run_sum=903.350113 ms
wrapper_overhead_sum=719.587551 ms

ms42 patch_match_initial.tsv:
cuda_request_validation_time_ms=441.954736 ms
cuda_wrapper_total_time_ms=902.324954 ms
cuda_wrapper_overhead_time_ms=718.562392 ms

ms43 summary:
initial_wall=447.090526 ms
initial_cuda_sum=200.449025 ms
stereo_run_sum=888.002476 ms
wrapper_overhead_sum=687.553451 ms
fused_points=155

ms43 patch_match_initial.tsv:
cuda_device_count_queries=1
cuda_device_count_cache_hits=17
cuda_request_validation_time_ms=403.682959 ms
cuda_wrapper_total_time_ms=886.850895 ms
cuda_wrapper_overhead_time_ms=686.401870 ms
cuda_host_pack_time_ms=5.436191 ms
cuda_result_materialization_time_ms=86.388447 ms

ms43 final patch_match.tsv:
cuda_device_count_queries=0
cuda_device_count_cache_hits=18
cuda_request_validation_time_ms=0.018279 ms
```

Conclusion:

- Initial and final/refinement CUDA diagnostics must be inspected separately.
- Device-count caching works: initial stage performs one runtime query and all
  later image calls hit the cache; refinement performs only cache hits.
- The large remaining initial validation time is the first CUDA runtime/device
  initialization cost. The next milestone should add explicit CUDA warmup and
  report `cuda_warmup_time_ms` separately from per-image PatchMatch timings.

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

### Milestone 38: Initial PatchMatch timing breakdown

Goal: identify where the remaining initial CUDA PatchMatch wall time is spent
after deferring redundant photometric stats.

Implemented:

- Added per-initial-job timing for source setup, full stereo call, and
  postprocess.
- Added the derived wrapper/CPU overhead field:
  `patch_match_initial_stereo_wrapper_overhead_sum_time_ms`.
- Wrote all timing fields into `summary.txt`.

New fields:

```text
patch_match_initial_source_setup_sum_time_ms
patch_match_initial_stereo_run_sum_time_ms
patch_match_initial_stereo_wrapper_overhead_sum_time_ms
patch_match_initial_postprocess_sum_time_ms
```

building-18 CUDA profiling:

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

Conclusion:

- Source vector assembly and postprocess/stat filtering are effectively noise at
  this resolution.
- The next optimization target is inside the PatchMatch CUDA call wrapper, not
  in `ReconMvsController` source setup.
- `stereo/patch_match.tsv` currently reflects the final/refinement results, so
  initial-stage CUDA phase sums should be exported separately before using TSV
  phase sums to explain initial wall time.

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

### Milestone 24: CUDA pose texture and COLMAP-like geometry cost

Goal: move CUDA PatchMatch pose/source geometry organization closer to COLMAP by
storing per-source camera geometry in a texture-backed layout instead of a
device struct array.

Implemented:

- Added a point-sampled layered pose texture in `patch_match_cuda.cu`.
- The pose texture packs 43 float parameters per source view:
  - intrinsics `K`: `fx, fy, cx, cy`,
  - relative reference-to-source rotation `R`,
  - relative translation `T`,
  - source camera center `C`,
  - projection matrix `P = K * [R | T]`,
  - inverse projection block `invP = [R^T * K^-1 | -R^T * T]`.
- CUDA kernels now read source intrinsics and relative pose through
  `cudaTextureObject_t` rather than through the old `DeviceSourceView` array.
- Source image metadata is reduced to image dimensions only, while all
  source transform information lives in the pose texture.
- The geometric consistency penalty now uses a more COLMAP-like
  forward/backward depth check:
  1. project the reference hypothesis into a source view with `P`,
  2. sample the source depth map,
  3. back-project the source depth through `invP`,
  4. compare the returned reference-camera depth against the current
     hypothesis depth.

Why this matters:

- Pose access is now organized like a compact texture cache, which prepares the
  CUDA path for rotated sweeps, probability maps, and source-dependent geometry
  weighting.
- Geometry cost no longer compares source depth only in source-camera `z`.
  It now measures source consistency after returning to the reference camera,
  which is closer to COLMAP's PatchMatch geometric consistency formulation.

Still not complete:

- The pose texture currently stores only the base orientation; rotated sweep
  transforms are not packed yet.
- Selection probability maps, incident/view-angle weighting, and full
  COLMAP-style geometric filtering remain future work.
- CUDA PatchMatch still supports only single-scale execution; pyramid execution
  continues to fall back to CPU in `auto` mode.

Verification:

```text
CUDA build:
ctest --test-dir build-wsl-cuda-mvs-tests -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 26

CPU build:
ctest --test-dir build-wsl -R 'PatchMatchBackendTest|CpuPatchMatchStereoTest|ReconMvsExeTest|MvsSweepScriptTest' --output-on-failure
100% tests passed, 0 tests failed out of 26
```
