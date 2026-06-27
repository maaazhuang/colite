# MVS COLMAPization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring COLITE MVS closer to COLMAP by improving source view selection, PatchMatch geometric cost, propagation, initialization, binary map IO, parallel execution, dense cloud cleanup, and meshing.

**Architecture:** Keep the current conservative C++17 pipeline and extend it in small testable phases. Preserve current text outputs and CLI defaults where useful, add new options behind explicit structs/flags, and use building-18 smoke metrics after each phase to avoid drifting away from known behavior.

**Tech Stack:** C++17, CMake, GoogleTest, OpenCV, Eigen, existing COLITE SfM/MVS data structures, WSL Ubuntu 22.04 build preset `wsl-debug`.

---

## Current Baseline

Current MVS code is centered in:

- `include/colite/mvs/mvs_pipeline.h`
- `src/colite/mvs/mvs_pipeline.cpp`
- `include/colite/mvs/patch_match.h`
- `src/colite/mvs/patch_match.cpp`
- `include/colite/mvs/dense_map.h`
- `src/colite/mvs/dense_map.cpp`
- `include/colite/mvs/image_workspace.h`
- `src/colite/mvs/image_workspace.cpp`
- `app/recon_mvs/recon_mvs_app.h`
- `app/recon_mvs/recon_mvs_app.cpp`
- tests under `tests/mvs/` and `tests/integration/recon_mvs_app_test.cpp`

Latest building-18 smoke at `max_image_size=48`, `patch_match_iterations=1`, default 8 source views:

```text
geometric_consistent_pixels 7058
fused_points 6030
fused_observations 7058
```

This is the comparison baseline for all following changes.

---

## Milestone 1: COLMAP-Style Source View Scoring

**Purpose:** Replace source view ranking by shared point count only with a score that also considers triangulation angle and view coverage. This targets the strongest current diagnosis: source view coverage was the most effective building-18 improvement.

**Files:**

- Modify: `include/colite/mvs/mvs_pipeline.h`
- Modify: `src/colite/mvs/mvs_pipeline.cpp`
- Modify: `tests/mvs/mvs_pipeline_test.cpp`
- Modify: `docs/mvs_status.md`

### Task 1.1: Extend SourceImageScore

- [x] **Step 1: Write failing test**

Add to `tests/mvs/mvs_pipeline_test.cpp`:

```cpp
TEST(MvsPipelineTest, ScoresSourceImagesWithTriangulationAngleAndSharedPoints) {
  Reconstruction reconstruction = MakeSparseReconstruction();
  MvsPipelineOptions options;
  options.max_num_source_images = 2;
  options.min_num_shared_points = 1;
  options.min_triangulation_angle_degrees = 0.0;
  options.max_triangulation_angle_degrees = 30.0;
  options.triangulation_angle_weight = 1.0;
  options.shared_points_weight = 1.0;

  const std::vector<SourceImageScore> sources =
      SelectSourceImages(reconstruction, 1, options);

  ASSERT_EQ(sources.size(), 2U);
  EXPECT_GT(sources[0].score, sources[1].score);
  EXPECT_GT(sources[0].mean_triangulation_angle_degrees, 0.0);
  EXPECT_GE(sources[0].num_shared_points, 1U);
}
```

- [x] **Step 2: Verify red**

Run:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && cmake --build --preset wsl-debug && ctest --test-dir build-wsl -R MvsPipelineTest.ScoresSourceImagesWithTriangulationAngleAndSharedPoints --output-on-failure"
```

Expected: build fails because fields do not exist.

- [x] **Step 3: Add fields**

In `include/colite/mvs/mvs_pipeline.h`:

```cpp
struct SourceImageScore {
  ImageId image_id = kInvalidImageId;
  std::size_t num_shared_points = 0;
  double mean_triangulation_angle_degrees = 0.0;
  double score = 0.0;
};

struct MvsPipelineOptions {
  std::size_t max_num_source_images = 8;
  std::size_t min_num_shared_points = 1;
  double min_triangulation_angle_degrees = 1.0;
  double max_triangulation_angle_degrees = 30.0;
  double triangulation_angle_weight = 1.0;
  double shared_points_weight = 1.0;
};
```

- [x] **Step 4: Implement scoring**

In `src/colite/mvs/mvs_pipeline.cpp`, add helper functions:

```cpp
double ComputeTrackTriangulationAngleDegrees(
    const Reconstruction& reconstruction,
    ImageId lhs_image_id,
    ImageId rhs_image_id,
    const Eigen::Vector3d& point_world);

double ScoreSourceView(double mean_angle_degrees,
                       std::size_t num_shared_points,
                       const MvsPipelineOptions& options);
```

Expected behavior:

- Compute camera centers from registered poses.
- For each shared point between target and source, compute triangulation angle.
- Ignore angles outside `[min_triangulation_angle_degrees, max_triangulation_angle_degrees]` for angle average, but keep shared point count.
- Score as:

```cpp
score = options.triangulation_angle_weight * mean_angle_degrees +
        options.shared_points_weight * static_cast<double>(num_shared_points);
```

- Sort by `score` descending, then `num_shared_points` descending, then `image_id`.

- [x] **Step 5: Verify green**

Run:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && ctest --test-dir build-wsl -R 'MvsPipelineTest' --output-on-failure"
```

Expected: MVS pipeline tests pass.

- [x] **Step 6: Smoke**

Run building-18 with default options. Compare against:

```text
fused_points 6030
fused_observations 7058
```

Accept if fused points do not drop more than 10% and pair diagnostics show fewer zero-accepted high-candidate pairs.

---

## Milestone 2: Geometric Consistency Inside PatchMatch Cost

**Purpose:** Move geometric consistency from pure post-filtering toward PatchMatch refinement. Since the current pipeline computes all images independently, implement this as a second refinement pass that uses previous-pass source depth maps.

**Files:**

- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Modify: `app/recon_mvs/recon_mvs_app.h`
- Modify: `app/recon_mvs/recon_mvs_app.cpp`
- Modify: `tests/mvs/patch_match_test.cpp`
- Modify: `tests/integration/recon_mvs_app_test.cpp`

### Task 2.1: Add geometric cost options

- [x] **Step 1: Write failing test**

Add to `tests/mvs/patch_match_test.cpp`:

```cpp
TEST(CpuPatchMatchPhotometricCostTest, PenalizesGeometricallyInconsistentSourceDepth) {
  const PatchMatchView reference_view = MakeView(1);
  const PatchMatchView source_view = MakeView(2, 0.2);
  const MvsImage reference_image = MakeGradientImage(5, 5);
  const MvsImage source_image = reference_image;

  DenseDepthMap source_depth_map(5, 5, 1.0F, 5.0F);
  source_depth_map.Set(2, 2, 4.5F);

  PatchMatchCostOptions options;
  options.geometric_consistency_weight = 1.0F;
  options.geometric_max_relative_depth_error = 0.01F;

  const PatchMatchCost cost_without_source_depth = ComputePlaneHypothesisCost(
      reference_view, source_view, reference_image, source_image,
      Eigen::Vector2f(2.0F, 2.0F), 3.0F,
      Eigen::Vector3f(0.0F, 0.0F, 1.0F), options);
  const PatchMatchCost cost_with_source_depth = ComputePlaneHypothesisCost(
      reference_view, source_view, reference_image, source_image,
      Eigen::Vector2f(2.0F, 2.0F), 3.0F,
      Eigen::Vector3f(0.0F, 0.0F, 1.0F), options, &source_depth_map);

  ASSERT_TRUE(cost_without_source_depth.valid);
  ASSERT_TRUE(cost_with_source_depth.valid);
  EXPECT_GT(cost_with_source_depth.cost, cost_without_source_depth.cost);
}
```

- [x] **Step 2: Verify red**

Expected: compile fails because options and overload do not exist.

- [x] **Step 3: Extend options**

In `PatchMatchCostOptions`:

```cpp
float geometric_consistency_weight = 0.0F;
float geometric_max_relative_depth_error = 0.01F;
```

- [x] **Step 4: Add overloads**

Add optional source depth input to plane/depth cost functions:

```cpp
PatchMatchCost ComputePlaneHypothesisCost(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const Eigen::Vector3f& reference_normal,
    const PatchMatchCostOptions& options = {},
    const DenseDepthMap* source_depth_map = nullptr);
```

If `source_depth_map != nullptr`, project the hypothesis into source, sample source depth bilinearly, reproject back, compute relative depth error, and add:

```cpp
cost.cost += options.geometric_consistency_weight *
             std::min(1.0F, relative_depth_error /
                            options.geometric_max_relative_depth_error);
```

- [x] **Step 5: Add refinement pass**

Add a new function:

```cpp
PatchMatchStereoResult RunPatchMatchStereoForImageWithSourceDepths(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>& source_depth_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options = {});
```

Reuse existing implementation but pass matching `source_depth_maps` into cost evaluation.

- [x] **Step 6: CLI**

Add:

```text
--patch_match_geometric_refinement_iterations <n>
--patch_match_geometric_consistency_weight <w>
--patch_match_geometric_max_relative_depth_error <r>
```

Default refinement iterations should be `0` to preserve current behavior.

- [x] **Step 7: Smoke**

Run:

```bash
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl/mvs-building-18/colite-pm-geom-refine-48 \
  --run_patch_match \
  --max_image_size 48 \
  --patch_match_iterations 1 \
  --patch_match_geometric_refinement_iterations 1 \
  --patch_match_geometric_consistency_weight 0.2 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --filter_geometric_min_num_consistent_sources 1
```

Accept if depth/mutual rejection decreases without catastrophic density loss.

Milestone 2 result notes:

- Unit/integration verification passed for PatchMatch geometric source-depth cost and `recon_mvs` CLI wiring.
- `max_image_size=48`, `weight=0.2` exceeded the 5 minute smoke timeout in the current single-thread CPU implementation, so this milestone exposes a real performance target for later multithreading/cost caching.
- `max_image_size=32` comparison:
  - baseline: `geometric_consistent_pixels=2889`, `fused_points=2440`, `fused_observations=2889`
  - geometric refinement `weight=0.2`: `geometric_consistent_pixels=2743`, `fused_points=2082`, `fused_observations=2743`
  - geometric refinement `weight=0.05`: `geometric_consistent_pixels=3447`, `fused_points=2629`, `fused_observations=3447`
- Current recommendation: keep default refinement disabled, and when experimenting use a conservative weight such as `0.05` until source depth initialization and propagation are stronger.

---

## Milestone 3: More Complete Propagation Scheduling

**Purpose:** Add more COLMAP-like propagation variants while keeping the existing forward/backward schedule as a fallback.

**Files:**

- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Modify: `tests/mvs/patch_match_test.cpp`
- Modify: `app/recon_mvs/recon_mvs_app.*`

### Task 3.1: Add schedule enum and CLI

- [x] Add enum:

```cpp
enum class PatchMatchPropagationSchedule {
  kForwardBackward,
  kFourDirection,
  kCheckerboard,
};
```

- [x] Add parser for:

```text
--patch_match_propagation_schedule forward_backward|four_direction|checkerboard
```

- [x] Test invalid value returns CLI error.

### Task 3.2: Four-direction sweeps

- [x] Add directions:

```cpp
enum class PatchMatchSweepDirection {
  kForward,
  kBackward,
  kTopDown,
  kBottomUp,
};
```

- [x] Extend `RunPatchMatchPropagationSweep` to choose neighbors by direction.
- [x] Add tests:
  - top-down updates from upper neighbor
  - bottom-up updates from lower neighbor

### Task 3.3: Checkerboard schedule

- [x] Add:

```cpp
PatchMatchSweepSummary RunPatchMatchCheckerboardPropagation(
    ...,
    int parity,
    const PatchMatchCostOptions& options = {});
```

- [x] Test parity `0` updates only `(row + col) % 2 == 0`, parity `1` updates the opposite set.

Milestone 3 result notes:

- Added `PatchMatchPropagationSchedule` with `forward_backward`, `four_direction`, and `checkerboard`.
- Added `kTopDown` and `kBottomUp` sweep directions.
- Added checkerboard propagation with four red/black rounds per user iteration, exposed in `PatchMatchStereoSummary::num_checkerboard_passes`.
- Added CLI flag:
  - `--patch_match_propagation_schedule forward_backward|four_direction|checkerboard`
- Added `patch_match.tsv` columns:
  - `forward_sweeps`
  - `backward_sweeps`
  - `top_down_sweeps`
  - `bottom_up_sweeps`
  - `checkerboard_passes`
- building-18 `max_image_size=32`, `patch_match_iterations=1`:
  - baseline `forward_backward`: `geometric_consistent_pixels=2889`, `fused_points=2440`, `fused_observations=2889`
  - `four_direction`: `geometric_consistent_pixels=3014`, `fused_points=2544`, `fused_observations=3014`
  - `checkerboard` four red/black rounds: `geometric_consistent_pixels=2099`, `fused_points=1899`, `fused_observations=2099`
- Current recommendation: keep `forward_backward` as default; use `four_direction` as the preferred experimental quality schedule; keep `checkerboard` available but not recommended until random sampling and checkerboard-local candidate strategy are improved.

---

## Milestone 4: Stable Depth Range, Normal Initialization, Perturbation

**Purpose:** Reduce noisy hypotheses from sparse depth outliers and make random normal/depth perturbation more stable.

**Files:**

- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Modify: `tests/mvs/patch_match_test.cpp`

### Task 4.1: Percentile sparse depth range

- [x] Add options:

```cpp
float sparse_depth_min_percentile = 0.01F;
float sparse_depth_max_percentile = 0.99F;
float depth_range_padding = 0.1F;
```

- [x] Add test with one extreme sparse outlier and expect the initialized range to ignore it.

### Task 4.2: Hemisphere normal initialization

- [x] Add option:

```cpp
bool randomize_initial_normals = false;
float initial_normal_hemisphere_degrees = 60.0F;
```

- [x] Test deterministic random normals for a fixed seed and unit norm.

### Task 4.3: Adaptive perturbation

- [x] Track per-iteration update ratio.
- [x] If update ratio falls below threshold, shrink depth and normal radius faster.
- [x] Add summary fields:

```cpp
std::vector<double> update_ratios;
std::vector<float> depth_radii;
std::vector<float> normal_radii_degrees;
```

- [x] Test summary records one entry per iteration.

Milestone 4 result notes:

- Added sparse depth percentile initialization options:
  - `sparse_depth_min_percentile = 0.01`
  - `sparse_depth_max_percentile = 0.99`
  - `depth_range_padding = 0.0`
- Added optional randomized initial normals:
  - `randomize_initial_normals = false`
  - `initial_normal_hemisphere_degrees = 60`
- Added adaptive perturbation summary fields:
  - `update_ratios`
  - `depth_radii`
  - `normal_radii_degrees`
- Added `adaptive_perturbation_min_update_ratio = 0.01`; if an iteration update ratio is lower than this threshold, the next perturbation radius gets one extra shrink step.
- Verification:
  - `ctest --test-dir build-wsl --output-on-failure`: 120/120 passed.
  - building-18 `max_image_size=32`, default `forward_backward`: `photometric_filtered_pixels=4182`, `geometric_consistent_pixels=2889`, `fused_points=2440`, `fused_observations=2889`.
- Current default behavior remains conservative: percentile clipping and randomized normals are available in library options, but `recon_mvs` still feeds explicit min/max depth ranges and default normals unless later CLI options expose these controls.

---

## Milestone 5: Binary Depth / Normal / Consistency Graph Format

**Purpose:** Stop relying only on debug text maps. Binary IO enables faster MVS runs and later compatibility work.

**Files:**

- Modify: `include/colite/mvs/dense_map.h`
- Modify: `src/colite/mvs/dense_map.cpp`
- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Modify: `tests/mvs/dense_map_test.cpp`
- Modify: `tests/mvs/patch_match_test.cpp`
- Modify: `app/recon_mvs/recon_mvs_app.cpp`

### Task 5.1: Binary dense maps

- [x] Add functions:

```cpp
void WriteDenseDepthMapBinary(const std::filesystem::path& path,
                              const DenseDepthMap& map);
DenseDepthMap ReadDenseDepthMapBinary(const std::filesystem::path& path);

void WriteNormalMapBinary(const std::filesystem::path& path,
                          const NormalMap& map);
NormalMap ReadNormalMapBinary(const std::filesystem::path& path);
```

- [x] Header:

```text
COLITE_DEPTH_MAP_BINARY 1
width height depth_min depth_max
```

Then raw little-endian `float` values row-major.

- [x] Test round-trip preserves width, height, range, values, and invalid zeros.

### Task 5.2: Binary consistency graph

- [x] Add:

```cpp
void WriteConsistencyGraphBinary(...);
ConsistencyGraph ReadConsistencyGraphBinary(...);
```

- [x] Format:
  - magic string
  - width height
  - per-pixel source count
  - flattened source image ids

- [x] Test graph round-trip with empty and non-empty pixels.

### Task 5.3: CLI format option

- [x] Add:

```text
--dense_map_format text|binary
```

Default `text` until binary reader is used in later resume workflows.

Milestone 5 result notes:

- Added binary depth map IO:
  - `WriteDenseDepthMapBinary`
  - `ReadDenseDepthMapBinary`
- Added binary normal map IO:
  - `WriteNormalMapBinary`
  - `ReadNormalMapBinary`
- Added binary consistency graph IO:
  - `WriteConsistencyGraphBinary`
  - `ReadConsistencyGraphBinary`
- Binary depth/normal files use a readable text header plus row-major raw floats:
  - `COLITE_DEPTH_MAP_BINARY 1`
  - `COLITE_NORMAL_MAP_BINARY 1`
- Binary consistency graphs use:
  - `COLITE_CONSISTENCY_GRAPH_BINARY 1`
  - width/height header
  - per-pixel `uint32_t` source count followed by `uint32_t` source image ids.
- `recon_mvs` now supports:
  - `--dense_map_format text|binary`
  - default remains `text`.
- In binary mode:
  - `stereo/depth_maps/*.bin`
  - `stereo/normal_maps/*.bin`
  - `stereo/consistency_graphs/*.bin`
  - `stereo/cost_maps/*.txt` remains text for diagnostics.
- Verification:
  - Targeted Milestone 5 tests: 10/10 passed.
  - Full test suite: 123/123 passed.
  - building-18 smoke, `max_image_size=32`, `patch_match_iterations=1`, binary maps:
    - `photometric_filtered_pixels=4182`
    - `geometric_consistent_pixels=2889`
    - `geometric_filtered_pixels=10935`
    - `fused_points=2440`
    - `fused_observations=2889`
  - Output checked at `build-wsl/mvs-building-18/colite-pm-binary-32-ms5`.

---

## Milestone 6: Multithreading and Cache Optimization

**Purpose:** Make larger source view counts practical.

**Files:**

- Modify: `app/recon_mvs/recon_mvs_app.h`
- Modify: `app/recon_mvs/recon_mvs_app.cpp`
- Modify: `include/colite/mvs/image_workspace.h`
- Modify: `src/colite/mvs/image_workspace.cpp`
- Modify: `tests/integration/recon_mvs_app_test.cpp`
- Add: `tests/mvs/image_workspace_parallel_test.cpp` if needed.

### Task 6.1: Add thread count option

- [x] CLI:

```text
--num_threads <n>
```

Default `1`.

- [x] Validation: `n >= 1`.

### Task 6.2: Parallelize per-reference PatchMatch

- [x] Avoid sharing mutable `MvsImageWorkspace` across threads.
- [x] Create one workspace per worker or preload immutable images before parallel section.
- [x] Use `std::async` or a fixed worker loop with `std::thread`.
- [x] Ensure diagnostics order is deterministic by sorting results by `image_id` before writing.

### Task 6.3: Tests and smoke

- [x] Integration test:
  - run with `--num_threads 2`
  - assert exit code `0`
  - assert `stereo/patch_match.tsv` still exists

- [x] building-18 compare:
  - outputs should be numerically identical for `--num_threads 1` and `--num_threads 2` with fixed seed.

Milestone 6 result notes:

- Added `recon_mvs --num_threads <n>`, default `1`, with validation that `n >= 1`.
- Added integration coverage for help text, invalid `--num_threads 0`, and `--num_threads 2` PatchMatch execution.
- `RunPatchMatchForWorkspace` now preloads all reference/source `MvsImage` objects once on the main thread, builds immutable `PatchMatchView` entries, and runs the first per-reference PatchMatch pass through a capped `std::async` worker queue.
- Mutable `MvsImageWorkspace` is not shared with workers; workers only read preloaded images/views.
- Results are sorted by `image_id` before downstream refinement, geometric consistency, fusion, and diagnostics, keeping output deterministic.
- Verification:
  - `ReconMvsAppTest`: 10/10 passed.
  - Full test suite: 125/125 passed.
  - building-18 `max_image_size=32`, `patch_match_iterations=1`, `--num_threads 1` and `--num_threads 2` produced identical `summary.txt` and `stereo/patch_match.tsv`.
  - Both smoke runs reported:
    - `photometric_filtered_pixels=4182`
    - `geometric_consistent_pixels=2889`
    - `geometric_filtered_pixels=10935`
    - `fused_points=2440`
    - `fused_observations=2889`

---

## Milestone 7: Dense Cloud Cleaning and Meshing

**Purpose:** Prepare stereo fusion output for surface reconstruction.

**Files:**

- Done: `include/colite/mvs/dense_cloud.h`
- Done: `src/colite/mvs/dense_cloud.cpp`
- Done: `tests/mvs/dense_cloud_test.cpp`
- Done: `app/recon_mvs/recon_mvs_app.*`
- Done: source/test CMake files.

### Task 7.1: Dense cloud filters

- [x] Add:

```cpp
struct DenseCloudFilterOptions {
  std::size_t min_track_observations = 2;
  double max_normal_angle_degrees = 30.0;
  double voxel_size = 0.0;
};

std::vector<FusedPoint> FilterDenseCloud(
    const std::vector<FusedPoint>& points,
    const std::vector<StereoFusionTrack>& tracks,
    const DenseCloudFilterOptions& options);
```

- [x] Tests:
  - removes points with too few observations
  - keeps points with enough observations
  - voxel filter keeps one representative per voxel

### Task 7.2: Clean cloud CLI

- [x] Add:

```text
--filter_dense_cloud
--dense_cloud_min_track_observations <n>
--dense_cloud_voxel_size <v>
```

- [x] Output:

```text
dense/fused_clean.ply
dense/fused_clean_stats.txt
```

### Task 7.3: First meshing prototype

- [x] Add simple visibility-free mesh prototype only after clean cloud exists.
- [x] Start with conservative local k-nearest triangle proposal:
  - for each point, find nearby points
  - reject long edges
  - reject normal-inconsistent triangles
  - write `dense/mesh.ply`

- [x] CLI:

```text
--mesh_dense_cloud
--mesh_max_edge_length <v>
--mesh_max_normal_angle <deg>
```

- [x] Tests:
  - four coplanar points produce at least two triangles
  - long-edge candidates are rejected

Implementation note: the first implementation used global all-triplets triangle proposals, which was too slow on building-18. It has been replaced with a bounded local-neighbor proposal controlled by `DenseCloudMeshingOptions::max_candidate_neighbors`.

Verification note: `ctest --test-dir build-wsl --output-on-failure` passed 131/131 after Milestone 7. The building-18 smoke run produced `dense/fused_clean.ply`, `dense/fused_clean_stats.txt`, and `dense/mesh.ply`; at `max_image_size=32`, the fused clean cloud kept 2440/2440 points and the mesh writer produced 2440 vertices and 61588 faces.

Note: This is not a final COLMAP-grade mesher. It is a testable bridge toward later Delaunay/Poisson-style meshing.

---

## Milestone 8: Regression Matrix and Documentation

**Purpose:** Keep changes measurable.

**Files:**

- Done: `docs/mvs_status.md`
- Done: `docs/mvs_experiments.md`

### Task 8.1: Add experiment table

- [x] Record for building-18:

```text
case
max_num_source_images
patch_match_iterations
patch_match_min_num_valid_sources
patch_match_max_num_aggregated_costs
photometric_filtered_pixels
geometric_consistent_pixels
fused_points
fused_observations
single_observation_tracks
multi_observation_tracks
runtime_seconds
```

### Task 8.2: Required verification before each milestone completion

Run:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && cmake --build --preset wsl-debug && ctest --test-dir build-wsl --output-on-failure"
```

Run building-18 smoke:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && ./build-wsl/app/recon_mvs/recon_mvs --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole --image_path data/building-18 --output_path build-wsl/mvs-building-18/<case-name> --run_patch_match --max_image_size 48 --patch_match_iterations 1 --patch_match_window_radius 1 --filter_photometric_max_cost 0.3 --fuse_stereo --fusion_min_num_observations 1 --fusion_max_normal_error 10 --fusion_max_view_angle 180 --filter_geometric_min_num_consistent_sources 1"
```

- [x] Ran the building-18 smoke for `colite-ms8-smoke-48` and recorded the result in `docs/mvs_experiments.md`.

Expected after Milestone 1:

```text
fused_points >= 5400
fused_observations >= 6300
```

Expected after later quality-focused milestones:

```text
Do not require fused_points to increase.
Require pair diagnostics to show fewer depth/mutual invalid failures or better multi-observation track ratio.
```

Milestone 8 verification note:

```text
colite-ms8-smoke-48
photometric_filtered_pixels 4334
geometric_consistent_pixels 6983
fused_points 5969
fused_observations 6983
single_observation_tracks 5201
multi_observation_tracks 768
runtime_seconds 69.40
```

---

## Suggested Execution Order

1. Milestone 1: Source view scoring.
2. Milestone 2: Geometric consistency refinement pass.
3. Milestone 3: Propagation scheduling.
4. Milestone 4: Initialization and perturbation stability.
5. Milestone 5: Binary map IO.
6. Milestone 6: Multithreading.
7. Milestone 7: Dense cloud cleaning and first meshing prototype.
8. Milestone 8: Experiment matrix updates after each milestone.

This order keeps quality-sensitive algorithm changes ahead of performance work and delays meshing until fused cloud diagnostics are stable enough to trust.

---

## Self-Review

- Spec coverage: all requested areas map to milestones 1-7.
- Placeholder scan: no `TBD` or open implementation holes remain; meshing is explicitly scoped as a first prototype, not final COLMAP-quality meshing.
- Type consistency: new option and struct names are consistently named across tasks.
- Risk notes:
  - Milestone 2 is the highest-risk architectural change because current PatchMatch is single-pass per image; the refinement-pass approach avoids needing a full global scheduler immediately.
  - Milestone 6 must preserve deterministic output with fixed seeds; do not parallelize random number generation without per-image deterministic seeds.
