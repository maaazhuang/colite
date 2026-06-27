# MVS Quality Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic MVS quality diagnostics so PatchMatch, stereo fusion, dense cleaning, and meshing can be compared without manual post-processing.

**Architecture:** Keep diagnostics in the existing MVS modules and `recon_mvs` output tree. Add small pure functions for fusion track summaries and mesh summaries, then write text files from `recon_mvs` beside the existing TSV/PLY artifacts. Preserve existing defaults and avoid changing reconstruction behavior.

**Tech Stack:** C++17, CMake, GoogleTest, Eigen, existing COLITE MVS data structures, WSL Ubuntu 22.04 preset `wsl-debug`.

---

## Scope

This plan implements the first quality-diagnostics slice, not a new meshing backend and not PatchMatch algorithm tuning. The output metrics will guide the next optimization phase.

## Files

- Modify: `include/colite/mvs/dense_cloud.h`
- Modify: `src/colite/mvs/dense_cloud.cpp`
- Modify: `app/recon_mvs/recon_mvs_app.cpp`
- Modify: `tests/mvs/dense_cloud_test.cpp`
- Modify: `tests/integration/recon_mvs_app_test.cpp`
- Modify: `docs/mvs_experiments.md`

---

## Task 1: Fusion Track Quality Summary

**Purpose:** Automatically summarize fusion track quality after `--fuse_stereo`.

- [x] Add `DenseFusionTrackQualityStats` to `include/colite/mvs/dense_cloud.h`:

```cpp
struct DenseFusionTrackQualityStats {
  std::size_t num_tracks = 0;
  std::size_t num_single_observation_tracks = 0;
  std::size_t num_multi_observation_tracks = 0;
  std::size_t num_observations = 0;
  std::size_t max_observations = 0;
  double mean_observations = 0.0;
  double multi_observation_track_ratio = 0.0;
  double mean_max_normal_angle_degrees = 0.0;
  double max_normal_angle_degrees = 0.0;
  double mean_max_view_angle_degrees = 0.0;
  double max_view_angle_degrees = 0.0;
};
```

- [x] Add:

```cpp
DenseFusionTrackQualityStats ComputeFusionTrackQualityStats(
    const std::vector<StereoFusionTrack>& tracks);

void WriteFusionTrackQualityStats(
    const std::filesystem::path& path,
    const DenseFusionTrackQualityStats& stats);
```

- [x] Test in `tests/mvs/dense_cloud_test.cpp`:

```cpp
TEST(DenseFusionTrackQualityTest, SummarizesSingleAndMultiObservationTracks)
```

Expected behavior: for tracks with observation counts `1, 2, 4`, stats report 3 tracks, 1 single-observation track, 2 multi-observation tracks, 7 observations, mean observations `7/3`, multi ratio `2/3`, and max observations 4.

- [x] Integration test: `ReconMvsAppTest.WritesStereoFusionPlyWhenRequested` asserts `dense/fusion_summary.txt` exists and contains `num_single_observation_tracks`.

---

## Task 2: Mesh Quality Summary

**Purpose:** Automatically summarize mesh size and triangle edge quality after `--mesh_dense_cloud`.

- [x] Add `DenseTriangleMeshStats` to `include/colite/mvs/dense_cloud.h`:

```cpp
struct DenseTriangleMeshStats {
  std::size_t num_vertices = 0;
  std::size_t num_triangles = 0;
  double mean_edge_length = 0.0;
  double max_edge_length = 0.0;
  double mean_triangle_area = 0.0;
  double min_triangle_area = 0.0;
  double max_triangle_area = 0.0;
};
```

- [x] Add:

```cpp
DenseTriangleMeshStats ComputeDenseTriangleMeshStats(
    const DenseTriangleMesh& mesh);

void WriteDenseTriangleMeshStats(const std::filesystem::path& path,
                                 const DenseTriangleMeshStats& stats);
```

- [x] Test in `tests/mvs/dense_cloud_test.cpp`:

```cpp
TEST(DenseCloudMeshingTest, ComputesTriangleMeshStats)
```

Expected behavior: a two-triangle square mesh has 4 vertices, 2 triangles, positive mean/max edge length, and positive area stats.

- [x] Integration test: `ReconMvsAppTest.WritesCleanDenseCloudAndMeshWhenRequested` asserts `dense/mesh_stats.txt` exists and contains `num_triangles`.

---

## Task 3: Documentation and Verification

- [x] Update `docs/mvs_experiments.md` with the new diagnostics files:

```text
dense/fusion_summary.txt
dense/mesh_stats.txt
```

- [x] Run targeted tests:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && cmake --build --preset wsl-debug && ctest --test-dir build-wsl -R 'DenseCloud|ReconMvsAppTest.(WritesStereoFusionPlyWhenRequested|WritesCleanDenseCloudAndMeshWhenRequested)' --output-on-failure"
```

- [x] Run full tests:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && ctest --test-dir build-wsl --output-on-failure"
```

- [x] Run a small building-18 smoke:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && ./build-wsl/app/recon_mvs/recon_mvs --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole --image_path data/building-18 --output_path build-wsl/mvs-building-18/colite-ms9-diagnostics-32 --run_patch_match --max_image_size 32 --patch_match_iterations 1 --patch_match_window_radius 1 --filter_photometric_max_cost 0.3 --fuse_stereo --fusion_min_num_observations 1 --fusion_max_normal_error 10 --fusion_max_view_angle 180 --filter_geometric_min_num_consistent_sources 1 --filter_dense_cloud --dense_cloud_min_track_observations 1 --mesh_dense_cloud --mesh_max_edge_length 0.5 --mesh_max_normal_angle 30 --num_threads 2"
```

Expected smoke outputs:

```text
dense/fusion_summary.txt
dense/fused_clean_stats.txt
dense/mesh_stats.txt
```

Observed smoke summary for `colite-ms9-diagnostics-32`:

```text
photometric_filtered_pixels 2491
geometric_consistent_pixels 2437
fused_points 2152
fused_observations 2437
num_single_observation_tracks 1926
num_multi_observation_tracks 226
multi_observation_track_ratio 0.1050185873605948
mesh_triangles 47045
mesh_mean_edge_length 0.30100532545381642
```

---

## Self-Review

- Spec coverage: implements the next-step diagnostic foundation before algorithm tuning.
- Placeholder scan: no TBD/TODO placeholders.
- Scope check: intentionally avoids changing PatchMatch or fusion behavior in this slice.
- Testability: pure stats functions have unit tests; `recon_mvs` output has integration coverage.
