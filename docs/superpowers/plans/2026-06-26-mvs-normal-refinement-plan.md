# MVS Normal Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve MVS normal-map quality by refining PatchMatch normals from the local depth surface before stereo fusion.

**Architecture:** Add a focused depth-to-normal refinement pass in `patch_match`, expose it through `PatchMatchStereoOptions`, and wire it into `recon_mvs` with conservative CLI controls. The pass estimates camera-space normals from left/right/up/down depth neighbors and only updates pixels whose local depth neighborhood is complete and not a sharp discontinuity.

**Tech Stack:** C++17, Eigen, existing COLITE MVS dense maps and PatchMatch pipeline, GoogleTest, CMake/CTest under WSL.

---

### Task 1: Add Depth-Based Normal Refinement API

**Files:**
- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Test: `tests/mvs/patch_match_test.cpp`

- [ ] Write failing tests:
  - `CpuPatchMatchNormalRefinementTest.EstimatesNormalFromTiltedDepthPlane`
  - `CpuPatchMatchNormalRefinementTest.KeepsOriginalNormalAcrossDepthDiscontinuity`
- [ ] Run:
  `ctest --test-dir build-wsl -R CpuPatchMatchNormalRefinementTest --output-on-failure`
  Expected before implementation: build or test failure because the API does not exist.
- [ ] Add `PatchMatchNormalRefinementOptions`, `PatchMatchNormalRefinementSummary`, and `RefineNormalMapFromDepth`.
- [ ] Compute local 3D points from `(col - 1,row)`, `(col + 1,row)`, `(col,row - 1)`, `(col,row + 1)` and cross their finite differences.
- [ ] Reject refinement when any neighbor is invalid or depth differs from the center by more than `max_relative_depth_difference`.
- [ ] Orient the refined normal to the same hemisphere as the existing normal before writing it.
- [ ] Run the new tests until they pass.

### Task 2: Integrate Into PatchMatch Stereo

**Files:**
- Modify: `include/colite/mvs/patch_match.h`
- Modify: `src/colite/mvs/patch_match.cpp`
- Test: `tests/mvs/patch_match_test.cpp`

- [ ] Add normal-refinement options to `PatchMatchStereoOptions`.
- [ ] Call `RefineNormalMapFromDepth` near the end of `RunPatchMatchStereoForImageImpl`, after PatchMatch propagation/refinement and before returning maps to app-level filters/fusion.
- [ ] Store refinement counts in `PatchMatchStereoSummary`.
- [ ] Add a stereo-level test proving the option changes at least one normal on a tilted synthetic depth surface.
- [ ] Run targeted PatchMatch tests.

### Task 3: Add recon_mvs CLI Controls

**Files:**
- Modify: `app/recon_mvs/recon_mvs_app.h`
- Modify: `app/recon_mvs/recon_mvs_app.cpp`
- Test: `tests/integration/recon_mvs_app_test.cpp`

- [ ] Add `--refine_normals_from_depth`.
- [ ] Add `--normal_refinement_max_relative_depth_difference <r>`.
- [ ] Include refinement summary fields in `summary.txt` and `stereo/patch_match.tsv`.
- [ ] Add help/validation tests.
- [ ] Run recon_mvs app tests.

### Task 4: Verify And Compare

**Files:**
- Modify: `docs/mvs_experiments.md`

- [ ] Run `cmake --build build-wsl --target colite_core_tests -j2`.
- [ ] Run `ctest --test-dir build-wsl --output-on-failure`.
- [ ] Run building-18 with the prior 32px smoke parameters plus `--refine_normals_from_depth`.
- [ ] Compare `multi_observation_track_ratio`, `num_rejected_normal`, and `num_rejected_depth` against `colite-ms11-normal-propagation-32`.
- [ ] Record the result in `docs/mvs_experiments.md`.
