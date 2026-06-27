# MVS Fusion Track Ratio Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:systematic-debugging before changing fusion behavior, and superpowers:test-driven-development for each behavior change.

**Goal:** Improve dense fusion multi-observation track ratio without merely filtering out single-observation tracks.

**Architecture:** Use fusion diagnostics to identify the dominant rejection cause, add a narrow COLMAP-like source pixel neighborhood search in stereo fusion, and keep the change configurable through `recon_mvs`.

**Tech Stack:** C++17, GoogleTest, existing COLITE MVS data structures, WSL Ubuntu 22.04 preset `wsl-debug`.

---

## Root Cause Evidence

Baseline case: `build-wsl/mvs-building-18/colite-ms10-fusion-radius0-32`

```text
num_candidates 3435
num_accepted 285
num_rejected_normal 249
num_rejected_depth 353
num_rejected_mutual_consistency 647
num_rejected_invalid 1561
num_rejected_visited 60
num_rejected_duplicate 280
num_single_observation_tracks 1926
num_multi_observation_tracks 226
multi_observation_track_ratio 0.1050185873605948
```

Interpretation: multi-observation tracks were limited primarily by invalid source pixels and mutual/depth checks. The implementation mismatch was that geometric consistency graph construction uses bilinear source depth sampling, while stereo fusion previously tested only the rounded source pixel.

---

## Implemented Change

- [x] Added `StereoFusionOptions::source_pixel_search_radius`, default `1`.
- [x] Added `recon_mvs --fusion_source_pixel_search_radius <n>`, default `1`, validated as `<= 5`.
- [x] Updated fusion source projection to search nearby source pixels by distance from the projected location.
- [x] Kept all existing depth, mutual consistency, normal, and view angle checks for accepted candidates.
- [x] Added reject-reason priority so later less-specific candidates do not overwrite more meaningful diagnostics.
- [x] Added unit test `CpuStereoFusionTest.SearchesNearbySourcePixelWhenRoundedProjectionInvalid`.

---

## Verification

Targeted tests:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && cmake --build --preset wsl-debug && ctest --test-dir build-wsl -R 'CpuStereoFusionTest.(SearchesNearbySourcePixelWhenRoundedProjectionInvalid|RejectsSourceObservationWithLargeNormalAngle)' --output-on-failure"
```

Full tests:

```text
ctest --test-dir build-wsl --output-on-failure
100% tests passed, 0 tests failed out of 134
```

building-18 optimized case: `build-wsl/mvs-building-18/colite-ms10-fusion-radius1-32-fixed`

```text
num_candidates 3435
num_accepted 438
num_rejected_normal 427
num_rejected_depth 1052
num_rejected_mutual_consistency 377
num_rejected_invalid 471
num_rejected_visited 345
num_rejected_duplicate 325
num_single_observation_tracks 1771
num_multi_observation_tracks 228
multi_observation_track_ratio 0.11405702851425713
fused_points 1999
fused_observations 2437
```

The improvement is modest but real: observations remain constant while fused tracks decrease, which means more pixels merge into shared tracks. Invalid rejection drops substantially; the next bottlenecks are depth and normal rejection.

---

## Next Optimization Direction

The next pass should focus on source depth agreement:

- compare nearest-pixel fusion depth checks with bilinear source depth checks
- add diagnostics for source depth relative error distribution
- evaluate whether fusion should use bilinear depth/normal sampling for candidate validation while still storing a discrete source pixel for visited tracking
