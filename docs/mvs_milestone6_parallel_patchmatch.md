# MVS Milestone 6: Parallel PatchMatch Scheduling

Updated: 2026-06-24

This note records the Milestone 6 implementation from
`docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md`.

## Scope

Milestone 6 adds a conservative multithreaded execution path for `recon_mvs`.
The first per-reference PatchMatch pass can now run in parallel while keeping
the downstream output deterministic.

## Implemented Behavior

CLI:
- `--num_threads <n>`
- Default: `1`
- Validation: `n >= 1`

Scheduling:
- The main thread collects PatchMatch target images from the MVS summary.
- The main thread preloads all required reference/source `MvsImage` objects
  through `MvsImageWorkspace`.
- The main thread builds `PatchMatchView` entries for all preloaded images.
- Worker tasks run the first per-reference PatchMatch pass through a capped
  `std::async` queue.
- Worker tasks only read preloaded images/views; mutable `MvsImageWorkspace`
  is not shared across threads.
- Results are sorted by `image_id` before geometric refinement, consistency
  checking, fusion, and diagnostics.

This keeps fixed-seed runs deterministic for `--num_threads 1` and
`--num_threads 2`.

## Tests

Added integration coverage in `tests/integration/recon_mvs_app_test.cpp`:
- help text includes `--num_threads`
- `--num_threads 0` is rejected
- `--num_threads 2` runs PatchMatch and writes `stereo/patch_match.tsv`

Verification:

```text
ctest --test-dir build-wsl -R 'ReconMvsAppTest' --output-on-failure
10/10 passed

ctest --test-dir build-wsl --output-on-failure
125/125 passed
```

## building-18 Smoke

Command shape:

```text
./build-wsl/app/recon_mvs/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl/mvs-building-18/colite-pm-thread<N>-32-ms6 \
  --run_patch_match \
  --max_image_size 32 \
  --patch_match_iterations 1 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --filter_geometric_min_num_consistent_sources 1 \
  --num_threads <N>
```

Outputs compared:
- `build-wsl/mvs-building-18/colite-pm-thread1-32-ms6/summary.txt`
- `build-wsl/mvs-building-18/colite-pm-thread2-32-ms6/summary.txt`
- `build-wsl/mvs-building-18/colite-pm-thread1-32-ms6/stereo/patch_match.tsv`
- `build-wsl/mvs-building-18/colite-pm-thread2-32-ms6/stereo/patch_match.tsv`

Both diffs were empty.

Shared smoke metrics:

```text
Images: 18
Sparse depth samples: 26653
PatchMatch images: 18
Photometric filtered pixels: 4182
Geometric consistent pixels: 2889
Geometric filtered pixels: 10935
Fused points: 2440
Fusion tracks: 2440
Fused observations: 2889
```

## Notes

Only the first PatchMatch pass is parallelized in this milestone. Geometric
refinement, consistency graph construction, and stereo fusion remain sequential
for now. Those stages can be parallelized later, but the current design already
removes shared mutable image cache access from PatchMatch workers.
