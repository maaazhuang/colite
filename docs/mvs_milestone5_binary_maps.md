# MVS Milestone 5: Binary Dense Maps

Updated: 2026-06-23

This note records the Milestone 5 implementation from
`docs/superpowers/plans/2026-06-23-mvs-colmapization-plan.md`.

## Scope

Milestone 5 adds binary IO for depth maps, normal maps, and consistency graphs,
then wires the format into `recon_mvs` behind an explicit CLI switch.

## Implemented APIs

Dense depth maps:
- `WriteDenseDepthMapBinary`
- `ReadDenseDepthMapBinary`

Normal maps:
- `WriteNormalMapBinary`
- `ReadNormalMapBinary`

Consistency graphs:
- `WriteConsistencyGraphBinary`
- `ReadConsistencyGraphBinary`

CLI:
- `--dense_map_format text|binary`
- Default remains `text`.

## Binary Formats

Depth map:

```text
COLITE_DEPTH_MAP_BINARY 1
width height depth_min depth_max
<row-major raw float depth values>
```

Normal map:

```text
COLITE_NORMAL_MAP_BINARY 1
width height
<row-major raw float nx ny nz triples>
```

Consistency graph:

```text
COLITE_CONSISTENCY_GRAPH_BINARY 1
width height
<per-pixel uint32 source_count followed by uint32 source image ids>
```

The binary payload is written in native little-endian format on the current
WSL/Linux development platform.

## Output Behavior

With `--dense_map_format binary`:
- `stereo/depth_maps/*.bin`
- `stereo/normal_maps/*.bin`
- `stereo/consistency_graphs/*.bin`
- `stereo/cost_maps/*.txt`

Cost maps intentionally remain text for now because they are still mainly used
as lightweight debugging and diagnostics artifacts.

## Verification

Targeted Milestone 5 tests:

```text
ctest --test-dir build-wsl -R 'DenseMapIoTest.RoundTripsDepthAndNormalMapsAsBinary|CpuPatchMatchGeometricConsistencyTest.RoundTripsConsistencyGraphAsBinary|ReconMvsAppTest' --output-on-failure
10/10 passed
```

Full test suite:

```text
ctest --test-dir build-wsl --output-on-failure
123/123 passed
```

building-18 binary smoke:

```text
output_path build-wsl/mvs-building-18/colite-pm-binary-32-ms5
max_image_size 32
patch_match_iterations 1
dense_map_format binary

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

The binary smoke matches the previous `max_image_size=32` default-path geometry
and fusion metrics, so this milestone changes output format without changing
PatchMatch or stereo fusion numerical behavior.

## Next Step

Milestone 6 should focus on multithreading and cache optimization. That is now
more important because geometric refinement and larger smoke resolutions are
already showing CPU runtime pressure.
