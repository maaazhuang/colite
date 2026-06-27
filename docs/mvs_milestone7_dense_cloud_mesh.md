# COLITE MVS Milestone 7: Dense Cloud Cleaning and Meshing Prototype

## Scope

Milestone 7 adds a first post-fusion surface preparation stage after stereo fusion:

- dense cloud filtering
- filtered dense cloud diagnostics
- a visibility-free local-neighbor triangle meshing prototype

This stage is intentionally conservative and testable. It is not a replacement for COLMAP's mature Delaunay/Poisson-style surface reconstruction path.

## Code Entry Points

- `include/colite/mvs/dense_cloud.h`
- `src/colite/mvs/dense_cloud.cpp`
- `app/recon_mvs/recon_mvs_app.cpp`
- `tests/mvs/dense_cloud_test.cpp`
- `tests/integration/recon_mvs_app_test.cpp`

## Dense Cloud Filtering

`DenseCloudFilterOptions` controls the cleanup pass:

- `min_track_observations`: removes fused points whose fusion track has too few supporting observations.
- `max_normal_angle_degrees`: removes tracks whose observed normals are too inconsistent.
- `voxel_size`: when positive, keeps one representative point per voxel to reduce duplicates.

`FilterDenseCloudWithStats` returns both the filtered points and a compact stats struct. The CLI writes these stats to `dense/fused_clean_stats.txt`.

## Meshing Prototype

`MeshDenseCloud` creates a simple `DenseTriangleMesh` from the filtered cloud:

- for each point, gather a bounded set of nearest candidate neighbors
- propose local triangles from those neighbors
- deduplicate triangles by sorted vertex ids
- reject degenerate triangles
- reject triangles with edges longer than `max_edge_length` when configured
- reject triangles whose vertex normals exceed `max_normal_angle_degrees`

The prototype uses `max_candidate_neighbors` to avoid the previous all-triplets behavior, which was not practical for building-18.

## recon_mvs CLI

New flags:

```text
--filter_dense_cloud
--dense_cloud_min_track_observations <n>
--dense_cloud_voxel_size <v>
--mesh_dense_cloud
--mesh_max_edge_length <v>
--mesh_max_normal_angle <deg>
```

`--mesh_dense_cloud` implies `--filter_dense_cloud` because the mesh is generated from the cleaned cloud.

New outputs under `dense/`:

```text
fused_clean.ply
fused_clean_stats.txt
mesh.ply
```

## Verification

Fresh full test command:

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && ctest --test-dir build-wsl --output-on-failure"
```

Result:

```text
100% tests passed, 0 tests failed out of 131
```

building-18 smoke run, using `max_image_size=32` and one PatchMatch iteration, produced:

```text
fused_points 2440
fusion_tracks 2440
fused_observations 2889
```

`dense/fused_clean_stats.txt`:

```text
num_input_points 2440
num_filtered_points 2440
num_removed_by_track_observations 0
num_removed_by_voxel 0
```

`dense/mesh.ply` header:

```text
element vertex 2440
element face 61588
```

## Current Limitations

- The mesh prototype does not use visibility, free-space constraints, graph cuts, Delaunay tetrahedralization, or Poisson reconstruction.
- Triangle density can still be high because every local neighbor pair can produce a candidate face.
- Filter stats currently aggregate non-finite, weak-track, and normal-inconsistent removals into the broad removal path before voxel filtering.
- The next useful step is to add mesh diagnostics and a stricter triangle quality filter before attempting a more COLMAP-like surface reconstruction backend.
