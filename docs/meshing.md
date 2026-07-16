# Poisson and visibility-aware Delaunay meshing

COLITE provides two production meshing paths after stereo fusion. Both consume
the oriented `dense/fused.ply` point cloud. Delaunay meshing additionally uses
the per-point image visibility produced by stereo fusion.

## Image undistortion

`--image_undistorter` runs before MVS and writes an undistorted workspace under
`undistorted/images`, `undistorted/sparse`, and `undistorted/images.tsv`. The
generated sparse model uses PINHOLE cameras and reprojects linked sparse
observations into the new image geometry. PatchMatch, fusion, meshing, and
texturing then consume the in-memory undistorted reconstruction and images.

The implementation uses a generic inverse map from every output PINHOLE ray
through COLITE's original camera projection. It therefore supports all camera
models whose projection is implemented, including fisheye and thin-prism
models. `--undistort_blank_pixels`, `--undistort_min_scale`,
`--undistort_max_scale`, and `--undistort_max_image_size` control field of view
and resolution. The sizing convention follows COLMAP's BSD-licensed image
undistorter and is rewritten for COLITE's camera API.

## Poisson

Poisson reconstruction is enabled by default with
`COLITE_ENABLE_POISSON_MESHING=ON`. The implementation calls the bundled
PoissonRecon library and then optionally runs SurfaceTrimmer. The vendored
source and command behavior track COLMAP 4.0, including shallow-tree
`fullDepth` handling and the `colors` / `parallel` flags. PoissonRecon is
MIT licensed; its source and license are kept under
`src/thirdparty/PoissonRecon` so a checkout is
self-contained.

Run it as part of MVS with:

```text
recon_mvs ... --run_patch_match --poisson_mesher
```

The result is `dense/meshed-poisson.ply`. The main tuning flags are
`--poisson_depth`, `--poisson_point_weight`, `--poisson_color`, and
`--poisson_trim`. `--poisson_color` is boolean (0 disables colors, 1 enables
them), matching COLMAP 4.0.

## Delaunay

The Delaunay implementation follows the visibility-driven formulation of
Labatut, Pons, and Keriven: it constructs a CGAL 3D Delaunay tetrahedralization,
integrates camera-to-point rays into source/sink and oriented-facet weights,
solves the s-t minimum cut with Boost's Boykov-Kolmogorov max-flow, and extracts
the oriented boundary facets.

Configure with a CGAL installation visible to CMake. When
`COLITE_ENABLE_DELAUNAY_MESHING=ON` but CGAL is absent, COLITE still builds and
reports a clear runtime error if Delaunay meshing is requested.

```text
recon_mvs ... --run_patch_match --delaunay_mesher
```

The outputs are `dense/meshed-delaunay.ply` and
`dense/meshed-delaunay-stats.txt`.

The implementation was written against COLITE's own point, visibility, camera,
mesh, and CLI types. COLMAP was used as the primary behavioral reference. No
OpenMVS source code is included or copied.

## Mesh simplification

`--mesh_simplifier` applies COLITE's quadric-error-metric (QEM) edge-collapse
simplifier to every mesh generated in the same run. It supports Poisson,
Delaunay, and the prototype dense-cloud mesher, writing a sibling
`*-simplified.ply` and statistics file while retaining the original mesh.

The default target is 50% of the input triangles. Set a relative target with
`--mesh_simplification_ratio`, or an absolute target with
`--mesh_simplification_target_triangles`. Boundaries are preserved by default;
`--mesh_simplification_allow_boundary_changes` relaxes that constraint.
`--mesh_simplification_max_error` optionally limits accumulated quadric error
(zero disables the limit), while `--mesh_simplification_max_normal_change`
bounds local face rotation.

The implementation checks the triangular link condition and rejects
degenerate or flipped faces. It is an independent implementation and does not
reuse OpenMVS/VCG source code.

## Mesh texturing

`--mesh_texturer` selects calibrated source views, groups adjacent faces with
the same selected image, packs those regions into a texture atlas, and bakes
the source pixels into an OBJ-compatible UV layout. A software inverse-depth
buffer checks projected vertices and centroids to reject occluded faces.
Neighbor-label smoothing reduces fragmented view assignments, while view
scoring uses OpenCV's worker pool.

For each enabled mesher, COLITE writes `dense/textured-<mesher>/mesh.obj`,
`mesh.mtl`, and `texture.png`. When `--mesh_simplifier` is also enabled, the
simplified mesh is textured. Atlas size, projected resolution, padding,
grazing-angle rejection, label smoothing, and occlusion tolerance are exposed
through the `--texture_*` options shown by `recon_mvs --help`. Atlases grow by
powers of two up to `--texture_max_atlas_size` before reducing texel scale.
The texturer also supports partial projections through
`min_visible_vertices`, configurable inpainting, and regularized global
cross-view color correction to reduce seams.

The implementation is written for COLITE's own camera and mesh types. COLMAP
4.0's BSD-licensed mesh texturer is the primary behavioral reference; no
OpenMVS texture source is copied.

## Standalone COLMAP-compatible commands

The unified executable exposes four thin wrappers for isolated tests:

```bash
colite poisson_mesher --input_path fused.ply --output_path poisson.ply
colite delaunay_mesher --input_path dense_workspace --output_path delaunay.ply
colite mesh_simplifier --input_path mesh.ply --output_path simplified.ply
colite mesh_texturer --input_path mesh.ply --output_path textured \
  --model_path dense_workspace/sparse --image_path dense_workspace/images
```

Run each command with `--help` for its COLMAP-style option prefixes. The PLY
reader accepts ASCII and binary little/big-endian meshes, so identical input
can be passed directly to both implementations.

For a repeatable A/B run against a COLMAP 4.0 executable:

```bash
python3 scripts/compare_colmap4_mvs.py \
  --colite build-wsl-release/src/colite/exe/colite \
  --colmap /path/to/colmap-4.0/bin/colmap \
  --workspace /path/to/dense-workspace \
  --mesh /path/to/common-mesh.ply \
  --output build-wsl-release/compare-mvs
```

The generated `comparison.json` records commands, return codes, elapsed time,
PLY vertex/face counts, atlas dimensions, file sizes, and log tails.
Use `scripts/create_synthetic_mvs_fixture.py` to create the deterministic smoke
fixture. The latest recorded results and interpretation are in
`docs/colmap4_mvs_ab.md`.
