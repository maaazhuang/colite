# Reconstruction pipeline architecture

COLITE exposes two stable pipeline boundaries and one composition command:

- `recon_sfm`: raw images to a registered sparse COLMAP text model.
- `recon_mvs`: sparse model and images to dense maps, fused cloud, and optional
  surface products.
- `recon_auto`: composes all independently runnable stages from raw images to
  a textured surface. Poisson is the default mesher; Delaunay or a fused-cloud
  terminal output can be selected explicitly.

Image undistortion is an SfM-to-MVS workspace transform. It is not part of the
geometric SfM solver, and operationally is the first MVS stage.

## Workspace contract

```text
OUTPUT/
  sfm/
    cameras.txt
    images.txt
    points3D.txt
  mvs/
    undistorted/
      images/
      sparse/
    stereo/
      depth_maps/
      normal_maps/
      cost_maps/
      consistency_graphs*/
    dense/
      fused.ply
      points_visibility.bin
  .colite/stages/
    recon_sfm.json
    image_undistorter.json
    patch_match_stereo.json
    stereo_fusion.json
    poisson_mesher.json (or delaunay_mesher.json)
    mesh_simplifier.json
    mesh_texturer.json
  surface/
    meshed-poisson.ply (or meshed-delaunay.ply)
    mesh-simplified.ply
    textured/
      mesh.obj
      mesh.mtl
      texture.png
```

The stage manifests contain a schema version, status, input and option
fingerprints, elapsed time, and declared outputs. `--resume` skips only when the
manifest fingerprints still match and every declared output exists. `--force`
always runs. The default is also to run, and `--dry_run` prints stage boundaries
without reading or writing reconstruction products.

## Commands

```bash
# Full raw-images-to-textured-mesh composition (Poisson by default).
colite recon_auto --image_path images --output_path result

# Select Delaunay, or stop with the fused point cloud.
colite recon_auto --image_path images --output_path result --mesher delaunay
colite recon_auto --image_path images --output_path result --stop_after fusion

# Inspect the plan or resume an interrupted run.
colite recon_auto --image_path images --output_path result --dry_run
colite recon_auto --image_path images --output_path result --resume

# Run only a range of stages.
colite recon_auto --image_path images --output_path result \
  --start_from stereo --stop_after texture

# Independent stages.
colite image_undistorter --image_path images --input_path result/sfm \
  --output_path result/mvs
colite patch_match_stereo \
  --model_path result/mvs/undistorted/sparse \
  --image_path result/mvs/undistorted/images --output_path result/mvs
colite stereo_fusion \
  --model_path result/mvs/undistorted/sparse \
  --image_path result/mvs/undistorted/images --output_path result/mvs
```

`stereo_fusion` reads persisted depth, normal, cost, and consistency maps. It
does not rerun PatchMatch. All three MVS stage commands forward their algorithm
options to the existing `recon_mvs` option parser, keeping one implementation
of the algorithms and one parameter vocabulary.

Surface processing is also independently composable. These commands support
the same `--resume`, `--force`, and `--dry_run` controls:

```bash
colite poisson_mesher --input_path result/mvs/dense/fused.ply \
  --output_path result/surface/poisson.ply
colite mesh_simplifier --input_path result/surface/poisson.ply \
  --output_path result/surface/poisson-simplified.ply
colite mesh_texturer --input_path result/surface/poisson-simplified.ply \
  --model_path result/mvs/undistorted/sparse \
  --image_path result/mvs/undistorted/images \
  --output_path result/surface/textured
```

## Module ownership

`recon_sfm` owns feature extraction, matching, mapping, triangulation, and
bundle adjustment. `image_undistorter` owns calibrated image resampling and the
dense workspace model. `patch_match_stereo` owns dense maps and consistency
graphs. `stereo_fusion` owns the fused cloud and visibility records. Meshing,
simplification, and texturing each own only their declared surface artifact.

The command layer is orchestration only. SfM/MVS algorithms remain under
`colite::controllers`, `colite::sfm`, and `colite::mvs`; stage execution and
manifests live under `colite::exe` and do not duplicate algorithm code.
