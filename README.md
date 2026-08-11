# Colite

English | [简体中文](README_zh-CN.md)

Colite is an experimental C++17 structure-from-motion and multi-view stereo
system. It reconstructs COLMAP-compatible sparse models, RGB dense point
clouds, Poisson or Delaunay meshes, and textured OBJ models from images.

The current release is an alpha-quality research and engineering build. It is
primarily developed and tested on Ubuntu 22.04 and WSL2, with optional CUDA
acceleration for SIFT, descriptor matching, and PatchMatch Stereo. Native
Windows builds have not yet been validated as a supported release target.

## Highlights

- Incremental SfM with multiple models, model continuation and robust Sim(3)
  merging.
- Twelve COLMAP-compatible camera models, EXIF focal initialization, optional
  GPS position priors, and auto/single/per-folder/per-image intrinsics.
- CPU and CUDA SIFT, exact CUDA brute-force descriptor matching, guided
  matching, and several image-pair strategies.
- CPU/CUDA PatchMatch, geometric consistency, RGB/normal Stereo Fusion,
  Poisson and Delaunay meshing, QEM simplification, and texture atlases.
- COLMAP text and binary sparse-model I/O plus resumable stage manifests.
- Independent model analysis, alignment, transformation, cropping, splitting,
  bundle adjustment, triangulation, and registration commands.

## Build

Install the required Ubuntu packages and build the CPU release:

```bash
./scripts/setup_ubuntu_22_04.sh
cmake --preset wsl-release
cmake --build --preset wsl-release --parallel 2
ctest --preset wsl-release
```

With a compatible NVIDIA driver and CUDA Toolkit available:

```bash
cmake --preset wsl-colite-cuda
cmake --build --preset wsl-colite-cuda --parallel 2
ctest --preset wsl-colite-cuda
```

The CUDA compiler and architecture are intentionally not hard-coded. Pass
`-DCMAKE_CUDA_COMPILER=...` or `-DCMAKE_CUDA_ARCHITECTURES=...` when automatic
detection is not appropriate. See [INSTALL.md](INSTALL.md) for dependencies,
installation and troubleshooting.

## Quick start

The repository does not distribute image datasets. Point `--image_path` at a
directory containing your own images:

```bash
COLITE=./build-wsl-colite-cuda/src/colite/exe/colite

$COLITE recon_auto \
  --image_path /path/to/images \
  --output_path output/my_reconstruction \
  --sfm.max_num_features 8192 \
  --sfm.feature_num_threads 8 \
  --mvs.patch_match_backend cuda \
  --mesher poisson \
  --simplify 1 \
  --texture 1
```

Use the CPU build and `--mvs.patch_match_backend cpu` when CUDA is unavailable.
Inspect the exact options supported by the current build with:

```bash
$COLITE --help
$COLITE recon_auto --help
```

The main outputs are:

```text
OUTPUT/
  sfm/                         sparse COLMAP model and RGB PLY
  mvs/dense/fused.ply          RGB and normal dense point cloud
  surface/meshed-poisson.ply   or meshed-delaunay.ply
  surface/textured/mesh.obj    textured mesh, MTL and atlas PNG
  .colite/stages/              resume manifests
```

## Documentation

- [Documentation index](docs/README.md)
- [Architecture and pipeline](docs/architecture_and_pipeline.md)
- [Command-line reference](docs/cli_reference.md)
- [Development environment](docs/development_environment.md)
- [Module implementation and known gaps](docs/module_implementation.md)
- [Benchmark methodology and comparisons](docs/benchmark_comparison.md)
- [Contributing](CONTRIBUTING.md)

## Current limitations

- Sparse reconstruction supports incremental multi-model, global, and
  hierarchical mapping. The global and hierarchical paths are initial
  production implementations and still need broader large-dataset validation.
- CUDA performance and memory usage vary substantially with image size, source
  count, GPU architecture, and cache configuration.
- Texture seams, visibility recall, Delaunay behavior on degenerate geometry,
  and very large datasets need broader validation.
- Compatibility with COLMAP formats does not imply numerical identity or
  equivalent reconstruction accuracy on every dataset.

See the detailed known gaps in
[docs/module_implementation.md](docs/module_implementation.md).

## License

Colite is licensed under the [BSD 3-Clause License](LICENSE). Third-party
notices and retained upstream licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
