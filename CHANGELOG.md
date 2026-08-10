# Changelog

All notable public changes to Colite are documented here.

## [Unreleased]

- Prepare the project for its first public GitHub release.
- Add project and third-party license documentation.
- Add portable CPU/CUDA presets, installation rules, and Linux CI.
- Exclude local datasets and generated reconstruction results from releases.

## [0.1.0] - 2026-08-10

Initial alpha release candidate:

- Complete incremental SfM and MVS command-line pipeline.
- CPU and CUDA SIFT, matching, and PatchMatch backends.
- Multiple sparse models, model continuation and COLMAP text/binary I/O.
- Twelve camera models, EXIF camera grouping, and optional GPS priors.
- RGB sparse/dense point clouds, Poisson/Delaunay meshing, simplification, and
  textured OBJ output.
- Resumable stage manifests and independent model/SfM maintenance commands.
