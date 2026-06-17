# Colite Architecture

Colite is a compact C++ reconstruction system organized around a minimal
SfM-first pipeline and a later CPU MVS baseline.

## Initial Module Layout

- `core`: shared math-facing types, camera models, versioning, logging hooks.
- `image`: image loading, metadata, undistortion, and camera initialization.
- `features`: feature extraction, descriptor storage, and feature caches.
- `matching`: pair generation, descriptor matching, and geometric verification.
- `sfm`: tracks, reconstruction state, triangulation, PnP, and bundle adjustment.
- `mvs`: depth estimation, depth fusion, meshing, and texturing.
- `io`: PLY/OBJ and COLMAP-like import/export.
- `app`: command-line tools.
- `tests`: focused unit and integration tests.

The first milestone intentionally contains only `core`, `app`, and `tests`.
New modules should be added as independent CMake targets or source groups when
their interfaces become concrete.

