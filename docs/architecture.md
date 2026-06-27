# Colite Architecture

Colite is a compact C++ reconstruction system organized around a minimal
SfM-first pipeline and a later CPU MVS baseline.

For the current implementation status, COLMAP comparison baseline, and
handoff notes, see `docs/current_status.md`.

## Module Layout

- `core`: shared math-facing types, camera models, versioning, logging hooks.
- `image`: image loading, metadata, undistortion, and camera initialization.
- `features`: feature extraction, descriptor storage, and feature caches.
- `matching`: pair generation, descriptor matching, and geometric verification.
- `sfm`: tracks, reconstruction state, triangulation, PnP, and bundle adjustment.
- `mvs`: source-view selection, sparse/dense depth estimation, depth fusion,
  meshing, and texturing.
- `io`: PLY/OBJ and COLMAP-like import/export.
- `app`: command-line tools.
- `tests`: focused unit and integration tests.

The first implementation milestone has grown into an end-to-end incremental SfM
baseline. The active modules are `core`, `features`, `matching`, `sfm`, `mvs`,
`app`, and `tests`. MVS currently contains the COLMAP-style workspace skeleton:
source-view selection and sparse depth samples from the SfM reconstruction. See
`docs/mvs_status.md` for the active MVS handoff notes.
