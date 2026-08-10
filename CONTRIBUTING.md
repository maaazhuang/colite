# Contributing to Colite

Contributions are welcome while the project is in alpha. Before opening a pull
request:

1. Read `docs/README.md`, `docs/architecture_and_pipeline.md` and
   `docs/development_guide.md`.
2. Keep algorithms in their existing core module; CLI files should remain thin
   wrappers.
3. Add deterministic unit or integration tests for behavior changes.
4. Build and run the CPU test suite. Run the CUDA suite as well when changing a
   CUDA path.
5. Run `git diff --check` and update command or module documentation when
   defaults or behavior change.

Do not commit datasets, reconstruction outputs, build directories, reference
source trees, IDE state or machine-specific presets. Test logs and generated
artifacts belong in a unique ignored `output/` subdirectory.

## Licensing and clean-room boundaries

By contributing, you agree that your contribution is licensed under the
project's BSD 3-Clause License and that you have the right to submit it.

COLMAP's BSD-licensed implementation may be used with retained attribution.
OpenMVS is AGPL-3.0 and must not be copied or adapted into Colite. OpenMVS may
only be used as an external behavioral benchmark; implement algorithms from
papers or other permissive sources and document the design origin.
