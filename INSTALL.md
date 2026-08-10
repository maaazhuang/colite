# Installing Colite

## Supported environment

The validated development environment is Ubuntu 22.04, including Ubuntu 22.04
under WSL2. GCC 11 or newer, CMake 3.20 or newer, and Ninja are recommended.
Native Windows/MSVC is not yet a validated release target.

CUDA is optional. CPU-only SfM and MVS remain available when no CUDA compiler
is detected.

## Ubuntu 22.04 dependencies

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config git \
  libopencv-dev libeigen3-dev libceres-dev libgoogle-glog-dev \
  libflann-dev liblz4-dev libgtest-dev \
  libcgal-dev libboost-graph-dev libgmp-dev libmpfr-dev
```

CGAL, Boost Graph, GMP and MPFR are needed for Delaunay meshing. If CGAL is
not found, Colite still builds but Delaunay meshing is disabled. PoseLib is an
optional dependency; fallback geometric solvers are used when it is absent.

PoissonRecon is bundled in `thirdparty/PoissonRecon` and needs no separate
download.

## CPU build

```bash
cmake --preset wsl-release
cmake --build --preset wsl-release --parallel 2
ctest --preset wsl-release
```

The CPU preset explicitly disables all CUDA backends so its result is
reproducible even on a machine where `nvcc` happens to be installed.

## CUDA build

Install an NVIDIA driver and CUDA Toolkit appropriate for your GPU, then run:

```bash
cmake --preset wsl-colite-cuda
cmake --build --preset wsl-colite-cuda --parallel 2
ctest --preset wsl-colite-cuda
```

If CUDA is installed outside the compiler search path:

```bash
cmake --preset wsl-colite-cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

To build for a specific GPU architecture:

```bash
cmake --preset wsl-colite-cuda -DCMAKE_CUDA_ARCHITECTURES=89
```

Examples include 75 for Turing, 86 for Ampere and 89 for Ada. Choose the value
for the deployment GPU and installed CUDA Toolkit instead of copying an
architecture blindly.

## Install executables

After building, install the command-line executables and documentation under a
prefix:

```bash
cmake --install build-wsl-release --prefix "$HOME/.local"
"$HOME/.local/bin/colite" --help
```

The default build-tree executable is also available at:

```text
build-wsl-release/src/colite/exe/colite
```

## Local presets

Machine-specific paths and architecture choices belong in the ignored
`CMakeUserPresets.json`, not in the shared `CMakePresets.json`.

## Test datasets

No image dataset is distributed with the public repository. Unit and synthetic
tests run without it. Dataset-dependent regression tests are skipped when the
local `data/` directory is absent. Keep private datasets under `data/`; the
whole directory is ignored by Git.

## Troubleshooting

- If CUDA is unexpectedly disabled, inspect `CMAKE_CUDA_COMPILER` and the
  `COLITE_ENABLE_CUDA_*` entries in the selected build's `CMakeCache.txt`.
- In a WSL non-login shell, add the CUDA `bin` directory to `PATH` and its
  library directory to `LD_LIBRARY_PATH`.
- If a build directory was configured with different CUDA settings, use a new
  build directory or explicitly update all relevant cache variables.
- Build and test inside the WSL filesystem when `/mnt/<drive>` I/O becomes a
  bottleneck; do not store test outputs in build directories.
