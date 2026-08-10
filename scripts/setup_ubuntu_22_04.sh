#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  git \
  libopencv-dev \
  libeigen3-dev \
  libceres-dev \
  libgoogle-glog-dev \
  libflann-dev \
  liblz4-dev \
  libgtest-dev \
  libcgal-dev \
  libboost-graph-dev \
  libgmp-dev \
  libmpfr-dev

cat <<'EOF'

Colite WSL dependencies are installed.

CPU build commands:
  cmake --preset wsl-release
  cmake --build --preset wsl-release --parallel 2
  ctest --preset wsl-release

CUDA is optional and is not installed by this script. After installing a
compatible NVIDIA driver and CUDA Toolkit, configure with:
  cmake --preset wsl-colite-cuda
  cmake --build --preset wsl-colite-cuda --parallel 2
  ctest --preset wsl-colite-cuda

Override the CUDA compiler or GPU architectures when needed, for example:
  cmake --preset wsl-colite-cuda \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=86

EOF
