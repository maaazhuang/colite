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
  libgtest-dev

cat <<'EOF'

Colite WSL dependencies are installed.

Suggested build commands:
  cd /mnt/d/Code/CXX/colite
  cmake -S . -B build-wsl -G Ninja -DCMAKE_BUILD_TYPE=Debug
  cmake --build build-wsl
  ctest --test-dir build-wsl --output-on-failure
  ./build-wsl-colite-cuda/src/colite/exe/recon_sfm --version

EOF
