# WSL Development Setup

The current project handoff notes are in `docs/current_status.md`. That file
records the installed COLMAP 3.11 reference build, building-18 comparison
results, and next development priorities.

Recommended baseline on Ubuntu 22.04:

```bash
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
```

Configure and build from WSL:

```bash
cd /mnt/d/Code/CXX/colite
cmake -S . -B build-wsl -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
./build-wsl/app/recon_sfm/recon_sfm --version
```

If full rebuilds become slow under `/mnt/d`, move the repository to the WSL
ext4 filesystem and keep this Windows checkout as a mirror or Git remote.

## Reference COLMAP 3.11

The comparison environment uses a source-built COLMAP 3.11 from
`references/colmap-3.11.0`, installed to `/usr/local/bin/colmap`.

```bash
which colmap
colmap -h | head -8
```

Expected version:

```text
COLMAP 3.11.0 -- Structure-from-Motion and Multi-View Stereo
```

COLMAP was configured without CUDA, GUI, or OpenGL for CLI-only CPU comparison.
