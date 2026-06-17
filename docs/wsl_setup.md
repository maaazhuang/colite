# WSL Development Setup

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
