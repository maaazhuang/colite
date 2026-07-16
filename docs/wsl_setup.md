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
  gdb \
  valgrind \
  libopencv-dev \
  libeigen3-dev \
  libceres-dev \
  libgoogle-glog-dev \
  libflann-dev \
  libgtest-dev \
  libcgal-dev \
  libboost-graph-dev
```

`libcgal-dev` and `libboost-graph-dev` are required for the visibility-aware
Delaunay mesher. Ubuntu installs the required GMP/MPFR dependencies
transitively. The COLMAP 4.0-aligned PoissonRecon is vendored in
`src/thirdparty/PoissonRecon` and is
built together with COLITE; it does not require a separate package, download,
or installation.

Configure and build from WSL:

```bash
cd /mnt/d/Code/CXX/colite
cmake -S . -B build-wsl -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
./build-wsl/app/recon_sfm/recon_sfm --version
```

## CUDA SIFT

CUDA SIFT、持久线程池和 AVX2/FMA 默认开启。没有 CUDA 编译器时，配置会明确提示并只关闭 CUDA 后端，CPU 后端仍可构建。WSL 非登录 shell 可能没有把 CUDA 加入 `PATH`；项目会自动探测 `/usr/local/cuda/bin/nvcc`，也可以显式配置：

```bash
cmake -S . -B build-wsl-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCOLITE_ENABLE_CUDA_SIFT=ON
cmake --build build-wsl-cuda -j$(nproc)
```

当前验证环境是 CUDA 13.2 和 RTX 4070 SUPER（compute capability 8.9）。`gdb`、`valgrind` 用于 CPU SIFT 的 ABI、越界和内存回归诊断；Valgrind 不用于 CUDA kernel 检查。

Poisson meshing is enabled by default. Delaunay meshing is enabled when CGAL is
available. To make both requirements explicit during configuration:

```bash
cmake -S . -B build-wsl -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCOLITE_ENABLE_POISSON_MESHING=ON \
  -DCOLITE_ENABLE_DELAUNAY_MESHING=ON
cmake --build build-wsl -j2
```

Focused mesher verification:

```bash
./build-wsl/tests/colite_core_tests \
  --gtest_filter='PoissonMeshingTest.*:DelaunayMeshingTest.*:MeshSimplification*.*'

ctest --test-dir build-wsl \
  -R 'Meshing|Delaunay|Poisson|ReconMvsExeTest' \
  --output-on-failure
```

The synthetic execution tests run PoissonRecon on an oriented sphere and run
the CGAL visibility/graph-cut Delaunay implementation on a sphere observed by
multiple cameras; both assert that a non-empty mesh is produced.
Mesh simplification has no additional system dependency; its tests simplify a
non-planar grid and verify target reduction, valid topology, error limiting,
and ASCII PLY ingestion.
Mesh texturing also uses the existing OpenCV and Eigen dependencies and requires no extra
package. Its focused tests are selected with `MeshTexturing*.*`.
Image undistortion uses that same OpenCV dependency. Run its focused tests with
`ImageUndistort*.*:ReconMvsExeTest.WritesUndistortedWorkspaceWhenRequested`.

## Optional PoseLib for COLMAP-aligned SfM

COLITE detects PoseLib with `find_package(PoseLib QUIET)`. When found, two-view
geometry uses the five-point essential solver and image registration uses P3P;
otherwise the portable Eigen eight-point and DLT fallbacks remain available.
The COLMAP-aligned accuracy numbers in
`docs/sfm_colmap311_validation_20260715.md` were measured with PoseLib enabled.

Verify the active build with:

```bash
grep -q COLITE_HAS_POSELIB build-wsl-cuda/build.ninja && \
  echo "PoseLib enabled"
```

The current WSL reference environment exposes PoseLib under `/usr/local`. A
repository-local installation can instead be selected without changing the
system installation by adding its prefix during configuration:

```bash
cmake -S . -B build-wsl-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build-wsl-deps/install"
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

The installed 3.11 executable is not sufficient for the four-stage comparison:
`mesh_simplifier` and `mesh_texturer` are COLMAP 4.0 commands. Build the local
`references/colmap-4.0.0` checkout and pass that executable to
`scripts/compare_colmap4_mvs.py`; the script does not download or install
anything.

The COLMAP 4.0 reference build additionally requires OpenImageIO on Ubuntu:

```bash
sudo apt install -y libopenimageio-dev openimageio-tools
```

This package is needed only to build the reference COLMAP executable, not
COLITE. In a non-interactive WSL session `sudo` may require running this command
once from the user's terminal before configuring `build-wsl-colmap4-reference`.

The private reference build used for A/B tests is configured and installed
inside the repository, leaving `/usr/local/bin/colmap` unchanged:

```bash
cmake -S references/colmap-4.0.0 -B build-wsl-colmap4-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-wsl-colmap4-reference/install" \
  -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF -DCGAL_ENABLED=ON
cmake --build build-wsl-colmap4-reference -j2
cmake --install build-wsl-colmap4-reference
```

Expected isolated versions:

```text
build-wsl-colmap4-reference/install/bin/colmap  -> COLMAP 4.0.0
/usr/local/bin/colmap                           -> COLMAP 3.11.0
```
## 隔离的 COLMAP 4.0 CUDA MVS 参考构建

以下构建仅写入仓库内的 `build-wsl-colmap4-cuda132-reference`，不会覆盖
`/usr/local/bin/colmap` 的 COLMAP 3.11：

```bash
cmake -S references/colmap-4.0.0 \
  -B build-wsl-colmap4-cuda132-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-wsl-colmap4-cuda132-reference/install" \
  -DCUDA_ENABLED=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DONNX_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF -DCGAL_ENABLED=OFF
cmake --build build-wsl-colmap4-cuda132-reference -j2
```

在当前 COLMAP 4.0 发布树中，CUDA PatchMatch 的无过滤路径创建零尺寸
`consistency_mask_`，但 `GpuMat` 构造器拒绝零尺寸。用于 A/B 的隔离参考树把该占位
改为 `GpuMat<uint8_t>(1, 1, 1)`；无过滤模板分支不会读取该占位。具体测试条件与结果见
[mvs_colmap4_core_validation_20260716.md](mvs_colmap4_core_validation_20260716.md)。

## colite CUDA MVS 构建

`COLITE_ENABLE_CUDA_MVS` 默认关闭。MVS A/B 必须显式开启，不能仅凭构建目录名判断：

```bash
cmake -S . -B build-wsl-mvs-cuda-20260716 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCOLITE_BUILD_TESTS=OFF \
  -DCOLITE_ENABLE_CUDA_SIFT=ON -DCOLITE_ENABLE_CUDA_MVS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-wsl-mvs-cuda-20260716 -j2
```
