# Colite 开发、编译与测试环境

更新时间：2026-07-18。本文记录可复现的主环境；版本变化后应同步更新。

## 当前主机与工具链

| 项目 | 当前配置 |
|---|---|
| Windows 工作区 | `D:\Code\CXX\colite` |
| WSL 路径 | `/mnt/d/Code/CXX/colite` |
| WSL | Ubuntu 22.04.5 LTS，发行版 `Ubuntu-22.04`，WSL2 |
| GCC/G++ | 11.4.0 |
| CMake / Ninja | 3.22.1 / 1.10.1 |
| CUDA | 13.2，`/usr/local/cuda-13.2/bin/nvcc`，`nvcc 13.2.51` |
| 验证 GPU | RTX 4070 SUPER，compute capability 8.9，驱动 610.74，显存 12,282 MiB |
| 系统 COLMAP | `/usr/local/bin/colmap`，3.11.0 |
| 主构建目录 | `build-wsl-colite-cuda` |

项目主线使用 WSL + CMake + Ninja。Windows checkout 便于 IDE 和文件管理；编译、测试、Git 和基准命令优先在 WSL 执行。

当前已验证的 WSL 运行方式为用户 `ma`：

```powershell
wsl.exe -d Ubuntu-22.04 -u ma -- bash -lc 'export PATH=/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib; export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu; nvcc -V; nvidia-smi'
```

Codex/任务调度启动的 WSL 非登录 shell 不一定继承用户登录环境；若 `nvcc` 显示找不到，应显式设置上述 `PATH` 和 `LD_LIBRARY_PATH`。GPU 验证结果为 RTX 4070 SUPER、驱动 610.74、12,282 MiB 显存。

## 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config git gdb valgrind \
  libopencv-dev libeigen3-dev libceres-dev libgoogle-glog-dev \
  libflann-dev liblz4-dev libgtest-dev \
  libcgal-dev libboost-graph-dev
```

- CGAL、Boost Graph、GMP/MPFR 用于 Delaunay；缺少 CGAL 时项目仍可构建，但 Delaunay 命令运行时不可用。
- PoissonRecon 已位于 `thirdparty/PoissonRecon`，不需要外部下载、安装或单独编译。
- PoseLib 是可选依赖。检测到时使用 five-point essential 和 P3P；否则使用 Eigen eight-point 和 DLT fallback。

## 推荐构建

主 CUDA Release 构建：

```bash
cd /mnt/d/Code/CXX/colite
cmake --preset wsl-colite-cuda
cmake --build --preset wsl-colite-cuda --parallel 2
ctest --preset wsl-colite-cuda
```

该 preset 显式开启：CUDA SIFT、CUDA 描述子匹配、CUDA MVS、SIFT 多线程、AVX2/FMA、Poisson、Delaunay 和测试，CUDA 架构为 89。

CPU/通用 Release：

```bash
cmake --preset wsl-release
cmake --build --preset wsl-release --parallel 2
ctest --test-dir build-wsl-release --output-on-failure
```

Debug：

```bash
cmake --preset wsl-debug
cmake --build --preset wsl-debug --parallel 2
ctest --test-dir build-wsl --output-on-failure
```

注意：顶层 `COLITE_ENABLE_CUDA_MVS` 默认是 `OFF`；只有 `wsl-colite-cuda` preset 或显式 `-DCOLITE_ENABLE_CUDA_MVS=ON` 才启用 CUDA PatchMatch。不要根据目录名称猜测，应检查：

```bash
grep -E 'CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|COLITE_ENABLE_CUDA' \
  build-wsl-colite-cuda/CMakeCache.txt
```

## CUDA 非登录 shell

WSL 非登录 shell 可能没有 CUDA PATH。CMake 会尝试 `/usr/local/cuda/bin/nvcc`，仍失败时执行：

```bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
which nvcc
nvcc --version
```

显式配置示例：

```bash
cmake -S . -B build-wsl-colite-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCOLITE_BUILD_TESTS=ON \
  -DCOLITE_ENABLE_CUDA_SIFT=ON -DCOLITE_ENABLE_CUDA_MATCHER=ON \
  -DCOLITE_ENABLE_CUDA_MVS=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

## 测试命令

完整测试：

```bash
cmake --build build-wsl-colite-cuda --parallel 2
ctest --test-dir build-wsl-colite-cuda --output-on-failure --parallel 2
```

2026-07-30 每图相机/EXIF/混合尺寸、CUDA matcher、SfM 并行/缓存、PatchMatch
几何覆盖、COLMAP dense-map 兼容和 Stereo Fusion 候选解耦后为 320/320；
全部测试通过，
其中 1 个“CUDA 不可用时应失败”的条件测试在 CUDA 可用环境中按设计跳过。
House 实测摘要同时记录 `feature_backend cuda_vlfeat`、
`cuda_sift_available 1`、`cuda_matcher_available 1` 和 `cpu_threads 8`，
不可只根据构建目录名称判断 CUDA 是否启用。

定向测试：

```bash
TESTS=./build-wsl-colite-cuda/tests/colite_core_tests

$TESTS --gtest_filter='ParallelRuntimeTest.*:SiftExtractorTest.*:SiftReferenceConsistencyTest.*'
$TESTS --gtest_filter='DescriptorMatcherTest.*:MatchingPairGeneratorTest.*:TwoViewGeometry*'
$TESTS --gtest_filter='Incremental*:*BundleAdjusterTest.*:Observation*'
$TESTS --gtest_filter='*PatchMatch*:*StereoFusion*:*DenseMap*:*ReconMvsExeTest*'
$TESTS --gtest_filter='PoissonMeshingTest.*:DelaunayMeshingTest.*:MeshSimplification*.*:MeshTexturing*.*'
```

CPU 内存问题可用 Valgrind；CUDA kernel 不使用 Valgrind：

```bash
valgrind --error-exitcode=99 --leak-check=full \
  $TESTS --gtest_filter='<CPU_TEST_FILTER>'
```

## 构建目录用途

| 目录 | 用途 | 是否长期保留 |
|---|---|---|
| `build-wsl-colite-cuda` | 当前 Colite Release + CUDA 主构建 | 是 |
| `build-wsl-release` | CPU/通用 Release 回归 | 可重建，按需保留 |
| `build-wsl-colmap4-reference` | 隔离 COLMAP 4.0 CPU/CGAL | 做 MVS A/B 时保留 |
| `build-wsl-colmap4-cuda132-reference` | 隔离 COLMAP 4.0 CUDA | 做 PatchMatch A/B 时保留 |
| `build-wsl-openmvg-2.1` | OpenMVG 2.1 参考构建 | 做外部 SfM 基准时保留 |
| `build-wsl-openmvs-2.4.0` | OpenMVS CPU 参考构建 | 做外部 MVS 基准时保留 |
| `build-wsl-openmvs-2.4.0-cuda` | OpenMVS CUDA 参考构建 | 做外部 MVS 基准时保留 |

构建目录只存放可重建的编译产物和依赖，不存放新的测试数据。所有测试中间结果、日志和最终产物进入 `output/`。

## 参考程序隔离

### COLMAP 3.11

系统参考命令为 `/usr/local/bin/colmap`，用于 SfM 对齐。检查：

```bash
which colmap
colmap -h | head -8
```

### COLMAP 4.0

COLMAP 4.0 安装在项目内构建前缀，不覆盖系统 3.11。CPU 构建需要 OpenImageIO：

```bash
sudo apt install -y libopenimageio-dev openimageio-tools

cmake -S references/colmap-4.0.0 -B build-wsl-colmap4-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build-wsl-colmap4-reference/install" \
  -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF \
  -DTESTS_ENABLED=OFF -DCGAL_ENABLED=ON
cmake --build build-wsl-colmap4-reference --parallel 2
cmake --install build-wsl-colmap4-reference
```

CUDA 13.2 构建使用 `build-wsl-colmap4-cuda132-reference`，显式指定 nvcc 和架构 89，并关闭 GUI、OpenGL、ONNX、测试。参考源码中的兼容修复只允许留在隔离参考树，不得混入 Colite 实现。

### OpenMVG 2.1

安装前缀为 `build-wsl-openmvg-2.1/install`。使用 Release、OpenMP，开启软件工具，关闭测试、示例和 GUI。它用于独立 SfM 基准及向 OpenMVS 转换。

### OpenMVS 2.4

额外依赖：

```bash
sudo apt install -y \
  libboost-iostreams-dev libboost-system-dev libboost-serialization-dev \
  libboost-program-options-dev libboost-math-dev libnanoflann-dev
```

CPU/CUDA 构建均使用项目内 `thirdparty/vcglib`，关闭 viewer、Python、Breakpad、测试和 IPO。CUDA 只加速 `DensifyPointCloud` 的深度估计；Fusion、网格和纹理仍主要在 CPU。

## 常见问题

- WSL 出现 `WSL_E_DISTRO_NOT_FOUND`：先执行 `wsl --list --verbose` 并重启 WSL/Windows；不要执行 `wsl --unregister`，它会删除发行版数据。
- Windows Git 报 dubious ownership：优先在 WSL 中运行 Git，不要为绕过问题破坏性修改仓库。
- `/mnt/d` 全量重编译慢：可把临时工作副本放在 WSL ext4；Windows checkout 继续作为镜像或远端。
- CUDA 结果看似被秒读：检查是否复用了 dense map/cache。性能 A/B 必须使用新的 `output/` 子目录。
- PoseLib 是否启用：检查 `build.ninja` 中的 `COLITE_HAS_POSELIB`；比较 SfM 精度时必须记录该状态。
