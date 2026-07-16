# COLITE WSL 开发与测试交接说明

更新时间：2026-07-13

本文用于在新的 Codex 对话或其他开发会话中快速恢复 COLITE 的本地开发上下文。项目主工作区位于 Windows 路径：

```text
D:\Code\CXX\colite
```

在 WSL 中对应路径为：

```bash
/mnt/d/Code/CXX/colite
```

## 当前 WSL 环境

建议继续使用 WSL + CMake + GCC/Ninja 开发，不建议切回 Windows VS 工程作为主线。

当前检测到的环境：

```text
WSL distro: Ubuntu-22.04
OS: Ubuntu 22.04.5 LTS
GCC/G++: 11.4.0
CMake: 3.22.1
Ninja: 1.10.1
COLMAP: 3.11.0, /usr/local/bin/colmap
CUDA Toolkit: 13.2, nvcc V13.2.51（用户交互 WSL 已确认）
Git branch: feature/conservative-colmap-sfm
Last known committed baseline: ad6bab9
```

注意：用户交互 WSL 中已确认 `nvcc --version` 可用：

```text
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 13.2, V13.2.51
```

但 Codex 通过非交互/受控命令读取环境时，可能因为 `PATH`、登录 shell 初始化文件、用户上下文或权限差异暂时找不到 `nvcc`。如果 CUDA 构建或 CUDA 测试异常，先在 WSL 中确认以下信息：

```bash
whoami
echo $PATH
which nvcc
nvcc --version
ls -l /usr/local/cuda/bin/nvcc
ls -l /usr/local/cuda-13.2/bin/nvcc
```

如果 `nvcc` 存在但 Codex 命令找不到，优先显式补充 CUDA 路径后再配置/构建：

```bash
export PATH=/usr/local/cuda-13.2/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## 依赖环境

基础依赖安装命令如下。多数依赖已安装，除非重装 WSL，一般不需要重复执行。

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

COLMAP 参考版本为源码安装的 3.11，路径通常为：

```bash
/usr/local/bin/colmap
```

检查命令：

```bash
which colmap
colmap -h | head -8
```

预期首行：

```text
COLMAP 3.11.0 -- Structure-from-Motion and Multi-View Stereo
```

## 常用构建目录

当前仓库主要存在以下构建目录：

```text
build-wsl          Debug，CPU/常规开发常用
build-wsl-release  Release，性能对比时使用
build-wsl-cuda     CUDA 相关开发和当前较完整测试常用
```

`build-wsl-cuda` 是否真正启用 CUDA，取决于配置该目录时 CMake 是否检测到 CUDA 编译器。若切换 shell、重启 WSL 或调整 CUDA `PATH`，建议重新检查：

```bash
grep -E 'CMAKE_CUDA_COMPILER|COLITE_ENABLE_CUDA' build-wsl-cuda/CMakeCache.txt
```

`CMakePresets.json` 中已有：

```bash
cmake --preset wsl-debug
cmake --build --preset wsl-debug

cmake --preset wsl-release
cmake --build --preset wsl-release
```

当前近期全量验证使用的是 `build-wsl-cuda`：

```bash
cd /mnt/d/Code/CXX/colite
cmake --build build-wsl-cuda -j2
ctest --test-dir build-wsl-cuda --output-on-failure
```

最近一次确认结果：

```text
205/205 tests passed
4 CUDA PatchMatch backend tests skipped
```

跳过的 4 个测试是条件性 CUDA 后端测试，通常表示当前构建/运行环境未满足测试内部的 CUDA 后端执行条件，不等价于全量失败。

## 常用测试命令

全量测试：

```bash
cd /mnt/d/Code/CXX/colite
cmake --build build-wsl-cuda -j2
ctest --test-dir build-wsl-cuda --output-on-failure
```

只跑 SfM / workspace / 命令层相关测试：

```bash
./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='SfmWorkspaceTest.*:SfmCommandExeTest.*:ColiteCommandExeTest.*:ColiteCompareExeTest.*'
```

只跑匹配与两视图几何：

```bash
./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='DescriptorMatcherTest.*:MatchingPairGeneratorTest.*:TwoViewGeometry*'
```

只跑增量 SfM：

```bash
./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='Incremental*:*BundleAdjusterTest.*:Observation*'
```

只跑 MVS / PatchMatch：

```bash
./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='*PatchMatch*:*StereoFusion*:*DenseMap*:*ReconMvsExeTest*'
```

提交前格式检查：

```bash
git diff --check
```

已知可能出现 CRLF/LF 警告，若 `git diff --check` 退出码为 0，一般不是阻塞问题。

## 主要命令行工具

统一入口：

```bash
./build-wsl-cuda/src/colite/exe/colite --help
```

分阶段 SfM 文件 workspace 流程：

```bash
./build-wsl-cuda/src/colite/exe/colite feature_extractor \
  --image_path data/building-18 \
  --workspace_path build-wsl-cuda/workspaces/building-18 \
  --max_num_features 4000 \
  --feature_extractor vlsift \
  --feature_num_threads 4

./build-wsl-cuda/src/colite/exe/colite exhaustive_matcher \
  --workspace_path build-wsl-cuda/workspaces/building-18 \
  --matcher flann

./build-wsl-cuda/src/colite/exe/colite mapper \
  --workspace_path build-wsl-cuda/workspaces/building-18
```

旧整链路 SfM：

```bash
./build-wsl-cuda/src/colite/exe/recon_sfm \
  --image_path data/building-18 \
  --output_path build-wsl-cuda/compare-building-18/colite-recon-sfm \
  --focal_length 2457.6 \
  --max_num_features 4000 \
  --feature_extractor vlsift \
  --matcher flann
```

COLMAP text model 对比工具：

```bash
./build-wsl-cuda/src/colite/exe/colite compare_colmap \
  --reference_path <colmap_text_model_dir> \
  --candidate_path <colite_text_model_dir> \
  --output_path <report_dir>
```

输出文件：

```text
summary.txt
image_diff.tsv
pair_diff.tsv
point_diff.tsv
pose_diff.tsv
```

MVS 命令：

```bash
./build-wsl-cuda/src/colite/exe/recon_mvs --help
```

一个轻量 smoke 示例：

```bash
./build-wsl-cuda/src/colite/exe/recon_mvs \
  --model_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --image_path data/building-18 \
  --output_path build-wsl-cuda/mvs-building-18/smoke \
  --run_patch_match \
  --max_image_size 48 \
  --patch_match_iterations 1 \
  --patch_match_window_radius 1 \
  --filter_photometric_max_cost 0.3 \
  --fuse_stereo \
  --fusion_min_num_observations 1 \
  --filter_geometric_min_num_consistent_sources 1
```

## building-18 数据与 COLMAP 对比

测试图像位于：

```text
data/building-18
```

常用相机参数：

```text
camera model: PINHOLE
focal length: 2457.6
principal point: 1536,1152
image size: 3072 x 2304
```

COLMAP 3.11 参考基线曾放在：

```text
build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided
```

若需要重新跑 COLMAP 对比，保持参数尽量一致：

```bash
colmap feature_extractor \
  --database_path <output>/database.db \
  --image_path data/building-18 \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params 2457.6,2457.6,1536,1152 \
  --SiftExtraction.max_num_features 4000 \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \
  --database_path <output>/database.db \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.cross_check 1 \
  --TwoViewGeometry.max_error 4 \
  --TwoViewGeometry.min_num_inliers 15 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path <output>/database.db \
  --image_path data/building-18 \
  --output_path <output>/sparse \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 1.0 \
  --Mapper.tri_min_angle 1.0
```

## 当前重要功能状态

SfM 主线已经具备：

- COLMAP 风格核心数据结构。
- 完整相机模型族注册，重点实现/验证多种投影模型。
- SIFT 后端切换：`auto`、`vlsift`、`cvsift`、`cuda_opencv`、`cuda_vlfeat`；默认 `auto` 优先使用 CUDA OpenCV 模式。
- 描述子匹配后端：自研 brute force 与 FLANN。
- exhaustive / sequential / spatial / vocab tree 匹配。
- 内部 E/F/H 两视图几何验证，支持退化、平面、全景分类。
- guided matching / weak pair 二阶段重匹配。
- Track / Observation Manager。
- 增量初始化、注册、三角化、re-triangulation。
- 局部 BA、全局 BA、BA 参数 mask、gauge fixing。
- 坏点/坏观测过滤。
- 分阶段 workspace 文件缓存，按配置 hash 复用 feature / match / geometry / sparse reconstruction。
- `compare_colmap` 固定差异报告。

MVS 主线正在推进 COLMAP 化，当前已有：

- COLMAP text sparse model 输入。
- source view 选择。
- CPU PatchMatch 基础流程。
- CUDA PatchMatch 相关路径。
- bilateral weighted NCC 与动态 source-selection probability 等近期实验。
- stereo fusion、dense cloud cleaning、初步 meshing。

MVS 相关文档可参考：

```text
docs/mvs_status.md
docs/mvs_experiments.md
docs/meshing.md
```

## Git 与工作树注意事项

在 Windows PowerShell 沙箱中直接运行 `git` 可能触发：

```text
fatal: detected dubious ownership in repository
```

这是因为 Windows 仓库所有者是用户 `Ma`，而 Codex 沙箱进程可能是另一个用户。优先在 WSL 中运行 git：

```bash
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/d/Code/CXX/colite && git status --short"
```

当前已知基线提交：

```text
ad6bab9 feat: advance colmap-style sfm diagnostics and caching
```

本文件创建前，工作树已经存在 MVS/CUDA/meshing 相关未提交改动和未跟踪文件。后续对话不要误以为这些改动由自己刚刚创建，也不要随意 `reset` 或删除。

提交前建议流程：

```bash
cd /mnt/d/Code/CXX/colite
git status --short
cmake --build build-wsl-cuda -j2
ctest --test-dir build-wsl-cuda --output-on-failure
git diff --check
git add <本次相关文件>
git commit -m "<message>"
```

不要使用：

```bash
git reset --hard
git checkout -- .
```

除非用户明确要求丢弃改动。

## WSL 与数据安全注意事项

- 不要执行 `wsl --unregister Ubuntu-22.04`，这会删除整个 WSL 发行版数据。
- 不要删除 `references/`，其中存放 COLMAP、SIFT、CudaSift 等参考源码。
- 不要把大体积构建目录、对比输出或中间 dense maps 提交进 git。
- `data/building-18` 是当前最常用的 smoke 数据集，不要清理。
- 如果 WSL 临时异常，优先尝试重启 WSL 或 Windows，不要重装发行版。

## 给新对话的建议开场

如果要让新的 Codex 对话接手，可以直接给它以下信息：

```text
项目在 D:\Code\CXX\colite，开发主环境为 WSL Ubuntu-22.04。
请先阅读 docs/wsl_handoff.md、docs/current_status.md、docs/mvs_status.md。
优先在 WSL 中运行 git/build/test，当前常用构建目录是 build-wsl-cuda。
不要重置工作树；当前可能已有 MVS/CUDA/meshing 未提交改动。
```
