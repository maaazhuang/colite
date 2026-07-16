# SIFT 后端、构建与验证

COLITE 现有五种 SIFT 选择：`auto`、`cvsift`、`vlsift`、`cuda_opencv` 和 `cuda_vlfeat`。默认值是 `auto`：运行时检测到 CUDA 设备便选择 `cuda_opencv`，否则选择 `cvsift`。旧名称 `cudasift`、`cuda_sift` 和 `cuda` 仍是 `cuda_opencv` 的兼容别名。

## 实现结构

- `include/colite/core/parallel.h`、`src/colite/core/parallel.cpp`：持久工作线程池和 `parallel_for_`；CPU SIFT 内核共享该实现。
- `src/colite/features/backends/sift_opencv.*`：OpenCV 语义的 CPU 重写版，包含多线程和 AVX2/FMA 路径。
- `src/colite/features/backends/sift_vlfeat.*`：VLFeat/COLMAP CPU SIFT 语义的优化版；公开头采用 opaque filter，避免与官方 VLFeat 的结构体 ABI 冲突。
- `src/colite/features/backends/cuda_sift/`：共享 CUDA 引擎，通过 traits 编译出 OpenCV 和 VLFeat 两种模式。旧 `cudasift/thirdparty` 实现已移除。
- `src/colite/features/backends/cuda_sift_extractor.cu`：COLITE 特征结构、RootSIFT、VLFeat descriptor bin 顺序和全局 response top-K 适配。

CPU AVX2 只加到两个 SIFT 源文件，不加到整个 `colite_core`，以避免调用方与库内部 Eigen 对齐 ABI 不一致。VLFeat 的梯度方向仍逐 lane 复用其标量 fast sqrt/atan2 数值契约；高斯、DoG、中心差分、极值检测和归一化继续使用 AVX2。

OpenCV-like 高斯边界使用周期折叠的 `REFLECT_101`/reflect 映射，支持卷积核宽于最小金字塔层，不会在极小图像层发生负索引或越界。

## 构建

CUDA、多线程和 AVX2 默认开启：

```bash
cmake -S . -B build-wsl-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-wsl-cuda -j$(nproc)
```

相关开关：

- `COLITE_ENABLE_CUDA_SIFT=ON`
- `COLITE_ENABLE_SIFT_PARALLEL=ON`
- `COLITE_ENABLE_AVX2=ON`

若没有 CUDA 编译器，CMake 会关闭 CUDA SIFT 并保留 CPU 后端。WSL 会自动尝试 `/usr/local/cuda/bin/nvcc`。

## 命令行

单步骤提取：

```bash
./build-wsl-cuda/src/colite/exe/colite feature_extractor \
  --image_path data/building-18 \
  --workspace_path output/features-auto \
  --feature_extractor auto \
  --max_num_features 8192 \
  --feature_num_threads -1
```

分别指定后端时，将 `--feature_extractor` 改为 `cvsift`、`vlsift`、`cuda_opencv` 或 `cuda_vlfeat`。`recon_sfm` 使用同名参数和相同的自动选择规则。CPU 模式中 `-1` 使用全部逻辑核；CUDA 模式固定单个外层图像 worker，避免多个任务争用同一 GPU。

## 测试

专项测试：

```bash
./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='ParallelRuntimeTest.*:SiftExtractorTest.*:SiftReferenceConsistencyTest.*'

ctest --test-dir build-wsl-cuda --output-on-failure -j4
```

测试覆盖：持久线程池、CPU 两后端、CUDA 两模式、`auto` GPU 优先、RootSIFT、response 全局 top-K、VLFeat 官方参考一致性，以及卷积核宽于最小金字塔层的边界回归。CUDA 13.2/RTX 4070 SUPER 构建中完整 CTest 为 268/268 通过；其中 24 个 CUDA PatchMatch 用例因本构建显式关闭 `COLITE_ENABLE_CUDA_MVS` 而跳过，与 CUDA SIFT 无关。

CPU 内存检查：

```bash
valgrind --error-exitcode=99 --leak-check=full \
  ./build-wsl-cuda/tests/colite_core_tests \
  --gtest_filter='SiftExtractorTest.CvSiftHandlesKernelWiderThanSmallestPyramidLevel:SiftReferenceConsistencyTest.VlFeatVariantMatchesOriginal'
```

验证结果为 `ERROR SUMMARY: 0 errors`、definitely lost 0 bytes。

## 性能记录

测试图像为 `data/building-18/P1180166.JPG`（3072×2304），`max_num_features=8192`，WSL 可用 20 个逻辑核。表中是 COLITE 命令报告的特征提取阶段时间；旧版基线与迁移后使用相同图像、参数和 workspace 冷缓存。

| 方法 | 线程 | 旧版 ms | 迁移后 ms | 相对旧版 | 验证 |
|---|---:|---:|---:|---:|---|
| CPU OpenCV | 1 | 18325 | 2262 | 8.10× | 通过 |
| CPU OpenCV | 20 | 18325 | 491 | 37.32× | 通过 |
| CPU VLFeat | 1 | 15487 | 7194 | 2.15× | 通过 |
| CPU VLFeat | 20 | 15487 | 915 | 16.93× | 通过 |
| CUDA OpenCV | GPU | — | 734 | — | 通过 |
| CUDA VLFeat | GPU | — | 730 | — | 通过 |
| auto（解析为 CUDA OpenCV） | GPU | — | 361 | — | 通过 |

这些是端到端 CLI 进程中的单次阶段记录，包含 CUDA 首次上下文和分配开销，不等同于进程内 warm-kernel microbenchmark。来源项目在相同分辨率、20 线程、`o_min=-1` 的多次进程内基准中记录 CPU OpenCV 约 396.7 ms、CPU VLFeat 约 453.4 ms；COLITE 的额外特征转换、RootSIFT、缓存写入边界及不同阈值/输出数量会影响 CLI 数值。

VLFeat 初次迁移后曾测得 5986 ms。原因是优化内核虽然支持并行，COLITE 封装仍串行调用每个 keypoint 的方向和描述子计算。封装现已改为预计算当前 octave 所需梯度、并行方向计算、一次性预分配输出以及并行描述子计算，因此同一 CLI 配置下降到 915 ms。

`max_num_features=8192` 下两种 CPU 后端的工作量并不相同：OpenCV 执行全局 top-K，输出严格为 8192 个方向点；COLMAP 风格 VLFeat 按完整 DoG level 保留 base keypoint，再为每点保留最多两个方向，本图输出 15247 个方向点。直接 API 计时分别为 393 ms 和 776 ms；VLFeat 输出量是 1.86 倍，耗时是 1.98 倍，单位输出成本已经接近。

## CPU 与 CUDA 重叠率

`SiftCpuCudaOverlapTest` 复用了 `D:/Code/CXX/Sift/demo/cuda_sift_compare.cpp` 的匹配口径：不限点数、`first_octave=-1`、全部 octave，候选门限为位置 1.5 px、尺度误差 8%、角度误差 8°；严格重叠门限为位置 0.05 px、尺度误差 0.1%、角度误差 0.1°，分母取 CPU/GPU 点数较大者。

在 `data/building-18/P1180166.JPG` 上的 COLITE 结果：

| 模式 | CPU 点数 | GPU 点数 | 匹配率 | 严格重叠率 | CPU | GPU |
|---|---:|---:|---:|---:|---:|---:|
| OpenCV | 76559 | 76841 | 99.9922% | 99.9740% | 564.6 ms | 213.4 ms |
| VLFeat | 89053 | 89042 | 99.9753% | 99.9607% | 574.7 ms | 56.4 ms |

VLFeat CUDA 适配层同步采用了 CPU/COLMAP 封装的 `+0.5` 像素中心坐标约定；修复前宽松匹配率为 99.9753%，但严格重叠率会因固定半像素偏移变为 0%。

同时在原 Sift 项目的 `P1180171.JPG` 上复跑其原生比较器，得到 OpenCV 49965 点、99.99% 匹配、99.95% 严格重叠；VLFeat 58975 点、99.98% 匹配、99.97% 严格重叠，与项目记录一致。

复现某个后端时必须使用新的 `--workspace_path`，否则命中特征缓存后测到的是缓存读取而不是提取。
