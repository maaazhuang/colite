# COLITE 当前开发状态记录

更新时间：2026-06-18

本文档用于后续开发或其他对话快速接续项目上下文。当前项目目标是实现一套自研的轻量三维重建系统，整体 SfM 路线保守参考 COLMAP，必要处吸收 openMVG 的清晰数据结构思想。当前开发平台以 WSL Ubuntu 22.04 + CMake + GCC/Ninja 为主。

## 开发环境

主机环境：

- Windows 工作区：`D:\Code\CXX\colite`
- WSL 发行版：`Ubuntu-22.04`
- 构建系统：CMake + Ninja
- 编译器：GCC/G++
- 主要依赖：OpenCV、Eigen、Ceres Solver、glog、FLANN、GTest
- 推荐构建目录：`build-wsl`

基础构建命令：

```bash
cd /mnt/d/Code/CXX/colite
cmake --preset wsl-debug
cmake --build --preset wsl-debug
ctest --test-dir build-wsl --output-on-failure
```

或者直接：

```bash
cmake -S . -B build-wsl -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
```

## COLMAP 参考环境

WSL 中已卸载 apt 源的 COLMAP 3.7，并从 `references/colmap-3.11.0` 源码编译安装 COLMAP 3.11：

```text
/usr/local/bin/colmap
COLMAP 3.11.0 -- Structure-from-Motion and Multi-View Stereo
without CUDA
```

COLMAP 3.11 构建配置：

- `CMAKE_BUILD_TYPE=Release`
- `CUDA_ENABLED=OFF`
- `GUI_ENABLED=OFF`
- `OPENGL_ENABLED=OFF`
- `CMAKE_INSTALL_PREFIX=/usr/local`

COLMAP 对比基线输出目录：

```text
build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided
```

对比时使用 building-18 数据集，参数保持：

```bash
colmap feature_extractor \
  --database_path build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided/database.db \
  --image_path data/building-18 \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params 2457.6,2457.6,1536,1152 \
  --SiftExtraction.max_num_features 4000 \
  --SiftExtraction.use_gpu 0

colmap exhaustive_matcher \
  --database_path build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided/database.db \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.cross_check 1 \
  --TwoViewGeometry.max_error 4 \
  --TwoViewGeometry.min_num_inliers 15 \
  --SiftMatching.guided_matching 1 \
  --SiftMatching.use_gpu 0

colmap mapper \
  --database_path build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided/database.db \
  --image_path data/building-18 \
  --output_path build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided/sparse \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 1.0 \
  --Mapper.tri_min_angle 1.0
```

注意：COLMAP 3.11 中二视图几何参数从 3.7 的 `SiftMatching.max_error/min_num_inliers` 迁移到 `TwoViewGeometry.max_error/min_num_inliers`。另外，即使设置 `SiftExtraction.max_num_features=4000`，building-18 上 COLMAP 实际输出约 4883 到 8227 个特征，这会影响与 COLITE 的前端公平对比。

## 已实现功能

### core

- 基本 ID 类型、相机、图像、二维点、三维点、track、pose、two-view geometry、match graph、reconstruction 数据结构。
- 数据结构主体按 COLMAP 组织。
- 相机模型已建立模型族逻辑，当前实现重点为：
  - `PINHOLE`
  - `SIMPLE_RADIAL`
- 当前暂未实现通过 `|` 动态指定优化参数的机制；优化参数仍主要由相机模型类型和 BA 配置决定。

### features

- SIFT 特征提取模块。
- 支持三种可切换 SIFT 后端：
  - `opencv`：保留原 OpenCV `cv::SIFT` 直接调用路径。
  - `vlsift`：接入从 `references/sift/sift_vlfeat` 复制到项目内的改造 VLFeat SIFT，并已按 COLMAP CPU SIFT 封装方式改进。
  - `cvsift`：接入从 `references/sift/sift_opencv` 复制到项目内的改造 OpenCV-like SIFT。
- 支持最大特征数量配置。
- 已做 RootSIFT/归一化相关处理。
- 当前与 COLMAP 差距主要仍集中在特征提取策略细节，例如尺度、方向、多方向 keypoint、特征数量语义等。

`vlsift` 当前已对齐 COLMAP CPU SIFT 的关键封装逻辑：

- 灰度输入按 `[0, 1]` 传入 VLFeat。
- 默认 peak threshold 使用 `0.02 / octave_resolution`。
- 默认 octave 数为 4。
- 每个 keypoint 最多保留 2 个方向。
- keypoint 坐标使用 COLMAP 的像素中心约定，写入时 `x/y + 0.5`。
- VLFeat descriptor 转为 UBC/SiftGPU bin 顺序。
- 特征数量限制按 DoG level 保留大尺度层，而不是简单截断到精确数量。

### matching

- 描述子匹配：
  - OpenCV 分支。
  - FLANN 分支。
  - 自研/独立 FLANN 路线按 COLMAP 思路推进，尽量降低对 OpenCV matcher 的依赖。
- 二视图几何验证。
- exhaustive pair 处理。
- guided matching。
- 二阶段 weak pair rematching。
- pair-level 诊断输出，便于与 COLMAP SQLite 中的 raw matches 和 verified inliers 对比。

### sfm

- 增量 SfM 初始化。
- 新图像注册。
- PnP/注册候选调度。
- 增量三角化。
- re-triangulation。
- 局部 BA。
- 全局 BA。
- 全局 BA 后过滤，再全局 BA。
- 坏点/坏观测过滤。
- re-triangulation 后局部过滤和三角角度分层。
- 更接近 COLMAP 的增量建图调度仍在持续收敛中。

### app/recon_sfm

`recon_sfm` 命令行已可完成 building-18 的端到端稀疏重建。

主要输出：

- `summary.txt`：注册图像数、点数、观测数、平均/最大重投影误差等。
- `pair_diagnostics.tsv`：每对图像的 raw matches、verified inliers、几何类型等。
- COLMAP text model。
- PLY 点云导出。

典型运行命令：

```bash
./build-wsl/app/recon_sfm/recon_sfm \
  --image_path data/building-18 \
  --output_path build-wsl/compare-building-18/controlled-colite-4000-pinhole \
  --focal_length 2457.6 \
  --max_num_features 4000 \
  --feature_extractor opencv \
  --matcher flann
```

8000 特征对比命令：

```bash
./build-wsl/app/recon_sfm/recon_sfm \
  --image_path data/building-18 \
  --output_path build-wsl/compare-building-18/controlled-colite-8000-pinhole \
  --focal_length 2457.6 \
  --max_num_features 8000 \
  --feature_extractor opencv \
  --matcher flann
```

## building-18 对比记录

数据集：

- 路径：`data/building-18`
- 图像数：18
- 图像尺寸：3072 x 2304
- 对比焦距：2457.6
- 主点：1536, 1152
- 相机模型：PINHOLE

### COLMAP 3.11 基线

输出目录：

```text
build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided
```

模型统计：

```text
registered_images 18
points3d 10971
num_observations 47172
mean_reprojection_error 0.664276 px
verified_pairs 136 / 153
verified_inliers_sum 95152
raw_matches_sum 82760
```

### COLITE max_num_features=4000

输出目录：

```text
build-wsl/compare-building-18/controlled-colite-4000-pinhole
```

结果：

```text
registered_images 18
points3d 9694
num_observations 26653
mean_reprojection_error 0.39192184318155171
max_reprojection_error 3.9554044136174462
verified_pairs 122 / 153
verified_inliers_sum 48845
raw_matches_sum 68767
```

与 COLMAP 3.11 的 pair-level 对比表：

```text
build-wsl/compare-building-18/controlled-pair-comparison-colite-vs-colmap311.tsv
```

主要结论：4000 特征时 COLITE 后端可完成 18/18 注册，但前端 verified inliers 明显低于 COLMAP，差距主要集中在相邻/近邻图像对。

### COLITE max_num_features=8000

输出目录：

```text
build-wsl/compare-building-18/controlled-colite-8000-pinhole
```

结果：

```text
registered_images 18
points3d 19119
num_observations 58490
mean_reprojection_error 0.42014804810277023
max_reprojection_error 4.4725107793290899
verified_pairs 135 / 153
verified_inliers_sum 115641
raw_matches_sum 153185
```

与 COLMAP 3.11 的 pair-level 对比表：

```text
build-wsl/compare-building-18/controlled-pair-comparison-colite8000-vs-colmap311.tsv
```

主要结论：8000 特征时 COLITE 的前端匹配数量显著提升，verified inliers 总量超过 COLMAP 3.11，但点数和观测数也明显增多，最大重投影误差升高，存在过补偿倾向。

### 三种特征后端快速对比

为了提高运行效率，使用 building-18 的 1024 宽缩小图像集，焦距按比例缩放为 `819.2`，统一 `max_num_features=1500`、`matcher=flann`。

```text
build-wsl/compare-building-18/building-18-width1024
```

当前结果：

```text
opencv: registered_images 18, points3d 3904, observations 11248, mean_error 0.32561495087900244, verified_pairs 122, verified_inliers 14569
vlsift-colmaplike: registered_images 18, points3d 5728, observations 18520, mean_error 0.36856673633651904, verified_pairs 139, verified_inliers 33098
cvsift: registered_images 18, points3d 3966, observations 9731, mean_error 0.33168452236870283, verified_pairs 123, verified_inliers 13560
```

改进前的 `vlsift` 在同一设置下只注册 14/18，`verified_pairs=119`、`verified_inliers=10130`。改进后已提升到 18/18，前端覆盖明显增强。

典型中后段近邻 pair：

```text
P1180174-P1180175: COLITE 1351, COLMAP 1335
P1180175-P1180176: COLITE 1461, COLMAP 1564
P1180180-P1180181: COLITE 2182, COLMAP 1654
P1180180-P1180182: COLITE 1308, COLMAP 902
P1180181-P1180182: COLITE 2089, COLMAP 1383
```

## 当前判断

1. 整体 SfM 流程已贯通：特征、匹配、几何验证、初始化、注册、三角化、局部 BA、全局 BA、过滤、导出均已具备。
2. COLITE 后端路线没有明显偏离 COLMAP，building-18 可以稳定 18/18 注册。
3. 4000 特征时主要短板是前端特征/匹配数量不足。
4. 8000 特征时匹配数量超过 COLMAP，但质量控制和过滤需要继续收紧。
5. 下一阶段不建议盲目继续增加特征数，应在 6000 左右测试平衡点，并继续对齐 COLMAP 的 SIFT/匹配/过滤细节。

## 后续优先事项

建议按以下顺序推进：

1. 特征提取 COLMAP 化：
   - 对齐 SIFT octave、peak threshold、edge threshold、first octave、尺度空间策略。
   - 研究 COLMAP 特征数超过 `max_num_features` 的原因，尤其多方向 keypoint 或内部保留策略。
   - 增加每图像特征数量、尺度、响应值分布诊断。

2. 匹配质量控制：
   - guided matching 后增加重复匹配/弱几何 pair 的更细过滤。
   - 分析 raw matches 大幅超过 COLMAP 但是否贡献稳定 track。
   - 按 pair 类型区分相邻、近邻、远邻阈值。

3. Track 与三角化过滤：
   - 继续对齐 COLMAP observation manager 的 track 合并、冲突处理、角度过滤。
   - 加强最终全局 BA 后的 observation-level 残差过滤。

4. 增量调度细节：
   - 继续比较注册顺序、候选评分、local BA 图像窗口选择。
   - 输出每轮注册图像、可见 2D-3D 数、三角化新增点、过滤点数。

5. 参数扫描：
   - 建议新增 `max_num_features=6000`、`7000` 的 building-18 对比。
   - 目标不是单纯追求更多点，而是在观测数、误差、pair 覆盖和点云质量之间找到更接近 COLMAP 的平衡。

## 常用验证命令

运行全部测试：

```bash
ctest --test-dir build-wsl --output-on-failure
```

运行 recon_sfm 集成测试：

```bash
ctest --test-dir build-wsl -R recon_sfm --output-on-failure
```

查看 COLMAP 模型统计：

```bash
colmap model_analyzer \
  --path build-wsl/compare-building-18/controlled-colmap-311-4000-pinhole-guided/sparse/0
```

检查 COLMAP 版本：

```bash
which colmap
colmap -h | head -8
```

## 注意事项

- Windows 上直接运行 git 可能因 Codex 沙箱用户不同触发 `dubious ownership` 警告；这不代表仓库损坏。
- 不要执行 `wsl --unregister Ubuntu-22.04`，否则会删除 WSL 发行版数据。
- 如果 WSL 短暂出现 `WSL_E_DISTRO_NOT_FOUND`，优先重启 WSL 或 Windows，不要贸然重装发行版。
- `references/`、`build-wsl/`、对比输出目录一般不进入 git。
