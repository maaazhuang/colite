# COLITE 与 COLMAP 3.11 SfM 分阶段验证

更新时间：2026-07-15

## 1. 结论摘要

本轮使用 `data/building-18` 的 18 张 3072×2304 JPEG，在同一 WSL 环境中比较当前 COLITE 与 `/usr/local/bin/colmap` 3.11.0。两边固定使用单个 PINHOLE 相机和相同内参：

```text
fx = fy = 2457.6
cx = 1536
cy = 1152
max_num_features = 4000
CPU threads = 20
```

主要结论：

1. COLITE CPU VLFeat 特征提取比 COLMAP 3.11 CPU 快约 1.82 倍，特征总数达到 COLMAP 的 97.94%。
2. 两边原始匹配总数达到 99.16% 一致，逐图像对匹配数 Pearson 相关系数为 0.9996，描述子匹配主体已经较接近。
3. COLITE 独立 exhaustive matcher 比 COLMAP 慢约 12.4 倍；当前 COLITE 对图像对串行匹配，是最明确的性能瓶颈。
4. COLITE 两视图几何只把 15 对判为 Essential，大量图像对被判为 Fundamental 或 Homography；COLMAP 的模型分类明显不同。这会影响初始化、track 和增量注册。
5. 独立三步骤流程中，COLITE 注册 14/18，COLMAP 注册 18/18。共同 14 张图的姿态很接近，但 COLITE 漏掉序列两端 4 张图。
6. COLITE 的 3D 点更多，但轨迹明显碎片化：49.5% 的点只有两个观测，COLMAP 仅 6.6%。
7. 当前集成 `recon_sfm` 只注册 4/18，比独立三步骤的 14/18 更差。集成命令的 relaxed nearby matching 和 weak-pair rematch 并未改善完整性，反而构成明确回归。

## 2. 测试产物

本轮输出保存在：

```text
build-wsl-cuda/sfm-colmap311-ab-20260715/
  colite/              COLITE 独立三步骤 workspace
  colmap/              COLMAP database、稀疏模型和日志
  colite-recon-sfm/    COLITE 集成 recon_sfm 结果
```

性能计时使用 `/usr/bin/time -v`。所有下游比较均使用 CPU VLFeat 特征，避免把 CUDA 与 COLMAP 3.11 的 CPU 路径混在同一个结论中。

注意：`data/building-18` 中还存在 `1.ini`。COLMAP image reader 会报告一次非图像读取错误，但实际仍正确处理 18 张 JPEG；正式基准应把非图像文件移出输入目录，消除额外噪声。

## 3. 特征提取

### 3.1 命令

COLITE：

```bash
./build-wsl-cuda/src/colite/exe/colite feature_extractor \
  --image_path data/building-18 \
  --workspace_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite \
  --focal_length 2457.6 \
  --max_num_features 4000 \
  --feature_extractor vlsift \
  --feature_num_threads -1
```

COLMAP：

```bash
/usr/local/bin/colmap feature_extractor \
  --database_path build-wsl-cuda/sfm-colmap311-ab-20260715/colmap/database.db \
  --image_path data/building-18 \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params 2457.6,2457.6,1536,1152 \
  --SiftExtraction.max_num_features 4000 \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.num_threads 20
```

### 3.2 结果

| 指标 | COLITE | COLMAP 3.11 | 说明 |
| --- | ---: | ---: | --- |
| Wall time | 14.58 s | 26.57 s | COLITE 快 1.82× |
| User time | 182.33 s | 288.26 s | 多线程累计 CPU 时间 |
| Peak RSS | 2,226,024 KB | 30,586,284 KB | COLMAP 同时创建 20 个 CPU extractor，内存非常高 |
| 特征总数 | 114,028 | 116,431 | COLITE/COLMAP = 97.94% |
| 平均每图 | 6,334.9 | 6,468.4 | 接近 |

按位置不超过 1.5 px、尺度相对误差不超过 8%、方向差不超过 8° 对齐关键点后：

| 指标 | 结果 |
| --- | ---: |
| COLITE 特征被 COLMAP 覆盖率 | 90.88% |
| 位置误差均值 / P95 | 0.167 px / 0.511 px |
| 尺度相对误差均值 / P95 | 0.88% / 3.04% |
| 方向误差均值 / P95 | 0.642° / 2.206° |
| 匹配关键点的描述子 byte-space L2 均值 | 17.60 |

结论：特征数量和匹配效果已经很接近，但生产分辨率下仍有约 9% 的关键点不能在上述门限内一一对应。后续应检查灰度转换、首 octave 上采样/平滑、DoG 极值阈值、边界处理和 COLMAP 3.11 vendored VLFeat 的细节；不过原始匹配已经高度一致，因此它不是当前 SfM 完整性的首要瓶颈。

## 4. 描述子匹配与两视图几何

### 4.1 公平参数

两边统一使用：

```text
Lowe ratio = 0.8
cross/mutual check = true
guided matching = false
two-view max error = 4 px
minimum inliers = 15
```

COLITE 的独立 `exhaustive_matcher` 没有 guided matching，因此本轮 COLMAP 也关闭该选项。

### 4.2 命令

```bash
./build-wsl-cuda/src/colite/exe/colite exhaustive_matcher \
  --workspace_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite \
  --matcher flann \
  --max_ratio 0.8
```

```bash
/usr/local/bin/colmap exhaustive_matcher \
  --database_path build-wsl-cuda/sfm-colmap311-ab-20260715/colmap/database.db \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.num_threads 20 \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.cross_check 1 \
  --SiftMatching.guided_matching 0 \
  --TwoViewGeometry.max_error 4 \
  --TwoViewGeometry.min_num_inliers 15
```

### 4.3 结果

| 指标 | COLITE | COLMAP 3.11 |
| --- | ---: | ---: |
| Wall time | 57.70 s | 4.64 s |
| User time | 56.66 s | 75.99 s |
| Peak RSS | 117,540 KB | 107,172 KB |
| 原始匹配 | 82,095 | 82,787 |
| 有原始匹配的图像对 | 153 | 150 |
| 有效几何图像对 | 133 | 135 |
| 有效几何内点 | 74,665 | 78,153 |

原始匹配总数比为 99.16%，共同图像对的匹配数 Pearson 相关系数为 0.9996。有效几何内点总数比为 95.56%，共同有效图像对的内点数相关系数为 0.9983。

COLITE 几何类型分布：

```text
CALIBRATED_ESSENTIAL       15 pairs
UNCALIBRATED_FUNDAMENTAL   52 pairs
PLANAR_HOMOGRAPHY          66 pairs
DEGENERATE                 20 pairs
```

COLMAP database 的配置分布：

```text
CALIBRATED / config 2      54 pairs
UNCALIBRATED / config 3    24 pairs
MULTIPLE / config 6        57 pairs
UNDEFINED / config 0       18 pairs
```

COLMAP 3.11 两视图几何默认使用 100 到 10,000 次自适应 RANSAC trial，并具有 MULTIPLE、watermark 和更完整的 E/F/H 竞争逻辑；COLITE 当前最多 1,000 次固定迭代，分类规则也更简单。不能只比较“内点总数”，必须优先对齐模型选择语义。

### 4.4 性能判断

COLITE 的 wall time 比 COLMAP 慢约 12.4 倍，但累计 CPU time 更少，说明主要问题是并行度不足而不是单个 pair 的计算量异常。当前独立 matcher 和集成 `recon_sfm` 都以双层循环串行处理图像对；COLMAP 创建 20 个 CPU matcher 并发处理。

## 5. 初始化、注册、三角化与 BA

### 5.1 命令

COLITE：

```bash
./build-wsl-cuda/src/colite/exe/colite mapper \
  --workspace_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite
```

COLMAP 固定内参，避免 COLMAP 优化焦距而 COLITE 保持常量造成口径差异：

```bash
mkdir -p build-wsl-cuda/sfm-colmap311-ab-20260715/colmap/sparse
/usr/local/bin/colmap mapper \
  --database_path build-wsl-cuda/sfm-colmap311-ab-20260715/colmap/database.db \
  --image_path data/building-18 \
  --output_path build-wsl-cuda/sfm-colmap311-ab-20260715/colmap/sparse \
  --Mapper.multiple_models 0 \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 0
```

### 5.2 结果

| 指标 | COLITE | COLMAP 3.11 |
| --- | ---: | ---: |
| Wall time | 5.66 s | 10.86 s |
| User time | 5.22 s | 11.49 s |
| Peak RSS | 154,660 KB | 87,724 KB |
| 注册图像 | 14/18 | 18/18 |
| 3D 点 | 11,233 | 9,148 |
| 观测 | 36,596 | 39,728 |
| 平均轨迹长度 | 3.258 | 4.343 |
| 两帧轨迹占比 | 49.51% | 6.59% |
| 长度至少 5 的轨迹占比 | 18.53% | 33.65% |

COLITE 未注册图像：

```text
P1180166.JPG
P1180167.JPG
P1180168.JPG
P1180183.JPG
```

COLITE 的点更多并不代表模型更完整。它的观测反而更少，而且接近一半的 3D 点只有两个观测，说明 track completion、track merging 或多视图重三角化明显弱于 COLMAP。

### 5.3 重投影误差

COLITE 导出的 `points3D.txt` 将每个点的 ERROR 写为 `-1`，因此 COLMAP `model_analyzer` 会错误显示 `0 px`。本轮直接从相机、姿态、3D 点和每条 2D 观测重新投影：

| 观测加权误差 | COLITE | COLMAP 3.11 |
| --- | ---: | ---: |
| Mean | 0.675 px | 0.838 px |
| Median | 0.555 px | 0.715 px |
| P95 | 1.696 px | 1.962 px |
| P99 | 2.485 px | 2.738 px |

COLMAP 自身 `model_analyzer` 根据点文件存储误差报告 0.79075 px。两种统计的聚合口径不同，跨实现对比时应统一使用逐观测重投影脚本。COLITE 的较低误差是在漏注册 4 张图并保留更短轨迹的条件下得到，不能单独解读为整体精度更高。

### 5.4 共同相机姿态

将 COLITE 的 14 个共同相机中心通过 7DoF 相似变换对齐到 COLMAP：

| 指标 | 结果 |
| --- | ---: |
| 相机中心平均误差 | 0.00771（COLMAP 模型尺度） |
| 相机中心误差 / 相邻相机中位基线 | 0.91% |
| 相机中心最大误差 | 0.01712 |
| 旋转平均差 | 0.117° |
| 旋转最大差 | 0.320° |

这说明 COLITE 对已经注册的主体序列恢复得较准确。主要问题是注册完整性、鲁棒估计和轨迹组织，而不是共同相机整体发生严重漂移。

## 6. 集成 `recon_sfm` 回归

命令：

```bash
./build-wsl-cuda/src/colite/exe/colite recon_sfm \
  --image_path data/building-18 \
  --output_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite-recon-sfm/model \
  --focal_length 2457.6 \
  --max_num_features 4000 \
  --feature_extractor vlsift \
  --feature_num_threads -1 \
  --matcher flann
```

结果：

| 指标 | 集成 recon_sfm |
| --- | ---: |
| Wall time | 85.94 s |
| User time | 263.87 s |
| Peak RSS | 2,331,832 KB |
| 注册图像 | 4/18 |
| 3D 点 | 3,868 |
| 观测 | 9,524 |
| 平均重投影误差 | 0.428 px |
| 原始匹配 | 86,341 |
| 几何内点 | 79,849 |

集成命令只注册 `P1180178.JPG` 到 `P1180181.JPG`。它比独立三步骤产生更多匹配和内点，但注册数从 14 降到 4，说明“更多内点”中存在会破坏 track 或 PnP 的不一致对应。

实现层面的差异：

1. 独立 `exhaustive_matcher` 默认全局 ratio 0.8。
2. 集成 `recon_sfm` 默认把文件序距离不超过 2 的 pair 放宽到 ratio 0.9。
3. 集成流程还会用不低于 0.95 的 ratio 对弱 pair 做 guided rematch。
4. 集成流程没有复用独立 controller 的同一套缓存和参数对象，因此两条路径会产生不同 MatchGraph。

此外，`nearby_match_window=0` 在主匹配逻辑中表示关闭 nearby relaxation，但在 `WeakPairRematcher::IsNearbyPair` 中表示所有 pair 都是 nearby，存在明确的参数语义冲突。

## 7. 优化优先级

### P0：统一独立步骤与集成流程

让 `recon_sfm` 直接编排 `feature_extractor`、matcher、geometry verifier 和 mapper 的公共 stage/controller，不再维护第二套内存内实现。所有 ratio、guided matching、弱对重匹配和随机种子必须写入同一配置及 cache hash。

建议验收：同一参数下，独立三步骤和 `recon_sfm` 的特征数、每对 raw matches、几何类型、内点索引、初始化 pair 和注册结果完全一致。

### P0：替换当前 DLT PnP + 100 次固定 RANSAC

当前注册器使用 6 点 DLT minimal sample、固定随机种子和仅 100 次 trial。实际诊断中常见上千个候选对应却只得到十几个到几十个内点；低内点率下，100 次抽到 6 个全内点样本的概率极低。

对标 COLMAP 3.11：

1. 标定相机使用 P3P minimal solver。
2. 使用 EPnP 做 LO-RANSAC 的局部重估计。
3. 自适应 RANSAC，默认 100 到 10,000 trials，置信度 0.99999。
4. 对内点做非线性 absolute pose refinement。
5. 加入动态 trial 终止、退化样本检查、cheirality 和 reprojection support 评分。

这是把 14/18 提升到 18/18 最直接的 mapper 优化方向。

### P0：对齐 E/F/H/MULTIPLE 两视图模型选择

不应继续仅通过调低/调高 `min_E_F_inlier_ratio` 修补。需要按 COLMAP 3.11 的 two-view geometry 流程实现或等价实现：

1. 独立、归一化且自适应的 E/F/H robust estimation。
2. 100 到 10,000 RANSAC trials 和 local optimization。
3. 按 support 比较 E/F/H，并支持 MULTIPLE，而不是过早把大量 pair 固定成 Homography。
4. 保留 watermark/重复纹理退化检测。
5. 用固定 raw matches 对两边输出逐 pair 比较模型类型、内点集合 Jaccard 和相对姿态。

### P1：并行化 exhaustive matching

第一阶段可以先用新的 `core/parallel` 对 153 个 pair 并行调度，每个 worker 使用线程私有 matcher/临时缓冲。进一步应像 COLMAP 一样缓存每张图的描述子索引，避免同一图像在不同 pair 中反复构建 FLANN 索引。

建议验收：20 线程 wall time 先从 57.7 s 降到 8 s 内，再以 COLMAP 4.64 s 为后续目标；raw match 索引必须与串行基线完全一致或达到逐 pair 99.99% 一致。

### P1：修复 track 分裂和多视图三角化

重点检查：

1. 构建 correspondence graph 时同一 feature 是否被重复创建为多个 component。
2. track merge 的冲突规则是否过于保守，或错误接受一张图中同一 track 的多个 feature。
3. 新图注册后是否完整执行 continue/complete/merge/retriangulate。
4. 三角化是否只依赖 pair，而没有基于完整 track 做鲁棒多视图估计。
5. observation filtering 后是否触发重新合并与重三角化。

验收目标：两帧 track 占比从 49.5% 降到 10% 以下，平均轨迹长度接近 COLMAP 的 4.34，同时保持 18/18 注册。

### P1：先约束 weak-pair rematch，再重新启用

在 PnP 和 track graph 修复前，默认集成流程不应无条件使用当前 0.9/0.95 relaxed matching。建议增加：

1. 原/新几何内点集合的一致性和相对姿态变化门限。
2. 新增匹配必须通过双向 guided check。
3. 除绝对内点增长外，还要求 inlier ratio 不下降。
4. 对初始化主链路做回归门禁；若注册数下降则拒绝替换旧几何。
5. 统一 `nearby_match_window=0` 的语义。

### P2：完善模型质量输出

导出 `points3D.txt` 时计算真实 per-point reprojection error，不能固定为 `-1`。同时在 summary 中固定输出：注册率、观测数、平均/中位 track length、长度 2 track 比例、逐观测 reprojection mean/P50/P95/P99、过滤点数和 BA 前后误差。

### P2：继续收敛 VLFeat 特征一致性

在 mapper P0/P1 完成后，再用相同图像和不限点数模式逐 octave 比较：

1. Gaussian pyramid 每层像素差。
2. DoG extrema 的 octave/layer/x/y。
3. border reject 原因统计。
4. orientation 数量和方向差。
5. descriptor 每维误差。

当前原始匹配已经达到 99.16% 总量一致，因此不建议先为最后约 9% 的关键点定位差异暂停 mapper 鲁棒性工作。

## 8. 后续单模块 A/B 测试设计

仅使用各自上游输出会让误差逐级累积，无法准确定位单模块。建议增加三类交换适配器：

1. feature adapter：COLMAP database keypoints/descriptors 与 COLITE FeatureSet 双向转换。
2. match adapter：固定同一组 raw matches，分别输入两边的 geometry verifier。
3. geometry adapter：固定同一组 verified matches/relative poses，分别输入 mapper。

建议测试矩阵：

| 测试 | 固定输入 | 被测模块 | 核心指标 |
| --- | --- | --- | --- |
| SIFT | 同一灰度图 | extractor | 点数、重复率、位置/尺度/方向、描述子误差、时间 |
| Matcher | 同一描述子 | descriptor matcher | pair match Jaccard、ratio 边界、时间 |
| Geometry | 同一 raw matches | E/F/H verifier | 类型、内点 Jaccard、相对姿态、时间 |
| Initializer | 同一 verified graph | initializer | pair 排名、初始点数、初始误差 |
| Registrar | 固定 2D-3D correspondences | PnP | 成功率、内点、姿态误差、trials、时间 |
| Triangulator | 固定相机和 tracks | triangulation | 点数、track length、角度、误差 |
| BA | 固定 reconstruction | local/global BA | cost、误差分位数、迭代数、时间 |

持续集成至少保留一个小型合成真值集和 `building-18` 真实集。最终门禁应同时要求：18/18 注册、共同姿态误差不退化、平均轨迹长度不低于基线、P95 重投影误差不过阈值，以及阶段 wall time 不回退。

## 9. P0/P1/P2 优化复测（2026-07-15）

本轮实现了以下改动：

1. P0 两视图几何：RANSAC 改为 100--10,000 次自适应迭代并增加局部优化；检测到 PoseLib 时使用五点 Essential，否则保留八点回退。
2. P0 图像注册：检测到 PoseLib 时使用 P3P 最小解，使用全内点 DLT 做局部优化，并采用自适应 RANSAC；最大迭代从 100 提高到 10,000。
3. P0 流程一致性：`recon_sfm` 与独立命令统一为 FLANN、ratio 0.8、默认不做 nearby relaxed rematch；weak-pair 替换还要求 inlier ratio 不下降。
4. P1 匹配并行化：所有 pair 先通过 `core::parallel_for_` 并行计算，再串行、确定性地写入 `MatchGraph`。
5. P1 track：合并点使用完整 track 多视图三角化；全局 completion 进行多轮传播；允许同一图像中距离不超过 1 px 的重复方向特征被识别为同一观测，而不是无条件拒绝合并。
6. P2 输出与边界：`points3D.txt` 写入逐 track RMS 重投影误差；增加极小图像 reflect-border 回归；VLFeat 优化/参考实现重叠门槛提高到 99.9%。

### 9.1 两视图几何与并行匹配

固定原始 matches 重新验证时，COLITE 得到 135 个有效 pair 和 77,625 个内点；COLMAP 3.11 为 135 个 pair 和 78,153 个内点，内点总数达到 99.32%。

复制 features 到无缓存工作区后运行完整 matcher：

```bash
./build-wsl-cuda/src/colite/exe/colite exhaustive_matcher \
  --workspace_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite-parallel \
  --matcher flann --max_ratio 0.8
```

| 指标 | 优化前 | 优化后 | COLMAP 3.11 |
| --- | ---: | ---: | ---: |
| Wall time | 57.70 s | 11.99 s | 4.64 s |
| CPU 利用率 | 约 98% | 830% | 多线程 |
| Raw matches | 82,095 | 82,069 | 82,787 |
| 有效几何内点 | 74,665 | 约 77.6k | 78,153 |

并行化带来 4.81 倍 wall-time 加速。与 COLMAP 的剩余效率差距主要来自每个 pair 内部的 FLANN/RANSAC 实现和任务粒度，而不是 pair 外层仍然串行。

### 9.2 Mapper 与端到端流程

独立 mapper 在固定优化后几何上的结果：

| 指标 | 优化前 COLITE | 优化后 COLITE | COLMAP 3.11 |
| --- | ---: | ---: | ---: |
| 注册图像 | 14/18 | 18/18 | 18/18 |
| Wall time | 5.66 s | 5.22 s | 10.86 s |
| 3D 点 | 11,233 | 15,578 | 9,148 |
| 观测 | 36,596 | 50,416 | 39,728 |
| 平均 track length | 3.258 | 3.236 | 4.343 |
| 长度 2 track 比例 | 49.51% | 52.43% | 6.59% |
| 模型存储平均误差 | 无效的 0 px | 0.655 px | 0.791 px |

注册完整性 P0 已解决，而且完整模型的观测覆盖显著增加。P1 的多视图三角化和合并已生效，但短 track 比例仍未达到低于 10% 的验收目标：点数增长快于长 track 合并，下一轮应实现显式 correspondence connected-component/track builder，并在建点前消除跨 pair 的重复 component，而不是继续放宽 merge 阈值。

从零运行集成命令：

```bash
./build-wsl-cuda/src/colite/exe/colite recon_sfm \
  --image_path data/building-18 \
  --output_path build-wsl-cuda/sfm-colmap311-ab-20260715/colite-recon-sfm-final/model \
  --focal_length 2457.6 --max_num_features 4000 \
  --feature_extractor vlsift --feature_num_threads -1 --matcher flann
```

结果由优化前的 4/18、86.43 秒改善为 18/18、30.93 秒（2.79 倍加速），输出 15,604 个点、51,541 个观测，平均 track length 3.303，COLMAP analyzer 读取的平均点误差为 0.660 px。

### 9.3 回归结果

`colite_core_tests` 共包含 274 项：250 项通过，24 项因本构建未启用 CUDA MVS 而跳过；独立的 `MvsSweepScriptTest` 通过。P0、P1 和 P2 新增/相关的几何、PnP、轨迹合并、并行框架、SIFT 极小图像和 VLFeat 99.9% 一致性测试均通过。

## 10. Sequential 与 vocabulary-tree matcher A/B（2026-07-15）

测试继续使用 building-18、ratio 0.8、mutual check、4 px 几何阈值和至少 15 个内点。所有正式计时均为单进程、无 matcher cache、Release 构建；最初并发运行得到的时间因 CPU 争用而废弃。

### 10.1 Sequential matcher

两边统一使用 overlap=5。COLMAP 额外设置 `quadratic_overlap=0` 和 `loop_detection=0`，使候选集合与 COLITE 的线性前向窗口完全相同。

| 指标 | COLITE | COLMAP 3.11 | COLITE/COLMAP |
| --- | ---: | ---: | ---: |
| Wall time | 6.44 s | 6.89 s | 0.935 |
| User CPU time | 59.63 s | 25.34 s | 2.35 |
| 平均 CPU 利用率 | 946% | 368% | 2.57 |
| Peak RSS | 196,100 KB | 106,400 KB | 1.84 |
| 候选/有 raw match 的 pair | 75 | 75 | 100% |
| Raw matches | 73,216 | 73,941 | 99.02% |
| 有效几何 pair | 75 | 75 | 100% |
| 有效几何内点 | 70,561 | 71,067 | 99.29% |
| Raw pair-count Pearson | \- | \- | 0.999424 |
| Inlier pair-count Pearson | \- | \- | 0.999550 |

COLITE wall time 只快约 6.5%，但消耗更多累计 CPU 和内存。原因是 COLITE 使用 `core::parallel_for_` 同时调度全部 pair，获得接近 9.5 核的平均并行度；COLMAP 3.11 的 sequential pairing 以图像批次推进，本次平均约 3.7 核。COLITE 是用更多并行资源换取小幅 wall-time 优势，不能解读为单 pair matcher 更高效。

下游 mapper：COLITE 注册 18/18，5.26 秒，生成 15,491 点、50,659 观测，平均 track length 3.270；COLMAP 同样注册 18/18，mapper wall time 9.93 秒。

### 10.2 Vocabulary-tree matcher

两边设置每图检索 10 张候选、每图最多 5,000 descriptors、256 visual words。两边的词汇模型格式和训练算法并不兼容，因此分别从各自的 building-18 descriptors 本地训练。

| 指标 | COLITE | COLMAP 3.11 |
| --- | ---: | ---: |
| 词汇模型构建 + 检索匹配总时间 | 15.93 s | 52.00 s |
| 其中独立 tree build | 包含在总时间 | 42.41 s |
| 已有 tree 后的 matcher | 当前无法单独复用临时训练计时 | 9.59 s |
| Peak RSS（matcher） | 226,176 KB | 152,192 KB |
| 有 raw match 的 pair | 112 | 108 |
| Raw matches | 74,023 | 72,057 |
| 有效几何 pair | 101 | 97 |
| 有效几何内点 | 70,538 | 68,519 |

候选 raw-pair 集合共有 89 对、并集 131 对，Jaccard 为 67.94%；有效几何集合共有 80 对、并集 118 对，Jaccard 为 67.80%。但是共同 pair 上 raw match count Pearson 为 0.999629，inlier count Pearson 为 0.999646。这说明低层描述子匹配和几何验证仍高度一致，主要差异来自图像检索候选。

上述数据对应优化前实现：当时 `vocab_tree_matcher` 对每图按步长截取最多 5,000 个 descriptors，执行 6 轮 deterministic flat k-means 后做 TF-IDF，因此不能把其总耗时与 COLMAP 的持久化层次 vocabulary tree 直接等价比较。COLMAP 本次使用全部 116,431 descriptors、11 轮量化并构建 FLANN 层次索引；首次构建慢，但 tree 可跨数据集/运行复用。若只比较已有 tree 的匹配阶段，COLMAP 的 9.59 秒快于优化前 COLITE 包含临时训练的 15.93 秒。

### 10.8 四项后续优化（2026-07-15）

- 新增 `vocab_tree_builder`，输出 `COLITE_VOCAB_TREE 2` 持久化层次树；matcher 可直接复用，且继续只读兼容 v1 平面词表。
- vocab retrieval 默认合并后续 2 帧，避免相邻视角因视觉词量化差异而从候选图中断开；可用 `--vocab_tree_sequential_overlap 0` 做纯 retrieval A/B。
- 新增候选召回与图结构诊断：retrieval pairs、时序边 recall/added、connected components、degree min/mean/max。
- sequential/vocab 的 FLANN 路径改为每图预量化一次并构建一次索引，图像对任务复用索引；这针对优化前 75 对匹配重复约 150 次量化/建索引造成的高 user CPU。索引查询按图加锁，避免并行任务并发访问同一 FLANN index 的未定义行为。

随后以用户上下文进入 Ubuntu-22.04 WSL，完成了新二进制编译与 building-18 复测：

| 路径 | wall | user CPU | CPU 利用率 | max RSS | 候选/验证 pair | 验证内点 |
|---|---:|---:|---:|---:|---:|---:|
| sequential 优化前 | 6.44 s | 59.63 s | 946% | 196100 KB | 75/75 | 70561 |
| sequential 索引复用后 | 6.47 s | 29.92 s | 482% | 168048 KB | 75/75 | 70584 |
| vocab 优化前（含临时平面聚类） | 15.93 s | 81.78 s | 534% | 226176 KB | 112/101 | 70538 |
| 层次词树 builder | 1.47 s | 0.90 s | 61% | 141640 KB | - | - |
| vocab 复用持久树后 | 11.89 s | 47.33 s | 408% | 196768 KB | 115/104 | 70948 |

Sequential wall 基本不变，因为几何验证仍是关键路径；但 user CPU 降低 49.8%，峰值内存降低 14.3%，达到了“降低实际计算量而非只增加并发”的目标。Vocab 纯 matcher 相对旧的混合计时 wall 降低 25.4%、user CPU 降低 42.1%；即使计入一次性的 1.47 秒建树，首次运行仍缩短 16.1%。

优化后 vocab 得到 113 个 retrieval pairs；33 条 overlap=2 时序保护边中已召回 31 条（93.94%），补入 2 条。最终候选图为 1 个连通分量，度数 min/mean/max=10/12.78/16。下游 mapper 仍注册 16/18、生成 12,972 点；失败日志显示剩余图像已有多个已注册邻居和数千验证内点，但 PnP RANSAC 失败，因此瓶颈已不是候选图连通性。后续应优化 correspondence-to-track 一致性/PnP，而不是继续盲目增加 vocab 候选数。

验证结果：WSL CUDA build 成功；匹配与命令定向测试 27/27 通过；完整 `ctest` 278/278 通过，其中 24 个需要实际 CUDA device 的测试按环境条件跳过。

COLITE vocab 候选图的下游 mapper 只注册 16/18，遗漏 `P1180166.JPG` 和 `P1180167.JPG`，生成 12,405 点、41,646 观测。虽然候选数和内点总量更高，但检索图没有为序列开头保留足够稳定的连通边。因此下一步不能继续以“更多候选/更多内点”为目标，而应：

1. 实现可持久化的层次 vocabulary tree 和独立 builder。
2. 对 retrieval score 做 COLMAP 式验证与候选截断，而不是 flat histogram top-N。
3. vocab 模式默认并入少量 sequential 邻接边，保证局部连通性。
4. 增加候选 recall、图连通分量、最小度数和 mapper 注册率门禁。

### 10.3 为什么不能概括为“COLITE 都快于 COLMAP”

当前实测应分阶段解读：

- CPU SIFT：COLITE 的多线程/AVX2 优化使 wall time 更短。
- Exhaustive matcher：COLITE 11.99 秒，COLMAP 4.64 秒，COLITE 仍慢约 2.58 倍。
- Sequential matcher：COLITE 仅快约 6.5%，代价是明显更高的 CPU time 和内存。
- Vocabulary tree 首次运行：COLITE 更快是因为实现更轻量、训练目标不同；复用完整 tree 后 COLMAP matcher 更快。
- Mapper：当前 COLITE 在 building-18 上更快，但仍生成更多短 track，质量结构尚未完全对齐。

因此 wall time 优势主要来自更激进的外层并行、轻量化实现或减少了 COLMAP 所做的工作；必须同时检查 CPU time、内存、候选集合、注册率和 track 质量，不能只比较一个耗时数字。
