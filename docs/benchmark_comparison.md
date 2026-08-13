# Colite 与开源重建项目对比测试

本文保留可指导后续开发的代表性结果。它不是持续增长的实验日志；每次新测试的原始命令、日志、JSON 和产物应保存到独立 `output/` 目录，并只把稳定结论更新到本文。

## 数据集与环境

| 数据集 | 图像 | 尺寸 | 常用相机 |
|---|---:|---:|---|
| `data/building-18` | 18 | 3072×2304 | PINHOLE，`f=2457.6`，`cx=1536`，`cy=1152` |
| `data/house` | 36 | 4592×3056 | PINHOLE，`f=5500`，`cx=2296`，`cy=1528` |

主要测试环境为 WSL Ubuntu 22.04、RTX 4070 SUPER、CUDA 13.2。参考版本为 COLMAP 3.11/4.0、OpenMVG 2.1 和 OpenMVS 2.4。

不同表格可能使用不同输入模型、工作尺寸和参数，只允许在各自表内比较，不能跨表直接推导速度或精度。

## SfM：COLMAP 3.11 结构对齐

`building-18` 使用相同缓存特征/几何，比较 IncrementalTriangulator point-centric Create、angular Continue 和 complete/merge 后的 Mapper 结果：

| 模型 | 注册 | 3D 点 | observations | 平均 track | track=2 | track≥4 | 平均重投影误差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| COLMAP 3.11 | 18/18 | 9,148 | 39,728 | 4.3428 | 6.59% | 52.97% | 0.7907 px |
| Colite sequential | 18/18 | 9,057 | 39,202 | 4.3284 | 7.07% | 53.40% | 0.8655 px |
| Colite vocab tree | 18/18 | 9,120 | 39,432 | 4.3237 | 7.86% | 52.84% | 0.8682 px |

Sim(3) 后 sequential 的平均/最大旋转误差为 `0.0388° / 0.0872°`，平均/最大相机中心误差占最大基线 `0.0162% / 0.0528%`。结构指标已经接近，剩余主要差距是平均重投影误差约高 9.5%、鲁棒多视图估计器和大规模 adjacency 缓存效率。

## SfM：House 公平前端测试

2026-07-17 使用两端 CPU VLFeat、8192 基础 keypoint 上限、全部 630 pair、ratio 0.8、cross-check、guided matching、4 线程和固定内参：

| 指标 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 注册图像 | 36/36 | 36/36 |
| 稀疏点 | 16,296 | 11,097 |
| observations | 43,725 | 25,870 |
| 平均 track | 2.683 | 2.331 |
| 平均/最大点误差 | 1.3453 / 11.3983 px | 1.0478 / 3.9362 px |
| 实际特征均值/图 | 12,120 | 13,480 |
| 总墙钟 | 699.64 s | 677.0 s |

结论：早期“Colite 3,404 点、COLMAP 11,765 点”主要受前端配置不一致影响；统一前端后 Colite 点数反而更多，但误差尾部明显更差。后续应优化最终 observation/track 过滤和统一 guided matching，而不是继续追求点数。

## SfM：House 共享 Camera 局部 BA 修复

2026-07-22 使用 CUDA VLFeat、最终 8192 descriptors/图、全部 630 pair、BF、ratio 0.8、相邻放宽窗口 2、EXIF 焦距初始化、焦距 BA 和 8 线程复测。旧实现允许不完整局部窗口修改 36 图共享 Camera 的焦距，随后每轮对全模型过滤；其 `8 local + 8 fixed` 窗口覆盖上限之后，累计删除 10,429 点和 46,168 个观测。

修复内容包括：不完整局部窗口冻结共享内参、局部过滤仅作用于调整图像所观测的点，以及全局 BA 前执行一次全模型 retriangulate。冷启动结果：

| 指标 | 修复前 | 共享内参/过滤修复 | 加全局 retriangulate |
|---|---:|---:|---:|
| 注册图像 | 36/36 | 36/36 | 36/36 |
| 稀疏点 | 5,458 | 10,070 | 10,212 |
| observations | 25,830 | 44,182 | 44,922 |
| 平均重投影误差 | 0.8240 px | 0.9066 px | 0.9380 px |
| 墙钟 | 938.98 s | 923.45 s | 929.11 s |
| 峰值 RSS | 616.2 MiB | 638.3 MiB | 625.1 MiB |

局部过滤删除事件从 49,232 个观测/10,956 点降为 9,453 个观测/816 点，单轮最大删观测从 13,918 降为 1,065。全局 retriangulate 新建 370 点、902 个观测，merge/filter 后净增 142 点，属于有效但次要的补充。PCA 投影显示修复后屋顶、道路和场地边界更连续。

本组仍不能直接与 COLMAP 40,578 点解释为 Mapper 的纯差距：COLMAP 组实际提取 368,100 descriptors，Colite 为 294,912，并且两者使用不同 CUDA SIFT、guided matching 和相机模型。历史统一 CPU VLFeat 前端时 Colite 曾为 16,296 点、COLMAP 为 11,097 点，说明剩余差异需要前端受控 A/B，而不是继续放宽几何过滤。

## SfM：CUDA 线程与 true guided matching 修复

2026-07-22 继续使用上述完全相同的 House 参数。确认 `cuda_vlfeat` 已编译且运行时可用；旧版 CUDA 提取把持久线程池设为 1 后未恢复，使后续 BF 和弱对重匹配退化为单线程。恢复 8 线程、复用每图量化描述子、加入 AVX2 点积并并行弱对后，无 true-guided 的墙钟从 929.11 s 降至 172.66 s，点数保持 10,212，证明这些修改不改变匹配结果语义。

剩余点数差距来自 guided matching：COLMAP 的 202,850 原始匹配经几何引导后形成 352,372 内点；旧 Colite 只有 72,992 内点。旧实现只对相邻 69 对做“全局宽松匹配后极线过滤”，不能恢复被极线外假近邻压掉的真实匹配。新实现沿初始 F 的极线走廊重新搜索描述子，并对全部具有可用初始几何的图像对执行。

| 指标 | 旧 Colite | true-guided Colite | COLMAP 4.0 对照 |
|---|---:|---:|---:|
| descriptors | 294,912 | 294,912 | 368,100 |
| verified inliers | 72,992 | 325,730 | 352,372 |
| 稀疏点 | 10,212 | 48,794 | 40,578 |
| observations | 44,922 | 147,407 | 150,537 |
| 平均 track | 4.399 | 3.021 | 约 3.71 |
| 平均重投影误差 | 0.9380 px | 1.1112 px | 不同前端，不直接判优 |
| 墙钟 | 929.11 s | 280.90 s | 82.27 s |
| 峰值 RSS | 625.1 MiB | 761.8 MiB | — |

最终 48,794 点较旧版增加 377.8%，比 COLMAP 多 20.2%；observations 仅少 2.1%。这确认主要缺口不是 GPS，也不是 Mapper 再次误删。点数更多不等于精度更高，仍需同时看误差尾部、轮廓和 MVS。最终分阶段耗时为 CUDA SIFT 12.94 s、BF 89.10 s、几何验证 36.38 s、guided 22.33 s、Mapper 124.97 s、RGB 取色 11.47 s；当前只有 SIFT 在 CUDA，BF 和 Mapper 仍为 CPU 主路径。完整数据位于 `output/colite_sfm_true_guided_house_20260722/`。

## SfM：CUDA 描述子匹配与默认焦距

2026-07-23 新增精确 CUDA BF：描述子仍按 CPU BF 的 uint8/512 规则量化，每图仅上传一次，warp 使用 `__dp4a` 计算整数点积，ratio、max-distance 和 mutual check 共用 CPU 判定。随机描述子单测验证 CUDA 与 CPU BF 匹配集合完全一致。House 使用 CUDA VLFeat 8192、全部 630 pair、邻近窗口 2、8 CPU 线程；不传焦距且默认开启焦距 BA。

最早 2026-07-19 对比效果差的直接原因不是 GPS，而是三条链路都显式固定 `f=5500 px` 并关闭焦距/畸变 BA。House 的 COLMAP 默认 EXIF 初值为 3059.454 px，OpenMVG 通过 Sony RX100 的 13.2 mm 传感器库与 EXIF 物理焦距得到 3139.829 px。Colite 旧实现按 `f35/35×max(width,height)` 换算且没有把 EXIF 标为 prior；现改为与 COLMAP 4.0 相同的 `f35/43.27×image_diagonal`，增加焦平面分辨率回退，并默认优化焦距。

本次初值为 3059.454 px（`exif_35mm` prior），最终 PINHOLE 均值约 3024.512 px；COLMAP SIMPLE_RADIAL 对照为 3025.244 px。CUDA BF 的首轮匹配为 11.691 s，旧 CPU BF 为 89.096 s，快 7.62 倍；全流程墙钟从 280.90 s 降至 216.91 s。最终 36/36 图、48,745 点、147,250 observations、325,715 verified/guided inliers，RGB PLY 有效。完整数据位于 `output/colite_cuda_matcher_focal_default_house_20260723/`。guided 极线走廊重搜索仍是 CPU 路径。

## SfM：几何、BA、三角化缓存与 RGB 取色优化

2026-07-23 在完全相同的 House 命令上继续定位非 CUDA BF 阶段。发现批量两视图验证串行处理 630 pair；局部/全局 Ceres BA 把线程数硬编码为 1；更大的重复开销是每次 `TriangulateRegisteredImage` 都从 325,715 个 verified/guided inliers 重建完整 correspondence adjacency。修复后，Mapper 生命周期只构建一次只读 adjacency，所有局部和全局三角化复用；稀疏 RGB 的 36 张图像解码和采样并行，颜色仍按原图像顺序归并。

| 阶段/指标 | CUDA BF 基线 | 几何+BA 并行 | +三角化缓存 | +并行 RGB/guided 缓存 |
|---|---:|---:|---:|---:|
| 几何验证 | 33.281 s | 4.960 s | 4.937 s | 4.877 s |
| guided 重匹配 | 20.250 s | 20.186 s | 19.860 s | 20.354 s |
| Mapper | 117.635 s | 94.621 s | 35.491 s | 34.318 s |
| RGB 取色 | 9.522 s | 10.799 s | 10.380 s | 1.791 s |
| 进程墙钟 | 216.91 s | 170.92 s | 110.44 s | 96.90 s |
| 注册图像 | 36/36 | 36/36 | 36/36 | 36/36 |
| 稀疏点 | 48,745 | 48,741 | 48,731 | 48,749 |
| observations | 147,250 | 147,247 | 147,244 | 147,305 |

最终墙钟比基线降低 55.3%，相对同机 COLMAP 全流程约 82.27 s 的差距由 134.64 s 缩小到 14.63 s。点数相对基线多 4（0.008%）；多线程 Ceres 的调度会造成极小浮点非确定性，三轮点数波动不代表算法阈值变化。guided 每图量化描述子和空间网格缓存后没有可测收益，说明其约 20 s 主要来自极线走廊候选点积和 E/F/H 几何复验；当前仍为 CPU guided 路径。并行 RGB 把最大 RSS 从约 0.78 GiB 提高到 1.01 GiB。三轮原始数据依次位于 `output/colite_sfm_runtime_optimized_house_20260723/`、`output/colite_sfm_runtime_cache_house_20260723/` 和 `output/colite_sfm_runtime_io_guided_house_20260723/`。

## MVS：COLMAP 4.0 单模块对齐

`building-18` 共用已注册位姿；去畸变最长边 800，PatchMatch 最长边 256，8 sources、3 iterations。以下是优化过程中最有指导意义的结果。

### 图像去畸变

输出尺寸 800×579，`fx/fy/cx/cy` 与 COLMAP 完全相同，共同像素区域 100%，整体 PSNR 36.756 dB。热缓存单次时间 Colite 5.09 s、COLMAP 3.29 s；冷缓存曾出现 Colite 5.43 s、COLMAP 22.98 s，因此不能用单次 I/O 主导结果下稳定速度结论。

### PatchMatch

三轮光度 + 三轮几何、bilateral NCC、directional HMM、CDF15 的一层结果：

| 指标 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟 | 41.98 s | 34.32 s |
| 有效覆盖率 | 55.593% | 68.062% |
| 稀疏点误差 median / P95 | 2.047% / 6.037% | 2.016% / 7.029% |

保留深度精度已达到参考水平，但一层覆盖率不足。启用两层 coarse-to-fine 后覆盖率为 62.485%，达到 COLMAP 的 91.81%，median/P95 为 2.047%/6.071%，墙钟增至 52.34 s。默认一层用于平衡耗时，质量 A/B 使用两层。

### Fusion

早期单线程 seed-track 为 58,206 点、8.76 s，COLMAP 为 65,647 点、0.90 s。改用 8 线程行带并行后墙钟降至 2.44 s，达到 `<3 s` 目标；重复输出点数差约 0.11%，来自并发 seed 认领顺序。

### Poisson

相同 58,206 点输入，`depth=8, point_weight=1, color=1, trim=0`：

| 指标 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 时间 | 7.85 s | 37.51 s |
| 顶点/面 | 17,769 / 35,386 | 完全相同 |
| 双向顶点距离 median/P95 | 4.08e-6 / 6.11e-6 | reference |

输出可视为数值等价；少量退化面来自 ASCII PLY 舍入。

### Delaunay、Simplifier、Texturer

| 模块 | 优化后 Colite | COLMAP 4.0 | 结论 |
|---|---:|---:|---|
| Delaunay 顶点/面 | 10,270 / 20,498 | 10,270 / 20,496 | 面数仅差 2，面积相对差约 0.0027% |
| Delaunay 非流形边 | 2 | 2 | graph-cut 和长面规则已高度对齐 |
| Simplifier 最大外推距离 | 0.025608 | 0.026227 | QEM 限制到边段后对齐 |
| Texturer 有效面（1000 面输入） | 429 | 511 | Colite 无假阳性，但 BVH 判定更保守 |
| Texturer 面集合 Jaccard | 83.95% | reference | 主要剩余差距为遮挡 recall |

## House 完整 Colite/COLMAP 4.0 MVS

2026-07-17 的完整 A/B 使用最长边 800、CUDA PatchMatch 5 iterations、Poisson `depth=13/trim=0`。这次 Colite 上游 SfM 只有 3,404 稀疏点，配置不如后续公平 SfM，因此只保留作为整链路规模参考。

| 阶段 | Colite CUDA | COLMAP 4.0 CUDA |
|---|---:|---:|
| SfM | 82.61 s | 352.19 s |
| undistorter | 20.64 s | 24.01 s |
| PatchMatch | 423.44 s | 373.21 s |
| Fusion | 22.81 s | 14.77 s |
| Poisson | 31.62 s | 134.23 s |
| Simplifier | 5.79 s | 4.03 s |
| Texturer | 12.16 s | 79.76 s |

关键质量规模：Colite/COLMAP 视差覆盖 31.47%/41.46%，融合点 99,537/106,895，Poisson 面 932,252/953,288，纹理有效面覆盖 61.81%/93.42%。该测试再次说明 PatchMatch 覆盖和纹理面召回是主要视觉差距。

## House 共享 COLMAP SfM 的 MVS 复测（2026-07-24）

使用同一 COLMAP 4.0 SfM 模型、36 张 House 图像、最长边 800、8 sources、
CUDA PatchMatch、3 轮光度 + 3 轮几何、8 CPU 线程。去畸变 PINHOLE 参数
完全一致，36 图 PSNR 35.639 dB，因此上游图像和相机不是主要差距。

| 指标 | Colite 原质量组合 | Colite 对齐变体 | COLMAP 4.0 |
|---|---:|---:|---:|
| PatchMatch 有效覆盖 | 19.605% | 48.152% | 85.957% |
| 稀疏深度误差 P95 | 24.045% | 2.681% | 2.703% |
| PatchMatch 墙钟 | 78.00 s | 151.70 s | 57.69 s |
| Fusion 点数 | 57,945 | 192,351 | 309,673 |
| Fusion 包围盒对角线 | 149.035 | 102.293 | 17.022 |

对齐变体关闭 directional HMM、使用单层和 4 轮 checkerboard，深度精度尾部
恢复到 COLMAP 水平，但覆盖仍只有其 56.0%，耗时为 2.63 倍。原组合中 source
probability 大量降到下限，说明“两层 + directional HMM + 单轮
checkerboard”在 House 上不稳健。Colite Fusion 主体与 COLMAP 的双向最近邻
中位距离约为参考包围盒的 0.13%，但少量远端离群点把包围盒扩大 6 倍并破坏
各自输入下的 Poisson 网格。

同一 COLMAP `fused.ply` 输入两端 Poisson 后，均为 71,021 顶点、141,765 面，
全顶点双向距离 median `4.22e-6`、P95 `6.55e-6`，确认 Poisson 实现不是
差距来源。当前优化顺序应为 CUDA PatchMatch 收敛/覆盖、kernel 调度和 Fusion
离群保护；本轮因存在明显差异，没有继续 OpenMVS。完整命令和原始结果位于
`output/mvs_colite_colmap_house_20260724/`。

### CUDA checkerboard 邻居预选（2026-07-26）

沿用上述共享模型和统一参数。诊断表明把 4 轮直接降为 1 轮会将墙钟降到
59.50 s，但覆盖从 48.152% 崩到 2.958%，因此瓶颈不能通过减少传播轮数解决。
优化后的 checkerboard 先按邻居已有代价排序，只对最优两个邻居执行完整
光度代价：

| 指标 | 四邻居基线 | 双邻居优化 | COLMAP 4.0 |
|---|---:|---:|---:|
| PatchMatch 墙钟 | 151.70 s | 88.80 s | 57.69 s |
| CUDA 传播累计 | 48.254 s | 27.326 s | 未分项 |
| 有效覆盖 | 48.152% | 44.819% | 85.957% |
| 稀疏深度 median / P95 | 0.411% / 2.681% | 0.462% / 2.730% | 0.221% / 2.689% |

墙钟降低 41.5%，传播累计时间降低 43.4%；单最优邻居虽然只需 62.56 s，
但覆盖降至 34.124%，不予保留。双邻居版融合为 166,639 点，主体到 COLMAP
最近邻 P95 仅为参考包围盒的 0.563%，但包围盒对角线仍为 100.803。历史
Fusion 点的 track 检查显示全部至少有 4 个观测，说明远端点是多视图一致的
错误表面，不能靠最小观测数过滤。新增的显式空间尾部过滤使用
`--dense_cloud_spatial_bounds_quantile 0.001`，仅移除 697/166,631 点
（0.418%），包围盒对角线由约 100.8 降到 15.433，接近 COLMAP 的 17.022；
候选到参考 P95 最近邻由 0.09577 降到 0.09132。该参数默认 0，不会静默裁剪
合法远景。完整命令、日志和指标位于
`output/mvs_patchmatch_optimization_house_20260726/`。

### PatchMatch 几何一致覆盖优化（2026-07-26）

继续使用同一 House 模型、36 图、最长边 800、8 sources、CUDA、8 CPU
线程、3 次光度 + 3 次几何、公开参数 `pyramid_levels=1`、4 轮
checkerboard、1 px/至少 2 source 的严格过滤口径。最终实现包含：

- 几何候选使用软重投影代价，不再要求优化前已经满足 1% 深度一致；
- 几何项与 `0.5*(1-ZNCC)` 同比例缩放，无效源深度记截断最大惩罚；
- 动态源选择默认启用 directional HMM；
- 移除无收益的 ±7.5° 补丁择优和固定 source/incident cost 惩罚；
- 几何阶段内部至少 3 层 coarse-to-fine，并执行 4 个同步 fixed-point pass。

| 指标 | 旧双邻居版 | 最终 Colite | COLMAP 4.0 |
|---|---:|---:|---:|
| 严格几何覆盖 | 44.819% | **85.892%** | 85.957% |
| 至少 1 个一致源 | 未记录 | 92.022% | 未记录 |
| 稀疏深度 median / P95 | 0.462% / 2.730% | 0.329% / 3.017% | 0.221% / 2.689% |
| PatchMatch 墙钟 | 88.80 s | 102.47 s | 57.69 s |
| 峰值 RSS | 未记录 | 3.12 GiB | 未记录 |

目标 85% 已达到，且相对 COLMAP 的覆盖差仅 0.065 个百分点。共同有效像素
上的 Colite/COLMAP 深度相对差 median/P95 为 0.253%/1.278%，说明提升不是
通过放宽 1 px 阈值换取。单一旋转补丁分支还把 CUDA 核心工作量从约 58 s
降至约 26 s，为多层和同步 pass 释放了预算。完整命令、失败 A/B、计时和
指标位于 `output/mvs_geometric_coverage_house_20260726/`。

### House 完整 MVS 与 OpenMVS 复测（2026-07-26）

继续使用同一 COLMAP 增量 SfM、36 图、800 px、CUDA、8 sources、3 次光度
+ 3 次几何和 8 CPU 线程。Colite/COLMAP 最终深度覆盖为
85.8916%/85.9568%，共同深度差 median/P95 为 0.2533%/1.2784%。
Colite 原始 Fusion 为 380,678 个 RGB+normal 点，但极少远端点把包围盒
对角线扩大到 73.014，Poisson 仅得到 6,002 顶点/11,988 面。

House 显式启用 `--dense_cloud_spatial_bounds_quantile 0.001` 后，仅删除
1,728/380,312 点（0.454%），包围盒恢复为 15.217（COLMAP 17.022），
Poisson 恢复为 102,001 顶点/203,615 面；COLMAP 为 71,023/141,769。
清理后 Colite/COLMAP 点云双向最近邻中位距离为参考尺度的
0.120%/0.112%，P95 为 0.546%/0.295%；Poisson 网格双向中位距离为
0.228%/0.221%，P95 为 1.279%/0.849%。

| 阶段 | Colite 推荐配置 | COLMAP 4.0 |
|---|---:|---:|
| 去畸变 | 25.50 s | 16.19 s |
| CUDA PatchMatch | 106.70 s | 56.89 s |
| Fusion（Colite 含尾部过滤） | 27.09 s | 4.17 s |
| Poisson | 7.85 s | 78.88 s |
| 全流程 | 167.14 s | 156.13 s |

全流程只慢 7.1% 是 Colite Poisson 更快所致；只比较稠密点生成，Colite
159.29 s、COLMAP 77.25 s，仍慢 2.06 倍，且 PatchMatch/Fusion 峰值内存
约 3 GiB。后续性能优先级仍是 Fusion 数据布局/并行和 PatchMatch 内存复用。

质量达到可比后继续使用同一 COLMAP SfM 输入测试 OpenMVS 2.4 CUDA。
`InterfaceCOLMAP` 0.36 s、`DensifyPointCloud`（含深度+Fusion）19.20 s、
`ReconstructMesh` 9.44 s；加共享 COLMAP 去畸变后为 45.19 s。输出
464,515 个 RGB+normal 点，较 COLMAP 点云的双向中位距离为参考尺度的
0.110%/0.105%，P95 为 0.411%/0.326%。OpenMVS Delaunay/graph-cut 网格
为 212,894 顶点/425,657 面；它不是 Poisson，不能只按面数横向判优。
完整命令、失败转换审计、资源日志和指标位于
`output/mvs_colite_colmap_openmvs_house_20260726/`。

### House Stereo Fusion 性能优化（2026-07-26）

复用上一轮相同的 36 张 House、800 px、8 线程、binary depth/normal 和
consistency graph，独立 `stereo_fusion` 顺序重复 3 次。最终保留的组合为：
Fusion RGB-only 图像加载、过滤后跳过 text cost map、consistency graph CSR
与批量读取、depth/normal 直接批量 I/O、worker scratch 复用、紧凑每像素缓存，
以及默认关闭重误差采样（可用 `--fusion_diagnostics` 恢复）。

| 指标（3 次中位数） | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 墙钟 | 26.06 s | 11.36 s | -56.4%，2.29× |
| User CPU | 89.67 s | 72.91 s | -18.7% |
| 峰值 RSS | 2,996,780 KB | 1,811,952 KB | -39.5% |
| 融合点 | 380,524 | 380,156 | -0.097% |
| 融合观测 | 6,799,952 | 6,799,086 | -0.013% |

最终内部中位数：RGB 0.430 s、depth/normal 0.861 s、一致性图 1.179 s、
核心融合 4.269 s、ASCII PLY/附属输出 2.041 s。上一轮 COLMAP 完整 Fusion
为 4.17 s；Colite 核心已接近，独立命令仍受模型/摘要读取和额外输出影响。
完整命令、逐版本 A/B、`/usr/bin/time -v`、RGB 点云和诊断位于
`output/mvs_fusion_optimization_house_20260726/`。

### House 原图半分辨率 MVS 复测（2026-07-26）

在上述优化代码上使用共享 COLMAP SfM、36 张 4592×3056 House 图像，将
去畸变最长边设为原图一半 2296 px；两端实际输出均为 2296×1527。PatchMatch
不再二次缩放，均使用 CUDA、8 sources、3 次光度 + 3 次几何、8 CPU 线程和
相同 Fusion/Poisson 参数。

| 阶段/指标 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 去畸变 | 23.32 s | 19.85 s |
| CUDA PatchMatch | 744.25 s | 251.53 s |
| PatchMatch 峰值 RSS | 13,217,612 KB | 3,207,788 KB |
| 有效深度覆盖 | 62.9533% | 61.9605% |
| 共同深度差 median/P95 | 0.1207% / 0.6366% | reference |
| Fusion | 51.82 s | 27.91 s |
| RGB+normal 点 | 2,174,486 | 1,615,872 |
| Poisson | 19.73 s | 91.61 s |
| 原始 MVS 总计 | 839.12 s | 390.90 s |

质量已达到可比：Colite 覆盖略高，稀疏深度误差 median/P95 为
0.2011%/2.5656%，COLMAP 为 0.1377%/2.3671%。主要差距已经转为扩展性：
Colite PatchMatch 为 2.96×，RSS 为 4.12×；Fusion 为 1.86×，RSS 为
3.49×。运行轨迹表明 Colite 把所有大尺寸 view 结果和几何图保留到阶段末尾，
而 COLMAP 按图像写出并由 cache 控制工作集。后续首要优化是 PatchMatch
按视图流式写出/逐出、consistency graph 避免双格式大缓冲，以及 Fusion
有界 view cache。

原始 Colite 点云仅有约 0.43% 空间尾部点，却把包围盒对角线扩至 164.118，
导致 `trim=0` Poisson 退化为 7,607 顶点/14,834 面。House 显式启用
`--dense_cloud_spatial_bounds_quantile 0.001` 后，点云为 2,165,330 点、
包围盒 18.559，Poisson 恢复为 70,528/140,739；COLMAP 为
81,253/162,169。清理后点云双向 median 为参考尺度的 0.0791%/0.0681%，
网格双向 median 为 0.2560%/0.2374%。该保护仍保持显式，不能对含合法远景
的场景默认启用。

完整命令、资源采样、JSON 和原始/清理后产物分别位于
`output/mvs_colite_colmap_house_half_20260726/` 与
`output/mvs_colite_colmap_house_half_spatial_filter_20260726/`。

## House 半分辨率 MVS 流式内存与远端点修复（2026-07-26）

同一 2296×1527 去畸变输入、8 sources、CUDA PatchMatch、8 CPU 线程。根因分析将旧点云的 9,375 个空间尾部点追溯到 seed depth：异常点 Colite 深度 median/P99 为 8.3889/81.6054，而 COLMAP 同像素有效深度 median/P99 为 4.3627/7.6326。生产路径此前把稀疏深度原始 extrema 直接作为 CUDA 搜索范围；最严重视图 raw max 106.1906，而 COLMAP 式 1%–99% + 25% stretch 得到 10.2918。修复后搜索范围已与 COLMAP 对齐。

| 指标 | 优化前 | 本轮 |
|---|---:|---:|
| PatchMatch wall | 744.25 s | 762.12 s |
| PatchMatch 峰值 RSS | 13,217,612 KB | 9,161,864 KB |
| Fusion wall | 51.82 s | 121.44 s |
| Fusion 峰值 RSS | 11,368,760 KB | 5,878,120 KB |
| Colite 点云 bbox 对角线 | 164.118 | 21.6517 |
| `fused.ply` | 286,004,228 B / 2,174,486 点 | 80,455,672 B / 2,979,831 点 |
| Poisson 顶点/面 | 7,607 / 14,834 | 65,812 / 131,310 |

PatchMatch native consistency graph 逐视图写出并在第二 pass 流式读取应用过滤，不再保留全部 graph，也不再默认写 COLMAP graph 副本；PatchMatch 退出后才启动 Fusion。Fusion 使用全局有界 LRU、活跃项 weak 去重索引和 worker 热集，并取消全像素 double XYZ 预计算。PatchMatch RSS 降 30.7%，Fusion RSS 降 48.3%，binary float PLY 在点数增加约 37% 时仍缩小 71.9%。

代价是 Fusion wall 仍为 2.34×：主要来自冷加载和按需反投影重复计算。下一步应使用 mmap/异步预取及有界的 float 几何 tile cache，不应恢复无界全像素 double 缓冲。完整命令、失败缓存迭代和原始结果位于：

- `output/mvs_outlier_rootcause_house_20260726/`
- `output/mvs_colite_house_half_streaming_20260726/`
- `output/mvs_colite_house_half_fusion_cache_weak_20260726/`
- `output/mvs_colite_house_half_fusion_cache_final_20260726/`

## House MVS 算法耗时优化（2026-07-26）

继续使用相同 COLMAP SfM 输入、36 图、8 sources、CUDA、8 CPU 线程、3 次光度
迭代、每轮 3 次几何迭代和严格至少 2 source 几何过滤。优化没有减少 CLI
指定的内层迭代，而是：

- 几何 refinement 继承前一轮 reference depth/normal，从 4 个重置式外层 pass
  先改为 2 个有初值 pass，扩展性复测后再改为 1 个有初值 pass；每个 pass
  内部仍保留 3 层金字塔和 CLI 指定的 3 次迭代；
- consistency graph 直接构建 CSR32，native v2 连续批量写入，构建及过滤按
  8 线程有界并行；
- binary 模式将 cost map 也改为 float binary；
- Fusion 预计算 PINHOLE 几何、对 source visited 做早拒绝、仅在诊断模式写
  大体积逐 track TSV。

| 2296×1527 PatchMatch 指标 | 旧版 | 继承初值、2 pass + CSR | 继承初值、1 pass | COLMAP 4.0 |
|---|---:|---:|---:|---:|
| wall | 744.25 s | 355.36 s | **248.98 s** | 251.53 s |
| 相对 COLMAP | 2.96× | 1.41× | **0.99×** | 1.00× |
| 峰值 RSS | 13,217,612 KB | 8,743,736 KB | 8,713,564 KB | 3,207,788 KB |
| 严格最终有效像素 | 旧口径不可直比 | 97,655,709 | 83,101,832 | 78,201,944 |
| 严格最终覆盖 | 旧口径不可直比 | 77.372% | **65.841%** | 61.961% |

单 pass 版阶段墙钟为初始 PatchMatch 84.21 s、refinement 113.42 s、graph
构建/写出 17.64 s、graph 读取/过滤 2.47 s、dense maps 输出 13.69 s。相对
双 pass 版再加速 29.9%，消除了大图相对 COLMAP 的耗时差距。严格覆盖降低
11.53 个百分点，但仍比相同输入、阈值和工作尺寸的 COLMAP 历史严格结果高
3.88 个百分点；共享稀疏点深度误差 median/P95 为 0.1692%/2.2393%，未出现
精度尾部恶化。质量脚本直接读取的 COLMAP 未过滤深度图有效率接近 100%，不能
与 Colite 的最终严格过滤覆盖混用。

800 px 同步从双 pass 的 47.26 s 降至 35.45 s，低于 COLMAP 的 57.69 s；
严格覆盖 85.905%，与 COLMAP 历史严格覆盖 85.957% 只差 0.051 个百分点。
因此单个继承 pass 是当前速度/覆盖最有价值的默认组合；继续保留第二个外层
pass 只适用于明确愿意用约 1.43× 时间换取更高覆盖的实验配置。

Fusion 的最终候选版为 107.05 s、峰值 RSS 5,901,492 KB、3,044,356 个 RGB+normal
点；相对旧的有界缓存版 121.44 s 加速 11.9%，但仍慢于 COLMAP 27.91 s。点云到
COLMAP 的最近邻距离（各采样 50,000 点）median/P95 为参考 bbox 对角线的
0.1096%/0.9792%。完整命令、时间文件、阶段 summary、质量 JSON 和拒绝的 A/B
位于 `output/mvs_runtime_optimization_house_half_20260726/`。单 pass 的完整
命令、时间、阶段摘要和质量指标位于
`output/mvs_large_scaling_house_20260726/`。

### House 2296 px 峰值内存与纹理全流程复测（2026-07-26）

继续使用同一共享 COLMAP SfM、36 图、2296×1527、CUDA、8 sources、8 CPU
线程、3 次光度 + 3 次几何、单个继承式外层几何 pass。内存优化把旧 normal
随 refinement job 移动并在完成时释放，CUDA 输入直接写 pinned staging，关闭
诊断时在设备端初始化动态 source probability 且不回读整图。PatchMatch
阈值、迭代和随机种子未变。

| PatchMatch 指标 | 优化前 Colite | 优化后 Colite | COLMAP 4.0 |
|---|---:|---:|---:|
| wall | 248.98 s | **246.39 s** | 248.13 s |
| 峰值 RSS | 8,713,564 KB | **6,565,756 KB** | 3,218,888 KB |
| GPU 采样峰值 | 未记录 | 3,007 MiB | 2,307 MiB |
| 严格覆盖 | 65.8411% | 65.8411% | 61.9605% |

RSS 相对优化前降低 24.6%，墙钟未回退。共同有效深度相对差
median/P95 为 0.0870%/0.4534%；共享稀疏点误差 Colite 为
0.1658%/2.2227%，COLMAP 为 0.1377%/2.3671%。

两端继续完成 Fusion、Poisson depth 8/trim 0、0.5 简化和纹理：

| 阶段 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 去畸变 | 28.26 s | 21.22 s |
| CUDA PatchMatch | 246.39 s | 248.13 s |
| Fusion | 115.18 s | 28.54 s |
| Poisson | 7.28 s | 89.85 s |
| 0.5 Mesh 简化 | 1.34 s | 0.69 s |
| 纹理 | 6.93 s | 9.58 s |
| 合计 | 405.38 s | 398.01 s |

Colite 清理后 RGB+normal 点云 2,357,955 点，COLMAP 为 1,615,849 点；
双向最近邻 median 分别为参考包围盒对角线的 0.1004%/0.0889%。简化网格
Colite/COLMAP 为 113,302/81,084 面。纹理有效面覆盖为 86.10%/92.61%，说明
Colite 纹理速度已快，但有效面召回仍低 6.51 个百分点。

曾尝试把 Fusion 每 worker 热集从 16 收紧到 2，以及容量 8 并按 reference
清空；两者均因 dense map/graph 冷加载抖动运行超过 15 分钟，已终止并撤销，
不能用数量级耗时换内存。完整命令、逐阶段 time/RSS/GPU 采样、RGB 点云、
Poisson/简化 Mesh、纹理和 JSON 指标位于
`output/mvs_memory_texture_compare_house_20260726/`。

### House 2296 px Fusion 热路径优化（2026-07-26）

复用上一节完全相同的 36 图、2296×1527 binary depth/normal/native
consistency graph，8 CPU 线程、cache 16、至少 5 observations、2 px、
relative depth 0.01、normal 10° 和显式空间尾部 0.001。每次运行使用独立
输出目录，PatchMatch maps 只读复用；最终版本连续运行 3 次并取中位数。

保留的优化为：COLMAP 式 `nth_element` 中位数、O(1) worker 弱索引 +
16 项强热集、稠密 ImageId 索引、最近 source 像素快路径、投影深度单次采样、
法线变换复用、内存 CSR32 offsets，以及 ≤64 views 的 uint64 可见性 mask。
默认 180° view gate 且关闭诊断时不采集视角统计，不改变任何几何接受条件。

| 指标 | 旧 Colite | 优化后 Colite（三轮中位数） | COLMAP 4.0 |
|---|---:|---:|---:|
| Fusion wall | 115.18 s | **48.94 s** | 28.54 s |
| Fusion core | 111.47 s | **40.55 s** | 未分项 |
| 峰值 RSS | 5,445,520 KB | **4,295,916 KB** | 3,266,828 KB |
| 融合点 | 2,368,811 | 2,367,658（中位点数） | 1,615,849 |
| 融合观测 | 40,285,004 | 40,294,338（中位观测数） | 未记录 |
| wall / 百万输出点 | 48.62 s | **20.67 s** | 17.66 s |

wall 降低 57.5%，RSS 降低 21.1%。三轮 wall 为 46.35/48.94/55.06 s，
点数最大差 864（0.0365%），属于已有并行 CAS seed 归属的微小非确定性。
优化后绝对 wall 仍为 COLMAP 的 1.71×，但 Colite 点数高 46.5%；按输出点
归一化后差距缩至约 17%。最终点云到本轮 COLMAP 的双向最近邻 median 分别为
COLMAP bbox 对角线的 0.1005%/0.0886%，P95 为 0.7511%/0.2169%，与优化前
0.1004%/0.0889%、0.7437%/0.2173% 同量级。

完整三轮 `/usr/bin/time -v`、阶段计时、点云、可见性、失败 A/B 和质量 JSON
位于 `output/mvs_fusion_runtime_optimized_house_20260726/`。

### House 原图尺寸完整 MVS 与纹理复测（2026-07-26）

使用相同 COLMAP SfM、36 图、去畸变实际 4592×3053、8 sources、CUDA、
8 CPU 线程、3 次光度/几何迭代、至少 2 个一致 source、Fusion 至少 5 个
observations、Poisson depth 8/trim 0、0.5 简化和纹理化。Colite 必须在
去畸变命令显式设置 `resolution_level=0`；只提高 `max_image_size` 仍会继承
控制器默认的缩小一级。

| 阶段 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 去畸变 | 28.04 s | 24.93 s |
| CUDA PatchMatch | 965.67 s | 780.09 s |
| Fusion | 147.56 s | 109.34 s |
| Poisson | 70.31 s | 90.87 s |
| 简化 | 1.43 s | 0.78 s |
| 纹理 | 17.89 s | 32.67 s |
| 合计 | 1230.90 s | 1038.68 s |

Colite 总流程慢 18.5%，稠密生成慢 24.8%；PatchMatch RSS 为
23,637,044 KB，对比 COLMAP 11,277,648 KB，仍是最主要扩展性问题。GPU
采样峰值为 7494/7136 MiB，双方均未回退 CPU。

Colite/COLMAP 有效深度覆盖为 35.6263%/27.0115%，共同区域深度相对差
median/P95 为 0.0600%/0.2895%；稀疏深度 P95 为 1.6815%/1.6353%。
Colite 空间尾部清理后为 4,833,106 个 RGB+normal 点，COLMAP 为
1,928,305 点。简化网格为 116,852/70,841 面，有纹理面比例为
84.98%/90.65%，Texturer 召回仍低 5.67 个百分点。

完整命令、逐阶段时间/RSS/GPU 采样、点云、网格、纹理和质量 JSON 位于
`output/mvs_full_resolution_compare_house_20260726/`。

## House GUI High 等价 MVS 与 Poisson 2×2（2026-07-26）

以 Windows GUI 产生的同一稀疏模型为输入，在 2400×1597、20 sources、
radius 5、5 photometric iterations、几何一致和统一 Fusion 阈值下重新冷启动：

| 阶段 | Colite | COLMAP 4.0 | 比率 |
|---|---:|---:|---:|
| 去畸变 | 25.35 s | 20.25 s | 1.25× |
| CUDA PatchMatch | 3537.21 s | 1504.87 s | 2.35× |
| Fusion | 1145.55 s | 30.79 s | 37.21× |
| MVS 合计 | 4708.11 s | 1555.91 s | 3.03× |

几何深度覆盖为 82.5200%/82.6285%，共同区域相对深度差 median/P95 仅
0.0886%/0.5821%；融合点数为 2,514,226/2,607,812。Colite PatchMatch
GPU 利用率均值 96.94%，确认使用 CUDA；性能差不是 CPU fallback。Fusion
的 1145.55 s 中约 1139.41 s 位于核心遍历/验证，峰值 RSS 12.61 GiB，
是当前 MVS 首要热点。Colite 点云包围盒对角线比 COLMAP 大 10.51%，
Fusion 空间尾部仍需继续收紧。

统一使用 COLMAP 4.0 Poisson（point weight 1、trim 10、8 线程）做
输入点云 × depth 的交叉实验：

| 点云 | depth 8（顶点/面/时间） | depth 13（顶点/面/时间） |
|---|---:|---:|
| Colite | 33,058 / 63,703 / 91.15 s | 6,739,391 / 13,340,877 / 1699.97 s |
| COLMAP | 41,098 / 79,582 / 97.21 s | 7,029,238 / 13,916,448 / 1572.96 s |

depth 13 相对 depth 8 增加约 171–204× 顶点和 175–209× 面，耗时增加
16–19×；固定 depth 13 时两套点云只造成约 4.3% 拓扑规模差。故 GUI High
网格细节的主效应来自 Poisson depth 13，而不是两端 PatchMatch 深度差；
`depth=8 + trim=10` 会过度裁剪，只适合作为预览。完整命令、日志、四个
网格及质量 JSON 位于 `output/mvs_gui_high_2x2_house_20260726/`。

### GUI High PatchMatch/Fusion 热路径优化（2026-07-27）

继续复用上节 36 图、2400×1597、20 sources、radius 5 和统一阈值，不复用
PatchMatch 结果重新冷启动。CUDA ZNCC 改为 FP32，并将每个候选/source 的逐
sample 完整投影改为一次平面单应组合和窗口增量投影。Fusion 将默认 source
搜索半径对齐为 0，跳过已完成 reference，取消默认 mutual graph 复验，并把
dense view 读取移出全局 LRU 锁。

| 阶段 | 优化前 Colite | 优化后 Colite | COLMAP 4.0 | 优化后/前 |
|---|---:|---:|---:|---:|
| CUDA PatchMatch | 3537.21 s | **772.17 s** | 1504.87 s | 0.218× |
| Fusion（36-view cache） | 1145.55 s | **36.47 s** | 30.79 s | 0.032× |
| 两阶段合计 | 4682.76 s | **808.64 s** | 1535.66 s | 0.173× |

PatchMatch 加速 4.58×，现在比 COLMAP 快 48.7%；GPU 显存峰值 5110 MiB，
利用率均值 86.9%。有效深度覆盖从 82.5200% 变为 82.4392%（-0.0808 个
百分点）；与 COLMAP 共同有效区域的相对深度差 median/P95 为
0.0888%/0.5821%，P95 与修改前基本相同。

最终 Fusion 为 2,529,490 点、54,964,653 observations，峰值 RSS
7,842,528 KiB。点云包围盒对角线从 21.7394 收紧到 20.2202，接近 COLMAP
19.6715；到 COLMAP 点云的双向最近邻 P95 为 0.0486/0.0413。对 36 图数据，
`image_cache_size=32` 仍会因 4 个视图反复淘汰而耗时 126.33 s，
`image_cache_size=36` 则为 36.47–40.39 s。一个按 reference 反复刷新未来
source 的 LRU 实验进一步退化到 160.36 s，已撤回，不属于最终实现。

完整命令、时间/RSS/GPU 日志、优化后 dense maps、RGB 点云和质量 JSON 位于
`output/mvs_runtime_optimized_gui_high_house_20260727/`。

### GUI High 平面网格起伏修复（2026-07-27）

对 House 36 图使用同一去畸变模型、2400×1597、20 sources、radius 5、5 次光度
迭代、5 次几何细化、CUDA 和 8 CPU 线程复测。Windows COLMAP 3.11 与 WSL
COLMAP 4.0 的定向点云局部指标接近，排除了 Windows/WSL、CUDA 版本和 Poisson
版本差异。根因位于 Colite 上游：

1. PatchMatch 旧默认把全部法线固定为 `(0,0,-1)`，且初始扰动仅 5°；COLMAP
   使用相机朝向半球随机法线和约 1 rad 起始扰动。弱纹理平面因此容易陷入有偏
   法线局部最优。
2. Fusion 旧实现每一跳相对 current point 校验重投影/深度，允许小误差沿图
   累积；COLMAP 始终相对 seed point 校验。旧行为会产生异常长 track，并合并
   邻近但不共面的观测。

修复后全尺寸 PatchMatch 为 810.83 s，平均 GPU 利用率 86.8%，显存峰值
5304 MiB，RSS 9,236,344 KiB；Fusion 核心命令为 44.08 s，RSS 8,566,176 KiB，
输出 2,057,161 个 binary float RGB+normal 点，平均 track 25.67 observations。

| 定向点云指标（50k 样本、24-NN） | 旧 Colite | WSL COLMAP | 修复后 Colite |
|---|---:|---:|---:|
| 局部平面归一化残差 median | 0.3412 | 0.2046 | **0.1631** |
| 存储法线与局部 PCA 法线夹角 median | 28.07° | 10.02° | **7.35°** |
| 邻域法线夹角 median | 17.16° | 10.28° | **6.59°** |

使用 point weight 1、trim 10、8 线程的 Poisson depth 13 生成
5,808,693 顶点/11,487,116 面（201.39 s），再以 ratio 0.01 简化为
63,328 顶点/112,107 面（147.08 s），纹理映射 6.39 s，输出 8192×3622 atlas。
对相同比例简化网格的 24-NN 局部平面残差 median，旧 Colite 为 0.2286，
COLMAP 为 0.2461，修复后 Colite 为 **0.1641**；该全场景指标包含真实屋檐锐边，
主要用于确认新旧同方向变化，不能代替指定屋顶 ROI 的真值误差。

完整命令、A/B、中间 dense maps、点云、网格、纹理、时间/RSS/GPU 日志和 JSON
位于 `output/mvs_mesh_planarity_house_20260727/`。

### 历史 GUI High 纹理三角形孔洞方案（2026-07-28，已废弃）

后续同网格 A/B 证明该方案通过过度接受可疑投影和逐三角 fallback 换取接近
100% 的 UV 覆盖，正是平面三角纹理碎片的主要来源。以下数据仅保留为历史对照，
当前默认行为以“House 同网格视图碎片修复（2026-07-29）”一节为准。

复用上一节修复后的 63,328 顶点/112,107 面 trim 10 简化网格，不重跑
PatchMatch/Fusion。原 OBJ 没有丢面或退化几何，但 21,388 个面被严格 BVH
遮挡选择拒绝，三个 UV 被压到统一占位 texel；其中 2,495 个未纹理连通块，
1,780 个恰好是单三角形，因此视觉上表现为三角形和异形孔洞。

修复分两层：默认 1% 目标端相对容差忽略近共面重建噪声；仍无可靠相机视图的
面使用 Poisson/QEM 保留的 RGB 顶点色烘焙独立 fallback patch。真实遮挡面、
背面和轮廓不通过邻面单边扩张，OBJ 拓扑不变。

| 指标 | 修复前 | 仅 1% 近面容差 | + RGB 顶点色 fallback |
|---|---:|---:|---:|
| 有效 UV 面 | 90,719 | 103,524 | **111,989** |
| 有效 UV 面比例 | 80.92% | 92.34% | **99.895%** |
| 未纹理几何面积比例 | 9.658% | 4.630% | **5.60e-9** |
| 可见退化面 | 21,388 | 8,583 | **118 个数值级微面** |

2% 容差只把真实相机纹理覆盖再提高 0.20 个百分点，收益不足以抵消遮挡穿透
风险，因此默认保持 1%。Poisson trim 5/0 仅把简化网格边界边从 22,785 降到
20,304/20,240，却把表面积从 56.48 扩大到 181.11/284.59、包围盒对角线从
11.84 扩大到 18.87/21.46，会产生大面积无图像支持外壳，因此未采用。

最终纹理耗时 7.53 s、峰值 RSS 1,041,852 KiB，atlas 为 8192×3853。完整
命令、trim/tolerance A/B、拓扑/UV JSON 和最终 OBJ/PNG 位于
`output/mvs_mesh_hole_fill_house_20260728/`。

## House 平面纹理局部接缝连续性

2026-07-28 在上述 112,107 面同一简化网格、同一相机、1% 遮挡容差和
8 线程下，保留原有每相机常量曝光校正，并增加“纹理区域 × seam 顶点”的
正则最小二乘残差场。分析脚本从 OBJ 中识别 atlas 两侧 UV 不连续的共享边，
在每条边两端和中点向各自三角形内部采样，共统计 48,794 条 seam、
146,382 对 BGR 样本。

| BGR L2 接缝误差 | 仅全局相机偏移 | + 局部 seam 偏移场 | 变化 |
|---|---:|---:|---:|
| 均值 | 32.600 | **24.854** | **-23.8%** |
| 中位数 | 18.962 | **12.303** | **-35.1%** |
| P90 | 82.940 | **68.012** | **-18.0%** |
| P95 | 110.368 | **92.889** | **-15.8%** |
| 最大值 | 396.144 | **326.075** | **-17.7%** |

纹理耗时由 7.53 s 增至 8.86 s（+17.7%，绝对增加 1.33 s），峰值 RSS 从
1,041,852 KiB 变为 1,041,720 KiB，基本不变。全量 WSL CUDA 回归共
307 项，306 通过、1 项按设计跳过、0 失败。命令、计时、JSON 和新 OBJ/PNG
位于 `output/mvs_planar_continuity_house_20260728/`。

许可证边界：局部偏移场依据 Waechter 2014 能量独立实现，COLMAP 4.0
BSD-3-Clause 代码只用于行为核对；OpenMVS 的 AGPL-3.0 实现未移植，后续也
仅采用论文/公开参数层面的 clean-room 设计。

## House 同网格视图碎片修复（2026-07-29）

固定上述 112,107 面网格、相机和去畸变图像，对提交 `eac2246` 的 Colite 与
WSL COLMAP 4.0 做同网格 A/B。修复前 Colite 将完整 1% 端点容差用于所有面，
并逐三角烘焙遮挡拒绝面的 RGB fallback；因此有效 UV 面达到 99.90%，但产生
23,264 个 atlas charts 和 17,142 个单面 charts。

修复后主选择的端点带限制为 0.5%，完整 1% 容差只用于近共面、同视图双邻面
支持且再次通过 BVH 的孔洞恢复；小于 64 面的近共面视图孤岛在整块候选有效且
总投影面积至少保留 50% 时合并，逐三角 fallback 默认关闭。

| 指标 | 修复前 Colite | 修复后 Colite | COLMAP 4.0 |
|---|---:|---:|---:|
| 有效纹理面比例 | 99.90% | **87.96%** | 88.82% |
| atlas chart | 23,264 | **11,300** | 14,742 |
| 单面 chart | 17,142 | **7,003** | 7,872 |
| 不超过 3 面 chart | 20,271 | **9,438** | 11,590 |
| 纹理流形边缝边率 | 31.12% | **17.10%** | 23.01% |

修复后与 COLMAP 的纹理面交并比为 94.28%，覆盖率只差 0.86 个百分点；纹理
耗时 6.05 s，峰值 RSS 1,014,392 KiB。完整参数扫描、日志、JSON 和最终
OBJ/PNG 位于 `output/texture_view_fragmentation_house_20260729/`。

## 历史默认大图 MVS 性能

2026-07-18 House 36 图，默认 `resolution_level=1`（约 2296×1528）、CUDA
PatchMatch、3 iterations、radius 2、8 sources、binary maps。该记录早于
当前几何覆盖算法和严格 COLMAP 对齐参数，只保留为历史实现记录，不能与上节
新鲜半分辨率基准直接比较：

| 指标 | 结果 |
|---|---:|
| 端到端墙钟 | 254.13 s |
| PatchMatch batch wall | 156.44 s |
| CUDA 时间合计 | 141.91 s |
| 峰值 RSS | 约 3.32 GiB |

默认每迭代仅一轮红黑传播并关闭热路径原子诊断。快速预览可用 `resolution_level=2`；全分辨率用 `0` 并建议额外限制 `max_image_size`。

## House SfM 逐阶段 CUDA 对比（2026-07-23）

输入为 `data/house` 36 图，特征上限 8192，CPU 线程 8，CUDA SIFT/CUDA brute-force、guided matching、增量式 SfM；两端均为新鲜输出，不复用数据库或阶段缓存。Colite 使用 `cuda_vlfeat`，COLMAP 4.0 使用其 CUDA SIFT，因此这是系统级流程对比，不是同特征输入的纯算法微基准。

| 阶段 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 特征提取 | 13.26 s | 18.41 s |
| 匹配 + 几何 + guided | 15.65 s | 7.82 s |
| 增量 Mapper + RGB | 37.07 s | 57.97 s |
| Colite 输出写入 | 0.80 s | 已包含于命令 |
| 总计 | 77.28 s | 84.20 s |

Colite 为 36/36、49,287 点、147,890 observations、平均误差 1.1148 px；COLMAP 为 36/36、40,188 点、149,512 observations、平均误差 1.1278 px。Colite 总流程快 6.92 s，但匹配前端仍比 COLMAP 多 7.83 s，是后续性能工作的首要热点。

本轮长期保留三项优化：cuBLAS INT8 双向精确点积使初始匹配 11.64 → 7.18 s；预计算极线项和无除法 Sampson 候选判断使 guided 6.58 → 5.16 s；只对新最优假设执行局部优化使 E/F/H 验证 4.93 → 3.31 s。cuBLAS 路径峰值 RSS 约 1.21 GiB，是用显存/主存换取匹配速度的明确取舍。完整命令、日志和产物位于 `output/sfm_stage_compare_cuda_guided_house_20260723` 与 `output/sfm_cuda_cublas_loransac_fast_house_20260723`。

## SfM：CUDA 前端持久工作区、融合 guided 与成对流水线（2026-07-24）

沿用上一节完全相同的 House、8192、8 线程、CUDA VLFeat、CUDA 精确匹配和增量 SfM 配置，按收益顺序落地三项优化：

1. 持久复用 CUDA 点积矩阵、双向输出、guided 缓冲和 cuBLAS handle；
2. 将 Sampson gate、uint8 描述子点积和双向归约融合到 guided CUDA kernel；
3. 按 image pair 流水执行初始 GPU 匹配、CPU 几何验证和 guided GPU 匹配。

| 版本 | 初始匹配 | 几何 | guided | 前端墙钟 | 全流程内部耗时 |
|---|---:|---:|---:|---:|---:|
| 2026-07-23 基线 | 7.180 s | 3.312 s | 5.162 s | 15.654 s | 77.279 s |
| 持久 CUDA workspace | 2.865 s | 3.298 s | 5.226 s | 11.389 s | 73.062 s |
| + 融合 CUDA guided | 2.822 s | 3.315 s | 1.496 s | 7.633 s | 69.769 s |
| + 成对流水线 | 3.867 work-s | 25.633 CPU-s | 2.110 work-s | 4.267 s | 66.983 s |
| COLMAP 4.0 对照 | — | — | — | 7.820 s | 84.200 s |

流水线版本的阶段列是累计工作量，包含 CPU worker 总量或 GPU mutex 等待，不能横向求和；`frontend_pipeline_seconds=4.267 s` 才是端到端前端墙钟。最终输出为 36/36、49,303 点、147,940 observations、332,718 verified/guided inliers、平均误差 1.1151 px，与优化前等价。相对基线，前端降低 72.7%，全流程降低 13.3%；当前 Colite 前端比本机 COLMAP 4.0 对照快 45.4%，因此后续不再优先重写几何估计器。

完整命令、`/usr/bin/time -v` 日志和模型依次位于：

- `output/sfm_cuda_persistent_workspace_house_20260724/`
- `output/sfm_cuda_fused_guided_house_20260724/`
- `output/sfm_cuda_pair_pipeline_house_20260724/`

## OpenMVG 2.1 SfM 外部基准

House 36 图，OpenMVG 标准 SIFT + exhaustive + geometric filter + incremental SfM：

- 36/36 注册，69,835 structure points，224,155 observations；
- 最终 BA RMSE 0.330936 px；
- 阶段墙钟合计 118.49 s；
- `ComputeSfM_DataColor` 生成 `colorized.ply`，其中额外包含 36 个相机可视化顶点。

它使用不同 SIFT、matcher、相机和 Mapper，不能把高点数直接解释为比 Colite/COLMAP 精度高。OpenMVG 彩色点云是独立后处理：优先覆盖最多未着色点的代表图像取色；Colite/COLMAP 4.0 对所有观测颜色求均值。

## OpenMVS 2.4 CPU/CUDA 外部基准

输入为同一 OpenMVG House scene，36 图、resolution level 1、5 views、3 photometric + 2 geometric iterations：

| 指标 | OpenMVS CPU | OpenMVS CUDA |
|---|---:|---:|
| Densify wall | 1,847.82 s | 108.02 s |
| ROI 后点数 | 6,052,135 | 5,919,855 |
| 峰值 RSS | 4.18 GiB | 4.35 GiB |

CUDA 比 CPU 快 17.11×。一次 88.91 s 的旧记录复用了 CPU depth cache，不是完整冷启动，不能作为结果。CUDA 端到端 MVS：转换 9.79 s、Densify 108.02 s、CPU ReconstructMesh 191.98 s、CPU TextureMesh（先 decimate 0.1）110.15 s，总计 419.94 s。直接纹理化约 648 万面曾在 atlas 阶段崩溃，简化后成功。

OpenMVS `DensifyPointCloud` 包含图像准备、PatchMatch、几何 pass、Fusion、ROI 和输出，不等同于 COLMAP 的单个 `patch_match_stereo` 阶段。

## 复现工具

| 脚本 | 用途 |
|---|---|
| `scripts/compare_colmap4_mvs.py` | 调用 Colite/COLMAP 四个表面命令并记录 JSON |
| `scripts/compare_undistorted_outputs.py` | 去畸变相机和像素比较 |
| `scripts/compare_patch_match_outputs.py` | 覆盖率、共同深度和稀疏残差 |
| `scripts/compare_fused_clouds.py` | 融合点云双向最近邻 |
| `scripts/compare_mesh_outputs.py` | 顶点、拓扑、面积和面集合 |
| `scripts/compare_textured_outputs.py` | UV 覆盖、atlas 和纹理统计 |
| `scripts/create_synthetic_mvs_fixture.py` | 确定性合成 MVS fixture |
| `scripts/evaluate_sphere_mesh.py` | 球面真值评估 |
| `scripts/analyze_fusion_outliers.py` | 将融合点云空间尾部追溯到 seed 视图、像素及 Colite/COLMAP 深度 |
| `scripts/mvs_sweep.py` | MVS 参数扫描 |

新测试不得继续写入 `build-*`。使用例如：

```text
output/colmap4_mvs_building18_YYYYMMDD/
output/openmvs_cuda_house_YYYYMMDD/
```

## 解释结论时的禁区

- 未做 Sim(3) 对齐的独立 SfM 点云不能直接比较欧氏距离。
- 两端独立 `point3D_id` 不能用于点对应。
- 更多稀疏点、稠密点或三角形不等于更准确。
- 热缓存单次 wall time 不能代表稳定速度。
- 不同工作尺寸、source 数、几何 pass 或上游位姿下的 PatchMatch/Fusion 点数不可直接横比。
- 没有真值时，与 COLMAP 更接近只能说明行为对齐，不能证明几何更正确。

## House COLMAP 4.0 Automatic High 基准

`output/colmap4_auto_high_house_20260728/` 使用 WSL2 Ubuntu 22.04、RTX 4070
SUPER、CUDA 13.2、8 线程和 36 张 4592×3056 house 图像完成了 COLMAP 4.0
Automatic High、Colite 独立全流程、共用 COLMAP 相机的受控 MVS 和 Poisson
交叉实验。MVS 上限 2400 px，实际约为原图一半尺寸。

- COLMAP/Colite 均注册 36/36；稀疏点为 39,505/49,299，观测为
  147,672/147,905，平均 track length 为 3.738/3.000。Colite 增加的是短轨迹，
  不能仅按点数判优。
- SfM 经 Sim(3) 对齐后的平均旋转差为 0.176°，平均相机中心误差为最大基线的
  0.0874%。
- 共用 COLMAP 相机时，COLMAP/Colite 稠密点为 2,667,152/2,073,041；
  Colite 少 22.3%，主要优化目标仍是 PatchMatch 覆盖/过滤和 Fusion 接受率。
- COLMAP/Colite PatchMatch 为 1450.6/809.3 s；Fusion 为约 22.5/45.0 s。
  Colite PatchMatch 更快，但受控峰值 RSS 为 9.05 GiB，且 Fusion 仍慢约 2 倍。
- 两套点云分别交叉输入两端 Poisson 后，顶点和面数差异均小于 0.001%，说明
  原生 Poisson 网格差异主要由输入稠密点云造成，不应重写 Poisson 几何算法。
- Colite Poisson 虽传入 `num_threads=8`，运行期曾观察到超过 800% CPU，
  底层线程约束需要继续核查。

完整参数、耗时、内存、GPU 采样、点云/网格/纹理路径和统计 JSON 见该输出目录的
`测试说明.md`。

## House High PatchMatch/Fusion 交叉诊断（2026-07-29）

复用 `output/colmap4_auto_high_house_20260728/` 的同一套 COLMAP 相机、2400×1597 图像和 High 参数，在 `output/mvs_fusion_cross_house_20260729/` 完成深度图、法线图和 Fusion 后端交叉。

| 深度/法线输入 | Fusion | RGB 点数 |
|---|---|---:|
| Colite（修复前逐像素 graph） | Colite | 2,073,022 |
| Colite（20 源图遍历） | Colite | 2,298,163 |
| Colite（再统一法线朝向） | Colite | 2,386,766 |
| Colite | COLMAP 4.0 | 2,395,492 |
| COLMAP | Colite | 2,591,903 |
| COLMAP | COLMAP 4.0 | 2,667,152 |

结论：

- 已完成几何过滤后再次只沿逐像素 consistency graph 遍历会重复约束。改为 20 张有序重叠源图投影验证后增加 225,141 点，Fusion core 从 62.04 秒降到 46.15 秒。
- PatchMatch 平面代价不能区分 `n/-n`。统一法线朝向参考相机后再增加 88,603 点；同一 Colite 输入下两套 Fusion 只差 8,726 点（0.36%），Fusion 点数差距基本消除。
- COLMAP 输入经 Colite Fusion 为 2,591,903 点，相对 COLMAP 自身少 2.82%，主要来自重叠图像选择和边界认领细节。
- 修复后受控 Colite 相对 COLMAP High 仍少 10.51%。深度×法线交叉显示，换用 COLMAP 法线可把 Colite 深度提升到 2,498,004 点，而 COLMAP 深度配 Colite 法线为 2,202,333 点；但符号修复后两套原生组合的法线角 P50 已接近（22.28°/22.66°），Colite 图的投影拒绝仍高 16.3%。下一质量热点是 PatchMatch 深度/平面法线的跨视图重投影一致性，不能把混合实验简单解释为单独法线方差。
- 四邻域深度差分法线只得到 2,105,576 点，屋瓦/深度噪声会放大局部法线抖动，因此该方案被否定。

完整脚本、命令、拒绝计数、误差分布、耗时、内存和点云见 `output/mvs_fusion_cross_house_20260729/测试说明.md`。

## Building-18 每图/共享相机策略（2026-07-30，历史消融）

此项记录默认 Auto 策略落地前的 Per-image/Single 消融。SfM 图像导入从“第一张
图的尺寸和 EXIF 强制共享给全部图像”改为支持每图独立 Camera。使用 18 张同设备、同尺寸、同 EXIF 的
building-18，固定 CUDA VLFeat 8192、CUDA matcher、8 线程和相同 Mapper：

| 指标 | Individual | Shared |
|---|---:|---:|
| 注册 | 18/18 | 18/18 |
| Camera | 18 | 1 |
| 点 / 观测 | 25,378 / 90,457 | 25,681 / 90,961 |
| 平均 track | 3.5644 | 3.5420 |
| 平均重投影 | 0.62582 px | 0.63007 px |
| wall / RSS | 29.21 s / 885,808 KiB | 29.33 s / 886,516 KiB |

两组前端 descriptor、raw match 和 verified inlier 完全一致；Individual BA 后
焦距范围 2501.25–2625.60 px，未出现发散。混合尺寸和逐图 EXIF 的确定性测试
共 29 项通过。完整日志和模型位于：

- `output/sfm_camera_policy_mixed_20260730/`
- `output/sfm_camera_policy_building18_20260730/`

### Fusion 候选解耦与 PatchMatch 源采样先验（2026-07-29）

进一步核对 COLMAP 4.0 后确认，PatchMatch 的 20 个 source 与 Fusion 默认检查
50 个重叠图像是两套独立参数。Colite 原先共用一个上限，使 Fusion 提前截断。
现在 Fusion 按共享稀疏点数独立重算最多 50 个候选，最小三角化角为 0°：

- 同一 Colite maps 从 20 候选的 2,386,766 点增至 2,473,488（+3.63%）。
- 同一 COLMAP maps 经 Colite Fusion 得到 2,666,592 点，对比 COLMAP 自身
  2,667,152 点仅少 560（-0.02%）。Fusion 主算法的点数差距已基本闭合。

PatchMatch 源视图名单也新增 `colmap` 模式：75% 分位三角化角至少 1°，再按
共享点数排序。House 的 720 个候选槽位与原 weighted 模式重合 98.61%，800 px
实验中只额外提升 0.34%，说明名单不是主要瓶颈。真正有效的是 CUDA CDF source
采样加入 COLMAP 式三角化角、入射角和局部单应分辨率先验：

| 800 px 组合 | PatchMatch | 平均代价 | Fusion 点 |
|---|---:|---:|---:|
| 原实现 | 98.05 s | 0.26348 | 287,344 |
| 几何先验 | 98.23 s | 0.22492 | 303,978 |
| COLMAP 名单 + 几何先验 | 98.94 s | 0.22484 | 305,007 |

几何先验以 0.18 秒代价增加 5.79% 融合点，已作为 CUDA PatchMatch 生产默认；
可用 `--patch_match_no_colmap_source_priors` 做兼容 A/B。全尺寸 2400×1597
冷启动验证使用 COLMAP source mode、20 个 PatchMatch source、50 个 Fusion
candidates 和原有 10°法向阈值：

| High 指标 | 优化后 Colite | 旧 Colite（50 candidates） | COLMAP 4.0 |
|---|---:|---:|---:|
| PatchMatch wall | 849.40 s | 约 809.3 s | 1450.6 s |
| PatchMatch 峰值 RSS | 9,142,008 KiB | 约 9.05 GiB | 未重跑 |
| Fusion wall / core | 68.52 / 54.64 s | 未单列 | 约 22.5 s |
| Fusion 峰值 RSS | 8,729,516 KiB | 未单列 | 未重跑 |
| RGB+normal 点 | **2,641,922** | 2,473,488 | 2,667,152 |

全尺寸点数提升 168,434（+6.81%），距 COLMAP 仅 -0.95%；PatchMatch 仍比
COLMAP 快约 41.4%。当前质量差距已从原始受控基准的 22.3% 收敛到 1%内。
后续 MVS 优先级转为 Fusion 50-view 工作集的时间/内存优化，以及在不破坏
10°严格法向一致性的前提下继续降低平面法线近阈值抖动。完整 A/B 和原始产物见：

- `output/mvs_source_selection_house_20260729/`
- `output/mvs_patchmatch_source_priors_house_20260729/`
- `output/mvs_patchmatch_priors_house_high_20260729/`

## House SfM 长轨迹优化（2026-07-30，Per-image 消融）

在当时的 `per_image` 配置、每图 EXIF、CUDA VLFeat SIFT 8192、CUDA matcher、
8 线程和增量式 SfM 下，严格重跑当前代码基线后，Colite 为 44,292 点、
131,478 观测、平均 track 2.968。通过联合多视图 merge、单冲突观测稳健合并、
逐观测局部过滤，以及局部 BA/全局过滤/最终 BA 后重新 complete/merge，最终得到
40,043 点、134,389 观测和平均 track 3.356；观测增加 2.21%，`track>=4` 的观测
占比由 41.96% 提升到 52.41%。唯一视图平均 track 由 2.925 提升到 3.284。

COLMAP 4.0 参考为 39,505 点、147,672 观测、平均 track 3.738。当前点数已接近，
剩余差距主要是少 13,283 个观测，而不是点数过多；同图重复观测仍为 2,886，对比
COLMAP 的 330，表明下一优化重点应是 verified/guided 匹配图的一对多冲突一致化，
不应继续靠强制 merge 压低点数。完整消融、命令、时间、内存、模型和分布 JSON
位于 `output/sfm_long_tracks_house_20260730/`。

## House guided/verified 匹配图一致性（2026-07-30）

使用 `data/house`、CUDA VLFeat 风格 SIFT、8192 特征、CUDA exhaustive/guided matcher、
8 CPU 线程、Per-image cameras 和增量式 SfM。相对长轨迹优化 v8：

| 指标 | v8 | pose-aware 一致性 | 变化 |
|---|---:|---:|---:|
| 注册图像 | 36 | 36 | 0 |
| 稀疏点 | 40,043 | 40,353 | +310 |
| 唯一视图观测 | 131,503 | 132,309 | +806 |
| 同图重复观测 | 2,886 | 0 | -100% |
| 唯一视图平均 track | 3.2840 | 3.2788 | -0.0052 |
| track≥4 观测 | 67,222 | 67,334 | +112 |
| SfM wall | 100.65 s | 106.81 s | +6.16 s |

House 的 332,718 条 verified edges 在 pair 内没有一对多冲突，但形成 1,527 个含同图竞争的
跨图组件。硬拆全局组件会删除 84k–109k 条 verified edges，并将最终观测降到
108k–117k，故未采用。最终实现保留 verified 证据，在 PnP/complete/merge 阶段按已知位姿
与重投影误差消解冲突。

## 12相机模型与默认 Auto 策略对照（2026-07-30）

现有12种模型已补齐统一投影、稳健反投影、BA、去畸变、参数合法性和读写测试；
COLMAP 4.0 成功对含全部12模型的合成 text model 完成 binary 往返，模型名、
参数数量和数值保持一致。完整 CTest 为343/343通过。

默认由历史 `PINHOLE + per_image + 不优化畸变` 改为
`SIMPLE_RADIAL + auto + 优化焦距/畸变`。House 冷启动中 Colite 与 COLMAP 4.0
都把36张图按 EXIF Auto 分成1个 Camera，初始焦距均为3059.4539605 px：

| 指标 | Colite | COLMAP 4.0 |
|---|---:|---:|
| 注册 / Camera | 36 / 1 | 36 / 1 |
| 稀疏点 / 观测 | 46,221 / 145,708 | 40,508 / 150,729 |
| 平均 track | 3.15242 | 3.72097 |
| track=2 / track≥4 点 | 28,308 / 11,226 | 14,559 / 15,085 |
| 平均重投影 | 1.13199 px | 1.14578 px |
| SfM wall | 133.41 s | 78.43 s |

相机分组和初始焦距已完全对齐，最终焦距仅差0.17%。Colite 仍表现为点更多但
短轨迹比例高、总观测少；差距主要来自 descriptor/匹配集合及多视图轨迹处理，
不是 EXIF 或默认焦距。完整模型、命令和资源日志位于：

- `output/camera_models_colmap4_ab_20260730/`
- `output/camera_models_house_colmap4_20260730/`
## P0–P5 收口验证（2026-08-01，跳过 P1）

本轮按确认目标执行 P0、P2–P5，明确冻结 P1 的 SIFT 特征上限与多方向语义。

- P0：SfM 模型支持 COLMAP 兼容 TXT、BIN、both 和 auto 读取；12 个现有相机模型、
  位姿、RGB、Point2D/Point3D、track 与含空格相对路径完成往返测试。`recon_sfm`
  持久化相机/图像、逐图特征、raw matches、verified geometry 和阶段哈希，支持
  `--resume`/`--force`；主要 MVS 阶段可读取已完成稀疏模型与 dense maps 继续执行。
- P2：严格一图一观测后 House 为约 46.2k 点、145.7k 观测、平均 track 3.15，
  相对旧基线约 2.97 明显提高。未达到 3.7 的剩余差距来自冻结的 P1 匹配拓扑，
  不再以同图重复观测或删点伪造长轨迹。
- P3：按图像稠密 correspondence adjacency 将 Mapper 76.39→73.87 s；COLMAP
  4.0 Mapper 为 54.55 s。限制重三角化、unordered adjacency 和 AutoDiff BA
  均因质量或耗时退化而回退。
- P4：保留既有视图流式落盘、有界 Fusion cache 和 binary float 输出。packed
  normal 在 240 s 未完成（旧基准 115.18 s），缩小 worker cache 会触发 I/O
  抖动，均回退；下一步必须是 tile cache/source-affinity 级重构。
- P5：800 px House 中，共面邻居传播虽增点 2.92%，但 24-NN 平面残差 median
  0.12006→0.13898；顺序 sweep 7 分钟未完成；边缘感知法线平滑使点数下降
  3.58% 且残差升至 0.12552。三项均回退，避免把错误深度平滑成连续伪面。

最终 CUDA Release 全量测试为 344/344 通过，另有 1 项因 CUDA 当前可用而条件
跳过，总耗时 14.19 s。原始命令、计时、RSS、点云和平面度 JSON 位于：
`output/p0_resume_baseline_20260801/`、`output/sfm_p3_*_house_20260801/`、
`output/mvs_p4_packed_normal_fusion_house_20260801/` 和
`output/mvs_p5_planar_support_house_20260801/`。

## House guided ratio 与前端拓扑隔离（2026-08-09）

使用 COLMAP 4.0 House 数据库的同一套关键点和描述子隔离测试后，Colite CUDA 原始匹配得到 223,256 条边，完整包含 COLMAP 的 223,043 条边，召回率 100%、精确率 99.9046%。因此 CUDA descriptor matcher 不是 Colite 自有图偏稀的原因。

COLMAP High 在 guided matching 中继续使用 `max_ratio=0.8`。同描述子 A/B 中，Colite ratio 0.8/0.9/0.95 相对 COLMAP verified 图的 precision 分别为 84.14%/53.51%/34.88%，Jaccard 为 71.37%/48.91%/32.94%，确认旧的隐式 0.95 不能代表 COLMAP High 同描述子语义。CLI 现开放独立 `--guided_match_ratio`，不再把 guided 比率隐式绑定到 nearby matching。

但 Colite 自有 CUDA VLFeat 严格 8192 特征/图时，全局 ratio 0.8/0.9/0.94 最终仅保留 79,351/114,910/139,555 observations，均低于 0.95 生产基线 146,640；对应 mean track 为 4.080/3.479/3.230。故 `recon_sfm` 暂保留 0.95 默认值以满足“观测总数不下降”，standalone matcher 的同描述子默认值为 0.8。跨图冲突硬剪枝也只得到 123,012 observations，已回退。

描述子 uint8/512 量化只使 raw matches 从 71,907 变为 71,909，可排除量化。CPU VLFeat 虽得到 232,513 raw matches，但实际平均输出 12,120 特征/图；COLMAP House 也因多方向语义平均超过 10k，而当前 CUDA 分支固定 8192。该差异属于本轮冻结的 P1（特征上限与多方向语义），不能通过扩容特征解决。

在保持 P1 冻结后，首轮图内 completion 对前四跳无候选的目标自适应扩展到六跳；随后新增最终位姿双描述子补全：已有 Point3D 投影到尚未观测的注册图像，只接受 2 px 邻域内、与至少两条现有轨迹描述子的归一化角距离第二优值不超过 0.301 rad 的未分配特征，再经过全局过滤和最终 BA。最终确定性复跑得到 46,058 点、152,013 observations、mean track **3.300469**、重复观测 0，平均重投影误差 1.118657 px；其中 pose-guided 阶段新增 5,196 observations，仅耗时 0.105 s。Mapper 61.989 s，为 COLMAP 54.55 s 的 1.136×。相对 146,640 / 3.179876 基线增加 5,373 observations，达到阶段目标且未靠删点或重复观测抬高指标。完整原始记录见 `output/p2_matcher_colmap_descriptors_*_house_20260809/`、`output/p2_guided_ratio*_house*_20260809/`、`output/p2_completion_adaptive_depth6_house_20260809/`、`output/p2_pose_guided_completion_angle0301_house_20260809/` 和 `output/p2_pose_guided_completion_final_house_20260809/`。

## P0/P2 第二轮性能与资源验收（2026-08-01）

本轮固定 House、8 CPU 线程、CUDA SIFT/CUDA matcher/CUDA PatchMatch、增量式 SfM，
继续冻结 P1 的 SIFT 8192 上限与多方向语义。三个 P0 与两个 P2 的验收结果如下：

| 项目 | Colite 结果 | COLMAP 4.0 参考 | 状态 |
|---|---:|---:|---|
| P0 Fusion（2296 px） | 30.63 s，3.74 GiB | 28.54 s | 达标，1.073× |
| P0 MVS 峰值内存（2296 px 全流程） | 6,146,208 KiB / 17:16.21 | 初始 Colite 7,435,184 KiB / 16:46.41 | 首阶段达标，5.86 GiB |
| P0 PatchMatch 平面度 | 2,492,697 点，median/P95 0.14450/0.37018 | 旧 Colite 0.18413/0.41458 | 达标，-21.52%/-10.71% |
| P2 Mapper | 61.989 s | 54.55 s | 达标，1.136× |
| P2 SfM 长轨迹 | 152,013 观测，mean track 3.300469，重复观测 0 | mean track 3.738 | 首阶段达标，观测较旧基线 +5,373 |

Fusion 不再加载几何过滤后已无判定用途的 consistency graph。内存式 MVS 将
PatchMatch depth/normal/cost 的所有权直接移交给 Fusion，CUDA refinement 通过
非拥有指针引用 source depth；大图只保留一个 refinement workspace，并在 CPU
consistency graph 构建前释放 device/pinned workspace。该组合将全流程峰值压到
5.53 GiB，同时 Fusion core 保持 29.42 s。

Mapper 的主要收益来自三处：全局 complete/merge 第一轮扫描全部点、后续仅扫描
真正发生 complete/merge 的 dirty points；默认 SIMPLE_PINHOLE、PINHOLE 和
SIMPLE_RADIAL 路径使用投影、相机参数及归一化四元数解析 Jacobian；首轮 graph
completion 对前四跳无候选的目标自适应扩展至六跳。全局 BA 后的 pose-guided 阶段再以
最终位姿、2 px 重投影窗口和双描述子一致性保守补全观测。House 最终保持 46,058 点，
观测由旧基线 146,640 增至 152,013，平均重投影误差 1.118657 px，未以删点、重复观测
或减少 BA 换取指标和耗时。

长轨迹方面，基于空间邻近或硬拆跨图冲突组件的强制合并会引入缺乏匹配证据的错误轨迹，
均已拒绝。最终位姿补全只接受轨迹中至少两条描述子共同支持的未分配特征，新增 5,196 个
观测仅耗时 0.105 s，将 mean track 提升到 3.300469，达到冻结 P1 条件下的首阶段目标。

完整目标、命令与原始结果位于：

- `output/optimization_p0_p2_house_20260801/`
- `output/p0_fusion_graphless_house_20260801/`
- `output/p0_mvs_bounded_workspace_house2296_20260801/`
- `output/p0_patchmatch_geom_weight06_house800_20260801/`
- `output/p0_patchmatch_geom_weight10_filter1_house800_20260801/`
- `output/p2_mapper_quaternion_jacobian_house_20260801/`
- `output/p2_pose_guided_completion_final_house_20260809/`

### 2026-08-02 继续优化结果

PatchMatch 旧 CUDA 路径把 `1.0` 同时当作光度无效哨兵和组合代价上界；几何项叠加后
大于 1 的合法 source 因而被误删。单独改为非有限哨兵可恢复覆盖率，但会暴露 source
深度噪声。最终组合进一步把几何项从光度 source 排名中解耦，以有效 source-depth
重投影误差的较低一半均值形成稳健跨视图共识。2296px 相对旧有界 workspace 基线：

- 点数 2,441,416→2,492,697（+2.10%），观测 49,035,385→51,376,638（+4.77%）；
- median 0.184127→0.144495（-21.52%），P95 0.414580→0.370183（-10.71%）；
- wall 17:13.50→17:16.21（+0.26%），RSS 5.53→5.86 GiB，仍低于 6 GiB。

SfM 全局 completion 可穿越已关联/冲突中间观测寻找后续安全候选。逐 Point3D 五跳 BFS
虽增加 1,050 观测，但使 Mapper 升到 80.35 s。最终实现改为 8 线程只读计算四跳
Point2D proposals、按误差确定性串行提交，得到 46,140 点、146,531 观测、mean
3.175791、重复观测 0；Mapper 61.921 s。最终模型的一跳 verified 上限只剩 1 个可补
观测和 1 对 disjoint merge，说明冻结 P1 后无法在不引入无证据合并的情况下达到 3.3。

对同一最终模型继续做冲突消解与空间合并上限审计：同图冲突 6,258 个中，仅 132 个
候选能以至少 0.25 px 改善当前观测；4,442 对“删除同图冲突后可合并”的轨迹会累计
丢失 7,064 个观测，mean 仅到 3.1847。6 px 空间邻近虽然可在数字上提供 1,870 次
不丢观测合并，但 3,462 个几何候选中仅 100 个具有四跳以内 verified 路径，其余会把
同一表面上的不同特征错误折叠。上述两种统计优化均被拒绝，生产结果维持 3.175791。
verified 图本身存在一个 170,496 特征节点、37,576 个现有 3D 点的巨型重复纹理组件，
因此也不能按完整连通分量建轨；若继续冲击 3.3，必须解除 P1 冻结并改善描述子匹配
拓扑，而不是继续放宽 Mapper 的几何门槛。

完整结果位于：

- `output/p0_patchmatch_robust_geom_house2296_20260802/`
- `output/p2_track_parallel_depth4_house_20260802/`
- `output/p2_mapper_quaternion_jacobian_house_20260801/verified_merge_upper_bound.json`

### House P2 匹配图归因与单次第五跳（2026-08-09）

COLMAP 4.0 数据库有 367,448 条 verified 边，Colite 自有图为 333,097 条；边数仅差
10.3%，但无三角闭环支持的比例分别为 26.8% 和 55.6%，三角支持 >=3 的边分别为
174,498 和 74,222。将 COLMAP 图原样转换给 Colite Mapper 后，Colite 输出
40,495 点、154,880 观测、mean track 3.824670、同图重复观测 0；COLMAP 模型为
39,505 点、147,672 观测、mean 3.738058。这证明当前 Mapper 能利用长轨迹证据，
自有结果未达 3.3 的主因是冻结 P1 后的 verified 图闭环拓扑。

不修改 P1 的生产改进仅在第一次全局 complete/merge 对四跳内无安全候选的目标搜索
第五跳，后续 BA/过滤后的 completion 仍固定四跳。House 得到 46,115 点、146,640
观测、mean 3.179876、重复观测 0，Mapper 61.653 s；相对四跳基线增加 109 观测，
同时比 61.921 s 略快。完整命令和数据位于：

- `output/p2_track_colmap_graph_audit_house_20260802/`
- `output/p2_track_single_extended_pass_house_20260809/`

另外测试了不改变 SIFT 数量的两跳闭环补边：候选必须有至少两个共同中间特征，且
通过描述子 0.7 和最终相机几何 2 px 门槛。22,863 条补边若从初始化开始启用会使
mean 3.179876→3.177407；仅用于已完成模型续建时，相对严格控制组也只净增 35 个
观测、mean +0.000550，wall 增加 0.56 s。该方案收益不足，未进入生产路径；原始
数据位于 `output/p2_track_cycle_closure_house_20260809/` 和
`output/p2_track_cycle_closure_postmap_mapper_house_20260809/`。

### 严格特征上限与 mean track（2026-08-09）

保持 CUDA VLFeat 算法不变，仅扫描严格输出上限。House 的 COLMAP 数据库共有
368,100 个描述子（均值 10,225/图）；Colite 上限 10,240 得到 368,640 个，差
0.147%，是当前最接近的点数口径。但完整 SfM 得到 57,828 点、188,281 观测、
mean track 3.255880，低于 8192 上限基线的 3.300469，也低于 COLMAP 的
3.738058。将 guided ratio 收紧到 0.8 虽得到 mean 4.332738，却只保留 101,945
观测（COLMAP 的 69.0%），属于删除短轨迹换均值，未采用为默认配置。因此后续
长轨迹优化应继续针对 verified 图闭环与冲突一致性，不应继续增加特征数或收紧
过滤。完整结果位于：

- `output/feature_cap_scan_house_20260809/`
- `output/feature_cap10240_sfm_house_20260809/`
- `output/feature_cap10240_guided08_sfm_house_20260809/`

### Building-18 学习前端 CPU/CUDA A/B（2026-08-13）

在每图严格 2048 特征、18/18 图注册的统一口径下，CvSift+FLANN 为 5,788 点、
18,414 观测、0.617 px、22.78 s。CPU ALIKED+BF 为 3,618 点、16,195 观测、
0.353 px、105.13 s；CPU ALIKED+LightGlue 为 5,371 点、23,790 观测、
0.582 px、204.18 s。

ONNX Runtime 1.27 CUDA 13 execution provider 在 RTX 4070 SUPER 上实际执行后，
ALIKED+LightGlue 的匹配阶段从 89.47 s 降到 9.21 s（9.7 倍），最终为 5,375 点、
23,779 观测、0.578 px。高分辨率 ALIKED 单 session 已使用约 11.69 GiB 显存，
提取耗时 167.78–179.29 s，慢于 CPU 多 session 的 94.41–100.68 s；但主机峰值
RSS 从约 27.9 GiB 降至约 2.3 GiB。完整 GPU 流水线 ALIKED+BF 和
ALIKED+LightGlue 分别为 204.23 s 和 196.67 s。结果说明 CUDA 对 LightGlue
收益明确，而 ALIKED 还需要输入缩放、I/O binding 和 memory arena 调优。

完整日志、GPU 200 ms 采样和报告位于：

- `output/sift_vs_learned_building18_20260813/`
- `output/sift_vs_learned_gpu_building18_20260813/`
