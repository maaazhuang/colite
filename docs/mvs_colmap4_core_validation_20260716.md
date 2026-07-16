# colite 与 COLMAP 4.0 去畸变 / MVS 核心对比（2026-07-16）

## 结论

本轮使用 `data/building-18` 的 18 张图、同一份已注册位姿和同一份 COLMAP
去畸变工作区，在 WSL Ubuntu 22.04、RTX 4070 SUPER、CUDA 13.2 上进行 A/B。

- **Image undistorter 尚未对齐 COLMAP 的输出相机策略。** colite 保留 800x600、
  单一焦距缩放；COLMAP 输出 800x579，并分别求 `fx/fy`、裁掉无效边界。colite
  当前明显更快，但这部分速度收益包含了更简单的相机拟合策略，不能直接视为同精度加速。
- **光度 PatchMatch 的中位精度已经接近，但错误长尾和效率仍落后。** 三轮质量档的
  稀疏点相对深度误差中位数为 colite 2.21%、COLMAP 2.43%；P95 分别为
  50.87% 和 22.50%。纯光度墙钟时间分别为 34.76s 和 15.46s。
- **几何第二遍、过滤和 fusion 尚有明显结构差距。** 最终有效深度覆盖率为
  colite 1.08%、COLMAP 68.06%，融合点数为 780 与 65,751。colite 保留下来的
  点大多落在 COLMAP 表面附近，但覆盖严重不足，当前不能认为 MVS 已对齐。

## 测试环境和隔离构建

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| WSL | Ubuntu 22.04 |
| CUDA | 13.2，`sm_89` |
| colite | `build-wsl-mvs-cuda-20260716`，当前源码，CUDA SIFT/MVS 开启 |
| COLMAP | 4.0.0，`build-wsl-colmap4-cuda132-reference`，CUDA 开启，ONNX/GUI/OpenGL 关闭 |
| 输入 | 18 张 COLMAP 去畸变图，800x579，共用相机、位姿和 source 数量 8 |
| PatchMatch 调度 | GPU 0，CPU 调度线程 1，最大边 256 |

COLMAP 4.0 发布树中 `PatchMatchCuda::InitWorkspaceMemory` 会创建 `(0,0,0)` 的
`GpuMat` 占位对象，但同版本的 `GpuMat` 构造器禁止零尺寸，导致全部视图启动失败。
隔离参考树将该占位改为 `(1,1,1)`；`filter=false` 的模板分支不会读取它，数值路径
不变。该修正只用于本次参考构建，未替换系统 COLMAP 3.11。

## 1. 图像去畸变

### 输入和方法

使用相同 OPENCV 相机：

```text
OPENCV 3072 2304 2457.6 2457.6 1536 1152 -0.18 0.04 0.001 -0.0005
```

两边均设置 `blank_pixels=0`、`min_scale=0.2`、`max_scale=2`、最大边 800。
输出比较脚本会先按两个 PINHOLE 相机的内参把候选图映射到参考相机，再在共同有效区域
计算像素差：

```bash
python3 scripts/compare_undistorted_outputs.py \
  --reference_images build-wsl-cuda/mvs-colmap-ab-20260716/colmap4-undistort/images \
  --reference_cameras build-wsl-cuda/mvs-colmap-ab-20260716/colmap4-undistort/sparse-text/cameras.txt \
  --candidate_images build-wsl-cuda/mvs-colmap-ab-20260716/colite-undistort/undistorted/images \
  --candidate_cameras build-wsl-cuda/mvs-colmap-ab-20260716/colite-undistort/undistorted/sparse/cameras.txt \
  --output build-wsl-cuda/mvs-colmap-ab-20260716/undistortion-comparison.json
```

### 结果

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | 3.12s | 22.98s |
| 输出尺寸 | 800x600 | 800x579 |
| 输出焦距 | fx=fy=605.929 | fx=593.982, fy=593.638 |
| 有效像素比例 | 99.62% | 通过裁剪控制 |

把 colite 输出重投影到 COLMAP 输出相机后，共同区域为 98.0%，RGB RMSE 10.93，
PSNR 27.36dB。该像素误差同时包含插值核、JPEG 编码和输出相机拟合差异，不能单独
当作几何误差。另一个限制是测试照片本身并非由上述畸变模型合成，因此本项验证的是
两种映射策略的一致性，不是相对于真实无畸变图的绝对误差。

## 2. PatchMatch 原始单轮档

双方均使用：窗口半径 2、15 个随机样本、1 轮、8 个 source、无几何项、无过滤。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | 4.54s | 3.93s |
| 有效深度覆盖率 | 100.00% | 99.97% |
| 稀疏点深度误差 median | 5.54% | 2.39% |
| 稀疏点深度误差 P90 | 234.88% | 36.07% |
| 稀疏点深度误差 P95 | 400.55% | 69.14% |

单轮随机初始化对实现细节很敏感，主要用于暴露收敛速度和错误长尾，不作为最终质量结论。

## 3. 三轮光度质量档

双方均使用半径 5、双边 NCC、`sigma_spatial=5`、`sigma_color=0.2`、15 个样本、
3 轮和 8 个 source。colite 开启动态 source selection，COLMAP 使用其原生概率消息传递。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间（纯光度） | 34.76s | 15.46s |
| CUDA 核心累计时间 | 32.93s | 12.38s |
| 有效深度覆盖率 | 100.00% | 99.87% |
| 稀疏点深度误差 median | **2.21%** | 2.43% |
| 稀疏点深度误差 P90 | 15.95% | **10.78%** |
| 稀疏点深度误差 P95 | 50.87% | **22.50%** |
| 稀疏点深度误差 mean | 16.25% | **7.28%** |

结论是中位深度已经接近，但 colite 的错误长尾约为 COLMAP 的 2 倍以上，且当前 CUDA
光度阶段约慢 2.25 倍。直接逐像素比较的中位相对差为 1.95%，P90 为 121.24%；高分位
包含弱纹理/遮挡区中不同局部极值，不宜要求两种随机 PatchMatch 输出逐像素完全相同。

## 4. 几何一致性和过滤

质量档均执行 3 轮几何第二遍和最少 2 个一致 source。COLMAP 使用
`filter_geom_consistency_max_cost=1`；colite 默认控制组使用 2px 重投影和 1% 相对深度。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 完整质量档墙钟 | 45.50s | 34.32s |
| 最终有效深度覆盖率 | **1.08%** | **68.06%** |
| 稀疏点被有效深度覆盖 | 515 / 39,202 | 35,098 / 39,202 |
| 已保留稀疏点误差 median | 1.52% | 2.02% |

colite 保留下来的少量像素精度不错，但召回率不可接受。将 colite 改为 1px 重投影并把
相对深度阈值放宽到 100% 后，有效覆盖率也只提高到 1.46%，说明阈值语义只能解释小部分
差距。主要结构差异仍包括：

1. COLMAP 的几何 cost 是前后向重投影误差，并在每个 sweep 内与 NCC 概率、三角角度、
   入射角和分辨率先验共同决定 source；colite 的几何项与最终一致性图是分开的简化路径。
2. COLMAP 对每个像素保留逐 source 的选择概率并以 Monte Carlo 抽样；colite 虽有动态
   概率，但传播、候选构成、概率消息和 best-cost 聚合语义仍不同。
3. colite 的几何图额外使用显式相对深度约束，且双线性深度采样、像素中心约定及边界
   处理需要用合成平面逐项验证。

## 5. Stereo Fusion

双方使用最少 2 个观测、2px 重投影、1% 深度、10 度法线阈值。

| 指标 | colite | COLMAP 4.0 |
|---|---:|---:|
| 墙钟时间 | 1.54s | 2.95s |
| 融合点数 | **780** | **65,751** |
| candidate→COLMAP 最近邻 median / P95 | 0.00200 / 0.00535 | — |
| candidate→COLMAP，按场景 bbox 对角线归一化 | 0.108% / 0.289% | — |
| COLMAP→candidate 最近邻 median / P95 | 0.06490 / 0.15262 | — |

colite 的 780 点多数位于 COLMAP 曲面附近，说明融合坐标和基本反投影没有整体错误；反向
距离很大，直接反映覆盖不足。Fusion 本身的 1.9 倍墙钟优势不能抵消输入只剩 1.08% 深度
像素，也不能作为同工作量速度比较。

## 优化优先级

### P0：先对齐几何第二遍与一致性图

1. 在 `ComputeGeomConsistencyCost` 层面复现 COLMAP 的**纯前后向重投影 cost**，将相对
   深度阈值移出 COLMAP 兼容模式；明确使用 `pixel + 0.5` 的纹理采样约定。
2. 几何第二遍应读取完整第一遍 photometric depth，逐 source 计算几何 cost，并在最后
   sweep 同时生成 consistency mask；不要用独立后处理近似替代核心判据。
3. 加入合成 fronto-parallel/斜平面测试：已知深度下每个 source 的 forward-backward
   reprojection 必须与 COLMAP CPU 参考公式在 1e-4px 内一致。
4. 新增召回率门槛：building-18 256px、min consistent=2 时，有效覆盖率先达到 COLMAP
   的 90% 相对水平，再讨论 fusion 调参。

### P0：压缩光度错误长尾

1. 对齐 COLMAP 每像素 source 概率的 forward/backward message、三角角/入射角/分辨率
   先验和 Monte Carlo 抽样，而不只是 NCC 概率 EMA。
2. 对齐每 sweep 的 5 类候选：当前、传播、深度扰动、法线扰动及二者组合，并复核深度
   扰动的 inverse-depth 分布和随 sweep 指数衰减。
3. 以稀疏点 P90/P95 为回归指标；中位数已经接近，后续优化不应只盯 median。

### P1：CUDA 效率

1. 当前三轮光度 CUDA 累计 32.93s，COLMAP 为 12.38s。优先 profile propagation 和
   perturbation；colite 诊断中它们占绝大多数时间。
2. 避免每个候选对所有 source 重复计算单应/窗口 NCC；复用 reference tile、积分量和
   source 几何变换，检查动态 source selection 是否产生不必要的全 source 扫描。
3. 分开报告 JPEG/工作区 I/O、host wrapper 和 CUDA kernel，避免缓存命中造成错误结论。

### P1：去畸变输出相机

1. 实现独立 x/y 边界搜索与最小有效 ROI，允许输出宽高变化和独立 `fx/fy`，对齐
   COLMAP 的裁剪/缩放策略。
2. 用规则网格以外的边界极值或自适应采样替代当前 64x64 空白比例近似。
3. 增加真正合成畸变的 checkerboard 测试，报告重投影几何误差，不再只使用 JPEG PSNR。

### P2：Fusion

等几何深度覆盖率对齐后，再比较 traversal、mutual consistency、track 去重、法线平均和
颜色融合。当前 fusion 输出差距主要由上游深度过滤造成，直接优化 fusion 收益很小。

## 可复现比较脚本

```bash
# 深度图覆盖、逐像素差和稀疏点深度残差
python3 scripts/compare_patch_match_outputs.py \
  --colite_depth_dir <colite>/stereo/depth_maps \
  --colmap_depth_dir <colmap>/stereo/depth_maps \
  --colmap_suffix geometric \
  --model_path <workspace>/sparse-text \
  --output comparison.json

# 融合点云双向最近邻距离（仅依赖 NumPy，分块计算）
python3 scripts/compare_fused_clouds.py \
  --candidate <colite>/dense/fused.ply \
  --reference <colmap>/fused.ply \
  --output cloud-comparison.json
```

本轮完整产物位于：
`build-wsl-cuda/mvs-colmap-ab-20260716/`。

## 2026-07-16 P0/P1 优化结果

本节记录按上述优先级实施后的同工作区复测。

### 已实现

- 修复几何过滤顺序依赖：所有 consistency graph 统一读取未过滤深度快照，完成全部
  graph 后才原地过滤，避免前面图像的孔洞级联到后续 source。
- 几何第二遍改为 COLMAP 风格的截断 forward-backward reprojection cost；关闭几何
  refinement 中 COLMAP 不存在的遮挡/冲突硬 gate。
- 最终 consistency mask 默认使用 1px 重投影误差；相对深度阈值默认 `0`（禁用），
  仍可通过命令行显式启用。
- 明确区分 COLMAP `num_samples=15` 的 source 抽样与 colite
  `patch_match_perturbation_samples` 的随机平面候选。效率档恢复每像素 1 组扰动候选，
  避免把 1 组联合/法线/深度候选错误放大为 15 组。
- 新增 `--patch_match_source_samples 15` 的逐像素 source CDF 抽样；它复用每个 source
  已计算的 NCC，不重复窗口计算。CDF 单独使用时覆盖率从 54.15% 降到 48.97%。
- 新增 `--patch_match_directional_hmm`：CUDA 为每个 source 保留独立 emission、
  backward-message 和 previous-probability 缓冲，使用 COLMAP 4.0 的 forward、
  backward 与 selection probability 公式，再进行逐像素 CDF 抽样。
- CUDA 显式 `four_direction` 路径去除了方向 0/1 重复评估的第二个邻居；由于并行
  四方向缺少行列 sweep 依赖、质量回退，默认仍保持 checkerboard。
- 去畸变改为完整边界扫描、独立 x/y 尺寸拟合和实际整数宽高比例缩放；warp 在源
  分辨率完成后再降采样，并对齐 COLMAP 的 `(0.5, 0.5)` 像素中心。

### 优化前后结果

| 指标 | 优化前 colite | 优化后 colite | COLMAP 4.0 |
|---|---:|---:|---:|
| 去畸变输出尺寸 | 800x600 | **800x579** | 800x579 |
| 去畸变内参 | 不同 | **四项完全一致** | reference |
| 去畸变 RMSE / PSNR | 10.93 / 27.36dB | **3.70 / 36.76dB** | reference |
| 去畸变墙钟 | 3.12s | 5.43s | 22.98s |
| 纯光度 PM 墙钟 | 34.76s（误用 15 组扰动） | **17.36s** | 15.46s |
| 光度稀疏点 median | 2.21% | 2.32% | 2.43% |
| 光度稀疏点 P95 | 50.87% | 74.69% | 22.50% |
| 几何过滤后覆盖率 | 1.08% | **54.15%** | 68.06% |
| 最终稀疏点 median / P95 | 1.52% / 5.08%（仅515点） | **1.55% / 6.22%（29366点）** | 2.02% / 7.03% |
| Fusion 点数 | 780 | **32,886** | 65,751 |

几何 P0 已从不可用提升到 COLMAP 覆盖率的 79.6%，且已保留区域的精度达到参考水平。
仍未完成的主要差距是光度阶段 P90/P95：减少误用的扰动候选后速度接近 COLMAP，但
错误长尾上升，因此继续实现了真正的逐像素 source HMM/CDF 抽样，而不是重新增加
随机平面候选。最终结果见下一节。

## 2026-07-16 HMM source probability message / Fusion 优化

### HMM + CDF15

质量档使用 `--patch_match_directional_hmm --patch_match_source_samples 15`。当前
checkerboard PatchMatch 在每轮结束后分别做行、列 forward-backward，因此概率公式已
对齐 COLMAP 4.0，但调度仍不是 COLMAP 的逐列/逐行 sweep 内联消息。

| 指标 | best-K 基线 | HMM + CDF15 | COLMAP 4.0 |
|---|---:|---:|---:|
| 纯光度墙钟 | 17.36s | 18.11s | 15.46s |
| 光度稀疏点 median | 2.32% | 2.69% | 2.43% |
| 光度稀疏点 P90 / P95 | 18.81% / 74.69% | **17.13% / 51.19%** | 10.78% / 22.50% |
| 完整几何墙钟 | 41.15s | 41.98s | 34.32s |
| 几何过滤后覆盖率 | 54.15% | **55.59%** | 68.06% |
| 最终稀疏点 median / P95 | 1.55% / 6.22% | **2.05% / 6.04%** | 2.02% / 7.03% |

HMM 把纯光度 P95 降低约 31.5%，并把最终覆盖率提高 1.44 个百分点；最终保留点的
median/P95 已与 COLMAP 同量级。由于纯光度长尾仍明显高于 COLMAP，且调度语义尚未
完全相同，当前保留为显式质量开关，不改变历史默认输出。

### Fusion

- 完整像素一次性缓存世界坐标、世界法线和颜色，缓存构建使用 `core/parallel_for`。
- visited 改为 COLMAP 的 enqueue 即标记语义，避免不足最小 track 的像素被重复遍历。
- 点、法线和颜色改为分量中位数聚合；pair diagnostics 改为固定矩阵，候选邻域复用
  thread-local 缓冲。
- 修复最关键的 track 漂移：所有观测只与 seed normal 比较。旧实现取
  `min(seed angle, current BFS angle)`，可沿曲面逐步漂移，实测最大错误 track 达
  44,951 个观测。

在 HMM + CDF15 深度上，Fusion 由漂移修复前 29,785 点提升到 **58,206** 点，达到
COLMAP 65,751 点的 **88.5%**。candidate→COLMAP 最近邻 median/P95 为
0.00374/0.01307，COLMAP→candidate 为 0.00344/0.01179；相较旧 32,886 点输出，双向
覆盖和表面距离均明显改善。单线程墙钟 8.76s，仍慢于 COLMAP 2.95s；当前剩余主要
性能差距是 seed-track 主循环尚未并行，像素缓存并行本身不足以覆盖更多 track 带来的
工作量。
