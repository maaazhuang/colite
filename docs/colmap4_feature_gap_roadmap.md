# Colite 跟进 COLMAP 4.0 特性差异与开发路线图

## 当前实施顺序（2026-10-18）

用户已暂缓数据库：先进行参数行为对齐，再做特征覆盖/GPU 匹配优化。最新结果见 [10-18 记录](sfm_alignment_20261018.md)，下面的数据库优先级属于此前审查建议。

## 审查入口（2026-10-13）

当前差异、互通边界和实施顺序以 [COLMAP 4.0 数据互通审查](colmap4_interop_audit_20261013.md) 为准。下文保留历史开发记录，其中“尚未实现 ALIKED”“顺序局部任务”、旧 CLI 示例及“P4 暂不开发”不能作为当前状态或范围约束。用户现已要求相同流程参数对齐，以及稀疏模型、特征和匹配数据互通；SQLite 适配和完整数据契约提高为 P0。

## 正确性与规模优化（2026-09-30）

- structure-less 回退改为尺度不变的平移射线交会；共线退化明确失败，候选通过
  多邻居正深度、重投影和视差验证后才注册。尚不是完整 generalized relative pose。
- 旋转异常边剔除后重新拟合，最终有效边传递给定位；定位改为稀疏 QR，并在每轮
  过滤后检查锚点连通分量、秩及正基线，不输出无约束相机。
- `recon_sfm --resume` 的分阶段身份补齐 ALIKED 输入尺寸、BA 参数名称与注册
  重试阈值；浮点参数按可往返精度编码。
- 局部 hierarchical 任务已经支持有界并行；下面的“顺序执行”属于历史状态。
  后续重点仍为真实大规模 A/B、未合并模型保留、模型图优化和学习前端 GPU 调度。

## 历史开发状态（2026-08-11）

开发分支：`codex/colmap4-feature-gap`。P4 按计划暂不开发。

| 阶段 | 当前边界 | 后续闭环 |
|---|---|---|
| P0 | 已完成：类型化 feature/matcher/descriptor 契约、workspace feature v2、校验和与损坏检测、v1 兼容读取、缓存身份升级 | 兼容性维护 |
| P1 | 核心链路已完成：可选 ONNX Runtime、ALIKED N16Rot/N32、ALIKED cosine BF、SIFT/ALIKED LightGlue、EXIF Orientation 1–8、独立命令及 `recon_sfm` 接入；官方模型清单、SHA-256 自动校验、CPU golden inference 和 CUDA EP 已加入 | 扩展弱纹理/宽基线数据集 A/B |
| P2 | 首条全局流水线已接通：view-graph 筛选、旋转平均、全局相机中心、track 三角化/合并、全局 BA，并支持 `recon_sfm/recon_auto --sfm.mapper global` | 更鲁棒的 view-graph calibration、固定旋转的位置/结构优化、分阶段重三角化与大数据集 A/B |
| P3 | `transitive_matcher`、重叠 view-graph 聚类器和 hierarchical mapper 已实现；局部增量重建、公共 verified track 的 Sim(3) 重试合并、最终 BA 已接通；structure-less 多邻居位姿回退已接入增量 mapper | generalized relative pose；局部任务并行/模型图联合优化；position prior absolute-pose refinement |

官方 COLMAP 3.13.0 的 ALIKED N16Rot/N32、ALIKED-LightGlue 与 SIFT-LightGlue 已通过
SHA-256 校验并完成真实 CPU golden inference；ALIKED+LightGlue CUDA golden 也已用
ONNX Runtime 1.27 CUDA 13 execution provider 实际执行。测试在缺少模型或 ONNX
构建时明确跳过，不把网络下载引入常规构建。building-18 已完成等 2048 特征预算的
CvSift、ALIKED+BF 和 ALIKED+LightGlue CPU/CUDA 端到端 A/B。

## 1. 结论

Colite 0.1.0 已经覆盖传统单相机重建的主干：SIFT、暴力/FLANN 匹配、guided
matching、增量 SfM、多模型、PatchMatch、Stereo Fusion、Poisson/Delaunay、QEM
简化和纹理映射。与 COLMAP 4.0 的主要差距不是基本重建流程，而是以下三条新能力链：

1. **类型化深度学习特征前端**：ALIKED N16Rot/N32、SIFT/ALIKED LightGlue、
   ONNX Runtime、模型资源管理以及能区分描述子类型和存储类型的基础设施。
2. **全局与大规模 SfM**：view graph calibration、rotation averaging、global
   positioning、GLOMAP 全局流水线，以及 hierarchical mapping/model clustering。
3. **鲁棒性与工程生态**：structure-less image registration、EXIF 方向自动旋转、
   SQLite 数据库互操作、导入/清理工具和 Python/GUI 等外围能力。

建议首先实现深度学习前端，但第一项提交应是特征数据模型和持久化协议升级，而不是直接
嵌入 ALIKED。当前 `FeatureSet` 只有无类型的 `float` 描述子矩阵，现有 SIFT 匹配又依赖
其量化语义；直接加入 ALIKED 会让维度、数值类型、距离度量和 matcher 兼容性变成隐式约定。

## 2. 对比范围与证据

- Colite 基线：当前分支创建时的 `main`，版本 `0.1.0`。
- COLMAP 基线：仓库内 `references/colmap-4.0.0`，其 changelog 标记版本日期为
  2026-03-14。
- 不纳入差异和开发目标：rig/多相机 frame 体系、rig 约束匹配、全景系统，以及
  FISHEYE、SIMPLE_DIVISION、DIVISION 等特殊/新增相机模型。
- 本文比较算法和可使用能力，不要求复制 COLMAP 的内部类层次、命令行拼写或 GUI。
- 主要证据入口：
  - COLMAP：`CHANGELOG.rst`、`src/colmap/feature`、
    `src/colmap/controllers/global_pipeline.*`、`hierarchical_pipeline.*`、
    `src/colmap/sfm/global_mapper.*`。
  - Colite：`include/colite/core/feature.h`、`src/colite/features`、
    `src/colite/matching`、`src/colite/sfm`、`src/colite/mvs` 和
    `src/colite/exe/command.cpp`。

## 3. 能力差异矩阵

| 能力 | COLMAP 4.0 | Colite 当前状态 | 判断 |
|---|---|---|---|
| SIFT CPU/GPU、RootSIFT | 支持 | CPU/CUDA，多套语义和一致性测试 | 已覆盖 |
| SIFT BF、ratio、cross-check、guided matching | 支持 | CPU/FLANN/CUDA，含极线走廊重搜索 | 已覆盖；实现路径不同 |
| exhaustive/sequential/spatial/vocab-tree pairing | 支持 | 四类均有，另有本地词树 builder | 基本覆盖 |
| transitive matcher | 独立命令 | 已实现受控 pair 扩展、真实匹配与几何验证 | 已覆盖，P3 |
| 类型化描述子 | SIFT/ALIKED 类型随存储和流水线传播，不匹配类型会失败 | 类型、标量、距离与归一化随 workspace v2 持久化 | 已覆盖，P0 |
| ALIKED N16Rot/N32 | ONNX，RGB 输入，float 描述子 | 已实现并通过官方模型 golden | 已覆盖，P1 |
| LightGlue | SIFT 与 ALIKED 的 ONNX matcher | 两种官方模型均已通过真实 inference golden | 已覆盖，P1 |
| ALIKED 可扩展检索 | 预训练 vocabulary tree | 当前词树围绕 SIFT/通用 float 接口设计 | 部分，P1/P3 |
| EXIF 方向自动旋转 | 提取/匹配时自动转正并回写特征坐标 | Orientation 1–8 在统一读取层处理 | 已覆盖，P1 |
| 增量 SfM、多模型、续建/合并 | 支持 | 支持 | 基本覆盖 |
| structure-less 注册回退 | 无 3-view overlap 时用 generalized relative pose 注册 | 主要依赖 2D-3D PnP | 缺失，P3 |
| position prior | 对齐、优化及 absolute pose refinement 可选使用 | GPS→ENU、Sim(3) 对齐、最终 BA；PnP refinement 未使用位置 prior | 部分，P3 |
| GLOMAP global mapper | view graph calibration→旋转平均→track 建立→全局定位→分阶段 BA→重三角化 | 首条全局流水线已接通，calibration/分阶段优化仍需增强 | 部分，P2 |
| view graph calibration | 可独立运行，由 F/E 估计焦距并重估相对位姿 | 无 | 缺失，P2 |
| rotation averaging/global positioning | 独立 estimator/CLI，并用于 global mapper | 已实现并接入 global mapper | 核心覆盖，P2 |
| hierarchical mapper | 重叠聚类、并行局部增量重建、层次合并 | 已实现重叠聚类、局部重建、Sim(3) 重试合并和最终 BA；局部任务暂顺序执行 | 核心覆盖，P3 |
| model clustering | 按场景连通性切分大型模型 | 已实现 verified view-graph 目标规模/重叠分区 | 核心覆盖，P3 |
| BA 点开关 | 可控制是否优化 3D 点 | local/global BA 已有 `refine_points` | 已覆盖；补 CLI 一致性检查 |
| BA 性能改进 | 单 pose 参数块、SIMPLE_RADIAL 分析 Jacobian、失败时 CPU fallback | 常用相机有分析 Jacobian；没有同等通用 solver 抽象/fallback | 部分，P2/P4 |
| LO-RANSAC 内部并行 | 支持 | image pair 级并行，单 pair RANSAC 循环串行 | 部分，P4 |
| QEM mesh simplification | 新增 | 已实现 | 已覆盖 |
| mesh texture mapping | 新增 | 已实现纹理图集、可见性和颜色校正 | 已覆盖，继续做质量 A/B |
| PatchMatch 并行输入 | 改进 | 有 workspace/cache，但未证明与 4.0 等价的并行读入 | 待基准，非功能阻塞 |
| JPEG 去畸变质量参数 | 支持 | `cv::imwrite`，无公开 JPEG quality 参数 | 小缺口，P4 |
| alpha 图像载入 | 支持 | 全链路主要使用 `IMREAD_COLOR`，alpha 被丢弃 | 小缺口，P4 |
| SQLite database 与维护/导入命令 | 原生数据库，creator/cleaner/merger/importer 等 | 自定义文件 workspace，无 SQLite 层 | 互操作缺口，P4 |
| Python API/GUI | pycolmap 与 Qt GUI | 仅 C++ CLI | 产品范围差异，默认不跟进 |

## 4. 开发目标与顺序

### P0：特征类型与持久化协议（必须先做）

目标：让 SIFT、ALIKED 和未来特征能够安全共存，旧 workspace 能明确迁移或报错。

1. 增加 `FeatureExtractorType`、`FeatureMatcherType` 和描述子标量/距离语义；至少表达
   `SIFT_U8_OR_QUANTIZED`、`ALIKED_N16ROT_F32`、`ALIKED_N32_F32`。
2. 将 `FeatureSet` 从单一 float 矩阵升级为带类型、行数、维度和标量类型的结构；
   保持 SIFT CUDA/FLANN 热路径无需反复转换。
3. 为 workspace feature 文件增加 magic、schema version、extractor type、dtype、shape、
   payload size 和校验；旧格式只允许确定性识别，禁止猜测类型。
4. matcher 工厂显式校验 extractor/matcher 组合；错误组合在计算前失败。
5. 配置 hash 纳入特征类型、模型身份、预处理方向、描述子格式和 matcher 类型。

验收：现有 SIFT 测试和基准不回退；新旧格式迁移/拒绝行为有测试；不同特征类型无法误配；
损坏或截断的 feature 文件明确报错。

### P1：ALIKED + LightGlue 深度学习前端

目标：达到 COLMAP 4.0 的单相机深度学习前端组合，不把 ONNX 变成强制依赖。

1. 增加可选 `COLITE_ENABLE_ONNX`，支持 CPU execution provider；CUDA provider 作为第二步，
   与现有 CUDA Toolkit/运行时版本做启动前检查。
2. 实现统一 ONNX session 封装：UTF-8 路径、输入输出名字/shape/dtype 校验、线程数控制、
   provider 诊断和异常转换。
3. 实现 ALIKED N16Rot/N32：RGB `[0,1]`、尺寸补齐到 32、padding 外关键点过滤、
   像素中心 `+0.5` 对齐、score top-K 和 float 描述子持久化。
4. 实现 ALIKED cosine BF，并接入现有几何验证、pair generator 和缓存。
5. 实现 LightGlue ONNX，支持 `SIFT_LIGHTGLUE`、`ALIKED_LIGHTGLUE`；明确 keypoint
   归一化、scale/orientation 和图像尺寸契约。
6. 处理 EXIF Orientation：统一图像转正，在写入 workspace 前把关键点映射回约定坐标系；
   同一变换必须供 ALIKED、SIFT 和 LightGlue 复用。
7. 模型文件只通过显式路径或带 SHA-256 的资源清单获取；离线环境给出可操作错误，测试不依赖网络。

验收：

- 用固定小图集分别完成四组端到端 SfM：SIFT+BF、SIFT+LightGlue、
  ALIKED+BF、ALIKED+LightGlue。
- CPU 推理结果可重复；CPU/CUDA 的关键点、匹配数和 SfM 注册率设容差门槛。
- 与 COLMAP 4.0 在同模型、同图像、同阈值下比较关键点数、共同关键点坐标、pair matches、
  verified inliers、注册图像数和重投影误差，不只比较最终点数。
- 90/180/270 度 EXIF 方向样例与物理旋转后的样例产生等价几何。

### P2：GLOMAP 风格全局 SfM

目标：新增 `global_mapper`，保留现有 incremental mapper，二者共享核心数据和 BA。

按依赖拆分：

1. 扩充 two-view geometry，可靠持久化相对位姿、标定/退化状态与必要置信信息。
2. `view_graph_calibrator`：从基本矩阵和相机分组估计焦距、分类边并重估相对位姿。
3. `rotation_averager`：最大生成树初始化、鲁棒旋转平均、异常边过滤；先做无 gravity 版本。
4. track establishment：一图一观测冲突处理、最小 track 长度和 view graph 清理。
5. global positioning：以 point-to-camera/track 约束联合估计相机与点位置。
6. 两阶段 BA：先固定旋转优化位置/结构，再联合优化；复用现有过滤、重三角化和颜色提取。
7. 在 `recon_sfm/recon_auto` 增加 `mapper=incremental|global`，保持输出格式一致。

验收：在 House 之外至少加入一个宽基线无序集和一个数百图数据集；对 COLMAP 4.0
比较注册率、最大连通模型、相对旋转误差、相机中心对齐误差、点/观测数、重投影误差、wall/RSS，
并设置固定随机种子的确定性测试。全局管线必须能单独定位到 calibration、rotation、position、BA
各阶段的失败，而不是只有端到端失败。

### P3：大规模与弱连接鲁棒性

1. structure-less registration fallback：PnP 因 3-view overlap 不足时，通过已注册邻图的
   generalized relative pose 给出候选位姿，再做几何验证和局部 BA。
2. hierarchical mapper：verified graph 重叠聚类、并行局部增量 SfM、公共 track/相机的
   Sim(3) 约束图、层次合并和最终 BA。
3. model clusterer：按目标规模和重叠率切分，而不是仅按连通分量断开。
4. transitive matcher：只生成受控的新 pair，再走真实描述子匹配与几何验证，避免把传递关系
   直接当作匹配证据。
5. 将位置 prior 加入 absolute pose refinement，并区分初始化、软约束、对齐三种用途。

验收：弱三视图重叠集能比纯 PnP 注册更多图且不显著增加错误位姿；层次重建在数百/数千图上
降低峰值内存或 wall time，并与非层次增量结果保持规定的注册率和几何误差。

### P4：互操作与次要工程项

- 评估 SQLite 兼容层。若目标是直接消费/产生 COLMAP database，这是高价值但独立的大项目；
  若仅追求算法能力，可先提供离线 import/export 工具，不把 SQLite 引入核心热路径。
- 增加已知 reconstruction 的独立 guided geometric verifier、feature/matches importer、
  database/model merger 等维护工具，按真实用户工作流选择，而非追求命令数量相等。
- 图像 I/O 增加 EXIF 方向、alpha 策略、JPEG quality 和更明确的格式诊断；是否从 OpenCV
  切换 OpenImageIO 应由吞吐/格式基准决定。
- 单 pair LO-RANSAC 并行、通用 BA solver summary、GPU BA 失败回退和共享库构建属于性能与
  发布增强，不阻塞 P1/P2 的功能闭环。
- Python API 和 GUI 默认不进入当前路线图，除非 Colite 的产品定位从 CLI/库转向交互式工具。

## 5. 建议的首批 issue/提交边界

1. `feature-types`：类型枚举、兼容矩阵和纯内存单测。
2. `feature-workspace-v2`：版本化读写、旧格式检测、损坏文件测试。
3. `onnx-runtime`：可选构建、session/provider 诊断、无网络模型路径策略。
4. `aliked-extractor`：N16Rot/N32 CPU，固定图像 golden test。
5. `aliked-bruteforce`：cosine BF + 几何验证 + exhaustive/sequential。
6. `lightglue`：先 ALIKED，再 SIFT；加入 pair-level COLMAP A/B。
7. `exif-orientation`：独立于 extractor 的预处理/坐标变换模块。
8. `deep-frontend-e2e`：四种组合的可重复端到端基准和文档。
9. `view-graph-calibration`：P2 的第一个独立算法里程碑。

## 6. 暂不做的事项

- rig、frame、非平凡多传感器 BA、同 frame pair 跳过和 rig geometric verification。
- FISHEYE、SIMPLE_DIVISION、DIVISION 及其他特殊相机模型差异。
- 为了表面兼容而复制 COLMAP GUI、所有 Python binding 或全部维护命令。
- 在 P0 数据契约完成前，把 ALIKED float 描述子强制量化成当前 SIFT 格式。
- 以“最终点数更多”作为唯一验收；前端、view graph、位姿和轨迹指标必须逐层对齐。
