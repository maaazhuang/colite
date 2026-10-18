# Colite / COLMAP 4.0 差异与数据互通审查（2026-10-13）

> 2026-10-18 范围调整：用户暂缓数据库，先做参数行为对齐与特征覆盖/GPU 匹配优化。本文是 10-13 的审查快照；当前实施结果见 [10-18 记录](sfm_alignment_20261018.md)。

## 结论与范围

当前 Colite 已有完整的单相机 SfM 主干，并实现 ALIKED/LightGlue、global/hierarchical mapping 的可运行链路；差距不能再概括为“缺少这些模块”。主要问题是数据交换完整性、参数/默认值语义，以及相同负载下的覆盖与吞吐。

**三文件稀疏模型 TXT/BIN 可双向交换；特征只能通过单向工具导入；完整 SfM 工程不能自由双向接续。** 稀疏模型的二维观测不包含描述子和完整匹配数据库，因此模型能打开不等于能从任意阶段切换程序。

本轮依据工作区源代码、仓库内 `references/colmap-4.0.0`、现有实际基准及本轮导入/互读验证。在线 COLMAP 文档当前显示 4.3.0.dev0；4.0 的类型、模型 ID 和 schema 判断以固定版本源码为准，没有把后续 LOMA 等特征作为 4.0 缺口。

用户约定：未正式发布，不保留旧设计兼容层；同流程参数名称与行为向 COLMAP 4.0 对齐，模块/程序名称无需一致。此前路线图的 SQLite/新增相机/rig 排除范围不再决定本轮互通目标。支持读写元数据与支持 rig 约束求解需分别验收。

## 当前差异

| 范围 | 当前 Colite | 与 COLMAP 4.0 的实质差异 / 影响 |
|---|---|---|
| SfM 模型 | TXT/BIN 三文件，12 种相机，保留 ID、位姿、观测、RGB、ERROR、track | 不保留 rigs/frames；COLMAP 源码还登记 ID 12–15 的 SIMPLE_DIVISION、DIVISION、SIMPLE_FISHEYE、FISHEYE，遇到这些模型不能直接接续 |
| 工程与特征存储 | cameras.txt / images.tsv / feature v2 / matches TSV | 没有原生 `database_path` SQLite 适配层，也没有 workspace → COLMAP database 出口 |
| SIFT | CPU VLFeat/CvSift、CUDA；内部 float32 | COLMAP 持久化 uint8×128。运行时 float 缓存可以保留，但交换数据需明确量化、归一化、形状和坐标语义 |
| ALIKED / LightGlue | 类型化特征、N16Rot/N32、两类 LightGlue、可选 ONNX | 模型 golden 不等于跨程序缓存互通；本轮仅测试 ALIKED 类型化数据库样本的导入和 CPU BF 消费，没有执行新的 ONNX 精度/吞吐比较 |
| 原始 / verified 匹配 | CPU/CUDA BF、几何验证、guided matching | 当前 TSV 未保存 F/E/H 和 optional 缺失值；导入脚本不搬运 raw matches，并对部分 config 降级或跳过 |
| 图像对生成 | exhaustive/sequential/spatial/vocab/transitive | 有功能链路；本地词树文件不兼容 COLMAP，spatial 最小邻居补足等规则也未完全对齐 |
| 增量构建 | 多模型、局部 BA、完成/合并/重三角化、最终 BA/过滤 | 初始化/注册阈值和尝试调度不同，缺少成熟的增长驱动全局 BA；多模型/合并后收敛仍需验证 |
| structure-less | 多邻居旋转一致性与尺度不变的平移射线交会，候选几何检查 | 不是 COLMAP 的 generalized relative pose 求解；默认开关也不同 |
| 全局构建 | 边筛选/旋转环检查、旋转平均、稀疏 QR 相机中心、固定旋转 BA、重三角化、最终 BA | `ViewGraphCalibrationOptions` 实际是边筛选，没有 COLMAP 的从 F/E 估计焦距并重估相对位姿；定位和分阶段鲁棒优化也不是完整 GLOMAP |
| 层次构建 | 重叠分区、有界并行局部重建、Sim(3) 合并/重试、最终 BA | 大规模模型图优化、未合并分量行为与实际质量/吞吐证据不足；不能按模块存在认定等价 |
| BA | Ceres、常用模型解析 Jacobian、小问题 DENSE_SCHUR、位置先验 | q/t 分块与 COLMAP 单 pose 参数块不同；损失、gauge、收敛控制、GPU solver/CPU fallback 未完全对齐 |
| 位置先验 | EXIF/GPS→ENU、对齐、最终 BA | SQLite pose_priors 的 sensor 关联、坐标系、协方差和重力不能完整搬运，PnP refinement 使用也不同 |
| MVS / 网格 | 去畸变、CPU/CUDA PatchMatch、fusion、Poisson/Delaunay、简化、纹理 | 主要流程已实现；本轮未对深度/法线/一致性缓存互读或同输入 MVS 性能做验收，SfM 互读结论不能推广过去 |
| 外围 | C++ CLI | pycolmap / GUI 是产品范围差异。优先完成数据契约，Python 用户可先通过 COLMAP 数据库和模型 API 使用数据 |

## 数据互通的具体边界

### 1. 稀疏模型：成立的是三文件契约

`src/colite/io/sfm_workspace.cpp` 已处理 little-endian、WXYZ、world-to-camera、非连续 ID、二维观测顺序与反向 track。纯转换保留 ERROR；优化后重新计算误差。

本轮重新用官方 COLMAP 可执行文件验证：12 种相机的构造样本，以及 House 的 36 张图像/40,034 个点，经 TXT 和 BIN 的 COLITE → COLMAP → COLITE 往返均通过。验证入口是 `scripts/check_colmap_model_interop.py`，结果在 `output/sfm_interop_audit_20261013/model_interop/report.json`。

限制：reader 只读 cameras/images/points3D，writer 会移除 rigs/frames。普通三文件模型可由 COLMAP 自动补齐 trivial rigs/frames，但非平凡 rig 的传感器外参、frame 分组与约束会丢失。不能称为 COLMAP 4.0 完整五文件的无损转换。

接续映射还依赖与模型对应的原始 keypoint 顺序。Colite 增量续建已按图像名重映射 ID，并在 `reconstruction_manager.cpp::ValidateAndCompleteSeedFeatures` 校验数量和逐行 xy；不是重新提取相似特征就能续建。相机共享关系也必须一致。

### 2. 描述子：4.0 的 cols 是字节数

固定源码 `feature/types.h` / `types.cc` / `scene/database_sqlite.cc` 定义：

| 特征 | COLMAP type | 数据库 rows × cols | Colite 表示 | 必须执行的转换 |
|---|---:|---|---|---|
| SIFT | 0 | N × 128，uint8 | type=1，N × 128 float32 | 导入为 uint8/512；返回需明确量化和取整。原始整数/512 可精确恢复，独立生成的 float 首次量化会有损 |
| ALIKED_N16ROT | 1 | N × 512 字节，实际 N × 128 float32 | type=2，N × 128 float32 | 按 float32 解释字节；不能数值转换每个 uint8 字节，也不能除以 512 |
| ALIKED_N32 | 2 | 同上 | type=3，N × 128 float32 | 同上，保留类型与描述子原值 |

数据库没有记录 SIFT 的 L1_ROOT/L2 提取配置。本轮工具默认声明 COLMAP 默认的 L1_ROOT，源任务若采用 L2，必须传 `--SiftExtraction.normalization L2`。真正共享工程应保存提取器版本、归一化、预处理、模型校验和和配置来源，不能仅靠维度猜测。

本轮已修正原单向脚本：读取 type 和字节宽度，支持 SIFT 与两类 ALIKED，写入带校验和的 feature v2，未知类型/截断数据明确失败。保留特征位置和行顺序，不重新排序或归一化。它仍不是双向数据库工具。

### 3. 关键点：xy、方向和完整仿射形状都需要契约

COLMAP 的像素中心为 (0.5,0.5)，2/4/6 列关键点分别表示位置、相似形状、完整仿射形状；方向角采用弧度。Colite 当前字段只有 x/y/scale/orientation/response，方向采用度。

本轮导入器修正了弧度→度，以及六列形状的平均轴尺度/方向转换；不改变 xy。完整 a11/a12/a21/a22 仍无法放入现有结构，shear/各向异性会丢失，response 在 COLMAP 中也没有同等字段。

此外，CvSift 与 cuda_opencv 包装仍直接采用低层坐标和 `keypoint.size`；VLFeat 路径加 0.5，并使用 sigma。不能在出口只统一乘一个常数：需要分别核对像素中心、直径/尺度、方向符号及描述子约定，然后在包装边界统一。该项尚未重构或验收，尤其影响 SIFT LightGlue 与词树几何验证。

### 4. 匹配：不能只回写内点索引

`matches` 与 `two_view_geometries` 是不同阶段，后者才是 COLMAP mapper 的输入。交换需要保留 canonical pair_id、uint32 的零基索引、config、F/E/H、qvec/tvec 以及 NULL/存在状态。

当前 `WriteWorkspaceVerifiedGeometries` 只保存 type、角度、pose、内点；结构里虽有 E/F/H，持久化没有写入。COLMAP 4.0 矩阵/相对位姿是 optional；Colite 的零矩阵/单位姿态不能代表可用结果。相对位姿缺失时，应通过已保存矩阵和观测重新估计，或者明确报告不可接续，不能伪造单位姿态。导入脚本的 config=6 降级为 planar，config=9 rig geometry 未接入；空 pair 和 watermark/退化记录也不能逐项往返。

这会阻断“COLMAP 提特征 → Colite 匹配 → COLMAP mapper”，也阻断复用数据库几何做 COLMAP 全局焦距 calibration。重算全部匹配可绕过部分缺口，但不符合用户自由切换/复用耗时阶段的目标。

## 参数与默认行为仍需对齐

首轮已删除同义旧名，接入标准选项和 `Colite.*` 扩展；不是完整参数替代。以下直接影响同参数实验与用户迁移：

| 选项 / 行为 | Colite | COLMAP 4.0 |
|---|---:|---:|
| Mapper.init_min_num_inliers | 15 | 100 |
| Mapper.init_min_tri_angle | 0.5° | 16° |
| Mapper.init_num_trials | 3 | 200 |
| Mapper.abs_pose_max_error / min_num_inliers | 4 px / 12 | 12 px / 30 |
| Mapper.filter_min_tri_angle | 0.5° | 1.5° |
| Mapper.min_model_size | 2 | 10 |
| Mapper.structure_less_registration_fallback | false | true |
| GPU / 线程 / 图像尺寸控制 | 多个特有控制，标准选项未全接入 | FeatureExtraction.* / FeatureMatching.* |
| BA 增长与收敛控制 | 局部与最终优化为主，暴露选项有限 | 增长触发、迭代/收敛调度及 BundleAdjustmentCeres.* |

目标是相同名称对应相同意义、单位、默认值与生效阶段。硬件开关应贯穿 SIFT/ALIKED/匹配实现；特有 CUDA/ONNX 内存、实现选择仍可用 `Colite.*`。只有 SQLite 真正接入后再接受标准 `database_path`，不能把 workspace 目录改名冒充数据库。

## 效率与质量：已有证据如何指导优化

以下是 **2026-10-01 的诊断结果**，没有在本轮重新进行性能计时；均为小数据集单次计时，COLMAP 是参考而非独立真值。

| 控制实验 | 已有差距 | 优化含义 |
|---|---|---|
| 相同描述子的 GPU 完整匹配 | Colite 慢约 2.8–3.4 倍；原始 BF 集合接近逐条相同 | 优先分解 BF、两视图估计、guided search、I/O 和调度；不能用独立前端更少的描述子掩盖吞吐差距 |
| GPU 独立前端 House | 同为 36 张注册，Colite 19,426 点 / COLMAP 40,034；verified pair 250 / 494 | 多主方向预算、尺度/空间覆盖与两视图验证是主要改进方向；低自重投影误差不证明更高真实精度 |
| 相同 verified 输入 mapper | 注册相同，点数差 <1.3%，位姿接近；Colite 慢约 18–37% | 核心几何链路已有可用水平，重点改善局部/全局 BA、track 完成/合并和模型调度 |
| 相同扰动初值的 BA | 收敛误差接近，时间约高 9–10% | 先统一 loss/gauge/收敛条件并重复计时，再考虑 pose 参数块、Jacobian 和 solver 优化 |
| CPU Building 共用描述子 | 原始 BF 逐条一致，完整匹配约快 16.3 倍 | CPU/GPU 结论必须分开；不外推至更多场景或所有 COLMAP CPU 配置 |

原始数据与限制见 [2026-10-01 SfM 实测](sfm_benchmark_20261001.md)。后续先用共同数据库隔离输入，正式结论至少三次中位数，并用 NAVI multiview 位姿标注、ETH3D 可见扫描几何检验准确度/覆盖。见 [真实参考数据约定](sfm_reference_datasets.md)。

## 建议实施顺序与验收

### P0：共享 COLMAP 4.0 数据库

在 I/O 边界建立 SQLite 读写适配层；标准数据库作为交换的权威数据，内部缓存作为可重建加速层。算法模块不需要依赖 COLMAP 库，也不必沿用其类/命令结构。已有 workspace 可作为调试缓存；不为未发布旧设计维护另一套永久公开契约。

第一阶段读写 cameras/images/keypoints/descriptors/matches/two_view_geometries；保存原始 ID、相机共享、相对路径和 keypoint 顺序，新增图像分配未占用 ID。事务提交与缓存失效需要根据数据身份，不能只根据路径。明确 schema/user_version 与 byte_cols；只支持已实现的类型，禁止按数据长度猜类型。

同时扩充关键点形状与 optional 两视图数据。对不能优化的 rig/sensor/pose prior 字段，可先原样往返或明确拒绝相关求解；透传元数据不是支持其几何约束。不能以“暂不做 rig 优化”为理由静默删除工程信息。

必须覆盖的交叉流程：

1. COLMAP 提取 → Colite 匹配 → COLMAP mapper。
2. Colite 提取 → COLMAP 匹配 → Colite mapper。
3. 两端读取同一 verified 数据库，各自 mapper；中途从另一端的模型接续。
4. SIFT、ALIKED N16Rot/N32，零特征/零匹配、非连续 ID、子目录/空格名称、未知类型/损坏 BLOB、交换 pair 顺序、NULL 几何；模型和数据库的 POINT2D_IDX 一一对应。

### P0：模型覆盖与无损契约

补齐四个相机模型的投影/反投影/导数与 TXT/BIN 读写；接入 rigs/frames I/O，或对不支持的非平凡模型明确拒绝操作。往返逐字段验证之外，要实际执行跨程序 BA、三角化和接续 mapper；PLY/NVM 不能替代这些数据。

### P1：统一参数行为并优化前端

补 GPU/线程/尺寸/SIFT 配置和 mapper 默认/初始化调度，去掉绕过标准控制的冗余开关。对齐多主方向预算与输出形状，再按相同 feature 数量定位 GPU matching 热点。保持原始 matches 与 verified matches 分阶段可复用。

### P1：稀疏构建与 BA

先解决既有周期 BA 试验的合并模型初始 residual/Jacobian 失败，再恢复增长与收敛调度。用共同输入检查局部 BA 共享相机保护、三角化/过滤误差尾部、注册失败和回退。BA 先对齐优化目标，再评估单 pose 参数块和求解器策略。

### P2：全局、层次与大规模

在完整 F/E/H 可复用的基础上补全局焦距 calibration、相对位姿重估、track 驱动的鲁棒定位和模型图优化。最后扩展大规模 ONNX/GPU 调度与 MVS 缓存互读；不能用这些模块替代先完成基础数据互通。

## 本轮已落地与验证

- 单向数据库工具修正 ALIKED type/float32 读取、SIFT 显式来源归一化、方向单位与六列形状的尺度/方向转换；使用 feature v2 校验和。未知类型/坏长度在输出前拒绝，已有 workspace 不覆盖。
- 新增 `ColmapDatabaseImportTest`：六个 Python 用例，SIFT/N16Rot/N32 分别经过实际原生 feature reader 与 BF matcher；覆盖非连续 ID、描述子原值、方向、损坏和已有数据保护。
- 构建成功，针对命令和导入的 25 项 CTest 通过；最后增加来源 L2 用例后，单独重跑导入测试通过。本轮没有重新跑全部 411 项或性能基准。
- 实际导入既有 COLMAP Building 数据库并执行 Colite mapper，注册 18/18，18,937 点，0 未注册，单模型。日志和输出在 `output/sfm_interop_audit_20261013/building_import/` 与同级 `building_mapper.log`。
- House 三文件模型与 12 种相机的双向 TXT/BIN 验证见同级 `model_interop/report.json`。这不验证 rig 约束求解或完整数据库往返。

完整 SQLite 双向接口、F/E/H/NULL 存储、仿射原值、rig/frame、四个额外模型与参数/算法对齐尚未实现，不应将本次单向脚本修正宣传为已完成全部互通。

来源：[COLMAP 数据库格式](https://colmap.github.io/database.html)、[COLMAP 模型格式](https://colmap.github.io/format.html)，以及固定版本 `references/colmap-4.0.0/CHANGELOG.rst`、`feature/types.*`、`scene/database_sqlite.cc`、`sensor/models.h`、`controllers/incremental_pipeline.h`、`sfm/incremental_mapper.h`。在线文档作为格式说明，4.0 能力清单以本地固定版本为准。
