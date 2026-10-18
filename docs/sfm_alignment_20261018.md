# 2026-10-18 参数行为对齐与前端优化

本轮按用户顺序推进：暂缓数据库，保留文件 workspace；先对齐已有 SfM 流程参数，再改善特征覆盖与 GPU 匹配。不增加旧名称兼容层。

## 参数与实际行为

- `FeatureExtraction.use_gpu / gpu_index / max_image_size` 接入提取流程。非正尺寸按 COLMAP 4.0 自动选择 SIFT 3200、ALIKED 1600；提取缩放后关键点回映原图坐标，Camera 保留原图尺寸。默认 SIFT 后端统一为 CUDA VLFeat / CPU VLFeat。
- `FeatureMatching.use_gpu / gpu_index / num_threads` 控制 SIFT 的真实 CPU/GPU 路径和匹配/几何并行预算；LightGlue 设备接到 ONNX provider，CPU 线程用于外层任务预算。GPU index 当前接受单整数，-1 自动选 0；CSV 多设备调度尚不支持。ALIKED BF 目前只支持 CPU，显式 GPU 请求会报错。
- 接入 `SiftExtraction.first_octave / num_octaves / octave_resolution / peak_threshold / edge_threshold / max_num_orientations / normalization`。默认分别为 -1 / 4 / 3 / 0.02÷3 / 10 / 2 / L1_ROOT。改变 octave_resolution 不隐式改变独立 peak_threshold 的默认值。L2 真正归一化为单位二范数并持久化正确元数据。
- first_octave 当前实现 -1、0，方向数为 1..4。显式 COLMAP SIFT 几何控制要求 VLFeat 模式；CV 模式保留为特有后端选择，不吞掉不生效的选项。
- 删除 `Colite.matcher_backend`、`Colite.aliked_max_image_size`、`Colite.learned_device / learned_gpu_index / learned_matcher_num_threads` 的公开入口，改用标准控制；保留特有 arena/I/O binding 和后端算法选择。
- guided matching 默认继承 `SiftMatching.max_ratio`，不再偷偷强制 cross_check=true；显式 `Colite.guided_match_ratio` 仍可覆盖候选比率。
- mapper / recon_sfm 共用初始化 100 内点 / 16° / 200 次候选预算，abs pose 12 px / 30 内点 / 0.25 比率，过滤 1.5°，最小模型 10 张（受总输入数限制），structure-less 回退默认 true。初始化失败后两轮依次减半内点门槛、角度门槛。默认 PnP 重试 multiplier=1，保持用户误差阈值。
- 提取与稀疏缓存身份升级，包含尺寸、GPU、SIFT 控制和算法版本，避免沿用旧结果。

这些是已接入控制的对齐，不是完整 COLMAP CLI 替代。init_num_trials 的候选/重建尝试组织、structure-less 求解器、完整周期 BA/收敛调度、所有 SIFT 仿射/DSP/upright/遮罩选项、random_seed、多 GPU、ALIKED GPU BF 等仍有差距；不通过近似别名宣称支持。

## 特征预算

原 CUDA VLFeat 路径按 response 将最终方向行硬截断到 8192，可能同时丢失粗尺度基础点和第二方向。

固定 COLMAP 4.0 源码 `feature/sift.cc` 使用 SiftGPU `-tc2`。`thirdparty/SiftGPU/SiftPyramid.cpp` 显示：展开多方向后还会再次按尺度层限额。本轮 CUDA 路径按该规则保留完整层，不拆散同一基础点的方向；边界层可使最终行数超过目标。CPU VLFeat 仍按基础点完整层筛选，两端输出数不必相等。

曾尝试直接使用 CPU 式基础点层预算：House 得到 504,870 个方向特征、53,806 点，但匹配/建图成本明显上升。该策略未采用；实验位于 `output/param_frontend_20261018/coverage_house/`，不能当作最终实现结果。

## GPU 匹配

1. 原始匹配只有两端行数都是 4 的倍数时才走整数 GEMM；其余退回逐描述子搜索。现在 signed descriptor 行补齐，GEMM 使用补齐尺寸，归约只遍历真实行/列，不把 padding 当作候选。
2. 列归约按 256 行分块，再合并 top-2，避免单线程串行扫描整列。保留并列分数、最小索引与第二最佳分数语义。
3. guided matching 使用同一整数 GEMM，再按原 Sampson gate 筛选；不再在大量几何候选上逐项计算 128 维 dot。
4. CUDA worker 激活指定设备，cuBLAS workspace 按设备保留。补充分段墙钟：读取、准备、原始匹配、验证、引导、图一致性、写出。

同一 House COLMAP 来源特征缓存、630 对，三次全新 workspace 重复：优化前中位数 **27.204 秒**，最终版本 **11.602 秒**，缩短 **57.35%**，约为原吞吐的 2.34 倍。六次运行的 raw 与 verified 两个 TSV 的 SHA-256 均一致。该缓存已经存在，本轮不调用数据库导入。基线在资源控制接通之后、特征预算/GPU 实现修改之前冻结，二进制保存在 `colite_profile_before`。

| 分段墙钟中位数，秒 | 优化前 | 最终 |
|---|---:|---:|
| 原始描述子匹配 | 15.925 | 2.309 |
| 两视图验证 | 2.397 | 2.383 |
| 引导匹配 | 4.329 | 2.634 |
| 图一致性 | 未单列 | 1.162 |
| 工作区写出 | 未单列 | 1.828 |

分段中位数不能直接相加等同于总中位数；总时间还包括输入读取、准备、启动和维护。下一轮需要同时关注 guided 的矩阵带宽、图一致性和 TSV 开销，不应继续把全部差距归因于原始 BF。

## 最终覆盖与几何回归

两套前端均显式使用 3200 尺寸、8192 预算；优化前特征也由同一当前 matcher / mapper 消费。House 最终验证复用了不可变基线 features/matches，重新运行当前 mapper，得到相同 24,308 点。COLMAP 列是既有参考重建，未在本轮重新运行。

| 数据集 / 指标 | 旧提取策略 + 当前后端 | 最终策略 + 当前后端 | COLMAP 4.0 既有参考 |
|---|---:|---:|---:|
| Building 特征行 | 147,456 | 199,089 | 195,446 |
| Building 有效 pair | 120 | 135 | 136 |
| Building 注册 / 点 | 18 / 15,394 | 18 / 21,251 | 18 / 18,765 |
| Building 观测均值 / P95，px | 0.389 / 1.109 | 0.583 / 1.656 | 0.609 / 1.789 |
| House 特征行 | 294,912 | 370,644 | 368,100 |
| House 有效 pair | 302 | 493 | 494 |
| House 注册 / 点 | 36 / 24,308 | 36 / 39,358 | 36 / 40,034 |
| House 观测均值 / P95，px | 1.038 / 2.763 | 1.320 / 3.084 | 1.247 / 2.843 |

最终两组均无相机后方观测、无超过 4 px 的观测。更多覆盖增加了误差较大的观测，不能只比较旧策略更低的自重投影误差。House 点数距参考约 -1.69%，但均值与尾部仍略高；当前只能判断覆盖接近、误差尾部仍需优化。

相机按图像名匹配并做 Sim(3) 对齐：Building / House 平均旋转差分别为 0.179° / 0.034°，中心均值差占最大基线 0.0345% / 0.0405%。这些是与参考的一致性，不是独立真值误差。独立重建的 point3D ID 不构成几何对应，不使用 common_points_by_id 判断准确度。

| 最终版本单次阶段墙钟，秒 | 提取 | 匹配 | mapper |
|---|---:|---:|---:|
| Building | 3.75 | 4.40 | 20.96 |
| House | 13.00 | 11.69 | 57.32 |

覆盖增加也增加后端工作量：使用同一当前后端的旧策略 Building / House 为 23.69 / 53.25 秒，最终为 29.11 / 82.01 秒。本轮不是“无代价增加点数”的优化，不能说全流程全面更快。提取和 mapper 此处均为单次诊断计时；与历史 COLMAP 时间的差异仅供后续定位，不是同期速度比赛。

House 的 Ceres 过程中出现一次 `Failed to compute a finite step` 线性求解警告，最终流程成功、模型和观测有限且全部注册。这不是先前周期 BA 的初始 residual/Jacobian 致命失败；其数值原因和误差尾部需要后续 BA 专项定位，本轮不归因为某一个模块。

## 实验约定与结果文件

- WSL Release / RTX 4070 SUPER；CPU 与 GPU 性能运行串行进行，不与构建或测试争用资源。
- 相同输入匹配入口：`scripts/benchmark_workspace_matching.py`。每次只链接/复制输入元数据和 feature BIN，使用独立 cache/output，保存完整 argv、退出码、SHA-256、分段统计、匹配文件摘要与中位数。
- 覆盖入口：`scripts/benchmark_frontend_coverage.py`。两类提取结果使用相同 3200 尺寸、8192 预算、8 提取线程、共享 SIMPLE_RADIAL、同一当前 matcher/mapper；用独立统计重算每条观测误差。可复用不可变基线 features/matches，再用相同当前 mapper 重新构建。
- 新测试覆盖非 4 倍数行、跨归约块并列、完整方向保留、缩放坐标回映、L2 元数据、标准配置及失效缓存。全量 CTest 415 项：409 通过、6 按环境跳过；最终诊断字段/CLI 边界校验增补后，相关 69 项及 44 项回归再次通过。ONNX golden/provider 未在当前构建启用。
- 更低自重投影误差或更多点都不证明真值精度更好；本轮仍是覆盖、效率与回归验证。真实位姿/扫描基准尚未新增。

权威结果目录为 `output/param_frontend_20261018/`：`shared_house_before/` / `shared_house_final/` 是相同负载匹配比较；`coverage_house_final/` / `coverage_building_final/` 是最终覆盖验证；`pose_*_after/` 是位姿对照；`final_summary.json` 汇总数值与环境。冻结基准二进制 `colite_sift_validated` 的 SHA-256 与匹配、House、Building 三项记录均一致。其后仅修正组合流程中 ALIKED 默认执行设备继承匹配设备的问题，使提取与匹配设备独立；基准使用的独立 SIFT 阶段未改。追加定向回归 40 项：36 通过、4 因 ONNX 环境跳过，当前构建未验证实际 ONNX 推理。当前二进制与冻结基准的摘要分别记录。硬件为 RTX 4070 SUPER / 20 逻辑 CPU，当前驱动 617.42。中间目录仅为试验，不用于替代最终结果。

## 历史比较修正

10-01 的脚本给 COLMAP 使用 max_image_size=-1，其实际含义是 SIFT 自动 3200；当时 Colite House 全分辨率。此前独立前端表不是严格同尺寸吞吐对比，已修正文档。共用特征与 verified 输入的控制实验不受这一点影响。本轮两端特征覆盖 A/B 均显式使用 3200。

## 后续重点

先根据分段计时继续处理图一致性和文件工作区开销，再改善增长驱动 BA、误差尾部和大图集调度。参数剩余项按真实实现逐项接入；数据库继续暂缓。相关来源与此前独立真值要求见 [10-13 审查](colmap4_interop_audit_20261013.md) 和 [真实参考集](sfm_reference_datasets.md)。
