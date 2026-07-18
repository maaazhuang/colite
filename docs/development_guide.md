# Colite 开发与验证规范

## 新任务开始前

1. 阅读 [文档入口](README.md)、[项目概况](project_overview.md)、[架构](architecture_and_pipeline.md) 和本文件。
2. 阅读根目录 `AGENTS.md`，确认输出目录规则。
3. 在 WSL 执行 `git status --short`，识别用户已有改动；不得把脏工作树当成可清理内容。
4. 检查目标构建的 `CMakeCache.txt`，确认 CUDA、CGAL、PoseLib 和 Build Type。
5. 先运行相关定向测试建立基线，再修改代码。
6. 查找 `references/` 中对应版本时记录许可证；OpenMVS 只可理解行为，不可复制代码。

## 实现流程

1. 明确输入、输出、坐标/单位、缓存和失败语义。
2. 优先修改现有算法内核，不在 CLI 薄封装中复制逻辑。
3. 新模块按“核心 API → 单元测试 → 控制器 → CLI stage → `recon_auto`”接入。
4. 影响缓存结果的算法修改必须更新 stage/config `algorithm_version`。
5. CPU 并行优先使用 `core/parallel_for`；只有不适合持久线程池时才引入专用调度。
6. CUDA 与 CPU 必须保持同一坐标、descriptor、阈值和输出契约；性能优化不能绕过质量判定。
7. 先加入最小确定性测试，再跑真实场景；复杂 bug 应保留针对根因的回归用例。

## 文件与代码边界

- 公共 API 放 `include/colite/<module>`，实现放对应 `src/colite/<module>`。
- 算法组合放 `controllers`，命令解析和 stage manifest 放 `exe`。
- 第三方源码必须有明确许可证并统一放在项目级 `thirdparty`；普通参考源码只放 `references`。
- 不在 `build-*`、`data`、`src` 或仓库根目录生成测试结果。
- 不直接编辑用户已有产物，不使用 `git reset --hard`、`git checkout -- .` 等破坏性命令。

## 测试输出规则

每次独立测试创建新的：

```text
output/<功能>_<数据或场景>_<YYYYMMDD>/
```

目录至少包含中文 `test_summary.md`，记录：

- Git 版本/工作树状态和构建配置；
- 输入数据、图像数量/尺寸和相机参数；
- 完整命令和全部非默认参数；
- CPU/GPU、线程、分辨率、迭代次数；
- 是否冷缓存、是否复用 stage/dense maps；
- wall time、峰值 RSS/显存（可获得时）；
- 输出文件及点/面/像素统计；
- 精度指标、限制和结论。

不同后端、分辨率、迭代或参数必须使用不同目录，防止缓存污染。构建目录仅保存可重建的编译产物。

## 推荐测试顺序

修改单模块时按以下顺序推进，前一级失败时不进入昂贵流程：

1. 编译目标和 `git diff --check`。
2. 新增/相关单元测试。
3. 相关集成/CLI 测试。
4. 小型合成 fixture 或缩小分辨率 smoke。
5. `data/building-18` 单步骤测试。
6. 需要时运行 House 或参考项目 A/B。
7. `ctest --test-dir build-wsl-colite-cuda --output-on-failure --parallel 2`。
8. 只有单步骤均通过后才运行 `recon_sfm`、`recon_mvs`、`recon_auto`。

完整流程建议验证顺序：feature → matcher → mapper → undistort → PatchMatch → Fusion → Poisson/Delaunay → simplify → texture → `recon_sfm` → `recon_mvs` → `recon_auto`。

## A/B 公平性

### SfM

至少固定：

- 图像列表、EXIF/固定相机、相机模型和共享相机策略；
- SIFT 类型、octave、first octave、threshold、方向数和特征上限语义；
- pair 列表、ratio、cross-check、guided matching 和线程；
- E/F/H 阈值、注册和三角化阈值；
- BA 内参开关和 PoseLib/求解器状态。

报告注册图像、实际 descriptor 数、raw/verified pairs、点数、观测、track 分布、重投影误差和耗时。两套独立 SfM 必须先按共同相机中心做 Sim(3) 对齐；跨模型 `point3D_id` 没有语义。

### MVS

至少固定：

- 同一稀疏模型/位姿或明确记录上游差异；
- 去畸变相机、工作尺寸、source 数、窗口、迭代、金字塔；
- CPU/CUDA、GPU、线程、HMM/CDF、几何 pass；
- Fusion 的观测、重投影、深度、法线阈值；
- 网格阶段使用相同点云/可见性和相同参数。

深度报告有效覆盖率、共同有效覆盖、稀疏点残差和共同像素差。点云/网格需对齐后报告双向距离、拓扑、面积、边界/非流形边。纹理报告有效 UV 面比例、atlas 尺寸、非黑像素和面集合 Jaccard。

### 性能

- 区分冷缓存、热缓存、进程启动和内核时间。
- CUDA 首次上下文、JIT/分配和 cache reuse 必须单独说明。
- vocabulary builder、图像转换和模型转换不要混入纯 matcher/算法计时。
- 至少重复三次并报告中位数后才能做稳定速度结论；单次结果只能定位数量级。

## 许可证与参考源码

- COLMAP BSD：可参考，但尽量按 Colite API 重写并保留第三方许可证说明。
- OpenMVG MPL-2.0：不要把修改后的源文件直接复制进核心。
- OpenMVS AGPL-3.0：不得复制函数、类或大段实现；只能参考论文、公开算法和数据流后独立实现。
- 对 OpenMVS 的任何“参考实现”提交都应能解释独立设计来源和测试依据。
- PoissonRecon MIT：随仓库分发时保留许可证。

## 高风险注意事项

- 不要给整个 `colite_core` 添加 AVX2，避免 Eigen 对齐 ABI 跨边界崩溃。
- 修改像素坐标时检查 COLMAP `(0.5,0.5)` 与 OpenCV `(0,0)` 的约定。
- 修改 observation/track 时保持 `Image` 与 `Point3D` 双向一致。
- 修改 CUDA source selection/传播时同时验证 CPU/CUDA 数值和点集重叠。
- 不要把“更多点”直接等同于更高精度；必须检查误差尾部、track 和真值/对齐指标。
- 不要把 OpenMVS 的 `DensifyPointCloud` 总时间直接与 COLMAP 单个 PatchMatch 阶段比较；两者包含阶段和缓存语义不同。

## 提交前检查

```bash
cd /mnt/d/Code/CXX/colite
git status --short
cmake --build build-wsl-colite-cuda --parallel 2
ctest --test-dir build-wsl-colite-cuda --output-on-failure --parallel 2
git diff --check
```

核对：只包含本任务文件；没有 `build-*`、`data`、`references`、`output` 大文件；文档参数与 `--help` 一致；测试说明在 `output/`；没有覆盖用户无关修改。
