# Codex 开发说明：自研三维重建系统（SfM + MVS）

## 1. 背景与目标

本项目目标是参考 COLMAP、openMVG、OpenMVS 的工程设计与算法实现，开发一个自研三维重建系统，覆盖从图像输入到稀疏重建、稠密重建、网格重建/纹理化的完整流程。

当前参考源码包括：

- `colmap-3.11.0.zip`
- `colmap-3.12.6.zip`
- `colmap-4.0.0.zip`
- `openMVG-2.1.zip`
- `openMVS-2.4.0.zip`

阶段性目标不是复制完整 COLMAP/OpenMVS，而是实现一个可维护、可裁剪、便于后续扩展的 C++ 三维重建框架。

---

## 2. 重要前提：Codex 能否访问上传的参考源码

不要假设 Codex 自动能访问 ChatGPT 对话中上传的 zip 源码。

在当前 ChatGPT 对话里，这些源码压缩包位于 `/mnt/data/`，但 Codex 是否能访问取决于实际运行方式：

1. 如果使用 Codex CLI / IDE 在本地开发，Codex 只能访问本地工作区中的文件。需要把这些 zip 解压或复制到当前项目目录，或明确放入 Codex 可见的 workspace。
2. 如果使用 Codex Cloud / ChatGPT Codex 任务，Codex 通常访问的是连接的 GitHub 仓库、任务环境或你显式提供的文件。ChatGPT 本轮对话上传的附件不应被视为自动传入 Codex 云端任务。
3. 最稳妥做法：在目标仓库中建立 `third_party_reference/` 或 `references/` 目录，放入解压后的参考源码，或只放入必要的源码片段/文档摘要，并在任务说明中明确路径。

建议目录：

```text
repo/
  docs/
    codex_3d_reconstruction_dev_brief.md
  references/
    colmap-3.11.0/
    colmap-3.12.6/
    colmap-4.0.0/
    openMVG-2.1/
    openMVS-2.4.0/
  src/
  tests/
```

如果不希望把大源码加入 Git，可以在本地保留 `references/`，并将其加入 `.gitignore`：

```gitignore
/references/
```

然后在 Codex 任务中说明：

> 参考源码已放在本地 workspace 的 `references/` 目录下；请只阅读参考，不要直接复制受许可证限制的代码。

---

## 3. 开源项目参考策略

### 3.1 总体结论

建议采用：

- SfM 主参考：COLMAP 3.11
- SfM 辅助参考：openMVG 2.1
- MVS 主参考：先参考 COLMAP 的流程与数据接口；中后期参考 OpenMVS 的工程分层与完整链路
- COLMAP 3.12.6 / 4.0：作为差异分析和后续优化参考，不作为第一阶段主线

### 3.2 为什么 SfM 优先参考 COLMAP 3.11

项目需求：不需要 rig、深度学习、Qt 特性，希望实现稳定、经典、可裁剪的 SfM 核心。

COLMAP 3.11 更适合作为主参考：

- 工程成熟，SfM/MVS 数据链路完整；
- 相比 4.0，新特性和工程抽象负担较小；
- 避免过早引入 rig、多传感器、深度学习、复杂 GUI 等非必要能力；
- 更适合抽取基础模块：特征、匹配、几何验证、增量式 SfM、BA、模型导入导出。

openMVG 适合作为辅助参考：

- 几何模块表达清晰；
- 多视几何和 SfM pipeline 更偏教学/库化；
- 可用于理解 essential/fundamental matrix、pose estimation、triangulation、global/incremental SfM 的结构；
- 但如果目标是完整工程落地，COLMAP 的端到端成熟度更高。

### 3.3 MVS 如何参考

MVS 阶段建议分两层推进：

第一阶段：以 COLMAP 的稀疏模型、相机模型、图像/位姿/深度图组织方式作为接口参考，先打通稠密重建最小链路。

第二阶段：参考 OpenMVS 的完整 MVS 流水线：

- scene 表达；
- dense point cloud；
- mesh reconstruction；
- mesh refinement；
- texture mapping；
- pipeline tools 的组织方式。

但 OpenMVS 许可证较严格，不建议直接复制代码进入闭源或许可证不兼容项目。更合理的方式是参考算法思想、数据流和模块边界，重新实现。

---

## 4. 许可证注意事项

开发时必须把“参考”和“复制”区分开。

建议原则：

1. COLMAP 可作为主要工程参考，但仍需注意第三方依赖许可证。
2. openMVG 是 MPL-2.0，修改其源码文件通常会触发 MPL 的文件级开源义务，不建议直接混入自研核心代码。
3. OpenMVS 是 AGPL-3.0，强 copyleft，闭源商业项目尤其需要谨慎。不要直接复制 OpenMVS 源码进入项目，除非明确接受 AGPL 义务或获得商业授权。
4. Codex 任务中明确要求：“参考源码仅用于理解结构和算法，不要逐行搬运或复制实现。”

推荐在仓库中添加 `docs/license_notes.md`，记录：

- 参考项目名称；
- 版本；
- 许可证；
- 使用方式：参考设计 / 参考接口 / 重新实现 / 不复制代码；
- 是否引入二进制或源码依赖。

---

## 5. 推荐系统架构

建议采用模块化 C++ 架构：

```text
src/
  core/
    types/
    math/
    logging/
    config/
  image/
    image_io/
    camera_models/
    undistortion/
  features/
    sift/
    orb_or_alternative/
    feature_database/
  matching/
    pair_generation/
    descriptor_matching/
    geometric_verification/
  sfm/
    scene_graph/
    tracks/
    reconstruction/
    incremental_mapper/
    triangulation/
    bundle_adjustment/
  mvs/
    workspace/
    depth_estimation/
    depth_fusion/
    meshing/
    mesh_refinement/
    texturing/
  io/
    colmap_import_export/
    ply/
    obj/
  app/
    cli/
  tests/
```

核心数据结构建议先实现：

```text
Camera
Image/View
FeatureKeypoint
Descriptor
ImagePair
TwoViewGeometry
Track
Point3D
Reconstruction
MVSWorkspace
DepthMap
NormalMap
Mesh
TextureAtlas
```

---

## 6. 第一阶段开发目标：最小可运行 SfM

### 6.1 输入输出

输入：

- 一组无序图像；
- 可选相机内参；
- 配置文件 `config.yaml`。

输出：

- 稀疏点云 `sparse.ply`；
- 相机位姿；
- 重建工程文件；
- 日志和中间结果。

### 6.2 必备模块

Codex 优先实现以下模块：

1. 图像读取与相机模型
   - 支持 pinhole / simple radial；
   - 支持 EXIF 焦距读取可后置；
   - 先允许用户手动输入内参。

2. 特征提取
   - 第一版可以封装 OpenCV SIFT；
   - 后续再考虑自研 SIFT 或其他特征。

3. 特征匹配
   - 暴力匹配 / FLANN；
   - ratio test；
   - mutual check；
   - 保存 pair matches。

4. 几何验证
   - fundamental / essential matrix；
   - RANSAC；
   - inlier matches；
   - relative pose recovery。

5. Track 构建
   - 将 pairwise matches 合并为 multi-view tracks；
   - 过滤短 track 和冲突 track。

6. 增量式 SfM
   - 选择初始图像对；
   - 初始化两视图；
   - PnP 注册新图像；
   - 三角化新点；
   - 局部 BA；
   - 全局 BA。

7. 输出
   - PLY 点云；
   - 简化版 COLMAP-like 文本模型；
   - 可视化检查工具可以后置。

### 6.3 验收标准

第一阶段完成标准：

- 能在小规模数据集上跑通完整 SfM；
- 输出相机位姿和稀疏点云；
- 点云方向、尺度虽然不绝对，但结构应基本正确；
- 有单元测试覆盖相机投影、三角化、PnP、BA 残差；
- 命令行可执行：

```bash
recon_sfm \
  --image_path data/images \
  --output_path output/sparse \
  --config config/sfm.yaml
```

---

## 7. 第二阶段开发目标：MVS 最小链路

MVS 不建议一开始追求完整 OpenMVS 级别效果。建议先实现可验证链路：

1. 从 SfM 输出读取相机内外参；
2. 图像去畸变；
3. 选择邻接视图；
4. 深度估计；
5. 深度图融合；
6. 输出稠密点云。

### 7.1 PatchMatch 与 CUDA

COLMAP 的高质量 PatchMatch MVS 通常依赖 GPU/CUDA 路径。若目标平台需要 CPU 运行，不应强依赖 CUDA。

建议路线：

- GPU 路线：后续实现 PatchMatch MVS，可参考 COLMAP 的算法组织；
- CPU 路线：先实现 plane-sweep / NCC / SGM 风格的深度估计作为可运行 baseline；
- 平替方案：在已知相机位姿和邻接视图条件下，实现多视图 plane sweep + cost aggregation；后续再加入 SGM/半全局代价聚合。

### 7.2 CPU MVS Baseline

第一版 CPU MVS 可做：

```text
For each reference image:
  Select N neighboring source images
  Generate depth hypotheses
  For each depth:
    Warp source images to reference view
    Compute photometric cost, e.g. NCC / ZNCC / Census
  Winner-takes-all depth selection
  Cross-view consistency check
Fuse depth maps into dense point cloud
```

验收标准：

- 能从 SfM 结果生成稠密点云；
- 不要求效果达到 COLMAP/OpenMVS；
- 能在 CPU 上运行小数据集；
- 深度图、中间 cost、融合结果可保存用于调试。

---

## 8. Codex 开发任务拆分

### Task 1：建立项目骨架

目标：创建 CMake C++ 项目，建立模块目录、基础类型和命令行入口。

要求：

- C++17 或 C++20；
- CMake；
- Eigen；
- OpenCV；
- Ceres；
- glog/spdlog 可选；
- 添加 `docs/architecture.md`；
- 添加基础单元测试框架。

验收：项目能编译，CLI 能输出版本号和帮助信息。

### Task 2：实现相机模型与几何基础

目标：实现 pinhole/simple radial 相机模型、投影/反投影、基础几何工具。

参考：COLMAP 3.11 的 camera model 设计；openMVG 的几何表达。

验收：

- 单元测试覆盖 projection/unprojection；
- 支持畸变参数；
- 提供统一 Camera 接口。

### Task 3：实现特征提取与匹配

目标：基于 OpenCV SIFT 建立第一版 feature pipeline。

验收：

- 对图像目录批量提取特征；
- 保存 keypoints/descriptors；
- 完成 pairwise matching；
- 输出 match statistics。

### Task 4：实现两视图几何验证

目标：实现 RANSAC + F/E matrix + relative pose recovery。

验收：

- 输入 matches，输出 inliers 和 relative pose；
- 对低 inlier pair 自动过滤；
- 保存 TwoViewGeometry。

### Task 5：实现 Track 构建

目标：把 pairwise matches 合并为多视图 tracks。

验收：

- 每个 track 不允许同一 image 出现多个 keypoint；
- 支持 track 长度过滤；
- 输出 track 统计。

### Task 6：实现增量 SfM

目标：打通 sparse reconstruction。

验收：

- 自动选择初始 pair；
- 支持 PnP 注册新图；
- 支持三角化；
- 支持局部/全局 BA；
- 输出 sparse.ply。

### Task 7：实现 COLMAP-like 导入导出

目标：支持与 COLMAP 文本模型进行互操作。

验收：

- 可导出 cameras.txt/images.txt/points3D.txt 类似格式；
- 可读取必要字段用于 MVS；
- 与内部 Reconstruction 结构转换明确。

### Task 8：实现 CPU MVS baseline

目标：从 SfM 结果生成稠密点云 baseline。

验收：

- 支持邻接视图选择；
- 支持 plane sweep depth estimation；
- 支持 depth consistency check；
- 支持 fuse depth maps；
- 输出 dense.ply。

---

## 9. 对 Codex 的实现约束

Codex 执行开发时必须遵守：

1. 不要直接复制 COLMAP/openMVG/OpenMVS 大段源码。
2. 可以阅读参考工程，提炼模块边界、接口设计和算法流程。
3. 每个任务先实现最小可运行版本，再逐步优化。
4. 每个模块必须有清晰单元测试或最小集成测试。
5. 不要引入 Qt/UI。
6. 不要引入 rig、多传感器、深度学习相关复杂特性。
7. MVS 第一版必须支持 CPU baseline，不强制 CUDA。
8. 所有依赖必须写入 CMake 和文档。
9. 对许可证敏感代码，必须重新实现，不要直接移植。

---

## 10. 推荐 Codex Prompt 模板

可以把下面内容直接作为 Codex 任务前缀：

```text
你正在开发一个自研 C++ 三维重建系统，目标是实现 SfM + MVS 的最小可运行框架。

参考源码位于 workspace 的 references/ 目录：
- references/colmap-3.11.0/
- references/colmap-3.12.6/
- references/colmap-4.0.0/
- references/openMVG-2.1/
- references/openMVS-2.4.0/

请注意：参考源码仅用于理解工程结构、算法流程和接口设计，不要直接复制大段源码，尤其不要复制 OpenMVS 的 AGPL 代码。

当前优先级：
1. SfM 主参考 COLMAP 3.11。
2. openMVG 仅辅助理解几何和 SfM pipeline。
3. MVS 第一版先做 CPU baseline，后续再考虑 GPU PatchMatch。
4. 不做 Qt、不做 rig、不做深度学习特性。

请先阅读 docs/codex_3d_reconstruction_dev_brief.md，然后完成本次指定任务。实现时要求：
- 代码模块化；
- CMake 可编译；
- 有最小测试；
- 保持接口清晰；
- 遇到复杂算法先实现 baseline，再留下 TODO 说明优化方向。
```

---

## 11. 当前推荐开发顺序

1. 项目骨架 + CMake + CLI；
2. Camera / Image / Reconstruction 数据结构；
3. 特征提取与匹配；
4. 几何验证；
5. Track 构建；
6. Incremental SfM；
7. BA；
8. COLMAP-like 导入导出；
9. MVS workspace；
10. CPU depth estimation baseline；
11. depth fusion；
12. mesh/texturing 后置。

---

## 12. 暂不做事项

第一阶段不要做：

- Qt GUI；
- rig/multi-camera sensor；
- 深度学习特征或深度估计；
- 大规模分布式重建；
- OpenMVS 代码直接移植；
- 高性能 CUDA PatchMatch；
- 完整 texture mapping。

这些能力可以在 SfM + CPU MVS baseline 稳定后再评估。
