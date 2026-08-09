# Colite 项目概况

更新时间：2026-07-30。

## 目标与定位

Colite 是一个 C++17 编写的轻量三维重建系统，覆盖从原始图像到稀疏点云、稠密点云、网格和纹理模型的完整 SfM + MVS 链路。项目强调：

- 算法模块可独立运行、测试和与参考实现做 A/B；
- 以 COLMAP 风格的数据结构、模型格式和经典几何算法为主线；
- CPU 路径可用，同时对 SIFT 和 PatchMatch 提供 CUDA 加速；
- 不依赖外部下载的 PoissonRecon，源码随仓库构建；
- 不引入 GUI、rig、多传感器和深度学习等当前非核心能力。

当前版本为 `0.1.0`。系统已经能完成 `recon_sfm`、`recon_mvs` 和 `recon_auto`，也支持所有主要阶段的单步骤调用。

## 已实现能力

| 领域 | 能力 |
|---|---|
| 图像与相机 | 递归目录/相对路径、扩展 EXIF、Auto/Single/Per-folder/Per-image 内参策略、12 种相机模型的投影/稳健反投影/BA/去畸变、COLMAP text/binary 模型读取 |
| 特征 | CPU OpenCV 语义 SIFT、CPU VLFeat 语义 SIFT、CUDA OpenCV/VLFeat 两种模式、RootSIFT、多线程、AVX2/FMA |
| 匹配 | BF、FLANN、exhaustive、sequential、spatial、vocabulary tree、guided matching、两视图几何验证 |
| SfM | 初始化、图像注册、PnP、point-centric 多视图三角化、track complete/merge、局部/全局 BA、观测过滤、多连通分量模型、已有模型续建、鲁棒 Sim(3) 模型合并 |
| MVS | source view 选择、CPU/CUDA PatchMatch、光度/几何一致性、binary/text dense maps、Stereo Fusion |
| 表面 | Poisson、可见性 Delaunay graph-cut、QEM 简化、标定图像纹理图集 |
| 编排 | SfM/MVS 两段流程、完整 `recon_auto`、SfM 多模型输出与 text-model 断点续建、单步骤 stage manifest、`--resume/--force/--dry_run` |
| 输出 | 带 RGB 的稀疏/稠密 PLY、法线、可见性、PLY 网格、OBJ/MTL/PNG 纹理 |

## 当前成熟度

- SfM 主体结构在 `building-18` 上已接近 COLMAP 3.11：注册率、点数、观测数、平均 track length 和相机位姿均已做 Sim(3) 对齐验证。
- SfM 可从 verified 匹配图的多个连通分量输出独立 sparse model，也可加载已有
  COLMAP text model 后继续注册当前图像集中的剩余图像；模型间有足够 verified
  3D-3D 对应时执行鲁棒 Sim(3) 合并。当前尚未实现 hierarchical/global mapper。
- PatchMatch 保留区域的深度误差已接近 COLMAP 4.0；CUDA source 采样已加入
  三角化角、入射角和局部单应分辨率先验。House High 共用相机基准的最终稠密
  点数与 COLMAP 相差 0.95%，同时 PatchMatch wall 更低。
- Fusion 已完成 seed-track 并行，并将 PatchMatch 20 个 source 与 Fusion
  默认 50 个候选解耦；同一 COLMAP maps 输入下点数与 COLMAP Fusion 相差
  0.02%。Poisson 基本数值等价；Delaunay 的图割和长面清理已高度接近 COLMAP。
- 纹理视图选择使用 BVH 遮挡和近表面容差；无可靠相机视图的面会以 RGB
  顶点色生成独立 fallback atlas patch，不再输出退化 UV 灰色孔洞。复杂模型
  的真实相机纹理召回和接缝质量仍需继续与 COLMAP 对比。
- House 等数据集没有几何真值，跨程序点云/网格比较必须先做 Sim(3) 对齐，不能仅凭点数或未对齐欧氏距离判断精度。

## 目录结构

```text
colite/
  include/colite/        公共 C++ 接口
  src/colite/
    core/                相机、几何、Reconstruction、并行运行时
    features/            SIFT 公共封装和 CPU/CUDA 后端
    matching/            pair 生成、描述子匹配、几何验证
    sfm/                 增量建图、三角化、BA、观测管理
    mvs/                 去畸变、PatchMatch、Fusion、网格、纹理
    io/                  workspace、模型与相机先验 I/O
    controllers/         算法阶段控制器
    exe/                 统一 CLI、单步骤包装和流程编排
  tests/                 单元和集成测试
  scripts/               A/B、格式比较、合成数据和参数扫描脚本
  data/                  本地测试图像（如 building-18、house）
  thirdparty/            项目级第三方源码（PoissonRecon、vcglib）
  references/            只读参考项目源码，不参与 Colite 常规构建
  output/                每次测试独立的日志、指标和产物；Git 忽略
  docs/                  长期维护文档
```

## 参考项目与许可证边界

| 项目 | 版本 | 主要用途 | 使用边界 |
|---|---:|---|---|
| COLMAP | 3.11 | SfM 行为和数据结构主参考 | 可参考 BSD 代码和行为，仍应按 Colite 类型重写并保留第三方许可证 |
| COLMAP | 4.0 | CUDA MVS、Poisson、网格和纹理 A/B | 主要做行为/数值对齐；隔离构建，不覆盖系统 3.11 |
| OpenMVG | 2.1 | SfM 数据流和独立基准 | MPL-2.0；不直接混入修改后的源文件 |
| OpenMVS | 2.4 | 完整 MVS 工程分层和外部基准 | AGPL-3.0；严禁直接复制实现到 Colite |
| PoissonRecon | 仓库内版本 | Poisson 表面重建 | MIT；源码和许可证位于 `thirdparty/PoissonRecon` |
| vcglib | 仓库内版本 | OpenMVS 隔离参考构建 | GPL-3.0；位于 `thirdparty/vcglib`，不链接到 Colite 主程序 |
| Sift 项目 | 本地优化版本 | 并行、AVX2、CUDA SIFT 迁移来源 | 属于本项目控制的参考实现，迁移后仍需保持测试和 ABI 边界 |

`references/` 默认被 Git 忽略。它用于阅读、构建参考程序和 A/B，不应成为 Colite 头文件或链接依赖。

## 设计取舍

- `recon_sfm` 负责从原图到已注册稀疏模型；图像去畸变是 SfM 到 MVS 的 workspace 转换，因此属于 MVS 的第一阶段。
- `recon_mvs` 负责去畸变之后的稠密深度、融合和表面处理。
- `recon_auto` 只编排现有阶段，不复制算法实现。
- 数据结构尽量保持 COLMAP 兼容，但内部缓存、诊断和文件 workspace 是 Colite 自有格式。
- 优先保证单步骤可复现，再组合成大流程；所有阶段均应能通过固定输入单独 A/B。

## 近期开发优先级

1. 在带真值、宽基线、重复纹理、混合相机和更大图像集上验证 SfM/MVS 鲁棒性。
2. 降低 50-view Fusion 工作集和全分辨率 CUDA PatchMatch 的内存与调度开销。
3. 继续提升 Texturer 的真实相机纹理召回和接缝质量，同时控制遮挡假阳性。
4. 继续完善模型/点云/网格经过 Sim(3) 对齐后的自动评估工具。
