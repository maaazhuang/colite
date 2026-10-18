# Colite

[English](README.md) | 简体中文

Colite 是一个实验性的 C++17 运动恢复结构（SfM）与多视图立体视觉（MVS）系统。它可以从图像重建兼容 COLMAP 的稀疏模型、带 RGB 信息的稠密点云、Poisson 或 Delaunay 网格，以及带纹理的 OBJ 模型。

当前版本属于 Alpha 阶段的研究与工程构建，主要在 Ubuntu 22.04 和 WSL2 上开发和测试。SIFT、描述子匹配与 PatchMatch Stereo 可选用 CUDA 加速。原生 Windows 构建尚未作为受支持的发布目标完成验证。

## 主要功能

- 增量式 SfM，支持多模型、已有模型续建和稳健 Sim(3) 合并。
- 12 种兼容 COLMAP 的相机模型，支持 EXIF 焦距初始化、可选 GPS 位置先验，以及 auto/single/per-folder/per-image 内参共享策略。
- CPU 与 CUDA SIFT、CUDA 精确暴力描述子匹配、guided matching 和多种图像对选择策略。
- CPU/CUDA PatchMatch、几何一致性、输出 RGB/法线的 Stereo Fusion、Poisson 与 Delaunay 网格、QEM 简化和纹理图集。
- COLMAP 文本与二进制稀疏模型读写，默认同时保存两种格式；支持 `model_converter` 转换和 `compare_colmap` 跨格式对比，以及可断点恢复的阶段清单。
- SfM 已删除同义旧参数与别名，使用 COLMAP 4.0 的 `ImageReader.*`、`FeatureExtraction.*`、`SiftMatching.*`、`AlikedMatching.*`、`TwoViewGeometry.*`、`Mapper.*` 名称；特有控制使用 `Colite.*`。尚未对齐的默认值和功能见[命令行参考](docs/cli_reference.md)，真实参考集选择见[评估约定](docs/sfm_reference_datasets.md)。
- 独立的模型分析、对齐、变换、裁剪、拆分、束调整、三角化和注册命令。

当前 SfM 参数和前端优化进展见 [2026-10-18 记录](docs/sfm_alignment_20261018.md)：统一 GPU/线程/图像尺寸控制，修正 CUDA SIFT 的尺度与多方向预算，并优化任意特征数的整数 GPU 匹配。数据库适配暂缓。

## 构建

安装 Ubuntu 依赖并构建 CPU Release：

```bash
./scripts/setup_ubuntu_22_04.sh
cmake --preset wsl-release
cmake --build --preset wsl-release --parallel 2
ctest --preset wsl-release
```

具备兼容的 NVIDIA 驱动和 CUDA Toolkit 时：

```bash
cmake --preset wsl-colite-cuda
cmake --build --preset wsl-colite-cuda --parallel 2
ctest --preset wsl-colite-cuda
```

项目不会硬编码 CUDA 编译器和 GPU 架构。自动检测不合适时，可传入 `-DCMAKE_CUDA_COMPILER=...` 或 `-DCMAKE_CUDA_ARCHITECTURES=...`。依赖、安装方式和故障排查见 [INSTALL.md](INSTALL.md)。

## 快速开始

仓库不发布图像数据集，请将 `--image_path` 指向自己的图像目录：

```bash
COLITE=./build-wsl-colite-cuda/src/colite/exe/colite

$COLITE recon_auto \
  --image_path /path/to/images \
  --output_path output/my_reconstruction \
  --sfm.SiftExtraction.max_num_features 8192 \
  --sfm.FeatureExtraction.num_threads 8 \
  --mvs.patch_match_backend cuda \
  --mesher poisson \
  --simplify 1 \
  --texture 1
```

没有 CUDA 时，请使用 CPU 构建并设置 `--mvs.patch_match_backend cpu`。当前构建支持的准确参数以帮助信息为准：

```bash
$COLITE --help
$COLITE recon_auto --help
```

主要输出如下：

```text
OUTPUT/
  sfm/                         COLMAP 稀疏模型与 RGB PLY
  mvs/dense/fused.ply          带 RGB 和法线的稠密点云
  surface/meshed-poisson.ply   或 meshed-delaunay.ply
  surface/textured/mesh.obj    纹理网格、MTL 与图集 PNG
  .colite/stages/              断点恢复阶段清单
```

## 文档

- [文档索引](docs/README.md)
- [架构与重建流程](docs/architecture_and_pipeline.md)
- [命令行参考](docs/cli_reference.md)
- [开发环境](docs/development_environment.md)
- [模块实现与已知差距](docs/module_implementation.md)
- [基准方法与对比结果](docs/benchmark_comparison.md)
- [贡献指南](CONTRIBUTING.md)

## 当前限制

- Mapper 支持增量式、多模型、global 和 hierarchical；后两者仍需更广泛的大规模与退化场景验证。
- CUDA 性能和显存占用会随图像尺寸、源视图数量、GPU 架构与缓存配置明显变化。
- 纹理接缝、可见性召回率、退化几何上的 Delaunay 行为和超大数据集仍需更广泛验证。
- 兼容 COLMAP 格式并不表示所有数据集上的数值结果或重建精度完全一致。

详细限制见 [docs/module_implementation.md](docs/module_implementation.md)。

## 许可证

Colite 采用 [BSD 3-Clause License](LICENSE)。第三方声明和保留的上游许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
