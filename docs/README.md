# Colite 开发文档

公共项目简介和安装入口分别位于仓库根目录的
[`README.md`](../README.md) 与 [`INSTALL.md`](../INSTALL.md)。本目录保存架构、
算法、CLI、验证环境和基准等长期技术文档。

本文档目录是项目长期维护入口。内容以当前代码和构建配置为准，不再保存逐日开发流水账；单次测试的完整命令、日志和产物统一放在仓库根目录的 `output/` 下。

## 推荐阅读顺序

1. [项目概况](project_overview.md)：目标、能力边界、目录结构、当前状态和参考项目策略。
2. [架构与重建流程](architecture_and_pipeline.md)：数据结构、SfM/MVS 分层、workspace、缓存和产物契约。
3. [命令行参考](cli_reference.md)：单步骤、组合流程、主要参数和典型命令。
4. [开发、编译与测试环境](development_environment.md)：WSL、CUDA、依赖、构建目录和参考程序隔离构建。
5. [模块实现说明](module_implementation.md)：SIFT、匹配、Mapper、PatchMatch、Fusion、网格和纹理的实现边界。
6. [开发与验证规范](development_guide.md)：新任务检查清单、测试顺序、A/B 口径、输出目录和许可证约束。
7. [开源项目对比测试](benchmark_comparison.md)：COLMAP、OpenMVG、OpenMVS 的代表性基准和剩余差距。

## 五分钟快速开始

内部已验证的 Windows 工作区：

```text
D:\Code\CXX\colite
```

WSL 路径与当前主构建：

```bash
cd /mnt/d/Code/CXX/colite
cmake --preset wsl-colite-cuda
cmake --build --preset wsl-colite-cuda --parallel 2
ctest --preset wsl-colite-cuda

COLITE=./build-wsl-colite-cuda/src/colite/exe/colite
$COLITE --help
```

公开仓库不分发 `data/` 下的数据集。下列 building-18 示例仅适用于已经在本地
准备相应数据的开发环境；普通用户应替换为自己的图像目录。

快速查看完整流程但不执行：

```bash
$COLITE recon_auto \
  --image_path data/building-18 \
  --output_path output/quickstart_plan_YYYYMMDD \
  --dry_run
```

## 新开发对话的建议开场

```text
项目位于 D:\Code\CXX\colite，主开发环境是 WSL Ubuntu-22.04，主构建目录是
build-wsl-colite-cuda。请先完整阅读 docs/README.md 及其链接的项目概况、架构、
开发环境和开发规范，再检查 AGENTS.md、git status、CMakeCache.txt 和相关测试。
不要重置已有工作树；参考源码只用于理解和 A/B，尤其不得直接复制 OpenMVS AGPL 代码。
所有新测试的日志、中间结果和产物必须进入新的 output/<测试名>_<YYYYMMDD>/，
并附中文测试说明。
```

## 文档维护原则

- CLI 默认值以 `$COLITE <command> --help` 和源码解析器为最终依据。
- 环境变化更新 `development_environment.md`；算法行为变化更新 `module_implementation.md`。
- 可复用的基准结论更新 `benchmark_comparison.md`，原始日志留在对应 `output/` 子目录。
- 不新增“某日计划”“某轮修复记录”类文档；有长期价值的结论直接归入上述主题文档。
- 文档中的基准数据必须同时记录版本、数据、参数、后端、分辨率和是否使用缓存。
