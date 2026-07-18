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

## 默认大图 MVS 性能

2026-07-18 House 36 图，默认 `resolution_level=1`（2296×1528）、CUDA PatchMatch、3 iterations、radius 2、8 sources、binary maps：

| 指标 | 结果 |
|---|---:|
| 端到端墙钟 | 254.13 s |
| PatchMatch batch wall | 156.44 s |
| CUDA 时间合计 | 141.91 s |
| 峰值 RSS | 约 3.32 GiB |

默认每迭代仅一轮红黑传播并关闭热路径原子诊断。快速预览可用 `resolution_level=2`；全分辨率用 `0` 并建议额外限制 `max_image_size`。

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
