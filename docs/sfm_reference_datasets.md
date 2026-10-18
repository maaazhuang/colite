# SfM 真实数据与参考质量评估

Building/House 是真实图像回归集，但目前没有独立参考几何或位姿。现有比较只能证明速度、重建覆盖和与 COLMAP 的一致性，不能证明绝对准确度。

## 推荐组合

| 数据集 | 官方参考数据 | 主要用途 | 使用限制 |
|---|---|---|---|
| [NAVI](https://github.com/google/navi) 的 multiview 集合 | 物体扫描与逐图精确 2D–3D 对齐/相机标注 | SfM 注册成功率、相对旋转/平移方向误差、对齐后的相机中心误差 | 选择同一物体固定不动、相机移动的 multiview 场景；wild_set 不满足普通静态 SfM 假设。真值有标注误差，不是数学上的精确值 |
| [ETH3D high-resolution multi-view](https://www.eth3d.net/overview) 的训练集 | 激光扫描点云、渲染深度、相机标定与外参文件 | 几何准确度、可见区域覆盖、弱纹理室内/室外场景 | 官方是 MVS 基准；稀疏 SfM 的 completeness 不能直接等同于稠密排行榜。相机标定经过图像/扫描对齐，不应声称是完全不含 SfM 的原始传感器位姿 |
| [Tanks and Temples](https://tanksandtemples.org/download/) 的训练集 | 独立激光扫描几何；另提供 COLMAP 参考重建/相机轨迹/对齐 | 较大真实场景、长序列与最终几何准确度 | 提供的 COLMAP 轨迹是参考结果，不是独立位姿真值；官方几何 F-score 更适合稠密结果。遵守其数据许可 |
| [IMC/PhotoTourism](https://github.com/vcg-uvic/image-matching-benchmark) | 大规模 COLMAP 重建得到的伪真值位姿/深度，及 stereo/multiview 评估代码 | 宽基线特征、匹配与相对位姿 mAA，旅游照片变化 | 适合标准化相对比较，不能作为“超越 COLMAP 真值精度”的唯一证据 |

优先顺序：NAVI multiview 检验位姿，ETH3D 训练集检验真实几何；再加入 Tanks and Temples 的较大场景与 IMC 的宽基线前端。当前尚未下载或完成这些数据的 Colite 基准，不能把选择方案写成已有实测结果。

来源细节：[NAVI 论文](https://arxiv.org/abs/2306.09109)、[ETH3D 数据构建/注册流程](https://github.com/ETH3D/dataset-pipeline)、[ETH3D 文档](https://www.eth3d.net/documentation)、[Tanks and Temples 评估工具](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation)。

## 评估约定

1. 两端使用相同图像清单、裁剪/缩放/遮罩、特征预算策略、CPU 线程/GPU、相机模型和内参精化配置。保存完整参数、二进制版本、时间与内存；至少三次运行报告中位数。单独报告实际输出特征数，因为基础 keypoint 预算与多主方向后的数量不同。
2. 运行独立前端全流程、共用特征的匹配、共用 verified matches 的 mapper，以及共用扰动初值的 BA。隔离输入后才能判断具体阶段的速度/质量差距。
3. 相对位姿比较报告旋转误差、平移方向角误差和指定阈值下 mAA/AUC；失败或未注册图像必须计入覆盖/失败指标，不能只对成功子集算好看的误差。
4. 全局 SfM 有尺度、旋转和平移自由度。用一个统一 Sim(3) 对齐后计算中心误差；分别报告相对误差与已知米制尺度下的绝对误差。未注册图像与不足以对齐的退化场景单列。
5. 有独立扫描/深度时，按共同可见区域、遮挡与官方有效掩码计算点到表面/深度的误差和阈值内比例。相同采样规则下报告稀疏覆盖；不得拿“稀疏点数更多”或“自重投影误差更低”替代真实几何精度。
6. 已知相机/扫描只用于评估。自标定全流程不得读入最终参考位姿；固定参考内参、固定参考位姿的实验必须分别标为 triangulation / BA 控制实验。

现有可复用入口为 `scripts/benchmark_sfm.py`、`compare_colmap` 和 TXT/BIN 模型互操作。后续接入公开集时，需要保存官方场景清单、下载版本/校验、参考来源类型，并适配图像名、坐标/像素中心约定与遮罩；不能只更换输入目录就声称完成官方评估。
