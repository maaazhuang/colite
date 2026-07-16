# Mapper 与 COLMAP 3.11 精度验证（2026-07-15）

## 结论

building-18 上出现的 17/18（sequential）和 16/18（vocab）注册不完整，根因是 COLITE PnP RANSAC 结束后的 DLT 重估会无条件覆盖已经找到的最佳 P3P 位姿。即使 DLT 重估后的内点更少，旧实现仍保留劣化模型，随后因内点不足拒绝注册。

修复为与 COLMAP 局部优化相同的保守接受准则：只有重估模型的内点数不少于当前最佳模型时才替换。修复没有放宽 `4 px` 初始阈值，也没有降低最小内点数。

修复后：

| 输入候选 | 修复前注册 | 修复后注册 | 修复后点数 |
|---|---:|---:|---:|
| sequential overlap=5 | 17/18 | 18/18 | 15,578 |
| hierarchical vocab + overlap=2 guard | 16/18 | 18/18 | 15,552 |

此前失败的 sequential 图像 18 在修复后使用 842 个 2D-3D 对应、725 个 PnP 内点一次注册成功；图像 1 和 3 分别以 1,537/1,762、478/848 内点/对应一次注册成功。Vocab 中图像 1 和 3 分别以 1,541/1,780、485/853 内点/对应一次注册成功。

## 精度比较方法

两个独立 SfM 模型存在任意全局尺度、旋转和平移，不能直接比较 `qvec/tvec`。旧 `compare_colmap` 直接相减位姿，得到的约 9°和数米差值混入了 gauge 差异，不代表重建误差。

当前比较器先使用共同图像的相机中心估计 Umeyama Sim(3)，将 COLITE 模型对齐到 COLMAP 坐标系，再计算：

- 相机旋转角误差；
- 相机中心欧氏误差及其相对 COLMAP 最大基线的比例；
- 模型平均重投影误差；
- 平均 track length。

`point3D_id` 是各重建内部递增编号，独立 Mapper 之间同 ID 不代表同一空间点，因此 `common_points_by_id` 和旧 `xyz_delta` 不能作为点云精度结论。相机位姿、重投影误差和 track 统计才是本轮使用的有效指标。

## building-18 对比结果

参考模型：COLMAP 3.11，18 张注册图像、9,148 点。候选模型使用相同 building-18 图像和 COLITE 已验证匹配。

| 指标 | sequential | vocab |
|---|---:|---:|
| 注册图像 | 18/18 | 18/18 |
| 平均旋转误差 | 0.1446° | 0.0509° |
| 中位旋转误差 | 0.1323° | 0.0502° |
| 最大旋转误差 | 0.2358° | 0.0942° |
| 平均相机中心误差 | 0.004385 | 0.003576 |
| 最大相机中心误差 | 0.012460 | 0.007958 |
| 平均中心误差 / 最大基线 | 0.0348% | 0.0284% |
| 最大中心误差 / 最大基线 | 0.0988% | 0.0631% |
| 平均重投影误差 | 0.6540 px | 0.6677 px |
| COLMAP 平均重投影误差 | 0.7907 px | 0.7907 px |
| 3D 点数 | 15,578 | 15,552 |
| 平均 track length | 3.309 | 3.319 |
| COLMAP 平均 track length | 4.343 | 4.343 |

相机位姿已经达到小于 0.25°最大旋转误差、相机中心小于 0.1%场景最大基线的对齐水平；平均重投影误差不高于 COLMAP。本轮 Mapper 的“注册率和位姿精度”可以视为已对齐。

尚未完全对齐的是模型结构：COLITE 生成约 70% 更多的短 track 点，平均 track length 比 COLMAP 低约 23.6%。这不是当前注册失败的原因，但会增加 BA 变量数并影响稠密度/稳定性。后续 P1 应对齐 COLMAP IncrementalTriangulator 的 continue/complete/merge 策略，重点统计重复 3D 点、可合并 track、每图观察覆盖率，而不是继续放宽 PnP 或重投影阈值。

## 命令

```bash
colite mapper --workspace_path <workspace>

colite compare_colmap \
  --reference_path <colmap_text_model> \
  --candidate_path <colite_sparse_model> \
  --output_path <report_dir>
```

输出 `summary.txt` 现在包含 Sim(3) 是否有效、尺度、旋转误差、相机中心误差/基线、重投影误差和 track length；`pose_diff.tsv` 中的候选位姿也是对齐后的位姿。
