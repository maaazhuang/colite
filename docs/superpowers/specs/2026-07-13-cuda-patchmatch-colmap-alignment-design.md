# CUDA PatchMatch COLMAP Alignment Design

## 目标

在保留 COLITE 现有 CUDA workspace、checkerboard 调度、几何 gate 和诊断能力的前提下，逐步对齐 COLMAP 3.11 的核心估计行为。第一阶段只处理三个相互关联且可独立验证的差距：平面感知传播、随机法线初始化、深度/法线解耦候选搜索。

## 设计原则

- CPU PatchMatch 继续作为算法正确性参考，不改变现有 CPU 默认行为。
- CUDA 默认调度仍保持 checkerboard，避免本阶段同时引入 rotated sweep 变量。
- 新行为复用现有 `PatchMatchInitializationOptions` 和 `PatchMatchPerturbationOptions`，不增加 CLI 开关。
- 所有 CUDA 随机数继续由像素索引和固定 seed 确定，保证测试可复现。
- 每个里程碑分别运行 CUDA 单元测试、CPU 回归测试和 building-18 A/B；质量不回归后再进入下一阶段。

## 第一阶段架构

### 平面感知传播

CUDA 从邻居读取 `(depth, normal)` 后，不再直接把邻居 depth 用作当前像素候选。设备函数先将邻居像素按其深度反投影到参考相机坐标系，再计算邻居平面与当前像素射线的交点深度。无穷、非正、近平行或超出深度范围的交点直接拒绝。

该公式与 CPU `ProjectPlaneDepthToPixel` 保持同一语义，因此倾斜平面可以沿传播方向连续延伸。

### 随机法线初始化

当 `randomize_initial_normals=false` 时保持固定默认法线。当选项为 true 时，CUDA 使用确定性随机扰动，在 `initial_normal_hemisphere_degrees` 限制内围绕默认法线采样并归一化。无效默认法线仍回退到既有默认方向。

### 五候选坐标下降

每个随机 sample 比较以下状态：

1. 当前 depth + 当前 normal，已由 cost map 表示；
2. 传播 depth + 传播 normal，由传播 pass 表示；
3. 随机 depth + 随机 normal；
4. 当前 depth + 随机 normal；
5. 随机 depth + 当前 normal。

后三种候选共享几何 pre-cost gate 和多源 cost 聚合函数。只有严格更低 cost 才更新状态。

## 后续阶段

1. bilateral weighted NCC 与 reference patch statistics 复用；
2. GPU 动态 selection probability、视角/尺度先验和 Monte Carlo source sampling；
3. forward-backward reprojection geometric cost；
4. COLMAP rotated sweep 与集成 consistency mask；
5. 关闭热路径诊断原子操作、二维 shared tile、模板化窗口和 GPU pyramid handoff；
6. 多 GPU 问题调度和统一质量/性能基准。

## 验收标准

- CUDA 随机初始化法线有限、单位长度、可复现，并且至少存在两个不同方向。
- CUDA 传播使用平面交点深度；合成倾斜平面用例中传播候选计数非零且输出保持有效。
- 开启一个 perturbation sample 时，每个有效像素最多评估三种解耦候选，而不是一种联合候选。
- CUDA 与 CPU PatchMatchBackend 测试通过。
- building-18 同参数 A/B 中，几何一致像素和 fused points 不低于当前基线的 95%，同时记录 CUDA 各阶段耗时变化。

