# CUDA PatchMatch COLMAP Alignment Implementation Plan

## 执行状态（2026-07-13）

- [x] Task 1：CUDA 随机法线初始化已实现并通过确定性、多样性测试。
- [x] Task 2：CUDA 平面感知传播已实现，并增加平面候选诊断计数。
- [x] Task 3：联合、仅深度、仅法线三类扰动候选已实现并通过计数测试。
- [x] Task 4：完成 CPU/CUDA 回归及 building-18 同参数 A/B，结果已写入 MVS 文档。

下一阶段目标：实现 bilateral weighted NCC，优先补齐参考窗口的空间权重、颜色权重和加权统计，再进行质量与耗时对比。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align COLITE CUDA PatchMatch propagation, initialization, and perturbation candidates with the corresponding COLMAP 3.11 estimator behavior.

**Architecture:** Keep CPU PatchMatch as the behavioral reference and change only the CUDA backend in the first milestone. Reuse existing option structs, deterministic random helpers, geometric gate, cost aggregation, counters, and workspace lifecycle.

**Tech Stack:** C++17, CUDA, CMake, GoogleTest, Eigen, OpenCV, WSL Ubuntu 22.04.

## Global Constraints

- 核心代码新增注释使用中文。
- 保留 checkerboard、four-direction 和 forward/backward 三种调度。
- 不修改 `references/colmap-3.11.0`。
- 每项行为先写失败测试，再实现最小改动。
- building-18 对比使用同一输入、source views、分辨率和过滤阈值。

---

### Task 1: CUDA 随机法线初始化

**Files:**
- Modify: `src/colite/mvs/patch_match_cuda.cu`
- Test: `tests/mvs/patch_match_test.cpp`

**Interfaces:**
- Consumes: `PatchMatchInitializationOptions::randomize_initial_normals`, `initial_normal_hemisphere_degrees`, `random_seed`
- Produces: initialized unit normals in the existing CUDA normal buffer

- [ ] 添加 CUDA 测试：固定 seed 下两次结果一致，单位法线有效，随机模式产生多个方向。
- [ ] 运行目标测试，确认旧实现因所有法线相同而失败。
- [ ] 扩展 `InitHypothesesKernel` 参数并调用现有确定性法线扰动函数。
- [ ] 运行目标测试和全部 `PatchMatchBackendTest`。

### Task 2: CUDA 平面感知传播

**Files:**
- Modify: `src/colite/mvs/patch_match_cuda.cu`
- Modify: `include/colite/mvs/patch_match.h`
- Test: `tests/mvs/patch_match_test.cpp`

**Interfaces:**
- Consumes: neighbor pixel, neighbor depth/normal, `DeviceCamera`
- Produces: current-ray plane intersection depth or invalid result

- [ ] 添加传播诊断字段和合成倾斜平面 CUDA 测试。
- [ ] 运行目标测试，确认旧 CUDA 直接复制邻居 depth 时失败。
- [ ] 新增设备函数：反投影邻居点，计算 `dot(n, X_neighbor) / dot(n, ray_current)`。
- [ ] 在 `TryUpdateFromNeighbor` 中以交点深度替代邻居原始深度，并记录有效平面候选。
- [ ] 运行 CUDA/CPU PatchMatchBackend 回归。

### Task 3: COLMAP 式五候选坐标下降

**Files:**
- Modify: `src/colite/mvs/patch_match_cuda.cu`
- Test: `tests/mvs/patch_match_test.cpp`

**Interfaces:**
- Consumes: current hypothesis, one random depth, one random normal
- Produces: best of joint、depth-only、normal-only candidates

- [ ] 添加 CUDA 测试，断言一个 sample 对有效像素产生三类 perturbation candidate evaluation。
- [ ] 运行测试，确认旧实现仅评估联合候选而失败。
- [ ] 抽取设备端候选评估 helper，统一几何 gate、cost 和 best-state 更新。
- [ ] 按 joint、normal-only、depth-only 顺序评估候选并保留严格最优结果。
- [ ] 运行全部 CUDA/CPU PatchMatchBackend 测试。

### Task 4: building-18 A/B 与文档

**Files:**
- Modify: `docs/mvs_status.md`
- Modify: `docs/mvs_experiments.md`

- [ ] 使用变更前记录作为 baseline，运行同参数 CUDA building-18。
- [ ] 比较几何一致像素、fused points、candidate/update ratio 和 CUDA cost/propagation/perturbation 时间。
- [ ] 若密度低于 baseline 95%，分别关闭随机初始化和解耦候选定位单因素回归。
- [ ] 将结果和下一阶段 bilateral weighted NCC 目标写入文档。
