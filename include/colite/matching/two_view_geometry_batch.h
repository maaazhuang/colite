#pragma once

#include <cstddef>
#include <map>

#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/core/types.h"
#include "colite/matching/two_view_geometry_verifier.h"

namespace colite {

// 批量两视图几何验证选项。它只负责调度单对验证器，
// 不改变特征提取、描述子匹配或后续增量 SfM 的职责边界。
struct TwoViewGeometryBatchOptions {
  TwoViewGeometryOptions geometry_options;

  // 第一版默认只保留可用于建图的验证边；退化边可以通过该开关保留以便调试。
  bool keep_degenerate = false;
};

struct TwoViewGeometryBatchSummary {
  std::size_t num_input_pairs = 0;
  std::size_t num_verified_pairs = 0;
  std::size_t num_degenerate_pairs = 0;
};

// 对 MatchGraph::matches 中的所有候选图像对做几何验证，并写回
// MatchGraph::verified_geometries。cameras_by_image 以 ImageId 为键，
// 代表每张图像实际使用的相机内参。
TwoViewGeometryBatchSummary VerifyMatchGraphGeometries(
    const std::map<ImageId, Camera>& cameras_by_image,
    MatchGraph* graph,
    const TwoViewGeometryBatchOptions& options = {});

}  // namespace colite
