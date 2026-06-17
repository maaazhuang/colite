#pragma once

#include <vector>

#include <Eigen/Core>

#include "colite/core/feature.h"

namespace colite {

struct GuidedMatchingOptions {
  // Sampson error 的像素阈值。这里使用像素坐标下的 F 矩阵，因此阈值与几何验证
  // 的 max_error_px 保持同一量纲。
  double max_sampson_error_px = 4.0;
};

// 使用已有基础矩阵对宽松匹配候选做 guided filtering。
// 该函数不重新搜索描述子，只负责把不满足极线几何的候选匹配剔除。
std::vector<FeatureMatch> FilterMatchesByGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& candidate_matches,
    const Eigen::Matrix3d& fundamental_matrix,
    const GuidedMatchingOptions& options = {});

}  // namespace colite
