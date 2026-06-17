#pragma once

#include <cstddef>
#include <map>

#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/guided_matching.h"
#include "colite/matching/two_view_geometry_verifier.h"

namespace colite {

struct WeakPairRematchingOptions {
  // 只对文件顺序近邻图像对启用二阶段重匹配；0 表示不限制邻域。
  std::size_t nearby_image_window = 2;

  // 新几何至少需要比旧几何多出的内点数，避免用噪声候选替换稳定结果。
  std::size_t min_inlier_gain = 5;

  DescriptorMatchingOptions relaxed_matching;
  GuidedMatchingOptions guided_matching;
  TwoViewGeometryOptions verification;
};

struct WeakPairRematchingSummary {
  std::size_t num_checked_pairs = 0;
  std::size_t num_rematched_pairs = 0;
  std::size_t num_improved_pairs = 0;
};

// 对弱相邻图像对执行二阶段重匹配：更宽松描述子候选 -> guided filtering ->
// 两视图几何重验证。只有内点数有明确收益时才更新 MatchGraph。
WeakPairRematchingSummary RematchWeakImagePairs(
    const std::map<ImageId, Camera>& cameras_by_image,
    const WeakPairRematchingOptions& options,
    MatchGraph* graph);

}  // namespace colite
