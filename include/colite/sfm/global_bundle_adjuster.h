#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "colite/core/reconstruction.h"

namespace colite {

// 全局 BA 参数。第一版固定相机内参，只优化已注册图像外参和 3D 点。
struct GlobalBundleAdjustmentOptions {
  int max_num_iterations = 50;
  double loss_scale = 4.0;
  bool refine_poses = true;
  bool refine_points = true;
  std::set<ImageId> constant_image_ids;
};

struct GlobalBundleAdjustmentSummary {
  bool success = false;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_residuals = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  std::string message;
};

GlobalBundleAdjustmentSummary AdjustGlobalBundle(
    const GlobalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
