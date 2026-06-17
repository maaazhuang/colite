#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "colite/core/reconstruction.h"

namespace colite {

// 局部 BA 选项。第一版只优化相机外参和 3D 点，内参固定；
// 后续再按相机模型参数组接入可选内参优化。
struct LocalBundleAdjustmentOptions {
  int max_num_iterations = 25;
  double loss_scale = 4.0;
  std::size_t max_num_images = 8;
  bool refine_poses = true;
  bool refine_points = true;

  // 用于固定 gauge 的图像。若为空，局部窗口中 ID 最小的图像会被固定。
  std::set<ImageId> constant_image_ids;
};

struct LocalBundleAdjustmentSummary {
  bool success = false;
  ImageId root_image_id = kInvalidImageId;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_residuals = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  std::string message;
};

// 围绕 root_image_id 构建一个小的局部 BA 窗口并优化。
// 局部窗口包含 root 图像、与其共视点相连的已注册图像，以及这些图像观测到的 3D 点。
LocalBundleAdjustmentSummary AdjustLocalBundle(
    ImageId root_image_id,
    const LocalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
