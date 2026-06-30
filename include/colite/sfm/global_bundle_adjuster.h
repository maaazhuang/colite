#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "colite/core/camera.h"
#include "colite/core/reconstruction.h"

namespace colite {

// 全局 BA 参数。默认仍固定相机内参；需要优化内参时通过 refine_camera_params
// 指定焦距、主点、畸变等参数组，语义参考 COLMAP 的 BA refine 选项。
struct GlobalBundleAdjustmentOptions {
  int max_num_iterations = 50;
  double loss_scale = 4.0;
  bool refine_poses = true;
  bool refine_points = true;
  CameraParameterGroup refine_camera_params = CameraParameterGroup::kNone;
  std::set<ImageId> constant_image_ids;
};

struct GlobalBundleAdjustmentSummary {
  bool success = false;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_variable_cameras = 0;
  std::size_t num_residuals = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  std::string message;
};

GlobalBundleAdjustmentSummary AdjustGlobalBundle(
    const GlobalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
