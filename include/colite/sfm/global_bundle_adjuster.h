#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/reconstruction.h"

namespace colite {

enum class BundleGaugeFixing {
  kNone,
  kFirstImage,
  kFirstTwoImages,
};

// 全局 BA 参数。默认仍固定相机内参；需要优化内参时通过 refine_camera_params
// 指定焦距、主点、畸变等参数组，语义参考 COLMAP 的 BA refine 选项。
struct GlobalBundleAdjustmentOptions {
  int max_num_iterations = 50;
  double loss_scale = 4.0;
  bool refine_poses = true;
  bool refine_points = true;
  CameraParameterGroup refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> refine_camera_param_names;
  std::set<ImageId> constant_image_ids;
  // Keep only the rotations of these images fixed while translations and
  // structure remain variable. Used by staged global SfM.
  std::set<ImageId> constant_rotation_image_ids;
  std::map<ImageId, std::vector<int>> constant_image_translation_idxs;
  BundleGaugeFixing gauge_fixing = BundleGaugeFixing::kFirstTwoImages;
  bool use_position_priors = false;
  double position_prior_stddev = 5.0;
  double position_prior_loss_scale = 2.0;
};

struct GlobalBundleAdjustmentSummary {
  bool success = false;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_variable_cameras = 0;
  std::size_t num_gauge_constant_translation_images = 0;
  std::size_t num_residuals = 0;
  std::size_t num_position_priors = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  double initial_mean_position_error = -1.0;
  double final_mean_position_error = -1.0;
  std::set<ImageId> gauge_constant_pose_image_ids;
  std::map<ImageId, std::vector<int>> gauge_constant_translation_idxs;
  std::string message;
};

GlobalBundleAdjustmentSummary AdjustGlobalBundle(
    const GlobalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
