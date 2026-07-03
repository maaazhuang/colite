#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/global_bundle_adjuster.h"

namespace colite {

// 局部 BA 选项。第一版只优化相机外参和 3D 点，内参固定；
// 后续再按相机模型参数组接入可选内参优化。
struct LocalBundleAdjustmentOptions {
  int max_num_iterations = 25;
  double loss_scale = 4.0;
  std::size_t max_num_images = 8;
  // 不参与位姿优化、但对局部 3D 点提供约束的 fixed images 数量上限。
  std::size_t max_num_fixed_images = 8;
  bool refine_poses = true;
  bool refine_points = true;
  CameraParameterGroup refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> refine_camera_param_names;
  std::map<ImageId, std::vector<int>> constant_image_translation_idxs;
  BundleGaugeFixing gauge_fixing = BundleGaugeFixing::kFirstTwoImages;

  // 用于固定 gauge 的图像。若为空，局部窗口中 ID 最小的图像会被固定。
  std::set<ImageId> constant_image_ids;
};

struct LocalBundleAdjustmentSummary {
  bool success = false;
  ImageId root_image_id = kInvalidImageId;
  std::set<ImageId> local_image_ids;
  std::set<ImageId> fixed_image_ids;
  std::size_t num_local_images = 0;
  std::size_t num_constant_images = 0;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_variable_cameras = 0;
  std::size_t num_gauge_constant_translation_images = 0;
  std::size_t num_residuals = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  std::set<ImageId> gauge_constant_pose_image_ids;
  std::map<ImageId, std::vector<int>> gauge_constant_translation_idxs;
  std::string message;
};

// 围绕 root_image_id 构建一个小的局部 BA 窗口并优化。
// 局部窗口包含 root 图像、与其共视点相连的已注册图像，以及这些图像观测到的 3D 点。
LocalBundleAdjustmentSummary AdjustLocalBundle(
    ImageId root_image_id,
    const LocalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
