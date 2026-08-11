#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include <Eigen/Geometry>

#include "colite/sfm/view_graph.h"

namespace colite {

struct RotationAveragingOptions {
  std::size_t max_num_iterations = 100;
  double convergence_threshold_deg = 1e-5;
  double max_relative_rotation_error_deg = 10.0;
};

struct RotationAveragingSummary {
  bool success = false;
  std::size_t num_iterations = 0;
  std::size_t num_connected_images = 0;
  std::size_t num_used_constraints = 0;
  std::size_t num_rejected_constraints = 0;
  double median_residual_deg = -1.0;
  double max_update_deg = -1.0;
};

RotationAveragingSummary EstimateGlobalRotations(
    const std::vector<RelativePoseConstraint>& constraints,
    ImageId anchor_image_id,
    const RotationAveragingOptions& options,
    std::map<ImageId, Eigen::Quaterniond>* world_to_camera_rotations);

}  // namespace colite
