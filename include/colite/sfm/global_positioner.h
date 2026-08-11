#pragma once

#include <map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "colite/sfm/view_graph.h"

namespace colite {

struct GlobalPositioningOptions {
  double scale_regularization = 1e-3;
  double max_direction_residual = 0.25;
  bool filter_outliers = true;
};

struct GlobalPositioningSummary {
  bool success = false;
  std::size_t num_positions = 0;
  std::size_t num_used_constraints = 0;
  std::size_t num_rejected_constraints = 0;
  double median_direction_residual = -1.0;
};

GlobalPositioningSummary EstimateGlobalCameraCenters(
    const std::vector<RelativePoseConstraint>& constraints,
    const std::map<ImageId, Eigen::Quaterniond>& world_to_camera_rotations,
    ImageId anchor_image_id,
    const GlobalPositioningOptions& options,
    std::map<ImageId, Eigen::Vector3d>* camera_centers);

}  // namespace colite
