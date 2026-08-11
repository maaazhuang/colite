#pragma once

#include <cstddef>
#include <vector>

#include "colite/core/match_graph.h"

namespace colite {

struct RelativePoseConstraint {
  ImageId image_id1 = kInvalidImageId;
  ImageId image_id2 = kInvalidImageId;
  Pose3 cam2_from_cam1;
  double weight = 1.0;
  std::size_t num_inliers = 0;
};

struct ViewGraphCalibrationOptions {
  std::size_t min_num_inliers = 15;
  double min_median_triangulation_angle_deg = 0.0;
  bool calibrated_only = false;
};

struct ViewGraphCalibrationSummary {
  std::size_t num_input_edges = 0;
  std::size_t num_accepted_edges = 0;
  std::size_t num_rejected_degenerate = 0;
  std::size_t num_rejected_inliers = 0;
  std::size_t num_rejected_angle = 0;
};

std::vector<RelativePoseConstraint> BuildRelativePoseConstraints(
    const MatchGraph& graph,
    const ViewGraphCalibrationOptions& options = {},
    ViewGraphCalibrationSummary* summary = nullptr);

}  // namespace colite
