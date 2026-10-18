#pragma once

#include <cstddef>
#include <string>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

struct StructurelessRegistrationOptions {
  std::size_t min_num_registered_neighbors = 2;
  std::size_t min_num_inliers_per_edge = 15;
  double max_rotation_disagreement_deg = 5.0;
  // Perpendicular ray error divided by the registered-neighbor baseline.
  double max_center_disagreement = 0.25;
};

struct StructurelessRegistrationSummary {
  bool success = false;
  std::size_t num_pose_hypotheses = 0;
  std::size_t num_consistent_hypotheses = 0;
  double median_rotation_disagreement_deg = -1.0;
  double median_center_disagreement = -1.0;
  Pose3 pose;
  std::string message;
};

// Builds an initial world-to-camera pose from verified relative poses to
// registered neighbors. It does not create observations or mark the image as
// registered; callers must geometrically validate the hypothesis first.
StructurelessRegistrationSummary EstimateStructurelessPose(
    const MatchGraph& graph,
    ImageId image_id,
    const Reconstruction& reconstruction,
    const StructurelessRegistrationOptions& options = {});

// Unique target observations passing positive-depth, reprojection and parallax
// checks, with support from the required number of registered neighbors.
std::size_t ValidateStructurelessPose(
    const MatchGraph& graph, ImageId image_id,
    const Reconstruction& reconstruction, const Pose3& pose,
    const StructurelessRegistrationOptions& options,
    double max_reprojection_error_px);

}  // namespace colite
