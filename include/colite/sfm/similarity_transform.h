#pragma once

#include <vector>

#include <Eigen/Core>

#include "colite/core/pose.h"
#include "colite/core/reconstruction.h"

namespace colite {

struct SimilarityTransform3 {
  double scale = 1.0;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  bool valid = false;

  Eigen::Vector3d TransformPoint(const Eigen::Vector3d& point) const;
  Pose3 TransformPose(const Pose3& pose) const;
};

SimilarityTransform3 EstimateSimilarityTransform(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& target_points,
    bool estimate_scale = true);

void TransformReconstruction(const SimilarityTransform3& transform,
                             Reconstruction* reconstruction);

struct PositionPriorAlignmentSummary {
  bool success = false;
  std::size_t num_priors = 0;
  std::size_t num_inliers = 0;
  double scale = 1.0;
  double mean_error = -1.0;
};

PositionPriorAlignmentSummary AlignReconstructionToPositionPriors(
    double max_error,
    Reconstruction* reconstruction);

}  // namespace colite
