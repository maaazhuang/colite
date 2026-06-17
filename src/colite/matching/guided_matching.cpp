#include "colite/matching/guided_matching.h"

#include <cmath>
#include <limits>

namespace colite {
namespace {

bool IsValidFeatureIndex(Point2DIdx idx, const FeatureSet& features) {
  return idx != kInvalidPoint2DIdx &&
         idx < static_cast<Point2DIdx>(features.keypoints.size());
}

double SquaredSampsonError(const Eigen::Matrix3d& fundamental_matrix,
                           const FeatureKeypoint& keypoint1,
                           const FeatureKeypoint& keypoint2) {
  const Eigen::Vector3d x1(keypoint1.x, keypoint1.y, 1.0);
  const Eigen::Vector3d x2(keypoint2.x, keypoint2.y, 1.0);
  const Eigen::Vector3d fx1 = fundamental_matrix * x1;
  const Eigen::Vector3d ftx2 = fundamental_matrix.transpose() * x2;
  const double residual = x2.transpose() * fundamental_matrix * x1;
  const double denominator =
      fx1.x() * fx1.x() + fx1.y() * fx1.y() +
      ftx2.x() * ftx2.x() + ftx2.y() * ftx2.y();
  if (denominator <= 0.0 || !std::isfinite(denominator)) {
    return std::numeric_limits<double>::infinity();
  }
  return residual * residual / denominator;
}

}  // namespace

std::vector<FeatureMatch> FilterMatchesByGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& candidate_matches,
    const Eigen::Matrix3d& fundamental_matrix,
    const GuidedMatchingOptions& options) {
  std::vector<FeatureMatch> guided_matches;
  if (candidate_matches.empty() || fundamental_matrix.norm() <= 0.0 ||
      options.max_sampson_error_px < 0.0) {
    return guided_matches;
  }

  const double max_squared_error =
      options.max_sampson_error_px * options.max_sampson_error_px;
  guided_matches.reserve(candidate_matches.size());
  for (const FeatureMatch& match : candidate_matches) {
    if (!IsValidFeatureIndex(match.idx1, features1) ||
        !IsValidFeatureIndex(match.idx2, features2)) {
      continue;
    }

    const double squared_error = SquaredSampsonError(
        fundamental_matrix, features1.keypoints[match.idx1],
        features2.keypoints[match.idx2]);
    if (squared_error <= max_squared_error) {
      guided_matches.push_back(match);
    }
  }
  return guided_matches;
}

}  // namespace colite
