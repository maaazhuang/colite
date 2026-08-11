#include "colite/sfm/structureless_registrar.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace colite {
namespace {

double RotationDistanceDeg(const Pose3& lhs, const Pose3& rhs) {
  const Eigen::Quaterniond delta = lhs.q_cw * rhs.q_cw.conjugate();
  return 2.0 * std::acos(std::clamp(std::abs(delta.normalized().w()), 0.0, 1.0)) *
         180.0 / 3.14159265358979323846;
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return -1.0;
  }
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  return values[middle];
}

}  // namespace

StructurelessRegistrationSummary EstimateStructurelessPose(
    const MatchGraph& graph,
    const ImageId image_id,
    const Reconstruction& reconstruction,
    const StructurelessRegistrationOptions& options) {
  StructurelessRegistrationSummary summary;
  if (!reconstruction.ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }
  std::vector<Pose3> hypotheses;
  for (const MatchGraphEdge& edge : FindVerifiedNeighbors(graph, image_id)) {
    if (edge.geometry == nullptr ||
        edge.geometry->inlier_matches.size() < options.min_num_inliers_per_edge) {
      continue;
    }
    const ImageId neighbor_id = edge.OtherImageId(image_id);
    if (!reconstruction.IsImageRegistered(neighbor_id)) {
      continue;
    }
    const Image& neighbor = reconstruction.GetImage(neighbor_id);
    if (!neighbor.cam_from_world.has_value()) {
      continue;
    }
    const Pose3 target_from_neighbor =
        edge.pair.image_id2 == image_id
            ? edge.geometry->cam2_from_cam1
            : edge.geometry->cam2_from_cam1.Inverse();
    hypotheses.push_back(target_from_neighbor * *neighbor.cam_from_world);
    hypotheses.back().q_cw.normalize();
  }
  summary.num_pose_hypotheses = hypotheses.size();
  if (hypotheses.size() < options.min_num_registered_neighbors) {
    summary.message = "Not enough registered view-graph neighbors";
    return summary;
  }

  std::size_t best_idx = 0;
  std::size_t best_support = 0;
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < hypotheses.size(); ++i) {
    std::size_t support = 0;
    double cost = 0.0;
    for (const Pose3& other : hypotheses) {
      const double rotation_error = RotationDistanceDeg(hypotheses[i], other);
      const double center_error =
          (hypotheses[i].CameraCenter() - other.CameraCenter()).norm();
      if (rotation_error <= options.max_rotation_disagreement_deg &&
          center_error <= options.max_center_disagreement) {
        ++support;
        cost += rotation_error + center_error;
      }
    }
    if (support > best_support || (support == best_support && cost < best_cost)) {
      best_idx = i;
      best_support = support;
      best_cost = cost;
    }
  }
  summary.pose = hypotheses[best_idx];
  summary.num_consistent_hypotheses = best_support;
  std::vector<double> rotation_errors;
  std::vector<double> center_errors;
  for (const Pose3& hypothesis : hypotheses) {
    const double rotation_error = RotationDistanceDeg(summary.pose, hypothesis);
    const double center_error =
        (summary.pose.CameraCenter() - hypothesis.CameraCenter()).norm();
    if (rotation_error <= options.max_rotation_disagreement_deg &&
        center_error <= options.max_center_disagreement) {
      rotation_errors.push_back(rotation_error);
      center_errors.push_back(center_error);
    }
  }
  summary.median_rotation_disagreement_deg = Median(std::move(rotation_errors));
  summary.median_center_disagreement = Median(std::move(center_errors));
  summary.success = best_support >= options.min_num_registered_neighbors;
  summary.message = summary.success
                        ? "Consistent structure-less pose hypothesis found"
                        : "Relative-pose hypotheses are inconsistent";
  return summary;
}

}  // namespace colite
