#include "colite/sfm/view_graph.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>

namespace colite {

namespace {

Eigen::Quaterniond RotationBetween(const RelativePoseConstraint& edge,
                                   const ImageId from,
                                   const ImageId to) {
  if (edge.image_id1 == from && edge.image_id2 == to) {
    return edge.cam2_from_cam1.q_cw;
  }
  return edge.cam2_from_cam1.q_cw.conjugate();
}

double RotationErrorDeg(const Eigen::Quaterniond& lhs,
                        const Eigen::Quaterniond& rhs) {
  const double dot = std::clamp(std::abs(lhs.dot(rhs)), 0.0, 1.0);
  return 2.0 * std::acos(dot) * 180.0 / 3.14159265358979323846;
}

}  // namespace

std::vector<RelativePoseConstraint> BuildRelativePoseConstraints(
    const MatchGraph& graph,
    const ViewGraphCalibrationOptions& options,
    ViewGraphCalibrationSummary* summary) {
  ViewGraphCalibrationSummary local_summary;
  std::vector<RelativePoseConstraint> constraints;
  constraints.reserve(graph.verified_geometries.size());
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    ++local_summary.num_input_edges;
    const bool usable_type =
        geometry.type == TwoViewGeometry::Type::kCalibratedEssential ||
        (!options.calibrated_only &&
         geometry.type == TwoViewGeometry::Type::kUncalibratedFundamental);
    if (!usable_type || !geometry.cam2_from_cam1.q_cw.coeffs().allFinite() ||
        !geometry.cam2_from_cam1.t_cw.allFinite() ||
        geometry.cam2_from_cam1.t_cw.norm() <= 1e-12) {
      ++local_summary.num_rejected_degenerate;
      continue;
    }
    if (geometry.inlier_matches.size() < options.min_num_inliers) {
      ++local_summary.num_rejected_inliers;
      continue;
    }
    if (geometry.median_triangulation_angle >= 0.0 &&
        geometry.median_triangulation_angle <
            options.min_median_triangulation_angle_deg) {
      ++local_summary.num_rejected_angle;
      continue;
    }
    RelativePoseConstraint constraint;
    constraint.image_id1 = pair.image_id1;
    constraint.image_id2 = pair.image_id2;
    constraint.cam2_from_cam1 = geometry.cam2_from_cam1;
    constraint.cam2_from_cam1.q_cw.normalize();
    constraint.num_inliers = geometry.inlier_matches.size();
    constraint.weight = std::sqrt(
        static_cast<double>(std::max<std::size_t>(1, constraint.num_inliers)));
    constraints.push_back(constraint);
    ++local_summary.num_accepted_edges;
  }
  if (options.max_rotation_cycle_error_deg > 0.0 &&
      options.min_rotation_cycle_support > 0 && constraints.size() >= 3) {
    std::map<std::pair<ImageId, ImageId>, std::size_t> edge_indices;
    std::map<ImageId, std::set<ImageId>> adjacency;
    for (std::size_t idx = 0; idx < constraints.size(); ++idx) {
      const RelativePoseConstraint& edge = constraints[idx];
      edge_indices[{edge.image_id1, edge.image_id2}] = idx;
      adjacency[edge.image_id1].insert(edge.image_id2);
      adjacency[edge.image_id2].insert(edge.image_id1);
    }
    const auto find_edge = [&](const ImageId image_id1,
                               const ImageId image_id2)
        -> const RelativePoseConstraint& {
      const auto key = std::minmax(image_id1, image_id2);
      return constraints[edge_indices.at({key.first, key.second})];
    };
    std::vector<RelativePoseConstraint> filtered;
    filtered.reserve(constraints.size());
    for (const RelativePoseConstraint& edge : constraints) {
      const std::set<ImageId>& neighbors1 = adjacency[edge.image_id1];
      const std::set<ImageId>& neighbors2 = adjacency[edge.image_id2];
      std::vector<ImageId> common_neighbors;
      std::set_intersection(neighbors1.begin(), neighbors1.end(),
                            neighbors2.begin(), neighbors2.end(),
                            std::back_inserter(common_neighbors));
      std::vector<double> errors;
      errors.reserve(common_neighbors.size());
      for (const ImageId neighbor : common_neighbors) {
        const RelativePoseConstraint& edge1 =
            find_edge(edge.image_id1, neighbor);
        const RelativePoseConstraint& edge2 =
            find_edge(neighbor, edge.image_id2);
        const Eigen::Quaterniond via_neighbor =
            RotationBetween(edge2, neighbor, edge.image_id2) *
            RotationBetween(edge1, edge.image_id1, neighbor);
        errors.push_back(RotationErrorDeg(
            edge.cam2_from_cam1.q_cw, via_neighbor.normalized()));
      }
      bool reject = false;
      if (errors.size() >= options.min_rotation_cycle_support) {
        const std::size_t middle = errors.size() / 2;
        std::nth_element(errors.begin(), errors.begin() + middle, errors.end());
        reject = errors[middle] > options.max_rotation_cycle_error_deg;
      }
      if (reject) {
        ++local_summary.num_rejected_rotation_cycle;
        --local_summary.num_accepted_edges;
      } else {
        filtered.push_back(edge);
      }
    }
    constraints = std::move(filtered);
  }
  if (summary != nullptr) {
    *summary = local_summary;
  }
  return constraints;
}

}  // namespace colite
