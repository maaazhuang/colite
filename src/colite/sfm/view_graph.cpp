#include "colite/sfm/view_graph.h"

#include <algorithm>
#include <cmath>

namespace colite {

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
  if (summary != nullptr) {
    *summary = local_summary;
  }
  return constraints;
}

}  // namespace colite
