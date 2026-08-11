#include "colite/sfm/global_positioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/QR>

namespace colite {
namespace {

struct PositionSolveResult {
  bool success = false;
  std::map<ImageId, Eigen::Vector3d> centers;
  std::vector<double> scales;
};

PositionSolveResult SolvePositions(
    const std::vector<RelativePoseConstraint>& constraints,
    const std::vector<bool>& active,
    const std::map<ImageId, Eigen::Quaterniond>& rotations,
    const ImageId anchor_image_id,
    const double scale_regularization) {
  std::map<ImageId, int> center_offsets;
  int num_center_parameters = 0;
  for (const auto& [image_id, rotation] : rotations) {
    (void)rotation;
    if (image_id != anchor_image_id) {
      center_offsets.emplace(image_id, num_center_parameters);
      num_center_parameters += 3;
    }
  }
  std::vector<int> scale_offsets(constraints.size(), -1);
  int num_active = 0;
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    if (active[edge_idx]) {
      scale_offsets[edge_idx] = num_center_parameters + num_active++;
    }
  }
  PositionSolveResult result;
  result.scales.assign(constraints.size(), 0.0);
  if (num_active == 0 || num_center_parameters == 0) {
    return result;
  }
  const int rows = 3 * num_active + num_active;
  const int cols = num_center_parameters + num_active;
  Eigen::MatrixXd system = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rows);
  int row = 0;
  int prior_row = 3 * num_active;
  const double prior_weight = std::sqrt(std::max(0.0, scale_regularization));
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    if (!active[edge_idx]) {
      continue;
    }
    const RelativePoseConstraint& edge = constraints[edge_idx];
    const Eigen::Vector3d direction =
        -(rotations.at(edge.image_id2).conjugate() *
          edge.cam2_from_cam1.t_cw.normalized());
    for (int axis = 0; axis < 3; ++axis) {
      if (edge.image_id1 != anchor_image_id) {
        system(row + axis, center_offsets.at(edge.image_id1) + axis) = -1.0;
      }
      if (edge.image_id2 != anchor_image_id) {
        system(row + axis, center_offsets.at(edge.image_id2) + axis) = 1.0;
      }
      system(row + axis, scale_offsets[edge_idx]) = -direction[axis];
    }
    system(prior_row, scale_offsets[edge_idx]) = prior_weight;
    rhs[prior_row] = prior_weight;
    row += 3;
    ++prior_row;
  }
  const Eigen::VectorXd solution = system.colPivHouseholderQr().solve(rhs);
  if (!solution.allFinite()) {
    return result;
  }
  result.centers.emplace(anchor_image_id, Eigen::Vector3d::Zero());
  for (const auto& [image_id, offset] : center_offsets) {
    result.centers.emplace(image_id, solution.segment<3>(offset));
  }
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    if (active[edge_idx]) {
      result.scales[edge_idx] = solution[scale_offsets[edge_idx]];
    }
  }
  result.success = true;
  return result;
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

GlobalPositioningSummary EstimateGlobalCameraCenters(
    const std::vector<RelativePoseConstraint>& constraints,
    const std::map<ImageId, Eigen::Quaterniond>& rotations,
    const ImageId anchor_image_id,
    const GlobalPositioningOptions& options,
    std::map<ImageId, Eigen::Vector3d>* camera_centers) {
  if (camera_centers == nullptr) {
    throw std::invalid_argument("camera_centers must not be null");
  }
  camera_centers->clear();
  GlobalPositioningSummary summary;
  if (rotations.count(anchor_image_id) == 0 || rotations.size() < 2) {
    return summary;
  }
  std::vector<bool> active(constraints.size(), false);
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    active[edge_idx] = rotations.count(constraints[edge_idx].image_id1) > 0 &&
                       rotations.count(constraints[edge_idx].image_id2) > 0 &&
                       constraints[edge_idx].cam2_from_cam1.t_cw.norm() > 1e-12;
  }
  PositionSolveResult solved = SolvePositions(
      constraints, active, rotations, anchor_image_id,
      options.scale_regularization);
  if (!solved.success) {
    return summary;
  }
  std::vector<double> residuals;
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    if (!active[edge_idx]) {
      continue;
    }
    const RelativePoseConstraint& edge = constraints[edge_idx];
    const Eigen::Vector3d direction =
        -(rotations.at(edge.image_id2).conjugate() *
          edge.cam2_from_cam1.t_cw.normalized());
    const Eigen::Vector3d delta = solved.centers.at(edge.image_id2) -
                                  solved.centers.at(edge.image_id1);
    const double residual = (delta - solved.scales[edge_idx] * direction).norm();
    if (options.filter_outliers &&
        residual > options.max_direction_residual) {
      active[edge_idx] = false;
      ++summary.num_rejected_constraints;
    } else {
      residuals.push_back(residual);
    }
  }
  if (options.filter_outliers && summary.num_rejected_constraints > 0) {
    solved = SolvePositions(constraints, active, rotations, anchor_image_id,
                            options.scale_regularization);
    if (!solved.success) {
      return summary;
    }
    residuals.clear();
    for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
      if (!active[edge_idx]) {
        continue;
      }
      const RelativePoseConstraint& edge = constraints[edge_idx];
      const Eigen::Vector3d direction =
          -(rotations.at(edge.image_id2).conjugate() *
            edge.cam2_from_cam1.t_cw.normalized());
      const Eigen::Vector3d delta = solved.centers.at(edge.image_id2) -
                                    solved.centers.at(edge.image_id1);
      residuals.push_back(
          (delta - solved.scales[edge_idx] * direction).norm());
    }
  }
  *camera_centers = std::move(solved.centers);
  summary.num_positions = camera_centers->size();
  summary.num_used_constraints =
      static_cast<std::size_t>(std::count(active.begin(), active.end(), true));
  summary.median_direction_residual = Median(std::move(residuals));
  summary.success = summary.num_positions >= 2 &&
                    summary.num_used_constraints > 0;
  return summary;
}

}  // namespace colite
