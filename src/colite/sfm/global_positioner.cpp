#include "colite/sfm/global_positioner.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>

#include <Eigen/SparseQR>

namespace colite {
namespace {

struct PositionSolveResult {
  bool success = false;
  std::map<ImageId, Eigen::Vector3d> centers;
  std::vector<double> scales;
};

void PruneDisconnected(
    const std::vector<RelativePoseConstraint>& constraints,
    const ImageId anchor, std::vector<bool>* active) {
  std::map<ImageId, std::vector<ImageId>> adjacency;
  for (std::size_t idx = 0; idx < constraints.size(); ++idx) {
    if (!(*active)[idx]) continue;
    const auto& edge = constraints[idx];
    adjacency[edge.image_id1].push_back(edge.image_id2);
    adjacency[edge.image_id2].push_back(edge.image_id1);
  }
  std::set<ImageId> connected{anchor};
  std::queue<ImageId> pending;
  pending.push(anchor);
  while (!pending.empty()) {
    const ImageId current = pending.front();
    pending.pop();
    for (const ImageId next : adjacency[current]) {
      if (connected.insert(next).second) pending.push(next);
    }
  }
  for (std::size_t idx = 0; idx < constraints.size(); ++idx) {
    (*active)[idx] = (*active)[idx] &&
        connected.count(constraints[idx].image_id1) > 0 &&
        connected.count(constraints[idx].image_id2) > 0;
  }
}

PositionSolveResult SolvePositions(
    const std::vector<RelativePoseConstraint>& constraints,
    const std::vector<bool>& active,
    const std::map<ImageId, Eigen::Quaterniond>& rotations,
    const ImageId anchor_image_id,
    const double scale_regularization) {
  std::map<ImageId, int> center_offsets;
  int num_center_parameters = 0;
  std::set<ImageId> connected{anchor_image_id};
  for (std::size_t idx = 0; idx < constraints.size(); ++idx) {
    if (active[idx]) {
      connected.insert(constraints[idx].image_id1);
      connected.insert(constraints[idx].image_id2);
    }
  }
  for (const ImageId image_id : connected) {
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
  Eigen::SparseMatrix<double> system(rows, cols);
  std::vector<Eigen::Triplet<double>> coefficients;
  coefficients.reserve(static_cast<std::size_t>(num_active) * 10);
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
        coefficients.emplace_back(
            row + axis, center_offsets.at(edge.image_id1) + axis, -1.0);
      }
      if (edge.image_id2 != anchor_image_id) {
        coefficients.emplace_back(
            row + axis, center_offsets.at(edge.image_id2) + axis, 1.0);
      }
      if (direction[axis] != 0.0) {
        coefficients.emplace_back(
            row + axis, scale_offsets[edge_idx], -direction[axis]);
      }
    }
    coefficients.emplace_back(prior_row, scale_offsets[edge_idx], prior_weight);
    rhs[prior_row] = prior_weight;
    row += 3;
    ++prior_row;
  }
  system.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  solver.compute(system);
  if (solver.info() != Eigen::Success || solver.rank() != cols) {
    return result;
  }
  const Eigen::VectorXd solution = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !solution.allFinite()) {
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
  if (!std::isfinite(options.scale_regularization) ||
      options.scale_regularization < 0.0 ||
      !std::isfinite(options.max_direction_residual) ||
      options.max_direction_residual < 0.0) {
    throw std::invalid_argument("Invalid global positioning options");
  }
  if (rotations.count(anchor_image_id) == 0 || rotations.size() < 2) {
    return summary;
  }
  std::vector<bool> active(constraints.size(), false);
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    active[edge_idx] = rotations.count(constraints[edge_idx].image_id1) > 0 &&
                       rotations.count(constraints[edge_idx].image_id2) > 0 &&
                       constraints[edge_idx].image_id1 != constraints[edge_idx].image_id2 &&
                       rotations.at(constraints[edge_idx].image_id1).coeffs().allFinite() &&
                       rotations.at(constraints[edge_idx].image_id2).coeffs().allFinite() &&
                       std::abs(rotations.at(constraints[edge_idx].image_id1).norm() - 1.0) < 1e-6 &&
                       std::abs(rotations.at(constraints[edge_idx].image_id2).norm() - 1.0) < 1e-6 &&
                       constraints[edge_idx].cam2_from_cam1.t_cw.allFinite() &&
                       constraints[edge_idx].cam2_from_cam1.t_cw.norm() > 1e-12;
  }
  PruneDisconnected(constraints, anchor_image_id, &active);
  std::vector<double> residuals;
  PositionSolveResult solved;
  for (;;) {
    solved = SolvePositions(
        constraints, active, rotations, anchor_image_id,
        options.scale_regularization);
    if (!solved.success) {
      return summary;
    }
    residuals.clear();
    bool rejected = false;
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
      if (solved.scales[edge_idx] <= 1e-12 || delta.norm() <= 1e-12 ||
          (options.filter_outliers && residual > options.max_direction_residual)) {
        active[edge_idx] = false;
        rejected = true;
      } else {
        residuals.push_back(residual);
      }
    }
    if (!rejected) break;
    PruneDisconnected(constraints, anchor_image_id, &active);
  }
  *camera_centers = std::move(solved.centers);
  summary.num_positions = camera_centers->size();
  summary.num_used_constraints =
      static_cast<std::size_t>(std::count(active.begin(), active.end(), true));
  summary.num_rejected_constraints = constraints.size() - summary.num_used_constraints;
  summary.median_direction_residual = Median(std::move(residuals));
  summary.success = summary.num_positions >= 2 &&
                    summary.num_used_constraints > 0;
  return summary;
}

}  // namespace colite
