#include "colite/sfm/rotation_averager.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Eigenvalues>

namespace colite {
namespace {

double QuaternionAngleDeg(const Eigen::Quaterniond& quaternion) {
  const double w = std::clamp(std::abs(quaternion.normalized().w()), 0.0, 1.0);
  return 2.0 * std::acos(w) * 180.0 / 3.14159265358979323846;
}

Eigen::Quaterniond WeightedQuaternionMean(
    const std::vector<std::pair<Eigen::Quaterniond, double>>& samples) {
  Eigen::Matrix4d accumulator = Eigen::Matrix4d::Zero();
  for (const auto& [sample, weight] : samples) {
    const Eigen::Quaterniond q = sample.normalized();
    const Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    accumulator.noalias() += std::max(0.0, weight) * v * v.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(accumulator);
  const Eigen::Vector4d v = solver.eigenvectors().col(3);
  return Eigen::Quaterniond(v[0], v[1], v[2], v[3]).normalized();
}

}  // namespace

RotationAveragingSummary EstimateGlobalRotations(
    const std::vector<RelativePoseConstraint>& constraints,
    const ImageId anchor_image_id,
    const RotationAveragingOptions& options,
    std::map<ImageId, Eigen::Quaterniond>* rotations) {
  if (rotations == nullptr) {
    throw std::invalid_argument("rotations must not be null");
  }
  rotations->clear();
  RotationAveragingSummary summary;
  if (constraints.empty() || options.max_num_iterations == 0) {
    return summary;
  }

  std::unordered_map<ImageId, std::vector<std::size_t>> adjacency;
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    adjacency[constraints[edge_idx].image_id1].push_back(edge_idx);
    adjacency[constraints[edge_idx].image_id2].push_back(edge_idx);
  }
  if (adjacency.count(anchor_image_id) == 0) {
    return summary;
  }
  rotations->emplace(anchor_image_id, Eigen::Quaterniond::Identity());
  std::queue<ImageId> queue;
  queue.push(anchor_image_id);
  while (!queue.empty()) {
    const ImageId current = queue.front();
    queue.pop();
    for (const std::size_t edge_idx : adjacency.at(current)) {
      const RelativePoseConstraint& edge = constraints[edge_idx];
      const bool forward = edge.image_id1 == current;
      const ImageId next = forward ? edge.image_id2 : edge.image_id1;
      if (rotations->count(next) > 0) {
        continue;
      }
      const Eigen::Quaterniond relative = edge.cam2_from_cam1.q_cw.normalized();
      rotations->emplace(
          next, (forward ? relative * rotations->at(current)
                         : relative.conjugate() * rotations->at(current))
                    .normalized());
      queue.push(next);
    }
  }
  summary.num_connected_images = rotations->size();

  std::vector<bool> active(constraints.size(), false);
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    active[edge_idx] = rotations->count(constraints[edge_idx].image_id1) > 0 &&
                       rotations->count(constraints[edge_idx].image_id2) > 0;
  }
  for (std::size_t iteration = 0; iteration < options.max_num_iterations;
       ++iteration) {
    std::map<ImageId, Eigen::Quaterniond> updated = *rotations;
    double max_update = 0.0;
    for (const auto& [image_id, rotation] : *rotations) {
      if (image_id == anchor_image_id) {
        continue;
      }
      std::vector<std::pair<Eigen::Quaterniond, double>> candidates;
      for (const std::size_t edge_idx : adjacency.at(image_id)) {
        if (!active[edge_idx]) {
          continue;
        }
        const RelativePoseConstraint& edge = constraints[edge_idx];
        const Eigen::Quaterniond relative =
            edge.cam2_from_cam1.q_cw.normalized();
        if (edge.image_id2 == image_id) {
          candidates.emplace_back(relative * rotations->at(edge.image_id1),
                                  edge.weight);
        } else {
          candidates.emplace_back(relative.conjugate() *
                                      rotations->at(edge.image_id2),
                                  edge.weight);
        }
      }
      if (!candidates.empty()) {
        updated[image_id] = WeightedQuaternionMean(candidates);
        max_update = std::max(
            max_update,
            QuaternionAngleDeg(updated[image_id] * rotation.conjugate()));
      }
    }
    *rotations = std::move(updated);
    summary.num_iterations = iteration + 1;
    summary.max_update_deg = max_update;
    if (max_update <= options.convergence_threshold_deg) {
      break;
    }
  }

  std::vector<double> residuals;
  for (std::size_t edge_idx = 0; edge_idx < constraints.size(); ++edge_idx) {
    if (!active[edge_idx]) {
      continue;
    }
    const RelativePoseConstraint& edge = constraints[edge_idx];
    const Eigen::Quaterniond predicted =
        rotations->at(edge.image_id2) *
        rotations->at(edge.image_id1).conjugate();
    const double residual = QuaternionAngleDeg(
        predicted * edge.cam2_from_cam1.q_cw.normalized().conjugate());
    if (residual <= options.max_relative_rotation_error_deg) {
      ++summary.num_used_constraints;
      residuals.push_back(residual);
    } else {
      ++summary.num_rejected_constraints;
    }
  }
  if (!residuals.empty()) {
    const std::size_t middle = residuals.size() / 2;
    std::nth_element(residuals.begin(), residuals.begin() + middle,
                     residuals.end());
    summary.median_residual_deg = residuals[middle];
  }
  summary.success = rotations->size() >= 2 && summary.num_used_constraints > 0;
  return summary;
}

}  // namespace colite
