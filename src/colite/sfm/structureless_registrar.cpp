#include "colite/sfm/structureless_registrar.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include <Eigen/Eigenvalues>
#include "colite/core/geometry.h"

namespace colite {
namespace {
struct PoseRay {
  Eigen::Quaterniond rotation;
  Eigen::Vector3d origin, direction;
};
double RotationDistanceDeg(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  return 2 * std::acos(std::clamp(std::abs(a.dot(b)), 0.0, 1.0)) * 180 / 3.14159265358979323846;
}
double Median(std::vector<double> values) {
  if (values.empty()) return -1;
  const auto middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  return values[middle];
}
bool IntersectRays(const std::vector<PoseRay>& rays,
                   const std::vector<std::size_t>& indices, Eigen::Vector3d* center) {
  Eigen::Matrix3d a = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  for (auto idx : indices) {
    const auto& ray = rays[idx];
    const Eigen::Matrix3d p = Eigen::Matrix3d::Identity() - ray.direction * ray.direction.transpose();
    a += p;
    b += p * ray.origin;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(a);
  // Parallel rays cannot determine position along their common baseline.
  if (solver.info() != Eigen::Success ||
      solver.eigenvalues().minCoeff() <= 1e-6 * solver.eigenvalues().maxCoeff()) return false;
  *center = solver.eigenvectors() * solver.eigenvalues().cwiseInverse().asDiagonal() *
            solver.eigenvectors().transpose() * b;
  return center->allFinite();
}
std::vector<std::size_t> FindSupport(const std::vector<PoseRay>& rays,
    const Eigen::Quaterniond& rotation, const Eigen::Vector3d& center,
    const StructurelessRegistrationOptions& options) {
  std::vector<std::size_t> support;
  for (std::size_t idx = 0; idx < rays.size(); ++idx) {
    const auto& ray = rays[idx];
    const Eigen::Vector3d delta = center - ray.origin;
    const double distance = delta.dot(ray.direction);
    if (distance > 1e-8 &&
        (delta - distance * ray.direction).norm() <= options.max_center_disagreement &&
        RotationDistanceDeg(rotation, ray.rotation) <= options.max_rotation_disagreement_deg)
      support.push_back(idx);
  }
  return support;
}
}  // namespace

StructurelessRegistrationSummary EstimateStructurelessPose(
    const MatchGraph& graph, ImageId image_id, const Reconstruction& reconstruction,
    const StructurelessRegistrationOptions& options) {
  if (options.min_num_registered_neighbors < 2 || options.min_num_inliers_per_edge == 0 ||
      !std::isfinite(options.max_rotation_disagreement_deg) || options.max_rotation_disagreement_deg <= 0 ||
      !std::isfinite(options.max_center_disagreement) || options.max_center_disagreement <= 0)
    throw std::invalid_argument("Invalid structure-less registration options");
  StructurelessRegistrationSummary summary;
  if (!reconstruction.ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }
  std::vector<PoseRay> rays;
  for (const auto& edge : FindVerifiedNeighbors(graph, image_id)) {
    if (!edge.geometry || edge.geometry->type != TwoViewGeometry::Type::kCalibratedEssential ||
        edge.geometry->inlier_matches.size() < options.min_num_inliers_per_edge ||
        !edge.geometry->cam2_from_cam1.q_cw.coeffs().allFinite() ||
        edge.geometry->cam2_from_cam1.q_cw.norm() <= 1e-12 ||
        !edge.geometry->cam2_from_cam1.t_cw.allFinite() ||
        edge.geometry->cam2_from_cam1.t_cw.norm() <= 1e-12) continue;
    const auto neighbor_id = edge.OtherImageId(image_id);
    if (!reconstruction.IsImageRegistered(neighbor_id)) continue;
    const auto& neighbor = reconstruction.GetImage(neighbor_id);
    if (!neighbor.cam_from_world ||
        !neighbor.cam_from_world->q_cw.coeffs().allFinite() ||
        std::abs(neighbor.cam_from_world->q_cw.norm() - 1) > 1e-6 ||
        !neighbor.cam_from_world->t_cw.allFinite()) continue;
    Pose3 relative = edge.geometry->cam2_from_cam1;
    relative.q_cw.normalize();
    if (edge.pair.image_id1 == image_id) relative = relative.Inverse();
    const auto rotation = (relative.q_cw * neighbor.cam_from_world->q_cw).normalized();
    rays.push_back({rotation, neighbor.cam_from_world->CameraCenter(),
                   -(rotation.conjugate() * relative.t_cw.normalized())});
  }
  summary.num_pose_hypotheses = rays.size();
  if (rays.size() < options.min_num_registered_neighbors) {
    summary.message = "Not enough registered view-graph neighbors";
    return summary;
  }
  // Normalize all positions by one neighbor baseline to preserve Sim(3) scale.
  const Eigen::Vector3d origin = rays.front().origin;
  double baseline = 0;
  for (const auto& ray : rays) baseline = std::max(baseline, (ray.origin - origin).norm());
  if (!std::isfinite(baseline) || baseline <= 1e-12) {
    summary.message = "Registered neighbors have no usable baseline";
    return summary;
  }
  for (auto& ray : rays) ray.origin = (ray.origin - origin) / baseline;
  std::vector<std::size_t> best_support;
  Eigen::Vector3d best_center = Eigen::Vector3d::Zero();
  Eigen::Quaterniond best_rotation = Eigen::Quaterniond::Identity();
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t lhs = 0; lhs < rays.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < rays.size(); ++rhs) {
      if (RotationDistanceDeg(rays[lhs].rotation, rays[rhs].rotation) >
          options.max_rotation_disagreement_deg) continue;
      Eigen::Vector3d center;
      if (!IntersectRays(rays, {lhs, rhs}, &center)) continue;
      const Eigen::Quaterniond rotation = rays[lhs].rotation.slerp(0.5, rays[rhs].rotation);
      auto support = FindSupport(rays, rotation, center, options);
      double cost = 0;
      for (auto idx : support) {
        const Eigen::Vector3d delta = center - rays[idx].origin;
        cost += (delta - delta.dot(rays[idx].direction) * rays[idx].direction).norm();
      }
      if (support.size() > best_support.size() ||
          (support.size() == best_support.size() && cost < best_cost)) {
        best_support = std::move(support);
        best_center = center;
        best_rotation = rotation;
        best_cost = cost;
      }
    }
  }
  if (best_support.size() < options.min_num_registered_neighbors ||
      !IntersectRays(rays, best_support, &best_center)) {
    summary.message = "Relative-pose rays are degenerate or inconsistent";
    return summary;
  }
  best_support = FindSupport(rays, best_rotation, best_center, options);
  if (best_support.size() < options.min_num_registered_neighbors) {
    summary.message = "Refined relative-pose rays are inconsistent";
    return summary;
  }
  std::vector<double> rotation_errors, center_errors;
  for (auto idx : best_support) {
    rotation_errors.push_back(RotationDistanceDeg(best_rotation, rays[idx].rotation));
    const Eigen::Vector3d delta = best_center - rays[idx].origin;
    center_errors.push_back((delta - delta.dot(rays[idx].direction) * rays[idx].direction).norm());
  }
  summary.pose.q_cw = best_rotation;
  summary.pose.t_cw = -(best_rotation * (origin + baseline * best_center));
  summary.num_consistent_hypotheses = best_support.size();
  summary.median_rotation_disagreement_deg = Median(std::move(rotation_errors));
  summary.median_center_disagreement = Median(std::move(center_errors));
  summary.success = summary.pose.t_cw.allFinite();
  summary.message = "Scale-independent relative-pose rays intersected";
  return summary;
}

std::size_t ValidateStructurelessPose(
    const MatchGraph& graph, ImageId image_id, const Reconstruction& reconstruction,
    const Pose3& pose, const StructurelessRegistrationOptions& options,
    double max_reprojection_error_px) {
  const auto* target_features = FindFeatures(graph, image_id);
  if (!target_features || !reconstruction.ExistsImage(image_id) ||
      !std::isfinite(max_reprojection_error_px) || max_reprojection_error_px <= 0) return 0;
  const auto& target_camera = reconstruction.GetCamera(reconstruction.GetImage(image_id).camera_id);
  std::set<Point2DIdx> validated;
  std::size_t supported_neighbors = 0;
  for (const auto& edge : FindVerifiedNeighbors(graph, image_id)) {
    const auto neighbor_id = edge.OtherImageId(image_id);
    if (!edge.geometry || edge.geometry->type != TwoViewGeometry::Type::kCalibratedEssential ||
        !reconstruction.IsImageRegistered(neighbor_id)) continue;
    const auto* neighbor_features = FindFeatures(graph, neighbor_id);
    const auto& neighbor = reconstruction.GetImage(neighbor_id);
    if (!neighbor_features || !neighbor.cam_from_world) continue;
    const auto& neighbor_camera = reconstruction.GetCamera(neighbor.camera_id);
    std::set<Point2DIdx> valid_for_neighbor, used_neighbor_indices;
    for (const auto& match : edge.geometry->inlier_matches) {
      const bool target_first = edge.pair.image_id1 == image_id;
      const auto target_idx = target_first ? match.idx1 : match.idx2;
      const auto neighbor_idx = target_first ? match.idx2 : match.idx1;
      if (target_idx >= target_features->keypoints.size() ||
          neighbor_idx >= neighbor_features->keypoints.size() ||
          used_neighbor_indices.count(neighbor_idx)) continue;
      const auto& tk = target_features->keypoints[target_idx];
      const auto& nk = neighbor_features->keypoints[neighbor_idx];
      const Eigen::Vector2d tp(tk.x, tk.y), np(nk.x, nk.y);
      if (!tp.allFinite() || !np.allFinite()) continue;
      const auto point = TriangulatePoint(target_camera, pose, tp,
          neighbor_camera, *neighbor.cam_from_world, np);
      if (!point || !point->allFinite()) continue;
      const auto te = ReprojectionError(target_camera, pose, *point, tp);
      const auto ne = ReprojectionError(neighbor_camera, *neighbor.cam_from_world, *point, np);
      if (te && ne && std::isfinite(*te) && std::isfinite(*ne) &&
          *te <= max_reprojection_error_px && *ne <= max_reprojection_error_px &&
          TriangulationAngleDegrees(pose, PixelToBearing(target_camera, tp),
              *neighbor.cam_from_world, PixelToBearing(neighbor_camera, np)) >= 0.1) {
        valid_for_neighbor.insert(target_idx);
        used_neighbor_indices.insert(neighbor_idx);
      }
    }
    if (valid_for_neighbor.size() >= options.min_num_inliers_per_edge) {
      ++supported_neighbors;
      validated.insert(valid_for_neighbor.begin(), valid_for_neighbor.end());
    }
  }
  return supported_neighbors >= options.min_num_registered_neighbors ? validated.size() : 0;
}
}  // namespace colite
