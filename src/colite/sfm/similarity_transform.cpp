#include "colite/sfm/similarity_transform.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <set>
#include <stdexcept>

#include <Eigen/Geometry>

namespace colite {

Eigen::Vector3d SimilarityTransform3::TransformPoint(
    const Eigen::Vector3d& point) const {
  return scale * rotation * point + translation;
}

Pose3 SimilarityTransform3::TransformPose(const Pose3& pose) const {
  Pose3 transformed;
  const Eigen::Matrix3d target_rotation =
      pose.q_cw.toRotationMatrix() * rotation.transpose();
  transformed.q_cw = Eigen::Quaterniond(target_rotation).normalized();
  transformed.t_cw =
      scale * pose.t_cw - target_rotation * translation;
  return transformed;
}

SimilarityTransform3 EstimateSimilarityTransform(
    const std::vector<Eigen::Vector3d>& source_points,
    const std::vector<Eigen::Vector3d>& target_points,
    const bool estimate_scale) {
  SimilarityTransform3 transform;
  if (source_points.size() != target_points.size() ||
      source_points.size() < 3) {
    return transform;
  }
  Eigen::MatrixXd source(3, source_points.size());
  Eigen::MatrixXd target(3, target_points.size());
  for (std::size_t idx = 0; idx < source_points.size(); ++idx) {
    if (!source_points[idx].allFinite() || !target_points[idx].allFinite()) {
      return transform;
    }
    source.col(idx) = source_points[idx];
    target.col(idx) = target_points[idx];
  }
  const Eigen::Matrix4d matrix = Eigen::umeyama(source, target, estimate_scale);
  if (!matrix.allFinite()) return transform;

  const Eigen::Matrix3d scaled_rotation = matrix.block<3, 3>(0, 0);
  const double scale = estimate_scale
                           ? (scaled_rotation.col(0).norm() +
                              scaled_rotation.col(1).norm() +
                              scaled_rotation.col(2).norm()) /
                                 3.0
                           : 1.0;
  if (!std::isfinite(scale) || scale <= 1e-12) return transform;
  Eigen::Matrix3d rotation = scaled_rotation / scale;
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
  rotation = svd.matrixU() * svd.matrixV().transpose();
  if (!rotation.allFinite() || rotation.determinant() <= 0.0) return transform;

  transform.scale = scale;
  transform.rotation = rotation;
  transform.translation = matrix.block<3, 1>(0, 3);
  transform.valid = transform.translation.allFinite();
  return transform;
}

void TransformReconstruction(const SimilarityTransform3& transform,
                             Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    throw std::invalid_argument("Reconstruction is null");
  }
  if (!transform.valid) {
    throw std::invalid_argument("Similarity transform is invalid");
  }
  for (const auto& [point_id, point] : reconstruction->Points3D()) {
    reconstruction->GetPoint3D(point_id).xyz =
        transform.TransformPoint(point.xyz);
  }
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    if (image.cam_from_world.has_value()) {
      image.cam_from_world = transform.TransformPose(*image.cam_from_world);
    }
  }
}

PositionPriorAlignmentSummary AlignReconstructionToPositionPriors(
    const double max_error,
    Reconstruction* reconstruction) {
  PositionPriorAlignmentSummary summary;
  if (reconstruction == nullptr) return summary;
  std::vector<Eigen::Vector3d> source;
  std::vector<Eigen::Vector3d> target;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    const Image& image = reconstruction->GetImage(image_id);
    if (!image.cam_from_world.has_value() ||
        !image.prior_position.has_value()) {
      continue;
    }
    source.push_back(image.cam_from_world->CameraCenter());
    target.push_back(*image.prior_position);
  }
  summary.num_priors = source.size();
  SimilarityTransform3 transform =
      EstimateSimilarityTransform(source, target, true);
  if (!transform.valid) return summary;

  std::vector<Eigen::Vector3d> inlier_source;
  std::vector<Eigen::Vector3d> inlier_target;
  const double threshold = std::isfinite(max_error) && max_error > 0.0
                               ? max_error
                               : std::numeric_limits<double>::infinity();
  const auto collect_inliers = [&](const SimilarityTransform3& candidate) {
    std::vector<std::size_t> indices;
    if (!candidate.valid) return indices;
    for (std::size_t idx = 0; idx < source.size(); ++idx) {
      if ((candidate.TransformPoint(source[idx]) - target[idx]).norm() <=
          threshold) {
        indices.push_back(idx);
      }
    }
    return indices;
  };
  std::vector<std::size_t> best_inliers = collect_inliers(transform);
  if (source.size() > 3 && best_inliers.size() < source.size()) {
    std::uint64_t state = 0x243f6a8885a308d3ULL;
    const std::size_t hypotheses = std::min<std::size_t>(512, source.size() * 8);
    for (std::size_t hypothesis = 0; hypothesis < hypotheses; ++hypothesis) {
      std::set<std::size_t> sample;
      while (sample.size() < 3) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        sample.insert(state % source.size());
      }
      std::vector<Eigen::Vector3d> sample_source;
      std::vector<Eigen::Vector3d> sample_target;
      for (const std::size_t idx : sample) {
        sample_source.push_back(source[idx]);
        sample_target.push_back(target[idx]);
      }
      const SimilarityTransform3 candidate =
          EstimateSimilarityTransform(sample_source, sample_target, true);
      const std::vector<std::size_t> inliers = collect_inliers(candidate);
      if (inliers.size() > best_inliers.size()) {
        transform = candidate;
        best_inliers = inliers;
      }
    }
  }
  for (const std::size_t idx : best_inliers) {
    inlier_source.push_back(source[idx]);
    inlier_target.push_back(target[idx]);
  }
  if (inlier_source.size() >= 3 && inlier_source.size() < source.size()) {
    const SimilarityTransform3 refined =
        EstimateSimilarityTransform(inlier_source, inlier_target, true);
    if (refined.valid) transform = refined;
  }
  if (inlier_source.size() < 3) return summary;

  TransformReconstruction(transform, reconstruction);
  double total_error = 0.0;
  std::size_t count = 0;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    const Image& image = reconstruction->GetImage(image_id);
    if (image.cam_from_world.has_value() && image.prior_position.has_value()) {
      const double error =
          (image.cam_from_world->CameraCenter() - *image.prior_position).norm();
      if (error <= threshold) {
        total_error += error;
        ++count;
      }
    }
  }
  summary.success = true;
  summary.num_inliers = count;
  summary.scale = transform.scale;
  summary.mean_error = count == 0 ? -1.0 : total_error / count;
  return summary;
}

}  // namespace colite
