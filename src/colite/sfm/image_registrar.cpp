#include "colite/sfm/image_registrar.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "colite/core/geometry.h"
#include "colite/sfm/observation_manager.h"

#ifdef COLITE_HAS_POSELIB
#include <PoseLib/solvers/p3p.h>
#endif

namespace colite {
namespace {

constexpr std::size_t kMinDltPnPPoints = 6;
#ifdef COLITE_HAS_POSELIB
constexpr std::size_t kMinPnPPoints = 3;
#else
constexpr std::size_t kMinPnPPoints = kMinDltPnPPoints;
#endif
constexpr double kMinProjectionScale = 1e-12;

struct PnPCorrespondence {
  Point3DId point3d_id = kInvalidPoint3DId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
};

void SyncImagePoints2DFromFeatures(const FeatureSet& features, Image* image) {
  if (image->points2d.size() < features.keypoints.size()) {
    image->points2d.resize(features.keypoints.size());
  }

  for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
    const FeatureKeypoint& keypoint = features.keypoints[i];
    Point2D& point2d = image->points2d[i];
    point2d.xy = Eigen::Vector2d(keypoint.x, keypoint.y);
    point2d.feature_idx = static_cast<Point2DIdx>(i);
  }
}

bool PairContainsImage(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id || pair.image_id2 == image_id;
}

ImageId OtherImageId(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id ? pair.image_id2 : pair.image_id1;
}

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

void AddCorrespondenceIfValid(const FeatureMatch& match,
                              bool target_is_first,
                              Image& target_image,
                              const Image& neighbor_image,
                              const Reconstruction& reconstruction,
                              std::map<Point2DIdx, PnPCorrespondence>*
                                  correspondences_by_target_idx) {
  const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
  const Point2DIdx neighbor_idx = target_is_first ? match.idx2 : match.idx1;
  if (target_idx == kInvalidPoint2DIdx ||
      neighbor_idx == kInvalidPoint2DIdx ||
      target_idx >= target_image.points2d.size() ||
      neighbor_idx >= neighbor_image.points2d.size()) {
    return;
  }

  const Point2D& neighbor_point2d = neighbor_image.points2d[neighbor_idx];
  if (!neighbor_point2d.HasPoint3D() ||
      !reconstruction.ExistsPoint3D(neighbor_point2d.point3d_id)) {
    return;
  }
  if (target_image.points2d[target_idx].HasPoint3D() ||
      correspondences_by_target_idx->find(target_idx) !=
          correspondences_by_target_idx->end()) {
    return;
  }

  const Point3D& point3d =
      reconstruction.GetPoint3D(neighbor_point2d.point3d_id);
  correspondences_by_target_idx->emplace(
      target_idx,
      PnPCorrespondence{point3d.id, target_idx, point3d.xyz,
                        target_image.points2d[target_idx].xy});
}

std::vector<PnPCorrespondence> CollectPnPCorrespondences(
    const MatchGraph& graph,
    ImageId image_id,
    Reconstruction* reconstruction) {
  std::vector<PnPCorrespondence> correspondences;
  if (!reconstruction->ExistsImage(image_id) ||
      reconstruction->IsImageRegistered(image_id)) {
    return correspondences;
  }

  const FeatureSet* features = FindFeatures(graph, image_id);
  if (features == nullptr) {
    return correspondences;
  }

  Image& target_image = reconstruction->GetImage(image_id);
  SyncImagePoints2DFromFeatures(*features, &target_image);

  std::map<Point2DIdx, PnPCorrespondence> unique_correspondences;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction->IsImageRegistered(neighbor_id)) {
      continue;
    }

    const Image& neighbor_image = reconstruction->GetImage(neighbor_id);
    const bool target_is_first = pair.image_id1 == image_id;
    for (const FeatureMatch& match : geometry.inlier_matches) {
      AddCorrespondenceIfValid(match, target_is_first, target_image,
                               neighbor_image, *reconstruction,
                               &unique_correspondences);
    }
  }

  correspondences.reserve(unique_correspondences.size());
  for (const auto& [target_idx, correspondence] : unique_correspondences) {
    (void)target_idx;
    correspondences.push_back(correspondence);
  }
  return correspondences;
}

std::vector<std::size_t> AllIndices(const std::size_t count) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

std::vector<std::size_t> SampleUniqueIndices(std::mt19937* rng,
                                             const std::size_t count,
                                             const std::size_t sample_size) {
  std::vector<std::size_t> indices = AllIndices(count);
  std::shuffle(indices.begin(), indices.end(), *rng);
  indices.resize(sample_size);
  return indices;
}

Eigen::Vector2d NormalizedImagePoint(const Camera& camera,
                                     const Eigen::Vector2d& pixel) {
  const Eigen::Vector3d ray = UnprojectPixel(camera, pixel, 1.0);
  return {ray.x() / ray.z(), ray.y() / ray.z()};
}

std::optional<Pose3> PoseFromProjectionMatrix(
    Eigen::Matrix<double, 3, 4> projection,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& indices) {
  Pose3 best_pose;
  std::size_t best_positive_depths = 0;

  for (const double sign : {1.0, -1.0}) {
    const Eigen::Matrix<double, 3, 4> signed_projection = sign * projection;
    const Eigen::Matrix3d m = signed_projection.leftCols<3>();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
    if (rotation.determinant() < 0.0) {
      continue;
    }

    const double scale = svd.singularValues().mean();
    if (scale <= 0.0 || !std::isfinite(scale)) {
      continue;
    }

    Pose3 pose;
    pose.q_cw = Eigen::Quaterniond(rotation).normalized();
    pose.t_cw = signed_projection.col(3) / scale;

    std::size_t positive_depths = 0;
    for (const std::size_t idx : indices) {
      if (pose.TransformPoint(correspondences[idx].xyz).z() > 0.0) {
        ++positive_depths;
      }
    }

    if (positive_depths > best_positive_depths) {
      best_positive_depths = positive_depths;
      best_pose = pose;
    }
  }

  if (best_positive_depths == 0) {
    return std::nullopt;
  }
  return best_pose;
}

std::optional<Pose3> EstimatePoseDlt(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& indices) {
  if (indices.size() < kMinDltPnPPoints) {
    return std::nullopt;
  }

  Eigen::MatrixXd a(static_cast<Eigen::Index>(2 * indices.size()), 12);
  for (std::size_t row = 0; row < indices.size(); ++row) {
    const PnPCorrespondence& correspondence = correspondences[indices[row]];
    const Eigen::Vector2d xy =
        NormalizedImagePoint(camera, correspondence.xy);
    const Eigen::Vector4d xyz_h = correspondence.xyz.homogeneous();

    a.row(static_cast<Eigen::Index>(2 * row)).setZero();
    a.row(static_cast<Eigen::Index>(2 * row)).segment<4>(4) = -xyz_h;
    a.row(static_cast<Eigen::Index>(2 * row)).segment<4>(8) = xy.y() * xyz_h;

    a.row(static_cast<Eigen::Index>(2 * row + 1)).setZero();
    a.row(static_cast<Eigen::Index>(2 * row + 1)).segment<4>(0) = xyz_h;
    a.row(static_cast<Eigen::Index>(2 * row + 1)).segment<4>(8) =
        -xy.x() * xyz_h;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  Eigen::Matrix<double, 3, 4> projection =
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(
          svd.matrixV().col(11).data());
  if (projection.leftCols<3>().norm() <= kMinProjectionScale) {
    return std::nullopt;
  }
  return PoseFromProjectionMatrix(projection, correspondences, indices);
}

std::vector<std::size_t> InliersForPose(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const Pose3& pose,
    const double max_reprojection_error_px) {
  std::vector<std::size_t> inliers;
  inliers.reserve(correspondences.size());
  for (std::size_t idx = 0; idx < correspondences.size(); ++idx) {
    const std::optional<double> error =
        ReprojectionError(camera, pose, correspondences[idx].xyz,
                          correspondences[idx].xy);
    if (error && *error <= max_reprojection_error_px) {
      inliers.push_back(idx);
    }
  }
  return inliers;
}

struct PnPRansacResult {
  Pose3 pose;
  std::vector<std::size_t> inlier_indices;
  std::size_t num_inliers_before_final_refinement = 0;
};

std::vector<Pose3> EstimateMinimalPoses(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& indices) {
  std::vector<Pose3> poses;
#ifdef COLITE_HAS_POSELIB
  if (indices.size() != 3) {
    return poses;
  }
  std::vector<Eigen::Vector3d> rays;
  std::vector<Eigen::Vector3d> points3d;
  rays.reserve(3);
  points3d.reserve(3);
  for (const std::size_t idx : indices) {
    rays.push_back(
        NormalizedImagePoint(camera, correspondences[idx].xy)
            .homogeneous()
            .normalized());
    points3d.push_back(correspondences[idx].xyz);
  }
  std::vector<poselib::CameraPose> solutions;
  poselib::p3p(rays, points3d, &solutions);
  poses.reserve(solutions.size());
  for (const poselib::CameraPose& solution : solutions) {
    const Eigen::Matrix<double, 3, 4> matrix = solution.Rt();
    Pose3 pose;
    pose.q_cw = Eigen::Quaterniond(matrix.leftCols<3>()).normalized();
    pose.t_cw = matrix.col(3);
    poses.push_back(pose);
  }
#else
  const std::optional<Pose3> pose =
      EstimatePoseDlt(camera, correspondences, indices);
  if (pose) {
    poses.push_back(*pose);
  }
#endif
  return poses;
}

PnPRansacResult EstimatePoseRansac(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const ImageRegistrarOptions& options) {
  PnPRansacResult best;
  if (correspondences.size() < kMinPnPPoints) {
    return best;
  }

  std::mt19937 rng(0);
  const int min_iterations = std::max(1, options.min_num_iterations);
  const int max_iterations =
      std::max(min_iterations, options.max_num_iterations);
  int required_iterations = max_iterations;
  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    const std::vector<std::size_t> sample =
        correspondences.size() == kMinPnPPoints
            ? AllIndices(correspondences.size())
            : SampleUniqueIndices(&rng, correspondences.size(),
                                  kMinPnPPoints);
    for (const Pose3& pose :
         EstimateMinimalPoses(camera, correspondences, sample)) {
      std::vector<std::size_t> inliers = InliersForPose(
          camera, correspondences, pose, options.max_reprojection_error_px);
      if (inliers.size() > best.inlier_indices.size()) {
        best.pose = pose;
        best.inlier_indices = std::move(inliers);

        // Local optimization uses all current inliers. The Eigen DLT solver
        // is non-minimal here and keeps builds without PoseLib functional.
        if (best.inlier_indices.size() >= kMinDltPnPPoints) {
          const std::optional<Pose3> local =
              EstimatePoseDlt(camera, correspondences, best.inlier_indices);
          if (local) {
            std::vector<std::size_t> local_inliers = InliersForPose(
                camera, correspondences, *local,
                options.max_reprojection_error_px);
            if (local_inliers.size() >= best.inlier_indices.size()) {
              best.pose = *local;
              best.inlier_indices = std::move(local_inliers);
            }
          }
        }

        if (best.inlier_indices.size() == correspondences.size()) {
          best.num_inliers_before_final_refinement =
              best.inlier_indices.size();
          return best;
        }
        const double ratio =
            static_cast<double>(best.inlier_indices.size()) /
            correspondences.size();
        const double sample_success =
            std::pow(ratio, static_cast<double>(kMinPnPPoints));
        if (sample_success > 0.0 && sample_success < 1.0) {
          const double confidence =
              std::clamp(options.confidence, 0.0, 1.0 - 1e-12);
          const double estimate = std::log(1.0 - confidence) /
                                  std::log(1.0 - sample_success);
          if (std::isfinite(estimate)) {
            required_iterations = std::clamp(
                static_cast<int>(std::ceil(estimate)), min_iterations,
                max_iterations);
            required_iterations =
                std::max(required_iterations, iteration + 1);
          }
        }
      }
    }
  }

  best.num_inliers_before_final_refinement = best.inlier_indices.size();
  if (best.inlier_indices.size() >= kMinDltPnPPoints) {
    const std::optional<Pose3> refined =
        EstimatePoseDlt(camera, correspondences, best.inlier_indices);
    if (refined) {
      std::vector<std::size_t> refined_inliers =
          InliersForPose(camera, correspondences, *refined,
                         options.max_reprojection_error_px);
      if (refined_inliers.size() >= best.inlier_indices.size()) {
        best.pose = *refined;
        best.inlier_indices = std::move(refined_inliers);
      }
    }
  }

  return best;
}

void LinkInlierCorrespondencesToImage(
    ImageId image_id,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& inlier_indices,
    Reconstruction* reconstruction) {
  const Image& image = reconstruction->GetImage(image_id);
  const Camera& camera = reconstruction->GetCamera(image.camera_id);
  std::map<Point3DId, std::pair<std::size_t, double>> best_by_point;
  for (const std::size_t correspondence_idx : inlier_indices) {
    const PnPCorrespondence& correspondence =
        correspondences[correspondence_idx];
    if (correspondence.point2d_idx >= image.points2d.size() ||
        !reconstruction->ExistsPoint3D(correspondence.point3d_id)) {
      continue;
    }
    const double error =
        ReprojectionError(
            camera, *image.cam_from_world,
            reconstruction->GetPoint3D(correspondence.point3d_id).xyz,
            image.points2d[correspondence.point2d_idx].xy)
            .value_or(std::numeric_limits<double>::infinity());
    const auto best_it = best_by_point.find(correspondence.point3d_id);
    if (best_it == best_by_point.end() || error < best_it->second.second ||
        (error == best_it->second.second &&
         correspondence.point2d_idx <
             correspondences[best_it->second.first].point2d_idx)) {
      best_by_point[correspondence.point3d_id] = {correspondence_idx, error};
    }
  }

  ObservationManager observation_manager(reconstruction);
  for (const auto& [point3d_id, selected] : best_by_point) {
    const PnPCorrespondence& correspondence =
        correspondences[selected.first];
    observation_manager.AddObservation(point3d_id, image_id,
                                       correspondence.point2d_idx);
  }
}

}  // namespace

ImageRegistrationSummary RegisterImageWithExistingPoints(
    const MatchGraph& graph,
    ImageId image_id,
    const ImageRegistrarOptions& options,
    Reconstruction* reconstruction) {
  ImageRegistrationSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }
  if (reconstruction->IsImageRegistered(image_id)) {
    summary.message = "Image is already registered";
    return summary;
  }

  const std::vector<PnPCorrespondence> correspondences =
      CollectPnPCorrespondences(graph, image_id, reconstruction);
  summary.num_correspondences = correspondences.size();
  if (correspondences.size() < options.min_num_correspondences) {
    summary.message = "Too few 2D-3D correspondences";
    return summary;
  }

  const Image& image = reconstruction->GetImage(image_id);
  const Camera& camera = reconstruction->GetCamera(image.camera_id);
  const PnPRansacResult result =
      EstimatePoseRansac(camera, correspondences, options);
  summary.num_ransac_inliers = result.num_inliers_before_final_refinement;
  summary.num_inliers = result.inlier_indices.size();
  if (summary.num_inliers < options.min_num_inliers) {
    summary.message = "PnP RANSAC failed or found too few inliers";
    return summary;
  }

  Image& mutable_image = reconstruction->GetImage(image_id);
  mutable_image.cam_from_world = result.pose;
  reconstruction->RegisterImage(image_id);
  LinkInlierCorrespondencesToImage(image_id, correspondences,
                                   result.inlier_indices, reconstruction);

  summary.success = true;
  summary.message = "Image registered";
  return summary;
}

}  // namespace colite
