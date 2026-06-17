#include "colite/sfm/image_registrar.h"

#include <map>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace colite {
namespace {

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
                              ImageId neighbor_image_id,
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
  (void)neighbor_image_id;
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
      AddCorrespondenceIfValid(match, target_is_first, neighbor_id,
                               target_image, neighbor_image, *reconstruction,
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

cv::Mat CameraMatrixToCv(const Camera& camera) {
  const Eigen::Matrix3d k = camera.CalibrationMatrix();
  cv::Mat matrix(3, 3, CV_64F);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      matrix.at<double>(row, col) = k(row, col);
    }
  }
  return matrix;
}

cv::Mat DistortionCoefficientsToCv(const Camera& camera) {
  cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
  if (camera.model_id == CameraModelId::kSimpleRadial &&
      camera.params.size() >= 4) {
    distortion.at<double>(0, 0) = camera.params[3];
  }
  return distortion;
}

Pose3 PoseFromPnP(const cv::Mat& rvec, const cv::Mat& tvec) {
  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);

  Pose3 pose;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation(row, col) = rotation_cv.at<double>(row, col);
    }
  }
  pose.q_cw = Eigen::Quaterniond(rotation).normalized();
  pose.t_cw = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1),
                              tvec.at<double>(2));
  return pose;
}

void LinkInlierCorrespondencesToImage(
    ImageId image_id,
    const std::vector<PnPCorrespondence>& correspondences,
    const cv::Mat& inlier_indices,
    Reconstruction* reconstruction) {
  Image& image = reconstruction->GetImage(image_id);
  for (int row = 0; row < inlier_indices.rows; ++row) {
    const int correspondence_idx = inlier_indices.at<int>(row, 0);
    const PnPCorrespondence& correspondence =
        correspondences[static_cast<std::size_t>(correspondence_idx)];
    Point2D& point2d = image.points2d[correspondence.point2d_idx];
    if (point2d.HasPoint3D()) {
      continue;
    }
    point2d.point3d_id = correspondence.point3d_id;

    Point3D& point3d = reconstruction->GetPoint3D(correspondence.point3d_id);
    bool already_has_observation = false;
    for (const TrackElement& element : point3d.track.elements) {
      if (element.image_id == image_id &&
          element.point2d_idx == correspondence.point2d_idx) {
        already_has_observation = true;
        break;
      }
    }
    if (!already_has_observation) {
      point3d.track.AddElement(image_id, correspondence.point2d_idx);
    }
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

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  object_points.reserve(correspondences.size());
  image_points.reserve(correspondences.size());
  for (const PnPCorrespondence& correspondence : correspondences) {
    object_points.emplace_back(static_cast<float>(correspondence.xyz.x()),
                               static_cast<float>(correspondence.xyz.y()),
                               static_cast<float>(correspondence.xyz.z()));
    image_points.emplace_back(static_cast<float>(correspondence.xy.x()),
                              static_cast<float>(correspondence.xy.y()));
  }

  cv::Mat rvec;
  cv::Mat tvec;
  cv::Mat inlier_indices;
  const bool solved = cv::solvePnPRansac(
      object_points, image_points, CameraMatrixToCv(camera),
      DistortionCoefficientsToCv(camera), rvec, tvec, false,
      options.max_num_iterations, static_cast<float>(options.max_reprojection_error_px),
      options.confidence, inlier_indices, cv::SOLVEPNP_EPNP);
  summary.num_inliers = static_cast<std::size_t>(inlier_indices.rows);
  if (!solved || summary.num_inliers < options.min_num_inliers) {
    summary.message = "PnP RANSAC failed or found too few inliers";
    return summary;
  }

  Image& mutable_image = reconstruction->GetImage(image_id);
  mutable_image.cam_from_world = PoseFromPnP(rvec, tvec);
  reconstruction->RegisterImage(image_id);
  LinkInlierCorrespondencesToImage(image_id, correspondences, inlier_indices,
                                   reconstruction);

  summary.success = true;
  summary.message = "Image registered";
  return summary;
}

}  // namespace colite
