#include "colite/sfm/incremental_initializer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "colite/core/geometry.h"

namespace colite {
namespace {

struct TriangulatedObservation {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  FeatureMatch match;
};

bool IsUsableInitialGeometry(const TwoViewGeometry& geometry,
                             const IncrementalInitializerOptions& options) {
  return geometry.type == TwoViewGeometry::Type::kCalibratedEssential &&
         geometry.inlier_matches.size() >= options.min_num_inliers &&
         geometry.median_triangulation_angle >=
             options.min_triangulation_angle_deg;
}

void EnsureImageAndCameraExist(const Reconstruction& reconstruction,
                               ImageId image_id) {
  if (!reconstruction.ExistsImage(image_id)) {
    throw std::out_of_range("Initial image is missing in reconstruction");
  }
  const Image& image = reconstruction.GetImage(image_id);
  if (!reconstruction.ExistsCamera(image.camera_id)) {
    throw std::out_of_range("Initial image references a missing camera");
  }
}

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

bool IsFinitePoint(const Eigen::Vector3d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
}

bool HasValidMatchIndices(const FeatureMatch& match,
                          const Image& image1,
                          const Image& image2) {
  return match.idx1 != kInvalidPoint2DIdx &&
         match.idx2 != kInvalidPoint2DIdx &&
         match.idx1 < image1.points2d.size() &&
         match.idx2 < image2.points2d.size();
}

bool IsObservationAvailable(const FeatureMatch& match,
                            const Image& image1,
                            const Image& image2) {
  return !image1.points2d[match.idx1].HasPoint3D() &&
         !image2.points2d[match.idx2].HasPoint3D();
}

bool PassesReprojectionCheck(const Camera& camera1,
                             const Pose3& cam1_from_world,
                             const Image& image1,
                             const Camera& camera2,
                             const Pose3& cam2_from_world,
                             const Image& image2,
                             const FeatureMatch& match,
                             const Eigen::Vector3d& xyz,
                             double max_reprojection_error_px) {
  const auto error1 =
      ReprojectionError(camera1, cam1_from_world, xyz,
                        image1.points2d[match.idx1].xy);
  const auto error2 =
      ReprojectionError(camera2, cam2_from_world, xyz,
                        image2.points2d[match.idx2].xy);
  return error1.has_value() && error2.has_value() &&
         *error1 <= max_reprojection_error_px &&
         *error2 <= max_reprojection_error_px;
}

bool PassesTriangulationAngleCheck(const Camera& camera1,
                                   const Pose3& cam1_from_world,
                                   const Image& image1,
                                   const Camera& camera2,
                                   const Pose3& cam2_from_world,
                                   const Image& image2,
                                   const FeatureMatch& match,
                                   double min_triangulation_angle_deg) {
  const Eigen::Vector3d bearing1 =
      PixelToBearing(camera1, image1.points2d[match.idx1].xy);
  const Eigen::Vector3d bearing2 =
      PixelToBearing(camera2, image2.points2d[match.idx2].xy);
  return TriangulationAngleDegrees(cam1_from_world, bearing1, cam2_from_world,
                                   bearing2) >= min_triangulation_angle_deg;
}

std::vector<TriangulatedObservation> CollectTriangulatedObservations(
    const TwoViewGeometry& geometry,
    const Camera& camera1,
    const Pose3& cam1_from_world,
    const Image& image1,
    const Camera& camera2,
    const Pose3& cam2_from_world,
    const Image& image2,
    const IncrementalInitializerOptions& options) {
  std::vector<TriangulatedObservation> observations;
  observations.reserve(geometry.inlier_matches.size());

  for (const FeatureMatch& match : geometry.inlier_matches) {
    if (!HasValidMatchIndices(match, image1, image2) ||
        !IsObservationAvailable(match, image1, image2) ||
        !PassesTriangulationAngleCheck(camera1, cam1_from_world, image1,
                                       camera2, cam2_from_world, image2, match,
                                       options.min_triangulation_angle_deg)) {
      continue;
    }

    const auto point = TriangulatePoint(camera1, cam1_from_world,
                                        image1.points2d[match.idx1].xy,
                                        camera2, cam2_from_world,
                                        image2.points2d[match.idx2].xy);
    if (!point.has_value() || !IsFinitePoint(*point)) {
      continue;
    }
    if (!PassesReprojectionCheck(camera1, cam1_from_world, image1, camera2,
                                 cam2_from_world, image2, match, *point,
                                 options.max_reprojection_error_px)) {
      continue;
    }

    observations.push_back({*point, match});
  }

  return observations;
}

void AddTriangulatedObservations(
    ImageId image_id1,
    ImageId image_id2,
    const std::vector<TriangulatedObservation>& observations,
    Reconstruction* reconstruction) {
  for (const TriangulatedObservation& observation : observations) {
    Track track;
    track.AddElement(image_id1, observation.match.idx1);
    track.AddElement(image_id2, observation.match.idx2);
    reconstruction->AddPoint3D(observation.xyz, track);
  }
}

}  // namespace

std::vector<ImagePair> SelectInitialImagePairCandidates(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options) {
  std::vector<std::pair<ImagePair, const TwoViewGeometry*>> candidates;

  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsUsableInitialGeometry(geometry, options)) {
      continue;
    }
    candidates.emplace_back(pair, &geometry);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const auto& lhs, const auto& rhs) {
              const TwoViewGeometry& lhs_geometry = *lhs.second;
              const TwoViewGeometry& rhs_geometry = *rhs.second;
              if (lhs_geometry.inlier_matches.size() !=
                  rhs_geometry.inlier_matches.size()) {
                return lhs_geometry.inlier_matches.size() >
                       rhs_geometry.inlier_matches.size();
              }
              if (lhs_geometry.median_triangulation_angle !=
                  rhs_geometry.median_triangulation_angle) {
                return lhs_geometry.median_triangulation_angle >
                       rhs_geometry.median_triangulation_angle;
              }
              return lhs.first < rhs.first;
            });

  std::vector<ImagePair> pairs;
  pairs.reserve(candidates.size());
  for (const auto& [pair, geometry] : candidates) {
    (void)geometry;
    pairs.push_back(pair);
  }
  return pairs;
}

std::optional<ImagePair> SelectInitialImagePair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options) {
  const std::vector<ImagePair> candidates =
      SelectInitialImagePairCandidates(graph, options);
  if (candidates.empty()) {
    return std::nullopt;
  }
  return candidates.front();
}

IncrementalInitializerSummary InitializeReconstructionFromPair(
    const MatchGraph& graph,
    ImagePair initial_pair,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction) {
  IncrementalInitializerSummary summary;
  summary.initial_pair = initial_pair;

  const TwoViewGeometry* geometry =
      FindVerifiedGeometry(graph, initial_pair.image_id1, initial_pair.image_id2);
  if (geometry == nullptr || !IsUsableInitialGeometry(*geometry, options)) {
    summary.message = "Initial pair has no usable calibrated geometry";
    return summary;
  }

  const FeatureSet* features1 = FindFeatures(graph, initial_pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, initial_pair.image_id2);
  if (features1 == nullptr || features2 == nullptr) {
    summary.message = "Initial pair features are missing";
    return summary;
  }

  EnsureImageAndCameraExist(*reconstruction, initial_pair.image_id1);
  EnsureImageAndCameraExist(*reconstruction, initial_pair.image_id2);

  Image& image1 = reconstruction->GetImage(initial_pair.image_id1);
  Image& image2 = reconstruction->GetImage(initial_pair.image_id2);
  SyncImagePoints2DFromFeatures(*features1, &image1);
  SyncImagePoints2DFromFeatures(*features2, &image2);

  Pose3 cam1_from_world;
  cam1_from_world.q_cw = Eigen::Quaterniond::Identity();
  cam1_from_world.t_cw = Eigen::Vector3d::Zero();
  const Pose3 cam2_from_world = geometry->cam2_from_cam1;

  const Camera& camera1 = reconstruction->GetCamera(image1.camera_id);
  const Camera& camera2 = reconstruction->GetCamera(image2.camera_id);
  const std::vector<TriangulatedObservation> observations =
      CollectTriangulatedObservations(*geometry, camera1, cam1_from_world,
                                      image1, camera2, cam2_from_world, image2,
                                      options);

  if (observations.size() < options.min_num_triangulated_points) {
    summary.num_triangulated_points = observations.size();
    summary.message = "Initial pair produced too few triangulated points";
    return summary;
  }

  image1.cam_from_world = cam1_from_world;
  image2.cam_from_world = cam2_from_world;
  reconstruction->RegisterImage(initial_pair.image_id1);
  reconstruction->RegisterImage(initial_pair.image_id2);
  AddTriangulatedObservations(initial_pair.image_id1, initial_pair.image_id2,
                              observations, reconstruction);

  summary.success = true;
  summary.num_triangulated_points = observations.size();
  summary.message = "Initial reconstruction created";
  return summary;
}

IncrementalInitializerSummary InitializeReconstructionFromBestPair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction) {
  IncrementalInitializerSummary final_summary;
  const std::vector<ImagePair> candidates =
      SelectInitialImagePairCandidates(graph, options);
  if (candidates.empty()) {
    final_summary.message = "No usable initial image pair";
    return final_summary;
  }

  const std::size_t max_trials =
      options.max_num_initialization_trials == 0
          ? candidates.size()
          : std::min(options.max_num_initialization_trials, candidates.size());
  for (std::size_t idx = 0; idx < max_trials; ++idx) {
    const ImagePair pair = candidates[idx];
    final_summary.attempted_initial_pairs.push_back(pair);
    final_summary.num_initialization_attempts =
        final_summary.attempted_initial_pairs.size();

    IncrementalInitializerSummary attempt =
        InitializeReconstructionFromPair(graph, pair, options, reconstruction);
    attempt.attempted_initial_pairs = final_summary.attempted_initial_pairs;
    attempt.num_initialization_attempts =
        final_summary.num_initialization_attempts;
    if (attempt.success) {
      return attempt;
    }
    final_summary = std::move(attempt);
  }

  if (final_summary.message.empty()) {
    final_summary.message = "No initial image pair could be initialized";
  }
  return final_summary;
}

}  // namespace colite
