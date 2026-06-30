#include "colite/sfm/incremental_triangulator.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "colite/core/geometry.h"
#include "colite/sfm/observation_manager.h"

namespace colite {
namespace {

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

bool PassesPairAngleLayer(const TwoViewGeometry& geometry,
                          const IncrementalTriangulatorOptions& options) {
  if (options.min_pair_median_triangulation_angle_deg <= 0.0 ||
      geometry.median_triangulation_angle < 0.0) {
    return true;
  }
  return geometry.median_triangulation_angle >=
         options.min_pair_median_triangulation_angle_deg;
}

bool HasPoseAndCamera(const Reconstruction& reconstruction, ImageId image_id) {
  if (!reconstruction.ExistsImage(image_id)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(image_id);
  return image.HasPose() && reconstruction.ExistsCamera(image.camera_id);
}

bool HasValidMatchIndices(const FeatureMatch& match,
                          const Image& image1,
                          const Image& image2) {
  return match.idx1 != kInvalidPoint2DIdx &&
         match.idx2 != kInvalidPoint2DIdx &&
         match.idx1 < image1.points2d.size() &&
         match.idx2 < image2.points2d.size();
}

bool BothObservationsAreNew(const FeatureMatch& match,
                            const Image& image1,
                            const Image& image2) {
  return !image1.points2d[match.idx1].HasPoint3D() &&
         !image2.points2d[match.idx2].HasPoint3D();
}

bool IsFinitePoint(const Eigen::Vector3d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
}

bool PassesTriangulationAngle(const Camera& camera1,
                              const Pose3& cam1_from_world,
                              const Eigen::Vector2d& pixel1,
                              const Camera& camera2,
                              const Pose3& cam2_from_world,
                              const Eigen::Vector2d& pixel2,
                              double min_angle_deg) {
  const Eigen::Vector3d bearing1 = PixelToBearing(camera1, pixel1);
  const Eigen::Vector3d bearing2 = PixelToBearing(camera2, pixel2);
  return TriangulationAngleDegrees(cam1_from_world, bearing1, cam2_from_world,
                                   bearing2) >= min_angle_deg;
}

bool PassesReprojectionCheck(const Camera& camera1,
                             const Pose3& cam1_from_world,
                             const Eigen::Vector2d& pixel1,
                             const Camera& camera2,
                             const Pose3& cam2_from_world,
                             const Eigen::Vector2d& pixel2,
                             const Eigen::Vector3d& xyz,
                             double max_error_px) {
  const auto error1 = ReprojectionError(camera1, cam1_from_world, xyz, pixel1);
  const auto error2 = ReprojectionError(camera2, cam2_from_world, xyz, pixel2);
  return error1.has_value() && error2.has_value() && *error1 <= max_error_px &&
         *error2 <= max_error_px;
}

bool TryCreatePointFromMatch(const ImagePair& pair,
                             const FeatureMatch& match,
                             const IncrementalTriangulatorOptions& options,
                             Reconstruction* reconstruction) {
  Image& image1 = reconstruction->GetImage(pair.image_id1);
  Image& image2 = reconstruction->GetImage(pair.image_id2);
  if (!HasValidMatchIndices(match, image1, image2) ||
      !BothObservationsAreNew(match, image1, image2)) {
    return false;
  }

  const Camera& camera1 = reconstruction->GetCamera(image1.camera_id);
  const Camera& camera2 = reconstruction->GetCamera(image2.camera_id);
  const Pose3& cam1_from_world = *image1.cam_from_world;
  const Pose3& cam2_from_world = *image2.cam_from_world;
  const Eigen::Vector2d pixel1 = image1.points2d[match.idx1].xy;
  const Eigen::Vector2d pixel2 = image2.points2d[match.idx2].xy;

  if (!PassesTriangulationAngle(camera1, cam1_from_world, pixel1, camera2,
                                cam2_from_world, pixel2,
                                options.min_triangulation_angle_deg)) {
    return false;
  }

  const auto point = TriangulatePoint(camera1, cam1_from_world, pixel1, camera2,
                                      cam2_from_world, pixel2);
  if (!point.has_value() || !IsFinitePoint(*point)) {
    return false;
  }
  if (!PassesReprojectionCheck(camera1, cam1_from_world, pixel1, camera2,
                               cam2_from_world, pixel2, *point,
                               options.create_max_reprojection_error_px)) {
    return false;
  }

  Track track;
  track.AddElement(pair.image_id1, match.idx1);
  track.AddElement(pair.image_id2, match.idx2);
  ObservationManager observation_manager(reconstruction);
  return observation_manager.AddPoint3D(*point, track).has_value();
}

bool TrackHasImageConflict(const Track& track1, const Track& track2) {
  for (const TrackElement& element1 : track1.elements) {
    for (const TrackElement& element2 : track2.elements) {
      if (element1.image_id == element2.image_id) {
        return true;
      }
    }
  }
  return false;
}

bool ReprojectsTrackWithinThreshold(const Reconstruction& reconstruction,
                                    const Track& track,
                                    const Eigen::Vector3d& xyz,
                                    double max_error_px) {
  for (const TrackElement& element : track.elements) {
    if (!reconstruction.ExistsImage(element.image_id)) {
      return false;
    }
    const Image& image = reconstruction.GetImage(element.image_id);
    if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
        element.point2d_idx >= image.points2d.size()) {
      return false;
    }

    const Camera& camera = reconstruction.GetCamera(image.camera_id);
    const auto error =
        ReprojectionError(camera, *image.cam_from_world, xyz,
                          image.points2d[element.point2d_idx].xy);
    if (!error.has_value() || *error > max_error_px) {
      return false;
    }
  }
  return true;
}

Eigen::Vector3d WeightedMergedPointPosition(const Point3D& point1,
                                            const Point3D& point2) {
  const double weight1 =
      static_cast<double>(std::max<std::size_t>(1, point1.track.Length()));
  const double weight2 =
      static_cast<double>(std::max<std::size_t>(1, point2.track.Length()));
  return (weight1 * point1.xyz + weight2 * point2.xyz) / (weight1 + weight2);
}

bool TryMergePoint3Ds(Point3DId target_point3d_id,
                      Point3DId source_point3d_id,
                      const IncrementalTriangulatorOptions& options,
                      Reconstruction* reconstruction) {
  if (target_point3d_id == source_point3d_id ||
      !reconstruction->ExistsPoint3D(target_point3d_id) ||
      !reconstruction->ExistsPoint3D(source_point3d_id)) {
    return false;
  }

  const Point3D& target_point3d =
      reconstruction->GetPoint3D(target_point3d_id);
  const Point3D& source_point3d =
      reconstruction->GetPoint3D(source_point3d_id);
  if (TrackHasImageConflict(target_point3d.track, source_point3d.track)) {
    return false;
  }

  const Eigen::Vector3d merged_xyz =
      WeightedMergedPointPosition(target_point3d, source_point3d);
  if (!ReprojectsTrackWithinThreshold(
          *reconstruction, target_point3d.track, merged_xyz,
          options.create_max_reprojection_error_px) ||
      !ReprojectsTrackWithinThreshold(
          *reconstruction, source_point3d.track, merged_xyz,
          options.create_max_reprojection_error_px)) {
    return false;
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.MergePoints(target_point3d_id, source_point3d_id,
                                         merged_xyz);
}

bool TryCompleteTargetObservation(const ImagePair& pair,
                                  ImageId target_image_id,
                                  const FeatureMatch& match,
                                  const IncrementalTriangulatorOptions& options,
                                  Reconstruction* reconstruction) {
  Image& image1 = reconstruction->GetImage(pair.image_id1);
  Image& image2 = reconstruction->GetImage(pair.image_id2);
  if (!HasValidMatchIndices(match, image1, image2)) {
    return false;
  }

  const bool target_is_first = pair.image_id1 == target_image_id;
  Image& target_image = target_is_first ? image1 : image2;
  const Image& source_image = target_is_first ? image2 : image1;
  const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
  const Point2DIdx source_idx = target_is_first ? match.idx2 : match.idx1;

  Point2D& target_point2d = target_image.points2d[target_idx];
  const Point2D& source_point2d = source_image.points2d[source_idx];
  if (target_point2d.HasPoint3D() || !source_point2d.HasPoint3D() ||
      !reconstruction->ExistsPoint3D(source_point2d.point3d_id)) {
    return false;
  }

  const Camera& target_camera =
      reconstruction->GetCamera(target_image.camera_id);
  const Point3D& source_point3d =
      reconstruction->GetPoint3D(source_point2d.point3d_id);
  const auto error = ReprojectionError(target_camera,
                                       *target_image.cam_from_world,
                                       source_point3d.xyz,
                                       target_point2d.xy);
  if (!error.has_value() ||
      *error > options.create_max_reprojection_error_px) {
    return false;
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.AddObservation(source_point2d.point3d_id,
                                            target_image_id, target_idx);
}

bool TryMergeMatchedExistingPoints(const ImagePair& pair,
                                   ImageId target_image_id,
                                   const FeatureMatch& match,
                                   const IncrementalTriangulatorOptions& options,
                                   Reconstruction* reconstruction) {
  Image& image1 = reconstruction->GetImage(pair.image_id1);
  Image& image2 = reconstruction->GetImage(pair.image_id2);
  if (!HasValidMatchIndices(match, image1, image2)) {
    return false;
  }

  const bool target_is_first = pair.image_id1 == target_image_id;
  const Image& target_image = target_is_first ? image1 : image2;
  const Image& source_image = target_is_first ? image2 : image1;
  const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
  const Point2DIdx source_idx = target_is_first ? match.idx2 : match.idx1;

  const Point2D& target_point2d = target_image.points2d[target_idx];
  const Point2D& source_point2d = source_image.points2d[source_idx];
  if (!target_point2d.HasPoint3D() || !source_point2d.HasPoint3D()) {
    return false;
  }

  const Point3DId kept_point3d_id =
      std::min(target_point2d.point3d_id, source_point2d.point3d_id);
  const Point3DId merged_point3d_id =
      std::max(target_point2d.point3d_id, source_point2d.point3d_id);
  return TryMergePoint3Ds(kept_point3d_id, merged_point3d_id, options,
                          reconstruction);
}

bool HasTrackUpdates(const IncrementalTriangulatorSummary& summary) {
  return summary.num_completed_observations > 0 ||
         summary.num_merged_points > 0 || summary.num_created_points > 0;
}

void AccumulateTrackUpdates(const IncrementalTriangulatorSummary& update,
                            IncrementalTriangulatorSummary* summary) {
  summary->num_created_points += update.num_created_points;
  summary->num_created_observations += update.num_created_observations;
  summary->num_completed_observations += update.num_completed_observations;
  summary->num_merged_points += update.num_merged_points;
  summary->num_merged_observations += update.num_merged_observations;
}

}  // namespace

IncrementalTriangulatorSummary TriangulateRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->IsImageRegistered(image_id) ||
      !HasPoseAndCamera(*reconstruction, image_id)) {
    summary.message = "Image is not registered or has no pose";
    return summary;
  }

  std::set<std::pair<ImageId, Point2DIdx>> tried_observations;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry) ||
        !PassesPairAngleLayer(geometry, options)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction->IsImageRegistered(neighbor_id) ||
        !HasPoseAndCamera(*reconstruction, neighbor_id)) {
      continue;
    }

    for (const FeatureMatch& match : geometry.inlier_matches) {
      const Point2DIdx target_idx =
          pair.image_id1 == image_id ? match.idx1 : match.idx2;
      const auto target_observation = std::make_pair(image_id, target_idx);
      if (tried_observations.find(target_observation) !=
          tried_observations.end()) {
        continue;
      }

      if (TryCreatePointFromMatch(pair, match, options, reconstruction)) {
        tried_observations.insert(target_observation);
        ++summary.num_created_points;
        summary.num_created_observations += 2;
      }
    }
  }

  for (std::size_t iteration = 1;
       summary.num_created_points > 0 &&
       iteration < options.max_track_update_iterations;
       ++iteration) {
    const IncrementalTriangulatorSummary completion =
        CompleteTracksForRegisteredImage(graph, image_id, options,
                                         reconstruction);
    if (!HasTrackUpdates(completion)) {
      break;
    }
    AccumulateTrackUpdates(completion, &summary);
  }

  summary.message = "Triangulation finished";
  return summary;
}

IncrementalTriangulatorSummary CompleteTracksForRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->IsImageRegistered(image_id) ||
      !HasPoseAndCamera(*reconstruction, image_id)) {
    summary.message = "Image is not registered or has no pose";
    return summary;
  }

  std::set<std::pair<ImageId, Point2DIdx>> tried_observations;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry) ||
        !PassesPairAngleLayer(geometry, options)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction->IsImageRegistered(neighbor_id) ||
        !HasPoseAndCamera(*reconstruction, neighbor_id)) {
      continue;
    }

    for (const FeatureMatch& match : geometry.inlier_matches) {
      const Point2DIdx target_idx =
          pair.image_id1 == image_id ? match.idx1 : match.idx2;
      const auto target_observation = std::make_pair(image_id, target_idx);
      if (tried_observations.find(target_observation) !=
          tried_observations.end()) {
        continue;
      }

      if (TryCompleteTargetObservation(pair, image_id, match, options,
                                       reconstruction)) {
        tried_observations.insert(target_observation);
        ++summary.num_completed_observations;
      } else if (TryMergeMatchedExistingPoints(pair, image_id, match, options,
                                               reconstruction)) {
        tried_observations.insert(target_observation);
        ++summary.num_merged_points;
        ++summary.num_merged_observations;
      }
    }
  }

  summary.message = "Track completion finished";
  return summary;
}

}  // namespace colite
