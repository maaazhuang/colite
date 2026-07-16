#include "colite/sfm/incremental_triangulator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include <Eigen/SVD>

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

bool IsFinitePoint(const Eigen::Vector3d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
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
  // Match COLMAP's merge policy: preserve the two optimized point estimates
  // using a track-length weighted position. Re-triangulating the combined
  // track here can move a good point enough to reject an otherwise valid
  // merge before the next bundle-adjustment pass.
  const Eigen::Vector3d merged_xyz =
      WeightedMergedPointPosition(target_point3d, source_point3d);
  if (!ReprojectsTrackWithinThreshold(
          *reconstruction, target_point3d.track, merged_xyz,
          options.merge_max_reprojection_error_px) ||
      !ReprojectsTrackWithinThreshold(
          *reconstruction, source_point3d.track, merged_xyz,
          options.merge_max_reprojection_error_px)) {
    return false;
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.MergePoints(target_point3d_id, source_point3d_id,
                                         merged_xyz);
}

bool TryCompleteObservation(const ImagePair& pair,
                            ImageId requested_image_id,
                            const FeatureMatch& match,
                            const IncrementalTriangulatorOptions& options,
                            Reconstruction* reconstruction) {
  Image& image1 = reconstruction->GetImage(pair.image_id1);
  Image& image2 = reconstruction->GetImage(pair.image_id2);
  if (!HasValidMatchIndices(match, image1, image2)) {
    return false;
  }

  const bool first_has_point = image1.points2d[match.idx1].HasPoint3D();
  const bool requested_is_first = pair.image_id1 == requested_image_id;
  const bool target_is_first = options.complete_neighbor_observations
                                   ? !first_has_point
                                   : requested_is_first;
  Image& target_image = target_is_first ? image1 : image2;
  const Image& source_image = target_is_first ? image2 : image1;
  const ImageId target_image_id = target_is_first ? pair.image_id1
                                                   : pair.image_id2;
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
      *error > options.complete_max_reprojection_error_px) {
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

using ObservationKey = std::pair<ImageId, Point2DIdx>;
using CorrespondenceAdjacency =
    std::map<ObservationKey, std::vector<ObservationKey>>;

CorrespondenceAdjacency BuildCorrespondenceAdjacency(
    const MatchGraph& graph) {
  CorrespondenceAdjacency adjacency;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsUsableGeometry(geometry)) {
      continue;
    }
    for (const FeatureMatch& match : geometry.inlier_matches) {
      if (match.idx1 == kInvalidPoint2DIdx ||
          match.idx2 == kInvalidPoint2DIdx) {
        continue;
      }
      const ObservationKey first{pair.image_id1, match.idx1};
      const ObservationKey second{pair.image_id2, match.idx2};
      adjacency[first].push_back(second);
      adjacency[second].push_back(first);
    }
  }
  return adjacency;
}

bool IsTwoViewObservation(const CorrespondenceAdjacency& adjacency,
                          const ObservationKey& observation) {
  const auto it = adjacency.find(observation);
  if (it == adjacency.end() || it->second.size() != 1) {
    return false;
  }
  const auto other_it = adjacency.find(it->second.front());
  return other_it != adjacency.end() && other_it->second.size() == 1;
}

bool IsUsableObservation(const ObservationKey& observation,
                         const Reconstruction& reconstruction) {
  if (!reconstruction.IsImageRegistered(observation.first) ||
      !HasPoseAndCamera(reconstruction, observation.first)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(observation.first);
  return observation.second < image.points2d.size();
}

std::vector<ObservationKey> CollectTransitiveCorrespondences(
    const ObservationKey& reference,
    const CorrespondenceAdjacency& adjacency,
    std::size_t max_transitivity,
    const Reconstruction& reconstruction) {
  std::vector<ObservationKey> correspondences;
  std::vector<ObservationKey> frontier{reference};
  std::set<ObservationKey> visited{reference};
  for (std::size_t depth = 0;
       depth < max_transitivity && !frontier.empty(); ++depth) {
    std::vector<ObservationKey> next_frontier;
    for (const ObservationKey& source : frontier) {
      const auto adjacency_it = adjacency.find(source);
      if (adjacency_it == adjacency.end()) {
        continue;
      }
      for (const ObservationKey& target : adjacency_it->second) {
        if (!visited.insert(target).second) {
          continue;
        }
        next_frontier.push_back(target);
        if (IsUsableObservation(target, reconstruction)) {
          correspondences.push_back(target);
        }
      }
    }
    frontier = std::move(next_frontier);
  }
  return correspondences;
}

double AngularErrorDegrees(const ObservationKey& observation,
                           const Eigen::Vector3d& xyz,
                           const Reconstruction& reconstruction) {
  if (!IsUsableObservation(observation, reconstruction)) {
    return std::numeric_limits<double>::infinity();
  }
  const Image& image = reconstruction.GetImage(observation.first);
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const Eigen::Vector3d point_camera =
      image.cam_from_world->TransformPoint(xyz);
  if (point_camera.z() <= 0.0 || !IsFinitePoint(point_camera)) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Vector3d bearing =
      PixelToBearing(camera, image.points2d[observation.second].xy);
  const double cosine = std::clamp(
      bearing.dot(point_camera.normalized()), -1.0, 1.0);
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  return std::acos(cosine) * kRadiansToDegrees;
}

double ObservationPairAngleDegrees(const ObservationKey& first,
                                   const ObservationKey& second,
                                   const Reconstruction& reconstruction) {
  const Image& image1 = reconstruction.GetImage(first.first);
  const Image& image2 = reconstruction.GetImage(second.first);
  const Camera& camera1 = reconstruction.GetCamera(image1.camera_id);
  const Camera& camera2 = reconstruction.GetCamera(image2.camera_id);
  return TriangulationAngleDegrees(
      *image1.cam_from_world,
      PixelToBearing(camera1, image1.points2d[first.second].xy),
      *image2.cam_from_world,
      PixelToBearing(camera2, image2.points2d[second.second].xy));
}

std::optional<Eigen::Vector3d> TriangulateObservationsLinear(
    const std::vector<ObservationKey>& observations,
    const Reconstruction& reconstruction) {
  if (observations.size() < 2) {
    return std::nullopt;
  }
  Eigen::MatrixXd equations(2 * observations.size(), 4);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const ObservationKey& observation = observations[index];
    const Image& image = reconstruction.GetImage(observation.first);
    const Camera& camera = reconstruction.GetCamera(image.camera_id);
    const Eigen::Vector3d bearing =
        PixelToBearing(camera, image.points2d[observation.second].xy);
    if (std::abs(bearing.z()) < 1e-12) {
      return std::nullopt;
    }
    Eigen::Matrix<double, 3, 4> projection;
    projection.leftCols<3>() =
        image.cam_from_world->q_cw.toRotationMatrix();
    projection.rightCols<1>() = image.cam_from_world->t_cw;
    const double x = bearing.x() / bearing.z();
    const double y = bearing.y() / bearing.z();
    equations.row(2 * index) = x * projection.row(2) - projection.row(0);
    equations.row(2 * index + 1) =
        y * projection.row(2) - projection.row(1);
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(equations,
                                               Eigen::ComputeFullV);
  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) < 1e-12) {
    return std::nullopt;
  }
  const Eigen::Vector3d xyz = homogeneous.hnormalized();
  return IsFinitePoint(xyz) ? std::optional<Eigen::Vector3d>(xyz)
                            : std::nullopt;
}

struct RobustTrackEstimate {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  std::vector<bool> inlier_mask;
  std::size_t num_inliers = 0;
  double inlier_error_sum = std::numeric_limits<double>::infinity();
};

RobustTrackEstimate ScoreTrackHypothesis(
    const Eigen::Vector3d& xyz,
    const std::vector<ObservationKey>& observations,
    double max_angle_error_deg,
    const Reconstruction& reconstruction) {
  RobustTrackEstimate estimate;
  estimate.xyz = xyz;
  estimate.inlier_mask.assign(observations.size(), false);
  estimate.inlier_error_sum = 0.0;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const double error =
        AngularErrorDegrees(observations[index], xyz, reconstruction);
    if (error <= max_angle_error_deg) {
      estimate.inlier_mask[index] = true;
      ++estimate.num_inliers;
      estimate.inlier_error_sum += error;
    }
  }
  return estimate;
}

bool IsBetterTrackEstimate(const RobustTrackEstimate& candidate,
                           const RobustTrackEstimate& best) {
  return candidate.num_inliers > best.num_inliers ||
         (candidate.num_inliers == best.num_inliers &&
          candidate.inlier_error_sum < best.inlier_error_sum);
}

std::optional<RobustTrackEstimate> EstimateRobustTrack(
    const std::vector<ObservationKey>& observations,
    const IncrementalTriangulatorOptions& options,
    const Reconstruction& reconstruction) {
  if (observations.size() < 2) {
    return std::nullopt;
  }

  RobustTrackEstimate best;
  best.inlier_mask.assign(observations.size(), false);
  constexpr std::size_t kMaxPairHypotheses = 10000;
  std::size_t num_hypotheses = 0;
  for (std::size_t first = 0; first + 1 < observations.size(); ++first) {
    for (std::size_t second = first + 1; second < observations.size();
         ++second) {
      if (++num_hypotheses > kMaxPairHypotheses) {
        break;
      }
      if (ObservationPairAngleDegrees(observations[first],
                                      observations[second], reconstruction) <
          options.min_triangulation_angle_deg) {
        continue;
      }
      const Image& image1 =
          reconstruction.GetImage(observations[first].first);
      const Image& image2 =
          reconstruction.GetImage(observations[second].first);
      const Camera& camera1 = reconstruction.GetCamera(image1.camera_id);
      const Camera& camera2 = reconstruction.GetCamera(image2.camera_id);
      const std::optional<Eigen::Vector3d> xyz = TriangulatePoint(
          camera1, *image1.cam_from_world,
          image1.points2d[observations[first].second].xy, camera2,
          *image2.cam_from_world,
          image2.points2d[observations[second].second].xy);
      if (!xyz || !IsFinitePoint(*xyz)) {
        continue;
      }
      const RobustTrackEstimate candidate = ScoreTrackHypothesis(
          *xyz, observations, options.create_max_angle_error_deg,
          reconstruction);
      if (candidate.num_inliers >= 2 &&
          IsBetterTrackEstimate(candidate, best)) {
        best = candidate;
      }
    }
    if (num_hypotheses > kMaxPairHypotheses) {
      break;
    }
  }
  if (best.num_inliers < 2) {
    return std::nullopt;
  }

  // COLMAP's LORANSAC locally re-estimates the model from all current
  // inliers. Two deterministic refinement rounds provide the same important
  // behavior without introducing another RANSAC framework dependency.
  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<ObservationKey> inliers;
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (best.inlier_mask[index]) {
        inliers.push_back(observations[index]);
      }
    }
    const std::optional<Eigen::Vector3d> refined_xyz =
        TriangulateObservationsLinear(inliers, reconstruction);
    if (!refined_xyz) {
      break;
    }
    const RobustTrackEstimate refined = ScoreTrackHypothesis(
        *refined_xyz, observations, options.create_max_angle_error_deg,
        reconstruction);
    if (refined.num_inliers < 2 || !IsBetterTrackEstimate(refined, best)) {
      break;
    }
    best = refined;
  }
  return best;
}

struct CreateTrackSummary {
  std::size_t num_points = 0;
  std::size_t num_observations = 0;
};

CreateTrackSummary CreatePointsFromCorrespondences(
    const std::vector<ObservationKey>& correspondences,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  std::vector<ObservationKey> candidates;
  candidates.reserve(correspondences.size());
  for (const ObservationKey& observation : correspondences) {
    if (!IsUsableObservation(observation, *reconstruction)) {
      continue;
    }
    const Image& image = reconstruction->GetImage(observation.first);
    if (!image.points2d[observation.second].HasPoint3D()) {
      candidates.push_back(observation);
    }
  }
  if (candidates.size() < 2 ||
      (options.ignore_two_view_tracks && candidates.size() == 2 &&
       IsTwoViewObservation(adjacency, candidates.front()))) {
    return {};
  }

  const std::optional<RobustTrackEstimate> estimate =
      EstimateRobustTrack(candidates, options, *reconstruction);
  if (!estimate) {
    return {};
  }
  Track track;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (estimate->inlier_mask[index]) {
      track.AddElement(candidates[index].first, candidates[index].second);
    }
  }
  ObservationManager observation_manager(reconstruction);
  if (!observation_manager.AddPoint3D(estimate->xyz, track).has_value()) {
    return {};
  }

  CreateTrackSummary summary{1, track.Length()};
  constexpr std::size_t kMinRecursiveTrackLength = 3;
  if (candidates.size() - track.Length() >= kMinRecursiveTrackLength) {
    const CreateTrackSummary recursive = CreatePointsFromCorrespondences(
        correspondences, adjacency, options, reconstruction);
    summary.num_points += recursive.num_points;
    summary.num_observations += recursive.num_observations;
  }
  return summary;
}

bool ContinueObservationAngular(
    const ObservationKey& reference,
    const std::vector<ObservationKey>& correspondences,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  if (!IsUsableObservation(reference, *reconstruction)) {
    return false;
  }
  Image& reference_image = reconstruction->GetImage(reference.first);
  if (reference_image.points2d[reference.second].HasPoint3D()) {
    return false;
  }

  Point3DId best_point3d_id = kInvalidPoint3DId;
  double best_error = std::numeric_limits<double>::infinity();
  std::set<Point3DId> tested_point_ids;
  for (const ObservationKey& correspondence : correspondences) {
    if (!IsUsableObservation(correspondence, *reconstruction)) {
      continue;
    }
    const Point2D& point2d = reconstruction->GetImage(correspondence.first)
                                 .points2d[correspondence.second];
    if (!point2d.HasPoint3D() ||
        !reconstruction->ExistsPoint3D(point2d.point3d_id) ||
        !tested_point_ids.insert(point2d.point3d_id).second) {
      continue;
    }
    const double error = AngularErrorDegrees(
        reference, reconstruction->GetPoint3D(point2d.point3d_id).xyz,
        *reconstruction);
    if (error < best_error ||
        (error == best_error && point2d.point3d_id < best_point3d_id)) {
      best_error = error;
      best_point3d_id = point2d.point3d_id;
    }
  }
  if (best_point3d_id == kInvalidPoint3DId ||
      best_error > options.continue_max_angle_error_deg) {
    return false;
  }
  ObservationManager observation_manager(reconstruction);
  return observation_manager.AddObservation(best_point3d_id, reference.first,
                                            reference.second);
}

std::size_t CompletePointTrack(
    Point3DId point3d_id,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  if (!reconstruction->ExistsPoint3D(point3d_id)) {
    return 0;
  }
  const Eigen::Vector3d xyz = reconstruction->GetPoint3D(point3d_id).xyz;
  std::vector<ObservationKey> frontier;
  std::set<ObservationKey> visited;
  for (const TrackElement& element :
       reconstruction->GetPoint3D(point3d_id).track.elements) {
    const ObservationKey observation{element.image_id, element.point2d_idx};
    frontier.push_back(observation);
    visited.insert(observation);
  }

  std::size_t num_completed = 0;
  ObservationManager observation_manager(reconstruction);
  for (std::size_t depth = 0;
       depth < options.complete_max_transitivity && !frontier.empty();
       ++depth) {
    std::vector<ObservationKey> next_frontier;
    for (const ObservationKey& source : frontier) {
      const auto adjacency_it = adjacency.find(source);
      if (adjacency_it == adjacency.end()) {
        continue;
      }
      for (const ObservationKey& target : adjacency_it->second) {
        if (!visited.insert(target).second ||
            !reconstruction->IsImageRegistered(target.first) ||
            !HasPoseAndCamera(*reconstruction, target.first)) {
          continue;
        }
        Image& image = reconstruction->GetImage(target.first);
        if (target.second >= image.points2d.size()) {
          continue;
        }
        Point2D& point2d = image.points2d[target.second];
        if (point2d.HasPoint3D()) {
          continue;
        }
        const Camera& camera = reconstruction->GetCamera(image.camera_id);
        const std::optional<double> error = ReprojectionError(
            camera, *image.cam_from_world, xyz, point2d.xy);
        if (!error.has_value() ||
            *error > options.complete_max_reprojection_error_px) {
          continue;
        }
        if (observation_manager.AddObservation(point3d_id, target.first,
                                               target.second)) {
          ++num_completed;
          next_frontier.push_back(target);
        }
      }
    }
    frontier = std::move(next_frontier);
  }
  return num_completed;
}

void MergePointTrack(
    Point3DId point3d_id,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    IncrementalTriangulatorSummary* summary) {
  while (reconstruction->ExistsPoint3D(point3d_id)) {
    std::set<Point3DId> candidates;
    const Track track = reconstruction->GetPoint3D(point3d_id).track;
    for (const TrackElement& element : track.elements) {
      const auto adjacency_it =
          adjacency.find({element.image_id, element.point2d_idx});
      if (adjacency_it == adjacency.end()) {
        continue;
      }
      for (const ObservationKey& observation : adjacency_it->second) {
        if (!reconstruction->ExistsImage(observation.first)) {
          continue;
        }
        const Image& image = reconstruction->GetImage(observation.first);
        if (observation.second >= image.points2d.size()) {
          continue;
        }
        const Point2D& point2d = image.points2d[observation.second];
        if (point2d.HasPoint3D() && point2d.point3d_id != point3d_id &&
            reconstruction->ExistsPoint3D(point2d.point3d_id)) {
          candidates.insert(point2d.point3d_id);
        }
      }
    }

    bool merged = false;
    for (const Point3DId candidate_id : candidates) {
      if (!reconstruction->ExistsPoint3D(candidate_id)) {
        continue;
      }
      const Point3DId target_id = std::min(point3d_id, candidate_id);
      const Point3DId source_id = std::max(point3d_id, candidate_id);
      const std::size_t merged_observations =
          reconstruction->GetPoint3D(target_id).track.Length() +
          reconstruction->GetPoint3D(source_id).track.Length();
      if (TryMergePoint3Ds(target_id, source_id, options,
                           reconstruction)) {
        point3d_id = target_id;
        ++summary->num_merged_points;
        summary->num_merged_observations += merged_observations;
        merged = true;
        break;
      }
    }
    if (!merged) {
      break;
    }
  }
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

  const CorrespondenceAdjacency adjacency =
      BuildCorrespondenceAdjacency(graph);
  const Image& reference_image = reconstruction->GetImage(image_id);
  for (Point2DIdx point2d_idx = 0;
       point2d_idx < reference_image.points2d.size(); ++point2d_idx) {
    const ObservationKey reference{image_id, point2d_idx};
    std::vector<ObservationKey> correspondences =
        CollectTransitiveCorrespondences(reference, adjacency,
                                         options.max_transitivity,
                                         *reconstruction);
    if (correspondences.empty()) {
      continue;
    }

    bool has_triangulated_correspondence = false;
    for (const ObservationKey& correspondence : correspondences) {
      const Point2D& point2d = reconstruction->GetImage(correspondence.first)
                                   .points2d[correspondence.second];
      if (point2d.HasPoint3D()) {
        has_triangulated_correspondence = true;
        break;
      }
    }
    if (has_triangulated_correspondence &&
        ContinueObservationAngular(reference, correspondences, options,
                                   reconstruction)) {
      ++summary.num_completed_observations;
    }

    correspondences.push_back(reference);
    const CreateTrackSummary created = CreatePointsFromCorrespondences(
        correspondences, adjacency, options, reconstruction);
    summary.num_created_points += created.num_points;
    summary.num_created_observations += created.num_observations;
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

      if (TryCompleteObservation(pair, image_id, match, options,
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

IncrementalTriangulatorSummary CompleteAndMergeAllTracks(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }
  const CorrespondenceAdjacency adjacency =
      BuildCorrespondenceAdjacency(graph);
  for (std::size_t iteration = 0;
       iteration < options.max_track_update_iterations; ++iteration) {
    const std::size_t updates_before =
        summary.num_completed_observations + summary.num_merged_points;
    std::vector<Point3DId> point_ids;
    point_ids.reserve(reconstruction->Points3D().size());
    for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
      (void)point3d;
      point_ids.push_back(point3d_id);
    }
    for (const Point3DId point3d_id : point_ids) {
      summary.num_completed_observations += CompletePointTrack(
          point3d_id, adjacency, options, reconstruction);
    }

    point_ids.clear();
    for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
      (void)point3d;
      point_ids.push_back(point3d_id);
    }
    for (const Point3DId point3d_id : point_ids) {
      if (reconstruction->ExistsPoint3D(point3d_id)) {
        MergePointTrack(point3d_id, adjacency, options, reconstruction,
                        &summary);
      }
    }
    const std::size_t updates_after =
        summary.num_completed_observations + summary.num_merged_points;
    if (updates_after == updates_before) {
      break;
    }
  }
  summary.message = "Global track completion and merging finished";
  return summary;
}

}  // namespace colite
