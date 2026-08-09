#include "colite/sfm/incremental_triangulator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include <Eigen/SVD>

#include "colite/core/geometry.h"
#include "colite/core/parallel.h"
#include "colite/sfm/observation_manager.h"

namespace colite {
namespace {

using ObservationKey = IncrementalTriangulatorCache::ObservationKey;
using CorrespondenceAdjacency =
    IncrementalTriangulatorCache::CorrespondenceAdjacency;

std::optional<Eigen::Vector3d> TriangulateObservationsLinear(
    const std::vector<ObservationKey>& observations,
    const Reconstruction& reconstruction);

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

struct MergePositionScore {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  double max_error_px = std::numeric_limits<double>::infinity();
  double squared_error_sum = std::numeric_limits<double>::infinity();
  bool valid = false;
};

MergePositionScore ScoreMergedPointPosition(
    const Reconstruction& reconstruction,
    const Track& first_track,
    const Track& second_track,
    const Eigen::Vector3d& xyz) {
  MergePositionScore score;
  score.xyz = xyz;
  score.max_error_px = 0.0;
  score.squared_error_sum = 0.0;
  score.valid = IsFinitePoint(xyz);
  if (!score.valid) {
    return score;
  }
  for (const Track* track : {&first_track, &second_track}) {
    for (const TrackElement& element : track->elements) {
      const Image& image = reconstruction.GetImage(element.image_id);
      const Camera& camera = reconstruction.GetCamera(image.camera_id);
      const std::optional<double> error =
          ReprojectionError(camera, *image.cam_from_world, xyz,
                            image.points2d[element.point2d_idx].xy);
      if (!error.has_value()) {
        score.valid = false;
        return score;
      }
      score.max_error_px = std::max(score.max_error_px, *error);
      score.squared_error_sum += *error * *error;
    }
  }
  return score;
}

bool IsBetterMergePosition(const MergePositionScore& candidate,
                           const MergePositionScore& best) {
  return candidate.valid &&
         (!best.valid ||
          candidate.max_error_px < best.max_error_px ||
          (candidate.max_error_px == best.max_error_px &&
           candidate.squared_error_sum < best.squared_error_sum));
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
  MergePositionScore best = ScoreMergedPointPosition(
      *reconstruction, target_point3d.track, source_point3d.track,
      WeightedMergedPointPosition(target_point3d, source_point3d));
  std::vector<ObservationKey> observations;
  observations.reserve(target_point3d.track.Length() +
                       source_point3d.track.Length());
  for (const Track* track :
       {&target_point3d.track, &source_point3d.track}) {
    for (const TrackElement& element : track->elements) {
      observations.emplace_back(element.image_id, element.point2d_idx);
    }
  }
  const std::optional<Eigen::Vector3d> triangulated_xyz =
      TriangulateObservationsLinear(observations, *reconstruction);
  if (triangulated_xyz.has_value()) {
    const MergePositionScore triangulated = ScoreMergedPointPosition(
        *reconstruction, target_point3d.track, source_point3d.track,
        *triangulated_xyz);
    if (IsBetterMergePosition(triangulated, best)) {
      best = triangulated;
    }
  }
  std::optional<TrackElement> outlier_to_remove;
  if (best.valid &&
      best.max_error_px > options.merge_max_reprojection_error_px &&
      observations.size() >= 4) {
    const std::vector<Eigen::Vector3d> robust_seeds = {
        WeightedMergedPointPosition(target_point3d, source_point3d),
        target_point3d.xyz, source_point3d.xyz, best.xyz};
    for (const Eigen::Vector3d& seed_xyz : robust_seeds) {
      std::vector<ObservationKey> inlier_observations;
      inlier_observations.reserve(observations.size() - 1);
      std::size_t num_target_inliers = 0;
      std::size_t num_source_inliers = 0;
      std::size_t num_outliers = 0;
      std::optional<TrackElement> seed_outlier;
      for (std::size_t index = 0; index < observations.size(); ++index) {
        const ObservationKey& observation = observations[index];
        const Image& image = reconstruction->GetImage(observation.first);
        const Camera& camera = reconstruction->GetCamera(image.camera_id);
        const std::optional<double> error =
            ReprojectionError(camera, *image.cam_from_world, seed_xyz,
                              image.points2d[observation.second].xy);
        if (!error.has_value() ||
            *error > options.merge_max_reprojection_error_px) {
          ++num_outliers;
          seed_outlier =
              TrackElement{observation.first, observation.second};
        } else {
          inlier_observations.push_back(observation);
          if (index < target_point3d.track.Length()) {
            ++num_target_inliers;
          } else {
            ++num_source_inliers;
          }
        }
      }

      if (num_outliers != 1 || num_target_inliers == 0 ||
          num_source_inliers == 0 || inlier_observations.size() < 3) {
        continue;
      }
      const std::optional<Eigen::Vector3d> robust_xyz =
          TriangulateObservationsLinear(inlier_observations,
                                        *reconstruction);
      if (robust_xyz.has_value()) {
        Track inlier_track;
        for (const ObservationKey& observation : inlier_observations) {
          inlier_track.AddElement(observation.first, observation.second);
        }
        const MergePositionScore robust_score = ScoreMergedPointPosition(
            *reconstruction, inlier_track, Track{}, *robust_xyz);
        if (robust_score.valid &&
            robust_score.max_error_px <=
                options.merge_max_reprojection_error_px) {
          best = robust_score;
          outlier_to_remove = seed_outlier;
          break;
        }
      }
    }
  }
  if (!best.valid ||
      best.max_error_px > options.merge_max_reprojection_error_px) {
    return false;
  }

  if (outlier_to_remove.has_value()) {
    const Image& image =
        reconstruction->GetImage(outlier_to_remove->image_id);
    const Point3DId owner_id =
        image.points2d[outlier_to_remove->point2d_idx].point3d_id;
    if ((owner_id != target_point3d_id && owner_id != source_point3d_id) ||
        !reconstruction->DeleteObservation(
            owner_id, outlier_to_remove->image_id,
            outlier_to_remove->point2d_idx)) {
      return false;
    }
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.MergePoints(target_point3d_id, source_point3d_id,
                                         best.xyz);
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

CorrespondenceAdjacency BuildCorrespondenceAdjacency(
    const MatchGraph& graph) {
  CorrespondenceAdjacency adjacency;
  for (const auto& [image_id, features] : graph.features) {
    adjacency.ResizeImage(image_id, features.keypoints.size());
  }
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
      adjacency.Mutable(first)->push_back(second);
      adjacency.Mutable(second)->push_back(first);
    }
  }
  return adjacency;
}

bool IsTwoViewObservation(const CorrespondenceAdjacency& adjacency,
                          const ObservationKey& observation) {
  const auto* neighbors = adjacency.Find(observation);
  if (neighbors == nullptr || neighbors->size() != 1) {
    return false;
  }
  const auto* other_neighbors = adjacency.Find(neighbors->front());
  return other_neighbors != nullptr && other_neighbors->size() == 1;
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
      const auto* neighbors = adjacency.Find(source);
      if (neighbors == nullptr) {
        continue;
      }
      for (const ObservationKey& target : *neighbors) {
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
  struct ImageInlier {
    std::size_t observation_index = 0;
    double error = std::numeric_limits<double>::infinity();
  };
  std::map<ImageId, ImageInlier> best_by_image;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const double error =
        AngularErrorDegrees(observations[index], xyz, reconstruction);
    if (error > max_angle_error_deg) {
      continue;
    }
    const ImageId image_id = observations[index].first;
    const auto existing = best_by_image.find(image_id);
    if (existing == best_by_image.end() ||
        error < existing->second.error ||
        (error == existing->second.error &&
         observations[index] <
             observations[existing->second.observation_index])) {
      best_by_image[image_id] = {index, error};
    }
  }
  for (const auto& [image_id, inlier] : best_by_image) {
    (void)image_id;
    estimate.inlier_mask[inlier.observation_index] = true;
    ++estimate.num_inliers;
    estimate.inlier_error_sum += inlier.error;
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
      const auto* neighbors = adjacency.Find(source);
      if (neighbors == nullptr) {
        continue;
      }
      for (const ObservationKey& target : *neighbors) {
        if (!visited.insert(target).second ||
            !reconstruction->IsImageRegistered(target.first) ||
            !HasPoseAndCamera(*reconstruction, target.first)) {
          continue;
        }
        Image& image = reconstruction->GetImage(target.first);
        if (target.second >= image.points2d.size()) {
          continue;
        }
        // Transitivity describes correspondence-graph traversal, not only a
        // chain of observations that can immediately be attached to this
        // point.  An intermediate feature may already belong to another point
        // or conflict with an image already present in the track, while a
        // later feature in the same verified component is still a valid new
        // observation.  Keep traversing such nodes; geometry and one-image-
        // one-observation checks below still guard every persisted update.
        next_frontier.push_back(target);
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
        }
      }
    }
    frontier = std::move(next_frontier);
  }
  return num_completed;
}

std::size_t CompleteUnassignedObservationsFromComponents(
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    std::set<Point3DId>* dirty_point_ids) {
  struct CompletionProposal {
    ObservationKey target{kInvalidImageId, kInvalidPoint2DIdx};
    std::array<Point3DId, 4> point3d_ids{};
    std::array<double, 4> errors{};
    std::size_t num_candidates = 0;
  };

  std::unordered_map<ImageId, std::size_t> image_offsets;
  image_offsets.reserve(adjacency.by_image.size());
  std::size_t total_observations = 0;
  for (const auto& [image_id, feature_neighbors] : adjacency.by_image) {
    image_offsets.emplace(image_id, total_observations);
    total_observations += feature_neighbors.size();
  }
  const auto flat_observation_index =
      [&image_offsets](const ObservationKey& observation) {
        return image_offsets.at(observation.first) + observation.second;
      };

  std::vector<ObservationKey> targets;
  targets.reserve(total_observations / 2);
  for (const auto& [image_id, feature_neighbors] : adjacency.by_image) {
    if (!reconstruction->IsImageRegistered(image_id) ||
        !HasPoseAndCamera(*reconstruction, image_id)) {
      continue;
    }
    Image& target_image = reconstruction->GetImage(image_id);
    const std::size_t num_features =
        std::min(feature_neighbors.size(), target_image.points2d.size());
    for (Point2DIdx point2d_idx = 0; point2d_idx < num_features;
         ++point2d_idx) {
      if (feature_neighbors[point2d_idx].empty() ||
          target_image.points2d[point2d_idx].HasPoint3D()) {
        continue;
      }
      targets.emplace_back(image_id, point2d_idx);
    }
  }

  std::vector<CompletionProposal> proposals(targets.size());
  const Reconstruction& read_only_reconstruction = *reconstruction;
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(targets.size())),
      [&](const parallel::Range& range) {
        std::vector<ObservationKey> frontier;
        std::vector<ObservationKey> next_frontier;
        std::vector<Point3DId> candidate_point_ids;
        frontier.reserve(64);
        next_frontier.reserve(64);
        candidate_point_ids.reserve(16);
        std::vector<std::uint32_t> visited_epochs(total_observations, 0U);
        std::uint32_t current_epoch = 0U;
        for (int target_index = range.start; target_index < range.end;
             ++target_index) {
          const ObservationKey target = targets[target_index];
          CompletionProposal& proposal = proposals[target_index];
          proposal.target = target;
          const Image& target_image =
              read_only_reconstruction.GetImage(target.first);
          const Camera& target_camera =
              read_only_reconstruction.GetCamera(target_image.camera_id);

          frontier.clear();
          next_frontier.clear();
          candidate_point_ids.clear();
          if (++current_epoch == 0U) {
            std::fill(visited_epochs.begin(), visited_epochs.end(), 0U);
            current_epoch = 1U;
          }
          frontier.push_back(target);
          visited_epochs[flat_observation_index(target)] = current_epoch;
          const std::size_t parallel_component_transitivity =
              std::min<std::size_t>(6, options.complete_max_transitivity);
          const auto score_candidate_points = [&]() {
            proposal.num_candidates = 0;
            for (const Point3DId candidate_id : candidate_point_ids) {
              const Point3D& candidate =
                  read_only_reconstruction.GetPoint3D(candidate_id);
              const bool already_observed_in_target =
                  std::any_of(candidate.track.elements.begin(),
                              candidate.track.elements.end(),
                              [target](const TrackElement& element) {
                                return element.image_id == target.first;
                              });
              if (already_observed_in_target) {
                continue;
              }
              const std::optional<double> error = ReprojectionError(
                  target_camera, *target_image.cam_from_world, candidate.xyz,
                  target_image.points2d[target.second].xy);
              if (!error.has_value() ||
                  *error > options.complete_max_reprojection_error_px) {
                continue;
              }
              std::size_t insert_index = proposal.num_candidates;
              while (
                  insert_index > 0 &&
                  (proposal.errors[insert_index - 1] > *error ||
                   (proposal.errors[insert_index - 1] == *error &&
                    proposal.point3d_ids[insert_index - 1] > candidate_id))) {
                if (insert_index < proposal.point3d_ids.size()) {
                  proposal.point3d_ids[insert_index] =
                      proposal.point3d_ids[insert_index - 1];
                  proposal.errors[insert_index] =
                      proposal.errors[insert_index - 1];
                }
                --insert_index;
              }
              if (insert_index < proposal.point3d_ids.size()) {
                proposal.point3d_ids[insert_index] = candidate_id;
                proposal.errors[insert_index] = *error;
              }
              proposal.num_candidates = std::min(
                  proposal.num_candidates + 1,
                  proposal.point3d_ids.size());
            }
          };
          bool candidates_scored = false;
          for (std::size_t depth = 0;
               depth < parallel_component_transitivity && !frontier.empty();
               ++depth) {
            next_frontier.clear();
            for (const ObservationKey& source : frontier) {
              const auto* neighbors = adjacency.Find(source);
              if (neighbors == nullptr) {
                continue;
              }
              for (const ObservationKey& neighbor : *neighbors) {
                const auto offset_it = image_offsets.find(neighbor.first);
                if (offset_it == image_offsets.end()) {
                  continue;
                }
                const std::size_t neighbor_flat_index =
                    offset_it->second + neighbor.second;
                if (neighbor_flat_index >= visited_epochs.size() ||
                    visited_epochs[neighbor_flat_index] == current_epoch) {
                  continue;
                }
                visited_epochs[neighbor_flat_index] = current_epoch;
                next_frontier.push_back(neighbor);
                if (!read_only_reconstruction.ExistsImage(neighbor.first)) {
                  continue;
                }
                const Image& neighbor_image =
                    read_only_reconstruction.GetImage(neighbor.first);
                if (neighbor.second >= neighbor_image.points2d.size()) {
                  continue;
                }
                const Point2D& neighbor_point2d =
                    neighbor_image.points2d[neighbor.second];
                if (neighbor_point2d.HasPoint3D() &&
                    read_only_reconstruction.ExistsPoint3D(
                        neighbor_point2d.point3d_id) &&
                    std::find(candidate_point_ids.begin(),
                              candidate_point_ids.end(),
                              neighbor_point2d.point3d_id) ==
                        candidate_point_ids.end()) {
                  candidate_point_ids.push_back(
                      neighbor_point2d.point3d_id);
                }
              }
            }
            frontier.swap(next_frontier);
            // Four hops cover almost every useful House proposal. Only pay
            // for deeper layers when the first four layers
            // contain no geometrically attachable point, avoiding a full
            // giant-component expansion for already solved targets.
            if (depth + 1 == 4 &&
                parallel_component_transitivity > 4) {
              score_candidate_points();
              candidates_scored = proposal.num_candidates > 0;
              if (candidates_scored) {
                break;
              }
              candidate_point_ids.clear();
            }
          }
          if (!candidates_scored) {
            score_candidate_points();
          }
        }
      },
      static_cast<double>(std::max(1, parallel::getNumThreads())));

  std::vector<std::size_t> proposal_order;
  proposal_order.reserve(proposals.size());
  for (std::size_t index = 0; index < proposals.size(); ++index) {
    if (proposals[index].num_candidates > 0) {
      proposal_order.push_back(index);
    }
  }
  std::stable_sort(
      proposal_order.begin(), proposal_order.end(),
      [&proposals](std::size_t first, std::size_t second) {
        const CompletionProposal& lhs = proposals[first];
        const CompletionProposal& rhs = proposals[second];
        return lhs.errors[0] < rhs.errors[0] ||
               (lhs.errors[0] == rhs.errors[0] && lhs.target < rhs.target);
      });

  std::size_t num_completed = 0;
  ObservationManager observation_manager(reconstruction);
  for (const std::size_t proposal_index : proposal_order) {
    const CompletionProposal& proposal = proposals[proposal_index];
    Image& target_image = reconstruction->GetImage(proposal.target.first);
    if (target_image.points2d[proposal.target.second].HasPoint3D()) {
      continue;
    }
    for (std::size_t candidate_index = 0;
         candidate_index < proposal.num_candidates; ++candidate_index) {
      const Point3DId candidate_id = proposal.point3d_ids[candidate_index];
      if (!reconstruction->ExistsPoint3D(candidate_id)) {
        continue;
      }
      const Point3D& candidate = reconstruction->GetPoint3D(candidate_id);
      const bool already_observed_in_target =
          std::any_of(candidate.track.elements.begin(),
                      candidate.track.elements.end(),
                      [&proposal](const TrackElement& element) {
                        return element.image_id == proposal.target.first;
                      });
      if (already_observed_in_target) {
        continue;
      }
      if (observation_manager.AddObservation(
              candidate_id, proposal.target.first, proposal.target.second)) {
        ++num_completed;
        dirty_point_ids->insert(candidate_id);
        break;
      }
    }
  }
  return num_completed;
}

Point3DId MergePointTrack(
    Point3DId point3d_id,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    IncrementalTriangulatorSummary* summary) {
  while (reconstruction->ExistsPoint3D(point3d_id)) {
    std::set<Point3DId> candidates;
    const Track track = reconstruction->GetPoint3D(point3d_id).track;
    for (const TrackElement& element : track.elements) {
      const auto* neighbors =
          adjacency.Find({element.image_id, element.point2d_idx});
      if (neighbors == nullptr) {
        continue;
      }
      for (const ObservationKey& observation : *neighbors) {
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
  return reconstruction->ExistsPoint3D(point3d_id) ? point3d_id
                                                    : kInvalidPoint3DId;
}

}  // namespace

IncrementalTriangulatorCache::IncrementalTriangulatorCache(
    const MatchGraph& graph)
    : correspondence_adjacency(BuildCorrespondenceAdjacency(graph)) {}

IncrementalTriangulatorSummary TriangulateRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->IsImageRegistered(image_id) ||
      !HasPoseAndCamera(*reconstruction, image_id)) {
    summary.message = "Image is not registered or has no pose";
    return summary;
  }

  const CorrespondenceAdjacency local_adjacency =
      cache == nullptr ? BuildCorrespondenceAdjacency(graph)
                       : CorrespondenceAdjacency{};
  const CorrespondenceAdjacency& adjacency =
      cache == nullptr ? local_adjacency : cache->correspondence_adjacency;
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
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }
  const CorrespondenceAdjacency local_adjacency =
      cache == nullptr ? BuildCorrespondenceAdjacency(graph)
                       : CorrespondenceAdjacency{};
  const CorrespondenceAdjacency& adjacency =
      cache == nullptr ? local_adjacency : cache->correspondence_adjacency;
  std::vector<Point3DId> active_point_ids;
  active_point_ids.reserve(reconstruction->Points3D().size());
  for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
    (void)point3d;
    active_point_ids.push_back(point3d_id);
  }
  for (std::size_t iteration = 0;
       iteration < options.max_track_update_iterations &&
       !active_point_ids.empty();
       ++iteration) {
    const std::size_t updates_before =
        summary.num_completed_observations + summary.num_merged_points;
    std::set<Point3DId> dirty_point_ids;
    if (iteration == 0) {
      summary.num_completed_observations +=
          CompleteUnassignedObservationsFromComponents(
              adjacency, options, reconstruction, &dirty_point_ids);
    } else {
      for (const Point3DId point3d_id : active_point_ids) {
        const std::size_t completed = CompletePointTrack(
            point3d_id, adjacency, options, reconstruction);
        summary.num_completed_observations += completed;
        if (completed > 0 && reconstruction->ExistsPoint3D(point3d_id)) {
          dirty_point_ids.insert(point3d_id);
        }
      }
    }

    for (const Point3DId point3d_id : active_point_ids) {
      if (reconstruction->ExistsPoint3D(point3d_id)) {
        const std::size_t merged_before = summary.num_merged_points;
        const Point3DId merged_target = MergePointTrack(
            point3d_id, adjacency, options, reconstruction, &summary);
        if (summary.num_merged_points > merged_before &&
            merged_target != kInvalidPoint3DId) {
          dirty_point_ids.insert(merged_target);
        }
      }
    }
    const std::size_t updates_after =
        summary.num_completed_observations + summary.num_merged_points;
    if (updates_after == updates_before) {
      break;
    }
    active_point_ids.assign(dirty_point_ids.begin(), dirty_point_ids.end());
  }
  summary.message = "Global track completion and merging finished";
  return summary;
}

IncrementalTriangulatorSummary CompleteTracksByPoseAndDescriptors(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr ||
      options.pose_guided_min_descriptor_support < 2 ||
      options.pose_guided_max_reprojection_error_px <= 0.0 ||
      options.pose_guided_max_descriptor_angle_rad <= 0.0) {
    summary.message = "Pose-guided track completion is disabled";
    return summary;
  }

  struct Candidate {
    ImageId image_id = kInvalidImageId;
    Point2DIdx point2d_idx = kInvalidPoint2DIdx;
    Point3DId point3d_id = kInvalidPoint3DId;
    double second_similarity = -1.0;
    double best_similarity = -1.0;
    double reprojection_error_px = std::numeric_limits<double>::infinity();
    std::size_t track_length = 0;
  };

  std::map<ImageId, std::vector<double>> descriptor_norms;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    const FeatureSet* features = FindFeatures(graph, image_id);
    if (features == nullptr || features->descriptors.cols() != 128) {
      continue;
    }
    std::vector<double>& norms = descriptor_norms[image_id];
    norms.resize(static_cast<std::size_t>(features->descriptors.rows()));
    for (Eigen::Index row = 0; row < features->descriptors.rows(); ++row) {
      norms[static_cast<std::size_t>(row)] =
          features->descriptors.row(row).norm();
    }
  }

  const std::vector<ImageId> image_ids(
      reconstruction->RegisteredImageIds().begin(),
      reconstruction->RegisteredImageIds().end());
  std::vector<std::vector<Candidate>> candidates_by_image(image_ids.size());
  const Reconstruction& read_only_reconstruction = *reconstruction;
  const double cell_size = options.pose_guided_max_reprojection_error_px;
  const double min_similarity =
      std::cos(options.pose_guided_max_descriptor_angle_rad);
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(image_ids.size())),
      [&](const parallel::Range& range) {
        for (int image_index = range.start; image_index < range.end;
             ++image_index) {
          const ImageId image_id = image_ids[image_index];
          const Image& image = read_only_reconstruction.GetImage(image_id);
          const Camera& camera =
              read_only_reconstruction.GetCamera(image.camera_id);
          const FeatureSet* target_features = FindFeatures(graph, image_id);
          const auto target_norms_it = descriptor_norms.find(image_id);
          if (target_features == nullptr ||
              target_norms_it == descriptor_norms.end()) {
            continue;
          }
          const std::size_t num_features = std::min(
              image.points2d.size(), target_features->keypoints.size());
          std::map<std::pair<int, int>, std::vector<Point2DIdx>> grid;
          for (Point2DIdx point2d_idx = 0; point2d_idx < num_features;
               ++point2d_idx) {
            if (image.points2d[point2d_idx].HasPoint3D()) {
              continue;
            }
            const FeatureKeypoint& keypoint =
                target_features->keypoints[point2d_idx];
            grid[{static_cast<int>(std::floor(keypoint.x / cell_size)),
                  static_cast<int>(std::floor(keypoint.y / cell_size))}]
                .push_back(point2d_idx);
          }

          std::vector<Candidate>& image_candidates =
              candidates_by_image[static_cast<std::size_t>(image_index)];
          for (const auto& [point3d_id, point3d] :
               read_only_reconstruction.Points3D()) {
            if (point3d.track.Length() <
                options.pose_guided_min_descriptor_support) {
              continue;
            }
            const bool already_observed = std::any_of(
                point3d.track.elements.begin(), point3d.track.elements.end(),
                [image_id](const TrackElement& element) {
                  return element.image_id == image_id;
                });
            if (already_observed) {
              continue;
            }
            const std::optional<Eigen::Vector2d> projected =
                ProjectWorldPoint(camera, *image.cam_from_world, point3d.xyz);
            if (!projected.has_value()) {
              continue;
            }
            const int cell_x =
                static_cast<int>(std::floor(projected->x() / cell_size));
            const int cell_y =
                static_cast<int>(std::floor(projected->y() / cell_size));
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                const auto grid_it = grid.find({cell_x + dx, cell_y + dy});
                if (grid_it == grid.end()) {
                  continue;
                }
                for (const Point2DIdx target_idx : grid_it->second) {
                  const Eigen::Vector2d target_xy(
                      target_features->keypoints[target_idx].x,
                      target_features->keypoints[target_idx].y);
                  const double reprojection_error =
                      (*projected - target_xy).norm();
                  if (reprojection_error >
                      options.pose_guided_max_reprojection_error_px) {
                    continue;
                  }
                  const double target_norm =
                      target_norms_it->second[target_idx];
                  if (!(target_norm > 0.0)) {
                    continue;
                  }
                  double best_similarity = -1.0;
                  double second_similarity = -1.0;
                  std::size_t descriptor_support = 0;
                  for (const TrackElement& element : point3d.track.elements) {
                    const FeatureSet* source_features =
                        FindFeatures(graph, element.image_id);
                    const auto source_norms_it =
                        descriptor_norms.find(element.image_id);
                    if (source_features == nullptr ||
                        source_norms_it == descriptor_norms.end() ||
                        element.point2d_idx >=
                            static_cast<Point2DIdx>(
                                source_features->descriptors.rows())) {
                      continue;
                    }
                    const double source_norm =
                        source_norms_it->second[element.point2d_idx];
                    if (!(source_norm > 0.0)) {
                      continue;
                    }
                    const double similarity = std::clamp(
                        static_cast<double>(
                            target_features->descriptors.row(target_idx).dot(
                                source_features->descriptors.row(
                                    element.point2d_idx))) /
                            (target_norm * source_norm),
                        -1.0, 1.0);
                    ++descriptor_support;
                    if (similarity > best_similarity) {
                      second_similarity = best_similarity;
                      best_similarity = similarity;
                    } else if (similarity > second_similarity) {
                      second_similarity = similarity;
                    }
                  }
                  if (descriptor_support <
                          options.pose_guided_min_descriptor_support ||
                      second_similarity < min_similarity) {
                    continue;
                  }
                  image_candidates.push_back(
                      {image_id, target_idx, point3d_id, second_similarity,
                       best_similarity, reprojection_error,
                       point3d.track.Length()});
                }
              }
            }
          }
        }
      });

  std::vector<Candidate> candidates;
  for (std::vector<Candidate>& image_candidates : candidates_by_image) {
    candidates.insert(candidates.end(), image_candidates.begin(),
                      image_candidates.end());
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& lhs, const Candidate& rhs) {
                     if (lhs.second_similarity != rhs.second_similarity) {
                       return lhs.second_similarity > rhs.second_similarity;
                     }
                     if (lhs.best_similarity != rhs.best_similarity) {
                       return lhs.best_similarity > rhs.best_similarity;
                     }
                     if (lhs.reprojection_error_px !=
                         rhs.reprojection_error_px) {
                       return lhs.reprojection_error_px <
                              rhs.reprojection_error_px;
                     }
                     if (lhs.track_length != rhs.track_length) {
                       return lhs.track_length > rhs.track_length;
                     }
                     return std::tie(lhs.image_id, lhs.point2d_idx,
                                     lhs.point3d_id) <
                            std::tie(rhs.image_id, rhs.point2d_idx,
                                     rhs.point3d_id);
                   });

  ObservationManager observation_manager(reconstruction);
  for (const Candidate& candidate : candidates) {
    if (!reconstruction->ExistsPoint3D(candidate.point3d_id) ||
        !reconstruction->ExistsImage(candidate.image_id)) {
      continue;
    }
    Image& image = reconstruction->GetImage(candidate.image_id);
    if (candidate.point2d_idx >= image.points2d.size() ||
        image.points2d[candidate.point2d_idx].HasPoint3D()) {
      continue;
    }
    const Point3D& point3d =
        reconstruction->GetPoint3D(candidate.point3d_id);
    const bool already_observed = std::any_of(
        point3d.track.elements.begin(), point3d.track.elements.end(),
        [&candidate](const TrackElement& element) {
          return element.image_id == candidate.image_id;
        });
    if (!already_observed && observation_manager.AddObservation(
                                 candidate.point3d_id, candidate.image_id,
                                 candidate.point2d_idx)) {
      ++summary.num_completed_observations;
    }
  }
  summary.message = "Pose-guided track completion finished";
  return summary;
}

}  // namespace colite
