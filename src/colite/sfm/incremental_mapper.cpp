#include "colite/sfm/incremental_mapper.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "colite/core/geometry.h"

namespace colite {
namespace {

struct ImageCandidate {
  ImageId image_id = kInvalidImageId;
  std::size_t num_visible_points = 0;
  std::size_t num_visible_observations = 0;
  double mean_track_length = 0.0;
  std::size_t num_registered_neighbors = 0;
  std::size_t num_verified_inliers_to_registered_images = 0;
  double score = 0.0;
  std::size_t num_registration_attempts = 0;
};

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

bool HasValidFeatureIndex(const FeatureSet* features, Point2DIdx idx) {
  return features != nullptr && idx != kInvalidPoint2DIdx &&
         idx < features->keypoints.size();
}

double ComputeCandidateScore(const ImageCandidate& candidate) {
  return static_cast<double>(candidate.num_visible_points) * 1000.0 +
         static_cast<double>(candidate.num_registered_neighbors) * 100.0 +
         candidate.mean_track_length * 20.0 +
         static_cast<double>(
             candidate.num_verified_inliers_to_registered_images) *
             0.01;
}

ImageCandidate AnalyzeImageCandidate(const MatchGraph& graph,
                                     ImageId image_id,
                                     const Reconstruction& reconstruction) {
  ImageCandidate candidate;
  candidate.image_id = image_id;
  if (!reconstruction.ExistsImage(image_id) ||
      reconstruction.IsImageRegistered(image_id)) {
    return candidate;
  }

  const FeatureSet* target_features = FindFeatures(graph, image_id);
  if (target_features == nullptr) {
    return candidate;
  }

  std::set<ImageId> registered_neighbors;
  std::set<Point2DIdx> visible_target_points;
  std::set<Point3DId> visible_point3d_ids;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction.ExistsImage(neighbor_id) ||
        !reconstruction.IsImageRegistered(neighbor_id)) {
      continue;
    }

    registered_neighbors.insert(neighbor_id);
    candidate.num_verified_inliers_to_registered_images +=
        geometry.inlier_matches.size();

    const Image& neighbor_image = reconstruction.GetImage(neighbor_id);
    const bool target_is_first = pair.image_id1 == image_id;
    for (const FeatureMatch& match : geometry.inlier_matches) {
      const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
      const Point2DIdx neighbor_idx = target_is_first ? match.idx2 : match.idx1;
      if (!HasValidFeatureIndex(target_features, target_idx) ||
          neighbor_idx == kInvalidPoint2DIdx ||
          neighbor_idx >= neighbor_image.points2d.size()) {
        continue;
      }

      const Point2D& neighbor_point = neighbor_image.points2d[neighbor_idx];
      if (neighbor_point.HasPoint3D() &&
          reconstruction.ExistsPoint3D(neighbor_point.point3d_id)) {
        visible_target_points.insert(target_idx);
        visible_point3d_ids.insert(neighbor_point.point3d_id);
      }
    }
  }

  candidate.num_visible_points = visible_target_points.size();
  for (const Point3DId point3d_id : visible_point3d_ids) {
    candidate.num_visible_observations +=
        reconstruction.GetPoint3D(point3d_id).track.Length();
  }
  if (!visible_point3d_ids.empty()) {
    candidate.mean_track_length =
        static_cast<double>(candidate.num_visible_observations) /
        static_cast<double>(visible_point3d_ids.size());
  }
  candidate.num_registered_neighbors = registered_neighbors.size();
  candidate.score = ComputeCandidateScore(candidate);
  return candidate;
}

std::vector<ImageCandidate> CollectImageCandidates(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction,
    const std::map<ImageId, std::size_t>* registration_trials = nullptr,
    std::size_t max_num_registration_trials_per_image = 0) {
  std::vector<ImageCandidate> candidates;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    (void)image;
    ImageCandidate candidate =
        AnalyzeImageCandidate(graph, image_id, reconstruction);
    if (registration_trials != nullptr) {
      const auto trial_it = registration_trials->find(image_id);
      if (trial_it != registration_trials->end()) {
        candidate.num_registration_attempts = trial_it->second;
      }
    }
    if (max_num_registration_trials_per_image > 0 &&
        candidate.num_registration_attempts >=
            max_num_registration_trials_per_image) {
      continue;
    }
    if (candidate.num_visible_points >=
        options.registration.min_num_correspondences) {
      candidates.push_back(candidate);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ImageCandidate& lhs, const ImageCandidate& rhs) {
              if (lhs.num_registration_attempts !=
                  rhs.num_registration_attempts) {
                return lhs.num_registration_attempts <
                       rhs.num_registration_attempts;
              }
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              if (lhs.num_visible_points != rhs.num_visible_points) {
                return lhs.num_visible_points > rhs.num_visible_points;
              }
              if (lhs.num_visible_observations !=
                  rhs.num_visible_observations) {
                return lhs.num_visible_observations >
                       rhs.num_visible_observations;
              }
              if (lhs.num_registered_neighbors !=
                  rhs.num_registered_neighbors) {
                return lhs.num_registered_neighbors >
                       rhs.num_registered_neighbors;
              }
              if (lhs.num_verified_inliers_to_registered_images !=
                  rhs.num_verified_inliers_to_registered_images) {
                return lhs.num_verified_inliers_to_registered_images >
                       rhs.num_verified_inliers_to_registered_images;
              }
              return lhs.image_id < rhs.image_id;
            });
  return candidates;
}

void CopyCandidateStatsToSummary(const ImageCandidate& candidate,
                                 IncrementalMapperSummary* summary) {
  summary->candidate_num_visible_points = candidate.num_visible_points;
  summary->candidate_num_visible_observations =
      candidate.num_visible_observations;
  summary->candidate_mean_track_length = candidate.mean_track_length;
  summary->candidate_num_registered_neighbors =
      candidate.num_registered_neighbors;
  summary->candidate_num_verified_inliers_to_registered_images =
      candidate.num_verified_inliers_to_registered_images;
  summary->candidate_score = candidate.score;
  summary->registration_attempts_before_success =
      candidate.num_registration_attempts;
}

ImageRegistrationFailureSummary MakeFailureSummary(
    const ImageCandidate& candidate,
    std::size_t attempt,
    const std::string& message) {
  ImageRegistrationFailureSummary failure;
  failure.image_id = candidate.image_id;
  failure.attempt = attempt;
  failure.candidate_num_visible_points = candidate.num_visible_points;
  failure.candidate_num_visible_observations =
      candidate.num_visible_observations;
  failure.candidate_mean_track_length = candidate.mean_track_length;
  failure.candidate_num_registered_neighbors =
      candidate.num_registered_neighbors;
  failure.candidate_num_verified_inliers_to_registered_images =
      candidate.num_verified_inliers_to_registered_images;
  failure.candidate_score = candidate.score;
  failure.message = message;
  return failure;
}

std::set<Point3DId> CollectPoint3DIds(const Reconstruction& reconstruction) {
  std::set<Point3DId> point3d_ids;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d;
    point3d_ids.insert(point3d_id);
  }
  return point3d_ids;
}

bool HasLargeReprojectionError(const Reconstruction& reconstruction,
                               const Point3D& point3d,
                               const TrackElement& element,
                               double max_error_px) {
  if (!reconstruction.ExistsImage(element.image_id)) {
    return true;
  }
  const Image& image = reconstruction.GetImage(element.image_id);
  if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
      element.point2d_idx >= image.points2d.size()) {
    return true;
  }
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const auto error =
      ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                        image.points2d[element.point2d_idx].xy);
  return !error.has_value() || *error > max_error_px;
}

double MaxTriangulationAngleDegrees(const Reconstruction& reconstruction,
                                    const Point3D& point3d) {
  double max_angle = 0.0;
  for (std::size_t i = 0; i < point3d.track.elements.size(); ++i) {
    const TrackElement& element1 = point3d.track.elements[i];
    if (!reconstruction.ExistsImage(element1.image_id)) {
      continue;
    }
    const Image& image1 = reconstruction.GetImage(element1.image_id);
    if (!image1.HasPose() || !reconstruction.ExistsCamera(image1.camera_id) ||
        element1.point2d_idx >= image1.points2d.size()) {
      continue;
    }
    const Camera& camera1 = reconstruction.GetCamera(image1.camera_id);
    const Eigen::Vector3d bearing1 =
        PixelToBearing(camera1, image1.points2d[element1.point2d_idx].xy);

    for (std::size_t j = i + 1; j < point3d.track.elements.size(); ++j) {
      const TrackElement& element2 = point3d.track.elements[j];
      if (!reconstruction.ExistsImage(element2.image_id)) {
        continue;
      }
      const Image& image2 = reconstruction.GetImage(element2.image_id);
      if (!image2.HasPose() ||
          !reconstruction.ExistsCamera(image2.camera_id) ||
          element2.point2d_idx >= image2.points2d.size()) {
        continue;
      }
      const Camera& camera2 = reconstruction.GetCamera(image2.camera_id);
      const Eigen::Vector3d bearing2 =
          PixelToBearing(camera2, image2.points2d[element2.point2d_idx].xy);
      max_angle = std::max(max_angle,
                           TriangulationAngleDegrees(*image1.cam_from_world,
                                                     bearing1,
                                                     *image2.cam_from_world,
                                                     bearing2));
    }
  }
  return max_angle;
}

ObservationFilterSummary FilterNewlyCreatedPoints(
    const std::set<Point3DId>& point_ids_before,
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction) {
  ObservationFilterSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  std::vector<Point3DId> points_to_delete;
  for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
    if (point_ids_before.find(point3d_id) != point_ids_before.end()) {
      continue;
    }
    bool has_bad_observation = false;
    for (const TrackElement& element : point3d.track.elements) {
      if (HasLargeReprojectionError(*reconstruction, point3d, element,
                                    options.max_reprojection_error_px)) {
        has_bad_observation = true;
        break;
      }
    }
    if (has_bad_observation ||
        point3d.track.Length() < options.min_track_length ||
        MaxTriangulationAngleDegrees(*reconstruction, point3d) <
            options.min_triangulation_angle_deg) {
      points_to_delete.push_back(point3d_id);
    }
  }

  for (const Point3DId point3d_id : points_to_delete) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const std::size_t track_length =
        reconstruction->GetPoint3D(point3d_id).track.Length();
    if (reconstruction->DeletePoint3D(point3d_id)) {
      ++summary.num_removed_points;
      summary.num_removed_observations += track_length;
    }
  }

  summary.success = true;
  summary.message = "Local retriangulation filtering finished";
  return summary;
}

}  // namespace

IncrementalMapperSummary RegisterNextImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalMapperOptions& options,
    Reconstruction* reconstruction) {
  IncrementalMapperSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }

  // 1. 通过已注册图像上的 2D-3D 关联估计新图像位姿。
  summary.registration =
      RegisterImageWithExistingPoints(graph, image_id, options.registration,
                                      reconstruction);
  if (!summary.registration.success) {
    summary.message = summary.registration.message;
    return summary;
  }

  // 2. 先把新图像能挂到已有 3D 点上的观测补齐，再创建新的 track。
  summary.track_completion = CompleteTracksForRegisteredImage(
      graph, image_id, options.triangulation, reconstruction);
  summary.triangulation = TriangulateRegisteredImage(
      graph, image_id, options.triangulation, reconstruction);
  const ObservationFilterOptions* local_retriangulation_filter =
      options.run_local_retriangulation_filter
          ? &options.local_retriangulation_filter
          : nullptr;
  summary.local_retriangulation = RetriangulateLocalNeighborhood(
      graph, image_id, options.triangulation, local_retriangulation_filter,
      reconstruction);

  // 3. 对新图像及其共视邻域做局部 BA，消化 PnP 和三角化带来的误差。
  if (options.run_local_bundle_adjustment) {
    summary.local_bundle_adjustment = AdjustLocalBundle(
        image_id, options.local_bundle_adjustment, reconstruction);
    if (!summary.local_bundle_adjustment.success) {
      summary.message = summary.local_bundle_adjustment.message;
      return summary;
    }
  }

  // 4. BA 后过滤坏观测和不稳定 3D 点，减少错误 track 继续污染后续注册。
  if (options.run_observation_filter) {
    summary.observation_filter =
        FilterObservationsAndPoints(options.observation_filter, reconstruction);
    if (!summary.observation_filter.success) {
      summary.message = summary.observation_filter.message;
      return summary;
    }
  }

  summary.success = true;
  summary.message = "Image registered and locally refined";
  return summary;
}

IncrementalTriangulatorSummary RetriangulateLocalNeighborhood(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    const ObservationFilterOptions* local_filter_options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;
  if (reconstruction == nullptr ||
      !reconstruction->IsImageRegistered(image_id)) {
    summary.message = "Image is not registered";
    return summary;
  }

  const std::set<Point3DId> point_ids_before =
      CollectPoint3DIds(*reconstruction);
  std::set<ImageId> local_image_ids;
  local_image_ids.insert(image_id);
  for (const MatchGraphEdge& edge : FindVerifiedNeighbors(graph, image_id)) {
    const ImageId neighbor_id = edge.OtherImageId(image_id);
    if (reconstruction->IsImageRegistered(neighbor_id)) {
      local_image_ids.insert(neighbor_id);
    }
  }

  // COLMAP 的增量三角化会围绕新注册图像反复 complete/merge/create。这里先做
  // 保守的局部版本：新图像和强邻居都重跑一轮，使邻居-邻居遗漏 track 进入局部 BA。
  for (const ImageId local_image_id : local_image_ids) {
    const IncrementalTriangulatorSummary completion =
        CompleteTracksForRegisteredImage(graph, local_image_id, options,
                                         reconstruction);
    summary.num_completed_observations += completion.num_completed_observations;
    summary.num_merged_points += completion.num_merged_points;
    summary.num_merged_observations += completion.num_merged_observations;

    const IncrementalTriangulatorSummary triangulation =
        TriangulateRegisteredImage(graph, local_image_id, options,
                                   reconstruction);
    summary.num_created_points += triangulation.num_created_points;
    summary.num_created_observations +=
        triangulation.num_created_observations;
  }

  if (local_filter_options != nullptr) {
    const ObservationFilterSummary filter_summary = FilterNewlyCreatedPoints(
        point_ids_before, *local_filter_options, reconstruction);
    summary.num_filtered_points = filter_summary.num_removed_points;
    summary.num_filtered_observations =
        filter_summary.num_removed_observations;
  }

  summary.message = "Local retriangulation finished";
  return summary;
}

std::optional<ImageId> SelectNextImage(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction) {
  const std::vector<ImageCandidate> candidates =
      CollectImageCandidates(graph, options, reconstruction);
  if (candidates.empty()) {
    return std::nullopt;
  }
  return candidates.front().image_id;
}

IncrementalMappingSummary RunIncrementalMapping(
    const MatchGraph& graph,
    const IncrementalMappingOptions& options,
    Reconstruction* reconstruction) {
  IncrementalMappingSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  if (reconstruction->RegisteredImageIds().empty()) {
    summary.initialization = InitializeReconstructionFromBestPair(
        graph, options.initialization, reconstruction);
    if (!summary.initialization.success) {
      summary.num_registered_images =
          reconstruction->RegisteredImageIds().size();
      summary.num_points3d = reconstruction->Points3D().size();
      summary.message = summary.initialization.message;
      return summary;
    }
  }

  std::map<ImageId, std::size_t> registration_trials;
  for (std::size_t i = 0; i < options.max_num_registrations; ++i) {
    const std::vector<ImageCandidate> candidates =
        CollectImageCandidates(graph, options.mapper, *reconstruction,
                               &registration_trials,
                               options.max_num_registration_trials_per_image);
    if (candidates.empty()) {
      break;
    }

    bool registered_any_image = false;
    for (const ImageCandidate& candidate : candidates) {
      IncrementalMapperSummary registration = RegisterNextImage(
          graph, candidate.image_id, options.mapper, reconstruction);
      if (!registration.success) {
        const std::size_t attempt = ++registration_trials[candidate.image_id];
        summary.failed_registrations.push_back(MakeFailureSummary(
            candidate, attempt, registration.message));
        continue;
      }

      registration.registration_order = summary.registrations.size() + 1;
      CopyCandidateStatsToSummary(candidate, &registration);
      summary.registrations.push_back(std::move(registration));
      registered_any_image = true;
      break;
    }

    if (!registered_any_image) {
      break;
    }
  }

  if (options.run_global_track_completion &&
      reconstruction->RegisteredImageIds().size() >= 2) {
    summary.global_track_completion.message =
        "Global track completion finished";
    const std::vector<ImageId> registered_image_ids(
        reconstruction->RegisteredImageIds().begin(),
        reconstruction->RegisteredImageIds().end());
    for (const ImageId image_id : registered_image_ids) {
      const IncrementalTriangulatorSummary completion =
          CompleteTracksForRegisteredImage(graph, image_id,
                                           options.mapper.triangulation,
                                           reconstruction);
      summary.global_track_completion.num_completed_observations +=
          completion.num_completed_observations;
    }
  }

  if (options.run_global_bundle_adjustment &&
      reconstruction->RegisteredImageIds().size() >= 2 &&
      !reconstruction->Points3D().empty()) {
    summary.global_bundle_adjustment = AdjustGlobalBundle(
        options.global_bundle_adjustment, reconstruction);
    if (!summary.global_bundle_adjustment.success) {
      summary.num_registered_images =
          reconstruction->RegisteredImageIds().size();
      summary.num_points3d = reconstruction->Points3D().size();
      summary.message = summary.global_bundle_adjustment.message;
      return summary;
    }

    if (options.run_global_observation_filter) {
      summary.global_observation_filter = FilterObservationsAndPoints(
          options.global_observation_filter, reconstruction);
      if (!summary.global_observation_filter.success) {
        summary.num_registered_images =
            reconstruction->RegisteredImageIds().size();
        summary.num_points3d = reconstruction->Points3D().size();
        summary.message = summary.global_observation_filter.message;
        return summary;
      }
    }

    if (options.run_final_global_bundle_adjustment &&
        reconstruction->RegisteredImageIds().size() >= 2 &&
        !reconstruction->Points3D().empty()) {
      summary.final_global_bundle_adjustment = AdjustGlobalBundle(
          options.global_bundle_adjustment, reconstruction);
      if (!summary.final_global_bundle_adjustment.success) {
        summary.num_registered_images =
            reconstruction->RegisteredImageIds().size();
        summary.num_points3d = reconstruction->Points3D().size();
        summary.message = summary.final_global_bundle_adjustment.message;
        return summary;
      }
    }
  }

  summary.num_registered_images = reconstruction->RegisteredImageIds().size();
  summary.num_points3d = reconstruction->Points3D().size();
  summary.success = summary.num_registered_images >= 2;
  summary.message = summary.success ? "Incremental mapping finished"
                                    : "Incremental mapping failed";
  return summary;
}

}  // namespace colite
