#include "colite/sfm/incremental_mapper.h"

#include <algorithm>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace colite {
namespace {

struct ImageCandidate {
  ImageId image_id = kInvalidImageId;
  std::size_t num_visible_points = 0;
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

std::size_t CountVisibleExistingPoints(const MatchGraph& graph,
                                       ImageId image_id,
                                       const Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(image_id) ||
      reconstruction.IsImageRegistered(image_id)) {
    return 0;
  }

  const FeatureSet* target_features = FindFeatures(graph, image_id);
  if (target_features == nullptr) {
    return 0;
  }

  std::set<Point2DIdx> visible_target_points;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction.ExistsImage(neighbor_id) ||
        !reconstruction.IsImageRegistered(neighbor_id)) {
      continue;
    }

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
      }
    }
  }

  return visible_target_points.size();
}

std::vector<ImageCandidate> CollectImageCandidates(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction) {
  std::vector<ImageCandidate> candidates;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    (void)image;
    const std::size_t num_visible_points =
        CountVisibleExistingPoints(graph, image_id, reconstruction);
    if (num_visible_points >= options.registration.min_num_correspondences) {
      candidates.push_back({image_id, num_visible_points});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ImageCandidate& lhs, const ImageCandidate& rhs) {
              if (lhs.num_visible_points != rhs.num_visible_points) {
                return lhs.num_visible_points > rhs.num_visible_points;
              }
              return lhs.image_id < rhs.image_id;
            });
  return candidates;
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

  for (std::size_t i = 0; i < options.max_num_registrations; ++i) {
    const std::vector<ImageCandidate> candidates =
        CollectImageCandidates(graph, options.mapper, *reconstruction);
    if (candidates.empty()) {
      break;
    }

    bool registered_any_image = false;
    for (const ImageCandidate& candidate : candidates) {
      IncrementalMapperSummary registration = RegisterNextImage(
          graph, candidate.image_id, options.mapper, reconstruction);
      if (!registration.success) {
        continue;
      }

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
