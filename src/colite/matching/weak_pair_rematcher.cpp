#include "colite/matching/weak_pair_rematcher.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <vector>

#include "colite/core/parallel.h"

namespace colite {
namespace {

bool IsNearbyPair(const ImagePair& pair, std::size_t nearby_image_window) {
  if (nearby_image_window == 0) {
    return true;
  }
  const ImageId diff = pair.image_id1 > pair.image_id2
                           ? pair.image_id1 - pair.image_id2
                           : pair.image_id2 - pair.image_id1;
  return diff <= nearby_image_window;
}

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

struct RematchCandidate {
  bool checked = false;
  bool rematched = false;
  bool improved = false;
  std::vector<FeatureMatch> matches;
  TwoViewGeometry geometry;
  GuidedDescriptorMatchingStats guided_stats;
  double geometry_verification_seconds = 0.0;
};

const Camera* FindCamera(const std::map<ImageId, Camera>& cameras_by_image,
                         ImageId image_id) {
  const auto it = cameras_by_image.find(image_id);
  return it == cameras_by_image.end() ? nullptr : &it->second;
}

}  // namespace

WeakPairRematchingSummary RematchWeakImagePairs(
    const std::map<ImageId, Camera>& cameras_by_image,
    const WeakPairRematchingOptions& options,
    MatchGraph* graph) {
  WeakPairRematchingSummary summary;
  if (graph == nullptr) {
    return summary;
  }

  const std::vector<ImagePair> pairs = [&graph]() {
    std::vector<ImagePair> result;
    result.reserve(graph->matches.size());
    for (const auto& [pair, matches] : graph->matches) {
      (void)matches;
      result.push_back(pair);
    }
    return result;
  }();

  std::vector<RematchCandidate> candidates(pairs.size());
  std::map<ImageId, std::unique_ptr<PreparedFeatureMatcher>>
      prepared_features;
  for (const auto& [image_id, features] : graph->features) {
    prepared_features.emplace(
        image_id, std::make_unique<PreparedFeatureMatcher>(
                      features, options.relaxed_matching.backend));
  }
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(pairs.size())),
      [&](const parallel::Range& range) {
        for (int pair_idx = range.start; pair_idx < range.end; ++pair_idx) {
          const ImagePair& pair =
              pairs[static_cast<std::size_t>(pair_idx)];
          RematchCandidate& candidate =
              candidates[static_cast<std::size_t>(pair_idx)];
          if (!IsNearbyPair(pair, options.nearby_image_window)) {
            continue;
          }
          candidate.checked = true;

          const FeatureSet* features1 = FindFeatures(*graph, pair.image_id1);
          const FeatureSet* features2 = FindFeatures(*graph, pair.image_id2);
          const auto prepared1_it = prepared_features.find(pair.image_id1);
          const auto prepared2_it = prepared_features.find(pair.image_id2);
          const Camera* camera1 = FindCamera(cameras_by_image, pair.image_id1);
          const Camera* camera2 = FindCamera(cameras_by_image, pair.image_id2);
          const TwoViewGeometry* old_geometry =
              FindVerifiedGeometry(*graph, pair.image_id1, pair.image_id2);
          if (features1 == nullptr || features2 == nullptr ||
              prepared1_it == prepared_features.end() ||
              prepared2_it == prepared_features.end() ||
              camera1 == nullptr || camera2 == nullptr ||
              old_geometry == nullptr || !IsUsableGeometry(*old_geometry) ||
              old_geometry->F.norm() <= 0.0) {
            continue;
          }
          const std::size_t old_num_inliers =
              old_geometry->inlier_matches.size();
          const std::vector<FeatureMatch>* old_matches =
              FindMatches(*graph, pair.image_id1, pair.image_id2);
          const std::size_t old_num_matches =
              old_matches == nullptr ? 0 : old_matches->size();
          const std::vector<FeatureMatch> guided_matches =
              MatchPreparedFeatureSetsGuidedFundamental(
                  *prepared1_it->second, *prepared2_it->second,
                  old_geometry->F,
                  options.guided_matching.max_sampson_error_px,
                  options.relaxed_matching, &candidate.guided_stats);
          candidate.rematched = true;
          if (guided_matches.size() <
              old_num_inliers + options.min_inlier_gain) {
            continue;
          }

          // Match COLMAP's guided matching pipeline: the first-pass verifier
          // estimates E/F/H and the guided matcher then updates only the
          // inlier correspondences under that verified geometry. Running a
          // second full E/F/H RANSAC here duplicates the expensive verifier
          // and can classify the same pair differently solely from a new
          // random sample order.
          TwoViewGeometry new_geometry = *old_geometry;
          new_geometry.inlier_matches = guided_matches;
          const double old_inlier_ratio =
              old_num_matches == 0
                  ? 0.0
                  : static_cast<double>(old_num_inliers) / old_num_matches;
          const double new_inlier_ratio =
              guided_matches.empty()
                  ? 0.0
                  : static_cast<double>(new_geometry.inlier_matches.size()) /
                        guided_matches.size();
          if (!IsUsableGeometry(new_geometry) ||
              new_geometry.inlier_matches.size() <
                  old_num_inliers + options.min_inlier_gain ||
              new_inlier_ratio + 1e-12 < old_inlier_ratio) {
            continue;
          }

          candidate.matches = guided_matches;
          candidate.geometry = new_geometry;
          candidate.improved = true;
        }
      });

  for (std::size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
    const RematchCandidate& candidate = candidates[pair_idx];
    summary.num_checked_pairs += candidate.checked ? 1 : 0;
    summary.num_rematched_pairs += candidate.rematched ? 1 : 0;
    summary.candidate_generation_work_seconds +=
        candidate.guided_stats.candidate_generation_seconds;
    summary.descriptor_search_work_seconds +=
        candidate.guided_stats.descriptor_search_seconds;
    summary.geometry_verification_work_seconds +=
        candidate.geometry_verification_seconds;
    if (!candidate.improved) {
      continue;
    }
    const ImagePair& pair = pairs[pair_idx];
    SetMatches(graph, pair.image_id1, pair.image_id2, candidate.matches);
    SetVerifiedGeometry(graph, pair.image_id1, pair.image_id2,
                        candidate.geometry);
    ++summary.num_improved_pairs;
  }

  return summary;
}

}  // namespace colite
