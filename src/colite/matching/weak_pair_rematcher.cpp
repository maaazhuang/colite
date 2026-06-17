#include "colite/matching/weak_pair_rematcher.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

  for (const ImagePair& pair : pairs) {
    if (!IsNearbyPair(pair, options.nearby_image_window)) {
      continue;
    }
    ++summary.num_checked_pairs;

    const FeatureSet* features1 = FindFeatures(*graph, pair.image_id1);
    const FeatureSet* features2 = FindFeatures(*graph, pair.image_id2);
    const Camera* camera1 = FindCamera(cameras_by_image, pair.image_id1);
    const Camera* camera2 = FindCamera(cameras_by_image, pair.image_id2);
    const TwoViewGeometry* old_geometry =
        FindVerifiedGeometry(*graph, pair.image_id1, pair.image_id2);
    if (features1 == nullptr || features2 == nullptr || camera1 == nullptr ||
        camera2 == nullptr || old_geometry == nullptr ||
        !IsUsableGeometry(*old_geometry) || old_geometry->F.norm() <= 0.0) {
      continue;
    }

    const std::size_t old_num_inliers = old_geometry->inlier_matches.size();
    const std::vector<FeatureMatch> relaxed_matches =
        MatchFeatureSets(*features1, *features2, options.relaxed_matching);
    ++summary.num_rematched_pairs;

    const std::vector<FeatureMatch> guided_matches =
        FilterMatchesByGuidedFundamental(*features1, *features2,
                                         relaxed_matches, old_geometry->F,
                                         options.guided_matching);
    if (guided_matches.size() < old_num_inliers + options.min_inlier_gain) {
      continue;
    }

    const TwoViewGeometry new_geometry = EstimateTwoViewGeometry(
        *camera1, *features1, *camera2, *features2, guided_matches,
        options.verification);
    if (!IsUsableGeometry(new_geometry) ||
        new_geometry.inlier_matches.size() <
            old_num_inliers + options.min_inlier_gain) {
      continue;
    }

    SetMatches(graph, pair.image_id1, pair.image_id2, guided_matches);
    SetVerifiedGeometry(graph, pair.image_id1, pair.image_id2, new_geometry);
    ++summary.num_improved_pairs;
  }

  return summary;
}

}  // namespace colite
