#include "colite/matching/two_view_geometry_batch.h"

#include <stdexcept>
#include <vector>

#include "colite/core/parallel.h"

namespace colite {
namespace {

bool IsVerifiedGeometryUsable(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

const Camera& RequireCamera(const std::map<ImageId, Camera>& cameras_by_image,
                            ImageId image_id) {
  const auto it = cameras_by_image.find(image_id);
  if (it == cameras_by_image.end()) {
    throw std::out_of_range("Missing camera for image");
  }
  return it->second;
}

const FeatureSet& RequireFeatures(const MatchGraph& graph, ImageId image_id) {
  const FeatureSet* features = FindFeatures(graph, image_id);
  if (features == nullptr) {
    throw std::out_of_range("Missing features for image");
  }
  return *features;
}

}  // namespace

TwoViewGeometryBatchSummary VerifyMatchGraphGeometries(
    const std::map<ImageId, Camera>& cameras_by_image,
    MatchGraph* graph,
    const TwoViewGeometryBatchOptions& options) {
  TwoViewGeometryBatchSummary summary;
  graph->verified_geometries.clear();

  using MatchEntry =
      std::pair<const ImagePair, std::vector<FeatureMatch>>;
  std::vector<const MatchEntry*> match_entries;
  match_entries.reserve(graph->matches.size());
  for (const auto& entry : graph->matches) {
    match_entries.push_back(&entry);
  }
  std::vector<TwoViewGeometry> geometries(match_entries.size());

  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(match_entries.size())),
      [&](const parallel::Range& range) {
        for (int index = range.start; index < range.end; ++index) {
          const auto& [pair, matches] = *match_entries[index];
          const Camera& camera1 =
              RequireCamera(cameras_by_image, pair.image_id1);
          const Camera& camera2 =
              RequireCamera(cameras_by_image, pair.image_id2);
          const FeatureSet& features1 =
              RequireFeatures(*graph, pair.image_id1);
          const FeatureSet& features2 =
              RequireFeatures(*graph, pair.image_id2);
          geometries[index] = EstimateTwoViewGeometry(
              camera1, features1, camera2, features2, matches,
              options.geometry_options);
        }
      });

  summary.num_input_pairs = match_entries.size();
  for (std::size_t index = 0; index < match_entries.size(); ++index) {
    const ImagePair& pair = match_entries[index]->first;
    const TwoViewGeometry& geometry = geometries[index];
    if (IsVerifiedGeometryUsable(geometry)) {
      graph->verified_geometries[pair] = geometry;
      ++summary.num_verified_pairs;
    } else {
      ++summary.num_degenerate_pairs;
      if (options.keep_degenerate) {
        graph->verified_geometries[pair] = geometry;
      }
    }
  }

  return summary;
}

}  // namespace colite
