#include "colite/matching/two_view_geometry_batch.h"

#include <stdexcept>

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

  for (const auto& [pair, matches] : graph->matches) {
    ++summary.num_input_pairs;

    const Camera& camera1 = RequireCamera(cameras_by_image, pair.image_id1);
    const Camera& camera2 = RequireCamera(cameras_by_image, pair.image_id2);
    const FeatureSet& features1 = RequireFeatures(*graph, pair.image_id1);
    const FeatureSet& features2 = RequireFeatures(*graph, pair.image_id2);

    const TwoViewGeometry geometry =
        EstimateTwoViewGeometry(camera1, features1, camera2, features2,
                                matches, options.geometry_options);
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
