#include "colite/core/match_graph.h"

#include <algorithm>
#include <stdexcept>

namespace colite {
namespace {

std::vector<FeatureMatch> ReorderMatchesIfNeeded(
    ImageId image_id1,
    ImageId image_id2,
    const std::vector<FeatureMatch>& matches) {
  if (image_id1 <= image_id2) {
    return matches;
  }

  std::vector<FeatureMatch> reordered;
  reordered.reserve(matches.size());
  for (const FeatureMatch& match : matches) {
    reordered.push_back({match.idx2, match.idx1});
  }
  return reordered;
}

bool IsUsableVerifiedGeometry(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

}  // namespace

std::size_t MatchGraphEdge::NumInliers() const {
  return geometry == nullptr ? 0U : geometry->inlier_matches.size();
}

ImageId MatchGraphEdge::OtherImageId(ImageId image_id) const {
  if (image_id == pair.image_id1) {
    return pair.image_id2;
  }
  if (image_id == pair.image_id2) {
    return pair.image_id1;
  }
  throw std::invalid_argument("Image id is not part of this match graph edge");
}

void SetFeatures(MatchGraph* graph, ImageId image_id, const FeatureSet& features) {
  graph->features[image_id] = features;
}

void SetMatches(MatchGraph* graph,
                ImageId image_id1,
                ImageId image_id2,
                const std::vector<FeatureMatch>& matches) {
  if (image_id1 == image_id2) {
    throw std::invalid_argument("Cannot store matches for the same image");
  }

  const ImagePair pair = ImagePair::Ordered(image_id1, image_id2);
  graph->matches[pair] = ReorderMatchesIfNeeded(image_id1, image_id2, matches);
}

void SetVerifiedGeometry(MatchGraph* graph,
                         ImageId image_id1,
                         ImageId image_id2,
                         const TwoViewGeometry& geometry) {
  if (image_id1 == image_id2) {
    throw std::invalid_argument("Cannot store geometry for the same image");
  }
  graph->verified_geometries[ImagePair::Ordered(image_id1, image_id2)] =
      geometry;
}

const FeatureSet* FindFeatures(const MatchGraph& graph, ImageId image_id) {
  const auto it = graph.features.find(image_id);
  return it == graph.features.end() ? nullptr : &it->second;
}

const std::vector<FeatureMatch>* FindMatches(const MatchGraph& graph,
                                             ImageId image_id1,
                                             ImageId image_id2) {
  const auto it = graph.matches.find(ImagePair::Ordered(image_id1, image_id2));
  return it == graph.matches.end() ? nullptr : &it->second;
}

const TwoViewGeometry* FindVerifiedGeometry(const MatchGraph& graph,
                                            ImageId image_id1,
                                            ImageId image_id2) {
  const auto it =
      graph.verified_geometries.find(ImagePair::Ordered(image_id1, image_id2));
  return it == graph.verified_geometries.end() ? nullptr : &it->second;
}

std::vector<MatchGraphEdge> FindVerifiedNeighbors(const MatchGraph& graph,
                                                  ImageId image_id) {
  std::vector<MatchGraphEdge> edges;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if ((pair.image_id1 == image_id || pair.image_id2 == image_id) &&
        IsUsableVerifiedGeometry(geometry)) {
      edges.push_back({pair, &geometry});
    }
  }

  std::sort(edges.begin(), edges.end(),
            [](const MatchGraphEdge& lhs, const MatchGraphEdge& rhs) {
              if (lhs.NumInliers() != rhs.NumInliers()) {
                return lhs.NumInliers() > rhs.NumInliers();
              }
              return lhs.pair < rhs.pair;
            });
  return edges;
}

}  // namespace colite
