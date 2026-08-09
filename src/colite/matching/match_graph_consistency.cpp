#include "colite/matching/match_graph_consistency.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace colite {
namespace {

struct Observation {
  ImageId image_id = kInvalidImageId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;

  bool operator<(const Observation& other) const {
    return std::tie(image_id, point2d_idx) <
           std::tie(other.image_id, other.point2d_idx);
  }
};

struct VerifiedEdge {
  ImagePair pair;
  FeatureMatch match;
  std::size_t geometry_match_idx = 0;
  std::size_t node1 = 0;
  std::size_t node2 = 0;
  std::size_t triangle_support = 0;
  std::size_t min_support = 0;
  std::size_t total_support = 0;
  double squared_sampson_error = std::numeric_limits<double>::infinity();
};

class DisjointSets {
 public:
  explicit DisjointSets(const std::vector<Observation>& observations)
      : parent_(observations.size()),
        rank_(observations.size(), 0),
        images_(observations.size()) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    for (std::size_t idx = 0; idx < observations.size(); ++idx) {
      images_[idx].insert(observations[idx].image_id);
    }
  }

  std::size_t Find(std::size_t node) {
    if (parent_[node] != node) {
      parent_[node] = Find(parent_[node]);
    }
    return parent_[node];
  }

  bool HasImageOverlap(std::size_t node1, std::size_t node2) {
    const std::size_t root1 = Find(node1);
    const std::size_t root2 = Find(node2);
    if (root1 == root2) {
      return false;
    }
    const auto* smaller = &images_[root1];
    const auto* larger = &images_[root2];
    if (smaller->size() > larger->size()) {
      std::swap(smaller, larger);
    }
    for (const ImageId image_id : *smaller) {
      if (larger->count(image_id) != 0) {
        return true;
      }
    }
    return false;
  }

  void Union(std::size_t node1, std::size_t node2) {
    std::size_t root1 = Find(node1);
    std::size_t root2 = Find(node2);
    if (root1 == root2) {
      return;
    }
    if (rank_[root1] < rank_[root2]) {
      std::swap(root1, root2);
    }
    parent_[root2] = root1;
    images_[root1].insert(images_[root2].begin(), images_[root2].end());
    images_[root2].clear();
    if (rank_[root1] == rank_[root2]) {
      ++rank_[root1];
    }
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<std::size_t> rank_;
  std::vector<std::set<ImageId>> images_;
};

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

bool IsValidMatch(const MatchGraph& graph,
                  const ImagePair& pair,
                  const FeatureMatch& match) {
  if (match.idx1 == kInvalidPoint2DIdx ||
      match.idx2 == kInvalidPoint2DIdx) {
    return false;
  }
  const FeatureSet* features1 = FindFeatures(graph, pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, pair.image_id2);
  return (features1 == nullptr ||
          match.idx1 < static_cast<Point2DIdx>(features1->keypoints.size())) &&
         (features2 == nullptr ||
          match.idx2 < static_cast<Point2DIdx>(features2->keypoints.size()));
}

double SquaredSampsonError(const MatchGraph& graph,
                           const VerifiedEdge& edge,
                           const TwoViewGeometry& geometry) {
  const FeatureSet* features1 = FindFeatures(graph, edge.pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, edge.pair.image_id2);
  if (features1 == nullptr || features2 == nullptr ||
      geometry.F.norm() <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const FeatureKeypoint& keypoint1 = features1->keypoints[edge.match.idx1];
  const FeatureKeypoint& keypoint2 = features2->keypoints[edge.match.idx2];
  const Eigen::Vector3d x1(keypoint1.x, keypoint1.y, 1.0);
  const Eigen::Vector3d x2(keypoint2.x, keypoint2.y, 1.0);
  const Eigen::Vector3d fx1 = geometry.F * x1;
  const Eigen::Vector3d ftx2 = geometry.F.transpose() * x2;
  const double residual = x2.transpose() * geometry.F * x1;
  const double denominator =
      fx1.head<2>().squaredNorm() + ftx2.head<2>().squaredNorm();
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return std::numeric_limits<double>::infinity();
  }
  return residual * residual / denominator;
}

bool BetterEdge(const VerifiedEdge& lhs, const VerifiedEdge& rhs) {
  if (lhs.triangle_support != rhs.triangle_support) {
    return lhs.triangle_support > rhs.triangle_support;
  }
  if (lhs.min_support != rhs.min_support) {
    return lhs.min_support > rhs.min_support;
  }
  if (lhs.total_support != rhs.total_support) {
    return lhs.total_support > rhs.total_support;
  }
  if (lhs.squared_sampson_error != rhs.squared_sampson_error) {
    return lhs.squared_sampson_error < rhs.squared_sampson_error;
  }
  return std::tie(lhs.pair.image_id1, lhs.pair.image_id2, lhs.match.idx1,
                  lhs.match.idx2) <
         std::tie(rhs.pair.image_id1, rhs.pair.image_id2, rhs.match.idx1,
                  rhs.match.idx2);
}

void MeasureInputComponents(
    const std::vector<Observation>& observations,
    const std::vector<VerifiedEdge>& edges,
    MatchGraphConsistencySummary* summary) {
  DisjointSets sets(observations);
  for (const VerifiedEdge& edge : edges) {
    sets.Union(edge.node1, edge.node2);
  }

  std::map<std::size_t, std::map<ImageId, std::size_t>> image_counts;
  for (std::size_t node = 0; node < observations.size(); ++node) {
    ++image_counts[sets.Find(node)][observations[node].image_id];
  }
  summary->num_input_components = image_counts.size();
  for (const auto& [root, counts] : image_counts) {
    (void)root;
    bool conflicted = false;
    for (const auto& [image_id, count] : counts) {
      (void)image_id;
      if (count > 1) {
        conflicted = true;
        summary->num_input_conflicting_observations += count - 1;
      }
    }
    summary->num_input_conflicted_components += conflicted ? 1 : 0;
  }
}

}  // namespace

MatchGraphConsistencySummary EnforceVerifiedMatchGraphConsistency(
    MatchGraph* graph) {
  MatchGraphConsistencySummary summary;
  if (graph == nullptr) {
    return summary;
  }

  std::map<Observation, std::size_t> node_indices;
  std::vector<Observation> observations;
  std::vector<VerifiedEdge> edges;
  auto get_node = [&](const Observation& observation) {
    const auto [it, inserted] =
        node_indices.emplace(observation, observations.size());
    if (inserted) {
      observations.push_back(observation);
    }
    return it->second;
  };

  for (const auto& [pair, geometry] : graph->verified_geometries) {
    if (!IsUsableGeometry(geometry)) {
      continue;
    }
    summary.num_input_edges += geometry.inlier_matches.size();
    for (std::size_t match_idx = 0;
         match_idx < geometry.inlier_matches.size(); ++match_idx) {
      const FeatureMatch& match = geometry.inlier_matches[match_idx];
      if (!IsValidMatch(*graph, pair, match)) {
        ++summary.num_invalid_edges;
        continue;
      }
      VerifiedEdge edge;
      edge.pair = pair;
      edge.match = match;
      edge.geometry_match_idx = match_idx;
      edge.node1 = get_node({pair.image_id1, match.idx1});
      edge.node2 = get_node({pair.image_id2, match.idx2});
      edge.squared_sampson_error =
          SquaredSampsonError(*graph, edge, geometry);
      edges.push_back(edge);
    }
  }
  if (edges.empty()) {
    for (auto& [pair, geometry] : graph->verified_geometries) {
      (void)pair;
      geometry.inlier_matches.clear();
    }
    return summary;
  }

  std::vector<std::set<std::size_t>> neighbors(observations.size());
  for (const VerifiedEdge& edge : edges) {
    neighbors[edge.node1].insert(edge.node2);
    neighbors[edge.node2].insert(edge.node1);
  }
  for (VerifiedEdge& edge : edges) {
    const auto& neighbors1 = neighbors[edge.node1];
    const auto& neighbors2 = neighbors[edge.node2];
    auto it1 = neighbors1.begin();
    auto it2 = neighbors2.begin();
    while (it1 != neighbors1.end() && it2 != neighbors2.end()) {
      if (*it1 < *it2) {
        ++it1;
      } else if (*it2 < *it1) {
        ++it2;
      } else {
        ++edge.triangle_support;
        ++it1;
        ++it2;
      }
    }
    edge.min_support =
        std::min(neighbors[edge.node1].size(), neighbors[edge.node2].size());
    edge.total_support =
        neighbors[edge.node1].size() + neighbors[edge.node2].size();
  }
  MeasureInputComponents(observations, edges, &summary);
  std::stable_sort(edges.begin(), edges.end(), BetterEdge);

  // First normalize each individual image pair to a one-to-one assignment.
  std::map<ImagePair, std::set<Point2DIdx>> used_first;
  std::map<ImagePair, std::set<Point2DIdx>> used_second;
  std::map<ImagePair, std::set<std::pair<Point2DIdx, Point2DIdx>>> used_exact;
  std::vector<VerifiedEdge> pair_unique_edges;
  pair_unique_edges.reserve(edges.size());
  for (const VerifiedEdge& edge : edges) {
    const std::pair<Point2DIdx, Point2DIdx> exact_key{
        edge.match.idx1, edge.match.idx2};
    if (used_exact[edge.pair].count(exact_key) != 0) {
      ++summary.num_exact_duplicate_edges;
      continue;
    }
    used_exact[edge.pair].insert(exact_key);
    if (used_first[edge.pair].count(edge.match.idx1) != 0 ||
        used_second[edge.pair].count(edge.match.idx2) != 0) {
      ++summary.num_pair_conflict_edges;
      continue;
    }
    used_first[edge.pair].insert(edge.match.idx1);
    used_second[edge.pair].insert(edge.match.idx2);
    pair_unique_edges.push_back(edge);
  }

  // Preserve cross-pair evidence here. A small number of bad bridge edges can
  // join many otherwise valid tracks, so deleting an entire global component
  // before camera poses are known is destructive. The mapper resolves those
  // same-image alternatives using reprojection geometry.
  DisjointSets consistent_sets(observations);
  std::map<ImagePair, std::set<std::pair<Point2DIdx, Point2DIdx>>> retained;
  for (const VerifiedEdge& edge : pair_unique_edges) {
    consistent_sets.Union(edge.node1, edge.node2);
    retained[edge.pair].insert({edge.match.idx1, edge.match.idx2});
    ++summary.num_output_edges;
  }

  std::set<std::size_t> output_roots;
  for (const VerifiedEdge& edge : pair_unique_edges) {
    if (retained[edge.pair].count(
            {edge.match.idx1, edge.match.idx2}) == 0) {
      continue;
    }
    output_roots.insert(consistent_sets.Find(edge.node1));
  }
  summary.num_output_components = output_roots.size();

  for (auto& [pair, geometry] : graph->verified_geometries) {
    std::vector<FeatureMatch> filtered;
    filtered.reserve(geometry.inlier_matches.size());
    const auto retained_it = retained.find(pair);
    if (retained_it != retained.end()) {
      std::set<std::pair<Point2DIdx, Point2DIdx>> emitted;
      for (const FeatureMatch& match : geometry.inlier_matches) {
        const std::pair<Point2DIdx, Point2DIdx> key{match.idx1, match.idx2};
        if (retained_it->second.count(key) != 0 &&
            emitted.insert(key).second) {
          filtered.push_back(match);
        }
      }
    }
    geometry.inlier_matches = std::move(filtered);
  }
  return summary;
}

}  // namespace colite
