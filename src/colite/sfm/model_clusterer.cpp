#include "colite/sfm/model_clusterer.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

namespace colite {

std::vector<ImageCluster> ClusterVerifiedViewGraph(
    const std::vector<ImageId>& image_ids,
    const MatchGraph& graph,
    const ModelClusteringOptions& options,
    ModelClusteringSummary* summary) {
  if (options.target_cluster_size == 0 ||
      options.max_cluster_size < options.target_cluster_size) {
    throw std::invalid_argument("Invalid model clustering size limits");
  }
  std::map<ImageId, std::set<ImageId>> adjacency;
  for (const ImageId image_id : image_ids) {
    adjacency[image_id];
  }
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (adjacency.count(pair.image_id1) == 0 ||
        adjacency.count(pair.image_id2) == 0 ||
        geometry.inlier_matches.empty()) {
      continue;
    }
    adjacency[pair.image_id1].insert(pair.image_id2);
    adjacency[pair.image_id2].insert(pair.image_id1);
  }

  std::set<ImageId> unassigned(image_ids.begin(), image_ids.end());
  std::vector<ImageCluster> clusters;
  while (!unassigned.empty()) {
    const ImageId seed = *std::max_element(
        unassigned.begin(), unassigned.end(), [&](ImageId lhs, ImageId rhs) {
          if (adjacency[lhs].size() != adjacency[rhs].size()) {
            return adjacency[lhs].size() < adjacency[rhs].size();
          }
          return lhs > rhs;
        });
    ImageCluster cluster;
    std::queue<ImageId> queue;
    std::set<ImageId> queued{seed};
    queue.push(seed);
    while (!queue.empty() &&
           cluster.core_image_ids.size() < options.target_cluster_size) {
      const ImageId current = queue.front();
      queue.pop();
      if (!unassigned.erase(current)) {
        continue;
      }
      cluster.core_image_ids.push_back(current);
      std::vector<ImageId> neighbors(adjacency[current].begin(),
                                     adjacency[current].end());
      std::sort(neighbors.begin(), neighbors.end(),
                [&](ImageId lhs, ImageId rhs) {
                  if (adjacency[lhs].size() != adjacency[rhs].size()) {
                    return adjacency[lhs].size() > adjacency[rhs].size();
                  }
                  return lhs < rhs;
                });
      for (const ImageId neighbor : neighbors) {
        if (unassigned.count(neighbor) > 0 && queued.insert(neighbor).second) {
          queue.push(neighbor);
        }
      }
    }
    cluster.image_ids = cluster.core_image_ids;
    std::vector<std::pair<std::size_t, ImageId>> boundary;
    const std::set<ImageId> core(cluster.core_image_ids.begin(),
                                 cluster.core_image_ids.end());
    for (const ImageId core_id : cluster.core_image_ids) {
      for (const ImageId neighbor : adjacency[core_id]) {
        if (core.count(neighbor) > 0) {
          continue;
        }
        std::size_t links = 0;
        for (const ImageId candidate_neighbor : adjacency[neighbor]) {
          links += core.count(candidate_neighbor) > 0 ? 1 : 0;
        }
        boundary.emplace_back(links, neighbor);
      }
    }
    std::sort(boundary.begin(), boundary.end(), [](const auto& lhs,
                                                    const auto& rhs) {
      return lhs.first != rhs.first ? lhs.first > rhs.first
                                    : lhs.second < rhs.second;
    });
    std::set<ImageId> added(cluster.image_ids.begin(), cluster.image_ids.end());
    for (const auto& [links, image_id] : boundary) {
      (void)links;
      if (cluster.image_ids.size() >= options.max_cluster_size ||
          cluster.image_ids.size() - cluster.core_image_ids.size() >=
              options.overlap) {
        break;
      }
      if (added.insert(image_id).second) {
        cluster.image_ids.push_back(image_id);
      }
    }
    std::sort(cluster.core_image_ids.begin(), cluster.core_image_ids.end());
    std::sort(cluster.image_ids.begin(), cluster.image_ids.end());
    clusters.push_back(std::move(cluster));
  }

  ModelClusteringSummary local_summary;
  local_summary.success = !clusters.empty() || image_ids.empty();
  local_summary.num_clusters = clusters.size();
  for (const ImageCluster& cluster : clusters) {
    local_summary.num_overlap_assignments +=
        cluster.image_ids.size() - cluster.core_image_ids.size();
    local_summary.max_cluster_size =
        std::max(local_summary.max_cluster_size, cluster.image_ids.size());
  }
  if (summary != nullptr) {
    *summary = local_summary;
  }
  return clusters;
}

}  // namespace colite
