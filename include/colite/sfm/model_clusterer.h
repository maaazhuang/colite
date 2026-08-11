#pragma once

#include <cstddef>
#include <vector>

#include "colite/core/match_graph.h"

namespace colite {

struct ModelClusteringOptions {
  std::size_t target_cluster_size = 100;
  std::size_t max_cluster_size = 150;
  std::size_t overlap = 10;
};

struct ImageCluster {
  std::vector<ImageId> core_image_ids;
  std::vector<ImageId> image_ids;
};

struct ModelClusteringSummary {
  bool success = false;
  std::size_t num_clusters = 0;
  std::size_t num_overlap_assignments = 0;
  std::size_t max_cluster_size = 0;
};

std::vector<ImageCluster> ClusterVerifiedViewGraph(
    const std::vector<ImageId>& image_ids,
    const MatchGraph& graph,
    const ModelClusteringOptions& options,
    ModelClusteringSummary* summary = nullptr);

}  // namespace colite
