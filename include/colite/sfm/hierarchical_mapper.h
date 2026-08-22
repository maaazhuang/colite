#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/sfm/incremental_mapper.h"
#include "colite/sfm/model_clusterer.h"
#include "colite/sfm/reconstruction_manager.h"

namespace colite {

struct HierarchicalMappingOptions {
  ModelClusteringOptions clustering;
  IncrementalMappingOptions local_mapping;
  ModelMergeOptions merging;
  GlobalBundleAdjustmentOptions final_bundle_adjustment;
  std::size_t min_num_images_per_local_model = 2;
  // Zero automatically uses at most two workers to bound the number of full
  // reconstruction copies held in memory.
  std::size_t max_parallel_local_mappings = 0;
  bool run_final_bundle_adjustment = true;
};

struct HierarchicalMappingSummary {
  bool success = false;
  ModelClusteringSummary clustering;
  std::vector<ImageCluster> clusters;
  std::vector<IncrementalMappingSummary> local_mappings;
  std::vector<ModelMergeSummary> merges;
  GlobalBundleAdjustmentSummary final_bundle_adjustment;
  std::size_t num_successful_local_models = 0;
  std::size_t num_merged_local_models = 0;
  std::size_t num_parallel_local_mappings = 0;
  std::size_t num_model_graph_edges = 0;
  std::size_t num_unmerged_local_models = 0;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
  std::string message;
};

// Runs overlapping local incremental reconstructions and merges them into the
// input schema reconstruction. The input must not contain registered images.
HierarchicalMappingSummary RunHierarchicalMapping(
    const MatchGraph& graph,
    const HierarchicalMappingOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
