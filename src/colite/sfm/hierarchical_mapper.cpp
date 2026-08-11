#include "colite/sfm/hierarchical_mapper.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace colite {

HierarchicalMappingSummary RunHierarchicalMapping(
    const MatchGraph& graph,
    const HierarchicalMappingOptions& options,
    Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    throw std::invalid_argument("reconstruction must not be null");
  }
  HierarchicalMappingSummary summary;
  if (!reconstruction->RegisteredImageIds().empty() ||
      !reconstruction->Points3D().empty()) {
    summary.message =
        "Hierarchical mapping requires an unregistered reconstruction";
    return summary;
  }
  std::vector<ImageId> image_ids;
  image_ids.reserve(reconstruction->Images().size());
  for (const auto& [image_id, image] : reconstruction->Images()) {
    (void)image;
    image_ids.push_back(image_id);
  }
  std::sort(image_ids.begin(), image_ids.end());
  summary.clusters = ClusterVerifiedViewGraph(
      image_ids, graph, options.clustering, &summary.clustering);
  if (summary.clusters.empty()) {
    summary.message = "View graph clustering produced no local tasks";
    return summary;
  }

  std::vector<Reconstruction> local_models;
  for (const ImageCluster& cluster : summary.clusters) {
    Reconstruction local = *reconstruction;
    IncrementalMappingOptions local_options = options.local_mapping;
    local_options.allowed_image_ids =
        std::set<ImageId>(cluster.image_ids.begin(), cluster.image_ids.end());
    IncrementalMappingSummary local_summary =
        RunIncrementalMapping(graph, local_options, &local);
    summary.local_mappings.push_back(local_summary);
    if (local_summary.success &&
        local.RegisteredImageIds().size() >=
            options.min_num_images_per_local_model) {
      local_models.push_back(std::move(local));
    }
  }
  summary.num_successful_local_models = local_models.size();
  if (local_models.empty()) {
    summary.message = "No local cluster produced a valid reconstruction";
    return summary;
  }
  std::sort(local_models.begin(), local_models.end(),
            [](const Reconstruction& lhs, const Reconstruction& rhs) {
              if (lhs.RegisteredImageIds().size() !=
                  rhs.RegisteredImageIds().size()) {
                return lhs.RegisteredImageIds().size() >
                       rhs.RegisteredImageIds().size();
              }
              return lhs.Points3D().size() > rhs.Points3D().size();
            });

  Reconstruction merged = std::move(local_models.front());
  std::vector<std::size_t> pending_model_indices;
  std::vector<ModelMergeSummary> last_merge_attempt(local_models.size());
  for (std::size_t model_idx = 1; model_idx < local_models.size(); ++model_idx) {
    pending_model_indices.push_back(model_idx);
  }
  while (!pending_model_indices.empty()) {
    bool merged_any_model = false;
    std::vector<std::size_t> next_pending_model_indices;
    for (const std::size_t model_idx : pending_model_indices) {
      ModelMergeSummary merge = MergeReconstructions(
          graph, options.merging, local_models[model_idx], &merged);
      if (merge.success) {
        ++summary.num_merged_local_models;
        summary.merges.push_back(std::move(merge));
        merged_any_model = true;
      } else {
        last_merge_attempt[model_idx] = std::move(merge);
        next_pending_model_indices.push_back(model_idx);
      }
    }
    if (!merged_any_model) {
      summary.num_unmerged_local_models += next_pending_model_indices.size();
      for (const std::size_t model_idx : next_pending_model_indices) {
        summary.merges.push_back(std::move(last_merge_attempt[model_idx]));
      }
      break;
    }
    pending_model_indices = std::move(next_pending_model_indices);
  }

  if (options.run_final_bundle_adjustment) {
    summary.final_bundle_adjustment =
        AdjustGlobalBundle(options.final_bundle_adjustment, &merged);
    if (!summary.final_bundle_adjustment.success) {
      summary.message = "Final hierarchical bundle adjustment failed: " +
                        summary.final_bundle_adjustment.message;
      return summary;
    }
  }
  summary.num_registered_images = merged.RegisteredImageIds().size();
  summary.num_points3d = merged.Points3D().size();
  *reconstruction = std::move(merged);
  summary.success = summary.num_registered_images >=
                    options.min_num_images_per_local_model;
  summary.message = summary.success ? "Hierarchical mapping completed"
                                    : "Merged model is too small";
  return summary;
}

}  // namespace colite
