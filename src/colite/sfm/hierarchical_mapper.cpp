#include "colite/sfm/hierarchical_mapper.h"

#include <algorithm>
#include <future>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace colite {

namespace {

std::size_t CountSharedImages(const Reconstruction& lhs,
                              const Reconstruction& rhs) {
  std::size_t count = 0;
  for (const ImageId image_id : lhs.RegisteredImageIds()) {
    if (rhs.IsImageRegistered(image_id)) {
      ++count;
    }
  }
  return count;
}

}  // namespace

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
  const Reconstruction schema = *reconstruction;
  const std::size_t hardware_threads =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const std::size_t requested_workers =
      options.max_parallel_local_mappings == 0
          ? std::min<std::size_t>(2, hardware_threads)
          : options.max_parallel_local_mappings;
  const std::size_t worker_count = std::min(
      std::max<std::size_t>(1, requested_workers), summary.clusters.size());
  summary.num_parallel_local_mappings = worker_count;
  summary.local_mappings.resize(summary.clusters.size());
  using LocalResult = std::pair<IncrementalMappingSummary, Reconstruction>;
  for (std::size_t begin = 0; begin < summary.clusters.size();
       begin += worker_count) {
    const std::size_t end =
        std::min(begin + worker_count, summary.clusters.size());
    std::vector<std::future<LocalResult>> futures;
    futures.reserve(end - begin);
    for (std::size_t cluster_idx = begin; cluster_idx < end; ++cluster_idx) {
      const ImageCluster cluster = summary.clusters[cluster_idx];
      futures.push_back(std::async(
          std::launch::async, [&graph, &options, &schema, cluster]() {
            Reconstruction local = schema;
            IncrementalMappingOptions local_options = options.local_mapping;
            local_options.allowed_image_ids = std::set<ImageId>(
                cluster.image_ids.begin(), cluster.image_ids.end());
            IncrementalMappingSummary local_summary =
                RunIncrementalMapping(graph, local_options, &local);
            return LocalResult(std::move(local_summary), std::move(local));
          }));
    }
    for (std::size_t offset = 0; offset < futures.size(); ++offset) {
      LocalResult result = futures[offset].get();
      summary.local_mappings[begin + offset] = std::move(result.first);
      if (summary.local_mappings[begin + offset].success &&
          result.second.RegisteredImageIds().size() >=
              options.min_num_images_per_local_model) {
        local_models.push_back(std::move(result.second));
      }
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
  for (std::size_t lhs = 0; lhs < local_models.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < local_models.size(); ++rhs) {
      if (CountSharedImages(local_models[lhs], local_models[rhs]) > 0) {
        ++summary.num_model_graph_edges;
      }
    }
  }

  Reconstruction merged = std::move(local_models.front());
  std::vector<std::size_t> pending_model_indices;
  std::vector<ModelMergeSummary> last_merge_attempt(local_models.size());
  for (std::size_t model_idx = 1; model_idx < local_models.size(); ++model_idx) {
    pending_model_indices.push_back(model_idx);
  }
  while (!pending_model_indices.empty()) {
    std::sort(pending_model_indices.begin(), pending_model_indices.end(),
              [&merged, &local_models](const std::size_t lhs,
                                       const std::size_t rhs) {
                const std::size_t lhs_overlap =
                    CountSharedImages(merged, local_models[lhs]);
                const std::size_t rhs_overlap =
                    CountSharedImages(merged, local_models[rhs]);
                if (lhs_overlap != rhs_overlap) {
                  return lhs_overlap > rhs_overlap;
                }
                if (local_models[lhs].RegisteredImageIds().size() !=
                    local_models[rhs].RegisteredImageIds().size()) {
                  return local_models[lhs].RegisteredImageIds().size() >
                         local_models[rhs].RegisteredImageIds().size();
                }
                return lhs < rhs;
              });
    bool merged_any_model = false;
    std::size_t merged_model_idx = local_models.size();
    for (const std::size_t model_idx : pending_model_indices) {
      ModelMergeSummary merge = MergeReconstructions(
          graph, options.merging, local_models[model_idx], &merged);
      if (merge.success) {
        ++summary.num_merged_local_models;
        summary.merges.push_back(std::move(merge));
        merged_any_model = true;
        merged_model_idx = model_idx;
        break;
      } else {
        last_merge_attempt[model_idx] = std::move(merge);
      }
    }
    if (!merged_any_model) {
      summary.num_unmerged_local_models += pending_model_indices.size();
      for (const std::size_t model_idx : pending_model_indices) {
        summary.merges.push_back(std::move(last_merge_attempt[model_idx]));
      }
      break;
    }
    pending_model_indices.erase(
        std::find(pending_model_indices.begin(),
                  pending_model_indices.end(), merged_model_idx));
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
