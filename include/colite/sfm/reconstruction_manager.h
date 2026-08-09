#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite {

struct ModelMergeOptions {
  std::size_t min_num_correspondences = 6;
  double max_alignment_error_ratio = 0.02;
  std::size_t max_num_hypotheses = 2048;
};

struct ModelMergeSummary {
  bool success = false;
  std::size_t num_correspondences = 0;
  std::size_t num_inliers = 0;
  std::size_t num_added_images = 0;
  std::size_t num_added_points = 0;
  std::size_t num_merged_points = 0;
  double scale = 1.0;
  double max_alignment_error = 0.0;
  std::string message;
};

// Estimate a robust Sim(3) from verified 3D-3D correspondences and merge
// source into target. Failure is non-destructive.
ModelMergeSummary MergeReconstructions(
    const MatchGraph& graph,
    const ModelMergeOptions& options,
    const Reconstruction& source,
    Reconstruction* target);

struct MultiModelMappingOptions {
  IncrementalMappingOptions mapping;
  ModelMergeOptions merging;
  std::size_t max_num_models = 50;
  std::size_t min_num_images_per_model = 2;
  bool merge_models = true;
};

struct MultiModelMappingResult {
  std::vector<Reconstruction> models;
  std::vector<IncrementalMappingSummary> mapping_summaries;
  std::vector<ModelMergeSummary> merge_summaries;
  std::size_t num_unregistered_images = 0;
};

// Continue an optional seed reconstruction, then repeatedly initialize
// remaining verified components. The schema reconstruction must contain every
// workspace camera and image; its registered images/points form the resume
// seed when present.
MultiModelMappingResult RunIncrementalMappingMultiple(
    const MatchGraph& graph,
    const MultiModelMappingOptions& options,
    Reconstruction schema_or_seed);

}  // namespace colite
