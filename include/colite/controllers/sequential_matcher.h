#pragma once

#include <cstddef>
#include <filesystem>

#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/learned_matcher.h"
#include "colite/matching/pair_generator.h"
#include "colite/matching/weak_pair_rematcher.h"
#include "colite/controllers/sfm_frontend_options.h"

namespace colite::controllers {

struct SequentialMatcherControllerOptions {
  SfmMatchingExecutionOptions execution;
  std::filesystem::path workspace_path;
  MatchingStrategy matching_strategy = MatchingStrategy::kSequential;
  std::size_t overlap = 10;
  bool quadratic_overlap = true;
  std::size_t spatial_max_num_neighbors = 50;
  double spatial_max_distance = 100.0;
  std::size_t vocab_tree_num_images = 100;
  std::size_t vocab_tree_num_visual_words = 1024;
  std::size_t vocab_tree_branching = 16;
  int vocab_tree_num_iterations = 6;
  std::size_t vocab_tree_max_num_features = 5000;
  std::size_t vocab_tree_sequential_overlap = 2;
  std::filesystem::path vocab_tree_path;
  std::size_t transitive_max_path_length = 2;
  std::size_t transitive_max_num_neighbors = 20;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kBruteForce;
  FeatureMatcherType matcher_type = FeatureMatcherType::kSiftBruteForce;
  std::filesystem::path learned_matcher_model_path;
  double min_cosine_similarity = 0.85;
  double lightglue_min_score = 0.1;
  int learned_matcher_num_threads = 1;
  OnnxExecutionProvider learned_execution_provider =
      OnnxExecutionProvider::kAuto;
  int learned_gpu_index = 0;
  std::size_t learned_cuda_memory_limit_mb = 8192;
  bool learned_use_io_binding = true;
  double max_ratio = 0.8;
  double max_distance = 0.7;
  TwoViewGeometryOptions two_view_geometry;
  bool mutual_check = true;
  bool guided_matching = false;
  // Candidate ratio under verified geometry; geometric filtering keeps the
  // same pixel gate. This extension is distinct from SiftMatching.max_ratio.
  double guided_match_ratio = 0.8;
  std::size_t guided_min_inlier_gain = 10;
};

struct SequentialMatcherControllerSummary {
  double feature_loading_seconds = 0;
  double descriptor_preparation_seconds = 0;
  double descriptor_matching_seconds = 0;
  double geometry_verification_seconds = 0;
  double guided_matching_seconds = 0;
  double graph_consistency_seconds = 0;
  double workspace_write_seconds = 0;
  bool success = false;
  bool raw_match_cache_hit = false;
  bool geometry_cache_hit = false;
  std::size_t num_images = 0;
  std::size_t num_pairs = 0;
  std::size_t num_verified_pairs = 0;
  std::size_t num_verified_inliers = 0;
  WeakPairRematchingSummary guided_matching;
  MatchingPairGeneratorSummary pair_generation;
};

class SequentialMatcherController {
 public:
  explicit SequentialMatcherController(
      SequentialMatcherControllerOptions options);

  SequentialMatcherControllerSummary Run() const;

 private:
  SequentialMatcherControllerOptions options_;
};

}  // namespace colite::controllers
