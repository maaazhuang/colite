#pragma once

#include <cstddef>
#include <filesystem>

#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/pair_generator.h"
#include "colite/matching/weak_pair_rematcher.h"

namespace colite::controllers {

struct SequentialMatcherControllerOptions {
  std::filesystem::path workspace_path;
  MatchingStrategy matching_strategy = MatchingStrategy::kSequential;
  std::size_t overlap = 5;
  std::size_t spatial_max_num_neighbors = 10;
  double spatial_max_distance = 0.0;
  std::size_t vocab_tree_num_images = 50;
  std::size_t vocab_tree_num_visual_words = 1024;
  std::size_t vocab_tree_branching = 16;
  int vocab_tree_num_iterations = 6;
  std::size_t vocab_tree_max_num_features = 5000;
  std::size_t vocab_tree_sequential_overlap = 2;
  std::filesystem::path vocab_tree_path;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kFlann;
  double max_ratio = 0.8;
  bool mutual_check = true;
  bool guided_matching = false;
  double guided_match_ratio = 0.8;
  std::size_t guided_min_inlier_gain = 10;
};

struct SequentialMatcherControllerSummary {
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
