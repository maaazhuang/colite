#pragma once

#include <cstddef>
#include <filesystem>

#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/pair_generator.h"

namespace colite::controllers {

struct SequentialMatcherControllerOptions {
  std::filesystem::path workspace_path;
  MatchingStrategy matching_strategy = MatchingStrategy::kSequential;
  std::size_t overlap = 5;
  std::size_t spatial_max_num_neighbors = 10;
  double spatial_max_distance = 0.0;
  std::size_t vocab_tree_num_images = 50;
  std::size_t vocab_tree_num_visual_words = 1024;
  std::size_t vocab_tree_max_num_features = 5000;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kFlann;
  double max_ratio = 0.8;
  bool mutual_check = true;
};

struct SequentialMatcherControllerSummary {
  bool success = false;
  std::size_t num_images = 0;
  std::size_t num_pairs = 0;
  std::size_t num_verified_pairs = 0;
  std::size_t num_verified_inliers = 0;
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
