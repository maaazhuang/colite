#pragma once

#include <cstddef>
#include <filesystem>

#include "colite/matching/descriptor_matcher.h"

namespace colite::controllers {

struct SequentialMatcherControllerOptions {
  std::filesystem::path workspace_path;
  std::size_t overlap = 5;
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
