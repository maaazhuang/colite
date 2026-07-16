#pragma once

#include <cstddef>
#include <filesystem>

#include "colite/features/sift_extractor.h"

namespace colite::controllers {

struct FeatureExtractorControllerOptions {
  std::filesystem::path image_path;
  std::filesystem::path workspace_path;
  double focal_length = 0.0;
  int max_num_features = 4096;
  int num_threads = -1;
  SiftBackend backend = SiftBackend::kAuto;
};

struct FeatureExtractorControllerSummary {
  bool success = false;
  bool cache_hit = false;
  std::size_t num_images = 0;
  double elapsed_seconds = 0.0;
};

class FeatureExtractorController {
 public:
  explicit FeatureExtractorController(FeatureExtractorControllerOptions options);

  FeatureExtractorControllerSummary Run() const;

 private:
  FeatureExtractorControllerOptions options_;
};

}  // namespace colite::controllers
