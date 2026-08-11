#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

#include "colite/core/feature.h"

namespace colite {

struct AlikedExtractionOptions {
  FeatureExtractorType extractor_type = FeatureExtractorType::kAlikedN16Rot;
  std::filesystem::path model_path;
  int max_num_features = 2048;
  float min_score = 0.2F;
  int num_threads = 1;
};

struct AlikedImageTensor {
  int original_width = 0;
  int original_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  // Contiguous RGB tensor in CHW order, normalized to [0, 1].
  std::vector<float> data;
};

bool IsOnnxRuntimeAvailable();
AlikedImageTensor PrepareAlikedImageTensor(const cv::Mat& image);
FeatureSet ExtractAlikedFeatures(const cv::Mat& image,
                                 const AlikedExtractionOptions& options);
FeatureSet ExtractAlikedFeaturesFromFile(
    const std::filesystem::path& image_path,
    const AlikedExtractionOptions& options);

}  // namespace colite
