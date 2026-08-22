#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

#include "colite/core/feature.h"
#include "colite/features/onnx_options.h"

namespace colite {

struct AlikedExtractionOptions {
  FeatureExtractorType extractor_type = FeatureExtractorType::kAlikedN16Rot;
  std::filesystem::path model_path;
  int max_num_features = 2048;
  float min_score = 0.2F;
  int num_threads = 1;
  OnnxExecutionProvider execution_provider = OnnxExecutionProvider::kAuto;
  int gpu_index = 0;
  // Resize the longest image edge before inference. Zero preserves the input
  // resolution. Detected keypoints are mapped back to source coordinates.
  int max_image_size = 0;
  // Limit the CUDA execution-provider arena. Zero uses the runtime default.
  std::size_t cuda_memory_limit_mb = 0;
  bool use_io_binding = true;
};

struct AlikedImageTensor {
  int original_width = 0;
  int original_height = 0;
  int inference_width = 0;
  int inference_height = 0;
  int padded_width = 0;
  int padded_height = 0;
  // Contiguous RGB tensor in CHW order, normalized to [0, 1].
  std::vector<float> data;
};

bool IsOnnxRuntimeAvailable();
bool IsOnnxCudaExecutionProviderAvailable();
AlikedImageTensor PrepareAlikedImageTensor(const cv::Mat& image,
                                           int max_image_size = 0);
FeatureSet ExtractAlikedFeatures(const cv::Mat& image,
                                 const AlikedExtractionOptions& options);
FeatureSet ExtractAlikedFeaturesFromFile(
    const std::filesystem::path& image_path,
    const AlikedExtractionOptions& options);

}  // namespace colite
