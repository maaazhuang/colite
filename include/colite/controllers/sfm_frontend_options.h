#pragma once

#include <optional>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "colite/features/sift_extractor.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite::controllers {

struct SfmFeatureExecutionOptions {
  std::optional<bool> use_gpu;
  int gpu_index = -1;
  int max_image_size = -1;
  bool custom_sift_geometry = false;
  SiftExtractionOptions sift = [] {
    SiftExtractionOptions options;
    options.peak_threshold = 0.02 / 3.0;
    return options;
  }();
};

struct SfmMatchingExecutionOptions {
  std::optional<bool> use_gpu;
  int gpu_index = -1;
  int num_threads = -1;
};

inline int EffectiveSfmImageSize(const SfmFeatureExecutionOptions& options,
                                 FeatureExtractorType type) {
  return options.max_image_size > 0 ? options.max_image_size
      : (type == FeatureExtractorType::kSift ? 3200 : 1600);
}

inline SiftExtractionOptions ResolveSfmSiftOptions(
    const SfmFeatureExecutionOptions& execution, SiftBackend backend, int maximum) {
  SiftExtractionOptions sift = execution.sift;
  const bool gpu_backend = backend == SiftBackend::kCudaSift || backend == SiftBackend::kCudaVlSift;
  if (execution.use_gpu.has_value() && backend != SiftBackend::kAuto &&
      *execution.use_gpu != gpu_backend) {
    throw std::invalid_argument("FeatureExtraction.use_gpu conflicts with Colite.feature_extractor");
  }
  if (backend == SiftBackend::kAuto) {
    const bool gpu = execution.use_gpu.value_or(IsCudaSiftAvailable());
    backend = gpu ? SiftBackend::kCudaVlSift : SiftBackend::kVlSift;
  }
  if ((backend == SiftBackend::kCvSift || backend == SiftBackend::kCudaSift) && execution.custom_sift_geometry)
    throw std::invalid_argument("COLMAP SIFT geometry controls require vlsift or cuda_vlfeat");
  if ((backend == SiftBackend::kCudaSift || backend == SiftBackend::kCudaVlSift) &&
      !IsCudaSiftAvailable()) throw std::invalid_argument("GPU SIFT is unavailable; set FeatureExtraction.use_gpu=0");
  if (execution.gpu_index < -1) throw std::invalid_argument("FeatureExtraction.gpu_index must be -1 or a single device index");
  sift.backend = backend;
  sift.cuda_device = std::max(0, execution.gpu_index);
  sift.max_num_features = maximum;
  sift.max_image_size = EffectiveSfmImageSize(execution, FeatureExtractorType::kSift);
  return sift;
}

inline DescriptorMatcherBackend ResolveSfmMatchingBackend(
    const SfmMatchingExecutionOptions& execution) {
  if (execution.gpu_index < -1) throw std::invalid_argument("FeatureMatching.gpu_index must be -1 or a single device index");
  const bool gpu = execution.use_gpu.value_or(IsCudaDescriptorMatcherAvailable());
  if (gpu && !IsCudaDescriptorMatcherAvailable()) {
    throw std::invalid_argument("GPU matching is unavailable; set FeatureMatching.use_gpu=0");
  }
  if (gpu) SetCudaDescriptorMatcherDevice(std::max(0, execution.gpu_index));
  return gpu ? DescriptorMatcherBackend::kCudaBruteForce : DescriptorMatcherBackend::kBruteForce;
}

inline std::string SfmFeatureExecutionKey(const SfmFeatureExecutionOptions& execution) {
  const auto& sift = execution.sift;
  std::ostringstream key;
  key << std::setprecision(17) << execution.use_gpu.value_or(IsCudaSiftAvailable()) << ','
      << execution.gpu_index << ',' << execution.max_image_size << ','
      << sift.vl_num_octaves << ',' << sift.num_octave_layers << ','
      << sift.upsample_first_octave << ',' << sift.peak_threshold << ','
      << sift.edge_threshold << ',' << sift.max_num_orientations << ',' << sift.use_root_sift;
  return key.str();
}

}  // namespace colite::controllers
