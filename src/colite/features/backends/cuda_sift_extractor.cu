#include "backends/cuda_sift_extractor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include "cuda_sift/cuda_sift.hpp"
#include "../sift_descriptor_conversion.h"

namespace colite {
namespace {

constexpr float kRadiansToDegrees = 57.295779513082320876F;
constexpr double kDefaultOpenCvContrastThreshold = 0.04;

float EffectiveCudaVlFeatPeakThreshold(
    const SiftExtractionOptions& options) {
  const double normalized =
      std::abs(options.contrast_threshold -
               kDefaultOpenCvContrastThreshold) < 1e-12
          ? 0.02 / static_cast<double>(options.num_octave_layers)
          : options.contrast_threshold;
  // The CUDA pyramid preserves the input's [0, 255] intensity range while
  // CPU VLFeat receives the same range as floats.
  return static_cast<float>(normalized * 255.0);
}

cv::Mat ToContinuousGray(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  if (gray.type() != CV_8UC1) {
    gray.convertTo(gray, CV_8UC1);
  }
  return gray.isContinuous() ? gray : gray.clone();
}

void TransformVlFeatDescriptorToUbc(float* descriptor) {
  std::array<float, 128> original{};
  std::copy_n(descriptor, original.size(), original.data());
  constexpr std::array<int, 8> permutation{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 8; ++k) {
        descriptor[8 * (j + 4 * i) + permutation[k]] =
            original[8 * (j + 4 * i) + k];
      }
    }
  }
}

features::cuda::CudaSiftMode CudaMode(const SiftBackend backend) {
  return backend == SiftBackend::kCudaVlSift
             ? features::cuda::CudaSiftMode::VLFeat
             : features::cuda::CudaSiftMode::OpenCV;
}

void RetainBestCudaFeatures(const std::size_t maximum,
                            features::cuda::CudaSiftResult* result) {
  if (maximum == 0 || result->keypoints.size() <= maximum) return;
  std::vector<std::size_t> order(result->keypoints.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](const std::size_t lhs,
                                                    const std::size_t rhs) {
    return std::abs(result->keypoints[lhs].response) >
           std::abs(result->keypoints[rhs].response);
  });

  std::vector<features::cuda::CudaSiftKeypoint> keypoints;
  std::vector<float> descriptors;
  keypoints.reserve(maximum);
  descriptors.reserve(maximum * 128);
  for (std::size_t rank = 0; rank < maximum; ++rank) {
    const std::size_t source = order[rank];
    keypoints.push_back(result->keypoints[source]);
    descriptors.insert(descriptors.end(),
                       result->descriptors.begin() + source * 128,
                       result->descriptors.begin() + (source + 1) * 128);
  }
  result->keypoints = std::move(keypoints);
  result->descriptors = std::move(descriptors);
}

}  // namespace

bool IsCudaSiftAvailable() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

FeatureSet ExtractCudaSiftFeatures(const cv::Mat& image,
                                   const SiftExtractionOptions& options) {
  if (!IsCudaSiftAvailable()) {
    throw std::runtime_error("CUDA SIFT was built, but no CUDA device is available.");
  }

  const SiftBackend backend = ResolveSiftBackend(options.backend);
  const bool vlfeat_mode = backend == SiftBackend::kCudaVlSift;
  features::cuda::CudaSiftConfig config;
  config.mode = CudaMode(backend);
  config.firstOctave = options.upsample_first_octave ? -1 : 0;
  config.octaveLayers = options.num_octave_layers;
  config.maxOctaves = vlfeat_mode ? options.vl_num_octaves
                                  : options.cuda_num_octaves;
  config.sigma = static_cast<float>(options.sigma);
  config.contrastThreshold = static_cast<float>(options.contrast_threshold);
  config.peakThreshold = vlfeat_mode
                             ? EffectiveCudaVlFeatPeakThreshold(options)
                             : options.cuda_threshold;
  config.edgeThreshold = static_cast<float>(options.edge_threshold);
  config.maxFeatures = options.max_num_features;

  const cv::Mat gray = ToContinuousGray(image);
  features::cuda::CudaSiftResult cuda_result =
      features::cuda::CudaSift(config, options.cuda_device)
          .detectAndCompute(gray.ptr<std::uint8_t>(), gray.cols, gray.rows);
  RetainBestCudaFeatures(static_cast<std::size_t>(options.max_num_features),
                         &cuda_result);

  FeatureSet result;
  result.keypoints.reserve(cuda_result.keypoints.size());
  for (const features::cuda::CudaSiftKeypoint& keypoint :
       cuda_result.keypoints) {
    FeatureKeypoint converted;
    // COLMAP's VLFeat wrapper stores pixel-center coordinates, whereas the
    // low-level VLFeat/CUDA kernels both use integer pixel coordinates.
    converted.x = keypoint.x + (vlfeat_mode ? 0.5F : 0.0F);
    converted.y = keypoint.y + (vlfeat_mode ? 0.5F : 0.0F);
    converted.scale = keypoint.size;
    converted.orientation =
        vlfeat_mode ? keypoint.angle * kRadiansToDegrees : keypoint.angle;
    if (converted.orientation < 0.0F) converted.orientation += 360.0F;
    converted.response = std::abs(keypoint.response);
    result.keypoints.push_back(converted);
  }

  std::vector<float> descriptors = cuda_result.descriptors;
  if (descriptors.size() != result.keypoints.size() * 128) {
    throw std::runtime_error("CUDA SIFT descriptor buffer has unexpected size.");
  }
  if (vlfeat_mode) {
    for (std::size_t i = 0; i < result.keypoints.size(); ++i) {
      TransformVlFeatDescriptorToUbc(descriptors.data() + i * 128);
    }
  }
  result.descriptors = features::ConvertSiftDescriptors(
      descriptors.data(), result.keypoints.size(), options.use_root_sift);
  return result;
}

}  // namespace colite
