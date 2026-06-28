#include "backends/cuda_sift_extractor.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include "cudasift/thirdparty/cudaImage.h"
#include "cudasift/thirdparty/cudaSift.h"
#include "../sift_descriptor_conversion.h"

namespace colite {
namespace {

std::once_flag cuda_init_once;

cv::Mat ToGrayFloatImage(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }

  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }

  cv::Mat gray_float;
  gray.convertTo(gray_float, CV_32FC1);
  if (!gray_float.isContinuous()) {
    gray_float = gray_float.clone();
  }
  return gray_float;
}

int CudaCandidateCapacity(const SiftExtractionOptions& options) {
  constexpr int kDefaultCapacity = 32768;
  constexpr int kMaxCapacity = 262144;
  if (options.max_num_features <= 0) {
    return kDefaultCapacity;
  }
  const long long expanded =
      std::max<long long>(kDefaultCapacity,
                          static_cast<long long>(options.max_num_features) * 4);
  return static_cast<int>(std::min<long long>(expanded, kMaxCapacity));
}

float CudaResponse(const SiftPoint& point) {
  // CudaSift 的 score 字段用于其 GPU matching，提取阶段不写入。
  // sharpness 是 DoG 极值强度，取绝对值后可作为跨尺度筛选响应。
  return std::abs(point.sharpness);
}

std::vector<int> RankedCudaPointIndices(const SiftPoint* points,
                                        const int num_points,
                                        const int max_num_features) {
  std::vector<int> indices(static_cast<std::size_t>(num_points));
  std::iota(indices.begin(), indices.end(), 0);
  std::stable_sort(indices.begin(), indices.end(), [&](const int lhs,
                                                       const int rhs) {
    const SiftPoint& lhs_point = points[lhs];
    const SiftPoint& rhs_point = points[rhs];
    const float lhs_response = CudaResponse(lhs_point);
    const float rhs_response = CudaResponse(rhs_point);
    if (lhs_response != rhs_response) {
      return lhs_response > rhs_response;
    }
    if (lhs_point.ypos != rhs_point.ypos) {
      return lhs_point.ypos < rhs_point.ypos;
    }
    if (lhs_point.xpos != rhs_point.xpos) {
      return lhs_point.xpos < rhs_point.xpos;
    }
    if (lhs_point.scale != rhs_point.scale) {
      return lhs_point.scale < rhs_point.scale;
    }
    return lhs_point.orientation < rhs_point.orientation;
  });

  if (max_num_features > 0 &&
      indices.size() > static_cast<std::size_t>(max_num_features)) {
    indices.resize(static_cast<std::size_t>(max_num_features));
  }
  return indices;
}

}  // namespace

bool IsCudaSiftAvailable() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

FeatureSet ExtractCudaSiftFeatures(const cv::Mat& image,
                                   const SiftExtractionOptions& options) {
  if (!IsCudaSiftAvailable()) {
    throw std::runtime_error("CudaSift backend was built, but no CUDA device "
                             "is available at runtime.");
  }

  std::call_once(cuda_init_once, [&]() { InitCuda(options.cuda_device); });

  const cv::Mat gray_float = ToGrayFloatImage(image);
  const int width = gray_float.cols;
  const int height = gray_float.rows;
  const int pitch = iAlignUp(width, 128);

  CudaImage cuda_image;
  cuda_image.Allocate(width, height, pitch, false, nullptr,
                      reinterpret_cast<float*>(gray_float.data));
  cuda_image.Download();

  SiftData sift_data;
  const int max_points = CudaCandidateCapacity(options);
  InitSiftData(sift_data, max_points, true, true);

  float* temp_memory = AllocSiftTempMemory(width, height,
                                           options.cuda_num_octaves,
                                           options.cuda_scale_up);
  ExtractSift(sift_data, cuda_image, options.cuda_num_octaves,
              options.cuda_init_blur, options.cuda_threshold,
              options.cuda_lowest_scale, options.cuda_scale_up, temp_memory);
  FreeSiftTempMemory(temp_memory);

  FeatureSet result;
  const std::vector<int> ranked_indices = RankedCudaPointIndices(
      sift_data.h_data, sift_data.numPts, options.max_num_features);
  result.keypoints.reserve(ranked_indices.size());

  std::vector<float> descriptors;
  descriptors.reserve(ranked_indices.size() * 128);
  for (const int point_idx : ranked_indices) {
    const SiftPoint& point = sift_data.h_data[point_idx];
    FeatureKeypoint keypoint;
    keypoint.x = point.xpos;
    keypoint.y = point.ypos;
    keypoint.scale = point.scale;
    keypoint.orientation = point.orientation;
    keypoint.response = CudaResponse(point);
    result.keypoints.push_back(keypoint);

    descriptors.insert(descriptors.end(), point.data, point.data + 128);
  }
  result.descriptors = features::ConvertSiftDescriptors(
      descriptors.data(), result.keypoints.size(), options.use_root_sift);

  FreeSiftData(sift_data);
  return result;
}

}  // namespace colite
