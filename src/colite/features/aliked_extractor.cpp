#include "colite/features/aliked_extractor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "onnx_runtime.h"
#include "colite/io/image_reader.h"
#include "colite/io/model_resource.h"

namespace colite {
namespace {

constexpr int kAlikedInputDivisor = 32;

void ValidateAlikedOptions(const AlikedExtractionOptions& options) {
  if (!IsAlikedFeature(options.extractor_type)) {
    throw std::invalid_argument("ALIKED extractor type must be N16Rot or N32");
  }
  if (options.max_num_features <= 0 || options.min_score < 0.0F ||
      options.min_score > 1.0F || options.num_threads == 0) {
    throw std::invalid_argument("Invalid ALIKED extraction options");
  }
}

#if defined(COLITE_ENABLE_ONNX)
features::OnnxModel& CachedAlikedModel(const AlikedExtractionOptions& options) {
  struct Cache {
    std::filesystem::path path;
    int num_threads = 0;
    std::unique_ptr<features::OnnxModel> model;
  };
  thread_local Cache cache;
  if (cache.model == nullptr || cache.path != options.model_path ||
      cache.num_threads != options.num_threads) {
    ValidateLearnedModelResource(options.model_path);
    cache.path = options.model_path;
    cache.num_threads = options.num_threads;
    cache.model = std::make_unique<features::OnnxModel>(
        options.model_path, options.num_threads);
  }
  return *cache.model;
}
#endif

}  // namespace

bool IsOnnxRuntimeAvailable() {
#if defined(COLITE_ENABLE_ONNX)
  return true;
#else
  return false;
#endif
}

AlikedImageTensor PrepareAlikedImageTensor(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("ALIKED input image is empty");
  }
  cv::Mat bgr;
  if (image.channels() == 1) {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 3) {
    bgr = image;
  } else if (image.channels() == 4) {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  } else {
    throw std::invalid_argument("ALIKED input must have 1, 3, or 4 channels");
  }
  if (bgr.depth() != CV_8U) {
    throw std::invalid_argument("ALIKED input must use 8-bit channels");
  }

  AlikedImageTensor tensor;
  tensor.original_width = bgr.cols;
  tensor.original_height = bgr.rows;
  tensor.padded_width =
      ((bgr.cols + kAlikedInputDivisor - 1) / kAlikedInputDivisor) *
      kAlikedInputDivisor;
  tensor.padded_height =
      ((bgr.rows + kAlikedInputDivisor - 1) / kAlikedInputDivisor) *
      kAlikedInputDivisor;
  const std::size_t plane = static_cast<std::size_t>(tensor.padded_width) *
                            static_cast<std::size_t>(tensor.padded_height);
  tensor.data.resize(3 * plane);
  constexpr float kInv255 = 1.0F / 255.0F;
  for (int y = 0; y < tensor.padded_height; ++y) {
    const int source_y = std::min(y, bgr.rows - 1);
    const cv::Vec3b* row = bgr.ptr<cv::Vec3b>(source_y);
    for (int x = 0; x < tensor.padded_width; ++x) {
      const int source_x = std::min(x, bgr.cols - 1);
      const cv::Vec3b pixel = row[source_x];
      const std::size_t offset = static_cast<std::size_t>(y) *
                                     static_cast<std::size_t>(tensor.padded_width) +
                                 static_cast<std::size_t>(x);
      tensor.data[offset] = pixel[2] * kInv255;
      tensor.data[plane + offset] = pixel[1] * kInv255;
      tensor.data[2 * plane + offset] = pixel[0] * kInv255;
    }
  }
  return tensor;
}

FeatureSet ExtractAlikedFeatures(const cv::Mat& image,
                                 const AlikedExtractionOptions& options) {
  ValidateAlikedOptions(options);
#if defined(COLITE_ENABLE_ONNX)
  features::OnnxModel& model = CachedAlikedModel(options);
  if (model.InputNames().size() != 3 || model.OutputNames().size() != 3) {
    throw std::runtime_error("ALIKED ONNX model must have 3 inputs and outputs");
  }
  features::ValidateOnnxNode(model.InputNames()[0], "image",
                             model.InputShapes()[0], {-1, 3, -1, -1});
  features::ValidateOnnxNode(model.InputNames()[1], "max_keypoints",
                             model.InputShapes()[1], {});
  features::ValidateOnnxNode(model.InputNames()[2], "min_score",
                             model.InputShapes()[2], {});
  features::ValidateOnnxNode(model.OutputNames()[0], "keypoints",
                             model.OutputShapes()[0], {-1, -1, 2});
  features::ValidateOnnxNode(model.OutputNames()[1], "descriptors",
                             model.OutputShapes()[1], {-1, -1, -1});
  features::ValidateOnnxNode(model.OutputNames()[2], "scores",
                             model.OutputShapes()[2], {-1, -1});

  AlikedImageTensor image_tensor = PrepareAlikedImageTensor(image);
  std::vector<std::int64_t> image_shape = {
      1, 3, image_tensor.padded_height, image_tensor.padded_width};
  std::int64_t max_keypoints = options.max_num_features;
  float min_score = options.min_score;
  const std::vector<std::int64_t> scalar_shape;
  const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory, image_tensor.data.data(), image_tensor.data.size(),
      image_shape.data(), image_shape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(
      memory, &max_keypoints, 1, scalar_shape.data(), scalar_shape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory, &min_score, 1, scalar_shape.data(), scalar_shape.size()));
  const std::vector<Ort::Value> outputs = model.Run(inputs);

  const std::vector<std::int64_t> keypoint_shape =
      outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<std::int64_t> descriptor_shape =
      outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<std::int64_t> score_shape =
      outputs[2].GetTensorTypeAndShapeInfo().GetShape();
  if (keypoint_shape.size() != 3 || keypoint_shape[0] != 1 ||
      keypoint_shape[2] != 2 || descriptor_shape.size() != 3 ||
      descriptor_shape[0] != 1 ||
      descriptor_shape[1] != keypoint_shape[1] || descriptor_shape[2] <= 0 ||
      score_shape != std::vector<std::int64_t>({1, keypoint_shape[1]})) {
    throw std::runtime_error("Invalid ALIKED output tensor shapes");
  }
  const auto num_keypoints = static_cast<int>(keypoint_shape[1]);
  const auto descriptor_dim = static_cast<int>(descriptor_shape[2]);
  const float* keypoint_data = outputs[0].GetTensorData<float>();
  const float* descriptor_data = outputs[1].GetTensorData<float>();
  const float* score_data = outputs[2].GetTensorData<float>();
  const float scale_x = 0.5F * (image_tensor.padded_width - 1);
  const float scale_y = 0.5F * (image_tensor.padded_height - 1);

  std::vector<int> valid_indices;
  valid_indices.reserve(num_keypoints);
  FeatureSet result;
  result.extractor_type = options.extractor_type;
  result.descriptor_metric = DescriptorMetric::kCosine;
  result.descriptor_normalization = DescriptorNormalization::kL2;
  for (int idx = 0; idx < num_keypoints; ++idx) {
    const float x = (keypoint_data[2 * idx] + 1.0F) * scale_x + 0.5F;
    const float y = (keypoint_data[2 * idx + 1] + 1.0F) * scale_y + 0.5F;
    if (x >= 0.0F && x < image_tensor.original_width && y >= 0.0F &&
        y < image_tensor.original_height) {
      result.keypoints.push_back({x, y, 1.0F, 0.0F, score_data[idx]});
      valid_indices.push_back(idx);
    }
  }
  result.descriptors.resize(static_cast<Eigen::Index>(valid_indices.size()),
                            descriptor_dim);
  for (std::size_t row = 0; row < valid_indices.size(); ++row) {
    std::memcpy(result.descriptors.row(static_cast<Eigen::Index>(row)).data(),
                descriptor_data + valid_indices[row] * descriptor_dim,
                static_cast<std::size_t>(descriptor_dim) * sizeof(float));
  }
  return result;
#else
  (void)image;
  throw std::runtime_error(
      "ALIKED extraction requires a build with COLITE_ENABLE_ONNX=ON");
#endif
}

FeatureSet ExtractAlikedFeaturesFromFile(
    const std::filesystem::path& image_path,
    const AlikedExtractionOptions& options) {
  const cv::Mat image = ReadImageWithExifOrientation(image_path,
                                                      cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    throw std::runtime_error("Failed to read ALIKED image: " +
                             image_path.string());
  }
  return ExtractAlikedFeatures(image, options);
}

}  // namespace colite
