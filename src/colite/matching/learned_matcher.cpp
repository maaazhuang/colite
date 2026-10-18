#include "colite/matching/learned_matcher.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../features/onnx_runtime.h"
#include "colite/io/model_resource.h"

namespace colite {
namespace {

void ValidateAlikedFeatures(const FeatureSet& features) {
  if (!IsAlikedFeature(features.extractor_type) ||
      features.descriptor_scalar_type != DescriptorScalarType::kFloat32 ||
      features.descriptor_metric != DescriptorMetric::kCosine) {
    throw std::invalid_argument("ALIKED matcher requires float cosine descriptors");
  }
  if (!features.keypoints.empty() &&
      features.descriptors.rows() !=
          static_cast<Eigen::Index>(features.keypoints.size())) {
    throw std::invalid_argument("Feature keypoint and descriptor rows do not match");
  }
}

struct BestCosineMatch {
  int index = -1;
  float cosine = -1.0F;
};

std::vector<BestCosineMatch> FindCosineMatchesOneWay(
    const DescriptorMatrix& query,
    const DescriptorMatrix& train,
    const CosineMatchingOptions& options) {
  std::vector<BestCosineMatch> result(
      static_cast<std::size_t>(query.rows()));
  if (query.size() == 0 || train.size() == 0) return result;
  std::vector<float> train_inv_norm(static_cast<std::size_t>(train.rows()));
  for (Eigen::Index row = 0; row < train.rows(); ++row) {
    const float norm = train.row(row).norm();
    train_inv_norm[static_cast<std::size_t>(row)] =
        norm > 0.0F ? 1.0F / norm : 0.0F;
  }
  for (Eigen::Index query_idx = 0; query_idx < query.rows(); ++query_idx) {
    const float query_norm = query.row(query_idx).norm();
    if (query_norm <= 0.0F) continue;
    const float query_inv_norm = 1.0F / query_norm;
    float best = -1.0F;
    float second = -1.0F;
    int best_idx = -1;
    for (Eigen::Index train_idx = 0; train_idx < train.rows(); ++train_idx) {
      const float cosine = std::clamp(
          query.row(query_idx).dot(train.row(train_idx)) * query_inv_norm *
              train_inv_norm[static_cast<std::size_t>(train_idx)],
          -1.0F, 1.0F);
      if (cosine > best) {
        second = best;
        best = cosine;
        best_idx = static_cast<int>(train_idx);
      } else if (cosine > second) {
        second = cosine;
      }
    }
    if (best_idx < 0 || best < options.min_cosine_similarity) continue;
    const float best_distance = std::sqrt(std::max(0.0F, 2.0F - 2.0F * best));
    const float second_distance =
        second < -0.5F
            ? std::numeric_limits<float>::infinity()
            : std::sqrt(std::max(0.0F, 2.0F - 2.0F * second));
    if (best_distance >= options.max_ratio * second_distance) continue;
    result[static_cast<std::size_t>(query_idx)] = {best_idx, best};
  }
  return result;
}

#if defined(COLITE_ENABLE_ONNX)
features::OnnxModel& CachedLightGlueModel(
    const LightGlueMatchingOptions& options) {
  struct Cache {
    std::filesystem::path path;
    int num_threads = 0;
    OnnxExecutionProvider execution_provider = OnnxExecutionProvider::kAuto;
    int gpu_index = -1;
    std::size_t cuda_memory_limit_mb = 0;
    bool use_io_binding = true;
    std::unique_ptr<features::OnnxModel> model;
  };
  thread_local Cache cache;
  if (cache.model == nullptr || cache.path != options.model_path ||
      cache.num_threads != options.num_threads ||
      cache.execution_provider != options.execution_provider ||
      cache.gpu_index != options.gpu_index ||
      cache.cuda_memory_limit_mb != options.cuda_memory_limit_mb ||
      cache.use_io_binding != options.use_io_binding) {
    ValidateLearnedModelResource(options.model_path);
    cache.path = options.model_path;
    cache.num_threads = options.num_threads;
    cache.execution_provider = options.execution_provider;
    cache.gpu_index = options.gpu_index;
    cache.cuda_memory_limit_mb = options.cuda_memory_limit_mb;
    cache.use_io_binding = options.use_io_binding;
    cache.model = std::make_unique<features::OnnxModel>(
        options.model_path, options.num_threads, options.execution_provider,
        options.gpu_index, options.cuda_memory_limit_mb * 1024U * 1024U,
        options.use_io_binding);
  }
  return *cache.model;
}

struct LightGlueInput {
  std::vector<float> keypoints;
  std::vector<float> descriptors;
  std::vector<float> scales;
  std::vector<float> orientations;
  float image_size[2] = {0.0F, 0.0F};
  int num_features = 0;
  int descriptor_dim = 0;
};

LightGlueInput PrepareLightGlueInput(const FeatureSet& features,
                                     const int width,
                                     const int height) {
  if (width <= 0 || height <= 0 || features.descriptors.rows() !=
                                     static_cast<Eigen::Index>(
                                         features.keypoints.size())) {
    throw std::invalid_argument("Invalid LightGlue feature/image dimensions");
  }
  LightGlueInput input;
  input.num_features = static_cast<int>(features.keypoints.size());
  input.descriptor_dim = static_cast<int>(features.descriptors.cols());
  input.image_size[0] = static_cast<float>(width);
  input.image_size[1] = static_cast<float>(height);
  input.keypoints.resize(static_cast<std::size_t>(input.num_features) * 2);
  input.scales.resize(input.num_features);
  input.orientations.resize(input.num_features);
  for (int idx = 0; idx < input.num_features; ++idx) {
    const FeatureKeypoint& keypoint = features.keypoints[idx];
    input.keypoints[2 * idx] = keypoint.x - 0.5F;
    input.keypoints[2 * idx + 1] = keypoint.y - 0.5F;
    input.scales[idx] = keypoint.scale;
    input.orientations[idx] =
        keypoint.orientation * static_cast<float>(3.14159265358979323846 / 180.0);
  }
  input.descriptors.resize(static_cast<std::size_t>(features.descriptors.size()));
  if (features.extractor_type == FeatureExtractorType::kSift &&
      features.descriptor_normalization == DescriptorNormalization::kNone) {
    for (Eigen::Index row = 0; row < features.descriptors.rows(); ++row) {
      const float l1 = features.descriptors.row(row).cwiseAbs().sum();
      for (Eigen::Index col = 0; col < features.descriptors.cols(); ++col) {
        input.descriptors[static_cast<std::size_t>(row) *
                              features.descriptors.cols() +
                          col] =
            l1 > 0.0F
                ? std::sqrt(std::max(0.0F, features.descriptors(row, col)) / l1)
                : 0.0F;
      }
    }
  } else {
    std::memcpy(input.descriptors.data(), features.descriptors.data(),
                input.descriptors.size() * sizeof(float));
  }
  return input;
}
#endif

}  // namespace

std::vector<FeatureMatch> MatchAlikedDescriptors(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const CosineMatchingOptions& options) {
  ValidateAlikedFeatures(features1);
  ValidateAlikedFeatures(features2);
  if (features1.extractor_type != features2.extractor_type ||
      (features1.descriptors.size() != 0 && features2.descriptors.size() != 0 &&
       features1.descriptors.cols() != features2.descriptors.cols())) {
    throw std::invalid_argument("ALIKED descriptor types or dimensions differ");
  }
  if (options.min_cosine_similarity < -1.0F ||
      options.min_cosine_similarity > 1.0F || options.max_ratio <= 0.0F ||
      options.max_ratio > 1.0F) {
    throw std::invalid_argument("Invalid cosine matching options");
  }
  const std::vector<BestCosineMatch> forward = FindCosineMatchesOneWay(
      features1.descriptors, features2.descriptors, options);
  std::vector<BestCosineMatch> backward;
  if (options.mutual_check) {
    backward = FindCosineMatchesOneWay(features2.descriptors,
                                       features1.descriptors, options);
  }
  std::vector<FeatureMatch> matches;
  for (std::size_t idx1 = 0; idx1 < forward.size(); ++idx1) {
    const int idx2 = forward[idx1].index;
    if (idx2 >= 0 &&
        (!options.mutual_check ||
         backward[static_cast<std::size_t>(idx2)].index ==
             static_cast<int>(idx1))) {
      matches.push_back({static_cast<Point2DIdx>(idx1),
                         static_cast<Point2DIdx>(idx2)});
    }
  }
  return matches;
}

std::vector<FeatureMatch> MatchFeatureSetsLightGlue(
    const FeatureSet& features1,
    const int image_width1,
    const int image_height1,
    const FeatureSet& features2,
    const int image_width2,
    const int image_height2,
    const LightGlueMatchingOptions& options) {
  if (features1.extractor_type != features2.extractor_type ||
      !IsFeatureMatcherCompatible(features1.extractor_type,
                                  options.matcher_type) ||
      options.min_score < 0.0F || options.min_score > 1.0F ||
      options.num_threads == 0) {
    throw std::invalid_argument("Invalid LightGlue feature contract or options");
  }
#if defined(COLITE_ENABLE_ONNX)
  features::OnnxModel& model = CachedLightGlueModel(options);
  const std::size_t input_count = model.InputNames().size();
  if ((input_count != 6 && input_count != 10) ||
      model.OutputNames().size() != 2) {
    throw std::runtime_error("LightGlue model must have 6 or 10 inputs and 2 outputs");
  }
  const std::vector<std::string> expected_names = {
      "kpts0", "kpts1", "desc0", "desc1", "image_size0", "image_size1",
      "scales0", "scales1", "oris0", "oris1"};
  const std::vector<std::vector<std::int64_t>> expected_shapes = {
      {-1, -1, 2}, {-1, -1, 2}, {-1, -1, -1}, {-1, -1, -1},
      {-1, 2}, {-1, 2}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};
  for (std::size_t idx = 0; idx < input_count; ++idx) {
    features::ValidateOnnxNode(model.InputNames()[idx], expected_names[idx],
                               model.InputShapes()[idx], expected_shapes[idx]);
  }
  features::ValidateOnnxNode(model.OutputNames()[0], "matches0",
                             model.OutputShapes()[0], {-1});
  features::ValidateOnnxNode(model.OutputNames()[1], "mscores0",
                             model.OutputShapes()[1], {-1});
  LightGlueInput input1 = PrepareLightGlueInput(
      features1, image_width1, image_height1);
  LightGlueInput input2 = PrepareLightGlueInput(
      features2, image_width2, image_height2);
  if (input1.descriptor_dim != input2.descriptor_dim) {
    throw std::invalid_argument("LightGlue descriptor dimensions differ");
  }
  const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
  std::vector<std::int64_t> keypoint_shape1 = {1, input1.num_features, 2};
  std::vector<std::int64_t> keypoint_shape2 = {1, input2.num_features, 2};
  std::vector<std::int64_t> descriptor_shape1 = {
      1, input1.num_features, input1.descriptor_dim};
  std::vector<std::int64_t> descriptor_shape2 = {
      1, input2.num_features, input2.descriptor_dim};
  std::vector<std::int64_t> image_size_shape = {1, 2};
  std::vector<std::int64_t> feature_shape1 = {1, input1.num_features};
  std::vector<std::int64_t> feature_shape2 = {1, input2.num_features};
  std::vector<Ort::Value> tensors;
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input1.keypoints.data(), input1.keypoints.size(),
      keypoint_shape1.data(), keypoint_shape1.size()));
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input2.keypoints.data(), input2.keypoints.size(),
      keypoint_shape2.data(), keypoint_shape2.size()));
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input1.descriptors.data(), input1.descriptors.size(),
      descriptor_shape1.data(), descriptor_shape1.size()));
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input2.descriptors.data(), input2.descriptors.size(),
      descriptor_shape2.data(), descriptor_shape2.size()));
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input1.image_size, 2, image_size_shape.data(), image_size_shape.size()));
  tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory, input2.image_size, 2, image_size_shape.data(), image_size_shape.size()));
  if (input_count == 10) {
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory, input1.scales.data(), input1.scales.size(),
        feature_shape1.data(), feature_shape1.size()));
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory, input2.scales.data(), input2.scales.size(),
        feature_shape2.data(), feature_shape2.size()));
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory, input1.orientations.data(), input1.orientations.size(),
        feature_shape1.data(), feature_shape1.size()));
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory, input2.orientations.data(), input2.orientations.size(),
        feature_shape2.data(), feature_shape2.size()));
  }
  const std::vector<Ort::Value> outputs = model.Run(tensors);
  const auto match_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<std::int64_t> match_shape = match_info.GetShape();
  const std::vector<std::int64_t> score_shape =
      outputs[1].GetTensorTypeAndShapeInfo().GetShape();
  if (match_shape != std::vector<std::int64_t>({input1.num_features}) ||
      score_shape != match_shape) {
    throw std::runtime_error("Invalid LightGlue output shapes");
  }
  const float* scores = outputs[1].GetTensorData<float>();
  std::vector<FeatureMatch> matches;
  const auto append = [&](int idx1, std::int64_t idx2) {
    if (idx2 >= 0 && idx2 < input2.num_features &&
        scores[idx1] >= options.min_score) {
      matches.push_back({static_cast<Point2DIdx>(idx1),
                         static_cast<Point2DIdx>(idx2)});
    }
  };
  if (match_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const std::int64_t* indices = outputs[0].GetTensorData<std::int64_t>();
    for (int idx = 0; idx < input1.num_features; ++idx) append(idx, indices[idx]);
  } else if (match_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* indices = outputs[0].GetTensorData<float>();
    for (int idx = 0; idx < input1.num_features; ++idx) {
      append(idx, static_cast<std::int64_t>(std::llround(indices[idx])));
    }
  } else {
    throw std::runtime_error("Unsupported LightGlue match index type");
  }
  return matches;
#else
  (void)image_width1;
  (void)image_height1;
  (void)image_width2;
  (void)image_height2;
  throw std::runtime_error(
      "LightGlue requires a build with COLITE_ENABLE_ONNX=ON");
#endif
}

}  // namespace colite
