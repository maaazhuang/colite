#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "colite/core/types.h"

namespace colite {

// Feature and descriptor metadata is persisted with every FeatureSet. Keeping
// it explicit prevents SIFT's quantized angular-distance semantics from being
// accidentally applied to float descriptors produced by learned extractors.
enum class FeatureExtractorType : std::uint32_t {
  kSift = 1,
  kAlikedN16Rot = 2,
  kAlikedN32 = 3,
};

enum class DescriptorScalarType : std::uint32_t {
  kFloat32 = 1,
};

enum class DescriptorMetric : std::uint32_t {
  kSiftAngular = 1,
  kCosine = 2,
};

enum class DescriptorNormalization : std::uint32_t {
  kNone = 1,
  kRootL1 = 2,
  kL2 = 3,
};

enum class FeatureMatcherType : std::uint32_t {
  kSiftBruteForce = 1,
  kSiftLightGlue = 2,
  kAlikedBruteForce = 3,
  kAlikedLightGlue = 4,
};

constexpr std::string_view FeatureExtractorTypeName(
    FeatureExtractorType type) {
  switch (type) {
    case FeatureExtractorType::kSift:
      return "sift";
    case FeatureExtractorType::kAlikedN16Rot:
      return "aliked_n16rot";
    case FeatureExtractorType::kAlikedN32:
      return "aliked_n32";
  }
  return "unknown";
}

constexpr bool IsAlikedFeature(FeatureExtractorType type) {
  return type == FeatureExtractorType::kAlikedN16Rot ||
         type == FeatureExtractorType::kAlikedN32;
}

constexpr bool IsFeatureMatcherCompatible(FeatureExtractorType extractor,
                                          FeatureMatcherType matcher) {
  if (extractor == FeatureExtractorType::kSift) {
    return matcher == FeatureMatcherType::kSiftBruteForce ||
           matcher == FeatureMatcherType::kSiftLightGlue;
  }
  if (IsAlikedFeature(extractor)) {
    return matcher == FeatureMatcherType::kAlikedBruteForce ||
           matcher == FeatureMatcherType::kAlikedLightGlue;
  }
  return false;
}

// 描述子矩阵。每一行是一条描述子，使用行主序便于传给 FLANN 等 C 风格接口。
using DescriptorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// 图像局部特征点。第一版保留 OpenCV 常用字段，后续可扩展仿射形状。
struct FeatureKeypoint {
  float x = 0.0F;
  float y = 0.0F;
  float scale = 1.0F;
  float orientation = 0.0F;
  // 特征响应强度，用于特征筛选、诊断和不同后端之间的排序对齐。
  float response = 0.0F;
};

// 两张图像之间的一条特征匹配，idx1/idx2 分别索引两张图像的特征点。
struct FeatureMatch {
  Point2DIdx idx1 = kInvalidPoint2DIdx;
  Point2DIdx idx2 = kInvalidPoint2DIdx;
};

// 单张图像的特征缓存。描述子使用内部矩阵，避免匹配核心依赖 OpenCV 数据结构。
struct FeatureSet {
  FeatureExtractorType extractor_type = FeatureExtractorType::kSift;
  DescriptorScalarType descriptor_scalar_type =
      DescriptorScalarType::kFloat32;
  DescriptorMetric descriptor_metric = DescriptorMetric::kSiftAngular;
  DescriptorNormalization descriptor_normalization =
      DescriptorNormalization::kRootL1;
  std::vector<FeatureKeypoint> keypoints;
  DescriptorMatrix descriptors;
};

}  // namespace colite
