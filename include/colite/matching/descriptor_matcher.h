#pragma once

#include <vector>

#include "colite/core/feature.h"

namespace colite {

// 描述子匹配后端。
// kOpenCVBruteForce：直接调用 OpenCV BFMatcher。
// kFlann：直接使用独立 FLANN 库做 KNN 搜索，流程参考 COLMAP 的 FLANN 索引思路。
enum class DescriptorMatcherBackend {
  kOpenCVBruteForce,
  kFlann,
};

struct DescriptorMatchingOptions {
  DescriptorMatcherBackend backend = DescriptorMatcherBackend::kOpenCVBruteForce;

  // Lowe ratio test 阈值。0.8 是 SIFT 常用默认值。
  double max_ratio = 0.8;

  // 是否启用互检。启用后要求 A->B 的最佳匹配和 B->A 的最佳匹配互相一致。
  bool mutual_check = true;
};

// 匹配两个 FeatureSet 的描述子，输出特征索引对。
std::vector<FeatureMatch> MatchFeatureSets(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const DescriptorMatchingOptions& options = {});

// 直接匹配两组描述子。描述子矩阵每一行对应一个特征点。
std::vector<FeatureMatch> MatchDescriptors(
    const DescriptorMatrix& descriptors1,
    const DescriptorMatrix& descriptors2,
    const DescriptorMatchingOptions& options = {});

}  // namespace colite
