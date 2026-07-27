#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>

#include "colite/core/feature.h"

namespace colite {

// 描述子匹配后端。
// kBruteForce：自实现 CPU 暴力匹配，流程参考 COLMAP CPU brute-force matcher。
// kFlann：直接使用独立 FLANN 库做 KNN 搜索，参数对齐 COLMAP CPU FLANN matcher。
enum class DescriptorMatcherBackend {
  kBruteForce,
  kOpenCVBruteForce = kBruteForce,  // 兼容旧命名，内部已不再依赖 OpenCV matcher。
  kFlann,
  kCudaBruteForce,
};

bool IsCudaDescriptorMatcherAvailable();

struct DescriptorMatchingOptions {
  DescriptorMatcherBackend backend = DescriptorMatcherBackend::kBruteForce;

  // Lowe ratio test 阈值。COLMAP SIFT 默认值为 0.8。
  double max_ratio = 0.8;

  // 最佳匹配距离上限。该阈值定义在 COLMAP 的长度 512 SIFT 描述子距离空间中。
  double max_distance = 0.7;

  // 是否启用互检。启用后要求 A->B 的最佳匹配和 B->A 的最佳匹配互相一致。
  bool mutual_check = true;
};

struct GuidedDescriptorMatchingStats {
  double candidate_generation_seconds = 0.0;
  double descriptor_search_seconds = 0.0;
};

// Pre-quantizes descriptors and, for FLANN, builds the search index once per
// image so a batch matcher does not repeat this work for every image pair.
class PreparedFeatureMatcher {
 public:
  PreparedFeatureMatcher(const FeatureSet& features,
                         DescriptorMatcherBackend backend);
  ~PreparedFeatureMatcher();
  PreparedFeatureMatcher(PreparedFeatureMatcher&&) noexcept;
  PreparedFeatureMatcher& operator=(PreparedFeatureMatcher&&) noexcept;
  PreparedFeatureMatcher(const PreparedFeatureMatcher&) = delete;
  PreparedFeatureMatcher& operator=(const PreparedFeatureMatcher&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend std::vector<FeatureMatch> MatchPreparedFeatureSets(
      const PreparedFeatureMatcher&,
      const PreparedFeatureMatcher&,
      const DescriptorMatchingOptions&);
  friend std::vector<FeatureMatch>
  MatchPreparedFeatureSetsGuidedFundamental(
      const PreparedFeatureMatcher&,
      const PreparedFeatureMatcher&,
      const Eigen::Matrix3d&,
      double,
      const DescriptorMatchingOptions&,
      GuidedDescriptorMatchingStats*);
};

std::vector<FeatureMatch> MatchPreparedFeatureSets(
    const PreparedFeatureMatcher& features1,
    const PreparedFeatureMatcher& features2,
    const DescriptorMatchingOptions& options = {});

std::vector<FeatureMatch> MatchPreparedFeatureSetsGuidedFundamental(
    const PreparedFeatureMatcher& features1,
    const PreparedFeatureMatcher& features2,
    const Eigen::Matrix3d& fundamental_matrix,
    double max_sampson_error_px,
    const DescriptorMatchingOptions& options = {},
    GuidedDescriptorMatchingStats* stats = nullptr);

// 匹配两个 FeatureSet 的描述子，输出特征索引对。
std::vector<FeatureMatch> MatchFeatureSets(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const DescriptorMatchingOptions& options = {});

// Re-search descriptors inside an epipolar corridor. This can recover a
// geometrically valid match hidden by an off-epipolar global distractor.
std::vector<FeatureMatch> MatchFeatureSetsGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const Eigen::Matrix3d& fundamental_matrix,
    double max_sampson_error_px,
    const DescriptorMatchingOptions& options = {});

// 直接匹配两组描述子。描述子矩阵每一行对应一个特征点。
std::vector<FeatureMatch> MatchDescriptors(
    const DescriptorMatrix& descriptors1,
    const DescriptorMatrix& descriptors2,
    const DescriptorMatchingOptions& options = {});

}  // namespace colite
