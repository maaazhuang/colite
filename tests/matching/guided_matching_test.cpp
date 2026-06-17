#include "colite/matching/guided_matching.h"

#include <gtest/gtest.h>

namespace colite {
namespace {

FeatureSet MakeFeatures(const std::vector<std::pair<float, float>>& points) {
  FeatureSet features;
  for (const auto& [x, y] : points) {
    features.keypoints.push_back({x, y, 1.0F, 0.0F});
  }
  return features;
}

TEST(GuidedMatchingTest, KeepsMatchesConsistentWithFundamentalMatrix) {
  const FeatureSet features1 =
      MakeFeatures({{10.0F, 20.0F}, {30.0F, 40.0F}, {50.0F, 60.0F}});
  const FeatureSet features2 =
      MakeFeatures({{15.0F, 20.2F}, {35.0F, 55.0F}, {55.0F, 59.5F}});
  const std::vector<FeatureMatch> candidates = {{0, 0}, {1, 1}, {2, 2}};

  // 水平极线约束：x2^T F x1 = y1 - y2。第 2 个候选的 y 方向偏差很大。
  Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
  fundamental(1, 2) = -1.0;
  fundamental(2, 1) = 1.0;

  GuidedMatchingOptions options;
  options.max_sampson_error_px = 1.0;

  const std::vector<FeatureMatch> guided = FilterMatchesByGuidedFundamental(
      features1, features2, candidates, fundamental, options);

  ASSERT_EQ(guided.size(), 2U);
  EXPECT_EQ(guided[0].idx1, 0U);
  EXPECT_EQ(guided[0].idx2, 0U);
  EXPECT_EQ(guided[1].idx1, 2U);
  EXPECT_EQ(guided[1].idx2, 2U);
}

}  // namespace
}  // namespace colite
