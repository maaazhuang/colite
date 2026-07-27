#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

DescriptorMatrix MakeBlockDescriptors(std::initializer_list<int> blocks) {
  DescriptorMatrix descriptors(static_cast<Eigen::Index>(blocks.size()), 128);
  descriptors.setZero();
  Eigen::Index row_idx = 0;
  for (const int block : blocks) {
    for (Eigen::Index offset = 0; offset < 16; ++offset) {
      descriptors(row_idx, static_cast<Eigen::Index>(block * 16) + offset) =
          0.25F;
    }
    ++row_idx;
  }
  return descriptors;
}

DescriptorMatrix MakeUnitSiftDescriptors(std::initializer_list<int> blocks) {
  DescriptorMatrix descriptors(static_cast<Eigen::Index>(blocks.size()), 128);
  descriptors.setZero();
  Eigen::Index row_idx = 0;
  for (const int block : blocks) {
    for (Eigen::Index offset = 0; offset < 16; ++offset) {
      descriptors(row_idx, static_cast<Eigen::Index>(block * 16) + offset) =
          0.25F;
    }
    ++row_idx;
  }
  return descriptors;
}

std::vector<std::pair<Point2DIdx, Point2DIdx>> ToPairs(
    const std::vector<FeatureMatch>& matches) {
  std::vector<std::pair<Point2DIdx, Point2DIdx>> pairs;
  for (const FeatureMatch& match : matches) {
    pairs.emplace_back(match.idx1, match.idx2);
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

TEST(DescriptorMatcherTest, BruteForceMatchesWithRatioAndMutualCheck) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0, 1, 2});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 1, 2, 3});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 0.8;
  options.mutual_check = true;

  const std::vector<FeatureMatch> matches =
      MatchDescriptors(descriptors1, descriptors2, options);

  EXPECT_EQ(ToPairs(matches),
            (std::vector<std::pair<Point2DIdx, Point2DIdx>>{
                {0, 0}, {1, 1}, {2, 2}}));
}

TEST(DescriptorMatcherTest, FlannMatchesSamePairsAsBruteForceBranch) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0, 1, 2});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 1, 2, 3});

  DescriptorMatchingOptions bf_options;
  bf_options.backend = DescriptorMatcherBackend::kBruteForce;
  bf_options.max_ratio = 0.8;
  bf_options.mutual_check = true;

  DescriptorMatchingOptions flann_options = bf_options;
  flann_options.backend = DescriptorMatcherBackend::kFlann;

  const std::vector<FeatureMatch> bf_matches =
      MatchDescriptors(descriptors1, descriptors2, bf_options);
  const std::vector<FeatureMatch> flann_matches =
      MatchDescriptors(descriptors1, descriptors2, flann_options);

  EXPECT_EQ(ToPairs(flann_matches), ToPairs(bf_matches));
}

TEST(DescriptorMatcherTest, PreparedFlannMatchesOneShotMatcher) {
  FeatureSet features1;
  FeatureSet features2;
  features1.descriptors = MakeBlockDescriptors({0, 1, 2});
  features2.descriptors = MakeBlockDescriptors({0, 1, 2, 3});
  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kFlann;
  const PreparedFeatureMatcher prepared1(features1, options.backend);
  const PreparedFeatureMatcher prepared2(features2, options.backend);
  EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2, options)),
            ToPairs(MatchFeatureSets(features1, features2, options)));
}

TEST(DescriptorMatcherTest, CudaMatchesCpuBruteForceExactly) {
  if (!IsCudaDescriptorMatcherAvailable()) {
    GTEST_SKIP() << "CUDA descriptor matcher is unavailable";
  }
  DescriptorMatrix descriptors1(40, 128);
  DescriptorMatrix descriptors2(44, 128);
  for (Eigen::Index row = 0; row < descriptors1.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors1.cols(); ++col) {
      descriptors1(row, col) = static_cast<float>(
          ((row * 37 + col * 17 + row * col) % 256));
    }
  }
  for (Eigen::Index row = 0; row < descriptors2.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors2.cols(); ++col) {
      descriptors2(row, col) = static_cast<float>(
          ((row * 29 + col * 13 + row * col * 3) % 256));
    }
  }
  descriptors2.topRows(20) = descriptors1.topRows(20);
  descriptors1.row(39).setConstant(255.0F);
  descriptors2.row(40).setZero();
  descriptors2.row(41).setConstant(1.0F);
  descriptors2.row(42).setConstant(254.0F);
  descriptors2.row(43).setConstant(255.0F);

  DescriptorMatchingOptions cpu_options;
  cpu_options.backend = DescriptorMatcherBackend::kBruteForce;
  DescriptorMatchingOptions cuda_options = cpu_options;
  cuda_options.backend = DescriptorMatcherBackend::kCudaBruteForce;

  EXPECT_EQ(ToPairs(MatchDescriptors(descriptors1, descriptors2, cuda_options)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2, cpu_options)));

  FeatureSet features1;
  FeatureSet features2;
  features1.descriptors = descriptors1;
  features2.descriptors = descriptors2;
  const PreparedFeatureMatcher prepared1(features1, cuda_options.backend);
  const PreparedFeatureMatcher prepared2(features2, cuda_options.backend);
  EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2,
                                             cuda_options)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2,
                                     cpu_options)));
}

TEST(DescriptorMatcherTest, RejectsAmbiguousMatchByRatio) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 0});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 0.4;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, RejectsMatchBeyondColmapMaxDistance) {
  const DescriptorMatrix descriptors1 = MakeUnitSiftDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeUnitSiftDescriptors({1, 2});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;
  options.max_distance = 0.7;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, RejectsEqualBestAndSecondBestByRatio) {
  const DescriptorMatrix descriptors1 = MakeUnitSiftDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeUnitSiftDescriptors({0, 0});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;
  options.max_distance = 1.0;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, GuidedSearchRecoversMatchBehindOffEpipolarDistractor) {
  FeatureSet features1;
  FeatureSet features2;
  features1.keypoints.push_back({10.0F, 10.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({20.0F, 10.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({20.0F, 100.0F, 1.0F, 0.0F});
  features1.descriptors = MakeUnitSiftDescriptors({0});
  features2.descriptors = MakeUnitSiftDescriptors({0, 0});
  for (Eigen::Index col = 0; col < 16; ++col) {
    features2.descriptors(0, col) = 0.24F;
  }

  DescriptorMatchingOptions options;
  options.max_ratio = 0.8;
  options.max_distance = 0.7;
  options.mutual_check = true;
  const std::vector<FeatureMatch> global =
      MatchFeatureSets(features1, features2, options);
  ASSERT_EQ(global.size(), 1U);
  EXPECT_EQ(global.front().idx2, 1U);

  Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
  fundamental(1, 2) = -1.0;
  fundamental(2, 1) = 1.0;
  const std::vector<FeatureMatch> guided =
      MatchFeatureSetsGuidedFundamental(features1, features2, fundamental,
                                        2.0, options);
  ASSERT_EQ(guided.size(), 1U);
  EXPECT_EQ(guided.front().idx1, 0U);
  EXPECT_EQ(guided.front().idx2, 0U);

  const PreparedFeatureMatcher prepared1(
      features1, DescriptorMatcherBackend::kBruteForce);
  const PreparedFeatureMatcher prepared2(
      features2, DescriptorMatcherBackend::kBruteForce);
  const std::vector<FeatureMatch> prepared_guided =
      MatchPreparedFeatureSetsGuidedFundamental(
          prepared1, prepared2, fundamental, 2.0, options);
  ASSERT_EQ(prepared_guided.size(), guided.size());
  EXPECT_EQ(prepared_guided.front().idx1, guided.front().idx1);
  EXPECT_EQ(prepared_guided.front().idx2, guided.front().idx2);

  if (IsCudaDescriptorMatcherAvailable()) {
    DescriptorMatchingOptions cuda_options = options;
    cuda_options.backend = DescriptorMatcherBackend::kCudaBruteForce;
    const std::vector<FeatureMatch> cuda_guided =
        MatchFeatureSetsGuidedFundamental(
            features1, features2, fundamental, 2.0, cuda_options);
    ASSERT_EQ(cuda_guided.size(), guided.size());
    EXPECT_EQ(cuda_guided.front().idx1, guided.front().idx1);
    EXPECT_EQ(cuda_guided.front().idx2, guided.front().idx2);
  }
}

}  // namespace
}  // namespace colite
