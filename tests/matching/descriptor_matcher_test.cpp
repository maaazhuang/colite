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

}  // namespace
}  // namespace colite
