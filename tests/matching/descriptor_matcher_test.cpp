#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

DescriptorMatrix MakeDescriptors(std::initializer_list<std::array<float, 4>> rows) {
  DescriptorMatrix descriptors(static_cast<Eigen::Index>(rows.size()), 4);
  Eigen::Index row_idx = 0;
  for (const auto& row : rows) {
    for (Eigen::Index col = 0; col < 4; ++col) {
      descriptors(row_idx, col) = row[static_cast<std::size_t>(col)];
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

TEST(DescriptorMatcherTest, OpenCVBruteForceMatchesWithRatioAndMutualCheck) {
  const DescriptorMatrix descriptors1 = MakeDescriptors({
      {0.0F, 0.0F, 0.0F, 0.0F},
      {10.0F, 0.0F, 0.0F, 0.0F},
      {0.0F, 10.0F, 0.0F, 0.0F},
  });
  const DescriptorMatrix descriptors2 = MakeDescriptors({
      {0.1F, 0.0F, 0.0F, 0.0F},
      {10.1F, 0.0F, 0.0F, 0.0F},
      {0.0F, 10.1F, 0.0F, 0.0F},
      {30.0F, 30.0F, 0.0F, 0.0F},
  });

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kOpenCVBruteForce;
  options.max_ratio = 0.8;
  options.mutual_check = true;

  const std::vector<FeatureMatch> matches =
      MatchDescriptors(descriptors1, descriptors2, options);

  EXPECT_EQ(ToPairs(matches),
            (std::vector<std::pair<Point2DIdx, Point2DIdx>>{
                {0, 0}, {1, 1}, {2, 2}}));
}

TEST(DescriptorMatcherTest, FlannMatchesSamePairsAsOpenCVBranch) {
  const DescriptorMatrix descriptors1 = MakeDescriptors({
      {0.0F, 0.0F, 0.0F, 0.0F},
      {10.0F, 0.0F, 0.0F, 0.0F},
      {0.0F, 10.0F, 0.0F, 0.0F},
  });
  const DescriptorMatrix descriptors2 = MakeDescriptors({
      {0.1F, 0.0F, 0.0F, 0.0F},
      {10.1F, 0.0F, 0.0F, 0.0F},
      {0.0F, 10.1F, 0.0F, 0.0F},
      {30.0F, 30.0F, 0.0F, 0.0F},
  });

  DescriptorMatchingOptions bf_options;
  bf_options.backend = DescriptorMatcherBackend::kOpenCVBruteForce;
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

TEST(DescriptorMatcherTest, RejectsAmbiguousMatchByRatio) {
  const DescriptorMatrix descriptors1 = MakeDescriptors({
      {0.0F, 0.0F, 0.0F, 0.0F},
  });
  const DescriptorMatrix descriptors2 = MakeDescriptors({
      {0.1F, 0.0F, 0.0F, 0.0F},
      {0.2F, 0.0F, 0.0F, 0.0F},
  });

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kOpenCVBruteForce;
  options.max_ratio = 0.4;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

}  // namespace
}  // namespace colite
