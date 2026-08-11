#include "colite/matching/learned_matcher.h"

#include <gtest/gtest.h>

namespace colite {
namespace {

FeatureSet MakeAlikedFeatures(const DescriptorMatrix& descriptors) {
  FeatureSet features;
  features.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  features.descriptor_metric = DescriptorMetric::kCosine;
  features.descriptor_normalization = DescriptorNormalization::kL2;
  features.descriptors = descriptors;
  return features;
}

TEST(LearnedMatcherTest, MatchesCosineDescriptorsWithMutualCheck) {
  DescriptorMatrix descriptors1(2, 3);
  DescriptorMatrix descriptors2(3, 3);
  descriptors1 << 1.0F, 0.0F, 0.0F,
                  0.0F, 1.0F, 0.0F;
  descriptors2 << 0.99F, 0.01F, 0.0F,
                  0.01F, 0.99F, 0.0F,
                  0.0F, 0.0F, 1.0F;
  const std::vector<FeatureMatch> matches = MatchAlikedDescriptors(
      MakeAlikedFeatures(descriptors1), MakeAlikedFeatures(descriptors2));
  ASSERT_EQ(matches.size(), 2U);
  EXPECT_EQ(matches[0].idx1, 0U);
  EXPECT_EQ(matches[0].idx2, 0U);
  EXPECT_EQ(matches[1].idx1, 1U);
  EXPECT_EQ(matches[1].idx2, 1U);
}

TEST(LearnedMatcherTest, AppliesCosineThresholdAndRatio) {
  DescriptorMatrix query(1, 2);
  DescriptorMatrix train(2, 2);
  query << 1.0F, 0.0F;
  train << 0.8F, 0.6F,
           0.79F, 0.61F;
  CosineMatchingOptions options;
  options.min_cosine_similarity = 0.7F;
  options.max_ratio = 0.8F;
  options.mutual_check = false;
  EXPECT_TRUE(MatchAlikedDescriptors(MakeAlikedFeatures(query),
                                     MakeAlikedFeatures(train), options)
                  .empty());
  options.max_ratio = 1.0F;
  EXPECT_EQ(MatchAlikedDescriptors(MakeAlikedFeatures(query),
                                   MakeAlikedFeatures(train), options)
                .size(),
            1U);
}

TEST(LearnedMatcherTest, RejectsSiftAndVariantMismatch) {
  DescriptorMatrix descriptors = DescriptorMatrix::Identity(2, 2);
  FeatureSet aliked = MakeAlikedFeatures(descriptors);
  FeatureSet other = aliked;
  other.extractor_type = FeatureExtractorType::kAlikedN32;
  EXPECT_THROW(MatchAlikedDescriptors(aliked, other), std::invalid_argument);
  FeatureSet sift;
  sift.descriptors = descriptors;
  EXPECT_THROW(MatchAlikedDescriptors(sift, sift), std::invalid_argument);
}

}  // namespace
}  // namespace colite
