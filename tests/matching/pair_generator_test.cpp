#include <vector>

#include <gtest/gtest.h>

#include "colite/core/image.h"
#include "colite/core/match_graph.h"
#include "colite/matching/pair_generator.h"

namespace colite {
namespace {

std::vector<Image> MakeImages(const std::size_t count) {
  std::vector<Image> images;
  images.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Image image;
    image.id = static_cast<ImageId>(i + 1);
    image.name = "image" + std::to_string(i + 1) + ".jpg";
    images.push_back(image);
  }
  return images;
}

std::vector<ImagePair> MakePairsTextFriendly(std::vector<ImagePair> pairs) {
  std::sort(pairs.begin(), pairs.end(), [](const ImagePair& lhs,
                                           const ImagePair& rhs) {
    return std::tie(lhs.image_id1, lhs.image_id2) <
           std::tie(rhs.image_id1, rhs.image_id2);
  });
  return pairs;
}

TEST(MatchingPairGeneratorTest, ExhaustiveGeneratesEveryUniquePair) {
  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kExhaustive;

  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(MakeImages(4), options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {1, 3}, {1, 4},
                                    {2, 3}, {2, 4}, {3, 4}}));
}

TEST(MatchingPairGeneratorTest, SequentialUsesForwardOverlapWindow) {
  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kSequential;
  options.sequential_overlap = 2;

  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(MakeImages(5), options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {1, 3}, {2, 3}, {2, 4},
                                    {3, 4}, {3, 5}, {4, 5}}));
}

TEST(MatchingPairGeneratorTest, SpatialConnectsNearestPriorsWithinDistance) {
  std::vector<Image> images = MakeImages(4);
  images[0].prior_position = Eigen::Vector3d(0.0, 0.0, 0.0);
  images[1].prior_position = Eigen::Vector3d(1.0, 0.0, 0.0);
  images[2].prior_position = Eigen::Vector3d(3.0, 0.0, 0.0);
  images[3].prior_position = Eigen::Vector3d(100.0, 0.0, 0.0);

  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kSpatial;
  options.spatial_max_num_neighbors = 2;
  options.spatial_max_distance = 3.5;

  const std::vector<ImagePair> pairs = GenerateMatchingPairs(images, options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {1, 3}, {2, 3}}));
}

TEST(MatchingPairGeneratorTest, SpatialRejectsImagesWithoutPriors) {
  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kSpatial;

  EXPECT_THROW(GenerateMatchingPairs(MakeImages(2), options),
               std::invalid_argument);
}

FeatureSet MakeOneClusterFeatures(const float a, const float b) {
  FeatureSet features;
  features.descriptors.resize(4, 4);
  features.descriptors.setZero();
  for (int row = 0; row < features.descriptors.rows(); ++row) {
    features.descriptors(row, 0) = a;
    features.descriptors(row, 1) = b;
    features.descriptors(row, 2) = static_cast<float>(row) * 0.01F;
  }
  return features;
}

TEST(MatchingPairGeneratorTest, VocabTreeRetrievesMostSimilarImages) {
  std::vector<Image> images = MakeImages(4);
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeOneClusterFeatures(1.0F, 0.0F));
  SetFeatures(&graph, 2, MakeOneClusterFeatures(0.95F, 0.05F));
  SetFeatures(&graph, 3, MakeOneClusterFeatures(0.0F, 1.0F));
  SetFeatures(&graph, 4, MakeOneClusterFeatures(0.05F, 0.95F));

  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kVocabTree;
  options.vocab_tree_num_images = 1;
  options.vocab_tree_num_visual_words = 2;

  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(images, graph, options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {3, 4}}));
}

}  // namespace
}  // namespace colite
