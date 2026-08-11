#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/image.h"
#include "colite/core/match_graph.h"
#include "colite/matching/pair_generator.h"
#include "colite/matching/vocabulary_tree.h"

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

TEST(MatchingPairGeneratorTest, TransitiveOnlyProposesNewVerifiedPathPairs) {
  const std::vector<Image> images = MakeImages(5);
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches.push_back({0, 0});
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  SetVerifiedGeometry(&graph, 2, 3, geometry);
  SetVerifiedGeometry(&graph, 3, 4, geometry);
  SetMatches(&graph, 2, 4, {{0, 0}});

  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kTransitive;
  options.transitive_max_path_length = 2;
  options.transitive_max_num_neighbors = 10;
  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(images, graph, options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 3}}));
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
  options.vocab_tree_sequential_overlap = 0;

  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(images, graph, options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {3, 4}}));
}

TEST(MatchingPairGeneratorTest, VocabTreeLoadsVisualWordsFromFile) {
  std::vector<Image> images = MakeImages(4);
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeOneClusterFeatures(1.0F, 0.0F));
  SetFeatures(&graph, 2, MakeOneClusterFeatures(0.95F, 0.05F));
  SetFeatures(&graph, 3, MakeOneClusterFeatures(0.0F, 1.0F));
  SetFeatures(&graph, 4, MakeOneClusterFeatures(0.05F, 0.95F));

  const std::filesystem::path vocab_path =
      std::filesystem::temp_directory_path() / "colite_test_vocab_tree.txt";
  std::ofstream vocab_file(vocab_path);
  vocab_file << "COLITE_VOCAB_TREE 1 4\n"
             << "1 0 0 0\n"
             << "0 1 0 0\n";
  vocab_file.close();

  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kVocabTree;
  options.vocab_tree_num_images = 1;
  options.vocab_tree_num_visual_words = 1;
  options.vocab_tree_path = vocab_path;
  options.vocab_tree_sequential_overlap = 0;

  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(images, graph, options);

  EXPECT_EQ(MakePairsTextFriendly(pairs),
            (std::vector<ImagePair>{{1, 2}, {3, 4}}));
}

TEST(MatchingPairGeneratorTest, PersistentHierarchicalTreeRoundTrips) {
  std::vector<Image> images = MakeImages(4);
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeOneClusterFeatures(1.0F, 0.0F));
  SetFeatures(&graph, 2, MakeOneClusterFeatures(0.95F, 0.05F));
  SetFeatures(&graph, 3, MakeOneClusterFeatures(0.0F, 1.0F));
  SetFeatures(&graph, 4, MakeOneClusterFeatures(0.05F, 0.95F));
  VocabularyTreeBuildOptions build_options;
  build_options.num_visual_words = 4;
  build_options.branching = 2;
  const VocabularyTree tree =
      BuildVocabularyTree(images, graph, build_options);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "colite_vocab_tree_v2.txt";
  tree.Write(path);
  const VocabularyTree loaded = VocabularyTree::Read(path);
  EXPECT_EQ(loaded.NumVisualWords(), tree.NumVisualWords());
  EXPECT_EQ(loaded.nodes.size(), tree.nodes.size());
  EXPECT_EQ(loaded.Assign({1.0F, 0.0F, 0.0F, 0.0F}),
            tree.Assign({1.0F, 0.0F, 0.0F, 0.0F}));
  std::filesystem::remove(path);
}

TEST(MatchingPairGeneratorTest, VocabTreeGuardConnectsTemporalSequence) {
  std::vector<Image> images = MakeImages(4);
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeOneClusterFeatures(1.0F, 0.0F));
  SetFeatures(&graph, 2, MakeOneClusterFeatures(0.0F, 1.0F));
  SetFeatures(&graph, 3, MakeOneClusterFeatures(1.0F, 0.0F));
  SetFeatures(&graph, 4, MakeOneClusterFeatures(0.0F, 1.0F));
  MatchingPairGeneratorOptions options;
  options.strategy = MatchingStrategy::kVocabTree;
  options.vocab_tree_num_images = 1;
  options.vocab_tree_num_visual_words = 2;
  options.vocab_tree_sequential_overlap = 1;
  MatchingPairGeneratorSummary summary;
  const std::vector<ImagePair> pairs =
      GenerateMatchingPairs(images, graph, options, &summary);
  EXPECT_EQ(summary.num_sequential_guard_pairs, 3);
  EXPECT_GT(summary.num_sequential_pairs_added, 0);
  EXPECT_EQ(summary.num_connected_components, 1);
  EXPECT_GE(summary.min_degree, 1);
  EXPECT_EQ(pairs.size(), summary.num_retrieval_pairs +
                              summary.num_sequential_pairs_added);
}

}  // namespace
}  // namespace colite
