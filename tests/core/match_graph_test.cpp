#include "colite/core/match_graph.h"

#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(MatchGraphTest, StoresMatchesWithOrderedImagePairAndSwapsReversedInput) {
  MatchGraph graph;
  SetMatches(&graph, 7, 3, {{2, 5}, {4, 9}});

  const std::vector<FeatureMatch>* matches = FindMatches(graph, 3, 7);

  ASSERT_NE(matches, nullptr);
  ASSERT_EQ(matches->size(), 2U);
  EXPECT_EQ((*matches)[0].idx1, 5U);
  EXPECT_EQ((*matches)[0].idx2, 2U);
  EXPECT_EQ((*matches)[1].idx1, 9U);
  EXPECT_EQ((*matches)[1].idx2, 4U);
}

TEST(MatchGraphTest, FindsVerifiedNeighborsSortedByInlierCount) {
  MatchGraph graph;

  TwoViewGeometry weak_geometry;
  weak_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  weak_geometry.inlier_matches = {{0, 0}, {1, 1}};
  SetVerifiedGeometry(&graph, 1, 2, weak_geometry);

  TwoViewGeometry strong_geometry;
  strong_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  strong_geometry.inlier_matches = {{0, 0}, {1, 1}, {2, 2}};
  SetVerifiedGeometry(&graph, 3, 1, strong_geometry);

  const std::vector<MatchGraphEdge> neighbors =
      FindVerifiedNeighbors(graph, 1);

  ASSERT_EQ(neighbors.size(), 2U);
  EXPECT_EQ(neighbors[0].OtherImageId(1), 3U);
  EXPECT_EQ(neighbors[0].NumInliers(), 3U);
  EXPECT_EQ(neighbors[1].OtherImageId(1), 2U);
  EXPECT_EQ(neighbors[1].NumInliers(), 2U);
}

}  // namespace
}  // namespace colite
