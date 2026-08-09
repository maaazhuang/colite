#include "colite/matching/match_graph_consistency.h"

#include <gtest/gtest.h>

namespace colite {
namespace {

FeatureSet MakeFeatures(std::size_t count) {
  FeatureSet features;
  features.keypoints.resize(count);
  return features;
}

TwoViewGeometry MakeGeometry(
    std::initializer_list<FeatureMatch> matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kUncalibratedFundamental;
  geometry.inlier_matches = matches;
  return geometry;
}

TEST(MatchGraphConsistencyTest, RemovesPairOneToManyConflicts) {
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeFeatures(3));
  SetFeatures(&graph, 2, MakeFeatures(3));
  SetVerifiedGeometry(
      &graph, 1, 2, MakeGeometry({{0, 0}, {0, 1}, {1, 1}, {2, 2}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_input_edges, 4);
  EXPECT_EQ(summary.num_pair_conflict_edges, 2);
  EXPECT_EQ(summary.num_output_edges, 2);
  const TwoViewGeometry* geometry = FindVerifiedGeometry(graph, 1, 2);
  ASSERT_NE(geometry, nullptr);
  EXPECT_EQ(geometry->inlier_matches.size(), 2);
}

TEST(MatchGraphConsistencyTest, ReportsButPreservesCrossPairConflict) {
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    SetFeatures(&graph, image_id, MakeFeatures(2));
  }
  SetVerifiedGeometry(&graph, 1, 2, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeGeometry({{1, 0}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_input_conflicted_components, 1);
  EXPECT_EQ(summary.num_input_conflicting_observations, 1);
  EXPECT_EQ(summary.num_cross_component_conflict_edges, 0);
  EXPECT_EQ(summary.num_output_edges, 3);
  EXPECT_EQ(summary.num_output_components, 1);
}

TEST(MatchGraphConsistencyTest, KeepsConsistentCycleEdges) {
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    SetFeatures(&graph, image_id, MakeFeatures(1));
  }
  SetVerifiedGeometry(&graph, 1, 2, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeGeometry({{0, 0}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_cross_component_conflict_edges, 0);
  EXPECT_EQ(summary.num_output_edges, 3);
  EXPECT_EQ(summary.num_output_components, 1);
}

}  // namespace
}  // namespace colite
