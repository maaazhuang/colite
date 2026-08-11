#include "colite/sfm/model_clusterer.h"

#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(ModelClustererTest, PartitionsCoreImagesAndAddsControlledOverlap) {
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches.push_back({0, 0});
  for (ImageId image_id = 1; image_id < 8; ++image_id) {
    SetVerifiedGeometry(&graph, image_id, image_id + 1, geometry);
  }
  ModelClusteringOptions options;
  options.target_cluster_size = 3;
  options.max_cluster_size = 4;
  options.overlap = 1;
  ModelClusteringSummary summary;
  const std::vector<ImageId> image_ids{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<ImageCluster> clusters =
      ClusterVerifiedViewGraph(image_ids, graph, options, &summary);

  ASSERT_TRUE(summary.success);
  EXPECT_GE(clusters.size(), 3U);
  std::set<ImageId> core_ids;
  for (const ImageCluster& cluster : clusters) {
    EXPECT_LE(cluster.image_ids.size(), 4U);
    EXPECT_LE(cluster.image_ids.size() - cluster.core_image_ids.size(), 1U);
    core_ids.insert(cluster.core_image_ids.begin(), cluster.core_image_ids.end());
  }
  EXPECT_EQ(core_ids, (std::set<ImageId>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_GT(summary.num_overlap_assignments, 0U);
}

TEST(ModelClustererTest, KeepsIsolatedImages) {
  MatchGraph graph;
  ModelClusteringOptions options;
  options.target_cluster_size = 2;
  options.max_cluster_size = 2;
  const auto clusters = ClusterVerifiedViewGraph({3, 7}, graph, options);
  ASSERT_EQ(clusters.size(), 2U);
  EXPECT_EQ(clusters[0].core_image_ids.size(), 1U);
  EXPECT_EQ(clusters[1].core_image_ids.size(), 1U);
}

}  // namespace
}  // namespace colite
