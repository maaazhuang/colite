#include "colite/mvs/dense_cloud.h"

#include <gtest/gtest.h>

namespace colite::mvs {
namespace {

FusedPoint MakePoint(double x, double y, double z) {
  FusedPoint point;
  point.xyz = Eigen::Vector3d(x, y, z);
  point.normal = Eigen::Vector3f(0.0F, 0.0F, 1.0F);
  return point;
}

TEST(DenseCloudFilterTest, RemovesPointsWithTooFewObservations) {
  const std::vector<FusedPoint> points = {MakePoint(0.0, 0.0, 0.0),
                                          MakePoint(1.0, 0.0, 0.0)};
  std::vector<StereoFusionTrack> tracks(2);
  tracks[0].point_idx = 0;
  tracks[0].num_observations = 1;
  tracks[1].point_idx = 1;
  tracks[1].num_observations = 3;

  DenseCloudFilterOptions options;
  options.min_track_observations = 2;

  const std::vector<FusedPoint> filtered =
      FilterDenseCloud(points, tracks, options);

  ASSERT_EQ(filtered.size(), 1U);
  EXPECT_DOUBLE_EQ(filtered.front().xyz.x(), 1.0);
}

TEST(DenseCloudFilterTest, KeepsOneRepresentativePerVoxel) {
  const std::vector<FusedPoint> points = {MakePoint(0.01, 0.01, 0.01),
                                          MakePoint(0.04, 0.02, 0.03),
                                          MakePoint(0.20, 0.00, 0.00)};
  std::vector<StereoFusionTrack> tracks(3);
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    tracks[i].point_idx = i;
    tracks[i].num_observations = 2;
  }

  DenseCloudFilterOptions options;
  options.min_track_observations = 2;
  options.voxel_size = 0.1;

  const std::vector<FusedPoint> filtered =
      FilterDenseCloud(points, tracks, options);

  ASSERT_EQ(filtered.size(), 2U);
  EXPECT_DOUBLE_EQ(filtered[0].xyz.x(), 0.01);
  EXPECT_DOUBLE_EQ(filtered[1].xyz.x(), 0.20);
}

TEST(DenseCloudFilterTest, RemovesCoordinateWiseSpatialTailsWhenRequested) {
  std::vector<FusedPoint> points;
  std::vector<StereoFusionTrack> tracks;
  for (int i = 0; i < 100; ++i) {
    points.push_back(MakePoint(static_cast<double>(i), 0.0, 0.0));
    tracks.emplace_back();
    tracks.back().point_idx = static_cast<std::size_t>(i);
    tracks.back().num_observations = 2;
  }
  points.push_back(MakePoint(-1000.0, 0.0, 0.0));
  points.push_back(MakePoint(1000.0, 0.0, 0.0));
  for (std::size_t i = 100; i < points.size(); ++i) {
    tracks.emplace_back();
    tracks.back().point_idx = i;
    tracks.back().num_observations = 2;
  }

  DenseCloudFilterOptions options;
  options.spatial_bounds_quantile = 0.01;
  const DenseCloudFilterResult filtered =
      FilterDenseCloudWithStats(points, tracks, options);

  ASSERT_EQ(filtered.points.size(), 100U);
  EXPECT_EQ(filtered.stats.num_removed_by_spatial_bounds, 2U);
  EXPECT_DOUBLE_EQ(filtered.points.front().xyz.x(), 0.0);
  EXPECT_DOUBLE_EQ(filtered.points.back().xyz.x(), 99.0);
}

TEST(DenseCloudMeshingTest, FourCoplanarPointsProduceTwoTriangles) {
  const std::vector<FusedPoint> points = {
      MakePoint(0.0, 0.0, 0.0), MakePoint(1.0, 0.0, 0.0),
      MakePoint(0.0, 1.0, 0.0), MakePoint(1.0, 1.0, 0.0)};

  DenseCloudMeshingOptions options;
  options.max_edge_length = 1.5;
  options.max_normal_angle_degrees = 5.0;

  const DenseTriangleMesh mesh = MeshDenseCloud(points, options);

  EXPECT_EQ(mesh.vertices.size(), 4U);
  EXPECT_GE(mesh.triangles.size(), 2U);
}

TEST(DenseCloudMeshingTest, RejectsTrianglesWithLongEdges) {
  const std::vector<FusedPoint> points = {
      MakePoint(0.0, 0.0, 0.0), MakePoint(1.0, 0.0, 0.0),
      MakePoint(0.0, 1.0, 0.0), MakePoint(5.0, 5.0, 0.0)};

  DenseCloudMeshingOptions options;
  options.max_edge_length = 1.5;
  options.max_normal_angle_degrees = 5.0;

  const DenseTriangleMesh mesh = MeshDenseCloud(points, options);

  for (const Eigen::Vector3i& triangle : mesh.triangles) {
    EXPECT_NE(triangle.x(), 3);
    EXPECT_NE(triangle.y(), 3);
    EXPECT_NE(triangle.z(), 3);
  }
}

TEST(DenseCloudMeshingTest, LimitsTrianglesToLocalNeighborCandidates) {
  std::vector<FusedPoint> points;
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      points.push_back(MakePoint(static_cast<double>(x),
                                 static_cast<double>(y), 0.0));
    }
  }

  DenseCloudMeshingOptions options;
  options.max_edge_length = 1.5;
  options.max_normal_angle_degrees = 5.0;
  options.max_candidate_neighbors = 4;

  const DenseTriangleMesh mesh = MeshDenseCloud(points, options);

  EXPECT_LT(mesh.triangles.size(), 25U * 6U);
  EXPECT_GT(mesh.triangles.size(), 0U);
}

TEST(DenseFusionTrackQualityTest,
     SummarizesSingleAndMultiObservationTracks) {
  std::vector<StereoFusionTrack> tracks(3);
  tracks[0].num_observations = 1;
  tracks[0].max_normal_angle_degrees = 0.0;
  tracks[0].max_view_angle_degrees = 5.0;
  tracks[1].num_observations = 2;
  tracks[1].max_normal_angle_degrees = 10.0;
  tracks[1].max_view_angle_degrees = 15.0;
  tracks[2].num_observations = 4;
  tracks[2].max_normal_angle_degrees = 20.0;
  tracks[2].max_view_angle_degrees = 25.0;

  const DenseFusionTrackQualityStats stats =
      ComputeFusionTrackQualityStats(tracks);

  EXPECT_EQ(stats.num_tracks, 3U);
  EXPECT_EQ(stats.num_single_observation_tracks, 1U);
  EXPECT_EQ(stats.num_multi_observation_tracks, 2U);
  EXPECT_EQ(stats.num_observations, 7U);
  EXPECT_EQ(stats.max_observations, 4U);
  EXPECT_DOUBLE_EQ(stats.mean_observations, 7.0 / 3.0);
  EXPECT_DOUBLE_EQ(stats.multi_observation_track_ratio, 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(stats.mean_max_normal_angle_degrees, 10.0);
  EXPECT_DOUBLE_EQ(stats.max_normal_angle_degrees, 20.0);
  EXPECT_DOUBLE_EQ(stats.mean_max_view_angle_degrees, 15.0);
  EXPECT_DOUBLE_EQ(stats.max_view_angle_degrees, 25.0);
}

TEST(DenseCloudMeshingTest, ComputesTriangleMeshStats) {
  DenseTriangleMesh mesh;
  mesh.vertices = {MakePoint(0.0, 0.0, 0.0), MakePoint(1.0, 0.0, 0.0),
                   MakePoint(0.0, 1.0, 0.0), MakePoint(1.0, 1.0, 0.0)};
  mesh.triangles = {Eigen::Vector3i(0, 1, 2), Eigen::Vector3i(1, 3, 2)};

  const DenseTriangleMeshStats stats = ComputeDenseTriangleMeshStats(mesh);

  EXPECT_EQ(stats.num_vertices, 4U);
  EXPECT_EQ(stats.num_triangles, 2U);
  EXPECT_GT(stats.mean_edge_length, 0.0);
  EXPECT_GT(stats.max_edge_length, 0.0);
  EXPECT_GT(stats.mean_triangle_area, 0.0);
  EXPECT_GT(stats.min_triangle_area, 0.0);
  EXPECT_GT(stats.max_triangle_area, 0.0);
}

}  // namespace
}  // namespace colite::mvs
