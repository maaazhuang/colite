#include "colite/sfm/hierarchical_mapper.h"

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"

namespace colite {
namespace {

TEST(HierarchicalMapperTest, ReconstructsOverlappingClustersAndMergesThem) {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 700;
  camera.params = {850.0, 850.0, 500.0, 350.0};
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  MatchGraph graph;
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 5; ++row) {
    for (int col = 0; col < 6; ++col) {
      points.emplace_back(-0.8 + 0.32 * col, -0.55 + 0.27 * row,
                          5.0 + 0.04 * col + 0.06 * row);
    }
  }
  constexpr std::size_t kNumImages = 8;
  std::vector<double> centers;
  for (std::size_t image_idx = 0; image_idx < kNumImages; ++image_idx) {
    const double center_x = 0.4 * static_cast<double>(image_idx);
    centers.push_back(center_x);
    Pose3 pose;
    pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
    Image image;
    image.id = static_cast<ImageId>(image_idx + 1);
    image.camera_id = camera.id;
    image.name = "image" + std::to_string(image_idx + 1) + ".jpg";
    reconstruction.AddImage(image);
    FeatureSet features;
    for (const Eigen::Vector3d& point : points) {
      const std::optional<Eigen::Vector2d> pixel =
          ProjectWorldPoint(camera, pose, point);
      ASSERT_TRUE(pixel.has_value());
      features.keypoints.push_back(
          {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
           1.0F, 0.0F, 1.0F});
    }
    SetFeatures(&graph, image.id, features);
  }
  for (std::size_t i = 0; i < kNumImages; ++i) {
    for (std::size_t j = i + 1; j < kNumImages; ++j) {
      TwoViewGeometry geometry;
      geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
      geometry.cam2_from_cam1.t_cw =
          Eigen::Vector3d(centers[i] - centers[j], 0.0, 0.0);
      geometry.median_triangulation_angle = 3.0;
      for (std::size_t point_idx = 0; point_idx < points.size(); ++point_idx) {
        geometry.inlier_matches.push_back(
            {static_cast<Point2DIdx>(point_idx),
             static_cast<Point2DIdx>(point_idx)});
      }
      SetVerifiedGeometry(&graph, static_cast<ImageId>(i + 1),
                          static_cast<ImageId>(j + 1), geometry);
    }
  }

  HierarchicalMappingOptions options;
  options.clustering.target_cluster_size = 4;
  options.clustering.max_cluster_size = 6;
  options.clustering.overlap = 2;
  options.local_mapping.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.local_mapping.mapper.run_local_bundle_adjustment = false;
  options.local_mapping.mapper.run_local_retriangulation_filter = false;
  options.local_mapping.mapper.run_observation_filter = false;
  options.local_mapping.run_global_observation_filter = false;
  options.local_mapping.run_pose_guided_track_completion = false;
  options.local_mapping.run_final_global_bundle_adjustment = false;
  options.merging.min_num_correspondences = 6;
  const HierarchicalMappingSummary summary =
      RunHierarchicalMapping(graph, options, &reconstruction);

  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.clustering.num_clusters, 2U);
  EXPECT_EQ(summary.num_successful_local_models, 2U);
  EXPECT_EQ(summary.num_merged_local_models, 1U);
  EXPECT_EQ(summary.num_unmerged_local_models, 0U);
  EXPECT_EQ(summary.num_registered_images, kNumImages);
  EXPECT_TRUE(summary.final_bundle_adjustment.success);
  EXPECT_GE(summary.num_points3d, points.size() / 2);
}

TEST(HierarchicalMapperTest, RejectsRegisteredInputModel) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);
  Image image;
  image.id = 1;
  image.camera_id = 1;
  image.cam_from_world = Pose3{};
  reconstruction.AddImage(image);
  reconstruction.RegisterImage(1);
  const HierarchicalMappingSummary summary = RunHierarchicalMapping(
      MatchGraph{}, HierarchicalMappingOptions{}, &reconstruction);
  EXPECT_FALSE(summary.success);
  EXPECT_NE(summary.message.find("unregistered"), std::string::npos);
}

}  // namespace
}  // namespace colite
