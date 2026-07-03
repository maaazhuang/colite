#include "colite/sfm/incremental_initializer.h"

#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"

namespace colite {
namespace {

Camera MakeCamera(CameraId id) {
  Camera camera;
  camera.id = id;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  return camera;
}

Image MakeImage(ImageId image_id, CameraId camera_id) {
  Image image;
  image.id = image_id;
  image.camera_id = camera_id;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  return image;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

void AddProjectedMatch(const Camera& camera,
                       const Pose3& cam1_from_world,
                       const Pose3& cam2_from_world,
                       const Eigen::Vector3d& point_world,
                       FeatureSet* features1,
                       FeatureSet* features2,
                       std::vector<FeatureMatch>* matches) {
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  const Point2DIdx idx =
      static_cast<Point2DIdx>(features1->keypoints.size());
  features1->keypoints.push_back(
      {static_cast<float>(pixel1->x()), static_cast<float>(pixel1->y()), 1.0F,
       0.0F});
  features2->keypoints.push_back(
      {static_cast<float>(pixel2->x()), static_cast<float>(pixel2->y()), 1.0F,
       0.0F});
  matches->push_back({idx, idx});
}

MatchGraph MakeVerifiedGraph(const Camera& camera) {
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      AddProjectedMatch(camera, cam1_from_world, cam2_from_world,
                        Eigen::Vector3d(-1.0 + 0.4 * x,
                                        -0.6 + 0.3 * y,
                                        4.0 + 0.07 * x + 0.11 * y +
                                            0.03 * x * y),
                        &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions verify_options;
  verify_options.max_error_px = 1.0;
  verify_options.min_num_inliers = 20;
  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              verify_options);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetMatches(&graph, 1, 2, matches);
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  return graph;
}

TEST(IncrementalInitializerTest, ChoosesBestInitialPairByInliersAndAngle) {
  MatchGraph graph;

  TwoViewGeometry weak_geometry;
  weak_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  weak_geometry.inlier_matches = std::vector<FeatureMatch>(24, {0, 0});
  weak_geometry.median_triangulation_angle = 1.0;
  SetVerifiedGeometry(&graph, 1, 2, weak_geometry);

  TwoViewGeometry strong_geometry = weak_geometry;
  strong_geometry.inlier_matches = std::vector<FeatureMatch>(30, {0, 0});
  strong_geometry.median_triangulation_angle = 2.0;
  SetVerifiedGeometry(&graph, 3, 4, strong_geometry);

  const auto pair = SelectInitialImagePair(graph);

  ASSERT_TRUE(pair.has_value());
  EXPECT_EQ(pair->image_id1, 3U);
  EXPECT_EQ(pair->image_id2, 4U);
}

TEST(IncrementalInitializerTest, InitializesReconstructionFromVerifiedPair) {
  const Camera camera = MakeCamera(1);
  MatchGraph graph = MakeVerifiedGraph(camera);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));

  IncrementalInitializerOptions options;
  options.min_num_inliers = 20;
  options.min_num_triangulated_points = 20;
  options.min_triangulation_angle_deg = 0.1;
  options.max_reprojection_error_px = 1.0;

  const IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.initial_pair.image_id1, 1U);
  EXPECT_EQ(summary.initial_pair.image_id2, 2U);
  EXPECT_GE(summary.num_triangulated_points, 20U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  EXPECT_TRUE(reconstruction.IsImageRegistered(2));
  EXPECT_EQ(reconstruction.RegisteredImageIds().size(), 2U);
  EXPECT_EQ(reconstruction.Points3D().size(), summary.num_triangulated_points);
  EXPECT_TRUE(reconstruction.GetImage(1).HasPose());
  EXPECT_TRUE(reconstruction.GetImage(2).HasPose());
  EXPECT_EQ(reconstruction.GetImage(1).points2d.size(),
            graph.features.at(1).keypoints.size());
  EXPECT_GT(reconstruction.GetImage(1).NumPoints3D(), 0U);
  EXPECT_GT(reconstruction.GetImage(2).NumPoints3D(), 0U);
}

TEST(IncrementalInitializerTest, TriesNextInitialPairWhenBestCandidateFails) {
  const Camera camera = MakeCamera(1);
  MatchGraph graph;

  TwoViewGeometry bad_best_geometry;
  bad_best_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  bad_best_geometry.inlier_matches = std::vector<FeatureMatch>(40, {0, 0});
  bad_best_geometry.median_triangulation_angle = 3.0;
  SetVerifiedGeometry(&graph, 1, 2, bad_best_geometry);

  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  FeatureSet features3;
  FeatureSet features4;
  std::vector<FeatureMatch> fallback_matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      AddProjectedMatch(camera, cam3_from_world, cam4_from_world,
                        Eigen::Vector3d(-1.0 + 0.4 * x,
                                        -0.6 + 0.3 * y,
                                        4.0 + 0.07 * x + 0.11 * y +
                                            0.03 * x * y),
                        &features3, &features4, &fallback_matches);
    }
  }
  TwoViewGeometryOptions verify_options;
  verify_options.max_error_px = 1.0;
  verify_options.min_num_inliers = 20;
  const TwoViewGeometry fallback_geometry = EstimateTwoViewGeometry(
      camera, features3, camera, features4, fallback_matches, verify_options);
  SetFeatures(&graph, 3, features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 3, 4, fallback_matches);
  SetVerifiedGeometry(&graph, 3, 4, fallback_geometry);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalInitializerOptions options;
  options.min_num_inliers = 20;
  options.min_num_triangulated_points = 20;
  options.min_triangulation_angle_deg = 0.1;
  options.max_reprojection_error_px = 1.0;
  options.max_num_initialization_trials = 3;

  const IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.initial_pair.image_id1, 3U);
  EXPECT_EQ(summary.initial_pair.image_id2, 4U);
  EXPECT_EQ(summary.num_initialization_attempts, 2U);
  ASSERT_EQ(summary.attempted_initial_pairs.size(), 2U);
  EXPECT_EQ(summary.attempted_initial_pairs[0], ImagePair::Ordered(1, 2));
  EXPECT_EQ(summary.attempted_initial_pairs[1], ImagePair::Ordered(3, 4));
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
}

TEST(IncrementalInitializerTest, FailsWhenNoUsableInitialPairExists) {
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kDegenerate;
  SetVerifiedGeometry(&graph, 1, 2, geometry);

  Reconstruction reconstruction;
  IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, {}, &reconstruction);

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.num_triangulated_points, 0U);
}

}  // namespace
}  // namespace colite
