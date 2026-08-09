#include "colite/sfm/reconstruction_manager.h"

#include <gtest/gtest.h>

#include "colite/core/geometry.h"

namespace colite {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  return camera;
}

Pose3 CameraAt(double x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-x, 0.0, 0.0);
  return pose;
}

std::vector<Eigen::Vector3d> ScenePoints(double x_offset = 0.0) {
  std::vector<Eigen::Vector3d> points;
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 6; ++x) {
      points.emplace_back(x_offset - 1.0 + 0.4 * x, -0.5 + 0.3 * y,
                          4.0 + 0.07 * x + 0.09 * y);
    }
  }
  return points;
}

FeatureSet Project(const Camera& camera,
                   const Pose3& pose,
                   const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const auto pixel = ProjectWorldPoint(camera, pose, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
         1.0F, 0.0F});
  }
  return features;
}

std::vector<FeatureMatch> IdentityMatches(std::size_t size) {
  std::vector<FeatureMatch> matches;
  for (std::size_t idx = 0; idx < size; ++idx) {
    matches.push_back({static_cast<Point2DIdx>(idx),
                       static_cast<Point2DIdx>(idx)});
  }
  return matches;
}

TwoViewGeometry Geometry(const Pose3& relative_pose,
                         std::size_t num_matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1 = relative_pose;
  geometry.median_triangulation_angle = 2.0;
  geometry.inlier_matches = IdentityMatches(num_matches);
  return geometry;
}

Reconstruction MakeSchema(std::size_t num_images) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  for (ImageId image_id = 1; image_id <= num_images; ++image_id) {
    Image image;
    image.id = image_id;
    image.camera_id = 1;
    image.name = "image_" + std::to_string(image_id) + ".jpg";
    reconstruction.AddImage(std::move(image));
  }
  return reconstruction;
}

MultiModelMappingOptions FastOptions() {
  MultiModelMappingOptions options;
  options.mapping.initialization.min_num_inliers = 15;
  options.mapping.initialization.min_num_triangulated_points = 15;
  options.mapping.initialization.min_triangulation_angle_deg = 0.1;
  options.mapping.initialization.max_reprojection_error_px = 1.0;
  options.mapping.mapper.registration.min_num_correspondences = 15;
  options.mapping.mapper.registration.min_num_inliers = 15;
  options.mapping.run_global_bundle_adjustment = false;
  options.mapping.run_final_global_bundle_adjustment = false;
  options.mapping.mapper.run_local_bundle_adjustment = false;
  options.mapping.mapper.run_observation_filter = false;
  options.mapping.mapper.run_local_retriangulation_filter = false;
  options.merge_models = false;
  return options;
}

TEST(ReconstructionManagerTest, ReconstructsDisconnectedComponentsAsModels) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> points1 = ScenePoints();
  const std::vector<Eigen::Vector3d> points2 = ScenePoints(0.4);
  MatchGraph graph;
  const std::vector<Pose3> poses = {
      CameraAt(0.0), CameraAt(1.0), CameraAt(5.0), CameraAt(6.0)};
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    SetFeatures(&graph, image_id,
                Project(camera, poses[image_id - 1],
                        image_id <= 2 ? points1 : points2));
  }
  SetVerifiedGeometry(&graph, 1, 2,
                      Geometry(CameraAt(1.0), points1.size()));
  SetVerifiedGeometry(&graph, 3, 4,
                      Geometry(CameraAt(1.0), points2.size()));

  const MultiModelMappingResult result = RunIncrementalMappingMultiple(
      graph, FastOptions(), MakeSchema(4));

  ASSERT_EQ(result.models.size(), 2U);
  EXPECT_EQ(result.models[0].RegisteredImageIds().size(), 2U);
  EXPECT_EQ(result.models[1].RegisteredImageIds().size(), 2U);
  EXPECT_EQ(result.num_unregistered_images, 0U);
}

TEST(ReconstructionManagerTest, ContinuesExistingSeedModel) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> points = ScenePoints();
  const Pose3 pose1 = CameraAt(0.0);
  const Pose3 pose2 = CameraAt(1.0);
  const Pose3 pose3 = CameraAt(0.5);
  MatchGraph graph;
  SetFeatures(&graph, 1, Project(camera, pose1, points));
  SetFeatures(&graph, 2, Project(camera, pose2, points));
  SetFeatures(&graph, 3, Project(camera, pose3, points));
  SetVerifiedGeometry(&graph, 1, 2, Geometry(CameraAt(1.0), points.size()));
  SetVerifiedGeometry(&graph, 1, 3, Geometry(pose3, points.size()));
  SetVerifiedGeometry(&graph, 2, 3, Geometry(CameraAt(-0.5), points.size()));

  Reconstruction seed = MakeSchema(3);
  IncrementalInitializerOptions init = FastOptions().mapping.initialization;
  ASSERT_TRUE(InitializeReconstructionFromPair(
                  graph, ImagePair::Ordered(1, 2), init, &seed)
                  .success);

  const MultiModelMappingResult result = RunIncrementalMappingMultiple(
      graph, FastOptions(), std::move(seed));

  ASSERT_EQ(result.models.size(), 1U);
  EXPECT_EQ(result.models[0].RegisteredImageIds().size(), 3U);
  EXPECT_TRUE(result.models[0].IsImageRegistered(3));
}

TEST(ReconstructionManagerTest, RejectsCheckpointWithDifferentFeatureOrder) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> points = ScenePoints();
  MatchGraph graph;
  SetFeatures(&graph, 1, Project(camera, CameraAt(0.0), points));
  SetFeatures(&graph, 2, Project(camera, CameraAt(1.0), points));
  SetVerifiedGeometry(&graph, 1, 2,
                      Geometry(CameraAt(1.0), points.size()));

  Reconstruction seed = MakeSchema(2);
  IncrementalInitializerOptions init = FastOptions().mapping.initialization;
  ASSERT_TRUE(InitializeReconstructionFromPair(
                  graph, ImagePair::Ordered(1, 2), init, &seed)
                  .success);
  graph.features.at(1).keypoints[0].x += 2.0F;

  EXPECT_THROW(
      RunIncrementalMappingMultiple(graph, FastOptions(), std::move(seed)),
      std::runtime_error);
}

TEST(ReconstructionManagerTest, MergesModelsUsingVerifiedPointPairs) {
  constexpr std::size_t kNumPoints = 12;
  Reconstruction target = MakeSchema(4);
  Reconstruction source = MakeSchema(4);
  for (ImageId image_id : {1U, 2U}) {
    Image& image = target.GetImage(image_id);
    image.cam_from_world = CameraAt(static_cast<double>(image_id - 1));
    image.points2d.resize(kNumPoints);
    target.RegisterImage(image_id);
  }
  for (ImageId image_id : {3U, 4U}) {
    Image& image = source.GetImage(image_id);
    image.cam_from_world = CameraAt(static_cast<double>(image_id - 3));
    image.points2d.resize(kNumPoints);
    source.RegisterImage(image_id);
  }

  const double scale = 2.0;
  const Eigen::Vector3d translation(3.0, -1.0, 0.5);
  for (std::size_t idx = 0; idx < kNumPoints; ++idx) {
    const Eigen::Vector3d source_xyz(
        0.2 * idx, 0.1 * static_cast<double>(idx % 3),
        3.0 + 0.03 * idx);
    const Eigen::Vector3d target_xyz = scale * source_xyz + translation;
    Track target_track;
    target_track.AddElement(1, idx);
    target_track.AddElement(2, idx);
    target.AddPoint3D(target_xyz, target_track);
    Track source_track;
    source_track.AddElement(3, idx);
    source_track.AddElement(4, idx);
    source.AddPoint3D(source_xyz, source_track);
  }

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, Geometry(Pose3{}, kNumPoints));
  ModelMergeOptions options;
  options.min_num_correspondences = 6;
  options.max_alignment_error_ratio = 0.01;
  const ModelMergeSummary summary =
      MergeReconstructions(graph, options, source, &target);

  EXPECT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.num_inliers, kNumPoints);
  EXPECT_NEAR(summary.scale, scale, 1e-9);
  EXPECT_EQ(target.RegisteredImageIds().size(), 4U);
  EXPECT_EQ(target.Points3D().size(), kNumPoints);
  for (const auto& [point_id, point] : target.Points3D()) {
    (void)point_id;
    EXPECT_EQ(point.track.Length(), 4U);
  }
}

}  // namespace
}  // namespace colite
