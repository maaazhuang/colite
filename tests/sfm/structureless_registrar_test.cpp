#include "colite/sfm/structureless_registrar.h"

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite {
namespace {

Pose3 CameraAt(const double center_x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
  return pose;
}

Image MakeImage(const ImageId id, const double center_x, const bool registered) {
  Image image;
  image.id = id;
  image.camera_id = 1;
  image.registered = registered;
  if (registered) {
    Pose3 pose;
    pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
    image.cam_from_world = pose;
  }
  return image;
}

TwoViewGeometry RelativeGeometry(const double translation_x) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1.t_cw =
      Eigen::Vector3d(translation_x, 0.0, 0.0);
  geometry.inlier_matches.resize(20);
  return geometry;
}

TEST(StructurelessRegistrarTest, CombinesRegisteredNeighborRelativePoses) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 0.0, true));
  reconstruction.AddImage(MakeImage(2, 2.0, true));
  reconstruction.AddImage(MakeImage(3, 0.0, false));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, RelativeGeometry(-1.0));
  SetVerifiedGeometry(&graph, 2, 3, RelativeGeometry(1.0));

  const StructurelessRegistrationSummary summary =
      EstimateStructurelessPose(graph, 3, reconstruction);

  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_consistent_hypotheses, 2U);
  EXPECT_NEAR(summary.pose.CameraCenter().x(), 1.0, 1e-10);
  EXPECT_NEAR(summary.median_center_disagreement, 0.0, 1e-10);
}

TEST(StructurelessRegistrarTest, RejectsSingleNeighborHypothesis) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 0.0, true));
  reconstruction.AddImage(MakeImage(3, 0.0, false));
  reconstruction.RegisterImage(1);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, RelativeGeometry(-1.0));

  const StructurelessRegistrationSummary summary =
      EstimateStructurelessPose(graph, 3, reconstruction);
  EXPECT_FALSE(summary.success);
}

TEST(StructurelessRegistrarTest, IncrementalMapperUsesExplicitFallback) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 800;
  camera.height = 600;
  camera.params = {700.0, 700.0, 400.0, 300.0};
  reconstruction.AddCamera(camera);
  const std::vector<double> centers{0.0, 2.0, 1.0};
  MatchGraph graph;
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 5; ++col) {
      points.emplace_back(-0.7 + 0.35 * col, -0.4 + 0.28 * row,
                          4.0 + 0.05 * col);
    }
  }
  for (std::size_t image_idx = 0; image_idx < centers.size(); ++image_idx) {
    Image image;
    image.id = static_cast<ImageId>(image_idx + 1);
    image.camera_id = 1;
    FeatureSet features;
    const Pose3 pose = CameraAt(centers[image_idx]);
    for (const Eigen::Vector3d& point : points) {
      const auto pixel = ProjectWorldPoint(camera, pose, point);
      ASSERT_TRUE(pixel.has_value());
      features.keypoints.push_back(
          {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
           1.0F, 0.0F, 1.0F});
      if (image_idx < 2) {
        image.points2d.push_back(
            {*pixel, static_cast<Point2DIdx>(features.keypoints.size() - 1),
             kInvalidPoint3DId});
      }
    }
    SetFeatures(&graph, image.id, features);
    if (image_idx < 2) {
      image.cam_from_world = pose;
    }
    reconstruction.AddImage(image);
    if (image_idx < 2) {
      reconstruction.RegisterImage(image.id);
    }
  }
  TwoViewGeometry geometry13 = RelativeGeometry(-1.0);
  TwoViewGeometry geometry23 = RelativeGeometry(1.0);
  geometry13.inlier_matches.clear();
  geometry23.inlier_matches.clear();
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    const FeatureMatch match{static_cast<Point2DIdx>(idx),
                             static_cast<Point2DIdx>(idx)};
    geometry13.inlier_matches.push_back(match);
    geometry23.inlier_matches.push_back(match);
  }
  SetVerifiedGeometry(&graph, 1, 3, geometry13);
  SetVerifiedGeometry(&graph, 2, 3, geometry23);

  IncrementalMapperOptions options;
  options.use_structureless_registration = true;
  options.structureless_registration.min_num_inliers_per_edge = 8;
  options.triangulation.min_triangulation_angle_deg = 0.1;
  options.run_local_bundle_adjustment = false;
  options.run_local_retriangulation_filter = false;
  options.run_observation_filter = false;
  const IncrementalMapperSummary summary =
      RegisterNextImage(graph, 3, options, &reconstruction);

  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_TRUE(summary.used_structureless_registration);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_GT(reconstruction.Points3D().size(), 0U);
}

}  // namespace
}  // namespace colite
