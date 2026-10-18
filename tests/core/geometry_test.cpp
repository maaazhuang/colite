#include "colite/core/geometry.h"

#include <cmath>

#include <gtest/gtest.h>

#include "colite/core/reconstruction.h"

namespace colite {
namespace {

Camera MakePinholeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {500.0, 500.0, 500.0, 400.0};
  return camera;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

TEST(PoseTest, InverseReturnsOriginalPoint) {
  Pose3 pose;
  pose.q_cw = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY());
  pose.t_cw = Eigen::Vector3d(1.0, 2.0, -0.5);

  const Eigen::Vector3d point_world(0.3, -0.2, 5.0);
  const Eigen::Vector3d point_camera = pose.TransformPoint(point_world);
  const Eigen::Vector3d round_trip = pose.Inverse().TransformPoint(point_camera);

  EXPECT_NEAR((round_trip - point_world).norm(), 0.0, 1e-12);
}

TEST(PoseTest, CompositionMatchesSequentialApplication) {
  Pose3 a;
  a.q_cw = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ());
  a.t_cw = Eigen::Vector3d(1.0, 0.0, 0.0);

  Pose3 b;
  b.q_cw = Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitX());
  b.t_cw = Eigen::Vector3d(0.0, 2.0, 0.0);

  const Eigen::Vector3d point(0.5, -0.3, 4.0);
  const Pose3 composed = a * b;

  EXPECT_NEAR((composed.TransformPoint(point) -
               a.TransformPoint(b.TransformPoint(point))).norm(),
              0.0,
              1e-12);
}

TEST(GeometryTest, ProjectsWorldPointWithPose) {
  const Camera camera = MakePinholeCamera();
  const Pose3 pose = CameraAt(Eigen::Vector3d::Zero());
  const Eigen::Vector3d point_world(0.5, -0.2, 5.0);

  const auto pixel = ProjectWorldPoint(camera, pose, point_world);

  ASSERT_TRUE(pixel.has_value());
  EXPECT_NEAR(pixel->x(), 550.0, 1e-12);
  EXPECT_NEAR(pixel->y(), 380.0, 1e-12);
}

TEST(GeometryTest, ReprojectionErrorIsMeasuredInPixels) {
  const Camera camera = MakePinholeCamera();
  const Pose3 pose = CameraAt(Eigen::Vector3d::Zero());
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);

  const auto error =
      ReprojectionError(camera, pose, point_world, Eigen::Vector2d(503.0, 404.0));

  ASSERT_TRUE(error.has_value());
  EXPECT_NEAR(*error, 5.0, 1e-12);
}

TEST(GeometryTest, ComputesColmapMeanPointTrackReprojectionError) {
  Reconstruction reconstruction;
  const Camera camera = MakePinholeCamera();
  reconstruction.AddCamera(camera);

  Image image1;
  image1.id = 1;
  image1.camera_id = camera.id;
  image1.cam_from_world = CameraAt(Eigen::Vector3d::Zero());
  image1.registered = true;
  image1.points2d.push_back({Eigen::Vector2d(503.0, 404.0), 0, 1});
  reconstruction.AddImage(image1);

  Image image2 = image1;
  image2.id = 2;
  image2.points2d[0].xy = Eigen::Vector2d(500.0, 400.0);
  reconstruction.AddImage(image2);

  Point3D point;
  point.id = 1;
  point.xyz = Eigen::Vector3d(0.0, 0.0, 5.0);
  point.track.AddElement(1, 0);
  point.track.AddElement(2, 0);
  const auto error = Point3DTrackReprojectionError(reconstruction, point);

  ASSERT_TRUE(error.has_value());
  // COLMAP averages the two pixel distances (5 and 0), rather than their squares.
  EXPECT_NEAR(*error, 2.5, 1e-12);
  point.reprojection_error = 99.0;
  reconstruction.AddPoint3D(point);
  UpdatePoint3DReprojectionErrors(&reconstruction);
  EXPECT_NEAR(reconstruction.GetPoint3D(1).reprojection_error, 2.5, 1e-12);
}

TEST(GeometryTest, TriangulatesPointFromTwoViews) {
  const Camera camera = MakePinholeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.2, -0.1, 5.0);

  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  const auto triangulated =
      TriangulatePoint(camera, cam1_from_world, *pixel1,
                       camera, cam2_from_world, *pixel2);

  ASSERT_TRUE(triangulated.has_value());
  EXPECT_NEAR((triangulated.value() - point_world).norm(), 0.0, 1e-9);
}

TEST(GeometryTest, ComputesTriangulationAngle) {
  const Camera camera = MakePinholeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);

  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  const double angle = TriangulationAngleDegrees(
      cam1_from_world,
      PixelToBearing(camera, *pixel1),
      cam2_from_world,
      PixelToBearing(camera, *pixel2));

  EXPECT_GT(angle, 0.0);
  EXPECT_LT(angle, 20.0);
}

}  // namespace
}  // namespace colite
