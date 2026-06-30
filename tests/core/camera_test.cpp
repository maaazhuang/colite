#include "colite/core/camera.h"

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

Camera PinholeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {1000.0, 980.0, 960.0, 540.0};
  return camera;
}

Camera SimpleRadialCamera() {
  Camera camera;
  camera.id = 2;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {1000.0, 960.0, 540.0, 0.01};
  return camera;
}

Camera RadialCamera() {
  Camera camera;
  camera.id = 3;
  camera.model_id = CameraModelId::kRadial;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {1000.0, 960.0, 540.0, 0.01, -0.001};
  return camera;
}

Camera OpenCVCamera() {
  Camera camera;
  camera.id = 4;
  camera.model_id = CameraModelId::kOpenCV;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {1000.0, 980.0, 960.0, 540.0,
                   0.01, -0.001, 0.0005, -0.0003};
  return camera;
}

TEST(CameraModelTest, RegistersCompleteModelFamily) {
  EXPECT_EQ(CameraModelNameToId("PINHOLE"), CameraModelId::kPinhole);
  EXPECT_EQ(CameraModelNameToId("FULL_OPENCV"), CameraModelId::kFullOpenCV);
  EXPECT_EQ(CameraModelNameToId("RAD_TAN_THIN_PRISM_FISHEYE"),
            CameraModelId::kRadTanThinPrismFisheye);
  EXPECT_TRUE(ExistsCameraModel(CameraModelId::kOpenCVFisheye));
}

TEST(CameraModelTest, TracksProjectionImplementationState) {
  EXPECT_TRUE(CameraModelProjectionImplemented(CameraModelId::kSimplePinhole));
  EXPECT_TRUE(CameraModelProjectionImplemented(CameraModelId::kPinhole));
  EXPECT_TRUE(CameraModelProjectionImplemented(CameraModelId::kSimpleRadial));
  EXPECT_TRUE(CameraModelProjectionImplemented(CameraModelId::kRadial));
  EXPECT_TRUE(CameraModelProjectionImplemented(CameraModelId::kOpenCV));
}

TEST(CameraModelTest, InitializesSingleAndDoubleFocalModels) {
  const std::vector<double> simple =
      InitializeCameraParams(CameraModelId::kSimpleRadial, 800.0, 1000, 500);
  EXPECT_EQ(simple, (std::vector<double>{800.0, 500.0, 250.0, 0.0}));

  const std::vector<double> pinhole =
      InitializeCameraParams(CameraModelId::kPinhole, 800.0, 1000, 500);
  EXPECT_EQ(pinhole, (std::vector<double>{800.0, 800.0, 500.0, 250.0}));
}

TEST(CameraModelTest, ExpandsParameterGroupsForSingleFocalModel) {
  const std::vector<std::size_t> indices = CameraParameterIndicesForGroups(
      CameraModelId::kSimpleRadial,
      CameraParameterGroup::kFocalLength | CameraParameterGroup::kDistortion);

  EXPECT_EQ(indices, (std::vector<std::size_t>{0, 3}));
}

TEST(CameraModelTest, ExpandsParameterGroupsForDoubleFocalModel) {
  const std::vector<std::size_t> indices = CameraParameterIndicesForGroups(
      CameraModelId::kOpenCV,
      CameraParameterGroup::kFocalLength |
          CameraParameterGroup::kPrincipalPoint |
          CameraParameterGroup::kDistortion);

  EXPECT_EQ(indices,
            (std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(CameraTest, PinholeProjectionAndUnprojectionRoundTrip) {
  const Camera camera = PinholeCamera();
  const Eigen::Vector3d point_camera(0.25, -0.1, 2.0);

  const Eigen::Vector2d pixel = ProjectPoint(camera, point_camera);
  const Eigen::Vector3d round_trip =
      UnprojectPixel(camera, pixel, point_camera.z());

  EXPECT_NEAR(round_trip.x(), point_camera.x(), 1e-12);
  EXPECT_NEAR(round_trip.y(), point_camera.y(), 1e-12);
  EXPECT_NEAR(round_trip.z(), point_camera.z(), 1e-12);
}

TEST(CameraTest, SimpleRadialProjectionAndUnprojectionRoundTrip) {
  const Camera camera = SimpleRadialCamera();
  const Eigen::Vector3d point_camera(0.15, -0.08, 2.0);

  const Eigen::Vector2d pixel = ProjectPoint(camera, point_camera);
  const Eigen::Vector3d round_trip =
      UnprojectPixel(camera, pixel, point_camera.z());

  EXPECT_NEAR(round_trip.x(), point_camera.x(), 1e-9);
  EXPECT_NEAR(round_trip.y(), point_camera.y(), 1e-9);
  EXPECT_NEAR(round_trip.z(), point_camera.z(), 1e-12);
}

TEST(CameraTest, RadialProjectionAndUnprojectionRoundTrip) {
  const Camera camera = RadialCamera();
  const Eigen::Vector3d point_camera(0.16, -0.07, 2.0);

  const Eigen::Vector2d pixel = ProjectPoint(camera, point_camera);
  const Eigen::Vector3d round_trip =
      UnprojectPixel(camera, pixel, point_camera.z());

  EXPECT_NEAR(round_trip.x(), point_camera.x(), 1e-9);
  EXPECT_NEAR(round_trip.y(), point_camera.y(), 1e-9);
  EXPECT_NEAR(round_trip.z(), point_camera.z(), 1e-12);
}

TEST(CameraTest, OpenCVProjectionAndUnprojectionRoundTrip) {
  const Camera camera = OpenCVCamera();
  const Eigen::Vector3d point_camera(0.18, -0.09, 2.0);

  const Eigen::Vector2d pixel = ProjectPoint(camera, point_camera);
  const Eigen::Vector3d round_trip =
      UnprojectPixel(camera, pixel, point_camera.z());

  EXPECT_NEAR(round_trip.x(), point_camera.x(), 1e-9);
  EXPECT_NEAR(round_trip.y(), point_camera.y(), 1e-9);
  EXPECT_NEAR(round_trip.z(), point_camera.z(), 1e-12);
}

TEST(CameraTest, ProjectionRejectsNonPositiveDepth) {
  const Camera camera = PinholeCamera();
  EXPECT_THROW(ProjectPoint(camera, Eigen::Vector3d(0.0, 0.0, 0.0)),
               std::invalid_argument);
}

TEST(CameraTest, UnimplementedProjectionThrows) {
  Camera camera;
  camera.id = 3;
  camera.model_id = CameraModelId::kFOV;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = InitializeCameraParams(camera.model_id, 1000.0, 1920, 1080);

  EXPECT_THROW(ProjectPoint(camera, Eigen::Vector3d(0.0, 0.0, 1.0)),
               std::domain_error);
}

}  // namespace
}  // namespace colite
