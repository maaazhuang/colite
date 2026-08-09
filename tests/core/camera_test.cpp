#include "colite/core/camera.h"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
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

std::vector<CameraModelId> AllCameraModels() {
  return {CameraModelId::kSimplePinhole,
          CameraModelId::kPinhole,
          CameraModelId::kSimpleRadial,
          CameraModelId::kRadial,
          CameraModelId::kOpenCV,
          CameraModelId::kOpenCVFisheye,
          CameraModelId::kFullOpenCV,
          CameraModelId::kFOV,
          CameraModelId::kSimpleRadialFisheye,
          CameraModelId::kRadialFisheye,
          CameraModelId::kThinPrismFisheye,
          CameraModelId::kRadTanThinPrismFisheye};
}

Camera NonzeroDistortionCamera(CameraModelId model_id) {
  Camera camera;
  camera.id = 20;
  camera.model_id = model_id;
  camera.width = 1600;
  camera.height = 1200;
  camera.params = InitializeCameraParams(model_id, 900.0, 1600, 1200);
  const CameraModelSpec& spec = GetCameraModelSpec(model_id);
  if (spec.focal_length_idxs.size() == 2) camera.params[1] = 880.0;
  for (std::size_t idx = 0; idx < spec.distortion_idxs.size(); ++idx) {
    camera.params[spec.distortion_idxs[idx]] =
        (idx % 2 == 0 ? 1.0 : -1.0) * 0.002 / (idx + 1);
  }
  if (model_id == CameraModelId::kFOV) camera.params[4] = 0.5;
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
  for (const CameraModelId model_id : AllCameraModels()) {
    EXPECT_TRUE(CameraModelProjectionImplemented(model_id))
        << CameraModelIdToName(model_id);
    EXPECT_TRUE(CameraModelUnprojectionImplemented(model_id))
        << CameraModelIdToName(model_id);
  }
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

TEST(CameraModelTest, ExpandsNamedRefineMaskForCompleteModelFamily) {
  CameraParameterRefinementMask mask;
  mask.groups = CameraParameterGroup::kFocalLength;
  mask.parameter_names = {"k4", "p2"};

  const std::vector<std::size_t> indices =
      CameraParameterIndicesForMask(CameraModelId::kFullOpenCV, mask);

  EXPECT_EQ(indices, (std::vector<std::size_t>{0, 1, 7, 9}));
}

TEST(CameraModelTest, ProjectsCompleteModelFamily) {
  const Eigen::Vector3d point_camera(0.2, -0.1, 2.0);

  for (const CameraModelId model_id :
       {CameraModelId::kSimplePinhole,
        CameraModelId::kPinhole,
        CameraModelId::kSimpleRadial,
        CameraModelId::kRadial,
        CameraModelId::kOpenCV,
        CameraModelId::kOpenCVFisheye,
        CameraModelId::kFullOpenCV,
        CameraModelId::kFOV,
        CameraModelId::kSimpleRadialFisheye,
        CameraModelId::kRadialFisheye,
        CameraModelId::kThinPrismFisheye,
        CameraModelId::kRadTanThinPrismFisheye}) {
    Camera camera;
    camera.id = 10;
    camera.model_id = model_id;
    camera.width = 1920;
    camera.height = 1080;
    camera.params = InitializeCameraParams(model_id, 1000.0, camera.width,
                                           camera.height);

    const Eigen::Vector2d pixel = ProjectPoint(camera, point_camera);

    EXPECT_TRUE(std::isfinite(pixel.x())) << CameraModelIdToName(model_id);
    EXPECT_TRUE(std::isfinite(pixel.y())) << CameraModelIdToName(model_id);
    EXPECT_NEAR(pixel.x(), 1060.0, 1.0) << CameraModelIdToName(model_id);
    EXPECT_NEAR(pixel.y(), 490.0, 1.0) << CameraModelIdToName(model_id);
  }
}

TEST(CameraModelTest, NonzeroDistortionProjectionUnprojectionRoundTrips) {
  const std::vector<Eigen::Vector3d> points = {
      {0.0, 0.0, 2.0}, {0.4, -0.2, 2.0}, {-0.7, 0.45, 2.5},
      {1.0, 0.7, 2.2}};
  for (const CameraModelId model_id : AllCameraModels()) {
    const Camera camera = NonzeroDistortionCamera(model_id);
    for (const Eigen::Vector3d& point : points) {
      const Eigen::Vector2d pixel = ProjectPoint(camera, point);
      const Eigen::Vector3d round_trip =
          UnprojectPixel(camera, pixel, point.z());
      EXPECT_TRUE(round_trip.isApprox(point, 1e-8))
          << CameraModelIdToName(model_id) << " point "
          << point.transpose() << " got " << round_trip.transpose();
    }
  }
}

TEST(CameraModelTest, RejectsInvalidCameraParameters) {
  Camera camera = NonzeroDistortionCamera(CameraModelId::kOpenCV);
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_FALSE(camera.HasBogusParams());

  camera.params[0] = 0.0;
  EXPECT_FALSE(camera.VerifyParams());
  camera = NonzeroDistortionCamera(CameraModelId::kOpenCV);
  camera.params[4] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(camera.VerifyParams());
  camera = NonzeroDistortionCamera(CameraModelId::kFOV);
  camera.params[4] = std::acos(-1.0);
  EXPECT_FALSE(camera.VerifyParams());
  camera = NonzeroDistortionCamera(CameraModelId::kPinhole);
  camera.width = 0;
  EXPECT_FALSE(camera.VerifyParams());
}

TEST(CameraModelTest, DetectsBogusButStructurallyValidParameters) {
  Camera camera = NonzeroDistortionCamera(CameraModelId::kSimpleRadial);
  camera.params[0] = 20.0;
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_TRUE(camera.HasBogusParams());
  camera.params[0] = 900.0;
  camera.params[3] = 2.0;
  EXPECT_TRUE(camera.VerifyParams());
  EXPECT_TRUE(camera.HasBogusParams());
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

TEST(CameraTest, InvalidModelProjectionThrows) {
  Camera camera;
  camera.id = 3;
  camera.model_id = CameraModelId::kInvalid;
  camera.width = 1920;
  camera.height = 1080;
  camera.params = {};

  EXPECT_THROW(ProjectPoint(camera, Eigen::Vector3d(0.0, 0.0, 1.0)),
               std::invalid_argument);
}

}  // namespace
}  // namespace colite
