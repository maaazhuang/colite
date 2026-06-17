#include "colite/core/reconstruction.h"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace colite {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = InitializeCameraParams(camera.model_id, 500.0, 640, 480);
  return camera;
}

Image MakeImage(ImageId image_id) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  image.points2d.push_back({Eigen::Vector2d(10.0, 20.0), 0, kInvalidPoint3DId});
  image.points2d.push_back({Eigen::Vector2d(30.0, 40.0), 1, kInvalidPoint3DId});
  return image;
}

TEST(ReconstructionTest, AddsPoint3DAndLinksImageObservations) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeImage(1));
  reconstruction.AddImage(MakeImage(2));

  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 1);

  const Point3DId point_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 2.0, 3.0), track);

  EXPECT_TRUE(reconstruction.ExistsPoint3D(point_id));
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, point_id);
  EXPECT_EQ(reconstruction.GetImage(2).points2d[1].point3d_id, point_id);
  EXPECT_EQ(reconstruction.GetPoint3D(point_id).track.Length(), 2);
}

TEST(ReconstructionTest, RegistersImageOnlyWhenPoseExists) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeImage(1));

  EXPECT_THROW(reconstruction.RegisterImage(1), std::invalid_argument);

  reconstruction.GetImage(1).cam_from_world = Pose3{};
  reconstruction.RegisterImage(1);

  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  EXPECT_TRUE(reconstruction.GetImage(1).registered);
}

}  // namespace
}  // namespace colite
