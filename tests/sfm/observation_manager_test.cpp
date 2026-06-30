#include "colite/sfm/observation_manager.h"

#include <gtest/gtest.h>

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

Image MakeImage(ImageId image_id) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.registered = true;
  image.cam_from_world = Pose3{};
  image.points2d.push_back({{500.0, 400.0}, 0, kInvalidPoint3DId});
  return image;
}

Reconstruction MakeReconstructionWithTwoImages() {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeImage(1));
  reconstruction.AddImage(MakeImage(2));
  return reconstruction;
}

}  // namespace

TEST(ObservationManagerTest, AddsObservationAndKeepsImageTrackConsistent) {
  Reconstruction reconstruction = MakeReconstructionWithTwoImages();
  Track track;
  track.AddElement(1, 0);
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);

  ObservationManager manager(&reconstruction);

  EXPECT_TRUE(manager.AddObservation(point3d_id, 2, 0));

  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id).track.Length(), 2U);
}

TEST(ObservationManagerTest, RejectsObservationLinkedToDifferentPoint) {
  Reconstruction reconstruction = MakeReconstructionWithTwoImages();
  Track track1;
  track1.AddElement(1, 0);
  const Point3DId point3d_id1 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track1);
  Track track2;
  track2.AddElement(2, 0);
  const Point3DId point3d_id2 =
      reconstruction.AddPoint3D(Eigen::Vector3d(1.0, 0.0, 5.0), track2);

  ObservationManager manager(&reconstruction);

  EXPECT_FALSE(manager.AddObservation(point3d_id1, 2, 0));

  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, point3d_id2);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id1).track.Length(), 1U);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id2).track.Length(), 1U);
}

TEST(ObservationManagerTest, MergesPointsAndReassignsSourceTrack) {
  Reconstruction reconstruction = MakeReconstructionWithTwoImages();
  Track track1;
  track1.AddElement(1, 0);
  const Point3DId target_point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track1);
  Track track2;
  track2.AddElement(2, 0);
  const Point3DId source_point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.1, 0.0, 5.0), track2);

  ObservationManager manager(&reconstruction);

  EXPECT_TRUE(manager.MergePoints(target_point3d_id, source_point3d_id,
                                  Eigen::Vector3d(0.05, 0.0, 5.0)));

  EXPECT_TRUE(reconstruction.ExistsPoint3D(target_point3d_id));
  EXPECT_FALSE(reconstruction.ExistsPoint3D(source_point3d_id));
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id,
            target_point3d_id);
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id,
            target_point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(target_point3d_id).track.Length(), 2U);
  EXPECT_NEAR(reconstruction.GetPoint3D(target_point3d_id).xyz.x(), 0.05,
              1e-12);
}

TEST(ObservationManagerTest, RejectsMergeWithImageConflict) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  Image image = MakeImage(1);
  image.points2d.push_back({{510.0, 400.0}, 1, kInvalidPoint3DId});
  reconstruction.AddImage(image);

  Track track1;
  track1.AddElement(1, 0);
  const Point3DId target_point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track1);
  Track track2;
  track2.AddElement(1, 1);
  const Point3DId source_point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.1, 0.0, 5.0), track2);

  ObservationManager manager(&reconstruction);

  EXPECT_FALSE(manager.MergePoints(target_point3d_id, source_point3d_id,
                                   Eigen::Vector3d(0.05, 0.0, 5.0)));

  EXPECT_TRUE(reconstruction.ExistsPoint3D(target_point3d_id));
  EXPECT_TRUE(reconstruction.ExistsPoint3D(source_point3d_id));
  EXPECT_EQ(reconstruction.GetPoint3D(target_point3d_id).track.Length(), 1U);
  EXPECT_EQ(reconstruction.GetPoint3D(source_point3d_id).track.Length(), 1U);
}

}  // namespace colite
