#include "colite/sfm/observation_filter.h"

#include <string>

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

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

Image MakeRegisteredImage(ImageId image_id, const Pose3& cam_from_world) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  image.cam_from_world = cam_from_world;
  image.registered = true;
  return image;
}

void AddProjectedObservation(const Camera& camera,
                             const Pose3& pose,
                             const Eigen::Vector3d& point,
                             Image* image) {
  const auto pixel = ProjectWorldPoint(camera, pose, point);
  ASSERT_TRUE(pixel.has_value());
  image->points2d.push_back(
      {*pixel, static_cast<Point2DIdx>(image->points2d.size()),
       kInvalidPoint3DId});
}

TEST(ObservationFilterTest,
     RemovesHighErrorObservationsAndDeletesInvalidPoints) {
  const Camera camera = MakeCamera();
  const Pose3 pose1 = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 pose2 = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 pose3 = CameraAt(Eigen::Vector3d(0.4, 0.2, 0.0));
  const Eigen::Vector3d point_a(-0.2, 0.1, 4.2);
  const Eigen::Vector3d point_b(0.35, -0.15, 4.6);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  Image image1 = MakeRegisteredImage(1, pose1);
  Image image2 = MakeRegisteredImage(2, pose2);
  Image image3 = MakeRegisteredImage(3, pose3);
  AddProjectedObservation(camera, pose1, point_a, &image1);
  AddProjectedObservation(camera, pose2, point_a, &image2);
  AddProjectedObservation(camera, pose3, point_a, &image3);
  AddProjectedObservation(camera, pose1, point_b, &image1);
  AddProjectedObservation(camera, pose2, point_b, &image2);

  // 第三个观测明显偏离，用来模拟局部 BA 后仍然不可信的外点观测。
  image3.points2d[0].xy += Eigen::Vector2d(80.0, -40.0);
  // 第二个点只剩两条观测，其中一条是外点，过滤后 track 长度不足，应删除整点。
  image2.points2d[1].xy += Eigen::Vector2d(-90.0, 60.0);

  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);

  Track track_a;
  track_a.AddElement(1, 0);
  track_a.AddElement(2, 0);
  track_a.AddElement(3, 0);
  const Point3DId point_a_id = reconstruction.AddPoint3D(point_a, track_a);

  Track track_b;
  track_b.AddElement(1, 1);
  track_b.AddElement(2, 1);
  const Point3DId point_b_id = reconstruction.AddPoint3D(point_b, track_b);

  ObservationFilterOptions options;
  options.max_reprojection_error_px = 2.0;
  options.min_track_length = 2;
  options.min_triangulation_angle_deg = 0.1;

  const ObservationFilterSummary summary =
      FilterObservationsAndPoints(options, &reconstruction);

  EXPECT_EQ(summary.num_removed_observations, 2U);
  EXPECT_EQ(summary.num_removed_points, 1U);
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point_a_id));
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point_b_id));
  EXPECT_FALSE(reconstruction.GetImage(3).points2d[0].HasPoint3D());
  EXPECT_FALSE(reconstruction.GetImage(1).points2d[1].HasPoint3D());
  EXPECT_FALSE(reconstruction.GetImage(2).points2d[1].HasPoint3D());
  EXPECT_EQ(reconstruction.GetPoint3D(point_a_id).track.Length(), 2U);
}

}  // namespace
}  // namespace colite
