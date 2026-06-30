#include "colite/sfm/local_bundle_adjuster.h"

#include <cmath>
#include <map>
#include <string>
#include <vector>

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

Image MakeImage(ImageId image_id, const Pose3& cam_from_world) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  image.cam_from_world = cam_from_world;
  image.registered = true;
  return image;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

std::vector<Eigen::Vector3d> MakeScenePoints() {
  std::vector<Eigen::Vector3d> points;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      points.emplace_back(-1.0 + 0.4 * x, -0.6 + 0.3 * y,
                          4.0 + 0.08 * x + 0.12 * y + 0.02 * x * y);
    }
  }
  return points;
}

void AddImageObservations(const Camera& camera,
                          const Pose3& true_pose,
                          Image* image,
                          const std::vector<Eigen::Vector3d>& points) {
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto pixel = ProjectWorldPoint(camera, true_pose, points[i]);
    ASSERT_TRUE(pixel.has_value());
    image->points2d.push_back(
        {*pixel, static_cast<Point2DIdx>(i), kInvalidPoint3DId});
  }
}

void AddTrackedPoint(Reconstruction* reconstruction,
                     const Camera& camera,
                     const std::map<ImageId, Pose3>& true_poses,
                     const Eigen::Vector3d& point,
                     const std::vector<ImageId>& image_ids) {
  Track track;
  for (const ImageId image_id : image_ids) {
    const auto pixel = ProjectWorldPoint(camera, true_poses.at(image_id), point);
    ASSERT_TRUE(pixel.has_value());
    Image& image = reconstruction->GetImage(image_id);
    const Point2DIdx point2d_idx =
        static_cast<Point2DIdx>(image.points2d.size());
    image.points2d.push_back({*pixel, point2d_idx, kInvalidPoint3DId});
    track.AddElement(image_id, point2d_idx);
  }
  reconstruction->AddPoint3D(point, track);
}

double MeanReprojectionError(const Reconstruction& reconstruction) {
  double total_error = 0.0;
  std::size_t num_errors = 0;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    for (const TrackElement& element : point3d.track.elements) {
      const Image& image = reconstruction.GetImage(element.image_id);
      const Camera& camera = reconstruction.GetCamera(image.camera_id);
      const auto error =
          ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                            image.points2d[element.point2d_idx].xy);
      if (error.has_value()) {
        total_error += *error;
        ++num_errors;
      }
    }
  }
  return total_error / static_cast<double>(num_errors);
}

TEST(LocalBundleAdjusterTest, ReducesReprojectionErrorForLocalWindow) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> true_points = MakeScenePoints();
  const Pose3 true_pose1 = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 true_pose2 = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 true_pose3 = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  Image image1 = MakeImage(1, true_pose1);
  Image image2 = MakeImage(2, true_pose2);
  Image image3 = MakeImage(3, true_pose3);
  AddImageObservations(camera, true_pose1, &image1, true_points);
  AddImageObservations(camera, true_pose2, &image2, true_points);
  AddImageObservations(camera, true_pose3, &image3, true_points);
  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);

  // 制造一个有误差的初始值：固定 image1 作为 gauge，扰动 image2/image3 和点。
  reconstruction.GetImage(2).cam_from_world->t_cw +=
      Eigen::Vector3d(0.05, -0.03, 0.02);
  reconstruction.GetImage(3).cam_from_world->t_cw +=
      Eigen::Vector3d(-0.04, 0.03, -0.02);

  for (std::size_t i = 0; i < true_points.size(); ++i) {
    Track track;
    track.AddElement(1, static_cast<Point2DIdx>(i));
    track.AddElement(2, static_cast<Point2DIdx>(i));
    track.AddElement(3, static_cast<Point2DIdx>(i));
    const Eigen::Vector3d noisy_point =
        true_points[i] + Eigen::Vector3d(0.03, -0.02, 0.04);
    reconstruction.AddPoint3D(noisy_point, track);
  }

  const double error_before = MeanReprojectionError(reconstruction);

  LocalBundleAdjustmentOptions options;
  options.max_num_iterations = 50;
  options.loss_scale = 1.0;
  options.constant_image_ids.insert(1);

  const LocalBundleAdjustmentSummary summary =
      AdjustLocalBundle(3, options, &reconstruction);

  const double error_after = MeanReprojectionError(reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_GT(summary.num_residuals, 0U);
  EXPECT_GT(summary.num_variable_points, 0U);
  EXPECT_GT(summary.num_variable_images, 0U);
  EXPECT_LT(summary.final_mean_reprojection_error,
            summary.initial_mean_reprojection_error);
  EXPECT_LT(error_after, error_before * 0.5);
}

TEST(LocalBundleAdjusterTest,
     SelectsLocalImagesBySharedPointsAndAddsFixedObservers) {
  const Camera camera = MakeCamera();
  const std::map<ImageId, Pose3> true_poses = {
      {1, CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0))},
      {2, CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0))},
      {3, CameraAt(Eigen::Vector3d(0.5, 0.2, 0.0))},
      {4, CameraAt(Eigen::Vector3d(-0.4, 0.1, 0.0))},
      {5, CameraAt(Eigen::Vector3d(2.0, 0.0, 0.0))},
  };

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (const auto& [image_id, pose] : true_poses) {
    reconstruction.AddImage(MakeImage(image_id, pose));
  }

  for (int i = 0; i < 10; ++i) {
    AddTrackedPoint(&reconstruction, camera, true_poses,
                    Eigen::Vector3d(-0.8 + 0.1 * i, -0.2, 4.5),
                    {1, 2});
  }
  for (int i = 0; i < 6; ++i) {
    AddTrackedPoint(&reconstruction, camera, true_poses,
                    Eigen::Vector3d(-0.4 + 0.1 * i, 0.1, 4.8),
                    {1, 3});
  }
  for (int i = 0; i < 2; ++i) {
    AddTrackedPoint(&reconstruction, camera, true_poses,
                    Eigen::Vector3d(0.2 + 0.1 * i, 0.3, 5.0),
                    {1, 4});
  }
  AddTrackedPoint(&reconstruction, camera, true_poses,
                  Eigen::Vector3d(0.5, -0.3, 5.2), {5});

  LocalBundleAdjustmentOptions options;
  options.max_num_images = 3;
  options.max_num_fixed_images = 2;
  options.max_num_iterations = 1;

  const LocalBundleAdjustmentSummary summary =
      AdjustLocalBundle(1, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.local_image_ids, (std::set<ImageId>{1, 2, 3}));
  EXPECT_EQ(summary.fixed_image_ids, (std::set<ImageId>{4}));
  EXPECT_EQ(summary.num_local_images, 3U);
  EXPECT_EQ(summary.num_constant_images, 1U);
  EXPECT_GT(summary.num_residuals, 2U * (10U + 6U));
}

}  // namespace
}  // namespace colite
