#include "colite/sfm/global_bundle_adjuster.h"

#include <cmath>
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

TEST(GlobalBundleAdjusterTest, ReducesReprojectionErrorForWholeModel) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> true_points = MakeScenePoints();
  const std::vector<Pose3> true_poses = {
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0)),
      CameraAt(Eigen::Vector3d(0.75, -0.2, 0.0))};

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  for (std::size_t i = 0; i < true_poses.size(); ++i) {
    Image image =
        MakeRegisteredImage(static_cast<ImageId>(i + 1), true_poses[i]);
    AddImageObservations(camera, true_poses[i], &image, true_points);
    reconstruction.AddImage(image);
  }

  // 制造一个全局有误差的模型：固定 image1，扰动其余图像和所有 3D 点。
  reconstruction.GetImage(2).cam_from_world->t_cw +=
      Eigen::Vector3d(0.05, -0.03, 0.02);
  reconstruction.GetImage(3).cam_from_world->t_cw +=
      Eigen::Vector3d(-0.04, 0.03, -0.02);
  reconstruction.GetImage(4).cam_from_world->t_cw +=
      Eigen::Vector3d(0.03, 0.02, -0.03);

  for (std::size_t i = 0; i < true_points.size(); ++i) {
    Track track;
    for (ImageId image_id = 1; image_id <= 4; ++image_id) {
      track.AddElement(image_id, static_cast<Point2DIdx>(i));
    }
    const Eigen::Vector3d noisy_point =
        true_points[i] + Eigen::Vector3d(0.03, -0.02, 0.04);
    reconstruction.AddPoint3D(noisy_point, track);
  }

  const double error_before = MeanReprojectionError(reconstruction);

  GlobalBundleAdjustmentOptions options;
  options.max_num_iterations = 50;
  options.loss_scale = 1.0;
  options.constant_image_ids.insert(1);

  const GlobalBundleAdjustmentSummary summary =
      AdjustGlobalBundle(options, &reconstruction);

  const double error_after = MeanReprojectionError(reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_GT(summary.num_residuals, 0U);
  EXPECT_GT(summary.num_variable_images, 0U);
  EXPECT_GT(summary.num_variable_points, 0U);
  EXPECT_LT(summary.final_mean_reprojection_error,
            summary.initial_mean_reprojection_error);
  EXPECT_LT(error_after, error_before * 0.5);
}

TEST(GlobalBundleAdjusterTest, RefinesOnlyRequestedCameraParameterGroups) {
  Camera true_camera = MakeCamera();
  const std::vector<Eigen::Vector3d> true_points = MakeScenePoints();
  const std::vector<Pose3> true_poses = {
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0))};

  Reconstruction reconstruction;
  Camera estimated_camera = true_camera;
  estimated_camera.params[0] = 630.0;
  estimated_camera.params[1] = 630.0;
  reconstruction.AddCamera(estimated_camera);

  for (std::size_t i = 0; i < true_poses.size(); ++i) {
    Image image =
        MakeRegisteredImage(static_cast<ImageId>(i + 1), true_poses[i]);
    AddImageObservations(true_camera, true_poses[i], &image, true_points);
    reconstruction.AddImage(image);
  }
  for (std::size_t i = 0; i < true_points.size(); ++i) {
    Track track;
    for (ImageId image_id = 1; image_id <= 3; ++image_id) {
      track.AddElement(image_id, static_cast<Point2DIdx>(i));
    }
    reconstruction.AddPoint3D(true_points[i], track);
  }

  const double initial_fx_error =
      std::abs(reconstruction.GetCamera(1).params[0] - true_camera.params[0]);
  const double fixed_cx = reconstruction.GetCamera(1).params[2];
  const double fixed_cy = reconstruction.GetCamera(1).params[3];

  GlobalBundleAdjustmentOptions options;
  options.max_num_iterations = 80;
  options.loss_scale = 0.0;
  options.refine_poses = false;
  options.refine_points = false;
  options.refine_camera_params = CameraParameterGroup::kFocalLength;

  const GlobalBundleAdjustmentSummary summary =
      AdjustGlobalBundle(options, &reconstruction);

  const Camera& refined_camera = reconstruction.GetCamera(1);
  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.num_variable_cameras, 1U);
  EXPECT_LT(std::abs(refined_camera.params[0] - true_camera.params[0]),
            initial_fx_error * 0.25);
  EXPECT_LT(std::abs(refined_camera.params[1] - true_camera.params[1]),
            initial_fx_error * 0.25);
  EXPECT_DOUBLE_EQ(refined_camera.params[2], fixed_cx);
  EXPECT_DOUBLE_EQ(refined_camera.params[3], fixed_cy);
}

TEST(GlobalBundleAdjusterTest, AppliesTwoImageGaugeWhenNoConstantsAreGiven) {
  const Camera camera = MakeCamera();
  const std::vector<Eigen::Vector3d> true_points = MakeScenePoints();
  const std::vector<Pose3> true_poses = {
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0))};

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  for (std::size_t i = 0; i < true_poses.size(); ++i) {
    Image image =
        MakeRegisteredImage(static_cast<ImageId>(i + 1), true_poses[i]);
    AddImageObservations(camera, true_poses[i], &image, true_points);
    reconstruction.AddImage(image);
  }
  for (std::size_t i = 0; i < true_points.size(); ++i) {
    Track track;
    for (ImageId image_id = 1; image_id <= 3; ++image_id) {
      track.AddElement(image_id, static_cast<Point2DIdx>(i));
    }
    reconstruction.AddPoint3D(true_points[i], track);
  }

  GlobalBundleAdjustmentOptions options;
  options.max_num_iterations = 1;

  const GlobalBundleAdjustmentSummary summary =
      AdjustGlobalBundle(options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.gauge_constant_pose_image_ids, (std::set<ImageId>{1}));
  ASSERT_EQ(summary.gauge_constant_translation_idxs.size(), 1U);
  EXPECT_EQ(summary.gauge_constant_translation_idxs.at(2),
            (std::vector<int>{0}));
  EXPECT_EQ(summary.num_gauge_constant_translation_images, 1U);
}

TEST(GlobalBundleAdjusterTest, RefinesNamedCameraParameterMask) {
  Camera true_camera = MakeCamera();
  true_camera.model_id = CameraModelId::kSimpleRadial;
  true_camera.params = {700.0, 500.0, 400.0, 0.04};

  const std::vector<Eigen::Vector3d> true_points = MakeScenePoints();
  const std::vector<Pose3> true_poses = {
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0))};

  Reconstruction reconstruction;
  Camera estimated_camera = true_camera;
  estimated_camera.params = {700.0, 500.0, 400.0, 0.0};
  reconstruction.AddCamera(estimated_camera);

  for (std::size_t i = 0; i < true_poses.size(); ++i) {
    Image image =
        MakeRegisteredImage(static_cast<ImageId>(i + 1), true_poses[i]);
    AddImageObservations(true_camera, true_poses[i], &image, true_points);
    reconstruction.AddImage(image);
  }
  for (std::size_t i = 0; i < true_points.size(); ++i) {
    Track track;
    for (ImageId image_id = 1; image_id <= 3; ++image_id) {
      track.AddElement(image_id, static_cast<Point2DIdx>(i));
    }
    reconstruction.AddPoint3D(true_points[i], track);
  }

  GlobalBundleAdjustmentOptions options;
  options.max_num_iterations = 80;
  options.loss_scale = 0.0;
  options.refine_poses = false;
  options.refine_points = false;
  options.refine_camera_param_names = {"k1"};

  const GlobalBundleAdjustmentSummary summary =
      AdjustGlobalBundle(options, &reconstruction);

  const Camera& refined_camera = reconstruction.GetCamera(1);
  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.num_variable_cameras, 1U);
  EXPECT_NEAR(refined_camera.params[0], 700.0, 1e-12);
  EXPECT_NEAR(refined_camera.params[1], 500.0, 1e-12);
  EXPECT_NEAR(refined_camera.params[2], 400.0, 1e-12);
  EXPECT_LT(std::abs(refined_camera.params[3] - true_camera.params[3]), 0.01);
}

TEST(GlobalBundleAdjusterTest, SupportsEveryRegisteredCameraModel) {
  const std::vector<CameraModelId> models = {
      CameraModelId::kSimplePinhole,
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
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const std::vector<Pose3> poses = {
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0)),
      CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0))};

  for (const CameraModelId model : models) {
    Camera true_camera;
    true_camera.id = 1;
    true_camera.model_id = model;
    true_camera.width = 1000;
    true_camera.height = 800;
    true_camera.params = InitializeCameraParams(model, 700.0, 1000, 800);
    for (const std::size_t idx :
         GetCameraModelSpec(model).distortion_idxs) {
      true_camera.params[idx] =
          model == CameraModelId::kFOV ? 0.3 : 0.0005;
    }

    Reconstruction reconstruction;
    Camera estimated_camera = true_camera;
    for (const std::size_t idx :
         GetCameraModelSpec(model).focal_length_idxs) {
      estimated_camera.params[idx] *= 0.95;
    }
    reconstruction.AddCamera(estimated_camera);
    for (std::size_t idx = 0; idx < poses.size(); ++idx) {
      Image image =
          MakeRegisteredImage(static_cast<ImageId>(idx + 1), poses[idx]);
      AddImageObservations(true_camera, poses[idx], &image, points);
      reconstruction.AddImage(image);
    }
    for (std::size_t idx = 0; idx < points.size(); ++idx) {
      Track track;
      for (ImageId image_id = 1; image_id <= poses.size(); ++image_id) {
        track.AddElement(image_id, static_cast<Point2DIdx>(idx));
      }
      reconstruction.AddPoint3D(points[idx], track);
    }
    const double before = MeanReprojectionError(reconstruction);

    GlobalBundleAdjustmentOptions options;
    options.max_num_iterations = 50;
    options.loss_scale = 0.0;
    options.refine_poses = false;
    options.refine_points = false;
    options.refine_camera_params = CameraParameterGroup::kFocalLength;
    const GlobalBundleAdjustmentSummary summary =
        AdjustGlobalBundle(options, &reconstruction);

    EXPECT_TRUE(summary.success) << CameraModelIdToName(model);
    EXPECT_LT(MeanReprojectionError(reconstruction), before * 0.1)
        << CameraModelIdToName(model);
  }
}

}  // namespace
}  // namespace colite
