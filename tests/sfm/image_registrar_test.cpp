#include "colite/sfm/image_registrar.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/incremental_initializer.h"

namespace colite {
namespace {

Camera MakeCamera(CameraId id) {
  Camera camera;
  camera.id = id;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  return camera;
}

Image MakeImage(ImageId image_id, CameraId camera_id) {
  Image image;
  image.id = image_id;
  image.camera_id = camera_id;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
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
                          4.0 + 0.07 * x + 0.11 * y + 0.03 * x * y);
    }
  }
  return points;
}

FeatureSet ProjectFeatures(const Camera& camera,
                           const Pose3& cam_from_world,
                           const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const auto pixel = ProjectWorldPoint(camera, cam_from_world, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()), 1.0F,
         0.0F});
  }
  return features;
}

std::vector<FeatureMatch> IdentityMatches(std::size_t count) {
  std::vector<FeatureMatch> matches;
  for (std::size_t i = 0; i < count; ++i) {
    const Point2DIdx idx = static_cast<Point2DIdx>(i);
    matches.push_back({idx, idx});
  }
  return matches;
}

TwoViewGeometry EstimateInitialGeometry(const Camera& camera,
                                        const FeatureSet& features1,
                                        const FeatureSet& features2,
                                        const std::vector<FeatureMatch>& matches) {
  TwoViewGeometryOptions options;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;
  return EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                                 options);
}

TwoViewGeometry MakeVerifiedGeometryWithInliers(
    const std::vector<FeatureMatch>& matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = matches;
  geometry.median_triangulation_angle = 1.0;
  return geometry;
}

TEST(ImageRegistrarTest, RegistersImageFromExistingPointsUsingPnPRansac) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches12 = IdentityMatches(points.size());
  const std::vector<FeatureMatch> matches13 = IdentityMatches(points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches12);
  SetMatches(&graph, 1, 3, matches13);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateInitialGeometry(camera, features1, features2,
                                              matches12));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometryWithInliers(matches13));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));

  IncrementalInitializerOptions init_options;
  init_options.min_num_inliers = 20;
  init_options.min_num_triangulated_points = 20;
  init_options.min_triangulation_angle_deg = 0.1;
  init_options.max_reprojection_error_px = 1.0;
  const IncrementalInitializerSummary init_summary =
      InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                       init_options, &reconstruction);
  ASSERT_TRUE(init_summary.success);

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;

  // Enough absolute inliers can still be a poor explanation of a large set of
  // correspondences. Enforce the COLMAP ratio before publishing the pose.
  MatchGraph outlier_graph = graph;
  FeatureSet noisy_features = features3;
  noisy_features.keypoints.back().x += 10000.0F;
  noisy_features.keypoints.back().y += 10000.0F;
  SetFeatures(&outlier_graph, 3, noisy_features);
  Reconstruction ratio_reconstruction = reconstruction;
  ImageRegistrarOptions ratio_options = register_options;
  ratio_options.min_num_inliers = 12;
  ratio_options.min_inlier_ratio = 1.0;
  const auto rejected = RegisterImageWithExistingPoints(
      outlier_graph, 3, ratio_options, &ratio_reconstruction);
  EXPECT_GE(rejected.num_inliers, 12U);
  EXPECT_LT(rejected.num_inliers, rejected.num_correspondences);
  EXPECT_FALSE(rejected.success);
  EXPECT_FALSE(ratio_reconstruction.IsImageRegistered(3));
  EXPECT_FALSE(ratio_reconstruction.GetImage(3).HasPose());

  const ImageRegistrationSummary summary =
      RegisterImageWithExistingPoints(graph, 3, register_options,
                                      &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.image_id, 3U);
  EXPECT_GE(summary.num_correspondences, 20U);
  EXPECT_GE(summary.num_ransac_inliers, 20U);
  EXPECT_GE(summary.num_inliers, 20U);
  EXPECT_GE(summary.num_inliers, summary.num_ransac_inliers);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.GetImage(3).HasPose());
  EXPECT_GE(reconstruction.GetImage(3).NumPoints3D(), 20U);
}

TEST(ImageRegistrarTest, FailsWhenThereAreTooFewCorrespondences) {
  MatchGraph graph;
  Reconstruction reconstruction;

  ImageRegistrationSummary summary =
      RegisterImageWithExistingPoints(graph, 7, {}, &reconstruction);

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.image_id, 7U);
  EXPECT_EQ(summary.num_correspondences, 0U);
}

}  // namespace
}  // namespace colite
