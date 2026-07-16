#include "colite/sfm/incremental_mapper.h"

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
    for (int x = 0; x < 8; ++x) {
      points.emplace_back(-1.4 + 0.4 * x, -0.6 + 0.3 * y,
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

FeatureSet PerturbFeatures(FeatureSet features, double amplitude_px) {
  for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
    const double sign = (i % 2 == 0) ? 1.0 : -1.0;
    features.keypoints[i].x += static_cast<float>(sign * amplitude_px);
    features.keypoints[i].y += static_cast<float>(-sign * amplitude_px * 0.5);
  }
  return features;
}

std::vector<FeatureMatch> IdentityMatches(std::size_t begin, std::size_t end) {
  std::vector<FeatureMatch> matches;
  for (std::size_t i = begin; i < end; ++i) {
    const Point2DIdx idx = static_cast<Point2DIdx>(i);
    matches.push_back({idx, idx});
  }
  return matches;
}

TwoViewGeometry EstimateGeometry(const Camera& camera,
                                 const FeatureSet& features1,
                                 const FeatureSet& features2,
                                 const std::vector<FeatureMatch>& matches) {
  TwoViewGeometryOptions options;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;
  return EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                                 options);
}

TwoViewGeometry MakeVerifiedGeometry(const std::vector<FeatureMatch>& matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = matches;
  geometry.median_triangulation_angle = 2.0;
  return geometry;
}

TEST(IncrementalMapperTest,
     RegistersImageTriangulatesAndRunsLocalBundleAdjustment) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));

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
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  // 模拟增量建图中常见的累积误差：已有位姿和点云并非完全干净。
  reconstruction.GetImage(2).cam_from_world->t_cw +=
      Eigen::Vector3d(0.04, -0.02, 0.02);
  std::vector<Point3DId> point3d_ids;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d;
    point3d_ids.push_back(point3d_id);
  }
  for (const Point3DId point3d_id : point3d_ids) {
    reconstruction.GetPoint3D(point3d_id).xyz +=
        Eigen::Vector3d(0.02, -0.015, 0.03);
  }

  const std::size_t num_points_before = reconstruction.Points3D().size();

  IncrementalMapperOptions mapper_options;
  mapper_options.registration.min_num_correspondences = 20;
  mapper_options.registration.min_num_inliers = 20;
  mapper_options.registration.max_reprojection_error_px = 3.0;
  mapper_options.triangulation.min_triangulation_angle_deg = 0.1;
  mapper_options.triangulation.create_max_angle_error_deg = 2.0;
  mapper_options.local_bundle_adjustment.max_num_iterations = 50;
  mapper_options.local_bundle_adjustment.loss_scale = 1.0;
  mapper_options.local_bundle_adjustment.constant_image_ids.insert(1);

  const IncrementalMapperSummary summary =
      RegisterNextImage(graph, 3, mapper_options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_TRUE(summary.registration.success);
  EXPECT_GT(summary.triangulation.num_created_points, 0U);
  EXPECT_TRUE(summary.local_bundle_adjustment.success);
  EXPECT_GT(summary.local_bundle_adjustment.num_residuals, 0U);
  EXPECT_TRUE(summary.observation_filter.success);
  EXPECT_LT(summary.local_bundle_adjustment.final_mean_reprojection_error,
            summary.local_bundle_adjustment.initial_mean_reprojection_error);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_GT(reconstruction.Points3D().size(), num_points_before);
}

TEST(IncrementalMapperTest, RunsCompleteIncrementalMappingLoop) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(0.75, -0.2, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const FeatureSet features4 = ProjectFeatures(camera, cam4_from_world, points);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetMatches(&graph, 1, 4, matches_all);
  SetMatches(&graph, 2, 4, matches_all);
  SetMatches(&graph, 3, 4, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3,
                      EstimateGeometry(camera, features1, features3,
                                       matches_all));
  SetVerifiedGeometry(&graph, 2, 3,
                      EstimateGeometry(camera, features2, features3,
                                       matches_all));
  SetVerifiedGeometry(&graph, 1, 4,
                      EstimateGeometry(camera, features1, features4,
                                       matches_all));
  SetVerifiedGeometry(&graph, 2, 4,
                      EstimateGeometry(camera, features2, features4,
                                       matches_all));
  SetVerifiedGeometry(&graph, 3, 4,
                      EstimateGeometry(camera, features3, features4,
                                       matches_all));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalMappingOptions options;
  options.initialization.min_num_inliers = 20;
  options.initialization.min_num_triangulated_points = 20;
  options.initialization.min_triangulation_angle_deg = 0.1;
  options.initialization.max_reprojection_error_px = 1.0;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 3.0;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.triangulation.create_max_angle_error_deg = 2.0;
  options.mapper.local_bundle_adjustment.max_num_iterations = 50;
  options.mapper.local_bundle_adjustment.loss_scale = 1.0;
  options.max_num_registrations = 4;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_TRUE(summary.initialization.success);
  EXPECT_TRUE(summary.global_bundle_adjustment.success);
  EXPECT_GT(summary.global_bundle_adjustment.num_residuals, 0U);
  EXPECT_TRUE(summary.global_observation_filter.success);
  EXPECT_TRUE(summary.final_global_bundle_adjustment.success);
  EXPECT_GT(summary.final_global_bundle_adjustment.num_residuals, 0U);
  EXPECT_EQ(summary.registrations.size(), 2U);
  EXPECT_EQ(summary.registrations[0].registration_order, 1U);
  EXPECT_EQ(summary.registrations[1].registration_order, 2U);
  EXPECT_GE(summary.registrations[0].candidate_num_visible_points,
            options.mapper.registration.min_num_correspondences);
  EXPECT_GE(summary.registrations[0].candidate_num_registered_neighbors, 2U);
  EXPECT_GE(
      summary.registrations[0].candidate_num_verified_inliers_to_registered_images,
      options.mapper.registration.min_num_inliers);
  EXPECT_GT(summary.registrations[0].candidate_num_visible_observations, 0U);
  EXPECT_GT(summary.registrations[0].candidate_mean_track_length, 0.0);
  EXPECT_GT(summary.registrations[0].candidate_score, 0.0);
  EXPECT_EQ(summary.num_registered_images, 4U);
  EXPECT_EQ(reconstruction.RegisteredImageIds().size(), 4U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  EXPECT_TRUE(reconstruction.IsImageRegistered(2));
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
  EXPECT_GT(summary.num_points3d, options.initialization.min_num_triangulated_points);
  for (const IncrementalMapperSummary& registration : summary.registrations) {
    EXPECT_TRUE(registration.success);
    EXPECT_TRUE(registration.local_bundle_adjustment.success);
    EXPECT_TRUE(registration.observation_filter.success);
  }
}

TEST(IncrementalMapperTest, DemotesFailedCandidatesAndStopsAfterTrialLimit) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(0.5, 0.1, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features4 = ProjectFeatures(camera, cam4_from_world, points);
  FeatureSet bad_features3;
  for (std::size_t i = 0; i < points.size(); ++i) {
    bad_features3.keypoints.push_back({12.0F, 12.0F, 1.0F, 0.0F});
  }

  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, bad_features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetMatches(&graph, 1, 4, matches_all);
  SetMatches(&graph, 2, 4, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 1, 4, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 4, MakeVerifiedGeometry(matches_all));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalInitializerOptions init_options;
  init_options.min_num_inliers = 20;
  init_options.min_num_triangulated_points = 20;
  init_options.min_triangulation_angle_deg = 0.1;
  init_options.max_reprojection_error_px = 1.0;
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  IncrementalMappingOptions options;
  options.max_num_registrations = 4;
  options.max_num_registration_trials_per_image = 1;
  options.run_global_track_completion = false;
  options.run_global_bundle_adjustment = false;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 0.25;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.triangulation.create_max_angle_error_deg = 2.0;
  options.mapper.run_local_bundle_adjustment = false;
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.run_observation_filter = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  ASSERT_EQ(summary.failed_registrations.size(), 1U);
  EXPECT_EQ(summary.failed_registrations[0].image_id, 3U);
  EXPECT_EQ(summary.failed_registrations[0].attempt, 1U);
  EXPECT_GT(summary.failed_registrations[0].candidate_score, 0.0);
  ASSERT_EQ(summary.registrations.size(), 1U);
  EXPECT_EQ(summary.registrations[0].image_id, 4U);
  EXPECT_EQ(summary.registrations[0].registration_attempts_before_success, 0U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
  EXPECT_FALSE(reconstruction.IsImageRegistered(3));
}

TEST(IncrementalMapperTest, RelaxesRegistrationThresholdsAcrossRetries) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 =
      PerturbFeatures(ProjectFeatures(camera, cam3_from_world, points), 0.8);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));

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
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  IncrementalMappingOptions options;
  options.max_num_registrations = 2;
  options.max_num_registration_trials_per_image = 2;
  options.registration_retry_reprojection_error_multiplier = 80.0;
  options.registration_retry_max_reprojection_error_px = 4.0;
  options.run_global_track_completion = false;
  options.run_global_bundle_adjustment = false;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 0.05;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.run_local_bundle_adjustment = false;
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.run_observation_filter = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  ASSERT_EQ(summary.failed_registrations.size(), 1U);
  EXPECT_EQ(summary.failed_registrations[0].image_id, 3U);
  EXPECT_EQ(summary.failed_registrations[0].attempt, 1U);
  EXPECT_DOUBLE_EQ(summary.failed_registrations[0].registration_max_reprojection_error_px,
                   0.05);
  ASSERT_EQ(summary.registrations.size(), 1U);
  EXPECT_EQ(summary.registrations[0].image_id, 3U);
  EXPECT_EQ(summary.registrations[0].registration_attempts_before_success, 1U);
  EXPECT_DOUBLE_EQ(summary.registrations[0].registration_max_reprojection_error_px,
                   4.0);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
}

TEST(IncrementalMapperTest, SelectsCandidateWithStrongerRegisteredConnectivity) {
  const Camera camera = MakeCamera(1);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (const ImageId image_id : {1U, 2U, 3U, 4U}) {
    Image image = MakeImage(image_id, 1);
    if (image_id == 1U || image_id == 4U) {
      image.registered = true;
      image.cam_from_world = CameraAt(Eigen::Vector3d::Zero());
      for (int i = 0; i < 10; ++i) {
        image.points2d.push_back(
            {{100.0 + i, 120.0 + i}, 0, kInvalidPoint3DId});
      }
    } else {
      for (int i = 0; i < 10; ++i) {
        image.points2d.push_back(
            {{200.0 + i, 220.0 + i}, 0, kInvalidPoint3DId});
      }
    }
    reconstruction.AddImage(image);
  }

  for (int i = 0; i < 10; ++i) {
    Track track1;
    track1.AddElement(1, static_cast<Point2DIdx>(i));
    reconstruction.AddPoint3D(Eigen::Vector3d(i, 0.0, 5.0), track1);

    Track track4;
    track4.AddElement(4, static_cast<Point2DIdx>(i));
    reconstruction.AddPoint3D(Eigen::Vector3d(i, 1.0, 5.0), track4);
  }

  MatchGraph graph;
  FeatureSet dummy_features;
  for (int i = 0; i < 10; ++i) {
    dummy_features.keypoints.push_back(
        {static_cast<float>(i), static_cast<float>(i), 1.0F, 0.0F});
  }
  SetFeatures(&graph, 2, dummy_features);
  SetFeatures(&graph, 3, dummy_features);

  const std::vector<FeatureMatch> all_matches = IdentityMatches(0, 10);
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(all_matches));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(IdentityMatches(0, 5)));
  SetVerifiedGeometry(&graph, 4, 3, MakeVerifiedGeometry(IdentityMatches(5, 10)));

  IncrementalMapperOptions options;
  options.registration.min_num_correspondences = 8;

  const std::optional<ImageId> next_image =
      SelectNextImage(graph, options, reconstruction);

  ASSERT_TRUE(next_image.has_value());
  EXPECT_EQ(*next_image, 3U);
}

TEST(IncrementalMapperTest, RetriangulatesRegisteredNeighborhoodPairs) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.5, 0.1, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, 10);
  const std::vector<FeatureMatch> seed_matches = IdentityMatches(0, 8);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(seed_matches));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(seed_matches));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  Image image3 = MakeImage(3, 1);
  image3.registered = true;
  image3.cam_from_world = cam3_from_world;
  for (std::size_t i = 0; i < points.size(); ++i) {
    image1.points2d.push_back({{features1.keypoints[i].x, features1.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    image2.points2d.push_back({{features2.keypoints[i].x, features2.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    image3.points2d.push_back({{features3.keypoints[i].x, features3.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
  }
  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);

  IncrementalTriangulatorOptions options;
  options.ignore_two_view_tracks = false;
  options.min_triangulation_angle_deg = 0.1;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 3, options, nullptr,
                                     &reconstruction);

  EXPECT_GE(summary.num_created_points, 10U);
  EXPECT_EQ(reconstruction.Points3D().size(), 10U);
}

TEST(IncrementalMapperTest, RetriangulationSkipsLowAnglePairLayer) {
  const Camera camera = MakeCamera(1);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  MatchGraph graph;
  TwoViewGeometry geometry = MakeVerifiedGeometry({{0, 0}});
  geometry.median_triangulation_angle = 0.2;
  SetVerifiedGeometry(&graph, 1, 2, geometry);

  IncrementalTriangulatorOptions options;
  options.min_triangulation_angle_deg = 0.0;
  options.min_pair_median_triangulation_angle_deg = 1.0;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 1, options, nullptr,
                                     &reconstruction);

  EXPECT_EQ(summary.num_created_points, 0U);
  EXPECT_TRUE(reconstruction.Points3D().empty());
}

TEST(IncrementalMapperTest, RetriangulationFiltersNewLowAnglePoints) {
  const Camera camera = MakeCamera(1);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(0.01, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.0, 0.0, 10.0);
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions triangulation_options;
  triangulation_options.ignore_two_view_tracks = false;
  triangulation_options.min_triangulation_angle_deg = 0.0;
  triangulation_options.create_max_angle_error_deg = 1.0;

  ObservationFilterOptions filter_options;
  filter_options.max_reprojection_error_px = 1.0;
  filter_options.min_triangulation_angle_deg = 1.0;
  filter_options.min_track_length = 2;

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 1, triangulation_options,
                                     &filter_options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_filtered_points, 1U);
  EXPECT_TRUE(reconstruction.Points3D().empty());
}

TEST(IncrementalMapperTest, CompletesTracksAcrossAllRegisteredImagesAtEnd) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);

  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  Track track1;
  track1.AddElement(1, 0);
  reconstruction.AddPoint3D(point_world, track1);

  Track track2;
  track2.AddElement(2, 0);
  reconstruction.AddPoint3D(point_world, track2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalMappingOptions options;
  options.max_num_registrations = 0;
  options.run_global_bundle_adjustment = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  const Point3DId merged_id = reconstruction.GetImage(1).points2d[0].point3d_id;
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, merged_id);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(merged_id));
  EXPECT_EQ(reconstruction.GetPoint3D(merged_id).track.Length(), 2U);
}

}  // namespace
}  // namespace colite
