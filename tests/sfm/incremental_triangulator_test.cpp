#include "colite/sfm/incremental_triangulator.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/image_registrar.h"
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

TEST(IncrementalTriangulatorTest,
     TriangulatesNewPointsAfterImageRegistration) {
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
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_existing));
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

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;
  ASSERT_TRUE(RegisterImageWithExistingPoints(graph, 3, register_options,
                                             &reconstruction)
                  .success);

  const std::size_t num_points_before = reconstruction.Points3D().size();

  IncrementalTriangulatorOptions tri_options;
  tri_options.min_triangulation_angle_deg = 0.1;
  tri_options.create_max_reprojection_error_px = 1.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 3, tri_options, &reconstruction);

  EXPECT_EQ(summary.image_id, 3U);
  EXPECT_GT(summary.num_created_points, 0U);
  EXPECT_EQ(summary.num_created_observations, summary.num_created_points * 2);
  EXPECT_GT(reconstruction.Points3D().size(), num_points_before);
  EXPECT_GT(reconstruction.GetImage(3).NumPoints3D(),
            register_options.min_num_inliers);
}

TEST(IncrementalTriangulatorTest,
     CompletesTracksForExistingPointsAfterCreatingNewPoints) {
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
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_existing));
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

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;
  ASSERT_TRUE(RegisterImageWithExistingPoints(graph, 3, register_options,
                                             &reconstruction)
                  .success);

  IncrementalTriangulatorOptions tri_options;
  tri_options.min_triangulation_angle_deg = 0.1;
  tri_options.create_max_reprojection_error_px = 1.0;
  ASSERT_GT(TriangulateRegisteredImage(graph, 3, tri_options, &reconstruction)
                .num_created_points,
            0U);
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));

  const std::size_t image1_points_before =
      reconstruction.GetImage(1).NumPoints3D();

  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, tri_options, &reconstruction);

  EXPECT_EQ(summary.image_id, 1U);
  EXPECT_GT(summary.num_completed_observations, 0U);
  EXPECT_GT(reconstruction.GetImage(1).NumPoints3D(), image1_points_before);

  std::size_t num_tracks_with_three_observations = 0;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    if (point3d.track.Length() >= 3) {
      ++num_tracks_with_three_observations;
    }
  }
  EXPECT_GT(num_tracks_with_three_observations, 0U);
}

TEST(IncrementalTriangulatorTest, MergesMatchedExistingPointsIntoLongerTrack) {
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
  const Point3DId point3d_id1 =
      reconstruction.AddPoint3D(point_world, track1);

  Track track2;
  track2.AddElement(2, 0);
  const Point3DId point3d_id2 =
      reconstruction.AddPoint3D(point_world, track2);
  ASSERT_NE(point3d_id1, point3d_id2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.create_max_reprojection_error_px = 1.0;

  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.image_id, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  const Point3DId merged_id = reconstruction.GetImage(1).points2d[0].point3d_id;
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, merged_id);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(merged_id));
  EXPECT_EQ(reconstruction.GetPoint3D(merged_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest,
     CompletesObservationAfterEarlierNeighborWithoutPointFails) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(2.0, 0.0, 0.0));
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  const auto pixel3 = ProjectWorldPoint(camera, cam3_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());
  ASSERT_TRUE(pixel3.has_value());

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

  Image image3 = MakeImage(3, 1);
  image3.registered = true;
  image3.cam_from_world = cam3_from_world;
  image3.points2d.push_back({*pixel3, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image3);

  Track track3;
  track3.AddElement(3, 0);
  const Point3DId point3d_id = reconstruction.AddPoint3D(point_world, track3);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.create_max_reprojection_error_px = 1.0;

  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest, SkipsUnregisteredImage) {
  MatchGraph graph;
  Reconstruction reconstruction;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 9, {}, &reconstruction);

  EXPECT_EQ(summary.image_id, 9U);
  EXPECT_EQ(summary.num_created_points, 0U);
  EXPECT_EQ(summary.num_created_observations, 0U);
}

}  // namespace
}  // namespace colite
