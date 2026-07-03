#include "colite/matching/two_view_geometry_verifier.h"

#include <cmath>
#include <stdexcept>
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

Pose3 CameraAt(const Eigen::Vector3d& center_world,
               const Eigen::Matrix3d& cam_from_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond(cam_from_world).normalized();
  pose.t_cw = -cam_from_world * center_world;
  return pose;
}

void AppendProjectedPoint(const Camera& camera,
                          const Pose3& cam1_from_world,
                          const Pose3& cam2_from_world,
                          const Eigen::Vector3d& point_world,
                          FeatureSet* features1,
                          FeatureSet* features2,
                          std::vector<FeatureMatch>* matches) {
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  const Point2DIdx idx =
      static_cast<Point2DIdx>(features1->keypoints.size());
  features1->keypoints.push_back(
      {static_cast<float>(pixel1->x()), static_cast<float>(pixel1->y()), 1.0F,
       0.0F});
  features2->keypoints.push_back(
      {static_cast<float>(pixel2->x()), static_cast<float>(pixel2->y()), 1.0F,
       0.0F});
  matches->push_back({idx, idx});
}

TEST(TwoViewGeometryVerifierTest, RecoversCalibratedGeometryFromInlierMatches) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(
          -1.0 + 0.4 * x, -0.6 + 0.3 * y,
          4.0 + 0.07 * x + 0.11 * y + 0.03 * x * y);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.E.norm(), 0.0);
  EXPECT_GT(geometry.F.norm(), 0.0);
  EXPECT_GT(geometry.median_triangulation_angle, 0.0);
}

TEST(TwoViewGeometryVerifierTest, ClassifiesPlanarSceneAsHomography) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(0.8, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(-1.2 + 0.45 * x, -0.7 + 0.35 * y,
                                        4.5);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPlanarHomography);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.H.norm(), 0.0);
  EXPECT_GT(geometry.median_triangulation_angle, 0.0);
}

TEST(TwoViewGeometryVerifierTest, ClassifiesPureRotationAsPanoramic) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const double angle_rad = 8.0 * M_PI / 180.0;
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Pose3 cam2_from_world =
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0), rotation);

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(-1.0 + 0.4 * x, -0.6 + 0.3 * y,
                                        5.0 + 0.2 * x + 0.15 * y);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPanoramic);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.H.norm(), 0.0);
  EXPECT_EQ(geometry.median_triangulation_angle, 0.0);
}

TEST(TwoViewGeometryVerifierTest, ReturnsDegenerateWhenMatchesAreInsufficient) {
  const Camera camera = MakeCamera();
  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;

  features1.keypoints.push_back({500.0F, 400.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({490.0F, 400.0F, 1.0F, 0.0F});
  matches.push_back({0, 0});

  TwoViewGeometryOptions options;
  options.min_num_inliers = 8;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kDegenerate);
  EXPECT_TRUE(geometry.inlier_matches.empty());
}

TEST(TwoViewGeometryVerifierTest, ThrowsWhenMatchIndexIsInvalid) {
  const Camera camera = MakeCamera();
  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;

  features1.keypoints.push_back({500.0F, 400.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({490.0F, 400.0F, 1.0F, 0.0F});
  matches.push_back({0, 7});

  EXPECT_THROW(EstimateTwoViewGeometry(camera, features1, camera, features2,
                                       matches),
               std::out_of_range);
}

}  // namespace
}  // namespace colite
