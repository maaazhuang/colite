#include "colite/matching/weak_pair_rematcher.h"

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

void AppendProjectedPoint(const Camera& camera,
                          const Pose3& cam1_from_world,
                          const Pose3& cam2_from_world,
                          const Eigen::Vector3d& point_world,
                          FeatureSet* features1,
                          FeatureSet* features2) {
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());
  features1->keypoints.push_back(
      {static_cast<float>(pixel1->x()), static_cast<float>(pixel1->y()), 1.0F,
       0.0F});
  features2->keypoints.push_back(
      {static_cast<float>(pixel2->x()), static_cast<float>(pixel2->y()), 1.0F,
       0.0F});
}

void FillDescriptorsForRelaxedRecovery(FeatureSet* features1,
                                       FeatureSet* features2) {
  features1->descriptors.resize(10, 1);
  features2->descriptors.resize(12, 1);
  for (int i = 0; i < 10; ++i) {
    features1->descriptors(i, 0) = static_cast<float>(10 * i);
    features2->descriptors(i, 0) = static_cast<float>(10 * i + 1);
  }

  // 第 8/9 个点的第二近邻较近：ratio=0.8 会拒绝，ratio=0.95 会接受。
  features2->descriptors(10, 0) = 81.1F;
  features2->descriptors(11, 0) = 91.1F;
  features2->keypoints.push_back(features2->keypoints[8]);
  features2->keypoints.push_back(features2->keypoints[9]);
}

TEST(WeakPairRematcherTest, ImprovesNearbyWeakPairWithGuidedCandidates) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  for (int i = 0; i < 10; ++i) {
    const Eigen::Vector3d point_world(-1.0 + 0.25 * i, -0.4 + 0.08 * i,
                                      4.0 + 0.13 * i + 0.01 * i * i);
    AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                         point_world, &features1, &features2);
  }
  FillDescriptorsForRelaxedRecovery(&features1, &features2);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);

  std::vector<FeatureMatch> initial_matches;
  for (Point2DIdx idx = 0; idx < 8; ++idx) {
    initial_matches.push_back({idx, idx});
  }
  SetMatches(&graph, 1, 2, initial_matches);

  TwoViewGeometryOptions geometry_options;
  geometry_options.min_num_inliers = 8;
  geometry_options.max_error_px = 1.0;
  const TwoViewGeometry initial_geometry = EstimateTwoViewGeometry(
      camera, features1, camera, features2, initial_matches, geometry_options);
  ASSERT_NE(initial_geometry.type, TwoViewGeometry::Type::kDegenerate);
  ASSERT_FALSE(initial_geometry.inlier_matches.empty());
  SetVerifiedGeometry(&graph, 1, 2, initial_geometry);

  WeakPairRematchingOptions options;
  options.nearby_image_window = 2;
  options.min_inlier_gain = 1;
  options.relaxed_matching.max_ratio = 0.95;
  options.relaxed_matching.mutual_check = true;
  options.guided_matching.max_sampson_error_px = 2.0;
  options.verification = geometry_options;

  const WeakPairRematchingSummary summary =
      RematchWeakImagePairs({{1, camera}, {2, camera}}, options, &graph);

  const std::vector<FeatureMatch>* rematched = FindMatches(graph, 1, 2);
  const TwoViewGeometry* geometry = FindVerifiedGeometry(graph, 1, 2);
  ASSERT_NE(rematched, nullptr);
  ASSERT_NE(geometry, nullptr);
  EXPECT_EQ(summary.num_checked_pairs, 1U);
  EXPECT_EQ(summary.num_improved_pairs, 1U);
  EXPECT_GE(rematched->size(), 10U);
  EXPECT_GE(geometry->inlier_matches.size(), 10U);
}

}  // namespace
}  // namespace colite
