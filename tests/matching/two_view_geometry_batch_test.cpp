#include "colite/matching/two_view_geometry_batch.h"

#include <map>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"

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

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

void AddProjectedMatch(const Camera& camera,
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

TEST(TwoViewGeometryBatchTest, VerifiesAllCandidatePairsIntoMatchGraph) {
  const Camera camera1 = MakeCamera(1);
  const Camera camera2 = MakeCamera(2);
  const Camera camera3 = MakeCamera(3);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> valid_matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      AddProjectedMatch(camera1, cam1_from_world, cam2_from_world,
                        Eigen::Vector3d(-1.0 + 0.4 * x,
                                        -0.6 + 0.3 * y,
                                        4.0 + 0.07 * x + 0.11 * y +
                                            0.03 * x * y),
                        &features1, &features2, &valid_matches);
    }
  }

  FeatureSet features3;
  features3.keypoints.push_back({500.0F, 400.0F, 1.0F, 0.0F});

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, valid_matches);
  SetMatches(&graph, 1, 3, {{0, 0}});

  std::map<ImageId, Camera> cameras_by_image = {
      {1, camera1},
      {2, camera2},
      {3, camera3},
  };

  TwoViewGeometryBatchOptions options;
  options.geometry_options.max_error_px = 1.0;
  options.geometry_options.min_num_inliers = 20;

  const TwoViewGeometryBatchSummary summary =
      VerifyMatchGraphGeometries(cameras_by_image, &graph, options);

  EXPECT_EQ(summary.num_input_pairs, 2U);
  EXPECT_EQ(summary.num_verified_pairs, 1U);
  EXPECT_EQ(summary.num_degenerate_pairs, 1U);
  EXPECT_NE(FindVerifiedGeometry(graph, 1, 2), nullptr);
  EXPECT_EQ(FindVerifiedGeometry(graph, 1, 3), nullptr);

  const std::vector<MatchGraphEdge> neighbors =
      FindVerifiedNeighbors(graph, 1);
  ASSERT_EQ(neighbors.size(), 1U);
  EXPECT_EQ(neighbors[0].OtherImageId(1), 2U);
  EXPECT_GE(neighbors[0].NumInliers(), 20U);
}

}  // namespace
}  // namespace colite
