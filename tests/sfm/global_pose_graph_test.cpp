#include "colite/sfm/global_positioner.h"
#include "colite/sfm/rotation_averager.h"
#include "colite/sfm/view_graph.h"

#include <map>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

RelativePoseConstraint MakeConstraint(
    const ImageId image_id1, const ImageId image_id2,
    const Eigen::Quaterniond& rotation,
    const Eigen::Vector3d& translation, const double weight = 1.0) {
  RelativePoseConstraint edge;
  edge.image_id1 = image_id1;
  edge.image_id2 = image_id2;
  edge.cam2_from_cam1.q_cw = rotation;
  edge.cam2_from_cam1.t_cw = translation;
  edge.weight = weight;
  edge.num_inliers = 100;
  return edge;
}

TEST(GlobalPoseGraphTest, BuildsCalibratedConstraintsWithDiagnostics) {
  MatchGraph graph;
  TwoViewGeometry accepted;
  accepted.type = TwoViewGeometry::Type::kCalibratedEssential;
  accepted.cam2_from_cam1.t_cw = Eigen::Vector3d(-1.0, 0.0, 0.0);
  accepted.median_triangulation_angle = 2.0;
  accepted.inlier_matches.resize(20);
  SetVerifiedGeometry(&graph, 1, 2, accepted);
  TwoViewGeometry rejected = accepted;
  rejected.inlier_matches.resize(3);
  SetVerifiedGeometry(&graph, 2, 3, rejected);

  ViewGraphCalibrationOptions options;
  options.min_num_inliers = 10;
  options.min_median_triangulation_angle_deg = 1.0;
  ViewGraphCalibrationSummary summary;
  const auto constraints =
      BuildRelativePoseConstraints(graph, options, &summary);

  ASSERT_EQ(constraints.size(), 1U);
  EXPECT_EQ(summary.num_input_edges, 2U);
  EXPECT_EQ(summary.num_accepted_edges, 1U);
  EXPECT_EQ(summary.num_rejected_inliers, 1U);
}

TEST(GlobalPoseGraphTest, AveragesConsistentRotationLoop) {
  const Eigen::Quaterniond yaw10(
      Eigen::AngleAxisd(10.0 * 3.14159265358979323846 / 180.0,
                        Eigen::Vector3d::UnitZ()));
  std::vector<RelativePoseConstraint> constraints{
      MakeConstraint(1, 2, yaw10, {-1.0, 0.0, 0.0}),
      MakeConstraint(2, 3, yaw10, {-1.0, 0.0, 0.0}),
      MakeConstraint(1, 3, yaw10 * yaw10, {-1.0, 0.0, 0.0})};
  std::map<ImageId, Eigen::Quaterniond> rotations;
  RotationAveragingOptions options;
  options.max_relative_rotation_error_deg = 1.0;
  const RotationAveragingSummary summary =
      EstimateGlobalRotations(constraints, 1, options, &rotations);

  ASSERT_TRUE(summary.success);
  ASSERT_EQ(rotations.size(), 3U);
  EXPECT_LT(summary.median_residual_deg, 1e-6);
  const Eigen::Quaterniond expected = yaw10 * yaw10;
  EXPECT_NEAR(std::abs(rotations.at(3).dot(expected)), 1.0, 1e-8);
}

TEST(GlobalPoseGraphTest, EstimatesCameraCentersFromTranslationDirections) {
  std::vector<RelativePoseConstraint> constraints{
      MakeConstraint(1, 2, Eigen::Quaterniond::Identity(),
                     {-1.0, 0.0, 0.0}),
      MakeConstraint(2, 3, Eigen::Quaterniond::Identity(),
                     {-1.0, 0.0, 0.0}),
      MakeConstraint(1, 3, Eigen::Quaterniond::Identity(),
                     {-1.0, 0.0, 0.0})};
  const std::map<ImageId, Eigen::Quaterniond> rotations{
      {1, Eigen::Quaterniond::Identity()},
      {2, Eigen::Quaterniond::Identity()},
      {3, Eigen::Quaterniond::Identity()}};
  std::map<ImageId, Eigen::Vector3d> centers;
  GlobalPositioningOptions options;
  options.filter_outliers = false;
  const GlobalPositioningSummary summary = EstimateGlobalCameraCenters(
      constraints, rotations, 1, options, &centers);

  ASSERT_TRUE(summary.success);
  EXPECT_NEAR(centers.at(1).norm(), 0.0, 1e-10);
  EXPECT_GT(centers.at(2).x(), 0.0);
  EXPECT_GT(centers.at(3).x(), centers.at(2).x());
  EXPECT_NEAR(centers.at(2).y(), 0.0, 1e-10);
}

}  // namespace
}  // namespace colite
