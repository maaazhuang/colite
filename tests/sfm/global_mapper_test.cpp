#include "colite/sfm/global_mapper.h"

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"

namespace colite {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 800;
  camera.height = 600;
  camera.params = {700.0, 700.0, 400.0, 300.0};
  return camera;
}

Pose3 CameraAt(const double center_x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
  return pose;
}

std::vector<Eigen::Vector3d> ScenePoints() {
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 6; ++col) {
      points.emplace_back(-0.8 + 0.32 * col, -0.45 + 0.3 * row,
                          4.0 + 0.08 * row + 0.04 * col);
    }
  }
  return points;
}

FeatureSet ProjectFeatures(const Camera& camera, const Pose3& pose,
                           const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const std::optional<Eigen::Vector2d> pixel =
        ProjectWorldPoint(camera, pose, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
         1.0F, 0.0F, 1.0F});
  }
  return features;
}

TwoViewGeometry MakeGeometry(const double center1, const double center2,
                             const std::size_t num_points) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1.t_cw =
      Eigen::Vector3d(center1 - center2, 0.0, 0.0);
  geometry.median_triangulation_angle = 5.0;
  for (std::size_t idx = 0; idx < num_points; ++idx) {
    geometry.inlier_matches.push_back(
        {static_cast<Point2DIdx>(idx), static_cast<Point2DIdx>(idx)});
  }
  return geometry;
}

TEST(GlobalMapperTest, RunsViewGraphPositioningTriangulationPipeline) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera();
  reconstruction.AddCamera(camera);
  const std::vector<double> centers{0.0, 1.0, 2.0};
  const std::vector<Eigen::Vector3d> points = ScenePoints();
  MatchGraph graph;
  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    Image image;
    image.id = static_cast<ImageId>(idx + 1);
    image.camera_id = camera.id;
    image.name = "image" + std::to_string(idx + 1) + ".jpg";
    reconstruction.AddImage(image);
    SetFeatures(&graph, image.id,
                ProjectFeatures(camera, CameraAt(centers[idx]), points));
  }
  SetVerifiedGeometry(&graph, 1, 2,
                      MakeGeometry(centers[0], centers[1], points.size()));
  SetVerifiedGeometry(&graph, 2, 3,
                      MakeGeometry(centers[1], centers[2], points.size()));
  SetVerifiedGeometry(&graph, 1, 3,
                      MakeGeometry(centers[0], centers[2], points.size()));

  GlobalMappingOptions options;
  options.view_graph.min_num_inliers = 8;
  options.triangulation.min_triangulation_angle_deg = 0.1;
  options.triangulation.ignore_two_view_tracks = true;
  const GlobalMappingSummary summary =
      RunGlobalMapping(graph, options, &reconstruction);

  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.num_registered_images, 3U);
  EXPECT_GE(summary.num_points3d, points.size() / 2);
  EXPECT_TRUE(summary.fixed_rotation_bundle_adjustment.success);
  EXPECT_FALSE(summary.retriangulation.message.empty());
  EXPECT_TRUE(summary.observation_filter.success);
  EXPECT_TRUE(summary.bundle_adjustment.success);
  EXPECT_EQ(reconstruction.RegisteredImageIds().size(), 3U);
  EXPECT_GT(reconstruction.GetImage(2).cam_from_world->CameraCenter().x(),
            reconstruction.GetImage(1).cam_from_world->CameraCenter().x());
  EXPECT_GT(reconstruction.GetImage(3).cam_from_world->CameraCenter().x(),
            reconstruction.GetImage(2).cam_from_world->CameraCenter().x());
}

TEST(GlobalMapperTest, ReportsMissingUsableViewGraph) {
  Reconstruction reconstruction;
  const GlobalMappingSummary summary =
      RunGlobalMapping(MatchGraph{}, GlobalMappingOptions{}, &reconstruction);
  EXPECT_FALSE(summary.success);
  EXPECT_NE(summary.message.find("View graph"), std::string::npos);
}

}  // namespace
}  // namespace colite
