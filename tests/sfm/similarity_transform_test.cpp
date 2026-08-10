#include "colite/sfm/similarity_transform.h"

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(SimilarityTransformTest, RecoversKnownTransform) {
  const std::vector<Eigen::Vector3d> source = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const double scale = 2.5;
  const Eigen::Vector3d translation(4.0, -3.0, 7.0);
  std::vector<Eigen::Vector3d> target;
  for (const Eigen::Vector3d& point : source) {
    target.push_back(scale * rotation * point + translation);
  }

  const SimilarityTransform3 transform =
      EstimateSimilarityTransform(source, target);

  ASSERT_TRUE(transform.valid);
  EXPECT_NEAR(transform.scale, scale, 1e-10);
  EXPECT_LT((transform.rotation - rotation).norm(), 1e-10);
  EXPECT_LT((transform.translation - translation).norm(), 1e-10);
}

TEST(SimilarityTransformTest, AlignsCameraCentersToPositionPriors) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kSimplePinhole;
  camera.width = 100;
  camera.height = 100;
  camera.params = {80.0, 50.0, 50.0};
  reconstruction.AddCamera(camera);
  const std::vector<Eigen::Vector3d> centers = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}, {0.5, 0.5, 1.0}};
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    Image image;
    image.id = idx + 1;
    image.camera_id = 1;
    image.registered = true;
    image.cam_from_world = Pose3{};
    image.cam_from_world->t_cw = -centers[idx];
    image.prior_position = 3.0 * rotation * centers[idx] +
                           Eigen::Vector3d(20.0, -5.0, 2.0);
    reconstruction.AddImage(image);
  }

  const PositionPriorAlignmentSummary summary =
      AlignReconstructionToPositionPriors(0.01, &reconstruction);

  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_inliers, centers.size());
  EXPECT_NEAR(summary.scale, 3.0, 1e-10);
  EXPECT_LT(summary.mean_error, 1e-9);
}

TEST(SimilarityTransformTest, RejectsPositionPriorOutlier) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kSimplePinhole;
  camera.width = 100;
  camera.height = 100;
  camera.params = {80.0, 50.0, 50.0};
  reconstruction.AddCamera(camera);
  const std::vector<Eigen::Vector3d> centers = {
      {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0},
      {1.0, 1.0, 2.0}, {2.0, 2.0, 1.0}};
  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    Image image;
    image.id = idx + 1;
    image.camera_id = 1;
    image.registered = true;
    image.cam_from_world = Pose3{};
    image.cam_from_world->t_cw = -centers[idx];
    image.prior_position = 2.0 * centers[idx] + Eigen::Vector3d(4.0, 5.0, 6.0);
    if (idx == centers.size() - 1) *image.prior_position += Eigen::Vector3d(100.0, 0.0, 0.0);
    reconstruction.AddImage(std::move(image));
  }

  const PositionPriorAlignmentSummary summary =
      AlignReconstructionToPositionPriors(0.1, &reconstruction);

  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_inliers, 4U);
  EXPECT_NEAR(summary.scale, 2.0, 1e-8);
  EXPECT_LT(summary.mean_error, 1e-8);
}

}  // namespace
}  // namespace colite
