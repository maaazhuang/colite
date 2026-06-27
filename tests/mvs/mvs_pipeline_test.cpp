#include "colite/mvs/mvs_pipeline.h"

#include <gtest/gtest.h>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 100;
  camera.height = 80;
  camera.params = InitializeCameraParams(camera.model_id, 50.0,
                                         camera.width, camera.height);
  return camera;
}

Image MakeRegisteredImage(ImageId image_id, double tx) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  image.registered = true;
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(tx, 0.0, 0.0);
  image.cam_from_world = pose;
  image.points2d.resize(3);
  return image;
}

Reconstruction MakeSparseReconstruction() {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeRegisteredImage(1, 0.0));
  reconstruction.AddImage(MakeRegisteredImage(2, -0.1));
  reconstruction.AddImage(MakeRegisteredImage(3, -0.2));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);

  Track track12;
  track12.AddElement(1, 0);
  track12.AddElement(2, 0);
  const Point3DId point12 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 2.0), track12);
  reconstruction.GetImage(1).points2d[0].xy = Eigen::Vector2d(50.0, 40.0);
  reconstruction.GetImage(1).points2d[0].point3d_id = point12;
  reconstruction.GetImage(2).points2d[0].point3d_id = point12;

  Track track123;
  track123.AddElement(1, 1);
  track123.AddElement(2, 1);
  track123.AddElement(3, 1);
  const Point3DId point123 =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.2, 0.0, 4.0), track123);
  reconstruction.GetImage(1).points2d[1].xy = Eigen::Vector2d(52.5, 40.0);
  reconstruction.GetImage(1).points2d[1].point3d_id = point123;
  reconstruction.GetImage(2).points2d[1].point3d_id = point123;
  reconstruction.GetImage(3).points2d[1].point3d_id = point123;

  Track track13;
  track13.AddElement(1, 2);
  track13.AddElement(3, 2);
  const Point3DId point13 =
      reconstruction.AddPoint3D(Eigen::Vector3d(-0.2, 0.0, 3.0), track13);
  reconstruction.GetImage(1).points2d[2].xy =
      Eigen::Vector2d(46.6666666667, 40.0);
  reconstruction.GetImage(1).points2d[2].point3d_id = point13;
  reconstruction.GetImage(3).points2d[2].point3d_id = point13;

  return reconstruction;
}

TEST(MvsPipelineTest, DefaultsToEightSourceImagesForDenserCoverage) {
  const MvsPipelineOptions options;

  EXPECT_EQ(options.max_num_source_images, 8U);
  EXPECT_DOUBLE_EQ(options.shared_points_weight, 1.0);
  EXPECT_DOUBLE_EQ(options.scale_consistency_weight, 2.0);
  EXPECT_DOUBLE_EQ(options.coverage_weight, 5.0);
  EXPECT_EQ(options.coverage_grid_size, 4U);
}

TEST(MvsPipelineTest, SelectsSourceImagesBySharedSparsePoints) {
  const Reconstruction reconstruction = MakeSparseReconstruction();
  MvsPipelineOptions options;
  options.max_num_source_images = 2;
  options.min_num_shared_points = 1;
  options.triangulation_angle_weight = 0.0;
  options.shared_points_weight = 1.0;
  options.scale_consistency_weight = 0.0;
  options.coverage_weight = 0.0;

  const std::vector<SourceImageScore> sources =
      SelectSourceImages(reconstruction, 1, options);

  ASSERT_EQ(sources.size(), 2U);
  EXPECT_EQ(sources[0].image_id, 2U);
  EXPECT_EQ(sources[0].num_shared_points, 2U);
  EXPECT_EQ(sources[1].image_id, 3U);
  EXPECT_EQ(sources[1].num_shared_points, 2U);
}

TEST(MvsPipelineTest, ScoresSourceImagesWithTriangulationAngleAndSharedPoints) {
  Reconstruction reconstruction = MakeSparseReconstruction();
  reconstruction.AddImage(MakeRegisteredImage(4, -0.8));
  reconstruction.RegisterImage(4);
  for (const Point3DId point3d_id : {Point3DId(1), Point3DId(2)}) {
    reconstruction.GetPoint3D(point3d_id).track.AddElement(4, 0);
  }

  MvsPipelineOptions options;
  options.max_num_source_images = 3;
  options.min_num_shared_points = 1;
  options.min_triangulation_angle_degrees = 0.0;
  options.max_triangulation_angle_degrees = 30.0;
  options.triangulation_angle_weight = 1.0;
  options.shared_points_weight = 0.01;
  options.scale_consistency_weight = 0.0;
  options.coverage_weight = 0.0;

  const std::vector<SourceImageScore> sources =
      SelectSourceImages(reconstruction, 1, options);

  ASSERT_EQ(sources.size(), 3U);
  EXPECT_EQ(sources[0].image_id, 4U);
  EXPECT_GT(sources[0].score, sources[1].score);
  EXPECT_GT(sources[0].mean_triangulation_angle_degrees,
            sources[1].mean_triangulation_angle_degrees);
  EXPECT_GE(sources[0].num_shared_points, 1U);
}

TEST(MvsPipelineTest, ScoresSourceImagesWithScaleAndCoverage) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeRegisteredImage(1, 0.0));
  reconstruction.AddImage(MakeRegisteredImage(2, -0.1));
  reconstruction.AddImage(MakeRegisteredImage(3, -0.1));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  reconstruction.RegisterImage(3);
  reconstruction.GetImage(3).cam_from_world->t_cw = Eigen::Vector3d(-0.1, 0.0, 8.0);
  reconstruction.GetImage(1).points2d.resize(8);
  reconstruction.GetImage(2).points2d.resize(4);
  reconstruction.GetImage(3).points2d.resize(4);

  const std::vector<Eigen::Vector3d> well_covered_points = {
      {-1.5, -1.0, 4.0},
      {1.5, -1.0, 4.0},
      {-1.5, 1.0, 4.0},
      {1.5, 1.0, 4.0},
  };
  for (std::size_t i = 0; i < well_covered_points.size(); ++i) {
    Track track;
    track.AddElement(1, static_cast<Point2DIdx>(i));
    track.AddElement(2, static_cast<Point2DIdx>(i));
    const Point3DId point3d_id =
        reconstruction.AddPoint3D(well_covered_points[i], track);
    reconstruction.GetImage(1).points2d[i].xy =
        Eigen::Vector2d(i % 2 == 0 ? 25.0 : 75.0, i < 2 ? 20.0 : 60.0);
    reconstruction.GetImage(1).points2d[i].point3d_id = point3d_id;
    reconstruction.GetImage(2).points2d[i].point3d_id = point3d_id;
  }

  for (std::size_t i = 0; i < 4; ++i) {
    Track track;
    track.AddElement(1, static_cast<Point2DIdx>(4 + i));
    track.AddElement(3, static_cast<Point2DIdx>(i));
    const Point3DId point3d_id =
        reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 12.0 + i), track);
    reconstruction.GetImage(1).points2d[4 + i].xy =
        Eigen::Vector2d(48.0 + static_cast<double>(i), 40.0);
    reconstruction.GetImage(1).points2d[4 + i].point3d_id = point3d_id;
    reconstruction.GetImage(3).points2d[i].point3d_id = point3d_id;
  }

  MvsPipelineOptions options;
  options.max_num_source_images = 2;
  options.min_num_shared_points = 1;
  options.triangulation_angle_weight = 0.0;
  options.shared_points_weight = 0.0;
  options.scale_consistency_weight = 1.0;
  options.coverage_weight = 10.0;
  options.coverage_grid_size = 2;

  const std::vector<SourceImageScore> sources =
      SelectSourceImages(reconstruction, 1, options);

  ASSERT_EQ(sources.size(), 2U);
  EXPECT_EQ(sources[0].image_id, 2U);
  EXPECT_GT(sources[0].coverage_score, sources[1].coverage_score);
  EXPECT_GT(sources[0].mean_scale_ratio, sources[1].mean_scale_ratio);
  EXPECT_GT(sources[0].score, sources[1].score);
}

TEST(MvsPipelineTest, BuildsSparseDepthMapFromVisiblePoints) {
  const Reconstruction reconstruction = MakeSparseReconstruction();

  const SparseDepthMap depth_map = BuildSparseDepthMap(reconstruction, 1);

  EXPECT_EQ(depth_map.image_id, 1U);
  EXPECT_EQ(depth_map.width, 100U);
  EXPECT_EQ(depth_map.height, 80U);
  ASSERT_EQ(depth_map.samples.size(), 3U);
  EXPECT_DOUBLE_EQ(depth_map.samples[0].depth, 2.0);
  EXPECT_DOUBLE_EQ(depth_map.samples[1].depth, 4.0);
  EXPECT_DOUBLE_EQ(depth_map.samples[2].depth, 3.0);
}

TEST(MvsPipelineTest, RunsSparseMvsWorkspacePass) {
  const Reconstruction reconstruction = MakeSparseReconstruction();

  const MvsPipelineSummary summary = RunMvsPipeline(reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.num_images, 3U);
  EXPECT_EQ(summary.num_sparse_depth_samples, 7U);
  ASSERT_EQ(summary.images.size(), 3U);
  EXPECT_DOUBLE_EQ(summary.images.front().min_depth, 2.0);
  EXPECT_DOUBLE_EQ(summary.images.front().max_depth, 4.0);
}

}  // namespace
}  // namespace colite::mvs
