#include "colite/mvs/patch_match.h"

#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

MvsImage MakeMvsImage() {
  MvsImage image;
  image.image_id = 1;
  image.name = "ref.png";
  image.gray = cv::Mat(2, 3, CV_32FC1, cv::Scalar(0.5F));
  image.scale = 1.0;
  return image;
}

MvsImage MakeGradientImage(std::size_t width, std::size_t height) {
  MvsImage image;
  image.image_id = 1;
  image.name = "gradient.png";
  image.gray = cv::Mat(static_cast<int>(height), static_cast<int>(width),
                       CV_32FC1);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      image.gray.at<float>(static_cast<int>(row), static_cast<int>(col)) =
          static_cast<float>(row * width + col) / 100.0F;
    }
  }
  image.scale = 1.0;
  return image;
}

PatchMatchView MakeView(ImageId image_id, double tx = 0.0) {
  PatchMatchView view;
  view.image_id = image_id;
  view.camera.id = image_id;
  view.camera.model_id = CameraModelId::kPinhole;
  view.camera.width = 5;
  view.camera.height = 5;
  view.camera.params =
      InitializeCameraParams(view.camera.model_id, 10.0, 5, 5);
  view.cam_from_world = Pose3{};
  view.cam_from_world.t_cw = Eigen::Vector3d(tx, 0.0, 0.0);
  return view;
}

Eigen::Vector3f NormalFromDegrees(float degrees) {
  constexpr float kPi = 3.14159265358979323846F;
  const float radians = degrees * kPi / 180.0F;
  return Eigen::Vector3f(std::sin(radians), 0.0F, std::cos(radians));
}

float PlaneDepthAtPixel(const PatchMatchView& view,
                        const Eigen::Vector3f& normal,
                        const Eigen::Vector2d& plane_pixel,
                        double plane_depth,
                        const Eigen::Vector2d& target_pixel) {
  const Eigen::Vector3d plane_point =
      UnprojectPixel(view.camera, plane_pixel, plane_depth);
  const Eigen::Vector3d target_ray =
      UnprojectPixel(view.camera, target_pixel, 1.0);
  const double numerator = normal.cast<double>().dot(plane_point);
  const double denominator = normal.cast<double>().dot(target_ray);
  return static_cast<float>(numerator / denominator);
}

TEST(CpuPatchMatchInitializationTest, InitializesDepthAndNormalMaps) {
  PatchMatchInitializationOptions options;
  options.random_seed = 7;

  const PatchMatchMaps maps =
      InitializePatchMatchMaps(MakeMvsImage(), 2.0F, 4.0F, options);

  EXPECT_EQ(maps.depth_map.Width(), 3U);
  EXPECT_EQ(maps.depth_map.Height(), 2U);
  EXPECT_FLOAT_EQ(maps.depth_map.DepthMin(), 2.0F);
  EXPECT_FLOAT_EQ(maps.depth_map.DepthMax(), 4.0F);
  EXPECT_EQ(maps.depth_map.ValidPixelCount(), 6U);
  EXPECT_EQ(maps.normal_map.ValidPixelCount(), 6U);
  for (std::size_t row = 0; row < maps.depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < maps.depth_map.Width(); ++col) {
      EXPECT_GE(maps.depth_map.Get(row, col), 2.0F);
      EXPECT_LE(maps.depth_map.Get(row, col), 4.0F);
      EXPECT_FLOAT_EQ(maps.normal_map.Get(row, col).x(), 0.0F);
      EXPECT_FLOAT_EQ(maps.normal_map.Get(row, col).y(), 0.0F);
      EXPECT_FLOAT_EQ(maps.normal_map.Get(row, col).z(), -1.0F);
    }
  }
}

TEST(CpuPatchMatchInitializationTest, IsDeterministicForSeed) {
  PatchMatchInitializationOptions options;
  options.random_seed = 11;

  const PatchMatchMaps maps1 =
      InitializePatchMatchMaps(MakeMvsImage(), 1.0F, 3.0F, options);
  const PatchMatchMaps maps2 =
      InitializePatchMatchMaps(MakeMvsImage(), 1.0F, 3.0F, options);

  ASSERT_EQ(maps1.depth_map.Data().size(), maps2.depth_map.Data().size());
  for (std::size_t i = 0; i < maps1.depth_map.Data().size(); ++i) {
    EXPECT_FLOAT_EQ(maps1.depth_map.Data()[i], maps2.depth_map.Data()[i]);
  }
}

TEST(CpuPatchMatchInitializationTest, InitializesFromSparseDepthRange) {
  SparseDepthMap sparse_depth_map;
  sparse_depth_map.image_id = 1;
  sparse_depth_map.width = 3;
  sparse_depth_map.height = 2;
  sparse_depth_map.samples.push_back({1, 1.0, 1.0, 2.0});
  sparse_depth_map.samples.push_back({2, 2.0, 1.0, 5.0});

  const PatchMatchMaps maps =
      InitializePatchMatchMaps(MakeMvsImage(), sparse_depth_map);

  EXPECT_FLOAT_EQ(maps.depth_map.DepthMin(), 2.0F);
  EXPECT_FLOAT_EQ(maps.depth_map.DepthMax(), 5.0F);
  EXPECT_EQ(maps.depth_map.ValidPixelCount(), 6U);
}

TEST(CpuPatchMatchInitializationTest,
     SparseDepthRangePercentilesIgnoreExtremeOutlier) {
  SparseDepthMap sparse_depth_map;
  sparse_depth_map.image_id = 1;
  sparse_depth_map.width = 3;
  sparse_depth_map.height = 2;
  sparse_depth_map.samples.push_back({1, 0.0, 0.0, 1.0});
  sparse_depth_map.samples.push_back({2, 1.0, 0.0, 2.0});
  sparse_depth_map.samples.push_back({3, 2.0, 0.0, 2.0});
  sparse_depth_map.samples.push_back({4, 0.0, 1.0, 2.0});
  sparse_depth_map.samples.push_back({5, 1.0, 1.0, 2.0});
  sparse_depth_map.samples.push_back({6, 2.0, 1.0, 1000.0});

  PatchMatchInitializationOptions options;
  options.sparse_depth_min_percentile = 0.2F;
  options.sparse_depth_max_percentile = 0.8F;
  options.depth_range_padding = 0.0F;

  const PatchMatchMaps maps =
      InitializePatchMatchMaps(MakeMvsImage(), sparse_depth_map, options);

  EXPECT_FLOAT_EQ(maps.depth_map.DepthMin(), 2.0F);
  EXPECT_FLOAT_EQ(maps.depth_map.DepthMax(), 2.0F);
}

TEST(CpuPatchMatchInitializationTest, RandomInitialNormalsAreDeterministic) {
  PatchMatchInitializationOptions options;
  options.random_seed = 17;
  options.randomize_initial_normals = true;
  options.initial_normal_hemisphere_degrees = 45.0F;
  options.default_normal = Eigen::Vector3f(0.0F, 0.0F, -1.0F);

  const PatchMatchMaps maps1 =
      InitializePatchMatchMaps(MakeMvsImage(), 2.0F, 4.0F, options);
  const PatchMatchMaps maps2 =
      InitializePatchMatchMaps(MakeMvsImage(), 2.0F, 4.0F, options);

  bool saw_randomized_normal = false;
  const float min_dot = std::cos(45.0F * 3.14159265358979323846F / 180.0F);
  for (std::size_t row = 0; row < maps1.normal_map.Height(); ++row) {
    for (std::size_t col = 0; col < maps1.normal_map.Width(); ++col) {
      const Eigen::Vector3f normal1 = maps1.normal_map.Get(row, col);
      const Eigen::Vector3f normal2 = maps2.normal_map.Get(row, col);
      EXPECT_NEAR(normal1.norm(), 1.0F, 1e-5F);
      EXPECT_NEAR(normal1.x(), normal2.x(), 1e-6F);
      EXPECT_NEAR(normal1.y(), normal2.y(), 1e-6F);
      EXPECT_NEAR(normal1.z(), normal2.z(), 1e-6F);
      EXPECT_GE(normal1.dot(options.default_normal.normalized()),
                min_dot - 1e-5F);
      saw_randomized_normal =
          saw_randomized_normal ||
          (normal1 - options.default_normal.normalized()).norm() > 1e-3F;
    }
  }
  EXPECT_TRUE(saw_randomized_normal);
}

TEST(CpuPatchMatchNormalRefinementTest, EstimatesNormalFromTiltedDepthPlane) {
  const PatchMatchView view = MakeView(1);
  DenseDepthMap depth_map(5, 5, 1.0F, 5.0F);
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      depth_map.Set(row, col, 3.0F + 0.05F * static_cast<float>(col));
    }
  }
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));

  PatchMatchNormalRefinementOptions options;
  options.max_relative_depth_difference = 1.0F;
  const PatchMatchNormalRefinementSummary summary =
      RefineNormalMapFromDepth(view, depth_map, options, &normal_map);

  const Eigen::Vector3d left =
      UnprojectPixel(view.camera, Eigen::Vector2d(1.0, 2.0),
                     static_cast<double>(depth_map.Get(2, 1)));
  const Eigen::Vector3d right =
      UnprojectPixel(view.camera, Eigen::Vector2d(3.0, 2.0),
                     static_cast<double>(depth_map.Get(2, 3)));
  const Eigen::Vector3d up =
      UnprojectPixel(view.camera, Eigen::Vector2d(2.0, 1.0),
                     static_cast<double>(depth_map.Get(1, 2)));
  const Eigen::Vector3d down =
      UnprojectPixel(view.camera, Eigen::Vector2d(2.0, 3.0),
                     static_cast<double>(depth_map.Get(3, 2)));
  Eigen::Vector3f expected = (right - left).cross(down - up).cast<float>();
  expected.normalize();
  if (expected.z() < 0.0F) {
    expected = -expected;
  }

  EXPECT_EQ(summary.num_pixels, 25U);
  EXPECT_EQ(summary.num_refined_pixels, 9U);
  EXPECT_NEAR(normal_map.Get(2, 2).x(), expected.x(), 1e-5F);
  EXPECT_NEAR(normal_map.Get(2, 2).y(), expected.y(), 1e-5F);
  EXPECT_NEAR(normal_map.Get(2, 2).z(), expected.z(), 1e-5F);
}

TEST(CpuPatchMatchNormalRefinementTest,
     KeepsOriginalNormalAcrossDepthDiscontinuity) {
  const PatchMatchView view = MakeView(1);
  DenseDepthMap depth_map(3, 3, 1.0F, 6.0F);
  depth_map.Fill(3.0F);
  depth_map.Set(1, 2, 5.0F);
  NormalMap normal_map(3, 3);
  const Eigen::Vector3f original_normal(0.0F, 0.0F, 1.0F);
  normal_map.Fill(original_normal);

  PatchMatchNormalRefinementOptions options;
  options.max_relative_depth_difference = 0.05F;
  const PatchMatchNormalRefinementSummary summary =
      RefineNormalMapFromDepth(view, depth_map, options, &normal_map);

  EXPECT_EQ(summary.num_pixels, 9U);
  EXPECT_EQ(summary.num_refined_pixels, 0U);
  EXPECT_TRUE(normal_map.IsValid(1, 1));
  EXPECT_NEAR(normal_map.Get(1, 1).x(), original_normal.x(), 1e-6F);
  EXPECT_NEAR(normal_map.Get(1, 1).y(), original_normal.y(), 1e-6F);
  EXPECT_NEAR(normal_map.Get(1, 1).z(), original_normal.z(), 1e-6F);
}

TEST(CpuPatchMatchPhotometricCostTest, SamplesGrayImageBilinearly) {
  const MvsImage image = MakeGradientImage(3, 3);

  const std::optional<float> sample = SampleGrayBilinear(image, 1.5F, 1.0F);

  ASSERT_TRUE(sample.has_value());
  EXPECT_NEAR(*sample, 0.045F, 1e-6F);
  EXPECT_FALSE(SampleGrayBilinear(image, -0.1F, 1.0F).has_value());
  EXPECT_FALSE(SampleGrayBilinear(image, 1.0F, 2.2F).has_value());
}

TEST(CpuPatchMatchPhotometricCostTest, ComputesZnccForMatchingPatches) {
  const MvsImage image = MakeGradientImage(5, 5);
  PatchMatchCostOptions options;
  options.window_radius = 1;
  options.window_step = 1;

  const PatchMatchCost same =
      ComputePatchZncc(image, image, {2.0F, 2.0F}, {2.0F, 2.0F}, options);

  EXPECT_TRUE(same.valid);
  EXPECT_NEAR(same.score, 1.0F, 1e-5F);
  EXPECT_NEAR(same.cost, 0.0F, 1e-5F);
}

TEST(CpuPatchMatchPhotometricCostTest, PenalizesInvertedPatches) {
  const MvsImage image = MakeGradientImage(5, 5);
  MvsImage inverted = image;
  inverted.gray = cv::Mat(image.gray.rows, image.gray.cols, CV_32FC1);
  for (int row = 0; row < image.gray.rows; ++row) {
    for (int col = 0; col < image.gray.cols; ++col) {
      inverted.gray.at<float>(row, col) =
          1.0F - image.gray.at<float>(row, col);
    }
  }

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchCost cost =
      ComputePatchZncc(image, inverted, {2.0F, 2.0F}, {2.0F, 2.0F}, options);

  EXPECT_TRUE(cost.valid);
  EXPECT_LT(cost.score, -0.99F);
  EXPECT_GT(cost.cost, 0.99F);
}

TEST(CpuPatchMatchPhotometricCostTest, ProjectsDepthHypothesisToSourceView) {
  const PatchMatchView ref_view = MakeView(1);
  const PatchMatchView src_view = MakeView(2);
  const Eigen::Vector2f projected =
      ProjectDepthToSourcePixel(ref_view, src_view, {2.0F, 2.0F}, 3.0F).value();

  EXPECT_NEAR(projected.x(), 2.0F, 1e-6F);
  EXPECT_NEAR(projected.y(), 2.0F, 1e-6F);
}

TEST(CpuPatchMatchPhotometricCostTest, ScoresDepthHypothesisWithGeometry) {
  const MvsImage image = MakeGradientImage(5, 5);
  PatchMatchCostOptions options;
  options.window_radius = 1;

  const PatchMatchCost cost =
      ComputeDepthHypothesisCost(MakeView(1), MakeView(2), image, image,
                                 {2.0F, 2.0F}, 3.0F, options);

  EXPECT_TRUE(cost.valid);
  EXPECT_NEAR(cost.cost, 0.0F, 1e-5F);
}

TEST(CpuPatchMatchPhotometricCostTest, PlaneProjectionUsesSurfaceNormal) {
  const PatchMatchView ref_view = MakeView(1);
  const PatchMatchView src_view = MakeView(2, -0.2);
  const Eigen::Vector2f center_xy(2.0F, 2.0F);
  const Eigen::Vector2f sample_xy(3.0F, 2.0F);

  const Eigen::Vector2f fronto_parallel =
      ProjectPlaneToSourcePixel(ref_view, src_view, center_xy, 3.0F,
                                Eigen::Vector3f(0.0F, 0.0F, 1.0F),
                                sample_xy)
          .value();
  const Eigen::Vector2f tilted =
      ProjectPlaneToSourcePixel(ref_view, src_view, center_xy, 3.0F,
                                Eigen::Vector3f(0.5F, 0.0F, 1.0F),
                                sample_xy)
          .value();

  EXPECT_NEAR(fronto_parallel.y(), tilted.y(), 1e-6F);
  EXPECT_GT(std::abs(fronto_parallel.x() - tilted.x()), 1e-3F);
}

TEST(CpuPatchMatchPhotometricCostTest, PlanePatchCostMatchesIdenticalViews) {
  const MvsImage image = MakeGradientImage(5, 5);
  PatchMatchCostOptions options;
  options.window_radius = 1;

  const PatchMatchCost cost =
      ComputePlaneHypothesisCost(MakeView(1), MakeView(2), image, image,
                                 {2.0F, 2.0F}, 3.0F,
                                 Eigen::Vector3f(0.0F, 0.0F, 1.0F), options);

  EXPECT_TRUE(cost.valid);
  EXPECT_NEAR(cost.cost, 0.0F, 1e-5F);
}

TEST(CpuPatchMatchPhotometricCostTest,
     PenalizesGeometricallyInconsistentSourceDepth) {
  const PatchMatchView reference_view = MakeView(1);
  const PatchMatchView source_view = MakeView(2);
  const MvsImage reference_image = MakeGradientImage(5, 5);
  const MvsImage source_image = reference_image;

  DenseDepthMap source_depth_map(5, 5, 1.0F, 5.0F);
  source_depth_map.Set(2, 2, 4.5F);
  source_depth_map.Set(2, 3, 4.5F);
  source_depth_map.Set(3, 2, 4.5F);
  source_depth_map.Set(3, 3, 4.5F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  options.geometric_consistency_weight = 1.0F;
  options.geometric_max_relative_depth_error = 0.01F;

  const PatchMatchCost cost_without_source_depth = ComputePlaneHypothesisCost(
      reference_view, source_view, reference_image, source_image,
      Eigen::Vector2f(2.0F, 2.0F), 3.0F,
      Eigen::Vector3f(0.0F, 0.0F, 1.0F), options);
  const PatchMatchCost cost_with_source_depth = ComputePlaneHypothesisCost(
      reference_view, source_view, reference_image, source_image,
      Eigen::Vector2f(2.0F, 2.0F), 3.0F,
      Eigen::Vector3f(0.0F, 0.0F, 1.0F), options, &source_depth_map);

  ASSERT_TRUE(cost_without_source_depth.valid);
  ASSERT_TRUE(cost_with_source_depth.valid);
  EXPECT_GT(cost_with_source_depth.cost, cost_without_source_depth.cost);
}

TEST(CpuPatchMatchPhotometricCostTest, AggregatesBestSourceViewCosts) {
  const MvsImage reference = MakeGradientImage(5, 5);
  const MvsImage good_source = reference;
  MvsImage bad_source = reference;
  bad_source.gray = cv::Mat(reference.gray.rows, reference.gray.cols, CV_32FC1);
  for (int row = 0; row < reference.gray.rows; ++row) {
    for (int col = 0; col < reference.gray.cols; ++col) {
      bad_source.gray.at<float>(row, col) =
          1.0F - reference.gray.at<float>(row, col);
    }
  }

  PatchMatchCostOptions options;
  options.window_radius = 1;
  options.max_num_aggregated_costs = 1;

  const MultiSourcePatchMatchCost cost =
      ComputeMultiSourcePlaneHypothesisCost(
          MakeView(1), {MakeView(2), MakeView(3)}, reference,
          {good_source, bad_source}, {2.0F, 2.0F}, 3.0F,
          Eigen::Vector3f(0.0F, 0.0F, 1.0F), options);

  EXPECT_TRUE(cost.valid);
  EXPECT_EQ(cost.num_total_sources, 2U);
  EXPECT_EQ(cost.num_valid_sources, 2U);
  EXPECT_EQ(cost.num_used_sources, 1U);
  EXPECT_NEAR(cost.cost, 0.0F, 1e-5F);
}

TEST(CpuPatchMatchBestCostMapTest, ComputesBestCostMapForCurrentHypotheses) {
  const MvsImage reference = MakeGradientImage(5, 5);
  const MvsImage source = reference;
  DenseDepthMap depth_map(5, 5, 2.0F, 4.0F);
  depth_map.Fill(3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const DenseCostMap cost_map =
      ComputeBestCostMap(MakeView(1), {MakeView(2)}, reference, {source},
                         depth_map, normal_map, options);

  EXPECT_EQ(cost_map.Width(), 5U);
  EXPECT_EQ(cost_map.Height(), 5U);
  EXPECT_EQ(cost_map.ValidPixelCount(), 9U);
  EXPECT_TRUE(cost_map.IsValid(2, 2));
  EXPECT_NEAR(cost_map.Get(2, 2), 0.0F, 1e-5F);
  EXPECT_FALSE(cost_map.IsValid(0, 0));
}

TEST(CpuPatchMatchBestCostMapTest, SkipsInvalidDepthOrNormalPixels) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 4.0F);
  depth_map.Fill(3.0F);
  depth_map.Set(2, 2, 0.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  normal_map.Set(2, 3, Eigen::Vector3f::Zero());

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const DenseCostMap cost_map =
      ComputeBestCostMap(MakeView(1), {MakeView(2)}, reference, {reference},
                         depth_map, normal_map, options);

  EXPECT_FALSE(cost_map.IsValid(2, 2));
  EXPECT_FALSE(cost_map.IsValid(2, 3));
}

TEST(CpuPatchMatchPropagationTest, ForwardSweepUpdatesFromLeftNeighbor) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(2, 1, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 1, 0.0F);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary summary = RunPatchMatchPropagationSweep(
      MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
      &normal_map, &cost_map, PatchMatchSweepDirection::kForward, options);

  EXPECT_GT(summary.num_candidate_evaluations, 0U);
  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 3.0F);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchPropagationTest, PropagatesNeighborPlaneDepthToCurrentRay) {
  const MvsImage reference = MakeGradientImage(5, 5);
  const PatchMatchView view = MakeView(1);
  const Eigen::Vector3f plane_normal = NormalFromDegrees(15.0F);
  const Eigen::Vector2d neighbor_pixel(1.0, 2.0);
  const Eigen::Vector2d target_pixel(2.0, 2.0);
  const float neighbor_depth = 3.0F;
  const float expected_target_depth =
      PlaneDepthAtPixel(view, plane_normal, neighbor_pixel, neighbor_depth,
                        target_pixel);

  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(2, 1, neighbor_depth);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  normal_map.SetNormalized(2, 1, plane_normal);
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 1, 0.0F);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary summary = RunPatchMatchPropagationSweep(
      view, {MakeView(2)}, reference, {reference}, &depth_map, &normal_map,
      &cost_map, PatchMatchSweepDirection::kForward, options);

  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_NEAR(depth_map.Get(2, 2), expected_target_depth, 1e-5F);
  EXPECT_NEAR(normal_map.Get(2, 2).x(), plane_normal.normalized().x(), 1e-5F);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchPropagationTest, BackwardSweepUpdatesFromRightNeighbor) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(2, 3, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 3, 0.0F);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary summary = RunPatchMatchPropagationSweep(
      MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
      &normal_map, &cost_map, PatchMatchSweepDirection::kBackward, options);

  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 3.0F);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchPropagationTest, TopDownSweepUpdatesFromUpperNeighbor) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(1, 2, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(1, 2, 0.0F);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary summary = RunPatchMatchPropagationSweep(
      MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
      &normal_map, &cost_map, PatchMatchSweepDirection::kTopDown, options);

  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 3.0F);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchPropagationTest, BottomUpSweepUpdatesFromLowerNeighbor) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(3, 2, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(3, 2, 0.0F);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary summary = RunPatchMatchPropagationSweep(
      MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
      &normal_map, &cost_map, PatchMatchSweepDirection::kBottomUp, options);

  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 3.0F);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchPropagationTest,
     CheckerboardPropagationUpdatesOnlyRequestedParity) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(5.0F);
  depth_map.Set(2, 1, 3.0F);
  depth_map.Set(2, 4, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 1, 0.0F);
  cost_map.Set(2, 2, 1.0F);
  cost_map.Set(2, 3, 1.0F);
  cost_map.Set(2, 4, 0.0F);

  PatchMatchCostOptions options;
  options.window_radius = 1;
  const PatchMatchSweepSummary parity_zero_summary =
      RunPatchMatchCheckerboardPropagation(
          MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
          &normal_map, &cost_map, 0, options);

  EXPECT_GT(parity_zero_summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 3.0F);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 3), 5.0F);

  const PatchMatchSweepSummary parity_one_summary =
      RunPatchMatchCheckerboardPropagation(
          MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
          &normal_map, &cost_map, 1, options);

  EXPECT_GT(parity_one_summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 3), 3.0F);
}

TEST(CpuPatchMatchPerturbationTest, RandomPerturbationImprovesCurrentCost) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(4.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions cost_options;
  cost_options.window_radius = 1;
  PatchMatchPerturbationOptions perturbation_options;
  perturbation_options.random_seed = 17;
  perturbation_options.depth_radius = 0.25F;
  perturbation_options.normal_radius_degrees = 5.0F;

  const PatchMatchPerturbationSummary summary =
      RunPatchMatchRandomPerturbation(
          MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
          &normal_map, &cost_map, perturbation_options, cost_options);

  EXPECT_GT(summary.num_candidate_evaluations, 0U);
  EXPECT_GT(summary.num_updates, 0U);
  EXPECT_LT(cost_map.Get(2, 2), 1.0F);
  EXPECT_GE(depth_map.Get(2, 2), depth_map.DepthMin());
  EXPECT_LE(depth_map.Get(2, 2), depth_map.DepthMax());
  EXPECT_TRUE(normal_map.IsValid(2, 2));
}

TEST(CpuPatchMatchPerturbationTest,
     EvaluatesMultiplePlaneCandidatesAndReportsUpdateMagnitude) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(4.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 2, 1.0F);

  PatchMatchCostOptions cost_options;
  cost_options.window_radius = 1;
  PatchMatchPerturbationOptions perturbation_options;
  perturbation_options.random_seed = 17;
  perturbation_options.depth_radius = 0.25F;
  perturbation_options.normal_radius_degrees = 5.0F;
  perturbation_options.num_samples_per_pixel = 3;

  const PatchMatchPerturbationSummary summary =
      RunPatchMatchRandomPerturbation(
          MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
          &normal_map, &cost_map, perturbation_options, cost_options);

  EXPECT_EQ(summary.num_candidate_evaluations, 3U);
  EXPECT_EQ(summary.num_updates, 1U);
  EXPECT_GT(summary.total_abs_depth_update, 0.0);
  EXPECT_GT(summary.max_abs_depth_update, 0.0);
  EXPECT_GE(summary.total_normal_angle_degrees, 0.0);
  EXPECT_GE(summary.max_normal_angle_degrees, 0.0);
  EXPECT_NEAR(summary.mean_abs_depth_update,
              summary.total_abs_depth_update /
              static_cast<double>(summary.num_updates),
              1e-12);
}

TEST(CpuPatchMatchPerturbationTest,
     RejectsCandidatesWithoutSourceDepthConsistencyDuringRefinement) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 2.0F, 6.0F);
  depth_map.Fill(4.0F);
  NormalMap normal_map(5, 5);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 2, 1.0F);

  DenseDepthMap source_depth_map(5, 5, 2.0F, 6.0F);
  source_depth_map.Fill(4.0F);
  std::vector<DenseDepthMap> source_depth_maps;
  source_depth_maps.push_back(source_depth_map);

  PatchMatchCostOptions cost_options;
  cost_options.window_radius = 1;
  cost_options.geometric_consistency_weight = 0.01F;
  cost_options.geometric_max_relative_depth_error = 1e-6F;
  PatchMatchPerturbationOptions perturbation_options;
  perturbation_options.random_seed = 17;
  perturbation_options.depth_radius = 0.25F;
  perturbation_options.normal_radius_degrees = 5.0F;
  perturbation_options.num_samples_per_pixel = 3;

  const PatchMatchPerturbationSummary summary =
      RunPatchMatchRandomPerturbation(
          MakeView(1), {MakeView(2)}, reference, {reference}, &depth_map,
          &normal_map, &cost_map, perturbation_options, cost_options,
          &source_depth_maps);

  EXPECT_EQ(summary.num_candidate_evaluations, 3U);
  EXPECT_EQ(summary.num_geometric_candidate_rejections, 3U);
  EXPECT_EQ(summary.num_updates, 0U);
  EXPECT_FLOAT_EQ(depth_map.Get(2, 2), 4.0F);
  EXPECT_FLOAT_EQ(cost_map.Get(2, 2), 1.0F);
}

TEST(CpuPatchMatchStereoTest, RunsIterationsAndProducesBestCostMap) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 2;
  options.cost.window_radius = 1;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.depth_map.Width(), 5U);
  EXPECT_EQ(result.normal_map.Height(), 5U);
  EXPECT_EQ(result.cost_map.ValidPixelCount(), 9U);
  EXPECT_EQ(result.summary.num_iterations, 2U);
  EXPECT_EQ(result.summary.num_forward_sweeps, 2U);
  EXPECT_EQ(result.summary.num_backward_sweeps, 2U);
  EXPECT_EQ(result.summary.num_perturbation_passes, 2U);
  EXPECT_GT(result.summary.num_candidate_evaluations, 0U);
  EXPECT_EQ(result.summary.num_valid_cost_pixels, 9U);
}

TEST(PatchMatchBackendTest, ParsesBackendNames) {
  EXPECT_EQ(PatchMatchBackendFromName("auto"), PatchMatchBackend::kAuto);
  EXPECT_EQ(PatchMatchBackendFromName("cpu"), PatchMatchBackend::kCpu);
  EXPECT_EQ(PatchMatchBackendFromName("cuda"), PatchMatchBackend::kCuda);
  EXPECT_EQ(PatchMatchBackendFromName("gpu"), PatchMatchBackend::kCuda);
  EXPECT_EQ(PatchMatchBackendToName(PatchMatchBackend::kAuto), "auto");
  EXPECT_EQ(PatchMatchBackendToName(PatchMatchBackend::kCpu), "cpu");
  EXPECT_EQ(PatchMatchBackendToName(PatchMatchBackend::kCuda), "cuda");
  EXPECT_THROW((void)PatchMatchBackendFromName("metal"),
               std::invalid_argument);
}

TEST(PatchMatchBackendTest, AutoFallsBackToCpuWhenCudaIsUnavailable) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kAuto;
  options.num_iterations = 0;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.summary.requested_backend, PatchMatchBackend::kAuto);
  if (IsCudaPatchMatchAvailable()) {
    EXPECT_EQ(result.summary.actual_backend, PatchMatchBackend::kCuda);
  } else {
    EXPECT_EQ(result.summary.actual_backend, PatchMatchBackend::kCpu);
  }
}

TEST(PatchMatchBackendTest, ForcedCpuUsesCpuBackend) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.num_iterations = 0;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {}, reference, {}, 2.0F, 6.0F,
                                  options);

  EXPECT_EQ(result.summary.requested_backend, PatchMatchBackend::kCpu);
  EXPECT_EQ(result.summary.actual_backend, PatchMatchBackend::kCpu);
}

TEST(PatchMatchBackendTest, ForcedCudaFailsWhenCudaIsUnavailable) {
  if (IsCudaPatchMatchAvailable()) {
    GTEST_SKIP() << "CUDA PatchMatch backend is available in this build.";
  }
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCuda;
  options.num_iterations = 0;

  EXPECT_THROW(RunPatchMatchStereoForImage(MakeView(1), {}, reference, {},
                                           2.0F, 6.0F, options),
               std::runtime_error);
  EXPECT_FALSE(CudaPatchMatchUnavailableReason().empty());
}

TEST(PatchMatchBackendTest, CudaBackendRunsBasicStereoWhenCompiled) {
  if (!IsCudaPatchMatchCompiled()) {
    GTEST_SKIP() << "CUDA PatchMatch backend is not compiled in this build.";
  }
  ASSERT_TRUE(IsCudaPatchMatchAvailable())
      << CudaPatchMatchUnavailableReason();

  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCuda;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 1;
  options.cost.window_radius = 1;
  options.perturbation.num_samples_per_pixel = 1;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.summary.requested_backend, PatchMatchBackend::kCuda);
  EXPECT_EQ(result.summary.actual_backend, PatchMatchBackend::kCuda);
  EXPECT_EQ(result.depth_map.Width(), 5U);
  EXPECT_EQ(result.normal_map.Height(), 5U);
  EXPECT_GT(result.cost_map.ValidPixelCount(), 0U);
  EXPECT_EQ(result.summary.num_iterations, 1U);
  EXPECT_GT(result.summary.num_candidate_evaluations, 0U);
  EXPECT_GT(result.summary.num_perturbation_passes, 0U);
}

TEST(PatchMatchBackendTest, CudaBackendRunsPyramidStereoWhenCompiled) {
  if (!IsCudaPatchMatchCompiled()) {
    GTEST_SKIP() << "CUDA PatchMatch backend is not compiled in this build.";
  }
  ASSERT_TRUE(IsCudaPatchMatchAvailable())
      << CudaPatchMatchUnavailableReason();

  const MvsImage reference = MakeGradientImage(8, 8);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCuda;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 1;
  options.pyramid_levels = 2;
  options.cost.window_radius = 1;
  options.perturbation.num_samples_per_pixel = 1;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.summary.requested_backend, PatchMatchBackend::kCuda);
  EXPECT_EQ(result.summary.actual_backend, PatchMatchBackend::kCuda);
  EXPECT_EQ(result.summary.num_pyramid_levels, 2U);
  EXPECT_EQ(result.depth_map.Width(), 8U);
  EXPECT_EQ(result.depth_map.Height(), 8U);
  EXPECT_GT(result.cost_map.ValidPixelCount(), 0U);
  EXPECT_EQ(result.summary.num_iterations, 2U);
  EXPECT_GT(result.summary.num_candidate_evaluations, 0U);
}

TEST(CpuPatchMatchStereoTest,
     PyramidInitializationUpsamplesCoarseDepthToFineLevel) {
  const MvsImage reference = MakeGradientImage(4, 4);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.num_iterations = 0;
  options.pyramid_levels = 2;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {}, reference, {}, 2.0F, 6.0F,
                                  options);

  EXPECT_EQ(result.summary.num_pyramid_levels, 2U);
  ASSERT_EQ(result.depth_map.Width(), 4U);
  ASSERT_EQ(result.depth_map.Height(), 4U);
  for (std::size_t row = 0; row < result.depth_map.Height(); row += 2) {
    for (std::size_t col = 0; col < result.depth_map.Width(); col += 2) {
      const float block_depth = result.depth_map.Get(row, col);
      EXPECT_FLOAT_EQ(result.depth_map.Get(row, col + 1), block_depth);
      EXPECT_FLOAT_EQ(result.depth_map.Get(row + 1, col), block_depth);
      EXPECT_FLOAT_EQ(result.depth_map.Get(row + 1, col + 1), block_depth);
    }
  }
}

TEST(CpuPatchMatchStereoTest, RefinesNormalsFromDepthWhenEnabled) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.num_iterations = 0;
  options.initialization.default_normal = Eigen::Vector3f(0.0F, 0.0F, 1.0F);
  options.normal_refinement.enabled = true;
  options.normal_refinement.max_relative_depth_difference = 1.0F;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {}, reference, {}, 3.0F, 3.0F,
                                  options);

  EXPECT_EQ(result.summary.num_normal_refinement_pixels, 25U);
  EXPECT_EQ(result.summary.num_normal_refined_pixels, 9U);
  EXPECT_EQ(result.summary.num_normal_refinement_skipped_pixels, 16U);
  EXPECT_TRUE(result.normal_map.IsValid(2, 2));
  EXPECT_NEAR(result.normal_map.Get(2, 2).z(), 1.0F, 1e-5F);
}

TEST(CpuPatchMatchStereoTest, CheckerboardScheduleRunsFourRedBlackRounds) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 1;
  options.cost.window_radius = 1;
  options.propagation_schedule =
      PatchMatchPropagationSchedule::kCheckerboard;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.summary.num_checkerboard_passes, 8U);
  EXPECT_EQ(result.summary.num_forward_sweeps, 0U);
  EXPECT_EQ(result.summary.num_backward_sweeps, 0U);
  EXPECT_GT(result.summary.num_candidate_evaluations, 0U);
}

TEST(CpuPatchMatchStereoTest, RecordsAdaptivePerturbationSummaryPerIteration) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 3;
  options.cost.window_radius = 1;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {MakeView(2)}, reference,
                                  {reference}, 2.0F, 6.0F, options);

  ASSERT_EQ(result.summary.update_ratios.size(), 3U);
  ASSERT_EQ(result.summary.depth_radii.size(), 3U);
  ASSERT_EQ(result.summary.normal_radii_degrees.size(), 3U);
  for (std::size_t i = 0; i < result.summary.update_ratios.size(); ++i) {
    EXPECT_GE(result.summary.update_ratios[i], 0.0);
    EXPECT_LE(result.summary.update_ratios[i], 1.0);
    EXPECT_GE(result.summary.depth_radii[i], options.min_depth_perturbation);
    EXPECT_GE(result.summary.normal_radii_degrees[i],
              options.min_normal_radius_degrees);
  }
  EXPECT_GE(result.summary.depth_radii.front(),
            result.summary.depth_radii.back());
}

TEST(CpuPatchMatchStereoTest, AdaptivePerturbationShrinksFasterWhenUpdatesAreLow) {
  const MvsImage reference = MakeGradientImage(5, 5);
  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 2;
  options.cost.window_radius = 1;
  options.initial_depth_perturbation_factor = 0.5F;
  options.perturbation_shrink = 0.5F;
  options.adaptive_perturbation_min_update_ratio = 0.5F;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImage(MakeView(1), {}, reference, {}, 2.0F, 6.0F,
                                  options);

  ASSERT_EQ(result.summary.update_ratios.size(), 2U);
  ASSERT_EQ(result.summary.depth_radii.size(), 2U);
  EXPECT_DOUBLE_EQ(result.summary.update_ratios.front(), 0.0);
  EXPECT_FLOAT_EQ(result.summary.depth_radii.front(), 2.0F);
  EXPECT_FLOAT_EQ(result.summary.depth_radii.back(), 0.5F);
}

TEST(CpuPatchMatchStereoTest, RunsGeometricRefinementWithSourceDepthMaps) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap source_depth_map(5, 5, 1.0F, 6.0F);
  source_depth_map.Fill(3.0F);

  PatchMatchStereoOptions options;
  options.backend = PatchMatchBackend::kCpu;
  options.initialization.random_seed = 19;
  options.perturbation.random_seed = 23;
  options.num_iterations = 1;
  options.cost.window_radius = 1;
  options.cost.geometric_consistency_weight = 0.2F;
  options.cost.geometric_max_relative_depth_error = 0.01F;

  const PatchMatchStereoResult result =
      RunPatchMatchStereoForImageWithSourceDepths(
          MakeView(1), {MakeView(2)}, reference, {reference},
          {source_depth_map}, 2.0F, 6.0F, options);

  EXPECT_EQ(result.depth_map.Width(), 5U);
  EXPECT_EQ(result.summary.num_iterations, 1U);
  EXPECT_GT(result.summary.num_candidate_evaluations, 0U);
  EXPECT_GT(result.cost_map.ValidPixelCount(), 0U);
}

TEST(CpuPatchMatchPhotometricStatsTest, ComputesCostDistribution) {
  DenseCostMap cost_map(3, 2);
  cost_map.Set(0, 0, 0.10F);
  cost_map.Set(0, 1, 0.20F);
  cost_map.Set(1, 0, 0.40F);

  const PatchMatchPhotometricStats stats =
      ComputePatchMatchPhotometricStats(cost_map);

  EXPECT_EQ(stats.num_pixels, 6U);
  EXPECT_EQ(stats.num_valid_cost_pixels, 3U);
  EXPECT_EQ(stats.num_invalid_cost_pixels, 3U);
  EXPECT_NEAR(stats.valid_ratio, 0.5, 1e-9);
  EXPECT_NEAR(stats.mean_cost, 0.233333333, 1e-6);
  EXPECT_NEAR(stats.median_cost, 0.20, 1e-6);
  EXPECT_NEAR(stats.min_cost, 0.10, 1e-6);
  EXPECT_NEAR(stats.max_cost, 0.40, 1e-6);
}

TEST(CpuPatchMatchPhotometricFilterTest, RemovesInvalidAndHighCostPixels) {
  DenseDepthMap depth_map(3, 1, 1.0F, 5.0F);
  depth_map.Set(0, 0, 2.0F);
  depth_map.Set(0, 1, 3.0F);
  depth_map.Set(0, 2, 4.0F);
  NormalMap normal_map(3, 1);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(3, 1);
  cost_map.Set(0, 0, 0.20F);
  cost_map.Set(0, 1, 0.80F);

  const PatchMatchPhotometricFilterSummary summary =
      FilterPatchMatchPhotometricResults(0.50F, &depth_map, &normal_map,
                                         &cost_map);

  EXPECT_EQ(summary.num_pixels, 3U);
  EXPECT_EQ(summary.num_kept_pixels, 1U);
  EXPECT_EQ(summary.num_filtered_pixels, 2U);
  EXPECT_TRUE(depth_map.IsValid(0, 0));
  EXPECT_TRUE(normal_map.IsValid(0, 0));
  EXPECT_TRUE(cost_map.IsValid(0, 0));
  EXPECT_FALSE(depth_map.IsValid(0, 1));
  EXPECT_FALSE(normal_map.IsValid(0, 1));
  EXPECT_FALSE(cost_map.IsValid(0, 1));
  EXPECT_FALSE(depth_map.IsValid(0, 2));
  EXPECT_FALSE(normal_map.IsValid(0, 2));
  EXPECT_FALSE(cost_map.IsValid(0, 2));
}

TEST(CpuPatchMatchGeometricConsistencyTest, BuildsConsistencyGraphFromMatchingDepthMaps) {
  DenseDepthMap reference_depth(5, 5, 1.0F, 5.0F);
  reference_depth.Fill(3.0F);
  DenseDepthMap source_depth(5, 5, 1.0F, 5.0F);
  source_depth.Fill(3.0F);

  PatchMatchGeometricConsistencyOptions options;
  options.max_reprojection_error_pixels = 0.25F;
  options.max_relative_depth_error = 0.001F;

  const ConsistencyGraph graph = ComputeGeometricConsistencyGraph(
      MakeView(1), {MakeView(2)}, reference_depth, {source_depth}, options);

  EXPECT_EQ(graph.Width(), 5U);
  EXPECT_EQ(graph.Height(), 5U);
  EXPECT_EQ(graph.Count(2, 2), 1U);
  EXPECT_EQ(graph.SourceImageIds(2, 2).front(), 2U);
  EXPECT_EQ(graph.NumConsistentPixels(1), 25U);
}

TEST(CpuPatchMatchGeometricConsistencyTest, RejectsInconsistentSourceDepth) {
  DenseDepthMap reference_depth(5, 5, 1.0F, 5.0F);
  reference_depth.Fill(3.0F);
  DenseDepthMap source_depth(5, 5, 1.0F, 5.0F);
  source_depth.Fill(4.0F);

  PatchMatchGeometricConsistencyOptions options;
  options.max_reprojection_error_pixels = 0.25F;
  options.max_relative_depth_error = 0.001F;

  const ConsistencyGraph graph = ComputeGeometricConsistencyGraph(
      MakeView(1), {MakeView(2)}, reference_depth, {source_depth}, options);

  EXPECT_EQ(graph.Count(2, 2), 0U);
  EXPECT_EQ(graph.NumConsistentPixels(1), 0U);
}

TEST(CpuPatchMatchGeometricConsistencyTest,
     RoundTripsConsistencyGraphAsBinary) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_consistency_graph_binary_test";
  std::filesystem::create_directories(root);

  ConsistencyGraph graph(3, 2);
  graph.Add(0, 0, 2);
  graph.Add(0, 0, 3);
  graph.Add(1, 2, 4);

  WriteConsistencyGraphBinary(root / "graph.bin", graph);
  const ConsistencyGraph loaded =
      ReadConsistencyGraphBinary(root / "graph.bin");

  EXPECT_EQ(loaded.Width(), 3U);
  EXPECT_EQ(loaded.Height(), 2U);
  EXPECT_EQ(loaded.Count(0, 0), 2U);
  EXPECT_EQ(loaded.SourceImageIds(0, 0)[0], 2U);
  EXPECT_EQ(loaded.SourceImageIds(0, 0)[1], 3U);
  EXPECT_EQ(loaded.Count(0, 1), 0U);
  EXPECT_EQ(loaded.Count(1, 2), 1U);
  EXPECT_EQ(loaded.SourceImageIds(1, 2).front(), 4U);
}

TEST(CpuPatchMatchGeometricFilterTest, FiltersPixelsWithoutEnoughConsistentSources) {
  DenseDepthMap depth_map(3, 1, 1.0F, 5.0F);
  depth_map.Fill(3.0F);
  NormalMap normal_map(3, 1);
  normal_map.Fill(Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(3, 1);
  cost_map.Set(0, 0, 0.1F);
  cost_map.Set(0, 1, 0.2F);
  cost_map.Set(0, 2, 0.3F);
  ConsistencyGraph graph(3, 1);
  graph.Add(0, 0, 2);
  graph.Add(0, 1, 2);
  graph.Add(0, 1, 3);

  const PatchMatchGeometricFilterSummary summary =
      FilterPatchMatchGeometricResults(graph, 2, &depth_map, &normal_map,
                                       &cost_map);

  EXPECT_EQ(summary.num_pixels, 3U);
  EXPECT_EQ(summary.num_kept_pixels, 1U);
  EXPECT_EQ(summary.num_filtered_pixels, 2U);
  EXPECT_FALSE(depth_map.IsValid(0, 0));
  EXPECT_TRUE(depth_map.IsValid(0, 1));
  EXPECT_FALSE(depth_map.IsValid(0, 2));
  EXPECT_FALSE(normal_map.IsValid(0, 0));
  EXPECT_TRUE(normal_map.IsValid(0, 1));
  EXPECT_FALSE(cost_map.IsValid(0, 2));
}

TEST(CpuStereoFusionTest, FusesConsistentDepthPixelsIntoWorldPoints) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 1.0F, 5.0F);
  depth_map.Set(2, 2, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 2, 0.1F);
  ConsistencyGraph graph(5, 5);
  graph.Add(2, 2, 2);

  const StereoFusionResult result =
      FuseStereoDepthMaps({MakeView(1)}, {reference}, {depth_map},
                          {normal_map}, {cost_map}, {graph});

  ASSERT_EQ(result.points.size(), 1U);
  EXPECT_EQ(result.summary.num_input_images, 1U);
  EXPECT_EQ(result.summary.num_depth_pixels, 1U);
  EXPECT_EQ(result.summary.num_fused_points, 1U);
  EXPECT_EQ(result.summary.num_fusion_tracks, 1U);
  EXPECT_EQ(result.summary.num_fused_observations, 1U);
  const FusedPoint& point = result.points.front();
  EXPECT_EQ(point.image_id, 1U);
  EXPECT_EQ(point.row, 2U);
  EXPECT_EQ(point.col, 2U);
  EXPECT_NEAR(point.xyz.z(), 3.0, 1e-9);
  EXPECT_NEAR(point.normal.z(), 1.0F, 1e-6F);
  EXPECT_EQ(point.color.x(), static_cast<std::uint8_t>(31));
  EXPECT_EQ(point.color.y(), static_cast<std::uint8_t>(31));
  EXPECT_EQ(point.color.z(), static_cast<std::uint8_t>(31));
}

TEST(CpuStereoFusionTest, MergesReciprocalConsistentPixelsIntoSingleTrack) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;
  image2.name = "source.png";

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  depth2.Set(2, 2, 3.0F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 2, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 2, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  ASSERT_EQ(result.points.size(), 1U);
  ASSERT_EQ(result.tracks.size(), 1U);
  ASSERT_EQ(result.pair_diagnostics.size(), 2U);
  EXPECT_EQ(result.summary.num_depth_pixels, 2U);
  EXPECT_EQ(result.summary.num_fusion_tracks, 1U);
  EXPECT_EQ(result.summary.num_fused_observations, 2U);
  EXPECT_EQ(result.summary.num_visited_pixels, 2U);
  EXPECT_EQ(result.summary.num_skipped_pixels, 0U);
  EXPECT_EQ(result.tracks.front().seed_image_id, 1U);
  EXPECT_EQ(result.tracks.front().num_observations, 2U);
  EXPECT_EQ(result.tracks.front().num_unique_images, 2U);
  EXPECT_EQ(result.pair_diagnostics[0].reference_image_id, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].source_image_id, 2U);
  EXPECT_EQ(result.pair_diagnostics[0].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].num_accepted, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].num_rejected_normal, 0U);
  EXPECT_EQ(result.pair_diagnostics[0].num_rejected_view, 0U);
  EXPECT_EQ(result.pair_diagnostics[1].reference_image_id, 2U);
  EXPECT_EQ(result.pair_diagnostics[1].source_image_id, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_accepted, 0U);
  EXPECT_EQ(result.pair_diagnostics[1].num_rejected_duplicate, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_rejected_other, 1U);
  EXPECT_NEAR(result.points.front().xyz.z(), 3.0, 1e-9);
}

TEST(CpuStereoFusionTest, RecordsFusionErrorSamplesForThresholdTuning) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;
  image2.name = "source.png";

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  depth2.Set(2, 2, 3.0F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 2, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 2, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  ASSERT_FALSE(result.error_samples.relative_depth_errors.empty());
  ASSERT_FALSE(result.error_samples.reprojection_errors_pixels.empty());
  ASSERT_FALSE(result.error_samples.normal_angles_degrees.empty());
  ASSERT_FALSE(result.error_samples.view_angles_degrees.empty());
  EXPECT_NEAR(result.error_samples.relative_depth_errors.front(), 0.0F, 1e-6F);
  EXPECT_NEAR(result.error_samples.reprojection_errors_pixels.front(), 0.0F,
              1e-6F);
  EXPECT_NEAR(result.error_samples.normal_angles_degrees.front(), 0.0F, 1e-5F);
  EXPECT_NEAR(result.error_samples.view_angles_degrees.front(), 0.0F, 1e-5F);
}

TEST(CpuStereoFusionTest, SearchesNearbySourcePixelWhenRoundedProjectionInvalid) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;
  image2.name = "source.png";

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  depth2.Set(2, 3, 3.0F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 3, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 3, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 3, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  options.source_pixel_search_radius = 1;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  ASSERT_EQ(result.points.size(), 1U);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().num_observations, 2U);
  EXPECT_EQ(result.summary.num_fused_observations, 2U);
  EXPECT_EQ(result.pair_diagnostics[0].num_accepted, 1U);
}

TEST(CpuStereoFusionTest, UsesBilinearSourceDepthForFusionValidation) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;
  image2.name = "source.png";

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  // 源图投影约为 (2.6, 2.0)。最近像素深度为 3.1，离散检查会失败；
  // 但 (2,2)-(3,2) 的双线性插值深度为 3.0，应与几何一致性图一致地通过。
  depth2.Set(2, 2, 2.85F);
  depth2.Set(2, 3, 3.10F);
  depth2.Set(3, 2, 2.85F);
  depth2.Set(3, 3, 3.10F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 3, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 3, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 3, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  options.source_pixel_search_radius = 1;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2, 0.18)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  ASSERT_EQ(result.points.size(), 1U);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().num_observations, 2U);
  EXPECT_EQ(result.summary.num_fused_observations, 2U);
  EXPECT_EQ(result.pair_diagnostics[0].num_accepted, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].num_rejected_depth, 0U);
}

TEST(CpuStereoFusionTest, PropagatesTrackAcrossLocallyConsistentNormals) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;
  MvsImage image3 = image1;
  image3.image_id = 3;

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  DenseDepthMap depth3(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  depth2.Set(2, 2, 3.0F);
  depth3.Set(2, 2, 3.0F);
  NormalMap normal1(5, 5);
  NormalMap normal2(5, 5);
  NormalMap normal3(5, 5);
  normal1.Set(2, 2, NormalFromDegrees(0.0F));
  normal2.Set(2, 2, NormalFromDegrees(8.0F));
  normal3.Set(2, 2, NormalFromDegrees(16.0F));
  DenseCostMap cost1(5, 5);
  DenseCostMap cost2(5, 5);
  DenseCostMap cost3(5, 5);
  cost1.Set(2, 2, 0.1F);
  cost2.Set(2, 2, 0.1F);
  cost3.Set(2, 2, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 2, 1);
  graph2.Add(2, 2, 3);
  ConsistencyGraph graph3(5, 5);
  graph3.Add(2, 2, 2);

  StereoFusionOptions options;
  options.min_num_fused_observations = 3;
  options.max_normal_error_degrees = 10.0F;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2), MakeView(3)}, {image1, image2, image3},
      {depth1, depth2, depth3}, {normal1, normal2, normal3},
      {cost1, cost2, cost3}, {graph1, graph2, graph3}, options);

  ASSERT_EQ(result.points.size(), 1U);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().num_observations, 3U);
  EXPECT_NEAR(result.tracks.front().max_normal_angle_degrees, 8.0, 1e-4);
}

TEST(CpuStereoFusionTest, RejectsSourceObservationWithLargeNormalAngle) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  depth2.Set(2, 2, 3.0F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 2, Eigen::Vector3f(1.0F, 0.0F, 0.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 2, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 2, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  options.max_normal_error_degrees = 10.0F;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  EXPECT_TRUE(result.points.empty());
  EXPECT_EQ(result.summary.num_rejected_normal_observations, 2U);
  EXPECT_EQ(result.summary.num_rejected_tracks, 2U);
  ASSERT_EQ(result.pair_diagnostics.size(), 2U);
  EXPECT_EQ(result.pair_diagnostics[0].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].num_rejected_normal, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_rejected_normal, 1U);
}

TEST(CpuStereoFusionTest, RejectsSourceObservationWithLargeViewAngle) {
  const MvsImage image1 = MakeGradientImage(5, 5);
  MvsImage image2 = image1;
  image2.image_id = 2;

  DenseDepthMap depth1(5, 5, 1.0F, 5.0F);
  depth1.Set(2, 2, 3.0F);
  DenseDepthMap depth2(5, 5, 1.0F, 5.0F);
  depth2.Set(2, 4, 3.0F);
  NormalMap normal1(5, 5);
  normal1.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  NormalMap normal2(5, 5);
  normal2.Set(2, 4, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost1(5, 5);
  cost1.Set(2, 2, 0.1F);
  DenseCostMap cost2(5, 5);
  cost2.Set(2, 4, 0.1F);
  ConsistencyGraph graph1(5, 5);
  graph1.Add(2, 2, 2);
  ConsistencyGraph graph2(5, 5);
  graph2.Add(2, 4, 1);

  StereoFusionOptions options;
  options.min_num_fused_observations = 2;
  options.max_view_angle_degrees = 1.0F;
  const StereoFusionResult result = FuseStereoDepthMaps(
      {MakeView(1), MakeView(2, 0.5)}, {image1, image2}, {depth1, depth2},
      {normal1, normal2}, {cost1, cost2}, {graph1, graph2}, options);

  EXPECT_TRUE(result.points.empty());
  EXPECT_EQ(result.summary.num_rejected_view_observations, 2U);
  EXPECT_EQ(result.summary.num_rejected_tracks, 2U);
  ASSERT_EQ(result.pair_diagnostics.size(), 2U);
  EXPECT_EQ(result.pair_diagnostics[0].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[0].num_rejected_view, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_candidates, 1U);
  EXPECT_EQ(result.pair_diagnostics[1].num_rejected_view, 1U);
}

TEST(CpuStereoFusionTest, SkipsPixelsWithoutConsistencySupport) {
  const MvsImage reference = MakeGradientImage(5, 5);
  DenseDepthMap depth_map(5, 5, 1.0F, 5.0F);
  depth_map.Set(2, 2, 3.0F);
  NormalMap normal_map(5, 5);
  normal_map.Set(2, 2, Eigen::Vector3f(0.0F, 0.0F, 1.0F));
  DenseCostMap cost_map(5, 5);
  cost_map.Set(2, 2, 0.1F);
  ConsistencyGraph graph(5, 5);

  StereoFusionOptions options;
  options.min_num_consistent_sources = 1;
  const StereoFusionResult result =
      FuseStereoDepthMaps({MakeView(1)}, {reference}, {depth_map},
                          {normal_map}, {cost_map}, {graph}, options);

  EXPECT_TRUE(result.points.empty());
  EXPECT_EQ(result.summary.num_depth_pixels, 1U);
  EXPECT_EQ(result.summary.num_skipped_pixels, 1U);
}

}  // namespace
}  // namespace colite::mvs
