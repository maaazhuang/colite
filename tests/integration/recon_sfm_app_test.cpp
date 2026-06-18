#include "recon_sfm_app.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/core/pose.h"

namespace colite::app {

double EstimateInitialFocalLengthFromImage(const std::filesystem::path& image_path,
                                           double user_focal_length);

namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

FeatureSet MakeRatioSensitiveFeatures1() {
  FeatureSet features;
  features.descriptors.resize(2, 2);
  features.descriptors << 0.0F, 0.0F,
                          100.0F, 100.0F;
  return features;
}

FeatureSet MakeRatioSensitiveFeatures2() {
  FeatureSet features;
  features.descriptors.resize(2, 2);
  features.descriptors << 1.0F, 0.0F,
                          1.2F, 0.0F;
  return features;
}

TEST(ReconSfmAppTest, PrintsHelp) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconSfmApp({"recon_sfm", "--help"}, out, err);

  EXPECT_EQ(exit_code, 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("--feature_extractor"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconSfmAppTest, RejectsMissingRequiredArguments) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code =
      RunReconSfmApp({"recon_sfm", "--image_path", "images"}, out, err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("--output_path"), std::string::npos);
}

TEST(ReconSfmAppTest, EstimatesInitialFocalLengthFromJpegExif) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";

  const double focal_length =
      EstimateInitialFocalLengthFromImage(image_path, 0.0);

  EXPECT_NEAR(focal_length, 2457.6, 1e-6);
}

TEST(ReconSfmAppTest, WritesColmapTextModel) {
  Reconstruction reconstruction;

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.registered = true;
  image1.cam_from_world = Pose3{};
  image1.points2d.push_back({{100.0, 120.0}, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  image2.registered = true;
  image2.cam_from_world = Pose3{};
  image2.cam_from_world->t_cw = Eigen::Vector3d(-1.0, 0.0, 0.0);
  image2.points2d.push_back({{105.0, 120.0}, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_colmap_export_test";
  WriteColmapTextModel(output_path, reconstruction);

  const std::string cameras = ReadTextFile(output_path / "cameras.txt");
  const std::string images = ReadTextFile(output_path / "images.txt");
  const std::string points = ReadTextFile(output_path / "points3D.txt");

  EXPECT_NE(cameras.find("# Camera list"), std::string::npos);
  EXPECT_NE(cameras.find("1 PINHOLE 640 480 500"), std::string::npos);
  EXPECT_NE(images.find("# Image list"), std::string::npos);
  EXPECT_NE(images.find("1 1 0 0 0 0 0 0 1 image1.jpg"),
            std::string::npos);
  EXPECT_NE(images.find("100 120 1"), std::string::npos);
  EXPECT_NE(points.find("# 3D point list"), std::string::npos);
  EXPECT_NE(points.find("1 0 0 5 255 255 255"), std::string::npos);
  EXPECT_NE(points.find("1 0 2 0"), std::string::npos);
}

TEST(ReconSfmAppTest, WritesQualitySummaryAndPlyPointCloud) {
  Reconstruction reconstruction;

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.registered = true;
  image1.cam_from_world = Pose3{};
  image1.points2d.push_back({{320.0, 240.0}, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  image2.registered = true;
  image2.cam_from_world = Pose3{};
  image2.cam_from_world->t_cw = Eigen::Vector3d(-1.0, 0.0, 0.0);
  image2.points2d.push_back({{220.0, 240.0}, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);

  const ReconstructionQualityStats stats =
      ComputeReconstructionQualityStats(reconstruction);
  EXPECT_EQ(stats.num_observations, 2U);
  EXPECT_NEAR(stats.mean_reprojection_error, 0.0, 1e-12);
  EXPECT_NEAR(stats.max_reprojection_error, 0.0, 1e-12);

  IncrementalMappingSummary summary;
  summary.success = true;
  summary.num_registered_images = 2;
  summary.num_points3d = 1;

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_summary_ply_test";
  WriteRunSummary(output_path, reconstruction, summary);
  WritePlyPointCloud(output_path / "points3D.ply", reconstruction);

  const std::string summary_text = ReadTextFile(output_path / "summary.txt");
  const std::string ply_text = ReadTextFile(output_path / "points3D.ply");

  EXPECT_NE(summary_text.find("num_observations 2"), std::string::npos);
  EXPECT_NE(summary_text.find("mean_reprojection_error 0"), std::string::npos);
  EXPECT_NE(summary_text.find("max_reprojection_error 0"), std::string::npos);
  EXPECT_NE(ply_text.find("ply"), std::string::npos);
  EXPECT_NE(ply_text.find("element vertex 1"), std::string::npos);
  EXPECT_NE(ply_text.find("0 0 5 255 255 255"), std::string::npos);
}

TEST(ReconSfmAppTest, WritesPairDiagnosticsForVerifiedMatches) {
  Reconstruction reconstruction;

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  reconstruction.AddImage(image1);

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  reconstruction.AddImage(image2);

  MatchGraph graph;
  SetMatches(&graph, 1, 2, {{0, 1}, {2, 3}, {4, 5}});

  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = {{0, 1}, {2, 3}};
  geometry.median_triangulation_angle = 1.5;
  SetVerifiedGeometry(&graph, 1, 2, geometry);

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_pair_diag_test";
  WritePairDiagnostics(output_path, reconstruction, graph);

  const std::string diagnostics =
      ReadTextFile(output_path / "pair_diagnostics.tsv");
  EXPECT_NE(diagnostics.find(
                "image_id1\timage_id2\timage_name1\timage_name2\traw_matches"
                "\tverified_inliers\tgeometry_type\tmedian_triangulation_angle"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("1\t2\timage1.jpg\timage2.jpg\t3\t2"
                             "\tCALIBRATED_ESSENTIAL\t1.5"),
            std::string::npos);
}

TEST(ReconSfmAppTest, BoostsOnlyNearbyPairMatchesWithRelaxedRatio) {
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeRatioSensitiveFeatures1());
  SetFeatures(&graph, 2, MakeRatioSensitiveFeatures2());
  SetFeatures(&graph, 4, MakeRatioSensitiveFeatures2());

  DescriptorMatchingOptions base_options;
  base_options.max_ratio = 0.8;
  base_options.mutual_check = true;

  NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = 1;
  nearby_options.max_ratio = 0.9;

  MatchAllImagePairs({1, 2, 4}, base_options, nearby_options, &graph);

  const std::vector<FeatureMatch>* nearby_matches = FindMatches(graph, 1, 2);
  const std::vector<FeatureMatch>* distant_matches = FindMatches(graph, 1, 4);
  ASSERT_NE(nearby_matches, nullptr);
  ASSERT_NE(distant_matches, nullptr);
  EXPECT_EQ(nearby_matches->size(), 1U);
  EXPECT_TRUE(distant_matches->empty());
}

}  // namespace
}  // namespace colite::app
