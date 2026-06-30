#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/exe/command.h"
#include "colite/exe/feature_extractor.h"
#include "colite/exe/mapper.h"
#include "colite/exe/sequential_matcher.h"
#include "colite/io/camera_prior.h"
#include "colite/io/sfm_workspace.h"

namespace colite {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

FeatureSet MakeTinyFeatures() {
  FeatureSet features;
  features.keypoints = {{10.0F, 20.0F, 1.5F, 0.1F, 2.0F},
                        {30.0F, 40.0F, 2.5F, 0.2F, 3.0F}};
  features.descriptors.resize(2, 128);
  features.descriptors.setZero();
  features.descriptors(0, 0) = 1.0F;
  features.descriptors(1, 127) = 2.0F;
  return features;
}

TEST(SfmWorkspaceTest, RoundTripsManifestFeaturesAndMatches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_workspace_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.file_path = "/tmp/image1.jpg";

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";
  image2.prior_position = Eigen::Vector3d(1.0, 2.0, 3.0);

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  MatchGraph graph;
  SetMatches(&graph, 1, 2, {{0, 1}, {1, 0}});
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1.t_cw = Eigen::Vector3d(1.0, 0.0, 0.0);
  geometry.median_triangulation_angle = 3.5;
  geometry.inlier_matches = {{0, 1}};
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);

  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  const std::vector<Image> images = ReadWorkspaceImages(workspace);
  const FeatureSet loaded_features = ReadWorkspaceFeatureSet(workspace, 1);
  MatchGraph loaded_graph;
  ReadWorkspaceMatches(workspace, &loaded_graph);
  ReadWorkspaceVerifiedGeometries(workspace, &loaded_graph);

  ASSERT_EQ(cameras.size(), 1U);
  EXPECT_EQ(cameras[0].width, 640U);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(images[1].name, "image2.jpg");
  ASSERT_TRUE(images[1].prior_position.has_value());
  EXPECT_NEAR(images[1].prior_position->x(), 1.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->y(), 2.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->z(), 3.0, 1e-12);
  ASSERT_EQ(loaded_features.keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(loaded_features.keypoints[1].response, 3.0F);
  ASSERT_EQ(loaded_features.descriptors.rows(), 2);
  EXPECT_FLOAT_EQ(loaded_features.descriptors(1, 127), 2.0F);

  const std::vector<FeatureMatch>* matches =
      FindMatches(loaded_graph, 1, 2);
  ASSERT_NE(matches, nullptr);
  EXPECT_EQ(matches->size(), 2U);
  const TwoViewGeometry* loaded_geometry =
      FindVerifiedGeometry(loaded_graph, 1, 2);
  ASSERT_NE(loaded_geometry, nullptr);
  EXPECT_EQ(loaded_geometry->type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_NEAR(loaded_geometry->median_triangulation_angle, 3.5, 1e-12);
  EXPECT_EQ(loaded_geometry->inlier_matches.size(), 1U);

  EXPECT_TRUE(std::filesystem::exists(workspace / "features" / "1.bin"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "raw_matches.tsv"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "verified_matches.tsv"));
}

TEST(SfmCommandExeTest, PrintsHelpForStageCommands) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunFeatureExtractor({"feature_extractor", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunSequentialMatcher({"sequential_matcher", "--help"}, out,
                                      err),
            0);
  EXPECT_NE(out.str().find("--overlap"), std::string::npos);
  EXPECT_NE(out.str().find("--matching_type"), std::string::npos);

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunMapper({"mapper", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
}

TEST(ColiteCommandExeTest, PrintsTopLevelHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("feature_extractor"), std::string::npos);
  EXPECT_NE(out.str().find("exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("recon_sfm"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesExhaustiveSpatialAndVocabTreeMatcherHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "exhaustive_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "spatial_matcher", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite spatial_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--spatial_max_distance"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--vocab_tree_num_images"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesSubcommandHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "recon_sfm", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("colite recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("--image_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, RejectsUnknownSubcommand) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "does_not_exist"}, out, err), 1);
  EXPECT_TRUE(out.str().empty());
  EXPECT_NE(err.str().find("Unknown command: does_not_exist"),
            std::string::npos);
  EXPECT_NE(err.str().find("Available commands:"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorUsesExifFocalPriorByDefault) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_exif_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  std::filesystem::copy_file(
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
          "P1180166.JPG",
      image_dir / "P1180166.JPG",
      std::filesystem::copy_options::overwrite_existing);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--workspace_path",
       workspace.string(),
       "--max_num_features",
       "32",
       "--feature_num_threads",
       "1"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  ASSERT_EQ(cameras.size(), 1U);
  ASSERT_GE(cameras[0].params.size(), 2U);
  EXPECT_NEAR(cameras[0].params[0], 2457.6, 1e-6);
  EXPECT_NEAR(cameras[0].params[1], 2457.6, 1e-6);
}

TEST(CameraPriorTest, EstimatesFocalLengthFromJpegExif) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";

  EXPECT_NEAR(EstimateInitialFocalLengthFromImage(image_path, 0.0), 2457.6,
              1e-6);
}

}  // namespace
}  // namespace colite
