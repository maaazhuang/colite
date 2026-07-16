#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/controllers/sequential_matcher.h"
#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/exe/command.h"
#include "colite/exe/compare_colmap.h"
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

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  file << contents;
}

void WriteTinyColmapModel(const std::filesystem::path& path,
                          const bool candidate) {
  WriteTextFile(path / "cameras.txt",
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                "1 PINHOLE 640 480 500 500 320 240\n");
  if (!candidate) {
    WriteTextFile(path / "images.txt",
                  "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                  "# POINTS2D[] as X Y POINT3D_ID\n"
                  "1 1 0 0 0 0 0 0 1 image1.jpg\n"
                  "10 20 1 30 40 2\n"
                  "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                  "15 25 1\n");
    WriteTextFile(path / "points3D.txt",
                  "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                  "1 0 0 5 255 255 255 0.5 1 0 2 0\n"
                  "2 1 0 5 255 255 255 0.7 1 1\n");
    return;
  }

  WriteTextFile(path / "images.txt",
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "# POINTS2D[] as X Y POINT3D_ID\n"
                "1 1 0 0 0 0.1 0 0 1 image1.jpg\n"
                "10 20 1 30 40 3\n"
                "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                "15 25 1 35 45 3\n"
                "3 1 0 0 0 2 0 0 1 image3.jpg\n"
                "\n");
  WriteTextFile(path / "points3D.txt",
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                "1 0.1 0 5 255 255 255 0.6 1 0 2 0\n"
                "3 2 0 5 255 255 255 0.9 1 1 2 1\n");
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

TEST(SfmWorkspaceTest, RoundTripsStageCacheMetadata) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_cache_test";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(workspace / "features");
  {
    std::ofstream file(workspace / "features" / "1.bin");
    file << "feature bytes";
  }

  const std::string hash = StableConfigHash(
      {{"stage", "features"}, {"backend", "vlsift"}, {"max_num_features", "64"}});
  WriteWorkspaceCacheMetadata(workspace, "features", hash,
                              {{"images", "1"},
                               {"feature_extraction_seconds", "0.25"}},
                              {"features/1.bin"});

  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace, "features");

  EXPECT_EQ(metadata.stage, "features");
  EXPECT_EQ(metadata.config_hash, hash);
  EXPECT_EQ(metadata.values.at("images"), "1");
  EXPECT_TRUE(WorkspaceCacheIsValid(workspace, "features", hash));
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash + "x"));

  std::filesystem::remove(workspace / "features" / "1.bin");
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash));
}

TEST(SfmWorkspaceTest, SequentialMatcherReusesRawAndGeometryCaches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_match_cache_test";
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

  Image image2 = image1;
  image2.id = 2;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  controllers::SequentialMatcherControllerOptions options;
  options.workspace_path = workspace;
  options.matching_strategy = MatchingStrategy::kExhaustive;
  options.matcher_backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;

  const controllers::SequentialMatcherControllerSummary first =
      controllers::SequentialMatcherController(options).Run();
  const controllers::SequentialMatcherControllerSummary second =
      controllers::SequentialMatcherController(options).Run();

  EXPECT_TRUE(first.success);
  EXPECT_FALSE(first.raw_match_cache_hit);
  EXPECT_FALSE(first.geometry_cache_hit);
  EXPECT_TRUE(second.success);
  EXPECT_TRUE(second.raw_match_cache_hit);
  EXPECT_TRUE(second.geometry_cache_hit);
  EXPECT_EQ(second.num_pairs, first.num_pairs);
  EXPECT_EQ(second.num_verified_inliers, first.num_verified_inliers);
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
  EXPECT_NE(out.str().find("vocab_tree_builder"), std::string::npos);
  EXPECT_NE(out.str().find("compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("recon_auto"), std::string::npos);
  EXPECT_NE(out.str().find("image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("delaunay_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_texturer"), std::string::npos);
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
  EXPECT_NE(out.str().find("--vocab_tree_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_builder", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("--output_path"), std::string::npos);
  EXPECT_NE(out.str().find("--branching"), std::string::npos);
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

TEST(ColiteCommandExeTest, DispatchesStandaloneMvsToolHelp) {
  for (const std::string& command : {"poisson_mesher", "delaunay_mesher",
                                     "mesh_simplifier", "mesh_texturer"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("--input_path"), std::string::npos) << command;
    EXPECT_NE(out.str().find("--output_path"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, DispatchesStagedPipelineHelp) {
  for (const std::string& command : {"recon_auto", "image_undistorter",
                                     "patch_match_stereo", "stereo_fusion"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("Usage:"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, ReconAutoDryRunPrintsAllStageBoundaries) {
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite({"colite", "recon_auto", "--image_path", "images",
                            "--output_path", "result", "--dry_run"},
                           out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_texturer"), std::string::npos);
}

TEST(ColiteCommandExeTest, SurfaceCommandsSupportDryRunStageContract) {
  struct Case {
    std::string command;
    std::vector<std::string> arguments;
    std::string stage;
  };
  const std::vector<Case> cases = {
      {"poisson_mesher",
       {"--input_path", "fused.ply", "--output_path", "mesh.ply"},
       "poisson_mesher"},
      {"delaunay_mesher",
       {"--input_path", "dense", "--output_path", "mesh.ply"},
       "delaunay_mesher"},
      {"mesh_simplifier",
       {"--input_path", "mesh.ply", "--output_path", "simple.ply"},
       "mesh_simplifier"},
      {"mesh_texturer",
       {"--input_path", "mesh.ply", "--output_path", "textured",
        "--model_path", "sparse", "--image_path", "images",
        "--MeshTextureMapping.visibility_resolution", "512",
        "--MeshTextureMapping.occlusion_relative_tolerance", "0.02"},
       "mesh_texturer"}};
  for (const Case& test_case : cases) {
    std::vector<std::string> command{"colite", test_case.command};
    command.insert(command.end(), test_case.arguments.begin(),
                   test_case.arguments.end());
    command.push_back("--dry_run");
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite(command, out, err), 0)
        << test_case.command << ": " << err.str();
    EXPECT_NE(out.str().find("PLAN " + test_case.stage), std::string::npos)
        << test_case.command;
  }
}

TEST(ColiteCommandExeTest, MeshSimplifierWritesAndResumesStageManifest) {
  const auto root = std::filesystem::temp_directory_path() /
                    "colite_mesh_simplifier_stage_test";
  std::filesystem::remove_all(root);
  const auto input = root / "input.ply";
  const auto output = root / "output.ply";
  WriteTextFile(input,
                "ply\nformat ascii 1.0\n"
                "element vertex 4\nproperty float x\nproperty float y\n"
                "property float z\nelement face 2\n"
                "property list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
                "3 0 1 2\n3 0 2 3\n");
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_ratio", "0.75"},
                out, err),
            0)
      << err.str();
  EXPECT_TRUE(std::filesystem::exists(output));
  EXPECT_TRUE(std::filesystem::exists(
      root / ".colite" / "stages" / "mesh_simplifier.json"));

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_ratio", "0.75", "--resume"},
                out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("SKIP mesh_simplifier"), std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ColiteCommandExeTest, ReconAutoCanSelectDelaunayOrStopAtFusion) {
  std::ostringstream delaunay_out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--mesher", "delaunay",
                 "--dry_run"},
                delaunay_out, err),
            0)
      << err.str();
  EXPECT_NE(delaunay_out.str().find("PLAN delaunay_mesher"),
            std::string::npos);

  std::ostringstream fusion_out;
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--stop_after", "fusion",
                 "--dry_run"},
                fusion_out, err),
            0)
      << err.str();
  EXPECT_EQ(fusion_out.str().find("PLAN poisson_mesher"), std::string::npos);
}

TEST(ColiteCompareExeTest, WritesFixedColmapDifferenceReports) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_compare_colmap_test";
  const std::filesystem::path reference = root / "reference";
  const std::filesystem::path candidate = root / "candidate";
  const std::filesystem::path output = root / "report";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(reference, false);
  WriteTinyColmapModel(candidate, true);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunCompareColmap({"compare_colmap",
                                   "--reference_path",
                                   reference.string(),
                                   "--candidate_path",
                                   candidate.string(),
                                   "--output_path",
                                   output.string()},
                                  out,
                                  err),
            0)
      << err.str();

  EXPECT_NE(out.str().find("Common images: 2"), std::string::npos);
  EXPECT_NE(out.str().find("Common points by id: 1"), std::string::npos);
  const std::string image_diff = ReadTextFile(output / "image_diff.tsv");
  const std::string pair_diff = ReadTextFile(output / "pair_diff.tsv");
  const std::string point_diff = ReadTextFile(output / "point_diff.tsv");
  const std::string pose_diff = ReadTextFile(output / "pose_diff.tsv");
  const std::string summary = ReadTextFile(output / "summary.txt");

  EXPECT_NE(image_diff.find("image_name\tstatus\tref_image_id"),
            std::string::npos);
  EXPECT_NE(image_diff.find("image3.jpg\tcandidate_only"), std::string::npos);
  EXPECT_NE(pair_diff.find("image1.jpg\timage2.jpg\t1\t2\t1"),
            std::string::npos);
  EXPECT_NE(point_diff.find("1\tcommon\t0.1"), std::string::npos);
  EXPECT_NE(point_diff.find("2\treference_only"), std::string::npos);
  EXPECT_NE(point_diff.find("3\tcandidate_only"), std::string::npos);
  EXPECT_NE(pose_diff.find("image1.jpg\t1\t1\t0\t0.1\t0.1"),
            std::string::npos);
  EXPECT_NE(summary.find("common_pairs 1"), std::string::npos);
  EXPECT_NE(summary.find("candidate_only_images 1"), std::string::npos);
}

TEST(ColiteCommandExeTest, DispatchesCompareColmapHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "compare_colmap", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("--reference_path"), std::string::npos);
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
