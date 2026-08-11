#include "colite/exe/recon_sfm.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/core/pose.h"
#include "colite/io/camera_prior.h"

namespace colite {

namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

FeatureSet MakeRatioSensitiveFeatures1() {
  FeatureSet features;
  features.descriptors.resize(1, 128);
  features.descriptors.setZero();
  for (int col = 0; col < 16; ++col) {
    features.descriptors(0, col) = 128.0F;
  }
  return features;
}

FeatureSet MakeRatioSensitiveFeatures2() {
  FeatureSet features;
  features.descriptors.resize(2, 128);
  features.descriptors.setZero();
  for (int col = 0; col < 16; ++col) {
    features.descriptors(0, col) = 128.0F;
    features.descriptors(1, col) = 128.0F;
  }
  // Build a ratio-sensitive pair: rejected at 0.8, accepted at 0.9.
  features.descriptors(0, 15) = 120.0F;
  features.descriptors(1, 15) = 116.0F;
  return features;
}

TEST(ReconSfmExeTest, PrintsHelp) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = exe::RunReconSfm({"recon_sfm", "--help"}, out, err);

  EXPECT_EQ(exit_code, 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("--feature_extractor"), std::string::npos);
  EXPECT_NE(out.str().find("--feature_type"), std::string::npos);
  EXPECT_NE(out.str().find("aliked_lightglue"), std::string::npos);
  EXPECT_NE(out.str().find("--mapper"), std::string::npos);
  EXPECT_NE(out.str().find("hierarchical"), std::string::npos);
  EXPECT_NE(out.str().find("--feature_num_threads"), std::string::npos);
  EXPECT_NE(out.str().find("--camera_mode"), std::string::npos);
  EXPECT_NE(out.str().find("--camera_model"), std::string::npos);
  EXPECT_NE(out.str().find("--camera_params"), std::string::npos);
  EXPECT_NE(out.str().find("--default_focal_length_factor"),
            std::string::npos);
  EXPECT_NE(out.str().find("--ba_refine_focal_length"), std::string::npos);
  EXPECT_NE(out.str().find("--ba_refine_principal_point"), std::string::npos);
  EXPECT_NE(out.str().find("--ba_refine_extra_params"), std::string::npos);
  EXPECT_NE(out.str().find("--ba_refine_params"), std::string::npos);
  EXPECT_NE(out.str().find("--max_registration_trials_per_image"),
            std::string::npos);
  EXPECT_NE(out.str().find("--max_initialization_trials"), std::string::npos);
  EXPECT_NE(out.str().find("--registration_retry_error_multiplier"),
            std::string::npos);
  EXPECT_NE(out.str().find("--registration_retry_max_error"),
            std::string::npos);
  EXPECT_NE(out.str().find("cuda_opencv"), std::string::npos);
  EXPECT_NE(out.str().find("cuda_vlfeat"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconSfmExeTest, PrintsHelpThroughExeLayer) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code =
      colite::exe::RunReconSfm({"recon_sfm", "--help"}, out, err);

  EXPECT_EQ(exit_code, 0);
  EXPECT_NE(out.str().find("colite recon_sfm"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconSfmExeTest, ComputesEffectiveFeatureExtractionThreads) {
  EXPECT_EQ(controllers::EffectiveFeatureExtractionThreads(1), 1);
  EXPECT_EQ(controllers::EffectiveFeatureExtractionThreads(2), 2);
  EXPECT_GT(controllers::EffectiveFeatureExtractionThreads(0), 0);
  EXPECT_GT(controllers::EffectiveFeatureExtractionThreads(-1), 0);
}

TEST(ReconSfmControllerTest, ComputesEffectiveFeatureExtractionThreads) {
  EXPECT_EQ(colite::controllers::EffectiveFeatureExtractionThreads(1), 1);
  EXPECT_EQ(colite::controllers::EffectiveFeatureExtractionThreads(2), 2);
  EXPECT_GT(colite::controllers::EffectiveFeatureExtractionThreads(0), 0);
  EXPECT_GT(colite::controllers::EffectiveFeatureExtractionThreads(-1), 0);
}

TEST(ReconSfmExeTest, RejectsMissingRequiredArguments) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code =
      exe::RunReconSfm({"recon_sfm", "--image_path", "images"}, out, err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("--output_path"), std::string::npos);
}

TEST(ReconSfmExeTest, RejectsInvalidRegistrationTrialLimit) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = exe::RunReconSfm(
      {"recon_sfm", "--image_path", "images", "--output_path", "out",
       "--max_registration_trials_per_image", "0"},
      out, err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("--max_registration_trials_per_image"),
            std::string::npos);
}

TEST(ReconSfmExeTest, RejectsIncompatibleLearnedFrontendTypes) {
  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunReconSfm(
      {"recon_sfm", "--image_path", "images", "--output_path", "out",
       "--feature_type", "aliked_n16rot", "--aliked_model_path", "a.onnx",
       "--matcher", "sift_lightglue", "--learned_matcher_model_path",
       "lightglue.onnx"},
      out, err);
  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("incompatible"), std::string::npos);
}

TEST(ReconSfmExeTest, RejectsUnknownMapperType) {
  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunReconSfm(
      {"recon_sfm", "--image_path", "images", "--output_path", "out",
       "--mapper", "unknown"},
      out, err);
  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("Unsupported mapper"), std::string::npos);
}

TEST(ReconSfmExeTest, RejectsInvalidHierarchicalClusterLimits) {
  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunReconSfm(
      {"recon_sfm", "--image_path", "images", "--output_path", "out",
       "--mapper", "hierarchical", "--hierarchical_target_cluster_size",
       "10", "--hierarchical_max_cluster_size", "5"},
      out, err);
  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("cluster size"), std::string::npos);
}

TEST(ReconSfmExeTest, RejectsInvalidRegistrationRetryOptions) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = exe::RunReconSfm(
      {"recon_sfm", "--image_path", "images", "--output_path", "out",
       "--registration_retry_error_multiplier", "0.5"},
      out, err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("--registration_retry_error_multiplier"),
            std::string::npos);
}

TEST(ReconSfmExeTest, EstimatesInitialFocalLengthFromJpegExif) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(image_path)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }

  const double focal_length =
      EstimateInitialFocalLengthFromImage(image_path, 0.0);

  EXPECT_NEAR(focal_length, 2484.86249133349, 1e-6);
}

TEST(ReconSfmExeTest, WritesColmapTextModel) {
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
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);
  reconstruction.GetPoint3D(point3d_id).color = {12, 34, 56};

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_colmap_export_test";
  controllers::WriteColmapTextModel(output_path, reconstruction);

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
  EXPECT_NE(points.find("1 0 0 5 12 34 56"), std::string::npos);
  EXPECT_NE(points.find("1 0 2 0"), std::string::npos);
}

TEST(ReconSfmExeTest, WritesQualitySummaryAndPlyPointCloud) {
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
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);
  reconstruction.GetPoint3D(point3d_id).color = {12, 34, 56};

  const controllers::ReconstructionQualityStats stats =
      controllers::ComputeReconstructionQualityStats(reconstruction);
  EXPECT_EQ(stats.num_observations, 2U);
  EXPECT_NEAR(stats.mean_reprojection_error, 0.0, 1e-12);
  EXPECT_NEAR(stats.max_reprojection_error, 0.0, 1e-12);

  IncrementalMappingSummary summary;
  summary.success = true;
  summary.num_registered_images = 2;
  summary.num_points3d = 1;

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_summary_ply_test";
  controllers::ReconSfmRunStats run_stats;
  run_stats.feature_backend = "cuda_vlfeat";
  run_stats.matcher_backend = "bf";
  run_stats.cuda_sift_available = true;
  run_stats.cpu_threads = 8;
  run_stats.feature_extraction_seconds = 1.25;
  run_stats.matching_seconds = 2.5;
  controllers::WriteRunSummary(output_path, reconstruction, summary,
                               run_stats);
  controllers::WritePlyPointCloud(output_path / "points3D.ply", reconstruction);

  const std::string summary_text = ReadTextFile(output_path / "summary.txt");
  const std::string ply_text = ReadTextFile(output_path / "points3D.ply");

  EXPECT_NE(summary_text.find("num_observations 2"), std::string::npos);
  EXPECT_NE(summary_text.find("feature_extraction_seconds 1.25"),
            std::string::npos);
  EXPECT_NE(summary_text.find("feature_backend cuda_vlfeat"),
            std::string::npos);
  EXPECT_NE(summary_text.find("cuda_sift_available 1"), std::string::npos);
  EXPECT_NE(summary_text.find("cpu_threads 8"), std::string::npos);
  EXPECT_NE(summary_text.find("matching_seconds 2.5"), std::string::npos);
  EXPECT_NE(summary_text.find("mean_reprojection_error 0"), std::string::npos);
  EXPECT_NE(summary_text.find("max_reprojection_error 0"), std::string::npos);
  EXPECT_NE(ply_text.find("ply"), std::string::npos);
  EXPECT_NE(ply_text.find("element vertex 1"), std::string::npos);
  EXPECT_NE(ply_text.find("0 0 5 12 34 56"), std::string::npos);
}

TEST(ReconSfmExeTest, ExtractsMeanRgbFromAllTrackObservations) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_sparse_rgb_test";
  std::filesystem::create_directories(root);
  ASSERT_TRUE(cv::imwrite((root / "image1.png").string(),
                          cv::Mat(2, 2, CV_8UC3, cv::Scalar(30, 20, 10))));
  ASSERT_TRUE(cv::imwrite((root / "image2.png").string(),
                          cv::Mat(2, 2, CV_8UC3, cv::Scalar(50, 40, 30))));

  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 2;
  camera.height = 2;
  camera.params = {10.0, 10.0, 1.0, 1.0};
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 2; ++image_id) {
    Image image;
    image.id = image_id;
    image.camera_id = 1;
    image.name = "image" + std::to_string(image_id) + ".png";
    image.file_path = (root / image.name).string();
    image.registered = true;
    image.cam_from_world = Pose3{};
    image.points2d.push_back({{0.5, 0.5}, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
    reconstruction.RegisterImage(image_id);
  }
  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(Eigen::Vector3d(0.0, 0.0, 5.0), track);

  controllers::ExtractPointColors(&reconstruction);

  const auto& color = reconstruction.GetPoint3D(point3d_id).color;
  EXPECT_EQ(color.x(), 20);
  EXPECT_EQ(color.y(), 30);
  EXPECT_EQ(color.z(), 40);
}

TEST(ReconSfmExeTest, WritesPairDiagnosticsForVerifiedMatches) {
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
  controllers::WritePairDiagnostics(output_path, reconstruction, graph);

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

TEST(ReconSfmExeTest, WritesRegistrationDiagnosticsForMappingRounds) {
  IncrementalMappingSummary summary;
  summary.success = true;

  IncrementalMapperSummary registration;
  registration.success = true;
  registration.registration_order = 2;
  registration.image_id = 7;
  registration.candidate_num_visible_points = 128;
  registration.candidate_num_visible_observations = 256;
  registration.candidate_mean_track_length = 2.0;
  registration.candidate_num_registered_neighbors = 4;
  registration.candidate_num_verified_inliers_to_registered_images = 512;
  registration.candidate_score = 128500.0;
  registration.registration_attempts_before_success = 1;
  registration.registration_max_reprojection_error_px = 3.5;
  registration.registration.num_correspondences = 140;
  registration.registration.num_ransac_inliers = 118;
  registration.registration.num_inliers = 120;
  registration.track_completion.num_completed_observations = 11;
  registration.track_completion.num_merged_points = 2;
  registration.track_completion.num_merged_observations = 3;
  registration.triangulation.num_created_points = 23;
  registration.triangulation.num_created_observations = 46;
  registration.local_retriangulation.num_created_points = 5;
  registration.local_retriangulation.num_created_observations = 10;
  registration.local_retriangulation.num_merged_points = 1;
  registration.local_retriangulation.num_merged_observations = 2;
  registration.local_retriangulation.num_filtered_points = 2;
  registration.local_bundle_adjustment.local_image_ids = {7, 8, 9};
  registration.local_bundle_adjustment.fixed_image_ids = {3, 4};
  registration.local_bundle_adjustment.num_local_images = 3;
  registration.local_bundle_adjustment.num_constant_images = 2;
  registration.local_bundle_adjustment.num_variable_images = 6;
  registration.local_bundle_adjustment.num_variable_points = 80;
  registration.local_bundle_adjustment.num_constant_cameras = 1;
  registration.local_bundle_adjustment.num_variable_cameras = 0;
  registration.local_bundle_adjustment.num_residuals = 260;
  registration.local_bundle_adjustment.initial_mean_reprojection_error = 1.25;
  registration.local_bundle_adjustment.final_mean_reprojection_error = 0.75;
  registration.observation_filter.num_removed_observations = 3;
  registration.observation_filter.num_removed_points = 1;
  summary.registrations.push_back(registration);

  ImageRegistrationFailureSummary failure;
  failure.image_id = 9;
  failure.attempt = 2;
  failure.candidate_num_visible_points = 64;
  failure.candidate_num_visible_observations = 96;
  failure.candidate_mean_track_length = 1.5;
  failure.candidate_num_registered_neighbors = 2;
  failure.candidate_num_verified_inliers_to_registered_images = 180;
  failure.candidate_score = 64200.0;
  failure.registration_max_reprojection_error_px = 6.0;
  failure.registration_num_correspondences = 72;
  failure.registration_num_ransac_inliers = 10;
  failure.registration_num_inliers = 10;
  failure.message = "Not enough inliers";
  summary.failed_registrations.push_back(failure);

  const std::filesystem::path output_path =
      std::filesystem::temp_directory_path() / "colite_registration_diag_test";
  controllers::WriteRegistrationDiagnostics(output_path, summary);

  const std::string diagnostics =
      ReadTextFile(output_path / "registration_diagnostics.tsv");
  EXPECT_NE(diagnostics.find(
                "order\timage_id\tcandidate_visible_points"
                "\tcandidate_visible_observations\tcandidate_mean_track_length"
                "\tcandidate_registered_neighbors"),
            std::string::npos);
  EXPECT_NE(diagnostics.find(
                "2\t7\t128\t256\t2\t4\t512\t128500\t1\t3.5\t140\t118\t120\t11\t2"
                "\t3\t23\t46\t5\t10\t1\t2\t2\t3\t2\t2\t6\t1\t0\t80\t260"
                "\t1.25\t0.75\t3\t1"),
            std::string::npos);

  const std::string failures =
      ReadTextFile(output_path / "registration_failures.tsv");
  EXPECT_NE(failures.find(
                "image_id\tattempt\tcandidate_visible_points"
                "\tcandidate_visible_observations\tcandidate_mean_track_length"),
            std::string::npos);
  EXPECT_NE(failures.find("9\t2\t64\t96\t1.5\t2\t180\t64200\t6\t72\t10\t10\tNot enough inliers"),
            std::string::npos);
}

TEST(ReconSfmExeTest, BoostsOnlyNearbyPairMatchesWithRelaxedRatio) {
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeRatioSensitiveFeatures1());
  SetFeatures(&graph, 2, MakeRatioSensitiveFeatures2());
  SetFeatures(&graph, 4, MakeRatioSensitiveFeatures2());

  DescriptorMatchingOptions base_options;
  base_options.max_ratio = 0.8;
  base_options.mutual_check = true;

  controllers::NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = 1;
  nearby_options.max_ratio = 0.9;

  controllers::MatchAllImagePairs({1, 2, 4}, base_options, nearby_options, &graph);

  const std::vector<FeatureMatch>* nearby_matches = FindMatches(graph, 1, 2);
  const std::vector<FeatureMatch>* distant_matches = FindMatches(graph, 1, 4);
  ASSERT_NE(nearby_matches, nullptr);
  ASSERT_NE(distant_matches, nullptr);
  EXPECT_EQ(nearby_matches->size(), 1U);
  EXPECT_TRUE(distant_matches->empty());
}

}  // namespace
}  // namespace colite
