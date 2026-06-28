#include "colite/exe/recon_mvs.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace colite::exe {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

void WriteSmallColmapTextModel(const std::filesystem::path& model_path) {
  std::filesystem::create_directories(model_path);
  {
    std::ofstream cameras(model_path / "cameras.txt");
    cameras << "# Camera list\n"
            << "1 PINHOLE 100 80 50 50 50 40\n";
  }
  {
    std::ofstream images(model_path / "images.txt");
    images << "# Image list\n"
           << "1 1 0 0 0 0 0 0 1 image1.jpg\n"
           << "50 40 1 52.5 40 2\n"
           << "2 1 0 0 0 0 0 0 1 image2.jpg\n"
           << "50 40 1 52.5 40 2\n";
  }
  {
    std::ofstream points(model_path / "points3D.txt");
    points << "# 3D point list\n"
           << "1 0 0 2 255 255 255 0 1 0 2 0\n"
           << "2 0.2 0 4 255 255 255 0 1 1 2 1\n";
  }
}

void WriteSmallImages(const std::filesystem::path& image_path) {
  std::filesystem::create_directories(image_path);
  cv::Mat image1(5, 5, CV_8UC1);
  for (int row = 0; row < image1.rows; ++row) {
    for (int col = 0; col < image1.cols; ++col) {
      image1.at<std::uint8_t>(row, col) =
          static_cast<std::uint8_t>(row * 20 + col * 10);
    }
  }
  const cv::Mat image2 = image1.clone();
  ASSERT_TRUE(cv::imwrite((image_path / "image1.jpg").string(), image1));
  ASSERT_TRUE(cv::imwrite((image_path / "image2.jpg").string(), image2));
}

TEST(ReconMvsExeTest, PrintsHelp) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconMvs({"recon_mvs", "--help"}, out, err);

  EXPECT_EQ(exit_code, 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("--model_path"), std::string::npos);
  EXPECT_NE(out.str().find("default 8"), std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_min_num_valid_sources"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_max_num_aggregated_costs"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_geometric_refinement_iterations"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_geometric_consistency_weight"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_propagation_schedule"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_pyramid_levels"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_perturbation_samples"),
            std::string::npos);
  EXPECT_NE(out.str().find("--refine_normals_from_depth"),
            std::string::npos);
  EXPECT_NE(out.str().find("--normal_refinement_max_relative_depth_difference"),
            std::string::npos);
  EXPECT_NE(out.str().find("--source_min_triangulation_angle"),
            std::string::npos);
  EXPECT_NE(out.str().find("--source_triangulation_angle_weight"),
            std::string::npos);
  EXPECT_NE(out.str().find("--source_scale_consistency_weight"),
            std::string::npos);
  EXPECT_NE(out.str().find("--source_coverage_weight"), std::string::npos);
  EXPECT_NE(out.str().find("--source_coverage_grid_size"), std::string::npos);
  EXPECT_NE(out.str().find("--dense_map_format"), std::string::npos);
  EXPECT_NE(out.str().find("--num_threads"), std::string::npos);
  EXPECT_NE(out.str().find("--filter_dense_cloud"), std::string::npos);
  EXPECT_NE(out.str().find("--mesh_dense_cloud"), std::string::npos);
  EXPECT_NE(out.str().find("--fusion_source_pixel_search_radius"),
            std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconMvsExeTest, RejectsInvalidPatchMatchPropagationSchedule) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconMvs(
      {"recon_mvs",
       "--model_path",
       "model",
       "--output_path",
       "mvs",
       "--patch_match_propagation_schedule",
       "diagonal"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("Invalid PatchMatch propagation schedule"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidThreadCount) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconMvs(
      {"recon_mvs",
       "--model_path",
       "model",
       "--output_path",
       "mvs",
       "--num_threads",
       "0"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("--num_threads must be positive"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RejectsInvalidNormalRefinementDepthThreshold) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconMvs(
      {"recon_mvs",
       "--model_path",
       "model",
       "--output_path",
       "mvs",
       "--normal_refinement_max_relative_depth_difference",
       "-0.1"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find(
                "--normal_refinement_max_relative_depth_difference must be "
                "non-negative"),
            std::string::npos);
}

TEST(ReconMvsExeTest, RunsSparseWorkspacePassFromColmapTextModel) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_exe_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--output_path",
                      output_path.string()},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Sparse depth samples: 4"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("num_sparse_depth_samples 4"),
            std::string::npos);
  const std::string source_images = ReadTextFile(output_path / "source_images.tsv");
  EXPECT_NE(source_images.find("target_image_id\tsource_image_id"
                               "\tnum_shared_points"
                               "\tmean_triangulation_angle_degrees"
                               "\tmean_scale_ratio\tcoverage_score\tscore"),
            std::string::npos);
  EXPECT_NE(source_images.find("1\t2\t2"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "sparse_depth_samples.tsv")
                .find("1\t1\t50\t40\t2"),
            std::string::npos);
}

TEST(ReconMvsExeTest, InitializesPatchMatchMapsWhenImagePathIsProvided) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_init_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--initialize_patch_match"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Initialized PatchMatch images: 2"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("initialized_patch_match_images 2"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.init.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.init.txt"));
}

TEST(ReconMvsExeTest, RunsPatchMatchStereoWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_pm_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_pyramid_levels",
                      "2",
                      "--patch_match_window_radius",
                      "1",
                      "--patch_match_perturbation_samples",
                      "2",
                      "--filter_photometric_max_cost",
                      "0.5",
                      "--patch_match_min_num_valid_sources",
                      "1",
                      "--patch_match_max_num_aggregated_costs",
                      "2",
                      "--patch_match_propagation_schedule",
                      "four_direction",
                      "--patch_match_geometric_refinement_iterations",
                      "1",
                      "--patch_match_geometric_consistency_weight",
                      "0.2",
                      "--patch_match_geometric_max_relative_depth_error",
                      "0.01",
                      "--refine_normals_from_depth",
                      "--normal_refinement_max_relative_depth_difference",
                      "1.0"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("PatchMatch images: 2"), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_images 2"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("photometric_filtered_pixels "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "cost_maps" /
                                      "image1.jpg.txt"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "cost_maps" /
                         "image1.jpg.txt")
                .find("COLITE_DENSE_COST_MAP_TEXT"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("\t25\t9\t16\t"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("valid_cost_pixels_before_filter"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("pyramid_levels"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("mean_cost_before_filter"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("top_down_sweeps"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("checkerboard_passes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("perturbation_mean_abs_depth_update"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("perturbation_geometric_candidate_rejections"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("normal_refined_pixels"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("normal_refined_pixels "),
            std::string::npos);
}

TEST(ReconMvsExeTest, RunsPatchMatchStereoWithTwoThreadsWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_pm_threads_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_window_radius",
                      "1",
                      "--num_threads",
                      "2"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("PatchMatch images: 2"), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "patch_match.tsv"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("image_id\tpyramid_levels\titerations\tvalid_cost_pixels"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesGeometricConsistencyGraphWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_geom_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_window_radius",
                      "1",
                      "--check_geometric_consistency",
                      "--filter_geometric_min_num_consistent_sources",
                      "1"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Geometric consistent pixels: "), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("geometric_consistent_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("geometric_filtered_pixels "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "consistency_graphs" /
                                      "image1.jpg.txt"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "consistency_graphs" /
                         "image1.jpg.txt")
                .find("COLITE_CONSISTENCY_GRAPH_TEXT"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("geometric_consistent_pixels"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesBinaryDenseMapsWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_binary_maps_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_window_radius",
                      "1",
                      "--check_geometric_consistency",
                      "--filter_geometric_min_num_consistent_sources",
                      "1",
                      "--dense_map_format",
                      "binary"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "normal_maps" /
                                      "image1.jpg.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "consistency_graphs" /
                                      "image1.jpg.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "cost_maps" /
                                      "image1.jpg.txt"));
}

TEST(ReconMvsExeTest, WritesStereoFusionPlyWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_fusion_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_window_radius",
                      "1",
                      "--fuse_stereo",
                      "--fusion_min_num_observations",
                      "1",
                      "--fusion_max_normal_error",
                      "10",
                      "--fusion_max_view_angle",
                      "180",
                      "--filter_geometric_min_num_consistent_sources",
                      "1"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_NE(out.str().find("Fused points: "), std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt").find("fused_points "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt").find("fusion_tracks "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fused_observations "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_visited_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_rejected_normal_observations "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("fusion_rejected_view_observations "),
            std::string::npos);
  const std::filesystem::path ply_path =
      output_path / "dense" / "fused.ply";
  ASSERT_TRUE(std::filesystem::exists(ply_path));
  const std::string ply = ReadTextFile(ply_path);
  EXPECT_NE(ply.find("element vertex "), std::string::npos);
  EXPECT_NE(ply.find("property float nx"), std::string::npos);
  EXPECT_NE(ply.find("property uchar red"), std::string::npos);
  const std::filesystem::path diagnostics_path =
      output_path / "dense" / "fusion_tracks.tsv";
  ASSERT_TRUE(std::filesystem::exists(diagnostics_path));
  const std::string diagnostics = ReadTextFile(diagnostics_path);
  EXPECT_NE(diagnostics.find("point_idx\tseed_image_id\tseed_row\tseed_col"),
            std::string::npos);
  EXPECT_NE(diagnostics.find("max_normal_angle_degrees"), std::string::npos);
  EXPECT_NE(diagnostics.find("max_view_angle_degrees"), std::string::npos);

  const std::filesystem::path pair_diagnostics_path =
      output_path / "dense" / "fusion_image_pairs.tsv";
  ASSERT_TRUE(std::filesystem::exists(pair_diagnostics_path));
  const std::string pair_diagnostics = ReadTextFile(pair_diagnostics_path);
  EXPECT_NE(pair_diagnostics.find("reference_image_id\tsource_image_id"
                                  "\tnum_candidates\tnum_accepted"),
            std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_normal"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_view"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_projection"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_depth"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_duplicate"), std::string::npos);
  EXPECT_NE(pair_diagnostics.find("num_rejected_visited"), std::string::npos);

  const std::filesystem::path track_lengths_path =
      output_path / "dense" / "fusion_track_lengths.tsv";
  ASSERT_TRUE(std::filesystem::exists(track_lengths_path));
  const std::string track_lengths = ReadTextFile(track_lengths_path);
  EXPECT_NE(track_lengths.find("num_observations\tnum_tracks"),
            std::string::npos);
  const std::filesystem::path fusion_summary_path =
      output_path / "dense" / "fusion_summary.txt";
  ASSERT_TRUE(std::filesystem::exists(fusion_summary_path));
  const std::string fusion_summary = ReadTextFile(fusion_summary_path);
  EXPECT_NE(fusion_summary.find("num_single_observation_tracks "),
            std::string::npos);
  EXPECT_NE(fusion_summary.find("multi_observation_track_ratio "),
            std::string::npos);

  const std::filesystem::path error_distribution_path =
      output_path / "dense" / "fusion_error_distribution.tsv";
  ASSERT_TRUE(std::filesystem::exists(error_distribution_path));
  const std::string error_distribution =
      ReadTextFile(error_distribution_path);
  EXPECT_NE(error_distribution.find("metric\trecord\tlabel\tlower\tupper"
                                    "\tcount\tvalue"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("relative_depth_error\thistogram"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("reprojection_error_pixels\tpercentile"
                                    "\tp95"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("normal_angle_degrees\tpercentile\tp95"),
            std::string::npos);
  EXPECT_NE(error_distribution.find("view_angle_degrees\tpercentile\tp95"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesCleanDenseCloudAndMeshWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_clean_mesh_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code =
      RunReconMvs({"recon_mvs",
                      "--model_path",
                      model_path.string(),
                      "--image_path",
                      image_path.string(),
                      "--output_path",
                      output_path.string(),
                      "--run_patch_match",
                      "--patch_match_iterations",
                      "1",
                      "--patch_match_window_radius",
                      "1",
                      "--fuse_stereo",
                      "--fusion_min_num_observations",
                      "1",
                      "--filter_geometric_min_num_consistent_sources",
                      "1",
                      "--filter_dense_cloud",
                      "--dense_cloud_min_track_observations",
                      "1",
                      "--dense_cloud_voxel_size",
                      "0",
                      "--mesh_dense_cloud",
                      "--mesh_max_edge_length",
                      "10",
                      "--mesh_max_normal_angle",
                      "180"},
                     out,
                     err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "fused_clean.ply"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "fused_clean_stats.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" / "mesh.ply"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "mesh_stats.txt"));
  EXPECT_NE(ReadTextFile(output_path / "dense" / "fused_clean_stats.txt")
                .find("num_filtered_points "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "dense" / "mesh.ply")
                .find("element face "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "dense" / "mesh_stats.txt")
                .find("num_triangles "),
            std::string::npos);
}

}  // namespace
}  // namespace colite::exe
