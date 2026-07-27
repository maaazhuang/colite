#include "colite/exe/recon_mvs.h"
#include "colite/exe/mvs_stages.h"

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
  EXPECT_NE(out.str().find("default checkerboard"), std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_backend"), std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_gpu_index"), std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_pyramid_levels"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_perturbation_samples"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_directional_hmm"),
            std::string::npos);
  EXPECT_NE(out.str().find("--patch_match_source_samples"),
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
  EXPECT_NE(out.str().find("--poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("--delaunay_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("--mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("--mesh_texturer"), std::string::npos);
  EXPECT_NE(out.str().find("--image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("--fusion_source_pixel_search_radius"),
            std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ReconMvsExeTest, RejectsInvalidPatchMatchBackend) {
  std::ostringstream out;
  std::ostringstream err;

  const int exit_code = RunReconMvs(
      {"recon_mvs",
       "--model_path",
       "model",
       "--output_path",
       "mvs",
       "--patch_match_backend",
       "metal"},
      out,
      err);

  EXPECT_EQ(exit_code, 1);
  EXPECT_NE(err.str().find("Invalid PatchMatch backend"), std::string::npos);
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
  std::filesystem::remove_all(root);
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
                      "--resolution_level",
                      "0",
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
                      "--resolution_level",
                      "0",
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
                .find("requested_backend"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("actual_backend"),
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
                .find("source_selection_tests"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("iteration_candidate_evaluations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("iteration_source_selection_acceptances"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("normal_refined_pixels"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_wrapper_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_request_validation_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pyramid_resize_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_single_scale_dispatch_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_result_materialization_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_output_to_vector_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_dense_map_fill_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_normal_refinement_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_valid_cost_count_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_host_initial_normal_upload_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_host_output_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_device_count_queries"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_device_count_cache_hits"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_evaluate_cost_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_texture_reuses"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_buffer_reuses"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_iteration_counter_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_output_map_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_host_uploads"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_texture_uploads"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_async_output_readbacks"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_host_upload_bytes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_texture_upload_bytes"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_pinned_host_staging_bytes"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "patch_match_initial.tsv"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_wrapper_total_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_host_pack_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_host_output_normalizations"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_result_materialization_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_output_to_vector_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_dense_map_fill_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_normal_refinement_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_valid_cost_count_time_ms"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_device_count_queries"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match_initial.tsv")
                .find("cuda_device_count_cache_hits"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_pool_size"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
                .find("cuda_workspace_pool_slot"),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("normal_refined_pixels "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_wall_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_sum_cuda_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_overlap_saved_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_refinement_overlap_ratio "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_deferred_photometric_stats_images "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("cuda_warmup_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_source_setup_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_stereo_run_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_stereo_wrapper_overhead_sum_time_ms "),
            std::string::npos);
  EXPECT_NE(ReadTextFile(output_path / "summary.txt")
                .find("patch_match_initial_postprocess_sum_time_ms "),
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
  const std::string summary = ReadTextFile(output_path / "summary.txt");
  EXPECT_NE(summary.find("patch_match_initial_wall_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_sum_cuda_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_overlap_saved_time_ms "),
            std::string::npos);
  EXPECT_NE(summary.find("patch_match_initial_overlap_ratio "),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" /
                                      "patch_match.tsv"));
  EXPECT_NE(ReadTextFile(output_path / "stereo" / "patch_match.tsv")
            .find("image_id\trequested_backend\tactual_backend"
                  "\tpyramid_levels\titerations\tbilateral_ncc"
                  "\tsigma_spatial"
                  "\tsigma_color\tdynamic_source_selection\tncc_sigma"
                  "\tprevious_selection_probability_weight"
                  "\tmin_selection_probability"
                  "\tmax_selection_probability\tvalid_cost_pixels"),
            std::string::npos);
}

TEST(ReconMvsExeTest, WritesGeometricConsistencyGraphWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_geom_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
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
                                      "image1.jpg.bin"));
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
  std::filesystem::remove_all(root);
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
  EXPECT_FALSE(std::filesystem::exists(output_path / "stereo" /
                                       "consistency_graphs_colmap" /
                                       "image1.jpg.bin"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "cost_maps" /
                                      "image1.jpg.bin"));
}

TEST(ReconMvsExeTest, WritesStereoFusionPlyWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_recon_mvs_fusion_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
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
                      "--fusion_diagnostics",
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
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" /
                                      "points_visibility.bin"));
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

TEST(ReconMvsExeTest, StandaloneFusionReusesPersistedPatchMatchMaps) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_standalone_fusion_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream patch_out;
  std::ostringstream err;
  ASSERT_EQ(RunPatchMatchStereo(
                {"patch_match_stereo", "--model_path", model_path.string(),
                 "--image_path", image_path.string(), "--output_path",
                 output_path.string(), "--patch_match_iterations", "1",
                 "--patch_match_window_radius", "1"},
                patch_out, err),
            0)
      << err.str();
  const auto depth_path = output_path / "stereo" / "depth_maps" /
                          "image1.jpg.txt";
  ASSERT_TRUE(std::filesystem::exists(depth_path));
  const auto depth_time = std::filesystem::last_write_time(depth_path);

  std::ostringstream fusion_out;
  err.str("");
  err.clear();
  EXPECT_EQ(RunStereoFusion(
                {"stereo_fusion", "--model_path", model_path.string(),
                 "--image_path", image_path.string(), "--output_path",
                 output_path.string(), "--fusion_min_num_observations", "1"},
                fusion_out, err),
            0)
      << err.str();
  EXPECT_EQ(std::filesystem::last_write_time(depth_path), depth_time);
  EXPECT_TRUE(std::filesystem::exists(output_path / "dense" / "fused.ply"));
  const std::filesystem::path timing_path =
      output_path / "dense" / "fusion_timing.txt";
  ASSERT_TRUE(std::filesystem::exists(timing_path));
  const std::string timing = ReadTextFile(timing_path);
  EXPECT_NE(timing.find("consistency_graph_loading_ms "), std::string::npos);
  EXPECT_NE(timing.find("fusion_core_ms "), std::string::npos);
  EXPECT_NE(timing.find("total_ms "), std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(
      output_path / ".colite" / "stages" / "patch_match_stereo.json"));
  EXPECT_TRUE(std::filesystem::exists(
      output_path / ".colite" / "stages" / "stereo_fusion.json"));

  std::ostringstream resume_out;
  err.str("");
  err.clear();
  EXPECT_EQ(RunStereoFusion(
                {"stereo_fusion", "--model_path", model_path.string(),
                 "--image_path", image_path.string(), "--output_path",
                 output_path.string(), "--fusion_min_num_observations", "1",
                 "--resume"},
                resume_out, err),
            0)
      << err.str();
  EXPECT_NE(resume_out.str().find("SKIP stereo_fusion"), std::string::npos);
}

TEST(ReconMvsExeTest, WritesUndistortedWorkspaceWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_undistorter_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
  WriteSmallColmapTextModel(model_path);
  WriteSmallImages(image_path);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = RunReconMvs(
      {"recon_mvs", "--model_path", model_path.string(), "--image_path",
       image_path.string(), "--output_path", output_path.string(),
       "--resolution_level", "0",
       "--image_undistorter", "--undistort_max_image_size", "64",
       "--initialize_patch_match"},
      out, err);

  EXPECT_EQ(exit_code, 0) << err.str();
  EXPECT_TRUE(std::filesystem::exists(output_path / "undistorted" / "images" /
                                      "image1.jpg"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "undistorted" / "images" /
                                      "image2.jpg"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "undistorted" / "sparse" /
                                      "cameras.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "undistorted" / "sparse" /
                                      "images.txt"));
  EXPECT_TRUE(std::filesystem::exists(output_path / "undistorted" / "sparse" /
                                      "points3D.txt"));
  EXPECT_NE(ReadTextFile(output_path / "undistorted" / "sparse" /
                         "cameras.txt")
                .find("PINHOLE 64 51"),
            std::string::npos);
  EXPECT_NE(out.str().find("Undistorted images: 2"), std::string::npos);
  EXPECT_NE(out.str().find("Initialized PatchMatch images: 2"),
            std::string::npos);
  EXPECT_TRUE(std::filesystem::exists(output_path / "stereo" / "depth_maps" /
                                      "image1.jpg.init.txt"));
  std::filesystem::remove_all(root);
}

TEST(ReconMvsExeTest, WritesCleanDenseCloudAndMeshWhenRequested) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recon_mvs_clean_mesh_test";
  const std::filesystem::path model_path = root / "model";
  const std::filesystem::path image_path = root / "images";
  const std::filesystem::path output_path = root / "mvs";
  std::filesystem::remove_all(root);
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
