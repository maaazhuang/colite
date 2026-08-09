#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>

namespace colite::controllers {

struct ReconMvsControllerOptions {
  std::filesystem::path model_path;
  std::filesystem::path image_path;
  std::filesystem::path output_path;
  std::size_t max_num_source_images = 8;
  std::size_t min_num_shared_points = 1;
  std::string source_selection_mode = "weighted";
  double source_min_triangulation_angle_degrees = 1.0;
  double source_max_triangulation_angle_degrees = 30.0;
  double source_triangulation_angle_weight = 1.0;
  double source_shared_points_weight = 1.0;
  double source_scale_consistency_weight = 2.0;
  double source_coverage_weight = 5.0;
  std::size_t source_coverage_grid_size = 4;
  bool initialize_patch_match = false;
  bool run_patch_match = false;
  std::string patch_match_backend = "auto";
  int patch_match_gpu_index = -1;
  std::size_t patch_match_iterations = 3;
  std::size_t patch_match_pyramid_levels = 1;
  int patch_match_window_radius = 2;
  bool patch_match_randomize_initial_normals = true;
  float patch_match_initial_normal_hemisphere_degrees = 90.0F;
  float patch_match_normal_perturbation_degrees = 57.2957795F;
  bool patch_match_bilateral_ncc = false;
  float patch_match_sigma_spatial = -1.0F;
  float patch_match_sigma_color = 0.2F;
  bool patch_match_dynamic_source_selection = false;
  bool patch_match_directional_hmm = false;
  bool patch_match_colmap_source_priors = true;
  std::size_t patch_match_source_samples = 0;
  float patch_match_ncc_sigma = 0.6F;
  float patch_match_previous_selection_probability_weight = 0.5F;
  float patch_match_min_selection_probability = 0.05F;
  float patch_match_max_selection_probability = 0.999F;
  std::size_t patch_match_perturbation_samples = 1;
  std::size_t patch_match_min_num_valid_sources = 1;
  std::size_t patch_match_max_num_aggregated_costs = 3;
  std::string patch_match_propagation_schedule = "checkerboard";
  std::size_t patch_match_checkerboard_rounds = 1;
  bool patch_match_diagnostics = false;
  std::size_t patch_match_geometric_refinement_iterations = 0;
    float patch_match_geometric_consistency_weight = 0.0F;
    bool patch_match_robust_geometric_aggregation = true;
  float patch_match_geometric_max_relative_depth_error = 0.01F;
  bool refine_normals_from_depth = false;
  float normal_refinement_max_relative_depth_difference = 0.005F;
  float filter_photometric_max_cost = -1.0F;
  bool check_geometric_consistency = false;
  std::size_t filter_geometric_min_num_consistent_sources = 0;
  bool fuse_stereo = false;
  std::size_t fusion_max_num_source_images = 50;
  std::size_t fusion_min_num_observations = 1;
  std::size_t fusion_max_num_observations = 100000;
  std::size_t fusion_max_traversal_depth = 100;
  float fusion_max_reprojection_error_pixels = 2.0F;
  float fusion_max_relative_depth_error = 0.01F;
  float fusion_max_normal_error_degrees = 10.0F;
  float fusion_max_view_angle_degrees = 180.0F;
  std::size_t fusion_source_pixel_search_radius = 0;
  bool fusion_diagnostics = false;
  bool filter_dense_cloud = false;
  std::size_t dense_cloud_min_track_observations = 2;
  double dense_cloud_voxel_size = 0.0;
  double dense_cloud_spatial_bounds_quantile = 0.0;
  bool mesh_dense_cloud = false;
  double mesh_max_edge_length = 0.0;
  double mesh_max_normal_angle_degrees = 30.0;
  bool poisson_mesher = false;
  double poisson_point_weight = 1.0;
  int poisson_depth = 13;
  bool poisson_color = true;
  double poisson_trim = 10.0;
  bool delaunay_mesher = false;
  double delaunay_max_proj_dist = 20.0;
  double delaunay_max_depth_dist = 0.05;
  double delaunay_visibility_sigma = 3.0;
  double delaunay_distance_sigma_factor = 1.0;
  double delaunay_quality_regularization = 1.0;
  double delaunay_max_side_length_factor = 25.0;
  double delaunay_max_side_length_percentile = 95.0;
  bool mesh_simplifier = false;
  double mesh_simplification_ratio = 0.5;
  std::size_t mesh_simplification_target_triangles = 0;
  double mesh_simplification_max_error = 0.0;
  bool mesh_simplification_preserve_boundary = true;
  double mesh_simplification_max_normal_change_degrees = 60.0;
  bool mesh_texturer = false;
  int texture_atlas_size = 4096;
  int texture_max_atlas_size = 32768;
  int texture_patch_padding = 2;
  int texture_inpaint_radius = 5;
  double texture_scale = 1.0;
  double texture_min_cos_normal_angle = 0.1;
  int texture_min_visible_vertices = 3;
  int texture_view_smoothing_iterations = 3;
  int texture_view_min_component_size = 64;
  int texture_visibility_resolution = 1024;
  double texture_occlusion_relative_tolerance = 0.01;
  bool texture_use_vertex_color_fallback = false;
  bool texture_apply_color_correction = true;
  double texture_color_correction_regularization = 0.1;
  float geometric_max_reprojection_error_pixels = 1.0F;
  float geometric_max_relative_depth_error = 0.0F;
  std::string dense_map_format = "text";
  std::size_t num_threads = 1;
  std::size_t image_cache_size = 16;
  // Level 1 (half width and height) is the default MVS working resolution.
  std::size_t resolution_level = 1;
  int max_image_size = -1;
  std::uint32_t random_seed = 42;
  bool image_undistorter = false;
  double undistort_blank_pixels = 0.0;
  double undistort_min_scale = 0.2;
  double undistort_max_scale = 2.0;
  int undistort_max_image_size = -1;
};

class ReconMvsController {
 public:
  explicit ReconMvsController(ReconMvsControllerOptions options);

  const ReconMvsControllerOptions& Options() const;

  int Run(std::ostream& out) const;

 private:
  ReconMvsControllerOptions options_;
};

}  // namespace colite::controllers
