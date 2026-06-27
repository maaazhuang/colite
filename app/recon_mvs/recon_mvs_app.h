#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace colite::app {

struct ReconMvsCliOptions {
  bool show_help = false;
  bool show_version = false;
  std::string model_path;
  std::string image_path;
  std::string output_path;
  std::size_t max_num_source_images = 8;
  std::size_t min_num_shared_points = 1;
  double source_min_triangulation_angle_degrees = 1.0;
  double source_max_triangulation_angle_degrees = 30.0;
  double source_triangulation_angle_weight = 1.0;
  double source_shared_points_weight = 1.0;
  double source_scale_consistency_weight = 2.0;
  double source_coverage_weight = 5.0;
  std::size_t source_coverage_grid_size = 4;
  bool initialize_patch_match = false;
  bool run_patch_match = false;
  std::size_t patch_match_iterations = 3;
  std::size_t patch_match_pyramid_levels = 1;
  int patch_match_window_radius = 2;
  std::size_t patch_match_perturbation_samples = 1;
  std::size_t patch_match_min_num_valid_sources = 1;
  std::size_t patch_match_max_num_aggregated_costs = 3;
  std::string patch_match_propagation_schedule = "forward_backward";
  std::size_t patch_match_geometric_refinement_iterations = 0;
  float patch_match_geometric_consistency_weight = 0.0F;
  float patch_match_geometric_max_relative_depth_error = 0.01F;
  bool refine_normals_from_depth = false;
  float normal_refinement_max_relative_depth_difference = 0.005F;
  float filter_photometric_max_cost = -1.0F;
  bool check_geometric_consistency = false;
  std::size_t filter_geometric_min_num_consistent_sources = 0;
  bool fuse_stereo = false;
  std::size_t fusion_min_num_observations = 1;
  std::size_t fusion_max_num_observations = 100000;
  std::size_t fusion_max_traversal_depth = 100;
  float fusion_max_reprojection_error_pixels = 2.0F;
  float fusion_max_relative_depth_error = 0.01F;
  float fusion_max_normal_error_degrees = 10.0F;
  float fusion_max_view_angle_degrees = 180.0F;
  std::size_t fusion_source_pixel_search_radius = 1;
  bool filter_dense_cloud = false;
  std::size_t dense_cloud_min_track_observations = 2;
  double dense_cloud_voxel_size = 0.0;
  bool mesh_dense_cloud = false;
  double mesh_max_edge_length = 0.0;
  double mesh_max_normal_angle_degrees = 30.0;
  float geometric_max_reprojection_error_pixels = 2.0F;
  float geometric_max_relative_depth_error = 0.01F;
  std::string dense_map_format = "text";
  std::size_t num_threads = 1;
  std::size_t image_cache_size = 16;
  int max_image_size = -1;
  std::uint32_t random_seed = 42;
};

int RunReconMvsApp(const std::vector<std::string>& args,
                   std::ostream& out,
                   std::ostream& err);

}  // namespace colite::app
