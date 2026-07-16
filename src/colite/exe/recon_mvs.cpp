#include "colite/exe/recon_mvs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/point2d.h"
#include "colite/core/point3d.h"
#include "colite/core/reconstruction.h"
#include "colite/core/version.h"
#include "colite/controllers/recon_mvs_controller.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/patch_match.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite recon_mvs " << colite::Version() << "\n"
      << "\n"
      << "Usage:\n"
      << "  " << executable << " --help\n"
      << "  " << executable << " --version\n"
      << "  " << executable
      << " --model_path <colmap_text_dir> --output_path <dir> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --max_num_source_images <n>  Maximum source images per target, "
         "default 8.\n"
      << "  --min_num_shared_points <n> Minimum shared sparse points for a "
         "source image, default 1.\n"
      << "  --source_min_triangulation_angle <deg> Minimum source-view "
         "triangulation angle used for scoring, default 1.\n"
      << "  --source_max_triangulation_angle <deg> Maximum source-view "
         "triangulation angle used for scoring, default 30.\n"
      << "  --source_triangulation_angle_weight <w> Source-view score "
         "weight for mean triangulation angle, default 1.\n"
      << "  --source_shared_points_weight <w> Source-view score weight for "
         "shared sparse points, default 1.\n"
      << "  --source_scale_consistency_weight <w> Source-view score weight "
         "for sparse projection scale consistency, default 2.\n"
      << "  --source_coverage_weight <w> Source-view score weight for target "
         "image sparse-point coverage, default 5.\n"
      << "  --source_coverage_grid_size <n> Grid size used by source-view "
         "coverage scoring, default 4.\n"
      << "  --image_path <dir>          Directory containing input images.\n"
      << "  --image_undistorter         Write undistorted/images and sparse model before MVS.\n"
      << "  --undistort_blank_pixels <r> Allowed blank output fraction, default 0.\n"
      << "  --undistort_min_scale <v>   Minimum focal scale, default 0.2.\n"
      << "  --undistort_max_scale <v>   Maximum focal scale, default 2.\n"
      << "  --undistort_max_image_size <n> Maximum undistorted side, default -1.\n"
      << "  --dense_map_format <fmt>    Dense map output format: text or "
         "binary, default text.\n"
      << "  --initialize_patch_match    Initialize dense depth/normal maps.\n"
      << "  --run_patch_match           Run PatchMatch stereo for each "
         "registered image.\n"
      << "  --patch_match_backend <name> PatchMatch backend: auto, cpu, or "
         "cuda; default auto prefers CUDA when available.\n"
      << "  --patch_match_gpu_index <n> CUDA GPU index for PatchMatch, "
         "default -1 lets the backend choose.\n"
      << "  --patch_match_iterations <n> PatchMatch iterations, default 3.\n"
      << "  --patch_match_pyramid_levels <n> Coarse-to-fine PatchMatch "
         "pyramid levels, default 1.\n"
      << "  --patch_match_window_radius <n> ZNCC window radius, default 2.\n"
      << "  --patch_match_bilateral_ncc Enable COLMAP-style bilateral "
         "weighted NCC; disabled by default.\n"
      << "  --patch_match_sigma_spatial <s> Bilateral NCC spatial sigma; "
         "default follows window radius.\n"
      << "  --patch_match_sigma_color <s> Bilateral NCC grayscale sigma, "
         "default 0.2.\n"
      << "  --patch_match_dynamic_source_selection Enable iterative NCC "
         "likelihood source probabilities; disabled by default.\n"
      << "  --patch_match_directional_hmm Apply CUDA row/column "
         "forward-backward source messages after each NCC update.\n"
      << "  --patch_match_source_samples <n> Per-pixel source CDF samples; "
         "default 0 uses best-K; 15 enables the experimental COLMAP-style "
         "CDF path.\n"
      << "  --patch_match_ncc_sigma <s> NCC likelihood sigma, default 0.6.\n"
      << "  --patch_match_previous_selection_probability_weight <w> Previous "
         "source probability EMA weight, default 0.5.\n"
      << "  --patch_match_min_selection_probability <p> Dynamic source "
         "probability floor, default 0.05.\n"
      << "  --patch_match_max_selection_probability <p> Dynamic source "
         "probability ceiling, default 0.999.\n"
      << "  --patch_match_perturbation_samples <n> Random plane candidates "
         "per valid pixel during perturbation, default 1.\n"
      << "  --patch_match_min_num_valid_sources <n> Minimum valid source "
         "views per PatchMatch cost, default 1.\n"
      << "  --patch_match_max_num_aggregated_costs <n> Best source costs "
         "aggregated per PatchMatch hypothesis, default 3; 0 uses all.\n"
      << "  --patch_match_propagation_schedule <name> Propagation schedule: "
         "forward_backward, four_direction, or checkerboard; default "
         "checkerboard.\n"
      << "  --patch_match_geometric_refinement_iterations <n> PatchMatch "
         "second-pass geometric refinement iterations, default 0.\n"
      << "  --patch_match_geometric_consistency_weight <w> Source-depth "
         "geometric consistency cost weight for refinement, default 0.\n"
      << "  --patch_match_geometric_max_relative_depth_error <r> Source-depth "
         "relative error normalization threshold, default 0.01.\n"
      << "  --refine_normals_from_depth Refine PatchMatch normals from the "
         "final depth map before filtering/fusion.\n"
      << "  --normal_refinement_max_relative_depth_difference <r> Maximum "
         "relative neighbor-depth difference for normal refinement, default "
         "0.005.\n"
      << "  --filter_photometric_max_cost <c> Filter pixels above cost c, "
         "disabled by default.\n"
      << "  --check_geometric_consistency Check depth maps across source "
         "views.\n"
      << "  --filter_geometric_min_num_consistent_sources <n> Filter pixels "
         "with fewer than n consistent sources.\n"
      << "  --fuse_stereo             Fuse filtered stereo depth maps into "
         "dense/fused.ply.\n"
      << "  --fusion_min_num_observations <n> Minimum observations per fused "
         "track, default 1.\n"
      << "  --fusion_max_num_observations <n> Maximum observations per fused "
         "track, default 100000.\n"
      << "  --fusion_max_traversal_depth <n> Maximum consistency graph "
         "traversal depth, default 100.\n"
      << "  --fusion_max_reprojection_error <px> Fusion reprojection "
         "threshold, default 2.\n"
      << "  --fusion_max_relative_depth_error <r> Fusion relative depth "
         "threshold, default 0.01.\n"
      << "  --fusion_max_normal_error <deg> Fusion normal angle threshold, "
         "default 10.\n"
      << "  --fusion_max_view_angle <deg> Fusion view angle threshold, "
         "default 180.\n"
      << "  --fusion_source_pixel_search_radius <n> Source pixel search "
         "radius for fusion, default 1.\n"
      << "  --filter_dense_cloud      Write dense/fused_clean.ply after "
         "track and voxel filtering.\n"
      << "  --dense_cloud_min_track_observations <n> Minimum observations "
         "per clean dense point, default 2.\n"
      << "  --dense_cloud_voxel_size <v> Optional clean cloud voxel size, "
         "default 0.\n"
      << "  --mesh_dense_cloud        Write a prototype dense/mesh.ply from "
         "the clean cloud.\n"
      << "  --mesh_max_edge_length <v> Prototype mesh max edge length, "
         "default 0 disables edge limit.\n"
      << "  --mesh_max_normal_angle <deg> Prototype mesh normal angle "
         "threshold, default 30.\n"
      << "  --poisson_mesher          Reconstruct dense/meshed-poisson.ply "
         "with PoissonRecon.\n"
      << "  --poisson_point_weight <w> Screened Poisson point weight, default 1.\n"
      << "  --poisson_depth <n>       Maximum Poisson octree depth, default 13.\n"
      << "  --poisson_color <0|1>     Enable Poisson vertex colors, default 1.\n"
      << "  --poisson_trim <v>        Remove low-density mesh regions, default 10.\n"
      << "  --delaunay_mesher         Reconstruct dense/meshed-delaunay.ply "
         "using visibility-aware CGAL meshing.\n"
      << "  --delaunay_max_proj_dist <px> Reprojected subsampling radius, default 20.\n"
      << "  --delaunay_max_depth_dist <r> Relative subsampling depth tolerance, default 0.05.\n"
      << "  --delaunay_visibility_sigma <v> Visibility confidence sigma, default 3.\n"
      << "  --delaunay_distance_sigma_factor <v> Surface distance scale, default 1.\n"
      << "  --delaunay_quality_regularization <v> Graph-cut smoothness, default 1.\n"
      << "  --delaunay_max_side_length_factor <v> Outlier face factor, default 25.\n"
      << "  --delaunay_max_side_length_percentile <p> Face baseline percentile, default 95.\n"
      << "  --mesh_simplifier        Simplify each generated mesh with QEM edge collapses.\n"
      << "  --mesh_simplification_ratio <r> Target triangle ratio, default 0.5.\n"
      << "  --mesh_simplification_target_triangles <n> Explicit target; overrides ratio.\n"
      << "  --mesh_simplification_max_error <e> Optional quadric-error limit, default 0 disables.\n"
      << "  --mesh_simplification_allow_boundary_changes Allow collapses that alter boundaries.\n"
      << "  --mesh_simplification_max_normal_change <deg> Maximum local face rotation, default 60.\n"
      << "  --mesh_texturer          Bake generated meshes to OBJ/MTL/PNG texture atlases.\n"
      << "  --texture_atlas_size <n> Square atlas dimension, default 4096.\n"
      << "  --texture_max_atlas_size <n> Maximum growing atlas side, default 32768.\n"
      << "  --texture_patch_padding <n> Atlas region padding, default 2.\n"
      << "  --texture_inpaint_radius <n> Atlas dilation radius, default 5.\n"
      << "  --texture_scale <v>      Source projection scale, default 1.\n"
      << "  --texture_min_cos_normal_angle <v> Grazing-angle threshold, default 0.1.\n"
      << "  --texture_min_visible_vertices <n> Required projected vertices, default 3.\n"
      << "  --texture_view_smoothing_iterations <n> Neighbor label smoothing, default 3.\n"
      << "  --texture_visibility_resolution <n> Depth-buffer longest side, default 1024.\n"
      << "  --texture_occlusion_tolerance <r> Relative depth tolerance, default 0.01.\n"
      << "  --texture_color_correction <0|1> Apply global seam correction, default 1.\n"
      << "  --texture_color_correction_regularization <v> Correction regularizer, default 0.1.\n"
      << "  --geometric_max_reprojection_error <px> Geometric consistency "
         "forward-backward reprojection threshold, default 1.\n"
      << "  --geometric_max_relative_depth_error <r> Optional geometric "
         "relative-depth threshold; default 0 disables it to match COLMAP.\n"
      << "  --num_threads <n>        Number of PatchMatch worker threads, "
         "default 1.\n"
      << "  --image_cache_size <n>      LRU image cache size, default 16.\n"
      << "  --max_image_size <n>        Resize longest image side before MVS.\n"
      << "  --random_seed <n>           PatchMatch initialization seed, "
         "default 42.\n";
}

std::string RequireValue(const std::vector<std::string>& args,
                         std::size_t* index,
                         std::string_view flag) {
  if (*index + 1 >= args.size()) {
    throw std::invalid_argument("Missing value for " + std::string(flag));
  }
  ++(*index);
  return args[*index];
}

ReconMvsCliOptions ParseOptions(const std::vector<std::string>& args) {
  ReconMvsCliOptions options;
  if (args.size() <= 1) {
    options.show_help = true;
    return options;
  }

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
    } else if (arg == "--version") {
      options.show_version = true;
    } else if (arg == "--model_path") {
      options.model_path = RequireValue(args, &i, arg);
    } else if (arg == "--image_path") {
      options.image_path = RequireValue(args, &i, arg);
    } else if (arg == "--output_path") {
      options.output_path = RequireValue(args, &i, arg);
    } else if (arg == "--image_undistorter") {
      options.image_undistorter = true;
    } else if (arg == "--undistort_blank_pixels") {
      options.undistort_blank_pixels =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--undistort_min_scale") {
      options.undistort_min_scale = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--undistort_max_scale") {
      options.undistort_max_scale = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--undistort_max_image_size") {
      options.undistort_max_image_size =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--max_num_source_images") {
      options.max_num_source_images =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--min_num_shared_points") {
      options.min_num_shared_points =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--source_min_triangulation_angle") {
      options.source_min_triangulation_angle_degrees =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_max_triangulation_angle") {
      options.source_max_triangulation_angle_degrees =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_triangulation_angle_weight") {
      options.source_triangulation_angle_weight =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_shared_points_weight") {
      options.source_shared_points_weight =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_scale_consistency_weight") {
      options.source_scale_consistency_weight =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_coverage_weight") {
      options.source_coverage_weight =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--source_coverage_grid_size") {
      options.source_coverage_grid_size =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--initialize_patch_match") {
      options.initialize_patch_match = true;
    } else if (arg == "--run_patch_match") {
      options.run_patch_match = true;
    } else if (arg == "--patch_match_backend") {
      options.patch_match_backend = RequireValue(args, &i, arg);
    } else if (arg == "--patch_match_gpu_index") {
      options.patch_match_gpu_index =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_iterations") {
      options.patch_match_iterations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_pyramid_levels") {
      options.patch_match_pyramid_levels =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_window_radius") {
      options.patch_match_window_radius =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_bilateral_ncc") {
      options.patch_match_bilateral_ncc = true;
    } else if (arg == "--patch_match_sigma_spatial") {
      options.patch_match_sigma_spatial =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_sigma_color") {
      options.patch_match_sigma_color =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_dynamic_source_selection") {
      options.patch_match_dynamic_source_selection = true;
    } else if (arg == "--patch_match_directional_hmm") {
      options.patch_match_directional_hmm = true;
      options.patch_match_dynamic_source_selection = true;
    } else if (arg == "--patch_match_source_samples") {
      options.patch_match_source_samples =
          static_cast<std::size_t>(
              std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_ncc_sigma") {
      options.patch_match_ncc_sigma = std::stof(RequireValue(args, &i, arg));
    } else if (arg ==
               "--patch_match_previous_selection_probability_weight") {
      options.patch_match_previous_selection_probability_weight =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_min_selection_probability") {
      options.patch_match_min_selection_probability =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_max_selection_probability") {
      options.patch_match_max_selection_probability =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_perturbation_samples") {
      options.patch_match_perturbation_samples =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_min_num_valid_sources") {
      options.patch_match_min_num_valid_sources =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_max_num_aggregated_costs") {
      options.patch_match_max_num_aggregated_costs =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_propagation_schedule") {
      options.patch_match_propagation_schedule = RequireValue(args, &i, arg);
    } else if (arg == "--patch_match_geometric_refinement_iterations") {
      options.patch_match_geometric_refinement_iterations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_geometric_consistency_weight") {
      options.patch_match_geometric_consistency_weight =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--patch_match_geometric_max_relative_depth_error") {
      options.patch_match_geometric_max_relative_depth_error =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--refine_normals_from_depth") {
      options.refine_normals_from_depth = true;
    } else if (arg == "--normal_refinement_max_relative_depth_difference") {
      options.normal_refinement_max_relative_depth_difference =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--filter_photometric_max_cost") {
      options.filter_photometric_max_cost =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--check_geometric_consistency") {
      options.check_geometric_consistency = true;
    } else if (arg == "--filter_geometric_min_num_consistent_sources") {
      options.filter_geometric_min_num_consistent_sources =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      options.check_geometric_consistency = true;
    } else if (arg == "--fuse_stereo") {
      options.fuse_stereo = true;
      options.check_geometric_consistency = true;
      if (options.filter_geometric_min_num_consistent_sources == 0) {
        options.filter_geometric_min_num_consistent_sources = 1;
      }
    } else if (arg == "--fusion_min_num_observations") {
      options.fusion_min_num_observations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--fusion_max_num_observations") {
      options.fusion_max_num_observations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--fusion_max_traversal_depth") {
      options.fusion_max_traversal_depth =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--fusion_max_reprojection_error") {
      options.fusion_max_reprojection_error_pixels =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--fusion_max_relative_depth_error") {
      options.fusion_max_relative_depth_error =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--fusion_max_normal_error") {
      options.fusion_max_normal_error_degrees =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--fusion_max_view_angle") {
      options.fusion_max_view_angle_degrees =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--fusion_source_pixel_search_radius") {
      options.fusion_source_pixel_search_radius =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--filter_dense_cloud") {
      options.filter_dense_cloud = true;
    } else if (arg == "--dense_cloud_min_track_observations") {
      options.dense_cloud_min_track_observations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--dense_cloud_voxel_size") {
      options.dense_cloud_voxel_size = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_dense_cloud") {
      options.mesh_dense_cloud = true;
      options.filter_dense_cloud = true;
    } else if (arg == "--mesh_max_edge_length") {
      options.mesh_max_edge_length = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_max_normal_angle") {
      options.mesh_max_normal_angle_degrees =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--poisson_mesher") {
      options.poisson_mesher = true;
      options.fuse_stereo = true;
      options.check_geometric_consistency = true;
      if (options.filter_geometric_min_num_consistent_sources == 0) {
        options.filter_geometric_min_num_consistent_sources = 1;
      }
    } else if (arg == "--poisson_point_weight") {
      options.poisson_point_weight = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--poisson_depth") {
      options.poisson_depth = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--poisson_color") {
      options.poisson_color = std::stoi(RequireValue(args, &i, arg)) != 0;
    } else if (arg == "--poisson_trim") {
      options.poisson_trim = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_mesher") {
      options.delaunay_mesher = true;
      options.fuse_stereo = true;
      options.check_geometric_consistency = true;
      if (options.filter_geometric_min_num_consistent_sources == 0) {
        options.filter_geometric_min_num_consistent_sources = 1;
      }
    } else if (arg == "--delaunay_max_proj_dist") {
      options.delaunay_max_proj_dist = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_max_depth_dist") {
      options.delaunay_max_depth_dist = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_visibility_sigma") {
      options.delaunay_visibility_sigma = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_distance_sigma_factor") {
      options.delaunay_distance_sigma_factor =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_quality_regularization") {
      options.delaunay_quality_regularization =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_max_side_length_factor") {
      options.delaunay_max_side_length_factor =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--delaunay_max_side_length_percentile") {
      options.delaunay_max_side_length_percentile =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_simplifier") {
      options.mesh_simplifier = true;
    } else if (arg == "--mesh_simplification_ratio") {
      options.mesh_simplification_ratio =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_simplification_target_triangles") {
      options.mesh_simplification_target_triangles =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--mesh_simplification_max_error") {
      options.mesh_simplification_max_error =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_simplification_allow_boundary_changes") {
      options.mesh_simplification_preserve_boundary = false;
    } else if (arg == "--mesh_simplification_max_normal_change") {
      options.mesh_simplification_max_normal_change_degrees =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--mesh_texturer") {
      options.mesh_texturer = true;
    } else if (arg == "--texture_atlas_size") {
      options.texture_atlas_size = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_max_atlas_size") {
      options.texture_max_atlas_size = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_patch_padding") {
      options.texture_patch_padding = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_inpaint_radius") {
      options.texture_inpaint_radius = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_scale") {
      options.texture_scale = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--texture_min_cos_normal_angle") {
      options.texture_min_cos_normal_angle =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--texture_min_visible_vertices") {
      options.texture_min_visible_vertices =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_view_smoothing_iterations") {
      options.texture_view_smoothing_iterations =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_visibility_resolution") {
      options.texture_visibility_resolution =
          std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--texture_occlusion_tolerance") {
      options.texture_occlusion_relative_tolerance =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--texture_color_correction") {
      options.texture_apply_color_correction =
          std::stoi(RequireValue(args, &i, arg)) != 0;
    } else if (arg == "--texture_color_correction_regularization") {
      options.texture_color_correction_regularization =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--geometric_max_reprojection_error") {
      options.geometric_max_reprojection_error_pixels =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--geometric_max_relative_depth_error") {
      options.geometric_max_relative_depth_error =
          std::stof(RequireValue(args, &i, arg));
    } else if (arg == "--dense_map_format") {
      options.dense_map_format = RequireValue(args, &i, arg);
    } else if (arg == "--num_threads") {
      options.num_threads =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--image_cache_size") {
      options.image_cache_size =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--max_image_size") {
      options.max_image_size = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--random_seed") {
      options.random_seed =
          static_cast<std::uint32_t>(std::stoul(RequireValue(args, &i, arg)));
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  return options;
}

void ValidateOptions(const ReconMvsCliOptions& options) {
  if (options.show_help || options.show_version) {
    return;
  }
  if (options.model_path.empty()) {
    throw std::invalid_argument("Missing required argument --model_path");
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path");
  }
  if (options.min_num_shared_points == 0) {
    throw std::invalid_argument("--min_num_shared_points must be positive");
  }
  if (!std::isfinite(options.source_min_triangulation_angle_degrees) ||
      !std::isfinite(options.source_max_triangulation_angle_degrees) ||
      options.source_min_triangulation_angle_degrees < 0.0 ||
      options.source_max_triangulation_angle_degrees <
          options.source_min_triangulation_angle_degrees ||
      options.source_max_triangulation_angle_degrees > 180.0) {
    throw std::invalid_argument(
        "Source triangulation angle range must be within [0, 180]");
  }
  if (!std::isfinite(options.source_triangulation_angle_weight) ||
      !std::isfinite(options.source_shared_points_weight) ||
      !std::isfinite(options.source_scale_consistency_weight) ||
      !std::isfinite(options.source_coverage_weight) ||
      options.source_triangulation_angle_weight < 0.0 ||
      options.source_shared_points_weight < 0.0 ||
      options.source_scale_consistency_weight < 0.0 ||
      options.source_coverage_weight < 0.0) {
    throw std::invalid_argument(
        "Source view score weights must be non-negative");
  }
  if (options.source_coverage_grid_size == 0) {
    throw std::invalid_argument(
        "--source_coverage_grid_size must be positive");
  }
  if (options.initialize_patch_match && options.image_path.empty()) {
    throw std::invalid_argument(
        "--initialize_patch_match requires --image_path");
  }
  if (options.image_undistorter && options.image_path.empty()) {
    throw std::invalid_argument(
        "--image_undistorter requires --image_path");
  }
  const bool undistortion_options_valid =
      std::isfinite(options.undistort_blank_pixels) &&
      options.undistort_blank_pixels >= 0.0 &&
      options.undistort_blank_pixels <= 1.0 &&
      std::isfinite(options.undistort_min_scale) &&
      std::isfinite(options.undistort_max_scale) &&
      options.undistort_min_scale > 0.0 &&
      options.undistort_min_scale <= options.undistort_max_scale &&
      options.undistort_max_image_size != 0;
  if (!undistortion_options_valid) {
    throw std::invalid_argument("Invalid image undistortion options");
  }
  if (options.run_patch_match && options.image_path.empty()) {
    throw std::invalid_argument("--run_patch_match requires --image_path");
  }
  if (options.patch_match_geometric_refinement_iterations > 0 &&
      !options.run_patch_match) {
    throw std::invalid_argument(
        "--patch_match_geometric_refinement_iterations requires "
        "--run_patch_match");
  }
  if (options.patch_match_window_radius < 0) {
    throw std::invalid_argument(
        "--patch_match_window_radius must be non-negative");
  }
  if (!std::isfinite(options.patch_match_sigma_spatial) ||
      !std::isfinite(options.patch_match_sigma_color) ||
      options.patch_match_sigma_color <= 0.0F) {
    throw std::invalid_argument(
        "PatchMatch bilateral NCC sigmas must be finite and color sigma "
        "must be positive");
  }
  if (!std::isfinite(options.patch_match_ncc_sigma) ||
      options.patch_match_ncc_sigma <= 0.0F ||
      !std::isfinite(
          options.patch_match_previous_selection_probability_weight) ||
      options.patch_match_previous_selection_probability_weight < 0.0F ||
      options.patch_match_previous_selection_probability_weight > 1.0F ||
      !std::isfinite(options.patch_match_min_selection_probability) ||
      !std::isfinite(options.patch_match_max_selection_probability) ||
      options.patch_match_min_selection_probability < 0.0F ||
      options.patch_match_max_selection_probability > 1.0F ||
      options.patch_match_min_selection_probability >
          options.patch_match_max_selection_probability) {
    throw std::invalid_argument(
        "PatchMatch dynamic source selection options are invalid");
  }
  try {
    (void)mvs::PatchMatchBackendFromName(options.patch_match_backend);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(error.what());
  }
  if (options.patch_match_gpu_index < -1) {
    throw std::invalid_argument("--patch_match_gpu_index must be >= -1");
  }
  if (options.patch_match_pyramid_levels == 0) {
    throw std::invalid_argument(
        "--patch_match_pyramid_levels must be positive");
  }
  if (options.patch_match_perturbation_samples == 0) {
    throw std::invalid_argument(
        "--patch_match_perturbation_samples must be positive");
  }
  if (options.patch_match_min_num_valid_sources == 0) {
    throw std::invalid_argument(
        "--patch_match_min_num_valid_sources must be positive");
  }
  if (options.patch_match_propagation_schedule != "forward_backward" &&
      options.patch_match_propagation_schedule != "four_direction" &&
      options.patch_match_propagation_schedule != "checkerboard") {
    throw std::invalid_argument(
        "Invalid PatchMatch propagation schedule: " +
        options.patch_match_propagation_schedule);
  }
  if (!std::isfinite(options.patch_match_geometric_consistency_weight) ||
      options.patch_match_geometric_consistency_weight < 0.0F) {
    throw std::invalid_argument(
        "--patch_match_geometric_consistency_weight must be non-negative");
  }
  if (!std::isfinite(
          options.patch_match_geometric_max_relative_depth_error) ||
      options.patch_match_geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "--patch_match_geometric_max_relative_depth_error must be positive");
  }
  if (!std::isfinite(
          options.normal_refinement_max_relative_depth_difference) ||
      options.normal_refinement_max_relative_depth_difference < 0.0F) {
    throw std::invalid_argument(
        "--normal_refinement_max_relative_depth_difference must be "
        "non-negative");
  }
  if (!std::isfinite(options.filter_photometric_max_cost) ||
      options.filter_photometric_max_cost < -1.0F) {
    throw std::invalid_argument(
        "--filter_photometric_max_cost must be non-negative when enabled");
  }
  if (options.check_geometric_consistency && !options.run_patch_match &&
      !options.fuse_stereo) {
    throw std::invalid_argument(
        "--check_geometric_consistency requires --run_patch_match");
  }
  if ((options.filter_dense_cloud || options.mesh_dense_cloud) &&
      !options.fuse_stereo) {
    throw std::invalid_argument(
        "Dense cloud filtering and meshing require --fuse_stereo");
  }
  const bool poisson_options_valid =
      std::isfinite(options.poisson_point_weight) &&
      options.poisson_point_weight >= 0.0 && options.poisson_depth > 0 &&
      std::isfinite(options.poisson_trim) && options.poisson_trim >= 0.0;
  if (!poisson_options_valid) {
    throw std::invalid_argument("Invalid Poisson meshing options");
  }
  const bool delaunay_options_valid =
      std::isfinite(options.delaunay_max_proj_dist) &&
      options.delaunay_max_proj_dist >= 0.0 &&
      std::isfinite(options.delaunay_max_depth_dist) &&
      options.delaunay_max_depth_dist >= 0.0 &&
      options.delaunay_max_depth_dist <= 1.0 &&
      std::isfinite(options.delaunay_visibility_sigma) &&
      options.delaunay_visibility_sigma > 0.0 &&
      std::isfinite(options.delaunay_distance_sigma_factor) &&
      options.delaunay_distance_sigma_factor > 0.0 &&
      std::isfinite(options.delaunay_quality_regularization) &&
      options.delaunay_quality_regularization >= 0.0 &&
      std::isfinite(options.delaunay_max_side_length_factor) &&
      options.delaunay_max_side_length_factor >= 0.0 &&
      std::isfinite(options.delaunay_max_side_length_percentile) &&
      options.delaunay_max_side_length_percentile >= 0.0 &&
      options.delaunay_max_side_length_percentile <= 100.0;
  if (!delaunay_options_valid) {
    throw std::invalid_argument("Invalid Delaunay meshing options");
  }
  const bool simplification_options_valid =
      std::isfinite(options.mesh_simplification_ratio) &&
      options.mesh_simplification_ratio > 0.0 &&
      options.mesh_simplification_ratio <= 1.0 &&
      std::isfinite(options.mesh_simplification_max_error) &&
      options.mesh_simplification_max_error >= 0.0 &&
      std::isfinite(options.mesh_simplification_max_normal_change_degrees) &&
      options.mesh_simplification_max_normal_change_degrees >= 0.0 &&
      options.mesh_simplification_max_normal_change_degrees <= 180.0;
  if (!simplification_options_valid) {
    throw std::invalid_argument("Invalid mesh simplification options");
  }
  if (options.mesh_simplifier && !options.poisson_mesher &&
      !options.delaunay_mesher && !options.mesh_dense_cloud) {
    throw std::invalid_argument(
        "--mesh_simplifier requires a mesh generation option");
  }
  const bool texturing_options_valid =
      options.texture_atlas_size >= 16 &&
      options.texture_max_atlas_size >= options.texture_atlas_size &&
      options.texture_patch_padding >= 0 &&
      options.texture_inpaint_radius >= 0 &&
      options.texture_patch_padding * 2 + 1 < options.texture_atlas_size &&
      std::isfinite(options.texture_scale) && options.texture_scale > 0.0 &&
      std::isfinite(options.texture_min_cos_normal_angle) &&
      options.texture_min_cos_normal_angle >= -1.0 &&
      options.texture_min_cos_normal_angle <= 1.0 &&
      options.texture_min_visible_vertices >= 1 &&
      options.texture_min_visible_vertices <= 3 &&
      options.texture_view_smoothing_iterations >= 0 &&
      options.texture_visibility_resolution >= 16 &&
      std::isfinite(options.texture_occlusion_relative_tolerance) &&
      options.texture_occlusion_relative_tolerance >= 0.0 &&
      std::isfinite(options.texture_color_correction_regularization) &&
      options.texture_color_correction_regularization > 0.0;
  if (!texturing_options_valid) {
    throw std::invalid_argument("Invalid mesh texturing options");
  }
  if (options.mesh_texturer && !options.poisson_mesher &&
      !options.delaunay_mesher && !options.mesh_dense_cloud) {
    throw std::invalid_argument(
        "--mesh_texturer requires a mesh generation option");
  }
  if (options.poisson_mesher && !mvs::IsPoissonMeshingAvailable()) {
    throw std::invalid_argument(
        "--poisson_mesher is unavailable in this build");
  }
  if (options.delaunay_mesher && !mvs::IsDelaunayMeshingAvailable()) {
    throw std::invalid_argument(
        "--delaunay_mesher requires a build configured with CGAL");
  }
  if (options.fusion_min_num_observations == 0) {
    throw std::invalid_argument(
        "--fusion_min_num_observations must be positive");
  }
  if (options.fusion_max_num_observations <
      options.fusion_min_num_observations) {
    throw std::invalid_argument(
        "--fusion_max_num_observations must be >= "
        "--fusion_min_num_observations");
  }
  if (!std::isfinite(options.fusion_max_reprojection_error_pixels) ||
      options.fusion_max_reprojection_error_pixels < 0.0F) {
    throw std::invalid_argument(
        "--fusion_max_reprojection_error must be non-negative");
  }
  if (!std::isfinite(options.fusion_max_relative_depth_error) ||
      options.fusion_max_relative_depth_error < 0.0F) {
    throw std::invalid_argument(
        "--fusion_max_relative_depth_error must be non-negative");
  }
  if (!std::isfinite(options.fusion_max_normal_error_degrees) ||
      options.fusion_max_normal_error_degrees < 0.0F ||
      options.fusion_max_normal_error_degrees > 180.0F) {
    throw std::invalid_argument(
        "--fusion_max_normal_error must be in [0, 180]");
  }
  if (!std::isfinite(options.fusion_max_view_angle_degrees) ||
      options.fusion_max_view_angle_degrees < 0.0F ||
      options.fusion_max_view_angle_degrees > 180.0F) {
    throw std::invalid_argument(
        "--fusion_max_view_angle must be in [0, 180]");
  }
  if (options.fusion_source_pixel_search_radius > 5) {
    throw std::invalid_argument(
        "--fusion_source_pixel_search_radius must be <= 5");
  }
  if (options.dense_cloud_min_track_observations == 0) {
    throw std::invalid_argument(
        "--dense_cloud_min_track_observations must be positive");
  }
  if (!std::isfinite(options.dense_cloud_voxel_size) ||
      options.dense_cloud_voxel_size < 0.0) {
    throw std::invalid_argument(
        "--dense_cloud_voxel_size must be non-negative");
  }
  if (!std::isfinite(options.mesh_max_edge_length) ||
      options.mesh_max_edge_length < 0.0) {
    throw std::invalid_argument("--mesh_max_edge_length must be non-negative");
  }
  if (!std::isfinite(options.mesh_max_normal_angle_degrees) ||
      options.mesh_max_normal_angle_degrees < 0.0 ||
      options.mesh_max_normal_angle_degrees > 180.0) {
    throw std::invalid_argument(
        "--mesh_max_normal_angle must be in [0, 180]");
  }
  if (!std::isfinite(options.geometric_max_reprojection_error_pixels) ||
      options.geometric_max_reprojection_error_pixels < 0.0F) {
    throw std::invalid_argument(
        "--geometric_max_reprojection_error must be non-negative");
  }
  if (!std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error < 0.0F) {
    throw std::invalid_argument(
        "--geometric_max_relative_depth_error must be non-negative");
  }
  if (options.dense_map_format != "text" &&
      options.dense_map_format != "binary") {
    throw std::invalid_argument(
        "--dense_map_format must be either text or binary");
  }
  if (options.num_threads == 0) {
    throw std::invalid_argument("--num_threads must be positive");
  }
}

}  // namespace

int RunReconMvs(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err) {
  const std::string executable = args.empty() ? "recon_mvs" : args.front();

  try {
    const ReconMvsCliOptions options = ParseOptions(args);
    if (options.show_help) {
      PrintHelp(executable, out);
      return 0;
    }
    if (options.show_version) {
      out << colite::Version() << "\n";
      return 0;
    }
    ValidateOptions(options);
    return controllers::ReconMvsController(options).Run(out);
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return 1;
  }
}

}  // namespace colite::exe
