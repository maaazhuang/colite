#include "recon_mvs_app.h"

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
#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/dense_map.h"
#include "colite/mvs/image_workspace.h"
#include "colite/mvs/mvs_pipeline.h"
#include "colite/mvs/patch_match.h"

namespace colite::app {
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
      << "  --dense_map_format <fmt>    Dense map output format: text or "
         "binary, default text.\n"
      << "  --initialize_patch_match    Initialize dense depth/normal maps.\n"
      << "  --run_patch_match           Run CPU PatchMatch stereo for each "
         "registered image.\n"
      << "  --patch_match_iterations <n> PatchMatch iterations, default 3.\n"
      << "  --patch_match_pyramid_levels <n> Coarse-to-fine PatchMatch "
         "pyramid levels, default 1.\n"
      << "  --patch_match_window_radius <n> ZNCC window radius, default 2.\n"
      << "  --patch_match_perturbation_samples <n> Random plane candidates "
         "per valid pixel during perturbation, default 1.\n"
      << "  --patch_match_min_num_valid_sources <n> Minimum valid source "
         "views per PatchMatch cost, default 1.\n"
      << "  --patch_match_max_num_aggregated_costs <n> Best source costs "
         "aggregated per PatchMatch hypothesis, default 3; 0 uses all.\n"
      << "  --patch_match_propagation_schedule <name> Propagation schedule: "
         "forward_backward, four_direction, or checkerboard; default "
         "forward_backward.\n"
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
      << "  --geometric_max_reprojection_error <px> Geometric consistency "
         "reprojection threshold, default 2.\n"
      << "  --geometric_max_relative_depth_error <r> Geometric consistency "
         "relative depth threshold, default 0.01.\n"
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
    } else if (arg == "--patch_match_iterations") {
      options.patch_match_iterations =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_pyramid_levels") {
      options.patch_match_pyramid_levels =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--patch_match_window_radius") {
      options.patch_match_window_radius =
          std::stoi(RequireValue(args, &i, arg));
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
  if (options.check_geometric_consistency && !options.run_patch_match) {
    throw std::invalid_argument(
        "--check_geometric_consistency requires --run_patch_match");
  }
  if (options.fuse_stereo && !options.run_patch_match) {
    throw std::invalid_argument("--fuse_stereo requires --run_patch_match");
  }
  if ((options.filter_dense_cloud || options.mesh_dense_cloud) &&
      !options.fuse_stereo) {
    throw std::invalid_argument(
        "Dense cloud filtering and meshing require --fuse_stereo");
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

std::vector<std::string> DataLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open model file: " + path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

void ReadCameras(const std::filesystem::path& model_path,
                 Reconstruction* reconstruction) {
  for (const std::string& line : DataLines(model_path / "cameras.txt")) {
    std::istringstream stream(line);
    Camera camera;
    std::string model_name;
    stream >> camera.id >> model_name >> camera.width >> camera.height;
    camera.model_id = CameraModelNameToId(model_name);
    if (!ExistsCameraModel(camera.model_id)) {
      throw std::runtime_error("Unsupported camera model: " + model_name);
    }
    double param = 0.0;
    while (stream >> param) {
      camera.params.push_back(param);
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid camera parameters in: " + line);
    }
    reconstruction->AddCamera(camera);
  }
}

Point3DId ParseColmapPoint3DId(std::int64_t raw_id) {
  if (raw_id < 0) {
    return kInvalidPoint3DId;
  }
  return static_cast<Point3DId>(raw_id);
}

void ReadImages(const std::filesystem::path& model_path,
                Reconstruction* reconstruction) {
  const std::vector<std::string> lines = DataLines(model_path / "images.txt");
  if (lines.size() % 2 != 0) {
    throw std::runtime_error("images.txt must contain two data lines per image");
  }

  for (std::size_t i = 0; i < lines.size(); i += 2) {
    std::istringstream header(lines[i]);
    Image image;
    Pose3 pose;
    header >> image.id >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >>
        pose.q_cw.z() >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z() >>
        image.camera_id >> image.name;
    if (!header || !reconstruction->ExistsCamera(image.camera_id)) {
      throw std::runtime_error("Invalid image entry in: " + lines[i]);
    }
    image.registered = true;
    image.cam_from_world = pose;

    std::istringstream points_line(lines[i + 1]);
    double x = 0.0;
    double y = 0.0;
    std::int64_t raw_point3d_id = -1;
    Point2DIdx point2d_idx = 0;
    while (points_line >> x >> y >> raw_point3d_id) {
      Point2D point2d;
      point2d.xy = Eigen::Vector2d(x, y);
      point2d.feature_idx = point2d_idx++;
      point2d.point3d_id = ParseColmapPoint3DId(raw_point3d_id);
      image.points2d.push_back(point2d);
    }

    reconstruction->AddImage(image);
    reconstruction->RegisterImage(image.id);
  }
}

void ReadPoints3D(const std::filesystem::path& model_path,
                  Reconstruction* reconstruction) {
  for (const std::string& line : DataLines(model_path / "points3D.txt")) {
    std::istringstream stream(line);
    Point3D point3d;
    int red = 0;
    int green = 0;
    int blue = 0;
    stream >> point3d.id >> point3d.xyz.x() >> point3d.xyz.y() >>
        point3d.xyz.z() >> red >> green >> blue >> point3d.reprojection_error;
    if (!stream) {
      throw std::runtime_error("Invalid point3D entry in: " + line);
    }
    point3d.color =
        Eigen::Matrix<std::uint8_t, 3, 1>(static_cast<std::uint8_t>(red),
                                          static_cast<std::uint8_t>(green),
                                          static_cast<std::uint8_t>(blue));

    ImageId image_id = kInvalidImageId;
    Point2DIdx point2d_idx = kInvalidPoint2DIdx;
    while (stream >> image_id >> point2d_idx) {
      point3d.track.AddElement(image_id, point2d_idx);
    }
    reconstruction->AddPoint3D(point3d);
  }
}

Reconstruction ReadColmapTextModel(const std::filesystem::path& model_path) {
  Reconstruction reconstruction;
  ReadCameras(model_path, &reconstruction);
  ReadImages(model_path, &reconstruction);
  ReadPoints3D(model_path, &reconstruction);
  return reconstruction;
}

void WriteMvsSummary(const std::filesystem::path& output_path,
                     const mvs::MvsPipelineSummary& summary,
                     std::size_t num_initialized_patch_match_images,
                     std::size_t num_patch_match_images,
                     std::size_t num_photometric_filtered_pixels,
                     std::size_t num_normal_refined_pixels,
                     std::size_t num_geometric_consistent_pixels,
                     std::size_t num_geometric_filtered_pixels,
                     const mvs::StereoFusionSummary& fusion_summary) {
  std::filesystem::create_directories(output_path);
  std::ofstream file(output_path / "summary.txt");
  file << std::setprecision(17)
       << "success " << (summary.success ? 1 : 0) << "\n"
       << "num_images " << summary.num_images << "\n"
       << "num_sparse_depth_samples " << summary.num_sparse_depth_samples
       << "\n"
       << "initialized_patch_match_images "
       << num_initialized_patch_match_images
       << "\n"
       << "patch_match_images "
       << num_patch_match_images
       << "\n"
       << "photometric_filtered_pixels "
       << num_photometric_filtered_pixels
       << "\n"
       << "normal_refined_pixels "
       << num_normal_refined_pixels
       << "\n"
       << "geometric_consistent_pixels "
       << num_geometric_consistent_pixels
       << "\n"
       << "geometric_filtered_pixels "
       << num_geometric_filtered_pixels
       << "\n"
       << "fused_points "
       << fusion_summary.num_fused_points
       << "\n"
       << "fusion_tracks "
       << fusion_summary.num_fusion_tracks
       << "\n"
       << "fused_observations "
       << fusion_summary.num_fused_observations
       << "\n"
       << "fusion_visited_pixels "
       << fusion_summary.num_visited_pixels
       << "\n"
       << "fusion_rejected_tracks "
       << fusion_summary.num_rejected_tracks
       << "\n"
       << "fusion_rejected_normal_observations "
       << fusion_summary.num_rejected_normal_observations
       << "\n"
       << "fusion_rejected_view_observations "
       << fusion_summary.num_rejected_view_observations
       << "\n";
}

void WriteSourceImages(const std::filesystem::path& output_path,
                       const mvs::MvsPipelineSummary& summary) {
  std::ofstream file(output_path / "source_images.tsv");
  file << "target_image_id\tsource_image_id\tnum_shared_points"
       << "\tmean_triangulation_angle_degrees"
       << "\tmean_scale_ratio\tcoverage_score\tscore\n";
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    for (const mvs::SourceImageScore& source : image_summary.source_images) {
      file << std::setprecision(17) << image_summary.image_id << "\t"
           << source.image_id << "\t"
           << source.num_shared_points << "\t"
           << source.mean_triangulation_angle_degrees << "\t"
           << source.mean_scale_ratio << "\t"
           << source.coverage_score << "\t"
           << source.score << "\n";
    }
  }
}

void WriteSparseDepthSamples(const std::filesystem::path& output_path,
                             const mvs::MvsPipelineSummary& summary) {
  std::ofstream file(output_path / "sparse_depth_samples.tsv");
  file << "image_id\tpoint3d_id\tx\ty\tdepth\n";
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    for (const mvs::SparseDepthSample& sample :
         image_summary.sparse_depth_map.samples) {
      file << std::setprecision(17) << image_summary.image_id << "\t"
           << sample.point3d_id << "\t" << sample.x << "\t" << sample.y
           << "\t" << sample.depth << "\n";
    }
  }
}

bool UseBinaryDenseMaps(const ReconMvsCliOptions& cli_options);

std::string DenseMapFileName(const std::string& image_name,
                             bool init_map,
                             bool binary);

void WriteDenseDepthMap(const std::filesystem::path& path,
                        const mvs::DenseDepthMap& depth_map,
                        bool binary);

void WriteNormalMap(const std::filesystem::path& path,
                    const mvs::NormalMap& normal_map,
                    bool binary);

std::size_t InitializePatchMatchMapsForWorkspace(
    const std::filesystem::path& output_path,
    const ReconMvsCliOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  mvs::PatchMatchInitializationOptions init_options;
  init_options.random_seed = cli_options.random_seed;

  const std::filesystem::path depth_dir =
      output_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      output_path / "stereo" / "normal_maps";
  const bool binary_dense_maps = UseBinaryDenseMaps(cli_options);
  std::size_t num_initialized = 0;
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    if (image_summary.sparse_depth_map.samples.empty()) {
      continue;
    }
    const mvs::MvsImage& image = workspace.GetImage(image_summary.image_id);
    const mvs::PatchMatchMaps maps =
        mvs::InitializePatchMatchMaps(image, image_summary.sparse_depth_map,
                                      init_options);
    const std::string file_name =
        DenseMapFileName(image.name, true, binary_dense_maps);
    WriteDenseDepthMap(depth_dir / file_name, maps.depth_map,
                       binary_dense_maps);
    WriteNormalMap(normal_dir / file_name, maps.normal_map,
                   binary_dense_maps);
    ++num_initialized;
  }
  return num_initialized;
}

mvs::PatchMatchView MakePatchMatchView(const Reconstruction& reconstruction,
                                       ImageId image_id,
                                       double image_scale) {
  const Image& image = reconstruction.GetImage(image_id);
  mvs::PatchMatchView view;
  view.image_id = image_id;
  view.camera = reconstruction.GetCamera(image.camera_id);
  if (!image.cam_from_world.has_value()) {
    throw std::runtime_error("PatchMatch image has no registered pose.");
  }
  view.cam_from_world = *image.cam_from_world;

  if (image_scale > 0.0 && image_scale != 1.0) {
    view.camera.width =
        static_cast<std::uint32_t>(std::lround(view.camera.width * image_scale));
    view.camera.height = static_cast<std::uint32_t>(
        std::lround(view.camera.height * image_scale));
    const CameraModelSpec& spec = GetCameraModelSpec(view.camera.model_id);
    for (const std::size_t idx : spec.focal_length_idxs) {
      view.camera.params.at(idx) *= image_scale;
    }
    for (const std::size_t idx : spec.principal_point_idxs) {
      view.camera.params.at(idx) *= image_scale;
    }
  }
  return view;
}

std::string FormatStat(double value) {
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

std::string FormatDistributionValue(double value) {
  if (std::isinf(value)) {
    return value > 0.0 ? "inf" : "-inf";
  }
  return FormatStat(value);
}

double Percentile(std::vector<float> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  if (values.size() == 1) {
    return static_cast<double>(values.front());
  }
  const double position =
      q * static_cast<double>(values.size() - 1U);
  const std::size_t lower_idx = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper_idx = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower_idx);
  return static_cast<double>(values[lower_idx]) * (1.0 - alpha) +
         static_cast<double>(values[upper_idx]) * alpha;
}

bool UseBinaryDenseMaps(const ReconMvsCliOptions& cli_options) {
  return cli_options.dense_map_format == "binary";
}

std::string DenseMapFileName(const std::string& image_name,
                             bool init_map,
                             bool binary) {
  return image_name + (init_map ? ".init" : "") + (binary ? ".bin" : ".txt");
}

void WriteDenseDepthMap(const std::filesystem::path& path,
                        const mvs::DenseDepthMap& depth_map,
                        bool binary) {
  if (binary) {
    mvs::WriteDenseDepthMapBinary(path, depth_map);
  } else {
    mvs::WriteDenseDepthMapText(path, depth_map);
  }
}

void WriteNormalMap(const std::filesystem::path& path,
                    const mvs::NormalMap& normal_map,
                    bool binary) {
  if (binary) {
    mvs::WriteNormalMapBinary(path, normal_map);
  } else {
    mvs::WriteNormalMapText(path, normal_map);
  }
}

mvs::PatchMatchPropagationSchedule ParsePatchMatchPropagationSchedule(
    const std::string& name) {
  if (name == "forward_backward") {
    return mvs::PatchMatchPropagationSchedule::kForwardBackward;
  }
  if (name == "four_direction") {
    return mvs::PatchMatchPropagationSchedule::kFourDirection;
  }
  if (name == "checkerboard") {
    return mvs::PatchMatchPropagationSchedule::kCheckerboard;
  }
  throw std::invalid_argument("Invalid PatchMatch propagation schedule: " +
                              name);
}

struct PatchMatchWorkspaceResult {
  ImageId image_id = kInvalidImageId;
  std::string dense_file_name;
  std::string text_file_name;
  mvs::PatchMatchView view;
  mvs::MvsImage image;
  std::vector<ImageId> source_image_ids;
  float min_depth = 0.0F;
  float max_depth = 0.0F;
  mvs::PatchMatchStereoResult result;
  mvs::ConsistencyGraph consistency_graph;
  mvs::PatchMatchPhotometricStats stats_before_filter;
  mvs::PatchMatchPhotometricStats stats_after_photometric_filter;
  std::size_t photometric_filtered_pixels = 0;
  std::size_t geometric_consistent_pixels = 0;
  std::size_t geometric_filtered_pixels = 0;
};

void WriteConsistencyGraphText(const std::filesystem::path& path,
                               const mvs::ConsistencyGraph& graph) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write consistency graph: " +
                             path.string());
  }
  file << "COLITE_CONSISTENCY_GRAPH_TEXT 1\n"
       << graph.Width() << " " << graph.Height() << "\n";
  for (std::size_t row = 0; row < graph.Height(); ++row) {
    for (std::size_t col = 0; col < graph.Width(); ++col) {
      const std::vector<ImageId>& source_ids = graph.SourceImageIds(row, col);
      if (source_ids.empty()) {
        continue;
      }
      file << row << " " << col << " " << source_ids.size();
      for (const ImageId source_id : source_ids) {
        file << " " << source_id;
      }
      file << "\n";
    }
  }
}

std::vector<const mvs::ImageMvsSummary*> CollectPatchMatchSummaries(
    const mvs::MvsPipelineSummary& summary) {
  std::vector<const mvs::ImageMvsSummary*> image_summaries;
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    if (image_summary.sparse_depth_map.samples.empty() ||
        image_summary.source_images.empty()) {
      continue;
    }
    image_summaries.push_back(&image_summary);
  }
  return image_summaries;
}

std::vector<ImageId> CollectRequiredMvsImageIds(
    const std::vector<const mvs::ImageMvsSummary*>& image_summaries) {
  std::vector<ImageId> image_ids;
  for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
    image_ids.push_back(image_summary->image_id);
    for (const mvs::SourceImageScore& source : image_summary->source_images) {
      image_ids.push_back(source.image_id);
    }
  }
  std::sort(image_ids.begin(), image_ids.end());
  image_ids.erase(std::unique(image_ids.begin(), image_ids.end()),
                  image_ids.end());
  return image_ids;
}

std::unordered_map<ImageId, mvs::MvsImage> PreloadMvsImages(
    const std::vector<ImageId>& image_ids,
    mvs::MvsImageWorkspace* workspace) {
  std::unordered_map<ImageId, mvs::MvsImage> images;
  images.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    images.emplace(image_id, workspace->GetImage(image_id));
  }
  return images;
}

std::unordered_map<ImageId, mvs::PatchMatchView> BuildPatchMatchViews(
    const Reconstruction& reconstruction,
    const std::unordered_map<ImageId, mvs::MvsImage>& images) {
  std::unordered_map<ImageId, mvs::PatchMatchView> views;
  views.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    views.emplace(image_id,
                  MakePatchMatchView(reconstruction, image_id, image.scale));
  }
  return views;
}

PatchMatchWorkspaceResult RunInitialPatchMatchForImage(
    const mvs::ImageMvsSummary& image_summary,
    const std::unordered_map<ImageId, mvs::MvsImage>& images,
    const std::unordered_map<ImageId, mvs::PatchMatchView>& views,
    const mvs::PatchMatchStereoOptions& patch_match_options,
    const ReconMvsCliOptions& cli_options,
    bool binary_dense_maps,
    bool run_geometric_refinement) {
  const mvs::MvsImage& reference_image = images.at(image_summary.image_id);
  const mvs::PatchMatchView& reference_view = views.at(image_summary.image_id);

  std::vector<mvs::PatchMatchView> source_views;
  std::vector<mvs::MvsImage> source_images;
  source_views.reserve(image_summary.source_images.size());
  source_images.reserve(image_summary.source_images.size());
  for (const mvs::SourceImageScore& source : image_summary.source_images) {
    source_views.push_back(views.at(source.image_id));
    source_images.push_back(images.at(source.image_id));
  }

  mvs::PatchMatchStereoResult result = mvs::RunPatchMatchStereoForImage(
      reference_view, source_views, reference_image, source_images,
      static_cast<float>(image_summary.min_depth),
      static_cast<float>(image_summary.max_depth), patch_match_options);
  mvs::PatchMatchPhotometricStats stats_before =
      mvs::ComputePatchMatchPhotometricStats(result.cost_map);
  mvs::PatchMatchPhotometricFilterSummary filter_summary;
  if (!run_geometric_refinement &&
      cli_options.filter_photometric_max_cost >= 0.0F) {
    filter_summary = mvs::FilterPatchMatchPhotometricResults(
        cli_options.filter_photometric_max_cost, &result.depth_map,
        &result.normal_map, &result.cost_map);
  } else {
    filter_summary.num_pixels = result.cost_map.NumPixels();
    filter_summary.num_kept_pixels = result.cost_map.ValidPixelCount();
  }
  const mvs::PatchMatchPhotometricStats stats_after =
      mvs::ComputePatchMatchPhotometricStats(result.cost_map);

  PatchMatchWorkspaceResult workspace_result;
  workspace_result.image_id = image_summary.image_id;
  workspace_result.dense_file_name =
      DenseMapFileName(reference_image.name, false, binary_dense_maps);
  workspace_result.text_file_name = reference_image.name + ".txt";
  workspace_result.view = reference_view;
  workspace_result.image = reference_image;
  workspace_result.min_depth = static_cast<float>(image_summary.min_depth);
  workspace_result.max_depth = static_cast<float>(image_summary.max_depth);
  for (const mvs::SourceImageScore& source : image_summary.source_images) {
    workspace_result.source_image_ids.push_back(source.image_id);
  }
  workspace_result.result = std::move(result);
  workspace_result.stats_before_filter = stats_before;
  workspace_result.stats_after_photometric_filter = stats_after;
  workspace_result.photometric_filtered_pixels =
      filter_summary.num_filtered_pixels;
  return workspace_result;
}

std::vector<PatchMatchWorkspaceResult> RunInitialPatchMatchForWorkspace(
    const std::vector<const mvs::ImageMvsSummary*>& image_summaries,
    const std::unordered_map<ImageId, mvs::MvsImage>& images,
    const std::unordered_map<ImageId, mvs::PatchMatchView>& views,
    const mvs::PatchMatchStereoOptions& patch_match_options,
    const ReconMvsCliOptions& cli_options,
    bool binary_dense_maps,
    bool run_geometric_refinement) {
  std::vector<PatchMatchWorkspaceResult> results;
  results.reserve(image_summaries.size());
  const std::size_t num_threads =
      std::max<std::size_t>(1, cli_options.num_threads);
  if (num_threads == 1 || image_summaries.size() <= 1) {
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      results.push_back(RunInitialPatchMatchForImage(
          *image_summary, images, views, patch_match_options, cli_options,
          binary_dense_maps, run_geometric_refinement));
    }
  } else {
    std::vector<std::future<PatchMatchWorkspaceResult>> futures;
    futures.reserve(num_threads);
    const auto collect_oldest_future = [&results, &futures]() {
      results.push_back(futures.front().get());
      futures.erase(futures.begin());
    };
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      futures.push_back(std::async(
          std::launch::async,
          [&images, &views, &patch_match_options, &cli_options,
           binary_dense_maps, run_geometric_refinement, image_summary]() {
            return RunInitialPatchMatchForImage(
                *image_summary, images, views, patch_match_options,
                cli_options, binary_dense_maps, run_geometric_refinement);
          }));
      if (futures.size() >= num_threads) {
        collect_oldest_future();
      }
    }
    while (!futures.empty()) {
      collect_oldest_future();
    }
  }

  std::sort(results.begin(), results.end(),
            [](const PatchMatchWorkspaceResult& lhs,
               const PatchMatchWorkspaceResult& rhs) {
              return lhs.image_id < rhs.image_id;
            });
  return results;
}

void WriteFusedPointCloudPly(const std::filesystem::path& path,
                             const std::vector<mvs::FusedPoint>& points) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fused point cloud: " +
                             path.string());
  }

  file << "ply\n"
       << "format ascii 1.0\n"
       << "element vertex " << points.size() << "\n"
       << "property double x\n"
       << "property double y\n"
       << "property double z\n"
       << "property float nx\n"
       << "property float ny\n"
       << "property float nz\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "end_header\n";
  file << std::setprecision(17);
  for (const mvs::FusedPoint& point : points) {
    file << point.xyz.x() << " " << point.xyz.y() << " " << point.xyz.z()
         << " " << point.normal.x() << " " << point.normal.y() << " "
         << point.normal.z() << " "
         << static_cast<int>(point.color.x()) << " "
         << static_cast<int>(point.color.y()) << " "
         << static_cast<int>(point.color.z()) << "\n";
  }
}

void WriteDenseCloudFilterStats(
    const std::filesystem::path& path,
    const mvs::DenseCloudFilterStats& stats) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense cloud filter stats: " +
                             path.string());
  }

  file << "num_input_points " << stats.num_input_points << "\n"
       << "num_filtered_points " << stats.num_filtered_points << "\n"
       << "num_removed_by_track_observations "
       << stats.num_removed_by_track_observations << "\n"
       << "num_removed_by_voxel " << stats.num_removed_by_voxel << "\n";
}

void WriteFusionTracksDiagnostics(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionTrack>& tracks) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion tracks diagnostics: " +
                             path.string());
  }

  file << "point_idx\tseed_image_id\tseed_row\tseed_col"
       << "\tnum_observations\tnum_unique_images\tmax_traversal_depth"
       << "\tmax_normal_angle_degrees\tmax_view_angle_degrees\n";
  file << std::setprecision(17);
  for (const mvs::StereoFusionTrack& track : tracks) {
    file << track.point_idx << "\t"
         << track.seed_image_id << "\t"
         << track.seed_row << "\t"
         << track.seed_col << "\t"
         << track.num_observations << "\t"
         << track.num_unique_images << "\t"
         << track.max_traversal_depth << "\t"
         << track.max_normal_angle_degrees << "\t"
         << track.max_view_angle_degrees << "\n";
  }
}

void WriteFusionImagePairsDiagnostics(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionPairDiagnostics>& diagnostics) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion image pair diagnostics: " +
                             path.string());
  }

  file << "reference_image_id\tsource_image_id\tnum_candidates\tnum_accepted"
       << "\tnum_rejected_normal\tnum_rejected_view"
       << "\tnum_rejected_projection\tnum_rejected_depth"
       << "\tnum_rejected_mutual_consistency\tnum_rejected_invalid"
       << "\tnum_rejected_visited\tnum_rejected_duplicate"
       << "\tnum_rejected_incomplete\tnum_rejected_other\n";
  for (const mvs::StereoFusionPairDiagnostics& pair : diagnostics) {
    file << pair.reference_image_id << "\t"
         << pair.source_image_id << "\t"
         << pair.num_candidates << "\t"
         << pair.num_accepted << "\t"
         << pair.num_rejected_normal << "\t"
         << pair.num_rejected_view << "\t"
         << pair.num_rejected_projection << "\t"
         << pair.num_rejected_depth << "\t"
         << pair.num_rejected_mutual_consistency << "\t"
         << pair.num_rejected_invalid << "\t"
         << pair.num_rejected_visited << "\t"
         << pair.num_rejected_duplicate << "\t"
         << pair.num_rejected_incomplete << "\t"
         << pair.num_rejected_other << "\n";
  }
}

void WriteFusionTrackLengthHistogram(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionTrack>& tracks) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion track length histogram: " +
                             path.string());
  }

  std::map<std::size_t, std::size_t> histogram;
  for (const mvs::StereoFusionTrack& track : tracks) {
    histogram[track.num_observations]++;
  }

  file << "num_observations\tnum_tracks\n";
  for (const auto& [num_observations, num_tracks] : histogram) {
    file << num_observations << "\t" << num_tracks << "\n";
  }
}

void WriteFusionMetricDistribution(
    std::ofstream* file,
    std::string_view metric,
    const std::vector<float>& samples,
    const std::vector<std::pair<double, double>>& bins) {
  const std::array<std::pair<std::string_view, double>, 5> percentiles = {
      std::make_pair(std::string_view("p50"), 0.50),
      std::make_pair(std::string_view("p75"), 0.75),
      std::make_pair(std::string_view("p90"), 0.90),
      std::make_pair(std::string_view("p95"), 0.95),
      std::make_pair(std::string_view("p99"), 0.99),
  };

  *file << metric << "\tsummary\tcount\t\t\t" << samples.size() << "\t"
        << samples.size() << "\n";
  for (const auto& [label, q] : percentiles) {
    *file << metric << "\tpercentile\t" << label << "\t\t\t"
          << samples.size() << "\t";
    if (!samples.empty()) {
      *file << Percentile(samples, q);
    }
    *file << "\n";
  }

  for (const auto& [lower, upper] : bins) {
    std::size_t count = 0;
    for (const float sample : samples) {
      if (!std::isfinite(sample)) {
        continue;
      }
      const double value = static_cast<double>(sample);
      if (value >= lower && value < upper) {
        count++;
      }
    }
    const std::string label =
        "[" + FormatDistributionValue(lower) + "," +
        FormatDistributionValue(upper) + ")";
    *file << metric << "\thistogram\t" << label << "\t"
          << FormatDistributionValue(lower) << "\t"
          << FormatDistributionValue(upper) << "\t" << count << "\t\n";
  }
}

void WriteFusionErrorDistribution(
    const std::filesystem::path& path,
    const mvs::StereoFusionErrorSamples& samples) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion error distribution: " +
                             path.string());
  }

  constexpr double kInf = std::numeric_limits<double>::infinity();
  const std::vector<std::pair<double, double>> relative_depth_bins = {
      {0.0, 0.0025}, {0.0025, 0.005}, {0.005, 0.01},
      {0.01, 0.02},  {0.02, 0.05},    {0.05, 0.1},
      {0.1, 0.2},    {0.2, kInf},
  };
  const std::vector<std::pair<double, double>> reprojection_bins = {
      {0.0, 0.25}, {0.25, 0.5}, {0.5, 1.0}, {1.0, 2.0},
      {2.0, 3.0},  {3.0, 5.0},  {5.0, 10.0}, {10.0, kInf},
  };
  const std::vector<std::pair<double, double>> angle_bins = {
      {0.0, 1.0},   {1.0, 2.0},   {2.0, 5.0},   {5.0, 10.0},
      {10.0, 20.0}, {20.0, 45.0}, {45.0, 90.0}, {90.0, 180.0},
      {180.0, kInf},
  };

  file << std::setprecision(17);
  file << "metric\trecord\tlabel\tlower\tupper\tcount\tvalue\n";
  WriteFusionMetricDistribution(&file, "relative_depth_error",
                                samples.relative_depth_errors,
                                relative_depth_bins);
  WriteFusionMetricDistribution(&file, "reprojection_error_pixels",
                                samples.reprojection_errors_pixels,
                                reprojection_bins);
  WriteFusionMetricDistribution(&file, "normal_angle_degrees",
                                samples.normal_angles_degrees, angle_bins);
  WriteFusionMetricDistribution(&file, "view_angle_degrees",
                                samples.view_angles_degrees, angle_bins);
}

std::size_t RunPatchMatchForWorkspace(
    const std::filesystem::path& output_path,
    const ReconMvsCliOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary,
    std::size_t* num_photometric_filtered_pixels,
    std::size_t* num_normal_refined_pixels,
    std::size_t* num_geometric_consistent_pixels,
    std::size_t* num_geometric_filtered_pixels,
    mvs::StereoFusionSummary* fusion_summary_out) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  mvs::PatchMatchStereoOptions patch_match_options;
  patch_match_options.initialization.random_seed = cli_options.random_seed;
  patch_match_options.perturbation.random_seed = cli_options.random_seed + 7919;
  patch_match_options.num_iterations = cli_options.patch_match_iterations;
  patch_match_options.pyramid_levels = cli_options.patch_match_pyramid_levels;
  patch_match_options.cost.window_radius = cli_options.patch_match_window_radius;
  patch_match_options.perturbation.num_samples_per_pixel =
      cli_options.patch_match_perturbation_samples;
  patch_match_options.cost.min_num_valid_sources =
      cli_options.patch_match_min_num_valid_sources;
  patch_match_options.cost.max_num_aggregated_costs =
      cli_options.patch_match_max_num_aggregated_costs;
  patch_match_options.normal_refinement.enabled =
      cli_options.refine_normals_from_depth;
  patch_match_options.normal_refinement.max_relative_depth_difference =
      cli_options.normal_refinement_max_relative_depth_difference;
  patch_match_options.propagation_schedule =
      ParsePatchMatchPropagationSchedule(
          cli_options.patch_match_propagation_schedule);
  const bool run_geometric_refinement =
      cli_options.patch_match_geometric_refinement_iterations > 0 &&
      cli_options.patch_match_geometric_consistency_weight > 0.0F;
  mvs::PatchMatchStereoOptions refinement_options = patch_match_options;
  refinement_options.num_iterations =
      cli_options.patch_match_geometric_refinement_iterations;
  refinement_options.cost.geometric_consistency_weight =
      cli_options.patch_match_geometric_consistency_weight;
  refinement_options.cost.geometric_max_relative_depth_error =
      cli_options.patch_match_geometric_max_relative_depth_error;

  const std::filesystem::path depth_dir =
      output_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      output_path / "stereo" / "normal_maps";
  const std::filesystem::path cost_dir = output_path / "stereo" / "cost_maps";
  const std::filesystem::path consistency_dir =
      output_path / "stereo" / "consistency_graphs";
  std::filesystem::create_directories(output_path / "stereo");
  const bool binary_dense_maps = UseBinaryDenseMaps(cli_options);

  std::size_t num_patch_match_images = 0;
  std::size_t total_photometric_filtered_pixels = 0;
  std::size_t total_normal_refined_pixels = 0;
  const std::vector<const mvs::ImageMvsSummary*> image_summaries =
      CollectPatchMatchSummaries(summary);
  const std::vector<ImageId> required_image_ids =
      CollectRequiredMvsImageIds(image_summaries);
  const std::unordered_map<ImageId, mvs::MvsImage> images =
      PreloadMvsImages(required_image_ids, &workspace);
  const std::unordered_map<ImageId, mvs::PatchMatchView> views =
      BuildPatchMatchViews(reconstruction, images);

  std::vector<PatchMatchWorkspaceResult> results =
      RunInitialPatchMatchForWorkspace(
          image_summaries, images, views, patch_match_options, cli_options,
          binary_dense_maps, run_geometric_refinement);
  std::unordered_map<ImageId, std::size_t> result_index_by_image_id;
  for (std::size_t i = 0; i < results.size(); ++i) {
    total_photometric_filtered_pixels +=
        results[i].photometric_filtered_pixels;
    total_normal_refined_pixels +=
        results[i].result.summary.num_normal_refined_pixels;
    result_index_by_image_id[results[i].image_id] = i;
  }
  num_patch_match_images = results.size();

  if (run_geometric_refinement) {
    for (PatchMatchWorkspaceResult& result : results) {
      std::vector<mvs::PatchMatchView> source_views;
      std::vector<mvs::MvsImage> source_images;
      std::vector<mvs::DenseDepthMap> source_depth_maps;
      for (const ImageId source_id : result.source_image_ids) {
        const auto it = result_index_by_image_id.find(source_id);
        if (it == result_index_by_image_id.end()) {
          continue;
        }
        const PatchMatchWorkspaceResult& source_result = results[it->second];
        source_views.push_back(source_result.view);
        source_images.push_back(source_result.image);
        source_depth_maps.push_back(source_result.result.depth_map);
      }
      if (source_views.empty()) {
        continue;
      }
      result.result = mvs::RunPatchMatchStereoForImageWithSourceDepths(
          result.view, source_views, result.image, source_images,
          source_depth_maps, result.min_depth, result.max_depth,
          refinement_options);
    }

    total_photometric_filtered_pixels = 0;
    total_normal_refined_pixels = 0;
    for (PatchMatchWorkspaceResult& result : results) {
      total_normal_refined_pixels +=
          result.result.summary.num_normal_refined_pixels;
      result.stats_before_filter =
          mvs::ComputePatchMatchPhotometricStats(result.result.cost_map);
      mvs::PatchMatchPhotometricFilterSummary filter_summary;
      if (cli_options.filter_photometric_max_cost >= 0.0F) {
        filter_summary = mvs::FilterPatchMatchPhotometricResults(
            cli_options.filter_photometric_max_cost, &result.result.depth_map,
            &result.result.normal_map, &result.result.cost_map);
      } else {
        filter_summary.num_pixels = result.result.cost_map.NumPixels();
        filter_summary.num_kept_pixels =
            result.result.cost_map.ValidPixelCount();
      }
      result.photometric_filtered_pixels =
          filter_summary.num_filtered_pixels;
      total_photometric_filtered_pixels +=
          result.photometric_filtered_pixels;
      result.stats_after_photometric_filter =
          mvs::ComputePatchMatchPhotometricStats(result.result.cost_map);
    }
  }

  std::size_t total_geometric_consistent_pixels = 0;
  std::size_t total_geometric_filtered_pixels = 0;
  if (cli_options.check_geometric_consistency) {
    mvs::PatchMatchGeometricConsistencyOptions geometric_options;
    geometric_options.max_reprojection_error_pixels =
        cli_options.geometric_max_reprojection_error_pixels;
    geometric_options.max_relative_depth_error =
        cli_options.geometric_max_relative_depth_error;

    for (PatchMatchWorkspaceResult& result : results) {
      std::vector<mvs::PatchMatchView> source_views;
      std::vector<mvs::DenseDepthMap> source_depth_maps;
      for (const ImageId source_id : result.source_image_ids) {
        const auto it = result_index_by_image_id.find(source_id);
        if (it == result_index_by_image_id.end()) {
          continue;
        }
        const PatchMatchWorkspaceResult& source_result = results[it->second];
        source_views.push_back(source_result.view);
        source_depth_maps.push_back(source_result.result.depth_map);
      }
      mvs::ConsistencyGraph graph =
          mvs::ComputeGeometricConsistencyGraph(
              result.view, source_views, result.result.depth_map,
              source_depth_maps, geometric_options);
      result.geometric_consistent_pixels = graph.NumConsistentPixels(1);
      total_geometric_consistent_pixels +=
          result.geometric_consistent_pixels;
      if (cli_options.filter_geometric_min_num_consistent_sources > 0) {
        const mvs::PatchMatchGeometricFilterSummary geometric_filter =
            mvs::FilterPatchMatchGeometricResults(
                graph,
                cli_options.filter_geometric_min_num_consistent_sources,
                &result.result.depth_map, &result.result.normal_map,
                &result.result.cost_map);
        result.geometric_filtered_pixels =
            geometric_filter.num_filtered_pixels;
        total_geometric_filtered_pixels +=
            result.geometric_filtered_pixels;
      }
      result.consistency_graph = graph;
      if (binary_dense_maps) {
        mvs::WriteConsistencyGraphBinary(
            consistency_dir / result.dense_file_name, graph);
      } else {
        WriteConsistencyGraphText(consistency_dir / result.text_file_name,
                                  graph);
      }
    }
  }

  mvs::StereoFusionSummary fusion_summary;
  if (cli_options.fuse_stereo) {
    std::vector<mvs::PatchMatchView> fusion_views;
    std::vector<mvs::MvsImage> fusion_images;
    std::vector<mvs::DenseDepthMap> fusion_depth_maps;
    std::vector<mvs::NormalMap> fusion_normal_maps;
    std::vector<mvs::DenseCostMap> fusion_cost_maps;
    std::vector<mvs::ConsistencyGraph> fusion_graphs;
    fusion_views.reserve(results.size());
    fusion_images.reserve(results.size());
    fusion_depth_maps.reserve(results.size());
    fusion_normal_maps.reserve(results.size());
    fusion_cost_maps.reserve(results.size());
    fusion_graphs.reserve(results.size());
    for (const PatchMatchWorkspaceResult& result : results) {
      fusion_views.push_back(result.view);
      fusion_images.push_back(result.image);
      fusion_depth_maps.push_back(result.result.depth_map);
      fusion_normal_maps.push_back(result.result.normal_map);
      fusion_cost_maps.push_back(result.result.cost_map);
      fusion_graphs.push_back(result.consistency_graph);
    }

    mvs::StereoFusionOptions fusion_options;
    fusion_options.min_num_consistent_sources =
        cli_options.filter_geometric_min_num_consistent_sources > 0
            ? cli_options.filter_geometric_min_num_consistent_sources
            : 1;
    fusion_options.min_num_fused_observations =
        cli_options.fusion_min_num_observations;
    fusion_options.max_num_fused_observations =
        cli_options.fusion_max_num_observations;
    fusion_options.max_traversal_depth =
        cli_options.fusion_max_traversal_depth;
    fusion_options.max_reprojection_error_pixels =
        cli_options.fusion_max_reprojection_error_pixels;
    fusion_options.max_relative_depth_error =
        cli_options.fusion_max_relative_depth_error;
    fusion_options.max_normal_error_degrees =
        cli_options.fusion_max_normal_error_degrees;
    fusion_options.max_view_angle_degrees =
        cli_options.fusion_max_view_angle_degrees;
    fusion_options.source_pixel_search_radius =
        cli_options.fusion_source_pixel_search_radius;
    const mvs::StereoFusionResult fusion = mvs::FuseStereoDepthMaps(
        fusion_views, fusion_images, fusion_depth_maps, fusion_normal_maps,
        fusion_cost_maps, fusion_graphs, fusion_options);
    fusion_summary = fusion.summary;
    WriteFusedPointCloudPly(output_path / "dense" / "fused.ply",
                            fusion.points);
    WriteFusionTracksDiagnostics(output_path / "dense" / "fusion_tracks.tsv",
                                  fusion.tracks);
    WriteFusionImagePairsDiagnostics(
        output_path / "dense" / "fusion_image_pairs.tsv",
        fusion.pair_diagnostics);
    WriteFusionTrackLengthHistogram(
        output_path / "dense" / "fusion_track_lengths.tsv",
        fusion.tracks);
    WriteFusionErrorDistribution(
        output_path / "dense" / "fusion_error_distribution.tsv",
        fusion.error_samples);
    mvs::WriteFusionTrackQualityStats(
        output_path / "dense" / "fusion_summary.txt",
        mvs::ComputeFusionTrackQualityStats(fusion.tracks));
    if (cli_options.filter_dense_cloud) {
      mvs::DenseCloudFilterOptions filter_options;
      filter_options.min_track_observations =
          cli_options.dense_cloud_min_track_observations;
      filter_options.max_normal_angle_degrees =
          cli_options.fusion_max_normal_error_degrees;
      filter_options.voxel_size = cli_options.dense_cloud_voxel_size;
      const mvs::DenseCloudFilterResult clean_cloud =
          mvs::FilterDenseCloudWithStats(fusion.points, fusion.tracks,
                                         filter_options);
      WriteFusedPointCloudPly(output_path / "dense" / "fused_clean.ply",
                              clean_cloud.points);
      WriteDenseCloudFilterStats(output_path / "dense" /
                                     "fused_clean_stats.txt",
                                 clean_cloud.stats);
      if (cli_options.mesh_dense_cloud) {
        mvs::DenseCloudMeshingOptions mesh_options;
        mesh_options.max_edge_length = cli_options.mesh_max_edge_length;
        mesh_options.max_normal_angle_degrees =
            cli_options.mesh_max_normal_angle_degrees;
        const mvs::DenseTriangleMesh mesh =
            mvs::MeshDenseCloud(clean_cloud.points, mesh_options);
        mvs::WriteDenseTriangleMeshPly(output_path / "dense" / "mesh.ply",
                                       mesh);
        mvs::WriteDenseTriangleMeshStats(
            output_path / "dense" / "mesh_stats.txt",
            mvs::ComputeDenseTriangleMeshStats(mesh));
      }
    }
  }

  std::ofstream diagnostics(output_path / "stereo" / "patch_match.tsv");
  diagnostics << "image_id\tpyramid_levels\titerations\tvalid_cost_pixels\tcandidate_evaluations"
              << "\tpropagation_updates\tperturbation_updates"
              << "\tperturbation_mean_abs_depth_update"
              << "\tperturbation_max_abs_depth_update"
              << "\tperturbation_mean_normal_angle_degrees"
              << "\tperturbation_max_normal_angle_degrees"
              << "\tperturbation_geometric_candidate_rejections"
              << "\tperturbation_geometric_candidate_acceptances"
              << "\tforward_sweeps\tbackward_sweeps"
              << "\ttop_down_sweeps\tbottom_up_sweeps"
              << "\tcheckerboard_passes"
              << "\tnormal_refinement_pixels"
              << "\tnormal_refined_pixels"
              << "\tnormal_refinement_skipped_pixels"
              << "\tvalid_cost_pixels_before_filter"
              << "\tvalid_cost_ratio_before_filter"
              << "\tmean_cost_before_filter"
              << "\tmedian_cost_before_filter"
              << "\tmax_cost_before_filter"
              << "\tphotometric_filtered_pixels"
              << "\tvalid_cost_pixels_after_filter"
              << "\tvalid_cost_ratio_after_filter"
              << "\tmean_cost_after_filter"
              << "\tmedian_cost_after_filter"
              << "\tmax_cost_after_filter"
              << "\tgeometric_consistent_pixels"
              << "\tgeometric_filtered_pixels"
              << "\tfinal_valid_cost_pixels\n";
  for (const PatchMatchWorkspaceResult& result : results) {
    const mvs::PatchMatchPhotometricStats final_stats =
        mvs::ComputePatchMatchPhotometricStats(result.result.cost_map);
    WriteDenseDepthMap(depth_dir / result.dense_file_name,
                       result.result.depth_map, binary_dense_maps);
    WriteNormalMap(normal_dir / result.dense_file_name,
                   result.result.normal_map, binary_dense_maps);
    mvs::WriteDenseCostMapText(cost_dir / result.text_file_name,
                               result.result.cost_map);
    diagnostics << result.image_id << "\t"
                << result.result.summary.num_pyramid_levels << "\t"
                << result.result.summary.num_iterations << "\t"
                << result.result.summary.num_valid_cost_pixels << "\t"
                << result.result.summary.num_candidate_evaluations << "\t"
                << result.result.summary.num_propagation_updates << "\t"
                << result.result.summary.num_perturbation_updates << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_mean_abs_depth_update)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_max_abs_depth_update)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_mean_normal_angle_degrees)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_max_normal_angle_degrees)
                << "\t"
                << result.result.summary
                       .num_perturbation_geometric_candidate_rejections
                << "\t"
                << result.result.summary
                       .num_perturbation_geometric_candidate_acceptances
                << "\t"
                << result.result.summary.num_forward_sweeps << "\t"
                << result.result.summary.num_backward_sweeps << "\t"
                << result.result.summary.num_top_down_sweeps << "\t"
                << result.result.summary.num_bottom_up_sweeps << "\t"
                << result.result.summary.num_checkerboard_passes << "\t"
                << result.result.summary.num_normal_refinement_pixels << "\t"
                << result.result.summary.num_normal_refined_pixels << "\t"
                << result.result.summary.num_normal_refinement_skipped_pixels
                << "\t"
                << result.stats_before_filter.num_valid_cost_pixels << "\t"
                << FormatStat(result.stats_before_filter.valid_ratio) << "\t"
                << FormatStat(result.stats_before_filter.mean_cost) << "\t"
                << FormatStat(result.stats_before_filter.median_cost) << "\t"
                << FormatStat(result.stats_before_filter.max_cost) << "\t"
                << result.photometric_filtered_pixels << "\t"
                << result.stats_after_photometric_filter.num_valid_cost_pixels
                << "\t"
                << FormatStat(
                       result.stats_after_photometric_filter.valid_ratio)
                << "\t"
                << FormatStat(result.stats_after_photometric_filter.mean_cost)
                << "\t"
                << FormatStat(
                       result.stats_after_photometric_filter.median_cost)
                << "\t"
                << FormatStat(result.stats_after_photometric_filter.max_cost)
                << "\t" << result.geometric_consistent_pixels << "\t"
                << result.geometric_filtered_pixels << "\t"
                << final_stats.num_valid_cost_pixels << "\n";
  }

  if (num_photometric_filtered_pixels != nullptr) {
    *num_photometric_filtered_pixels = total_photometric_filtered_pixels;
  }
  if (num_normal_refined_pixels != nullptr) {
    *num_normal_refined_pixels = total_normal_refined_pixels;
  }
  if (num_geometric_consistent_pixels != nullptr) {
    *num_geometric_consistent_pixels = total_geometric_consistent_pixels;
  }
  if (num_geometric_filtered_pixels != nullptr) {
    *num_geometric_filtered_pixels = total_geometric_filtered_pixels;
  }
  if (fusion_summary_out != nullptr) {
    *fusion_summary_out = fusion_summary;
  }
  return num_patch_match_images;
}

int RunPipeline(const ReconMvsCliOptions& cli_options, std::ostream& out) {
  const Reconstruction reconstruction =
      ReadColmapTextModel(cli_options.model_path);

  mvs::MvsPipelineOptions mvs_options;
  mvs_options.max_num_source_images = cli_options.max_num_source_images;
  mvs_options.min_num_shared_points = cli_options.min_num_shared_points;
  mvs_options.min_triangulation_angle_degrees =
      cli_options.source_min_triangulation_angle_degrees;
  mvs_options.max_triangulation_angle_degrees =
      cli_options.source_max_triangulation_angle_degrees;
  mvs_options.triangulation_angle_weight =
      cli_options.source_triangulation_angle_weight;
  mvs_options.shared_points_weight =
      cli_options.source_shared_points_weight;
  mvs_options.scale_consistency_weight =
      cli_options.source_scale_consistency_weight;
  mvs_options.coverage_weight = cli_options.source_coverage_weight;
  mvs_options.coverage_grid_size = cli_options.source_coverage_grid_size;
  const mvs::MvsPipelineSummary summary =
      mvs::RunMvsPipeline(reconstruction, mvs_options);

  const std::filesystem::path output_path(cli_options.output_path);
  std::size_t num_initialized_patch_match_images = 0;
  if (cli_options.initialize_patch_match) {
    num_initialized_patch_match_images = InitializePatchMatchMapsForWorkspace(
        output_path, cli_options, reconstruction, summary);
  }
  std::size_t num_patch_match_images = 0;
  std::size_t num_photometric_filtered_pixels = 0;
  std::size_t num_normal_refined_pixels = 0;
  std::size_t num_geometric_consistent_pixels = 0;
  std::size_t num_geometric_filtered_pixels = 0;
  mvs::StereoFusionSummary fusion_summary;
  if (cli_options.run_patch_match) {
    num_patch_match_images = RunPatchMatchForWorkspace(
        output_path, cli_options, reconstruction, summary,
        &num_photometric_filtered_pixels, &num_normal_refined_pixels,
        &num_geometric_consistent_pixels, &num_geometric_filtered_pixels,
        &fusion_summary);
  }

  WriteMvsSummary(output_path, summary, num_initialized_patch_match_images,
                  num_patch_match_images, num_photometric_filtered_pixels,
                  num_normal_refined_pixels, num_geometric_consistent_pixels,
                  num_geometric_filtered_pixels, fusion_summary);
  WriteSourceImages(output_path, summary);
  WriteSparseDepthSamples(output_path, summary);

  out << "Images: " << summary.num_images << "\n"
      << "Sparse depth samples: " << summary.num_sparse_depth_samples << "\n"
      << "Initialized PatchMatch images: "
      << num_initialized_patch_match_images << "\n"
      << "PatchMatch images: " << num_patch_match_images << "\n"
      << "Photometric filtered pixels: "
      << num_photometric_filtered_pixels << "\n"
      << "Geometric consistent pixels: "
      << num_geometric_consistent_pixels << "\n"
      << "Geometric filtered pixels: "
      << num_geometric_filtered_pixels << "\n"
      << "Fused points: " << fusion_summary.num_fused_points << "\n"
      << "Fusion tracks: " << fusion_summary.num_fusion_tracks << "\n"
      << "Fused observations: "
      << fusion_summary.num_fused_observations << "\n"
      << "Output: " << output_path.string() << "\n";
  return summary.success ? 0 : 2;
}

}  // namespace

int RunReconMvsApp(const std::vector<std::string>& args,
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
    return RunPipeline(options, out);
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return 1;
  }
}

}  // namespace colite::app
