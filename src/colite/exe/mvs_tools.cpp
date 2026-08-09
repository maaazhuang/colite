#include "colite/exe/mvs_tools.h"

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "colite/core/camera.h"
#include "colite/core/pose.h"
#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/mesh_texturer.h"
#include "colite/mvs/patch_match.h"
#include "colite/exe/stage.h"
#include "colite/io/sfm_workspace.h"

namespace colite::exe {
namespace {

using Arguments = std::unordered_map<std::string, std::string>;

Arguments ParseArguments(const std::vector<std::string>& args) {
  Arguments values;
  for (std::size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--help" || args[i] == "-h" ||
        args[i] == "--resume" || args[i] == "--force" ||
        args[i] == "--dry_run") {
      values[args[i] == "-h" ? "--help" : args[i]] = "1";
      continue;
    }
    if (args[i].rfind("--", 0) != 0 || i + 1 >= args.size()) {
      throw std::invalid_argument("Expected --option value, got: " + args[i]);
    }
    const std::string option = args[i];
    values[option] = args[++i];
  }
  return values;
}

StageOptions ParseStageOptions(const Arguments& args) {
  StageOptions options;
  options.resume = args.count("--resume") != 0;
  options.force = args.count("--force") != 0;
  options.dry_run = args.count("--dry_run") != 0;
  if (options.resume && options.force) {
    throw std::invalid_argument("--resume and --force are mutually exclusive");
  }
  return options;
}

std::vector<std::string> AlgorithmArguments(
    const std::vector<std::string>& argv) {
  std::vector<std::string> result;
  for (const std::string& argument : argv) {
    if (argument != "--resume" && argument != "--force" &&
        argument != "--dry_run") {
      result.push_back(argument);
    }
  }
  return result;
}

std::filesystem::path StageRoot(const Arguments& args,
                                const std::filesystem::path& output,
                                bool output_is_directory = false) {
  if (args.count("--stage_root")) return args.at("--stage_root");
  return output_is_directory ? output : output.parent_path();
}

std::string Require(const Arguments& args, const std::string& name) {
  const auto it = args.find(name);
  if (it == args.end() || it->second.empty()) {
    throw std::invalid_argument("Missing required option: " + name);
  }
  return it->second;
}

std::filesystem::path ResolveModelPath(std::filesystem::path path) {
  if (!std::filesystem::exists(path / "cameras.txt") &&
      !std::filesystem::exists(path / "cameras.bin") &&
      std::filesystem::is_directory(path / "0")) {
    path /= "0";
  }
  return path;
}

template <typename T>
T Number(const Arguments& args, const std::string& name, T fallback) {
  const auto it = args.find(name);
  if (it == args.end()) return fallback;
  std::istringstream stream(it->second);
  T value{};
  if (!(stream >> value) || !stream.eof()) {
    throw std::invalid_argument("Invalid value for " + name + ": " + it->second);
  }
  return value;
}

struct ModelImage {
  std::string name;
  mvs::PatchMatchView view;
};

std::vector<ModelImage> ReadModelImages(const std::filesystem::path& model_path,
                                        bool load_pixels,
                                        const std::filesystem::path& image_path,
                                        std::vector<mvs::MeshTextureView>* textures) {
  const Reconstruction reconstruction = ReadColmapReconstruction(model_path);
  std::vector<ModelImage> images;
  images.reserve(reconstruction.RegisteredImageIds().size());
  for (const ImageId image_id : reconstruction.RegisteredImageIds()) {
    const Image& source = reconstruction.GetImage(image_id);
    if (!source.cam_from_world.has_value()) {
      throw std::runtime_error("Registered model image has no pose");
    }
    ModelImage image;
    image.name = source.name;
    image.view.image_id = source.id;
    image.view.camera = reconstruction.GetCamera(source.camera_id);
    image.view.cam_from_world = *source.cam_from_world;
    images.push_back(std::move(image));
  }
  for (const ModelImage& image : images) {
    if (load_pixels) {
      cv::Mat pixels = cv::imread((image_path / image.name).string(),
                                  cv::IMREAD_COLOR);
      if (pixels.empty()) {
        throw std::runtime_error("Failed to read texture image: " +
                                 (image_path / image.name).string());
      }
      if (pixels.cols != static_cast<int>(image.view.camera.width) ||
          pixels.rows != static_cast<int>(image.view.camera.height)) {
        throw std::runtime_error("Texture image dimensions do not match model: " +
                                 image.name);
      }
      textures->push_back({image.view, std::move(pixels)});
    }
  }
  return images;
}

template <typename Function>
int GuardedRun(Function&& function, std::ostream& err) {
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << '\n';
    return 1;
  }
}

}  // namespace

int RunPoissonMesher(const std::vector<std::string>& argv,
                     std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseArguments(argv);
    if (args.count("--help")) {
      out << "Usage: poisson_mesher --input_path fused.ply --output_path mesh.ply "
             "[--PoissonMeshing.depth 13] [--PoissonMeshing.point_weight 1] "
             "[--PoissonMeshing.color 1] [--PoissonMeshing.trim 10] "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path input = Require(args, "--input_path");
    const std::filesystem::path output = Require(args, "--output_path");
    mvs::PoissonMeshingOptions options;
    options.depth = Number(args, "--PoissonMeshing.depth", options.depth);
    options.point_weight = Number(args, "--PoissonMeshing.point_weight", options.point_weight);
    options.color = Number<int>(args, "--PoissonMeshing.color", options.color ? 1 : 0) != 0;
    options.trim = Number(args, "--PoissonMeshing.trim", options.trim);
    options.num_threads = Number(args, "--PoissonMeshing.num_threads", options.num_threads);
    StageSpec spec{"poisson_mesher", StageRoot(args, output), {input},
                   {output}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      if (!mvs::PoissonMeshing(options, input, output)) {
        throw std::runtime_error("Poisson reconstruction failed");
      }
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunMeshSimplifier(const std::vector<std::string>& argv,
                      std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseArguments(argv);
    if (args.count("--help")) {
      out << "Usage: mesh_simplifier --input_path mesh.ply --output_path simplified.ply "
             "[--MeshSimplification.target_ratio 0.5] "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path input_path = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    mvs::MeshSimplificationOptions options;
    options.target_ratio = Number(args, "--MeshSimplification.target_ratio", options.target_ratio);
    options.target_num_triangles = Number<std::size_t>(args, "--MeshSimplification.target_num_triangles", 0);
    options.max_error = Number(args, "--MeshSimplification.max_error", options.max_error);
    options.preserve_boundary = Number<int>(args, "--MeshSimplification.preserve_boundary", 1) != 0;
    options.max_normal_change_degrees = Number(args, "--MeshSimplification.max_normal_change", options.max_normal_change_degrees);
    StageSpec spec{"mesh_simplifier", StageRoot(args, output_path),
                   {input_path}, {output_path}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      const mvs::DenseTriangleMesh input =
          mvs::ReadDenseTriangleMeshPly(input_path);
      mvs::WriteDenseTriangleMeshPly(output_path,
                                     mvs::SimplifyMesh(input, options));
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunDelaunayMesher(const std::vector<std::string>& argv,
                      std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseArguments(argv);
    if (args.count("--help")) {
      out << "Usage: delaunay_mesher --input_path dense_workspace --output_path mesh.ply "
             "[--model_path dense_workspace/sparse] "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path workspace = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    const std::filesystem::path model = ResolveModelPath(
        args.count("--model_path")
            ? std::filesystem::path(args.at("--model_path"))
            : workspace / "sparse");
    const std::filesystem::path visibility_path =
        std::filesystem::exists(workspace / "points_visibility.bin")
            ? workspace / "points_visibility.bin"
            : workspace / "fused.ply.vis";
    mvs::DelaunayMeshingOptions options;
    options.max_proj_dist = Number(args, "--DelaunayMeshing.max_proj_dist", options.max_proj_dist);
    options.max_depth_dist = Number(args, "--DelaunayMeshing.max_depth_dist", options.max_depth_dist);
    options.visibility_sigma = Number(args, "--DelaunayMeshing.visibility_sigma", options.visibility_sigma);
    options.distance_sigma_factor = Number(args, "--DelaunayMeshing.distance_sigma_factor", options.distance_sigma_factor);
    options.quality_regularization = Number(args, "--DelaunayMeshing.quality_regularization", options.quality_regularization);
    options.max_side_length_factor = Number(args, "--DelaunayMeshing.max_side_length_factor", options.max_side_length_factor);
    options.max_side_length_percentile = Number(args, "--DelaunayMeshing.max_side_length_percentile", options.max_side_length_percentile);
    options.num_threads = Number(args, "--DelaunayMeshing.num_threads", options.num_threads);
    StageSpec spec{"delaunay_mesher", StageRoot(args, output_path),
                   {workspace / "fused.ply", visibility_path, model},
                   {output_path}, AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      const mvs::DenseTriangleMesh cloud =
          mvs::ReadDenseTriangleMeshPly(workspace / "fused.ply");
      const auto visibility =
          mvs::ReadPointsVisibilityColmapBinary(visibility_path);
      std::vector<mvs::MeshTextureView> unused;
      const auto model_images = ReadModelImages(model, false, {}, &unused);
      std::vector<mvs::PatchMatchView> views;
      views.reserve(model_images.size());
      for (const ModelImage& image : model_images) views.push_back(image.view);
      mvs::WriteDenseTriangleMeshPly(
          output_path,
          mvs::DelaunayMeshing(options, cloud.vertices, visibility, views));
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

int RunMeshTexturer(const std::vector<std::string>& argv,
                    std::ostream& out, std::ostream& err) {
  int stage_result = 0;
  const int guard_result = GuardedRun([&] {
    const Arguments args = ParseArguments(argv);
    if (args.count("--help")) {
      out << "Usage: mesh_texturer --input_path mesh.ply --output_path output_dir "
             "--model_path sparse --image_path images "
             "[--resume|--force] [--dry_run]\n";
      return;
    }
    const std::filesystem::path input_path = Require(args, "--input_path");
    const std::filesystem::path output_path = Require(args, "--output_path");
    const std::filesystem::path model_path =
        ResolveModelPath(Require(args, "--model_path"));
    const std::filesystem::path image_path = Require(args, "--image_path");
    mvs::MeshTexturingOptions options;
    options.atlas_size = Number(args, "--MeshTextureMapping.atlas_size", options.atlas_size);
    options.max_atlas_size = Number(args, "--MeshTextureMapping.max_atlas_size", options.max_atlas_size);
    options.atlas_patch_padding = Number(args, "--MeshTextureMapping.atlas_patch_padding", options.atlas_patch_padding);
    options.inpaint_radius = Number(args, "--MeshTextureMapping.inpaint_radius", options.inpaint_radius);
    options.texture_scale = Number(args, "--MeshTextureMapping.texture_scale_factor", options.texture_scale);
    options.min_cos_normal_angle = Number(args, "--MeshTextureMapping.min_cos_normal_angle", options.min_cos_normal_angle);
    options.min_visible_vertices = Number(args, "--MeshTextureMapping.min_visible_vertices", options.min_visible_vertices);
    options.view_selection_smoothing_iterations = Number(args, "--MeshTextureMapping.view_selection_smoothing_iterations", options.view_selection_smoothing_iterations);
    options.view_selection_min_component_size = Number(
        args, "--MeshTextureMapping.view_selection_min_component_size",
        options.view_selection_min_component_size);
    options.visibility_resolution = Number(
        args, "--MeshTextureMapping.visibility_resolution",
        options.visibility_resolution);
    options.occlusion_relative_tolerance = Number(
        args, "--MeshTextureMapping.occlusion_relative_tolerance",
        options.occlusion_relative_tolerance);
    options.use_vertex_color_fallback = Number<int>(
        args, "--MeshTextureMapping.use_vertex_color_fallback",
        options.use_vertex_color_fallback ? 1 : 0) != 0;
    options.apply_color_correction = Number<int>(args, "--MeshTextureMapping.apply_color_correction", 1) != 0;
    options.color_correction_regularization = Number(args, "--MeshTextureMapping.color_correction_regularization", options.color_correction_regularization);
    options.num_threads = Number(args, "--MeshTextureMapping.num_threads", options.num_threads);
    StageSpec spec{"mesh_texturer", StageRoot(args, output_path, true),
                   {input_path, model_path, image_path},
                   {output_path / "mesh.obj", output_path / "mesh.mtl",
                    output_path / "texture.png"},
                   AlgorithmArguments(argv)};
    stage_result = RunStage(spec, ParseStageOptions(args), [&] {
      std::vector<mvs::MeshTextureView> views;
      ReadModelImages(model_path, true, image_path, &views);
      const mvs::DenseTriangleMesh mesh =
          mvs::ReadDenseTriangleMeshPly(input_path);
      mvs::WriteTexturedTriangleMesh(output_path,
                                     mvs::TextureMesh(mesh, views, options));
      return 0;
    }, out, err);
  }, err);
  return guard_result != 0 ? guard_result : stage_result;
}

}  // namespace colite::exe
