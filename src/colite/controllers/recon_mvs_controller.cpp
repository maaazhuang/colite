#include "colite/controllers/recon_mvs_controller.h"

#include <algorithm>
#include <array>
#include <chrono>
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
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "colite/core/camera.h"
#include "colite/core/geometry.h"
#include "colite/core/point2d.h"
#include "colite/core/point3d.h"
#include "colite/core/reconstruction.h"
#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/dense_map.h"
#include "colite/mvs/image_workspace.h"
#include "colite/mvs/image_undistorter.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/mesh_texturer.h"
#include "colite/mvs/mvs_pipeline.h"
#include "colite/mvs/patch_match.h"
#include "colite/mvs/patch_match_cuda.h"

namespace colite::controllers {

ReconMvsController::ReconMvsController(ReconMvsControllerOptions options)
    : options_(std::move(options)) {}

const ReconMvsControllerOptions& ReconMvsController::Options() const {
  return options_;
}

namespace {
using ControllerClock = std::chrono::steady_clock;

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

void WriteColmapTextModel(const std::filesystem::path& model_path,
                          const Reconstruction& reconstruction) {
  std::filesystem::create_directories(model_path);
  std::ofstream cameras(model_path / "cameras.txt");
  std::ofstream images(model_path / "images.txt");
  std::ofstream points(model_path / "points3D.txt");
  if (!cameras || !images || !points) {
    throw std::runtime_error("Failed to write undistorted sparse model.");
  }
  cameras << std::setprecision(17);
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    cameras << camera_id << ' ' << CameraModelIdToName(camera.model_id) << ' '
            << camera.width << ' ' << camera.height;
    for (const double parameter : camera.params) cameras << ' ' << parameter;
    cameras << '\n';
  }
  images << std::setprecision(17);
  for (const ImageId image_id : reconstruction.RegisteredImageIds()) {
    const Image& image = reconstruction.GetImage(image_id);
    const Pose3& pose = *image.cam_from_world;
    images << image.id << ' ' << pose.q_cw.w() << ' ' << pose.q_cw.x() << ' '
           << pose.q_cw.y() << ' ' << pose.q_cw.z() << ' ' << pose.t_cw.x()
           << ' ' << pose.t_cw.y() << ' ' << pose.t_cw.z() << ' '
           << image.camera_id << ' ' << image.name << '\n';
    for (const Point2D& point : image.points2d) {
      images << point.xy.x() << ' ' << point.xy.y() << ' ';
      if (point.HasPoint3D()) images << point.point3d_id;
      else images << -1;
      images << ' ';
    }
    images << '\n';
  }
  points << std::setprecision(17);
  for (const auto& [point_id, point] : reconstruction.Points3D()) {
    points << point_id << ' ' << point.xyz.x() << ' ' << point.xyz.y() << ' '
           << point.xyz.z() << ' ' << static_cast<int>(point.color.x()) << ' '
           << static_cast<int>(point.color.y()) << ' '
           << static_cast<int>(point.color.z()) << ' '
           << point.reprojection_error;
    for (const TrackElement& element : point.track.elements) {
      points << ' ' << element.image_id << ' ' << element.point2d_idx;
    }
    points << '\n';
  }
}

std::size_t UndistortWorkspaceImages(
    const ReconMvsControllerOptions& cli_options,
    const std::filesystem::path& output_path,
    Reconstruction* reconstruction) {
  mvs::ImageUndistortionOptions options;
  options.blank_pixels = cli_options.undistort_blank_pixels;
  options.min_scale = cli_options.undistort_min_scale;
  options.max_scale = cli_options.undistort_max_scale;
  options.max_image_size = cli_options.undistort_max_image_size;
  options.resolution_level = cli_options.resolution_level;
  const auto original_cameras = reconstruction->Cameras();
  const std::filesystem::path image_output =
      output_path / "undistorted" / "images";
  std::filesystem::create_directories(image_output);
  std::ofstream diagnostics(output_path / "undistorted" / "images.tsv");
  diagnostics << "image_id\tname\twidth\theight\tvalid_pixel_ratio\n";
  std::size_t count = 0;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    const Camera& original_camera = original_cameras.at(image.camera_id);
    const std::filesystem::path source_path =
        image.file_path.empty() ? cli_options.image_path / image.name
                                : std::filesystem::path(image.file_path);
    cv::Mat source = cv::imread(source_path.string(), cv::IMREAD_COLOR);
    if (source.empty()) {
      throw std::runtime_error("Failed to read image for undistortion: " +
                               source_path.string());
    }
    const cv::Size expected_size(static_cast<int>(original_camera.width),
                                 static_cast<int>(original_camera.height));
    if (source.size() != expected_size) {
      cv::resize(source, source, expected_size, 0.0, 0.0, cv::INTER_AREA);
    }
    mvs::UndistortedImage undistorted =
        mvs::UndistortImage(original_camera, source, options);
    const std::filesystem::path image_output_path = image_output / image.name;
    std::filesystem::create_directories(image_output_path.parent_path());
    if (!cv::imwrite(image_output_path.string(), undistorted.image)) {
      throw std::runtime_error("Failed to write undistorted image: " +
                               image_output_path.string());
    }
    reconstruction->GetCamera(image.camera_id) = undistorted.camera;
    image.file_path = image_output_path.string();
    diagnostics << image.id << '\t' << image.name << '\t'
                << undistorted.camera.width << '\t'
                << undistorted.camera.height << '\t'
                << std::setprecision(17) << undistorted.valid_pixel_ratio
                << '\n';
    ++count;
  }
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    const Camera& camera = reconstruction->GetCamera(image.camera_id);
    const Camera& original_camera = original_cameras.at(image.camera_id);
    for (Point2D& point : image.points2d) {
      const std::optional<Eigen::Vector2d> undistorted =
          mvs::UndistortPixel(original_camera, camera, point.xy);
      if (undistorted.has_value()) point.xy = *undistorted;
      if (!point.HasPoint3D() ||
          !reconstruction->ExistsPoint3D(point.point3d_id)) {
        continue;
      }
      const std::optional<Eigen::Vector2d> projected = ProjectWorldPoint(
          camera, *image.cam_from_world,
          reconstruction->GetPoint3D(point.point3d_id).xyz);
      if (projected.has_value()) point.xy = *projected;
    }
  }
  WriteColmapTextModel(output_path / "undistorted" / "sparse",
                       *reconstruction);
  return count;
}

struct PatchMatchSchedulingSummary {
  double cuda_warmup_time_ms = 0.0;
  double initial_wall_time_ms = 0.0;
  double initial_sum_cuda_time_ms = 0.0;
  double initial_overlap_saved_time_ms = 0.0;
  double initial_overlap_ratio = 0.0;
  std::size_t initial_deferred_photometric_stats_images = 0;
  double initial_source_setup_sum_time_ms = 0.0;
  double initial_stereo_run_sum_time_ms = 0.0;
  double initial_stereo_wrapper_overhead_sum_time_ms = 0.0;
  double initial_postprocess_sum_time_ms = 0.0;
  double refinement_wall_time_ms = 0.0;
  double refinement_sum_cuda_time_ms = 0.0;
  double refinement_overlap_saved_time_ms = 0.0;
  double refinement_overlap_ratio = 0.0;
};

void WriteMvsSummary(const std::filesystem::path& output_path,
                     const mvs::MvsPipelineSummary& summary,
                     std::size_t num_undistorted_images,
                     std::size_t num_initialized_patch_match_images,
                     std::size_t num_patch_match_images,
                     std::size_t num_photometric_filtered_pixels,
                     std::size_t num_normal_refined_pixels,
                     std::size_t num_geometric_consistent_pixels,
                     std::size_t num_geometric_filtered_pixels,
                     const PatchMatchSchedulingSummary& scheduling_summary,
                     const mvs::StereoFusionSummary& fusion_summary) {
  std::filesystem::create_directories(output_path);
  std::ofstream file(output_path / "summary.txt");
  file << std::setprecision(17)
       << "success " << (summary.success ? 1 : 0) << "\n"
       << "num_images " << summary.num_images << "\n"
       << "num_sparse_depth_samples " << summary.num_sparse_depth_samples
       << "\n"
       << "undistorted_images " << num_undistorted_images << "\n"
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
       << "cuda_warmup_time_ms "
       << scheduling_summary.cuda_warmup_time_ms
       << "\n"
       << "patch_match_initial_wall_time_ms "
       << scheduling_summary.initial_wall_time_ms
       << "\n"
       << "patch_match_initial_sum_cuda_time_ms "
       << scheduling_summary.initial_sum_cuda_time_ms
       << "\n"
       << "patch_match_initial_overlap_saved_time_ms "
       << scheduling_summary.initial_overlap_saved_time_ms
       << "\n"
       << "patch_match_initial_overlap_ratio "
       << scheduling_summary.initial_overlap_ratio
       << "\n"
       << "patch_match_initial_deferred_photometric_stats_images "
       << scheduling_summary.initial_deferred_photometric_stats_images
       << "\n"
       << "patch_match_initial_source_setup_sum_time_ms "
       << scheduling_summary.initial_source_setup_sum_time_ms
       << "\n"
       << "patch_match_initial_stereo_run_sum_time_ms "
       << scheduling_summary.initial_stereo_run_sum_time_ms
       << "\n"
       << "patch_match_initial_stereo_wrapper_overhead_sum_time_ms "
       << scheduling_summary.initial_stereo_wrapper_overhead_sum_time_ms
       << "\n"
       << "patch_match_initial_postprocess_sum_time_ms "
       << scheduling_summary.initial_postprocess_sum_time_ms
       << "\n"
       << "patch_match_refinement_wall_time_ms "
       << scheduling_summary.refinement_wall_time_ms
       << "\n"
       << "patch_match_refinement_sum_cuda_time_ms "
       << scheduling_summary.refinement_sum_cuda_time_ms
       << "\n"
       << "patch_match_refinement_overlap_saved_time_ms "
       << scheduling_summary.refinement_overlap_saved_time_ms
       << "\n"
       << "patch_match_refinement_overlap_ratio "
       << scheduling_summary.refinement_overlap_ratio
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

bool UseBinaryDenseMaps(const ReconMvsControllerOptions& cli_options);

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
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
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

std::vector<mvs::MeshTextureView> LoadMeshTextureViews(
    const Reconstruction& reconstruction,
    const std::vector<mvs::PatchMatchView>& geometry_views,
    const std::filesystem::path& image_path) {
  std::vector<mvs::MeshTextureView> texture_views;
  texture_views.reserve(geometry_views.size());
  for (const mvs::PatchMatchView& geometry : geometry_views) {
    const Image& reconstruction_image =
        reconstruction.GetImage(geometry.image_id);
    const std::filesystem::path path =
        reconstruction_image.file_path.empty()
            ? image_path / reconstruction_image.name
            : std::filesystem::path(reconstruction_image.file_path);
    cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read texture image: " +
                               path.string());
    }
    const cv::Size expected_size(static_cast<int>(geometry.camera.width),
                                 static_cast<int>(geometry.camera.height));
    if (image.size() != expected_size) {
      cv::resize(image, image, expected_size, 0.0, 0.0, cv::INTER_AREA);
    }
    texture_views.push_back({geometry, std::move(image)});
  }
  return texture_views;
}

std::string FormatStat(double value) {
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

template <typename T>
std::string JoinDiagnosticVector(const std::vector<T>& values) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << values[i];
  }
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

bool UseBinaryDenseMaps(const ReconMvsControllerOptions& cli_options) {
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
  bool photometric_stats_deferred = false;
  double initial_source_setup_time_ms = 0.0;
  double initial_stereo_run_time_ms = 0.0;
  double initial_postprocess_time_ms = 0.0;
  std::size_t geometric_consistent_pixels = 0;
  std::size_t geometric_filtered_pixels = 0;
};

void WriteInitialPatchMatchDiagnostics(
    const std::filesystem::path& output_path,
    const std::vector<PatchMatchWorkspaceResult>& results) {
  std::filesystem::create_directories(output_path / "stereo");
  std::ofstream diagnostics(output_path / "stereo" /
                            "patch_match_initial.tsv");
  diagnostics << "image_id\trequested_backend\tactual_backend"
              << "\tpyramid_levels\titerations\tvalid_cost_pixels"
              << "\tplane_propagation_candidates"
              << "\tperturbation_candidate_evaluations"
              << "\tsource_selection_probability_updates"
              << "\titeration_source_selection_probability_means"
              << "\tfinal_source_selection_probability_means"
              << "\tcuda_upload_time_ms"
              << "\tcuda_initialization_time_ms"
              << "\tcuda_evaluate_cost_time_ms"
              << "\tcuda_propagation_time_ms"
              << "\tcuda_perturbation_time_ms"
              << "\tcuda_download_time_ms"
              << "\tcuda_total_time_ms"
              << "\tcuda_wrapper_total_time_ms"
              << "\tcuda_wrapper_overhead_time_ms"
              << "\tcuda_request_validation_time_ms"
              << "\tcuda_pyramid_resize_time_ms"
              << "\tcuda_host_pack_time_ms"
              << "\tcuda_single_scale_dispatch_time_ms"
              << "\tcuda_result_materialization_time_ms"
              << "\tcuda_output_to_vector_time_ms"
              << "\tcuda_dense_map_fill_time_ms"
              << "\tcuda_normal_refinement_time_ms"
              << "\tcuda_valid_cost_count_time_ms"
              << "\tcuda_host_initial_normal_upload_normalizations"
              << "\tcuda_host_output_normalizations"
              << "\tcuda_device_count_queries"
              << "\tcuda_device_count_cache_hits"
              << "\tcuda_workspace_pool_size"
              << "\tcuda_workspace_pool_slot\n";
  for (const PatchMatchWorkspaceResult& result : results) {
    const mvs::PatchMatchStereoSummary& summary = result.result.summary;
    const double wrapper_overhead_time_ms =
        std::max(0.0,
                 summary.cuda_wrapper_total_time_ms -
                     summary.cuda_total_time_ms);
    diagnostics << result.image_id << "\t"
                << mvs::PatchMatchBackendToName(summary.requested_backend)
                << "\t"
                << mvs::PatchMatchBackendToName(summary.actual_backend)
                << "\t"
                << summary.num_pyramid_levels << "\t"
                << summary.num_iterations << "\t"
                << summary.num_valid_cost_pixels << "\t"
                << summary.num_plane_propagation_candidates << "\t"
                << summary.num_perturbation_candidate_evaluations << "\t"
                << summary.num_source_selection_probability_updates << "\t"
                << JoinDiagnosticVector(
                       summary.iteration_source_selection_probability_means)
                << "\t"
                << JoinDiagnosticVector(
                       summary.final_source_selection_probability_means)
                << "\t"
                << FormatStat(summary.cuda_upload_time_ms) << "\t"
                << FormatStat(summary.cuda_initialization_time_ms) << "\t"
                << FormatStat(summary.cuda_evaluate_cost_time_ms) << "\t"
                << FormatStat(summary.cuda_propagation_time_ms) << "\t"
                << FormatStat(summary.cuda_perturbation_time_ms) << "\t"
                << FormatStat(summary.cuda_download_time_ms) << "\t"
                << FormatStat(summary.cuda_total_time_ms) << "\t"
                << FormatStat(summary.cuda_wrapper_total_time_ms) << "\t"
                << FormatStat(wrapper_overhead_time_ms) << "\t"
                << FormatStat(summary.cuda_request_validation_time_ms)
                << "\t"
                << FormatStat(summary.cuda_pyramid_resize_time_ms) << "\t"
                << FormatStat(summary.cuda_host_pack_time_ms) << "\t"
                << FormatStat(summary.cuda_single_scale_dispatch_time_ms)
                << "\t"
                << FormatStat(summary.cuda_result_materialization_time_ms)
                << "\t"
                << FormatStat(summary.cuda_output_to_vector_time_ms) << "\t"
                << FormatStat(summary.cuda_dense_map_fill_time_ms) << "\t"
                << FormatStat(summary.cuda_normal_refinement_time_ms) << "\t"
                << FormatStat(summary.cuda_valid_cost_count_time_ms) << "\t"
                << summary.cuda_host_initial_normal_upload_normalizations
                << "\t"
                << summary.cuda_host_output_normalizations << "\t"
                << summary.cuda_device_count_queries << "\t"
                << summary.cuda_device_count_cache_hits << "\t"
                << summary.cuda_workspace_pool_size << "\t"
                << summary.cuda_workspace_pool_slot << "\n";
  }
}

using CudaWorkspacePool = std::vector<mvs::CudaPatchMatchWorkspacePtr>;

bool UsesCudaWorkspacePool(const mvs::PatchMatchStereoOptions& options) {
  return options.backend != mvs::PatchMatchBackend::kCpu;
}

CudaWorkspacePool CreateCudaWorkspacePool(
    const mvs::PatchMatchStereoOptions& options,
    const ReconMvsControllerOptions& cli_options,
    std::size_t num_images) {
  CudaWorkspacePool pool;
  if (!UsesCudaWorkspacePool(options) || num_images == 0) {
    return pool;
  }
  const std::size_t pool_size = std::max<std::size_t>(
      1, std::min({std::max<std::size_t>(1, cli_options.num_threads),
                   num_images, static_cast<std::size_t>(2)}));
  pool.reserve(pool_size);
  for (std::size_t i = 0; i < pool_size; ++i) {
    pool.push_back(mvs::MakeCudaPatchMatchWorkspace());
  }
  return pool;
}

void FillOverlapSummary(double wall_time_ms,
                        double sum_cuda_time_ms,
                        double* wall_time_out,
                        double* sum_cuda_time_out,
                        double* overlap_saved_time_out,
                        double* overlap_ratio_out) {
  *wall_time_out = wall_time_ms;
  *sum_cuda_time_out = sum_cuda_time_ms;
  *overlap_saved_time_out = std::max(0.0, sum_cuda_time_ms - wall_time_ms);
  *overlap_ratio_out =
      sum_cuda_time_ms <= 0.0 ? 0.0
                              : *overlap_saved_time_out / sum_cuda_time_ms;
}

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
    const ReconMvsControllerOptions& cli_options,
    bool binary_dense_maps,
    bool run_geometric_refinement,
    mvs::CudaPatchMatchWorkspace* cuda_workspace,
    std::size_t cuda_workspace_pool_size,
    std::size_t cuda_workspace_pool_slot) {
  const ControllerClock::time_point source_setup_start =
      ControllerClock::now();
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
  const double source_setup_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                source_setup_start)
          .count();

  const ControllerClock::time_point stereo_run_start = ControllerClock::now();
  mvs::PatchMatchStereoResult result =
      mvs::RunPatchMatchStereoForImageWithCudaWorkspace(
      reference_view, source_views, reference_image, source_images,
      static_cast<float>(image_summary.min_depth),
      static_cast<float>(image_summary.max_depth), cuda_workspace,
      patch_match_options);
  if (result.summary.actual_backend == mvs::PatchMatchBackend::kCuda &&
      cuda_workspace_pool_size > 0) {
    result.summary.cuda_workspace_pool_size = cuda_workspace_pool_size;
    result.summary.cuda_workspace_pool_slot = cuda_workspace_pool_slot;
  }
  const double stereo_run_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                stereo_run_start)
          .count();

  const ControllerClock::time_point postprocess_start = ControllerClock::now();
  mvs::PatchMatchPhotometricFilterSummary filter_summary;
  mvs::PatchMatchPhotometricStats stats_before;
  mvs::PatchMatchPhotometricStats stats_after;
  const bool defer_photometric_stats = run_geometric_refinement;
  if (!defer_photometric_stats) {
    stats_before = mvs::ComputePatchMatchPhotometricStats(result.cost_map);
    if (cli_options.filter_photometric_max_cost >= 0.0F) {
      filter_summary = mvs::FilterPatchMatchPhotometricResults(
          cli_options.filter_photometric_max_cost, &result.depth_map,
          &result.normal_map, &result.cost_map);
    } else {
      filter_summary.num_pixels = result.cost_map.NumPixels();
      filter_summary.num_kept_pixels = result.cost_map.ValidPixelCount();
    }
    stats_after = mvs::ComputePatchMatchPhotometricStats(result.cost_map);
  } else {
    filter_summary.num_pixels = result.cost_map.NumPixels();
    filter_summary.num_kept_pixels = result.cost_map.ValidPixelCount();
  }

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
  workspace_result.photometric_stats_deferred = defer_photometric_stats;
  workspace_result.initial_source_setup_time_ms = source_setup_time_ms;
  workspace_result.initial_stereo_run_time_ms = stereo_run_time_ms;
  workspace_result.initial_postprocess_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                postprocess_start)
          .count();
  return workspace_result;
}

std::vector<PatchMatchWorkspaceResult> RunInitialPatchMatchForWorkspace(
    const std::vector<const mvs::ImageMvsSummary*>& image_summaries,
    const std::unordered_map<ImageId, mvs::MvsImage>& images,
    const std::unordered_map<ImageId, mvs::PatchMatchView>& views,
    const mvs::PatchMatchStereoOptions& patch_match_options,
    const ReconMvsControllerOptions& cli_options,
    bool binary_dense_maps,
    bool run_geometric_refinement,
    const std::vector<mvs::CudaPatchMatchWorkspace*>& cuda_workspace_pool,
    PatchMatchSchedulingSummary* scheduling_summary) {
  const ControllerClock::time_point batch_start = ControllerClock::now();
  std::vector<PatchMatchWorkspaceResult> results;
  results.reserve(image_summaries.size());
  const std::size_t num_threads =
      std::max<std::size_t>(1, cli_options.num_threads);
  if (!cuda_workspace_pool.empty()) {
    const std::size_t pool_size = cuda_workspace_pool.size();
    std::vector<std::future<PatchMatchWorkspaceResult>> futures;
    futures.reserve(pool_size);
    std::size_t next_slot = 0;
    const auto collect_oldest_future = [&results, &futures]() {
      results.push_back(futures.front().get());
      futures.erase(futures.begin());
    };
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      if (futures.size() >= pool_size) {
        collect_oldest_future();
      }
      const std::size_t slot = next_slot;
      next_slot = (next_slot + 1) % pool_size;
      futures.push_back(std::async(
          std::launch::async,
          [&images, &views, &patch_match_options, &cli_options,
           binary_dense_maps, run_geometric_refinement, image_summary,
           &cuda_workspace_pool, pool_size, slot]() {
            return RunInitialPatchMatchForImage(
                *image_summary, images, views, patch_match_options,
                cli_options, binary_dense_maps, run_geometric_refinement,
                cuda_workspace_pool[slot], pool_size, slot);
          }));
    }
    while (!futures.empty()) {
      collect_oldest_future();
    }
  } else if (num_threads == 1 ||
      image_summaries.size() <= 1) {
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      results.push_back(RunInitialPatchMatchForImage(
          *image_summary, images, views, patch_match_options, cli_options,
          binary_dense_maps, run_geometric_refinement, nullptr, 0, 0));
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
                cli_options, binary_dense_maps, run_geometric_refinement,
                nullptr, 0, 0);
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
  if (scheduling_summary != nullptr) {
    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                  batch_start)
            .count();
    double sum_cuda_time_ms = 0.0;
    for (const PatchMatchWorkspaceResult& result : results) {
      if (result.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda) {
        sum_cuda_time_ms += result.result.summary.cuda_total_time_ms;
        scheduling_summary->initial_stereo_wrapper_overhead_sum_time_ms +=
            std::max(0.0, result.initial_stereo_run_time_ms -
                              result.result.summary.cuda_total_time_ms);
      }
      scheduling_summary->initial_source_setup_sum_time_ms +=
          result.initial_source_setup_time_ms;
      scheduling_summary->initial_stereo_run_sum_time_ms +=
          result.initial_stereo_run_time_ms;
      scheduling_summary->initial_postprocess_sum_time_ms +=
          result.initial_postprocess_time_ms;
    }
    FillOverlapSummary(
        wall_time_ms, sum_cuda_time_ms,
        &scheduling_summary->initial_wall_time_ms,
        &scheduling_summary->initial_sum_cuda_time_ms,
        &scheduling_summary->initial_overlap_saved_time_ms,
        &scheduling_summary->initial_overlap_ratio);
  }
  return results;
}

struct PatchMatchRefinementJob {
  std::size_t result_index = 0;
  mvs::PatchMatchView reference_view;
  mvs::MvsImage reference_image;
  std::vector<mvs::PatchMatchView> source_views;
  std::vector<mvs::MvsImage> source_images;
  std::vector<std::size_t> source_result_indices;
  float min_depth = 0.0F;
  float max_depth = 0.0F;
};

struct PatchMatchRefinementResult {
  std::size_t result_index = 0;
  mvs::PatchMatchStereoResult result;
};

PatchMatchRefinementResult RunPatchMatchRefinementJob(
    const PatchMatchRefinementJob& job,
    const std::vector<mvs::DenseDepthMap>& initial_depth_maps,
    const mvs::PatchMatchStereoOptions& refinement_options,
    mvs::CudaPatchMatchWorkspace* cuda_workspace,
    std::size_t cuda_workspace_pool_size,
    std::size_t cuda_workspace_pool_slot) {
  PatchMatchRefinementResult output;
  output.result_index = job.result_index;
  std::vector<mvs::DenseDepthMap> source_depth_maps;
  source_depth_maps.reserve(job.source_result_indices.size());
  for (const std::size_t source_result_index : job.source_result_indices) {
    source_depth_maps.push_back(initial_depth_maps.at(source_result_index));
  }
  output.result =
      mvs::RunPatchMatchStereoForImageWithSourceDepthsAndCudaWorkspace(
          job.reference_view, job.source_views, job.reference_image,
          job.source_images, source_depth_maps, job.min_depth,
          job.max_depth, cuda_workspace, refinement_options);
  if (output.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda &&
      cuda_workspace_pool_size > 0) {
    output.result.summary.cuda_workspace_pool_size =
        cuda_workspace_pool_size;
    output.result.summary.cuda_workspace_pool_slot =
        cuda_workspace_pool_slot;
  }
  return output;
}

std::vector<PatchMatchRefinementResult> RunPatchMatchRefinementJobs(
    const std::vector<PatchMatchRefinementJob>& jobs,
    const std::vector<mvs::DenseDepthMap>& initial_depth_maps,
    const mvs::PatchMatchStereoOptions& refinement_options,
    const ReconMvsControllerOptions& cli_options,
    const std::vector<mvs::CudaPatchMatchWorkspace*>& cuda_workspace_pool,
    PatchMatchSchedulingSummary* scheduling_summary) {
  const ControllerClock::time_point batch_start = ControllerClock::now();
  std::vector<PatchMatchRefinementResult> outputs;
  outputs.reserve(jobs.size());
  const std::size_t num_threads =
      std::max<std::size_t>(1, cli_options.num_threads);
  const auto collect_oldest_future =
      [&outputs](std::vector<std::future<PatchMatchRefinementResult>>*
                     futures) {
        outputs.push_back(futures->front().get());
        futures->erase(futures->begin());
      };

  if (!cuda_workspace_pool.empty()) {
    const std::size_t pool_size = cuda_workspace_pool.size();
    std::vector<std::future<PatchMatchRefinementResult>> futures;
    futures.reserve(pool_size);
    std::size_t next_slot = 0;
    for (const PatchMatchRefinementJob& job : jobs) {
      if (futures.size() >= pool_size) {
        collect_oldest_future(&futures);
      }
      const std::size_t slot = next_slot;
      next_slot = (next_slot + 1) % pool_size;
      const PatchMatchRefinementJob* job_ptr = &job;
      futures.push_back(std::async(
          std::launch::async,
          [job_ptr, &initial_depth_maps, &refinement_options,
           &cuda_workspace_pool, pool_size, slot]() {
            return RunPatchMatchRefinementJob(
                *job_ptr, initial_depth_maps, refinement_options,
                cuda_workspace_pool[slot], pool_size, slot);
          }));
    }
    while (!futures.empty()) {
      collect_oldest_future(&futures);
    }
  } else if (num_threads == 1 || jobs.size() <= 1) {
    for (const PatchMatchRefinementJob& job : jobs) {
      outputs.push_back(RunPatchMatchRefinementJob(
          job, initial_depth_maps, refinement_options, nullptr, 0, 0));
    }
  } else {
    std::vector<std::future<PatchMatchRefinementResult>> futures;
    futures.reserve(num_threads);
    for (const PatchMatchRefinementJob& job : jobs) {
      const PatchMatchRefinementJob* job_ptr = &job;
      futures.push_back(std::async(
          std::launch::async,
          [job_ptr, &initial_depth_maps, &refinement_options]() {
            return RunPatchMatchRefinementJob(
                *job_ptr, initial_depth_maps, refinement_options, nullptr, 0,
                0);
          }));
      if (futures.size() >= num_threads) {
        collect_oldest_future(&futures);
      }
    }
    while (!futures.empty()) {
      collect_oldest_future(&futures);
    }
  }

  if (scheduling_summary != nullptr) {
    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                  batch_start)
            .count();
    double sum_cuda_time_ms = 0.0;
    for (const PatchMatchRefinementResult& output : outputs) {
      if (output.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda) {
        sum_cuda_time_ms += output.result.summary.cuda_total_time_ms;
      }
    }
    FillOverlapSummary(
        wall_time_ms, sum_cuda_time_ms,
        &scheduling_summary->refinement_wall_time_ms,
        &scheduling_summary->refinement_sum_cuda_time_ms,
        &scheduling_summary->refinement_overlap_saved_time_ms,
        &scheduling_summary->refinement_overlap_ratio);
  }
  return outputs;
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
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary,
    std::size_t* num_photometric_filtered_pixels,
    std::size_t* num_normal_refined_pixels,
    std::size_t* num_geometric_consistent_pixels,
    std::size_t* num_geometric_filtered_pixels,
    PatchMatchSchedulingSummary* scheduling_summary_out,
    mvs::StereoFusionSummary* fusion_summary_out) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  mvs::PatchMatchStereoOptions patch_match_options;
  patch_match_options.backend =
      mvs::PatchMatchBackendFromName(cli_options.patch_match_backend);
  patch_match_options.cuda_gpu_index = cli_options.patch_match_gpu_index;
  patch_match_options.initialization.random_seed = cli_options.random_seed;
  patch_match_options.perturbation.random_seed = cli_options.random_seed + 7919;
  patch_match_options.num_iterations = cli_options.patch_match_iterations;
  patch_match_options.pyramid_levels = cli_options.patch_match_pyramid_levels;
  patch_match_options.cost.window_radius = cli_options.patch_match_window_radius;
  patch_match_options.cost.bilateral_weighting =
      cli_options.patch_match_bilateral_ncc;
  patch_match_options.cost.sigma_spatial =
      cli_options.patch_match_sigma_spatial;
  patch_match_options.cost.sigma_color = cli_options.patch_match_sigma_color;
  patch_match_options.source_selection.dynamic_probability =
      cli_options.patch_match_dynamic_source_selection;
  patch_match_options.source_selection.directional_hmm =
      cli_options.patch_match_directional_hmm;
  patch_match_options.source_selection.num_samples =
      cli_options.patch_match_source_samples;
  patch_match_options.source_selection.ncc_sigma =
      cli_options.patch_match_ncc_sigma;
  patch_match_options.source_selection.previous_probability_weight =
      cli_options.patch_match_previous_selection_probability_weight;
  patch_match_options.source_selection.min_probability =
      cli_options.patch_match_min_selection_probability;
  patch_match_options.source_selection.max_probability =
      cli_options.patch_match_max_selection_probability;
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
  patch_match_options.checkerboard_propagation_rounds =
      cli_options.patch_match_checkerboard_rounds;
  patch_match_options.collect_diagnostics =
      cli_options.patch_match_diagnostics;
  const bool run_geometric_refinement =
      cli_options.patch_match_geometric_refinement_iterations > 0 &&
      cli_options.patch_match_geometric_consistency_weight > 0.0F;
  mvs::PatchMatchStereoOptions refinement_options = patch_match_options;
  refinement_options.num_iterations =
      cli_options.patch_match_geometric_refinement_iterations;
  refinement_options.cost.geometric_consistency_weight =
      cli_options.patch_match_geometric_consistency_weight;
  refinement_options.cost.geometric_max_reprojection_error = 3.0F;
  refinement_options.cost.geometric_max_relative_depth_error =
      cli_options.patch_match_geometric_max_relative_depth_error;
  // COLMAP treats geometry as a soft forward-backward reprojection cost. It
  // does not reject every candidate with an occluded or conflicting source.
  refinement_options.cost.geometric_reject_occluded_candidates = false;
  refinement_options.cost.geometric_reject_depth_conflicts = false;

  const std::filesystem::path depth_dir =
      output_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      output_path / "stereo" / "normal_maps";
  const std::filesystem::path cost_dir = output_path / "stereo" / "cost_maps";
  const std::filesystem::path consistency_dir =
      output_path / "stereo" / "consistency_graphs";
  const std::filesystem::path colmap_consistency_dir =
      output_path / "stereo" / "consistency_graphs_colmap";
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
  CudaWorkspacePool cuda_workspaces =
      CreateCudaWorkspacePool(patch_match_options, cli_options,
                              image_summaries.size());
  std::vector<mvs::CudaPatchMatchWorkspace*> cuda_workspace_ptrs;
  cuda_workspace_ptrs.reserve(cuda_workspaces.size());
  for (const mvs::CudaPatchMatchWorkspacePtr& workspace_ptr :
       cuda_workspaces) {
    cuda_workspace_ptrs.push_back(workspace_ptr.get());
  }
  if (scheduling_summary_out != nullptr &&
      patch_match_options.backend != mvs::PatchMatchBackend::kCpu &&
      !image_summaries.empty()) {
    std::string unavailable_reason;
    scheduling_summary_out->cuda_warmup_time_ms =
        mvs::WarmupCudaPatchMatch(patch_match_options.cuda_gpu_index,
                                  &unavailable_reason);
    if (!unavailable_reason.empty() &&
        patch_match_options.backend == mvs::PatchMatchBackend::kCuda) {
      throw std::runtime_error(unavailable_reason);
    }
  }

  std::vector<PatchMatchWorkspaceResult> results =
      RunInitialPatchMatchForWorkspace(
          image_summaries, images, views, patch_match_options, cli_options,
          binary_dense_maps, run_geometric_refinement, cuda_workspace_ptrs,
          scheduling_summary_out);
  WriteInitialPatchMatchDiagnostics(output_path, results);
  std::unordered_map<ImageId, std::size_t> result_index_by_image_id;
  for (std::size_t i = 0; i < results.size(); ++i) {
    total_photometric_filtered_pixels +=
        results[i].photometric_filtered_pixels;
    total_normal_refined_pixels +=
        results[i].result.summary.num_normal_refined_pixels;
    result_index_by_image_id[results[i].image_id] = i;
    if (scheduling_summary_out != nullptr &&
        results[i].photometric_stats_deferred) {
      ++scheduling_summary_out->initial_deferred_photometric_stats_images;
    }
  }
  num_patch_match_images = results.size();
  std::vector<ImageId> image_ids_by_mvs_index;
  image_ids_by_mvs_index.reserve(results.size());
  for (const PatchMatchWorkspaceResult& result : results) {
    image_ids_by_mvs_index.push_back(result.image_id);
  }

  if (run_geometric_refinement) {
    std::vector<mvs::DenseDepthMap> initial_depth_maps;
    initial_depth_maps.reserve(results.size());
    for (const PatchMatchWorkspaceResult& result : results) {
      initial_depth_maps.push_back(result.result.depth_map);
    }
    std::vector<PatchMatchRefinementJob> refinement_jobs;
    refinement_jobs.reserve(results.size());
    for (std::size_t result_idx = 0; result_idx < results.size();
         ++result_idx) {
      const PatchMatchWorkspaceResult& result = results[result_idx];
      PatchMatchRefinementJob job;
      job.result_index = result_idx;
      job.reference_view = result.view;
      job.reference_image = result.image;
      job.min_depth = result.min_depth;
      job.max_depth = result.max_depth;
      for (const ImageId source_id : result.source_image_ids) {
        const auto it = result_index_by_image_id.find(source_id);
        if (it == result_index_by_image_id.end()) {
          continue;
        }
        const PatchMatchWorkspaceResult& source_result = results[it->second];
        job.source_views.push_back(source_result.view);
        job.source_images.push_back(source_result.image);
        job.source_result_indices.push_back(it->second);
      }
      if (job.source_views.empty()) {
        continue;
      }
      refinement_jobs.push_back(std::move(job));
    }
    std::vector<PatchMatchRefinementResult> refinement_results =
        RunPatchMatchRefinementJobs(
            refinement_jobs, initial_depth_maps, refinement_options,
            cli_options, cuda_workspace_ptrs, scheduling_summary_out);
    for (PatchMatchRefinementResult& refinement_result :
         refinement_results) {
      results[refinement_result.result_index].result =
          std::move(refinement_result.result);
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

    // Every consistency graph must observe the same unfiltered geometric
    // refinement result. Filtering results in-place while iterating makes the
    // output depend on image order and causes invalid pixels to cascade into
    // later source-depth lookups.
    std::vector<mvs::DenseDepthMap> geometric_source_depth_maps;
    geometric_source_depth_maps.reserve(results.size());
    for (const PatchMatchWorkspaceResult& result : results) {
      geometric_source_depth_maps.push_back(result.result.depth_map);
    }

    std::vector<mvs::ConsistencyGraph> geometric_graphs;
    geometric_graphs.reserve(results.size());

    for (const PatchMatchWorkspaceResult& result : results) {
      std::vector<mvs::PatchMatchView> source_views;
      std::vector<mvs::DenseDepthMap> source_depth_maps;
      for (const ImageId source_id : result.source_image_ids) {
        const auto it = result_index_by_image_id.find(source_id);
        if (it == result_index_by_image_id.end()) {
          continue;
        }
        const PatchMatchWorkspaceResult& source_result = results[it->second];
        source_views.push_back(source_result.view);
        source_depth_maps.push_back(geometric_source_depth_maps[it->second]);
      }
      geometric_graphs.push_back(mvs::ComputeGeometricConsistencyGraph(
          result.view, source_views,
          geometric_source_depth_maps[result_index_by_image_id.at(
              result.image_id)],
          source_depth_maps, geometric_options));
    }

    for (std::size_t result_idx = 0; result_idx < results.size();
         ++result_idx) {
      PatchMatchWorkspaceResult& result = results[result_idx];
      mvs::ConsistencyGraph& graph = geometric_graphs[result_idx];
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
      mvs::WriteConsistencyGraphColmapBinary(
          colmap_consistency_dir / result.dense_file_name, graph,
          image_ids_by_mvs_index);
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
    mvs::WritePointsVisibilityColmapBinary(
        output_path / "dense" / "points_visibility.bin",
        fusion.points_visibility);
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
    mvs::MeshSimplificationOptions simplification_options;
    simplification_options.target_ratio =
        cli_options.mesh_simplification_ratio;
    simplification_options.target_num_triangles =
        cli_options.mesh_simplification_target_triangles;
    simplification_options.max_error =
        cli_options.mesh_simplification_max_error;
    simplification_options.preserve_boundary =
        cli_options.mesh_simplification_preserve_boundary;
    simplification_options.max_normal_change_degrees =
        cli_options.mesh_simplification_max_normal_change_degrees;
    mvs::MeshTexturingOptions texturing_options;
    texturing_options.atlas_size = cli_options.texture_atlas_size;
    texturing_options.max_atlas_size = cli_options.texture_max_atlas_size;
    texturing_options.atlas_patch_padding =
        cli_options.texture_patch_padding;
    texturing_options.inpaint_radius = cli_options.texture_inpaint_radius;
    texturing_options.texture_scale = cli_options.texture_scale;
    texturing_options.min_cos_normal_angle =
        cli_options.texture_min_cos_normal_angle;
    texturing_options.min_visible_vertices =
        cli_options.texture_min_visible_vertices;
    texturing_options.view_selection_smoothing_iterations =
        cli_options.texture_view_smoothing_iterations;
    texturing_options.visibility_resolution =
        cli_options.texture_visibility_resolution;
    texturing_options.occlusion_relative_tolerance =
        cli_options.texture_occlusion_relative_tolerance;
    texturing_options.apply_color_correction =
        cli_options.texture_apply_color_correction;
    texturing_options.color_correction_regularization =
        cli_options.texture_color_correction_regularization;
    texturing_options.num_threads = static_cast<int>(cli_options.num_threads);
    std::vector<mvs::MeshTextureView> texture_views;
    if (cli_options.mesh_texturer) {
      texture_views = LoadMeshTextureViews(reconstruction, fusion_views,
                                           cli_options.image_path);
    }
    const auto write_textured_mesh =
        [&](const mvs::DenseTriangleMesh& mesh, const std::string& name) {
          if (!cli_options.mesh_texturer) return;
          const mvs::TexturedTriangleMesh textured =
              mvs::TextureMesh(mesh, texture_views, texturing_options);
          mvs::WriteTexturedTriangleMesh(
              output_path / "dense" / ("textured-" + name), textured);
        };
    if (cli_options.poisson_mesher) {
      mvs::PoissonMeshingOptions options;
      options.point_weight = cli_options.poisson_point_weight;
      options.depth = cli_options.poisson_depth;
      options.color = cli_options.poisson_color;
      options.trim = cli_options.poisson_trim;
      options.num_threads = static_cast<int>(cli_options.num_threads);
      if (!mvs::PoissonMeshing(options, output_path / "dense" / "fused.ply",
                               output_path / "dense" /
                                   "meshed-poisson.ply")) {
        throw std::runtime_error("Poisson meshing failed.");
      }
      if (cli_options.mesh_simplifier || cli_options.mesh_texturer) {
        mvs::DenseTriangleMesh output_mesh = mvs::ReadDenseTriangleMeshPly(
            output_path / "dense" / "meshed-poisson.ply");
        if (cli_options.mesh_simplifier) {
          output_mesh =
              mvs::SimplifyMesh(output_mesh, simplification_options);
          mvs::WriteDenseTriangleMeshPly(
              output_path / "dense" / "meshed-poisson-simplified.ply",
              output_mesh);
          mvs::WriteDenseTriangleMeshStats(
              output_path / "dense" /
                  "meshed-poisson-simplified-stats.txt",
              mvs::ComputeDenseTriangleMeshStats(output_mesh));
        }
        write_textured_mesh(output_mesh, "poisson");
      }
    }
    if (cli_options.delaunay_mesher) {
      mvs::DelaunayMeshingOptions options;
      options.max_proj_dist = cli_options.delaunay_max_proj_dist;
      options.max_depth_dist = cli_options.delaunay_max_depth_dist;
      options.visibility_sigma = cli_options.delaunay_visibility_sigma;
      options.distance_sigma_factor =
          cli_options.delaunay_distance_sigma_factor;
      options.quality_regularization =
          cli_options.delaunay_quality_regularization;
      options.max_side_length_factor =
          cli_options.delaunay_max_side_length_factor;
      options.max_side_length_percentile =
          cli_options.delaunay_max_side_length_percentile;
      options.num_threads = static_cast<int>(cli_options.num_threads);
      const mvs::DenseTriangleMesh mesh = mvs::DelaunayMeshing(
          options, fusion.points, fusion.points_visibility, fusion_views);
      mvs::WriteDenseTriangleMeshPly(
          output_path / "dense" / "meshed-delaunay.ply", mesh);
      mvs::WriteDenseTriangleMeshStats(
          output_path / "dense" / "meshed-delaunay-stats.txt",
          mvs::ComputeDenseTriangleMeshStats(mesh));
      mvs::DenseTriangleMesh output_mesh = mesh;
      if (cli_options.mesh_simplifier) {
        output_mesh = mvs::SimplifyMesh(mesh, simplification_options);
        mvs::WriteDenseTriangleMeshPly(
            output_path / "dense" / "meshed-delaunay-simplified.ply",
            output_mesh);
        mvs::WriteDenseTriangleMeshStats(
            output_path / "dense" /
                "meshed-delaunay-simplified-stats.txt",
            mvs::ComputeDenseTriangleMeshStats(output_mesh));
      }
      write_textured_mesh(output_mesh, "delaunay");
    }
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
        mvs::DenseTriangleMesh output_mesh = mesh;
        if (cli_options.mesh_simplifier) {
          output_mesh = mvs::SimplifyMesh(mesh, simplification_options);
          mvs::WriteDenseTriangleMeshPly(
              output_path / "dense" / "mesh-simplified.ply", output_mesh);
          mvs::WriteDenseTriangleMeshStats(
              output_path / "dense" / "mesh-simplified-stats.txt",
              mvs::ComputeDenseTriangleMeshStats(output_mesh));
        }
        write_textured_mesh(output_mesh, "mesh");
      }
    }
  }

  std::ofstream diagnostics(output_path / "stereo" / "patch_match.tsv");
  diagnostics << "image_id\trequested_backend\tactual_backend"
              << "\tpyramid_levels\titerations"
              << "\tbilateral_ncc\tsigma_spatial\tsigma_color"
              << "\tdynamic_source_selection\tncc_sigma"
              << "\tprevious_selection_probability_weight"
              << "\tmin_selection_probability"
              << "\tmax_selection_probability"
              << "\tvalid_cost_pixels\tcandidate_evaluations"
              << "\tplane_propagation_candidates"
              << "\tperturbation_candidate_evaluations"
              << "\tpropagation_updates\tperturbation_updates"
              << "\tsource_selection_tests"
              << "\tsource_selection_acceptances"
              << "\tsource_selection_rejections"
              << "\tsource_selection_probability_updates"
              << "\tperturbation_mean_abs_depth_update"
              << "\tperturbation_max_abs_depth_update"
              << "\tperturbation_mean_normal_angle_degrees"
              << "\tperturbation_max_normal_angle_degrees"
              << "\tperturbation_geometric_candidate_rejections"
              << "\tperturbation_geometric_candidate_acceptances"
              << "\tforward_sweeps\tbackward_sweeps"
              << "\ttop_down_sweeps\tbottom_up_sweeps"
              << "\tcheckerboard_passes"
              << "\tcuda_upload_time_ms"
              << "\tcuda_initialization_time_ms"
              << "\tcuda_evaluate_cost_time_ms"
              << "\tcuda_propagation_time_ms"
              << "\tcuda_perturbation_time_ms"
              << "\tcuda_download_time_ms"
              << "\tcuda_total_time_ms"
              << "\tcuda_wrapper_total_time_ms"
              << "\tcuda_request_validation_time_ms"
              << "\tcuda_pyramid_resize_time_ms"
              << "\tcuda_host_pack_time_ms"
              << "\tcuda_single_scale_dispatch_time_ms"
              << "\tcuda_result_materialization_time_ms"
              << "\tcuda_output_to_vector_time_ms"
              << "\tcuda_dense_map_fill_time_ms"
              << "\tcuda_normal_refinement_time_ms"
              << "\tcuda_valid_cost_count_time_ms"
              << "\tcuda_host_initial_normal_upload_normalizations"
              << "\tcuda_host_output_normalizations"
              << "\tcuda_device_count_queries"
              << "\tcuda_device_count_cache_hits"
              << "\tcuda_shared_reference_tile_pixels"
              << "\tcuda_workspace_texture_allocations"
              << "\tcuda_workspace_texture_reuses"
              << "\tcuda_workspace_buffer_allocations"
              << "\tcuda_workspace_buffer_reuses"
              << "\tcuda_iteration_counter_snapshots"
              << "\tcuda_iteration_counter_readbacks"
              << "\tcuda_packed_output_pixels"
              << "\tcuda_output_map_readbacks"
              << "\tcuda_async_host_uploads"
              << "\tcuda_async_texture_uploads"
              << "\tcuda_async_counter_readbacks"
              << "\tcuda_async_output_readbacks"
              << "\tcuda_pinned_host_upload_bytes"
              << "\tcuda_pinned_texture_upload_bytes"
              << "\tcuda_pinned_host_staging_bytes"
              << "\tcuda_stream_kernel_launches"
              << "\tcuda_workspace_pool_size"
              << "\tcuda_workspace_pool_slot"
              << "\titeration_levels"
              << "\titeration_indices"
              << "\titeration_candidate_evaluations"
              << "\titeration_propagation_updates"
              << "\titeration_perturbation_updates"
              << "\titeration_source_selection_tests"
              << "\titeration_source_selection_acceptances"
              << "\titeration_source_selection_rejections"
              << "\titeration_source_selection_probability_means"
              << "\tfinal_source_selection_probability_means"
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
                << mvs::PatchMatchBackendToName(
                       result.result.summary.requested_backend)
                << "\t"
                << mvs::PatchMatchBackendToName(
                       result.result.summary.actual_backend)
                << "\t"
                << result.result.summary.num_pyramid_levels << "\t"
                << result.result.summary.num_iterations << "\t"
                << (cli_options.patch_match_bilateral_ncc ? 1 : 0) << "\t"
                << FormatStat(
                       cli_options.patch_match_sigma_spatial > 0.0F
                           ? cli_options.patch_match_sigma_spatial
                           : static_cast<float>(std::max(
                                 1, cli_options.patch_match_window_radius)))
                << "\t"
                << FormatStat(cli_options.patch_match_sigma_color) << "\t"
                << (cli_options.patch_match_dynamic_source_selection ? 1 : 0)
                << "\t"
                << FormatStat(cli_options.patch_match_ncc_sigma) << "\t"
                << FormatStat(
                       cli_options
                           .patch_match_previous_selection_probability_weight)
                << "\t"
                << FormatStat(
                       cli_options.patch_match_min_selection_probability)
                << "\t"
                << FormatStat(
                       cli_options.patch_match_max_selection_probability)
                << "\t"
                << result.result.summary.num_valid_cost_pixels << "\t"
                << result.result.summary.num_candidate_evaluations << "\t"
                << result.result.summary.num_plane_propagation_candidates
                << "\t"
                << result.result.summary
                       .num_perturbation_candidate_evaluations
                << "\t"
                << result.result.summary.num_propagation_updates << "\t"
                << result.result.summary.num_perturbation_updates << "\t"
                << result.result.summary.num_source_selection_tests << "\t"
                << result.result.summary.num_source_selection_acceptances
                << "\t"
                << result.result.summary.num_source_selection_rejections
                << "\t"
                << result.result.summary
                       .num_source_selection_probability_updates
                << "\t"
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
                << FormatStat(result.result.summary.cuda_upload_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_initialization_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_evaluate_cost_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_propagation_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_perturbation_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_download_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_total_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_wrapper_total_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_request_validation_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_pyramid_resize_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_host_pack_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .cuda_single_scale_dispatch_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .cuda_result_materialization_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_output_to_vector_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_dense_map_fill_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_normal_refinement_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_valid_cost_count_time_ms)
                << "\t"
                << result.result.summary
                       .cuda_host_initial_normal_upload_normalizations
                << "\t"
                << result.result.summary.cuda_host_output_normalizations
                << "\t"
                << result.result.summary.cuda_device_count_queries << "\t"
                << result.result.summary.cuda_device_count_cache_hits
                << "\t"
                << result.result.summary.cuda_shared_reference_tile_pixels
                << "\t"
                << result.result.summary.cuda_workspace_texture_allocations
                << "\t"
                << result.result.summary.cuda_workspace_texture_reuses << "\t"
                << result.result.summary.cuda_workspace_buffer_allocations
                << "\t"
                << result.result.summary.cuda_workspace_buffer_reuses
                << "\t"
                << result.result.summary.cuda_iteration_counter_snapshots
                << "\t"
                << result.result.summary.cuda_iteration_counter_readbacks
                << "\t"
                << result.result.summary.cuda_packed_output_pixels
                << "\t"
                << result.result.summary.cuda_output_map_readbacks
                << "\t"
                << result.result.summary.cuda_async_host_uploads
                << "\t"
                << result.result.summary.cuda_async_texture_uploads
                << "\t"
                << result.result.summary.cuda_async_counter_readbacks
                << "\t"
                << result.result.summary.cuda_async_output_readbacks
                << "\t"
                << result.result.summary.cuda_pinned_host_upload_bytes
                << "\t"
                << result.result.summary.cuda_pinned_texture_upload_bytes
                << "\t"
                << result.result.summary.cuda_pinned_host_staging_bytes
                << "\t"
                << result.result.summary.cuda_stream_kernel_launches
                << "\t"
                << result.result.summary.cuda_workspace_pool_size
                << "\t"
                << result.result.summary.cuda_workspace_pool_slot
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_pyramid_levels)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_indices)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_candidate_evaluations)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_propagation_updates)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_perturbation_updates)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_source_selection_tests)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_acceptances)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_rejections)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_probability_means)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .final_source_selection_probability_means)
                << "\t"
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

mvs::StereoFusionSummary FusePersistedStereoMaps(
    const std::filesystem::path& output_path,
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  const std::vector<const mvs::ImageMvsSummary*> image_summaries =
      CollectPatchMatchSummaries(summary);
  const std::vector<ImageId> required_image_ids =
      CollectRequiredMvsImageIds(image_summaries);
  const auto images = PreloadMvsImages(required_image_ids, &workspace);
  const auto views = BuildPatchMatchViews(reconstruction, images);
  const bool binary = UseBinaryDenseMaps(cli_options);

  std::vector<mvs::PatchMatchView> fusion_views;
  std::vector<mvs::MvsImage> fusion_images;
  std::vector<mvs::DenseDepthMap> depth_maps;
  std::vector<mvs::NormalMap> normal_maps;
  std::vector<mvs::DenseCostMap> cost_maps;
  std::vector<mvs::ConsistencyGraph> consistency_graphs;
  std::vector<ImageId> image_ids;
  for (const auto* image_summary : image_summaries) {
    const mvs::MvsImage& image = images.at(image_summary->image_id);
    const std::string dense_name =
        DenseMapFileName(image.name, false, binary);
    const auto depth_path = output_path / "stereo" / "depth_maps" / dense_name;
    const auto normal_path = output_path / "stereo" / "normal_maps" / dense_name;
    const auto cost_path = output_path / "stereo" / "cost_maps" /
                           (image.name + ".txt");
    fusion_views.push_back(views.at(image_summary->image_id));
    fusion_images.push_back(image);
    depth_maps.push_back(binary ? mvs::ReadDenseDepthMapBinary(depth_path)
                                : mvs::ReadDenseDepthMapText(depth_path));
    normal_maps.push_back(binary ? mvs::ReadNormalMapBinary(normal_path)
                                 : mvs::ReadNormalMapText(normal_path));
    cost_maps.push_back(mvs::ReadDenseCostMapText(cost_path));
    image_ids.push_back(image_summary->image_id);
  }
  for (std::size_t i = 0; i < fusion_images.size(); ++i) {
    const std::string dense_name =
        DenseMapFileName(fusion_images[i].name, false, binary);
    const auto native_path = output_path / "stereo" /
                             "consistency_graphs" / dense_name;
    const auto colmap_path = output_path / "stereo" /
                             "consistency_graphs_colmap" / dense_name;
    if (std::filesystem::exists(native_path) && binary) {
      consistency_graphs.push_back(
          mvs::ReadConsistencyGraphBinary(native_path));
    } else if (std::filesystem::exists(colmap_path)) {
      consistency_graphs.push_back(
          mvs::ReadConsistencyGraphColmapBinary(colmap_path, image_ids));
    } else {
      consistency_graphs.emplace_back(depth_maps[i].Width(),
                                      depth_maps[i].Height());
    }
  }

  mvs::StereoFusionOptions options;
  options.min_num_consistent_sources =
      cli_options.filter_geometric_min_num_consistent_sources > 0
          ? cli_options.filter_geometric_min_num_consistent_sources
          : 1;
  options.min_num_fused_observations = cli_options.fusion_min_num_observations;
  options.max_num_fused_observations = cli_options.fusion_max_num_observations;
  options.max_traversal_depth = cli_options.fusion_max_traversal_depth;
  options.max_reprojection_error_pixels =
      cli_options.fusion_max_reprojection_error_pixels;
  options.max_relative_depth_error = cli_options.fusion_max_relative_depth_error;
  options.max_normal_error_degrees = cli_options.fusion_max_normal_error_degrees;
  options.max_view_angle_degrees = cli_options.fusion_max_view_angle_degrees;
  options.source_pixel_search_radius =
      cli_options.fusion_source_pixel_search_radius;
  const mvs::StereoFusionResult fusion = mvs::FuseStereoDepthMaps(
      fusion_views, fusion_images, depth_maps, normal_maps, cost_maps,
      consistency_graphs, options);
  WriteFusedPointCloudPly(output_path / "dense" / "fused.ply", fusion.points);
  mvs::WritePointsVisibilityColmapBinary(
      output_path / "dense" / "points_visibility.bin",
      fusion.points_visibility);
  WriteFusionTracksDiagnostics(output_path / "dense" / "fusion_tracks.tsv",
                                fusion.tracks);
  WriteFusionImagePairsDiagnostics(
      output_path / "dense" / "fusion_image_pairs.tsv",
      fusion.pair_diagnostics);
  WriteFusionTrackLengthHistogram(
      output_path / "dense" / "fusion_track_lengths.tsv", fusion.tracks);
  WriteFusionErrorDistribution(
      output_path / "dense" / "fusion_error_distribution.tsv",
      fusion.error_samples);
  mvs::WriteFusionTrackQualityStats(
      output_path / "dense" / "fusion_summary.txt",
      mvs::ComputeFusionTrackQualityStats(fusion.tracks));
  return fusion.summary;
}

}  // namespace

int ReconMvsController::Run(std::ostream& out) const {
  ReconMvsControllerOptions cli_options = options_;
  Reconstruction reconstruction = ReadColmapTextModel(cli_options.model_path);
  const std::filesystem::path output_path(cli_options.output_path);
  std::size_t num_undistorted_images = 0;
  if (cli_options.image_undistorter) {
    num_undistorted_images =
        UndistortWorkspaceImages(cli_options, output_path, &reconstruction);
    cli_options.image_path = output_path / "undistorted" / "images";
    cli_options.max_image_size = -1;
    cli_options.resolution_level = 0;
  }

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
  PatchMatchSchedulingSummary scheduling_summary;
  mvs::StereoFusionSummary fusion_summary;
  if (cli_options.run_patch_match) {
    num_patch_match_images = RunPatchMatchForWorkspace(
        output_path, cli_options, reconstruction, summary,
        &num_photometric_filtered_pixels, &num_normal_refined_pixels,
        &num_geometric_consistent_pixels, &num_geometric_filtered_pixels,
        &scheduling_summary, &fusion_summary);
  } else if (cli_options.fuse_stereo) {
    fusion_summary = FusePersistedStereoMaps(
        output_path, cli_options, reconstruction, summary);
  }

  WriteMvsSummary(output_path, summary, num_undistorted_images,
                  num_initialized_patch_match_images, num_patch_match_images,
                  num_photometric_filtered_pixels, num_normal_refined_pixels,
                  num_geometric_consistent_pixels,
                  num_geometric_filtered_pixels, scheduling_summary,
                  fusion_summary);
  WriteSourceImages(output_path, summary);
  WriteSparseDepthSamples(output_path, summary);

  out << "Images: " << summary.num_images << "\n"
      << "Undistorted images: " << num_undistorted_images << "\n"
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


}  // namespace colite::controllers
