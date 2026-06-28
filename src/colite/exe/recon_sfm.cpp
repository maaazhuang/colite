#include "colite/exe/recon_sfm.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <limits>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "colite/core/camera.h"
#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/core/version.h"
#include "colite/features/sift_extractor.h"
#include "colite/io/camera_prior.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/two_view_geometry_batch.h"
#include "colite/matching/weak_pair_rematcher.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite recon_sfm " << colite::Version() << "\n"
      << "\n"
      << "Usage:\n"
      << "  " << executable << " --help\n"
      << "  " << executable << " --version\n"
      << "  " << executable
      << " --image_path <dir> --output_path <dir> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --focal_length <px>       Initial focal length. If omitted, use "
         "JPEG EXIF 35mm focal prior when available, otherwise "
         "1.2 * max(width, height).\n"
      << "  --max_num_features <n>   SIFT feature limit, default 4096.\n"
      << "  --feature_num_threads <n>\n"
         "                            CPU feature extraction threads, default -1 "
         "uses hardware concurrency.\n"
      << "  --feature_extractor <vlsift|cvsift|cudasift>\n"
         "                            SIFT extraction backend, default cvsift.\n"
      << "  --matcher <bf|flann>     Descriptor matcher backend, default bf.\n"
      << "  --nearby_match_window <n>\n"
         "                            Re-match neighboring image pairs within "
         "this file-order window, default 2; 0 disables it.\n"
      << "  --nearby_match_ratio <r> Relaxed Lowe ratio for neighboring pairs, "
         "default 0.9.\n";
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

ReconSfmCliOptions ParseOptions(const std::vector<std::string>& args) {
  ReconSfmCliOptions options;
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
    } else if (arg == "--image_path") {
      options.image_path = RequireValue(args, &i, arg);
    } else if (arg == "--output_path") {
      options.output_path = RequireValue(args, &i, arg);
    } else if (arg == "--focal_length") {
      options.focal_length = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--max_num_features") {
      options.max_num_features = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--feature_num_threads") {
      options.feature_num_threads = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--feature_extractor") {
      options.feature_backend =
          SiftBackendFromName(RequireValue(args, &i, arg));
    } else if (arg == "--nearby_match_window") {
      options.nearby_match_window =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--nearby_match_ratio") {
      options.nearby_match_ratio = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--matcher") {
      const std::string matcher = RequireValue(args, &i, arg);
      if (matcher == "bf") {
        options.use_flann = false;
      } else if (matcher == "flann") {
        options.use_flann = true;
      } else {
        throw std::invalid_argument("Unsupported matcher: " + matcher);
      }
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  return options;
}

void ValidateOptions(const ReconSfmCliOptions& options) {
  if (options.show_help || options.show_version) {
    return;
  }
  if (options.image_path.empty()) {
    throw std::invalid_argument("Missing required argument --image_path");
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path");
  }
  if (options.max_num_features <= 0) {
    throw std::invalid_argument("--max_num_features must be positive");
  }
  if (options.focal_length < 0.0) {
    throw std::invalid_argument("--focal_length must be non-negative");
  }
  if (options.nearby_match_ratio <= 0.0 || options.nearby_match_ratio > 1.0) {
    throw std::invalid_argument("--nearby_match_ratio must be in (0, 1]");
  }
}

Camera MakeCameraFromImage(const std::filesystem::path& image_path,
                           double focal_length) {
  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = static_cast<std::uint32_t>(image.cols);
  camera.height = static_cast<std::uint32_t>(image.rows);
  const double default_focal =
      EstimateInitialFocalLengthFromImage(image_path, focal_length);
  camera.params = InitializeCameraParams(
      camera.model_id, default_focal, camera.width, camera.height);
  camera.has_prior_focal_length = focal_length > 0.0;
  return camera;
}

void EnsureSameImageSize(const std::vector<std::filesystem::path>& image_paths,
                         const Camera& camera) {
  for (const std::filesystem::path& image_path : image_paths) {
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + image_path.string());
    }
    if (image.cols != static_cast<int>(camera.width) ||
        image.rows != static_cast<int>(camera.height)) {
      throw std::runtime_error(
          "All images must have the same size in the first CLI version.");
    }
  }
}

Image MakeImage(ImageId image_id,
                CameraId camera_id,
                const std::filesystem::path& image_path) {
  Image image;
  image.id = image_id;
  image.camera_id = camera_id;
  image.name = image_path.filename().string();
  image.file_path = image_path.string();
  return image;
}

std::string Point3DIdForColmap(Point3DId point3d_id) {
  if (point3d_id == kInvalidPoint3DId) {
    return "-1";
  }
  return std::to_string(point3d_id);
}

std::string TwoViewGeometryTypeName(const TwoViewGeometry& geometry) {
  switch (geometry.type) {
    case TwoViewGeometry::Type::kUndefined:
      return "UNDEFINED";
    case TwoViewGeometry::Type::kDegenerate:
      return "DEGENERATE";
    case TwoViewGeometry::Type::kUncalibratedFundamental:
      return "UNCALIBRATED_FUNDAMENTAL";
    case TwoViewGeometry::Type::kCalibratedEssential:
      return "CALIBRATED_ESSENTIAL";
  }
  return "UNKNOWN";
}

}  // namespace

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph) {
  DescriptorMatchingOptions relaxed_nearby_options = base_options;
  relaxed_nearby_options.max_ratio = nearby_options.max_ratio;
  relaxed_nearby_options.mutual_check = true;

  for (std::size_t i = 0; i < image_ids.size(); ++i) {
    for (std::size_t j = i + 1; j < image_ids.size(); ++j) {
      const ImageId image_id1 = image_ids[i];
      const ImageId image_id2 = image_ids[j];
      const FeatureSet* features1 = FindFeatures(*graph, image_id1);
      const FeatureSet* features2 = FindFeatures(*graph, image_id2);
      if (features1 == nullptr || features2 == nullptr) {
        continue;
      }

      // 近邻图像通常承担增量建图主链路。这里对相邻/近邻 pair 放宽
      // ratio，但仍保持互检，让几何验证获得更多候选而不过度引入单向歧义匹配。
      const bool use_nearby_relaxation =
          nearby_options.image_index_window > 0 &&
          j - i <= nearby_options.image_index_window;
      const DescriptorMatchingOptions& pair_options =
          use_nearby_relaxation ? relaxed_nearby_options : base_options;
      SetMatches(graph, image_id1, image_id2,
                 MatchFeatureSets(*features1, *features2, pair_options));
    }
  }
}

double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length) {
  return colite::EstimateInitialFocalLengthFromImage(image_path,
                                                     user_focal_length);
}

ReconstructionQualityStats ComputeReconstructionQualityStats(
    const Reconstruction& reconstruction) {
  ReconstructionQualityStats stats;
  double total_error = 0.0;
  double max_error = 0.0;

  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    for (const TrackElement& element : point3d.track.elements) {
      if (!reconstruction.ExistsImage(element.image_id)) {
        continue;
      }
      const Image& image = reconstruction.GetImage(element.image_id);
      if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
          element.point2d_idx >= image.points2d.size()) {
        continue;
      }
      const Camera& camera = reconstruction.GetCamera(image.camera_id);
      const auto error =
          ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                            image.points2d[element.point2d_idx].xy);
      if (!error.has_value()) {
        continue;
      }
      total_error += *error;
      max_error = std::max(max_error, *error);
      ++stats.num_observations;
    }
  }

  if (stats.num_observations > 0) {
    stats.mean_reprojection_error =
        total_error / static_cast<double>(stats.num_observations);
    stats.max_reprojection_error = max_error;
  }
  return stats;
}

int EffectiveFeatureExtractionThreads(const int requested_threads) {
  if (requested_threads > 0) {
    return requested_threads;
  }
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  return hardware_threads == 0 ? 1 : static_cast<int>(hardware_threads);
}

void WriteRunSummary(const std::filesystem::path& output_path,
                     const Reconstruction& reconstruction,
                     const IncrementalMappingSummary& summary,
                     const double feature_extraction_seconds) {
  std::filesystem::create_directories(output_path);
  const ReconstructionQualityStats stats =
      ComputeReconstructionQualityStats(reconstruction);

  std::ofstream summary_file(output_path / "summary.txt");
  summary_file << std::setprecision(17)
               << "success " << (summary.success ? 1 : 0) << "\n"
               << "registered_images " << summary.num_registered_images << "\n"
               << "points3d " << summary.num_points3d << "\n"
               << "feature_extraction_seconds "
               << feature_extraction_seconds << "\n"
               << "incremental_registrations "
               << summary.registrations.size() << "\n"
               << "num_observations " << stats.num_observations << "\n"
               << "mean_reprojection_error "
               << stats.mean_reprojection_error << "\n"
               << "max_reprojection_error " << stats.max_reprojection_error
               << "\n";
}

void WriteColmapTextModel(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction) {
  std::filesystem::create_directories(output_path);

  std::ofstream cameras_file(output_path / "cameras.txt");
  cameras_file << "# Camera list with one line of data per camera:\n"
               << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
               << "# Number of cameras: " << reconstruction.Cameras().size()
               << "\n";
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    cameras_file << camera_id << " " << CameraModelIdToName(camera.model_id)
                 << " " << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      cameras_file << " " << std::setprecision(17) << param;
    }
    cameras_file << "\n";
  }

  std::ofstream images_file(output_path / "images.txt");
  images_file << "# Image list with two lines of data per image:\n"
              << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
              << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
              << "# Number of images: " << reconstruction.RegisteredImageIds().size()
              << "\n";
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.registered || !image.cam_from_world.has_value()) {
      continue;
    }
    const Pose3& pose = *image.cam_from_world;
    images_file << std::setprecision(17) << image_id << " "
                << pose.q_cw.w() << " " << pose.q_cw.x() << " "
                << pose.q_cw.y() << " " << pose.q_cw.z() << " "
                << pose.t_cw.x() << " " << pose.t_cw.y() << " "
                << pose.t_cw.z() << " " << image.camera_id << " "
                << image.name << "\n";
    for (std::size_t i = 0; i < image.points2d.size(); ++i) {
      const Point2D& point2d = image.points2d[i];
      if (i > 0) {
        images_file << " ";
      }
      images_file << std::setprecision(17) << point2d.xy.x() << " "
                  << point2d.xy.y() << " "
                  << Point3DIdForColmap(point2d.point3d_id);
    }
    images_file << "\n";
  }

  std::ofstream points_file(output_path / "points3D.txt");
  points_file << "# 3D point list with one line of data per point:\n"
              << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
                 "(IMAGE_ID, POINT2D_IDX)\n"
              << "# Number of points: " << reconstruction.Points3D().size()
              << "\n";
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    points_file << point3d_id << " " << std::setprecision(17)
                << point3d.xyz.x() << " " << point3d.xyz.y() << " "
                << point3d.xyz.z() << " 255 255 255 "
                << point3d.reprojection_error;
    for (const TrackElement& element : point3d.track.elements) {
      points_file << " " << element.image_id << " " << element.point2d_idx;
    }
    points_file << "\n";
  }
}

void WritePairDiagnostics(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction,
                          const MatchGraph& graph) {
  std::filesystem::create_directories(output_path);

  std::ofstream diagnostics_file(output_path / "pair_diagnostics.tsv");
  diagnostics_file
      << "image_id1\timage_id2\timage_name1\timage_name2\traw_matches"
         "\tverified_inliers\tgeometry_type\tmedian_triangulation_angle\n";

  for (const auto& [pair, matches] : graph.matches) {
    const auto image1_it = reconstruction.Images().find(pair.image_id1);
    const auto image2_it = reconstruction.Images().find(pair.image_id2);
    const std::string image_name1 =
        image1_it == reconstruction.Images().end() ? "" : image1_it->second.name;
    const std::string image_name2 =
        image2_it == reconstruction.Images().end() ? "" : image2_it->second.name;

    const TwoViewGeometry* geometry =
        FindVerifiedGeometry(graph, pair.image_id1, pair.image_id2);
    const std::size_t verified_inliers =
        geometry == nullptr ? 0U : geometry->inlier_matches.size();
    const std::string geometry_type =
        geometry == nullptr ? "UNVERIFIED" : TwoViewGeometryTypeName(*geometry);
    const double median_angle =
        geometry == nullptr ? -1.0 : geometry->median_triangulation_angle;

    diagnostics_file << pair.image_id1 << "\t" << pair.image_id2 << "\t"
                     << image_name1 << "\t" << image_name2 << "\t"
                     << matches.size() << "\t" << verified_inliers << "\t"
                     << geometry_type << "\t" << std::setprecision(17)
                     << median_angle << "\n";
  }
}

void WritePlyPointCloud(const std::filesystem::path& ply_path,
                        const Reconstruction& reconstruction) {
  std::filesystem::create_directories(ply_path.parent_path());

  std::ofstream ply_file(ply_path);
  ply_file << "ply\n"
           << "format ascii 1.0\n"
           << "element vertex " << reconstruction.Points3D().size() << "\n"
           << "property double x\n"
           << "property double y\n"
           << "property double z\n"
           << "property uchar red\n"
           << "property uchar green\n"
           << "property uchar blue\n"
           << "end_header\n";

  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    ply_file << std::setprecision(17) << point3d.xyz.x() << " "
             << point3d.xyz.y() << " " << point3d.xyz.z()
             << " 255 255 255\n";
  }
}

namespace {

class FeatureExtractionJobQueue {
 public:
  explicit FeatureExtractionJobQueue(const std::size_t num_jobs) {
    for (std::size_t idx = 0; idx < num_jobs; ++idx) {
      jobs_.push(idx);
    }
  }

  std::optional<std::size_t> Pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (jobs_.empty()) {
      return std::nullopt;
    }
    const std::size_t job_idx = jobs_.front();
    jobs_.pop();
    return job_idx;
  }

 private:
  std::queue<std::size_t> jobs_;
  std::mutex mutex_;
};

std::vector<FeatureSet> ExtractFeaturesForImages(
    const std::vector<std::filesystem::path>& image_paths,
    const SiftExtractionOptions& sift_options,
    const int requested_num_threads) {
  std::vector<FeatureSet> features(image_paths.size());
  if (image_paths.empty()) {
    return features;
  }

  int num_threads = EffectiveFeatureExtractionThreads(requested_num_threads);
  // 当前 CudaSift 后端包含全局 CUDA 状态，GPU 模式先保持串行更稳定。
  if (sift_options.backend == SiftBackend::kCudaSift) {
    num_threads = 1;
  }
  const std::size_t num_workers =
      std::min<std::size_t>(image_paths.size(),
                            static_cast<std::size_t>(std::max(1, num_threads)));

  FeatureExtractionJobQueue job_queue(image_paths.size());
  std::exception_ptr first_exception;
  std::mutex exception_mutex;

  const auto worker = [&]() {
    while (true) {
      {
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (first_exception != nullptr) {
          return;
        }
      }

      const std::optional<std::size_t> job_idx = job_queue.Pop();
      if (!job_idx.has_value()) {
        return;
      }

      try {
        features[*job_idx] =
            ExtractSiftFeaturesFromFile(image_paths[*job_idx], sift_options);
      } catch (...) {
        std::lock_guard<std::mutex> lock(exception_mutex);
        if (first_exception == nullptr) {
          first_exception = std::current_exception();
        }
        return;
      }
    }
  };

  std::vector<std::thread> workers;
  workers.reserve(num_workers);
  for (std::size_t worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
    workers.emplace_back(worker);
  }
  for (std::thread& thread : workers) {
    thread.join();
  }

  if (first_exception != nullptr) {
    std::rethrow_exception(first_exception);
  }
  return features;
}

int RunPipeline(const ReconSfmCliOptions& options,
                std::ostream& out) {
  const std::filesystem::path image_dir(options.image_path);
  const std::filesystem::path output_dir(options.output_path);
  const std::vector<std::filesystem::path> image_paths =
      ListImageFiles(image_dir);
  if (image_paths.size() < 2) {
    throw std::runtime_error("Need at least two input images.");
  }

  Camera camera = MakeCameraFromImage(image_paths.front(), options.focal_length);
  EnsureSameImageSize(image_paths, camera);

  SiftExtractionOptions sift_options;
  sift_options.max_num_features = options.max_num_features;
  sift_options.backend = options.feature_backend;

  MatchGraph graph;
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  std::vector<ImageId> image_ids;
  image_ids.reserve(image_paths.size());
  for (std::size_t i = 0; i < image_paths.size(); ++i) {
    const ImageId image_id = static_cast<ImageId>(i + 1);
    image_ids.push_back(image_id);
    reconstruction.AddImage(MakeImage(image_id, camera.id, image_paths[i]));
  }

  const auto feature_start = std::chrono::steady_clock::now();
  const std::vector<FeatureSet> extracted_features = ExtractFeaturesForImages(
      image_paths, sift_options, options.feature_num_threads);
  const auto feature_end = std::chrono::steady_clock::now();
  const double feature_extraction_seconds =
      std::chrono::duration<double>(feature_end - feature_start).count();
  for (std::size_t i = 0; i < extracted_features.size(); ++i) {
    SetFeatures(&graph, image_ids[i], extracted_features[i]);
  }

  DescriptorMatchingOptions matching_options;
  matching_options.backend = options.use_flann
                                 ? DescriptorMatcherBackend::kFlann
                                 : DescriptorMatcherBackend::kBruteForce;
  NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = options.nearby_match_window;
  nearby_options.max_ratio = options.nearby_match_ratio;
  MatchAllImagePairs(image_ids, matching_options, nearby_options, &graph);

  std::map<ImageId, Camera> cameras_by_image;
  for (const ImageId image_id : image_ids) {
    cameras_by_image.emplace(image_id, camera);
  }
  TwoViewGeometryBatchOptions geometry_options;
  geometry_options.keep_degenerate = true;
  VerifyMatchGraphGeometries(cameras_by_image, &graph, geometry_options);

  WeakPairRematchingOptions weak_pair_options;
  weak_pair_options.nearby_image_window = options.nearby_match_window;
  weak_pair_options.min_inlier_gain = 10;
  weak_pair_options.relaxed_matching = matching_options;
  weak_pair_options.relaxed_matching.max_ratio =
      std::max(options.nearby_match_ratio, 0.95);
  weak_pair_options.relaxed_matching.mutual_check = true;
  weak_pair_options.guided_matching.max_sampson_error_px =
      geometry_options.geometry_options.max_error_px;
  weak_pair_options.verification = geometry_options.geometry_options;
  RematchWeakImagePairs(cameras_by_image, weak_pair_options, &graph);

  IncrementalMappingOptions mapping_options;
  // 局部 re-triangulation 比首次建点更积极，因此单独增加一层角度分层：
  // 低 median angle 的两视图边先不参与局部重三角化，新建点也要通过 1 度角度门。
  mapping_options.mapper.triangulation.min_pair_median_triangulation_angle_deg =
      1.0;
  mapping_options.mapper.local_retriangulation_filter
      .min_triangulation_angle_deg = 1.0;
  IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, mapping_options, &reconstruction);
  std::filesystem::create_directories(output_dir);
  WriteRunSummary(output_dir, reconstruction, summary,
                  feature_extraction_seconds);
  WritePairDiagnostics(output_dir, reconstruction, graph);
  WriteColmapTextModel(output_dir, reconstruction);
  WritePlyPointCloud(output_dir / "points3D.ply", reconstruction);

  out << "Images: " << image_paths.size() << "\n"
      << "Registered images: " << summary.num_registered_images << "\n"
      << "Points3D: " << summary.num_points3d << "\n"
      << "Feature extraction seconds: " << std::setprecision(6)
      << feature_extraction_seconds << "\n"
      << "Mean reprojection error: "
      << std::setprecision(17)
      << ComputeReconstructionQualityStats(reconstruction)
             .mean_reprojection_error
      << "\n"
      << "Output: " << output_dir.string() << "\n";
  return summary.success ? 0 : 2;
}

}  // namespace

int RunReconSfm(const std::vector<std::string>& args,
                   std::ostream& out,
                   std::ostream& err) {
  const std::string executable = args.empty() ? "recon_sfm" : args.front();

  try {
    const ReconSfmCliOptions options = ParseOptions(args);
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

}  // namespace colite::exe
