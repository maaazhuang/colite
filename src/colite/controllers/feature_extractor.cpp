#include "colite/controllers/feature_extractor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/core/camera.h"
#include "colite/core/parallel.h"
#include "colite/io/camera_prior.h"
#include "colite/io/sfm_workspace.h"

namespace colite::controllers {
namespace {

std::string ToConfigValue(double value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string ToConfigValue(std::filesystem::file_time_type value) {
  return std::to_string(value.time_since_epoch().count());
}

std::vector<std::pair<std::string, std::string>> FeatureCacheEntries(
    const FeatureExtractorControllerOptions& options,
    const std::vector<std::filesystem::path>& image_paths) {
  std::vector<std::pair<std::string, std::string>> entries;
  entries.emplace_back("stage", "features");
  entries.emplace_back("camera_prior_version", "camera_models_auto_exif_v4");
  entries.emplace_back("image_path", options.image_path.lexically_normal().string());
  entries.emplace_back("focal_length", ToConfigValue(options.focal_length));
  entries.emplace_back(
      "camera_mode",
      std::string(CameraSharingModeName(options.camera_sharing_mode)));
  entries.emplace_back(
      "camera_model",
      std::string(CameraModelIdToName(options.camera_model_id)));
  std::ostringstream camera_params;
  camera_params << std::setprecision(17);
  for (const double value : options.camera_params) camera_params << value << ',';
  entries.emplace_back("camera_params", camera_params.str());
  entries.emplace_back("default_focal_length_factor",
                       ToConfigValue(options.default_focal_length_factor));
  entries.emplace_back("max_num_features", std::to_string(options.max_num_features));
  entries.emplace_back("backend", SiftBackendName(options.backend));
  entries.emplace_back("resolved_backend",
                       SiftBackendName(ResolveSiftBackend(options.backend)));
  entries.emplace_back("num_images", std::to_string(image_paths.size()));
  for (std::size_t i = 0; i < image_paths.size(); ++i) {
    const std::filesystem::path& image_path = image_paths[i];
    entries.emplace_back("image_" + std::to_string(i) + "_path",
                         image_path.lexically_normal().string());
    entries.emplace_back("image_" + std::to_string(i) + "_size",
                         std::to_string(std::filesystem::file_size(image_path)));
    entries.emplace_back(
        "image_" + std::to_string(i) + "_mtime",
        ToConfigValue(std::filesystem::last_write_time(image_path)));
  }
  return entries;
}

std::vector<std::filesystem::path> FeatureRequiredFiles(
    std::size_t num_images) {
  std::vector<std::filesystem::path> files = {"cameras.txt",
                                             "images.tsv",
                                             "features/summary.txt"};
  for (std::size_t i = 0; i < num_images; ++i) {
    files.emplace_back("features/" + std::to_string(i + 1) + ".bin");
  }
  return files;
}

std::optional<FeatureExtractorControllerSummary> ReadCachedFeatureSummary(
    const std::filesystem::path& workspace_path,
    const std::string& config_hash) {
  if (!WorkspaceCacheIsValid(workspace_path, "features", config_hash)) {
    return std::nullopt;
  }
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, "features");
  FeatureExtractorControllerSummary summary;
  summary.success = true;
  summary.cache_hit = true;
  summary.num_images = static_cast<std::size_t>(
      std::stoull(metadata.values.at("images")));
  summary.num_cameras = static_cast<std::size_t>(
      std::stoull(metadata.values.at("cameras")));
  if (const auto it = metadata.values.find("features");
      it != metadata.values.end()) {
    summary.num_features =
        static_cast<std::size_t>(std::stoull(it->second));
  }
  summary.elapsed_seconds =
      std::stod(metadata.values.at("feature_extraction_seconds"));
  return summary;
}

class ImageJobQueue {
 public:
  explicit ImageJobQueue(std::size_t num_jobs) {
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

}  // namespace

FeatureExtractorController::FeatureExtractorController(
    FeatureExtractorControllerOptions options)
    : options_(std::move(options)) {}

FeatureExtractorControllerSummary FeatureExtractorController::Run() const {
  FeatureExtractorControllerSummary summary;
  const std::vector<std::filesystem::path> image_paths =
      ListImageFiles(options_.image_path);
  if (image_paths.empty()) {
    throw std::runtime_error("No input images found");
  }
  LOG(INFO) << "Feature extraction: images=" << image_paths.size()
            << " backend="
            << SiftBackendName(ResolveSiftBackend(options_.backend))
            << " max_features=" << options_.max_num_features
            << " requested_threads=" << options_.num_threads
            << " workspace=" << options_.workspace_path.string();
  const std::string config_hash =
      StableConfigHash(FeatureCacheEntries(options_, image_paths));
  if (const auto cached_summary =
          ReadCachedFeatureSummary(options_.workspace_path, config_hash)) {
    LOG(INFO) << "Feature extraction cache hit: images="
              << cached_summary->num_images
              << " features=" << cached_summary->num_features;
    return *cached_summary;
  }

  SfmCameraSetupOptions camera_options;
  camera_options.sharing_mode = options_.camera_sharing_mode;
  camera_options.model_id = options_.camera_model_id;
  camera_options.focal_length = options_.focal_length;
  camera_options.camera_params = options_.camera_params;
  camera_options.image_root = options_.image_path;
  camera_options.default_focal_length_factor =
      options_.default_focal_length_factor;
  const SfmCameraSetup camera_setup =
      BuildSfmCameraSetup(image_paths, camera_options);

  WriteWorkspaceCameras(options_.workspace_path, camera_setup.cameras);
  WriteWorkspaceImages(options_.workspace_path, camera_setup.images);

  SiftExtractionOptions sift_options;
  sift_options.backend = options_.backend;
  sift_options.max_num_features = options_.max_num_features;

  int num_threads = EffectiveFeatureExtractionThreads(options_.num_threads);
  const SiftBackend resolved_backend = ResolveSiftBackend(sift_options.backend);
#if defined(COLITE_ENABLE_SIFT_PARALLEL)
  // SIFT 内部固定线程池按连续图像区域切分；外层保持单图，避免两层并行争抢核心。
  parallel::setNumThreads(resolved_backend == SiftBackend::kCudaSift ||
                                  resolved_backend == SiftBackend::kCudaVlSift
                              ? 1
                              : num_threads);
  num_threads = 1;
#else
  if (resolved_backend == SiftBackend::kCudaSift ||
      resolved_backend == SiftBackend::kCudaVlSift) {
    num_threads = 1;
  }
#endif
  const std::size_t num_workers =
      std::min<std::size_t>(image_paths.size(),
                            static_cast<std::size_t>(std::max(1, num_threads)));

  const auto start = std::chrono::steady_clock::now();
  ImageJobQueue jobs(image_paths.size());
  std::exception_ptr first_exception;
  std::mutex exception_mutex;
  std::atomic<std::size_t> num_features{0};

  const auto worker = [&]() {
    while (true) {
      const std::optional<std::size_t> job_idx = jobs.Pop();
      if (!job_idx.has_value()) {
        return;
      }
      try {
        const FeatureSet features =
            ExtractSiftFeaturesFromFile(image_paths[*job_idx], sift_options);
        num_features.fetch_add(features.keypoints.size(),
                               std::memory_order_relaxed);
        VLOG(1) << "Extracted image " << (*job_idx + 1) << "/"
                << image_paths.size() << ": " << features.keypoints.size()
                << " features from "
                << image_paths[*job_idx].filename().string();
        WriteWorkspaceFeatureSet(options_.workspace_path,
                                 static_cast<ImageId>(*job_idx + 1),
                                 features);
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
  for (std::size_t i = 0; i < num_workers; ++i) {
    workers.emplace_back(worker);
  }
  for (std::thread& thread : workers) {
    thread.join();
  }
  if (first_exception != nullptr) {
    std::rethrow_exception(first_exception);
  }

  const auto end = std::chrono::steady_clock::now();
  summary.success = true;
  summary.num_images = image_paths.size();
  summary.num_cameras = camera_setup.cameras.size();
  summary.num_features = num_features.load(std::memory_order_relaxed);
  summary.elapsed_seconds = std::chrono::duration<double>(end - start).count();
  std::filesystem::create_directories(
      WorkspaceFeaturesPath(options_.workspace_path));
  std::ofstream summary_file(WorkspaceFeaturesPath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "images " << summary.num_images << "\n"
               << "cameras " << summary.num_cameras << "\n"
               << "camera_mode "
               << CameraSharingModeName(options_.camera_sharing_mode) << "\n"
               << "features " << summary.num_features << "\n"
               << "feature_extraction_seconds " << summary.elapsed_seconds
               << "\n";
  WriteWorkspaceCacheMetadata(
      options_.workspace_path, "features", config_hash,
      {{"images", std::to_string(summary.num_images)},
       {"cameras", std::to_string(summary.num_cameras)},
       {"features", std::to_string(summary.num_features)},
       {"feature_extraction_seconds", ToConfigValue(summary.elapsed_seconds)}},
      FeatureRequiredFiles(image_paths.size()));
  LOG(INFO) << "Feature extraction complete: images=" << summary.num_images
            << " cameras=" << summary.num_cameras
            << " features=" << summary.num_features
            << " elapsed_seconds=" << summary.elapsed_seconds;
  return summary;
}

}  // namespace colite::controllers
