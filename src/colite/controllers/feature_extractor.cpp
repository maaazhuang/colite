#include "colite/controllers/feature_extractor.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/core/camera.h"
#include "colite/io/camera_prior.h"
#include "colite/io/sfm_workspace.h"

namespace colite::controllers {
namespace {

Camera MakeSingleCamera(const std::filesystem::path& image_path,
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
  camera.params = InitializeCameraParams(camera.model_id, default_focal,
                                         camera.width, camera.height);
  camera.has_prior_focal_length = focal_length > 0.0;
  return camera;
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

  const Camera camera = MakeSingleCamera(image_paths.front(),
                                         options_.focal_length);
  std::vector<Image> images;
  images.reserve(image_paths.size());
  for (std::size_t i = 0; i < image_paths.size(); ++i) {
    Image image;
    image.id = static_cast<ImageId>(i + 1);
    image.camera_id = camera.id;
    image.name = image_paths[i].filename().string();
    image.file_path = image_paths[i].string();
    images.push_back(image);
  }

  WriteWorkspaceCameras(options_.workspace_path, {camera});
  WriteWorkspaceImages(options_.workspace_path, images);

  SiftExtractionOptions sift_options;
  sift_options.backend = options_.backend;
  sift_options.max_num_features = options_.max_num_features;

  int num_threads = EffectiveFeatureExtractionThreads(options_.num_threads);
  if (sift_options.backend == SiftBackend::kCudaSift) {
    num_threads = 1;
  }
  const std::size_t num_workers =
      std::min<std::size_t>(image_paths.size(),
                            static_cast<std::size_t>(std::max(1, num_threads)));

  const auto start = std::chrono::steady_clock::now();
  ImageJobQueue jobs(image_paths.size());
  std::exception_ptr first_exception;
  std::mutex exception_mutex;

  const auto worker = [&]() {
    while (true) {
      const std::optional<std::size_t> job_idx = jobs.Pop();
      if (!job_idx.has_value()) {
        return;
      }
      try {
        const FeatureSet features =
            ExtractSiftFeaturesFromFile(image_paths[*job_idx], sift_options);
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
  summary.elapsed_seconds = std::chrono::duration<double>(end - start).count();
  std::filesystem::create_directories(
      WorkspaceFeaturesPath(options_.workspace_path));
  std::ofstream summary_file(WorkspaceFeaturesPath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "images " << summary.num_images << "\n"
               << "feature_extraction_seconds " << summary.elapsed_seconds
               << "\n";
  return summary;
}

}  // namespace colite::controllers
