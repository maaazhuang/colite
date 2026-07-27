#include "colite/controllers/recon_sfm_controller.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "colite/core/camera.h"
#include "colite/core/parallel.h"
#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/features/sift_extractor.h"
#include "colite/io/camera_prior.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/two_view_geometry_batch.h"
#include "colite/matching/weak_pair_rematcher.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

ReconSfmController::ReconSfmController(ReconSfmControllerOptions options)
    : options_(std::move(options)) {}

const ReconSfmControllerOptions& ReconSfmController::Options() const {
  return options_;
}

int ReconSfmController::EffectiveFeatureExtractionThreads() const {
  return controllers::EffectiveFeatureExtractionThreads(
      options_.feature_num_threads);
}

void ReconSfmController::MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    MatchGraph* graph) const {
  NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = options_.nearby_match_window;
  nearby_options.max_ratio = options_.nearby_match_ratio;
  controllers::MatchAllImagePairs(image_ids, base_options, nearby_options,
                                  graph);
}

namespace {
class ScopedParallelThreads {
 public:
  explicit ScopedParallelThreads(const int num_threads)
      : previous_num_threads_(parallel::getNumThreads()) {
    parallel::setNumThreads(num_threads);
  }

  ~ScopedParallelThreads() {
    parallel::setNumThreads(previous_num_threads_);
  }

  ScopedParallelThreads(const ScopedParallelThreads&) = delete;
  ScopedParallelThreads& operator=(const ScopedParallelThreads&) = delete;

 private:
  int previous_num_threads_;
};

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
  const InitialFocalLengthEstimate focal =
      colite::EstimateInitialFocalLengthPriorFromImage(image_path,
                                                       focal_length);
  camera.params = InitializeCameraParams(
      camera.model_id, focal.pixels, camera.width, camera.height);
  camera.has_prior_focal_length = focal.has_prior;
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

std::optional<Eigen::Vector3d> SampleRgbBilinear(const cv::Mat& image_bgr,
                                                  double x,
                                                  double y) {
  if (image_bgr.empty() || image_bgr.type() != CV_8UC3 ||
      !std::isfinite(x) || !std::isfinite(y)) {
    return std::nullopt;
  }

  // COLMAP defines the upper-left pixel center as (0.5, 0.5), whereas OpenCV
  // addresses the same pixel center as (0, 0).
  x -= 0.5;
  y -= 0.5;
  if (x < 0.0 || y < 0.0 || x > image_bgr.cols - 1.0 ||
      y > image_bgr.rows - 1.0) {
    return std::nullopt;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image_bgr.cols - 1);
  const int y1 = std::min(y0 + 1, image_bgr.rows - 1);
  const double dx = x - x0;
  const double dy = y - y0;
  const cv::Vec3b p00 = image_bgr.at<cv::Vec3b>(y0, x0);
  const cv::Vec3b p01 = image_bgr.at<cv::Vec3b>(y0, x1);
  const cv::Vec3b p10 = image_bgr.at<cv::Vec3b>(y1, x0);
  const cv::Vec3b p11 = image_bgr.at<cv::Vec3b>(y1, x1);

  Eigen::Vector3d rgb;
  for (int rgb_channel = 0; rgb_channel < 3; ++rgb_channel) {
    const int bgr_channel = 2 - rgb_channel;
    const double top = (1.0 - dx) * p00[bgr_channel] +
                       dx * p01[bgr_channel];
    const double bottom = (1.0 - dx) * p10[bgr_channel] +
                          dx * p11[bgr_channel];
    rgb[rgb_channel] = (1.0 - dy) * top + dy * bottom;
  }
  return rgb;
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
    case TwoViewGeometry::Type::kPlanarHomography:
      return "PLANAR_HOMOGRAPHY";
    case TwoViewGeometry::Type::kPanoramic:
      return "PANORAMIC";
    case TwoViewGeometry::Type::kMultiple:
      return "MULTIPLE";
  }
  return "UNKNOWN";
}

}  // namespace

void ExtractPointColors(Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    throw std::invalid_argument("reconstruction must not be null");
  }

  struct ColorSample {
    Point3DId point3d_id = kInvalidPoint3DId;
    Eigen::Vector3d rgb = Eigen::Vector3d::Zero();
  };
  const std::vector<ImageId> image_ids(
      reconstruction->RegisteredImageIds().begin(),
      reconstruction->RegisteredImageIds().end());
  std::vector<std::vector<ColorSample>> samples_by_image(image_ids.size());
  const Reconstruction& const_reconstruction = *reconstruction;
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(image_ids.size())),
      [&](const parallel::Range& range) {
        for (int image_idx = range.start; image_idx < range.end; ++image_idx) {
          const Image& image =
              const_reconstruction.GetImage(image_ids[image_idx]);
          if (image.file_path.empty()) {
            continue;
          }
          const cv::Mat pixels =
              cv::imread(image.file_path, cv::IMREAD_COLOR);
          if (pixels.empty()) {
            continue;
          }
          std::vector<ColorSample>& samples = samples_by_image[image_idx];
          samples.reserve(image.points2d.size());
          for (const Point2D& point2d : image.points2d) {
            if (!point2d.HasPoint3D() ||
                !const_reconstruction.ExistsPoint3D(point2d.point3d_id)) {
              continue;
            }
            const std::optional<Eigen::Vector3d> rgb =
                SampleRgbBilinear(pixels, point2d.xy.x(), point2d.xy.y());
            if (rgb.has_value()) {
              samples.push_back({point2d.point3d_id, *rgb});
            }
          }
        }
      });

  std::unordered_map<Point3DId, Eigen::Vector3d> color_sums;
  std::unordered_map<Point3DId, std::size_t> color_counts;
  color_sums.reserve(reconstruction->Points3D().size());
  color_counts.reserve(reconstruction->Points3D().size());

  // 按原图像顺序归并，保持浮点累加和最终 RGB 与串行实现一致。
  for (const std::vector<ColorSample>& samples : samples_by_image) {
    for (const ColorSample& sample : samples) {
      const auto sum_it = color_sums.find(sample.point3d_id);
      if (sum_it == color_sums.end()) {
        color_sums.emplace(sample.point3d_id, sample.rgb);
        color_counts.emplace(sample.point3d_id, 1);
      } else {
        sum_it->second += sample.rgb;
        ++color_counts.at(sample.point3d_id);
      }
    }
  }

  for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
    (void)point3d;
    Point3D& mutable_point = reconstruction->GetPoint3D(point3d_id);
    const auto count_it = color_counts.find(point3d_id);
    if (count_it == color_counts.end() || count_it->second == 0) {
      mutable_point.color.setZero();
      continue;
    }
    Eigen::Vector3d color = color_sums.at(point3d_id) /
                            static_cast<double>(count_it->second);
    for (int channel = 0; channel < 3; ++channel) {
      mutable_point.color[channel] = static_cast<std::uint8_t>(
          std::lround(std::clamp(color[channel], 0.0, 255.0)));
    }
  }
}

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph) {
  DescriptorMatchingOptions relaxed_nearby_options = base_options;
  relaxed_nearby_options.max_ratio = nearby_options.max_ratio;
  relaxed_nearby_options.mutual_check = true;

  std::vector<std::pair<std::size_t, std::size_t>> pair_indices;
  for (std::size_t i = 0; i < image_ids.size(); ++i) {
    for (std::size_t j = i + 1; j < image_ids.size(); ++j) {
      pair_indices.emplace_back(i, j);
    }
  }

  std::vector<std::vector<FeatureMatch>> pair_matches(pair_indices.size());
  std::vector<PreparedFeatureMatcher> prepared_matchers;
  prepared_matchers.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    const FeatureSet* features = FindFeatures(*graph, image_id);
    if (features == nullptr) {
      throw std::runtime_error("Missing features for exhaustive matching.");
    }
    prepared_matchers.emplace_back(*features, base_options.backend);
  }
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(pair_indices.size())),
      [&](const parallel::Range& range) {
        for (int pair_idx = range.start; pair_idx < range.end; ++pair_idx) {
          const auto [i, j] = pair_indices[static_cast<std::size_t>(pair_idx)];
          const bool use_nearby_relaxation =
              nearby_options.image_index_window > 0 &&
              j - i <= nearby_options.image_index_window;
          const DescriptorMatchingOptions& pair_options =
              use_nearby_relaxation ? relaxed_nearby_options : base_options;
          pair_matches[static_cast<std::size_t>(pair_idx)] =
              MatchPreparedFeatureSets(prepared_matchers[i],
                                       prepared_matchers[j], pair_options);
        }
      });

  // MatchGraph uses ordered maps. Publish the read-only parallel results in a
  // deterministic serial pass instead of locking the map per image pair.
  for (std::size_t pair_idx = 0; pair_idx < pair_indices.size(); ++pair_idx) {
    const auto [i, j] = pair_indices[pair_idx];
    SetMatches(graph, image_ids[i], image_ids[j], pair_matches[pair_idx]);
  }
}

namespace {

template <typename T>
class PipelineQueue {
 public:
  void Push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }
      values_.push(std::move(value));
    }
    condition_.notify_one();
  }

  bool Pop(T* value) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !values_.empty(); });
    if (values_.empty()) {
      return false;
    }
    *value = std::move(values_.front());
    values_.pop();
    return true;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    condition_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::queue<T> values_;
  bool closed_ = false;
};

struct FrontendPipelineJob {
  std::size_t image_index1 = 0;
  std::size_t image_index2 = 0;
  std::vector<FeatureMatch> matches;
  TwoViewGeometry geometry;
  GuidedDescriptorMatchingStats guided_stats;
  std::size_t initial_num_matches = 0;
  bool geometry_usable = false;
  bool guided_attempted = false;
  bool guided_improved = false;
  double matching_seconds = 0.0;
  double geometry_seconds = 0.0;
  double guided_seconds = 0.0;
};

struct FrontendPipelineSummary {
  double wall_seconds = 0.0;
  double matching_work_seconds = 0.0;
  double geometry_work_seconds = 0.0;
  double guided_work_seconds = 0.0;
  std::size_t num_initial_matches = 0;
  WeakPairRematchingSummary guided;
};

bool PipelineGeometryUsable(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

FrontendPipelineSummary RunCudaFrontendPipeline(
    const std::vector<ImageId>& image_ids,
    const std::map<ImageId, Camera>& cameras_by_image,
    const DescriptorMatchingOptions& base_matching_options,
    const NearbyPairMatchingOptions& nearby_options,
    const TwoViewGeometryBatchOptions& geometry_options,
    const WeakPairRematchingOptions& guided_options,
    MatchGraph* graph) {
  const auto pipeline_start = std::chrono::steady_clock::now();
  DescriptorMatchingOptions nearby_matching_options = base_matching_options;
  nearby_matching_options.max_ratio = nearby_options.max_ratio;
  nearby_matching_options.mutual_check = true;

  std::vector<PreparedFeatureMatcher> prepared_features;
  prepared_features.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    const FeatureSet* features = FindFeatures(*graph, image_id);
    if (features == nullptr) {
      throw std::runtime_error("Missing features for frontend pipeline.");
    }
    prepared_features.emplace_back(*features, base_matching_options.backend);
  }

  std::vector<FrontendPipelineJob> jobs;
  for (std::size_t i = 0; i < image_ids.size(); ++i) {
    for (std::size_t j = i + 1; j < image_ids.size(); ++j) {
      FrontendPipelineJob job;
      job.image_index1 = i;
      job.image_index2 = j;
      jobs.push_back(std::move(job));
    }
  }

  PipelineQueue<std::size_t> geometry_queue;
  PipelineQueue<std::size_t> guided_queue;
  std::mutex exception_mutex;
  std::exception_ptr worker_exception;
  std::atomic<bool> failed = false;
  const int num_geometry_threads = std::max(1, parallel::getNumThreads());
  std::atomic<int> geometry_workers_left = num_geometry_threads;

  const auto record_exception = [&](std::exception_ptr exception) {
    {
      std::lock_guard<std::mutex> lock(exception_mutex);
      if (worker_exception == nullptr) {
        worker_exception = std::move(exception);
      }
    }
    failed.store(true);
  };

  std::vector<std::thread> geometry_workers;
  geometry_workers.reserve(num_geometry_threads);
  for (int thread_idx = 0; thread_idx < num_geometry_threads; ++thread_idx) {
    geometry_workers.emplace_back([&] {
      try {
        std::size_t job_idx = 0;
        while (geometry_queue.Pop(&job_idx)) {
          if (failed.load()) {
            continue;
          }
          FrontendPipelineJob& job = jobs[job_idx];
          const ImageId image_id1 = image_ids[job.image_index1];
          const ImageId image_id2 = image_ids[job.image_index2];
          const auto geometry_start = std::chrono::steady_clock::now();
          job.geometry = EstimateTwoViewGeometry(
              cameras_by_image.at(image_id1), *FindFeatures(*graph, image_id1),
              cameras_by_image.at(image_id2), *FindFeatures(*graph, image_id2),
              job.matches, geometry_options.geometry_options);
          job.geometry_seconds = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - geometry_start).count();
          job.geometry_usable = PipelineGeometryUsable(job.geometry);
          if (job.geometry_usable && job.geometry.F.norm() > 0.0) {
            guided_queue.Push(job_idx);
          }
        }
      } catch (...) {
        record_exception(std::current_exception());
      }
      if (geometry_workers_left.fetch_sub(1) == 1) {
        guided_queue.Close();
      }
    });
  }

  std::thread guided_worker([&] {
    try {
      std::size_t job_idx = 0;
      while (guided_queue.Pop(&job_idx)) {
        if (failed.load()) {
          continue;
        }
        FrontendPipelineJob& job = jobs[job_idx];
        const auto guided_start = std::chrono::steady_clock::now();
        std::vector<FeatureMatch> guided_matches =
            MatchPreparedFeatureSetsGuidedFundamental(
                prepared_features[job.image_index1],
                prepared_features[job.image_index2], job.geometry.F,
                guided_options.guided_matching.max_sampson_error_px,
                guided_options.relaxed_matching, &job.guided_stats);
        job.guided_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - guided_start).count();
        job.guided_attempted = true;

        const std::size_t old_num_inliers =
            job.geometry.inlier_matches.size();
        if (guided_matches.size() <
            old_num_inliers + guided_options.min_inlier_gain) {
          continue;
        }
        const double old_inlier_ratio =
            job.initial_num_matches == 0
                ? 0.0
                : static_cast<double>(old_num_inliers) /
                      job.initial_num_matches;
        const double new_inlier_ratio =
            guided_matches.empty() ? 0.0 : 1.0;
        if (new_inlier_ratio + 1e-12 < old_inlier_ratio) {
          continue;
        }
        job.matches = std::move(guided_matches);
        job.geometry.inlier_matches = job.matches;
        job.guided_improved = true;
      }
    } catch (...) {
      record_exception(std::current_exception());
    }
  });

  try {
    for (std::size_t job_idx = 0; job_idx < jobs.size(); ++job_idx) {
      if (failed.load()) {
        break;
      }
      FrontendPipelineJob& job = jobs[job_idx];
      const bool use_nearby_relaxation =
          nearby_options.image_index_window > 0 &&
          job.image_index2 - job.image_index1 <=
              nearby_options.image_index_window;
      const DescriptorMatchingOptions& matching_options =
          use_nearby_relaxation ? nearby_matching_options
                                : base_matching_options;
      const auto matching_start = std::chrono::steady_clock::now();
      job.matches = MatchPreparedFeatureSets(
          prepared_features[job.image_index1],
          prepared_features[job.image_index2], matching_options);
      job.matching_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - matching_start).count();
      job.initial_num_matches = job.matches.size();
      geometry_queue.Push(job_idx);
    }
  } catch (...) {
    record_exception(std::current_exception());
  }
  geometry_queue.Close();
  for (std::thread& worker : geometry_workers) {
    worker.join();
  }
  guided_worker.join();
  if (worker_exception != nullptr) {
    std::rethrow_exception(worker_exception);
  }

  FrontendPipelineSummary summary;
  graph->matches.clear();
  graph->verified_geometries.clear();
  for (const FrontendPipelineJob& job : jobs) {
    const ImageId image_id1 = image_ids[job.image_index1];
    const ImageId image_id2 = image_ids[job.image_index2];
    SetMatches(graph, image_id1, image_id2, job.matches);
    if (job.geometry_usable || geometry_options.keep_degenerate) {
      SetVerifiedGeometry(graph, image_id1, image_id2, job.geometry);
    }
    summary.matching_work_seconds += job.matching_seconds;
    summary.geometry_work_seconds += job.geometry_seconds;
    summary.guided_work_seconds += job.guided_seconds;
    summary.num_initial_matches += job.initial_num_matches;
    ++summary.guided.num_checked_pairs;
    if (job.guided_attempted) {
      ++summary.guided.num_rematched_pairs;
    }
    if (job.guided_improved) {
      ++summary.guided.num_improved_pairs;
    }
    summary.guided.candidate_generation_work_seconds +=
        job.guided_stats.candidate_generation_seconds;
    summary.guided.descriptor_search_work_seconds +=
        job.guided_stats.descriptor_search_seconds;
  }
  summary.wall_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - pipeline_start).count();
  return summary;
}

}  // namespace

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
                     const ReconSfmRunStats& run_stats) {
  std::filesystem::create_directories(output_path);
  const ReconstructionQualityStats stats =
      ComputeReconstructionQualityStats(reconstruction);

  std::ofstream summary_file(output_path / "summary.txt");
  summary_file << std::setprecision(17)
               << "success " << (summary.success ? 1 : 0) << "\n"
               << "registered_images " << summary.num_registered_images << "\n"
               << "points3d " << summary.num_points3d << "\n"
               << "feature_backend " << run_stats.feature_backend << "\n"
               << "cuda_sift_available "
               << (run_stats.cuda_sift_available ? 1 : 0) << "\n"
               << "cuda_matcher_available "
               << (run_stats.cuda_matcher_available ? 1 : 0) << "\n"
               << "initial_focal_length "
               << run_stats.initial_focal_length << "\n"
               << "initial_focal_source " << run_stats.initial_focal_source
               << "\n"
               << "has_prior_focal_length "
               << (run_stats.has_prior_focal_length ? 1 : 0) << "\n"
               << "cpu_threads " << run_stats.cpu_threads << "\n"
               << "matcher_backend " << run_stats.matcher_backend << "\n"
               << "num_extracted_features "
               << run_stats.num_extracted_features << "\n"
               << "num_raw_matches " << run_stats.num_raw_matches << "\n"
               << "num_verified_inliers " << run_stats.num_verified_inliers
               << "\n"
               << "feature_extraction_seconds "
               << run_stats.feature_extraction_seconds << "\n"
               << "frontend_pipeline_seconds "
               << run_stats.frontend_pipeline_seconds << "\n"
               << "matching_seconds " << run_stats.matching_seconds << "\n"
               << "geometry_verification_seconds "
               << run_stats.geometry_verification_seconds << "\n"
               << "weak_pair_rematching_seconds "
               << run_stats.weak_pair_rematching_seconds << "\n"
               << "weak_candidate_generation_work_seconds "
               << run_stats.weak_candidate_generation_work_seconds << "\n"
               << "weak_descriptor_search_work_seconds "
               << run_stats.weak_descriptor_search_work_seconds << "\n"
               << "weak_geometry_verification_work_seconds "
               << run_stats.weak_geometry_verification_work_seconds << "\n"
               << "weak_pairs_checked " << run_stats.num_weak_pairs_checked
               << "\n"
               << "weak_pairs_rematched "
               << run_stats.num_weak_pairs_rematched << "\n"
               << "weak_pairs_improved " << run_stats.num_weak_pairs_improved
               << "\n"
               << "incremental_mapping_seconds "
               << run_stats.incremental_mapping_seconds << "\n"
               << "color_extraction_seconds "
               << run_stats.color_extraction_seconds << "\n"
               << "output_write_seconds " << run_stats.output_write_seconds
               << "\n"
               << "total_seconds " << run_stats.total_seconds << "\n"
               << "incremental_registrations "
               << summary.registrations.size() << "\n"
               << "initialization_attempts "
               << summary.initialization.num_initialization_attempts << "\n"
               << "global_retriangulated_points "
               << summary.global_retriangulation.num_created_points << "\n"
               << "global_retriangulated_observations "
               << summary.global_retriangulation.num_created_observations
               << "\n"
               << "global_completed_observations "
               << summary.global_track_completion.num_completed_observations
               << "\n"
               << "global_merged_points "
               << summary.global_track_completion.num_merged_points << "\n"
               << "global_filter_removed_observations "
               << summary.global_observation_filter.num_removed_observations
               << "\n"
               << "global_filter_removed_points "
               << summary.global_observation_filter.num_removed_points << "\n"
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
    const double track_error =
        Point3DTrackReprojectionError(reconstruction, point3d).value_or(-1.0);
    points_file << point3d_id << " " << std::setprecision(17)
                << point3d.xyz.x() << " " << point3d.xyz.y() << " "
                << point3d.xyz.z() << " "
                << static_cast<int>(point3d.color.x()) << " "
                << static_cast<int>(point3d.color.y()) << " "
                << static_cast<int>(point3d.color.z()) << " "
                << track_error;
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

void WriteRegistrationDiagnostics(
    const std::filesystem::path& output_path,
    const IncrementalMappingSummary& summary) {
  std::filesystem::create_directories(output_path);

  std::ofstream diagnostics_file(output_path / "registration_diagnostics.tsv");
  diagnostics_file
      << "order\timage_id\tcandidate_visible_points"
         "\tcandidate_visible_observations\tcandidate_mean_track_length"
         "\tcandidate_registered_neighbors"
         "\tcandidate_verified_inliers_to_registered"
         "\tcandidate_score\tregistration_attempts_before_success"
         "\tregistration_max_reprojection_error_px"
         "\tregistration_correspondences\tregistration_ransac_inliers"
         "\tregistration_inliers"
         "\tcompleted_observations\ttrack_merged_points"
         "\ttrack_merged_observations"
         "\ttriangulated_points\ttriangulated_observations"
         "\tlocal_retriangulated_points\tlocal_retriangulated_observations"
         "\tlocal_retriangulation_merged_points"
         "\tlocal_retriangulation_merged_observations"
         "\tlocal_retriangulation_filtered_points"
         "\tlocal_ba_local_images\tlocal_ba_fixed_images"
         "\tlocal_ba_constant_images\tlocal_ba_variable_images"
         "\tlocal_ba_constant_cameras\tlocal_ba_variable_cameras"
         "\tlocal_ba_points\tlocal_ba_residuals"
         "\tlocal_ba_initial_mean_error\tlocal_ba_final_mean_error"
         "\tfilter_removed_observations\tfilter_removed_points\n";

  for (std::size_t idx = 0; idx < summary.registrations.size(); ++idx) {
    const IncrementalMapperSummary& registration = summary.registrations[idx];
    const std::size_t order = registration.registration_order == 0
                                  ? idx + 1
                                  : registration.registration_order;
    diagnostics_file
        << std::setprecision(17) << order << "\t" << registration.image_id
        << "\t" << registration.candidate_num_visible_points << "\t"
        << registration.candidate_num_visible_observations << "\t"
        << registration.candidate_mean_track_length << "\t"
        << registration.candidate_num_registered_neighbors << "\t"
        << registration.candidate_num_verified_inliers_to_registered_images
        << "\t" << registration.candidate_score << "\t"
        << registration.registration_attempts_before_success << "\t"
        << registration.registration_max_reprojection_error_px << "\t"
        << registration.registration.num_correspondences << "\t"
        << registration.registration.num_ransac_inliers << "\t"
        << registration.registration.num_inliers << "\t"
        << registration.track_completion.num_completed_observations << "\t"
        << registration.track_completion.num_merged_points << "\t"
        << registration.track_completion.num_merged_observations << "\t"
        << registration.triangulation.num_created_points << "\t"
        << registration.triangulation.num_created_observations << "\t"
        << registration.local_retriangulation.num_created_points << "\t"
        << registration.local_retriangulation.num_created_observations << "\t"
        << registration.local_retriangulation.num_merged_points << "\t"
        << registration.local_retriangulation.num_merged_observations << "\t"
        << registration.local_retriangulation.num_filtered_points << "\t"
        << registration.local_bundle_adjustment.num_local_images << "\t"
        << registration.local_bundle_adjustment.fixed_image_ids.size() << "\t"
        << registration.local_bundle_adjustment.num_constant_images << "\t"
        << registration.local_bundle_adjustment.num_variable_images << "\t"
        << registration.local_bundle_adjustment.num_constant_cameras << "\t"
        << registration.local_bundle_adjustment.num_variable_cameras << "\t"
        << registration.local_bundle_adjustment.num_variable_points << "\t"
        << registration.local_bundle_adjustment.num_residuals << "\t"
        << registration.local_bundle_adjustment.initial_mean_reprojection_error
        << "\t"
        << registration.local_bundle_adjustment.final_mean_reprojection_error
        << "\t" << registration.observation_filter.num_removed_observations
        << "\t" << registration.observation_filter.num_removed_points << "\n";
  }

  std::ofstream failures_file(output_path / "registration_failures.tsv");
  failures_file
      << "image_id\tattempt\tcandidate_visible_points"
         "\tcandidate_visible_observations\tcandidate_mean_track_length"
         "\tcandidate_registered_neighbors"
         "\tcandidate_verified_inliers_to_registered"
         "\tcandidate_score\tregistration_max_reprojection_error_px"
         "\tregistration_correspondences\tregistration_ransac_inliers"
         "\tregistration_inliers"
         "\tmessage\n";
  for (const ImageRegistrationFailureSummary& failure :
       summary.failed_registrations) {
    failures_file << std::setprecision(17) << failure.image_id << "\t"
                  << failure.attempt << "\t"
                  << failure.candidate_num_visible_points << "\t"
                  << failure.candidate_num_visible_observations << "\t"
                  << failure.candidate_mean_track_length << "\t"
                  << failure.candidate_num_registered_neighbors << "\t"
                  << failure.candidate_num_verified_inliers_to_registered_images
                  << "\t" << failure.candidate_score << "\t"
                  << failure.registration_max_reprojection_error_px << "\t"
                  << failure.registration_num_correspondences << "\t"
                  << failure.registration_num_ransac_inliers << "\t"
                  << failure.registration_num_inliers << "\t"
                  << failure.message << "\n";
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
             << point3d.xyz.y() << " " << point3d.xyz.z() << " "
             << static_cast<int>(point3d.color.x()) << " "
             << static_cast<int>(point3d.color.y()) << " "
             << static_cast<int>(point3d.color.z()) << "\n";
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
  const SiftBackend resolved_backend = ResolveSiftBackend(sift_options.backend);
#if defined(COLITE_ENABLE_SIFT_PARALLEL)
  // CPU 优化实现已在尺度层、行带和描述子阶段并行，外层单图调度避免嵌套并行。
  const ScopedParallelThreads feature_thread_scope(
      resolved_backend == SiftBackend::kCudaSift ||
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

}  // namespace

int ReconSfmController::Run(std::ostream& out) const {
  const ReconSfmControllerOptions& options = options_;
  const std::filesystem::path image_dir(options.image_path);
  const std::filesystem::path output_dir(options.output_path);
  const std::vector<std::filesystem::path> image_paths =
      ListImageFiles(image_dir);
  if (image_paths.size() < 2) {
    throw std::runtime_error("Need at least two input images.");
  }

  const int cpu_threads =
      controllers::EffectiveFeatureExtractionThreads(
          options.feature_num_threads);
  const ScopedParallelThreads run_thread_scope(cpu_threads);
  const auto total_start = std::chrono::steady_clock::now();

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


  ReconSfmRunStats run_stats;
  const SiftBackend resolved_feature_backend =
      ResolveSiftBackend(options.feature_backend);
  run_stats.feature_backend = SiftBackendName(resolved_feature_backend);
  run_stats.cuda_sift_available = IsCudaSiftAvailable();
  run_stats.cuda_matcher_available = IsCudaDescriptorMatcherAvailable();
  const InitialFocalLengthEstimate initial_focal =
      EstimateInitialFocalLengthPriorFromImage(image_paths.front(),
                                               options.focal_length);
  run_stats.initial_focal_length = initial_focal.pixels;
  run_stats.initial_focal_source =
      InitialFocalLengthSourceName(initial_focal.source);
  run_stats.has_prior_focal_length = initial_focal.has_prior;
  run_stats.cpu_threads = cpu_threads;
  run_stats.feature_extraction_seconds = feature_extraction_seconds;
  for (const FeatureSet& features : extracted_features) {
    run_stats.num_extracted_features += features.keypoints.size();
  }

  DescriptorMatchingOptions matching_options;
  matching_options.backend = options.matcher_backend;
  switch (options.matcher_backend) {
    case DescriptorMatcherBackend::kBruteForce:
      run_stats.matcher_backend = "bf";
      break;
    case DescriptorMatcherBackend::kFlann:
      run_stats.matcher_backend = "flann";
      break;
    case DescriptorMatcherBackend::kCudaBruteForce:
      run_stats.matcher_backend = "cuda";
      break;
  }
  NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = options.nearby_match_window;
  nearby_options.max_ratio = options.nearby_match_ratio;

  std::map<ImageId, Camera> cameras_by_image;
  for (const ImageId image_id : image_ids) {
    cameras_by_image.emplace(image_id, camera);
  }
  TwoViewGeometryBatchOptions geometry_options;
  geometry_options.keep_degenerate = true;

  WeakPairRematchingOptions weak_pair_options;
  // Guided descriptor re-search applies to every pair with a usable initial
  // geometry. The nearby window only controls first-pass ratio relaxation.
  weak_pair_options.nearby_image_window = 0;
  weak_pair_options.min_inlier_gain = 10;
  weak_pair_options.relaxed_matching = matching_options;
  weak_pair_options.relaxed_matching.max_ratio =
      std::max(options.nearby_match_ratio, 0.95);
  weak_pair_options.relaxed_matching.mutual_check = true;
  weak_pair_options.guided_matching.max_sampson_error_px =
      geometry_options.geometry_options.max_error_px;
  weak_pair_options.verification = geometry_options.geometry_options;
  WeakPairRematchingSummary weak_pair_summary;

  if (matching_options.backend ==
      DescriptorMatcherBackend::kCudaBruteForce) {
    const FrontendPipelineSummary pipeline_summary =
        RunCudaFrontendPipeline(image_ids, cameras_by_image, matching_options,
                                nearby_options, geometry_options,
                                weak_pair_options, &graph);
    run_stats.frontend_pipeline_seconds = pipeline_summary.wall_seconds;
    run_stats.matching_seconds = pipeline_summary.matching_work_seconds;
    run_stats.geometry_verification_seconds =
        pipeline_summary.geometry_work_seconds;
    run_stats.weak_pair_rematching_seconds =
        pipeline_summary.guided_work_seconds;
    run_stats.num_raw_matches = pipeline_summary.num_initial_matches;
    weak_pair_summary = pipeline_summary.guided;
  } else {
    const auto matching_start = std::chrono::steady_clock::now();
    controllers::MatchAllImagePairs(image_ids, matching_options,
                                    nearby_options, &graph);
    run_stats.matching_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - matching_start).count();
    for (const auto& [pair, matches] : graph.matches) {
      (void)pair;
      run_stats.num_raw_matches += matches.size();
    }

    const auto geometry_start = std::chrono::steady_clock::now();
    VerifyMatchGraphGeometries(cameras_by_image, &graph, geometry_options);
    run_stats.geometry_verification_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - geometry_start).count();

    const auto rematching_start = std::chrono::steady_clock::now();
    weak_pair_summary =
        RematchWeakImagePairs(cameras_by_image, weak_pair_options, &graph);
    run_stats.weak_pair_rematching_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - rematching_start).count();
    run_stats.frontend_pipeline_seconds =
        run_stats.matching_seconds +
        run_stats.geometry_verification_seconds +
        run_stats.weak_pair_rematching_seconds;
  }
  run_stats.num_weak_pairs_checked = weak_pair_summary.num_checked_pairs;
  run_stats.num_weak_pairs_rematched = weak_pair_summary.num_rematched_pairs;
  run_stats.num_weak_pairs_improved = weak_pair_summary.num_improved_pairs;
  run_stats.weak_candidate_generation_work_seconds =
      weak_pair_summary.candidate_generation_work_seconds;
  run_stats.weak_descriptor_search_work_seconds =
      weak_pair_summary.descriptor_search_work_seconds;
  run_stats.weak_geometry_verification_work_seconds =
      weak_pair_summary.geometry_verification_work_seconds;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    (void)pair;
    run_stats.num_verified_inliers += geometry.inlier_matches.size();
  }

  IncrementalMappingOptions mapping_options;
  // 局部 re-triangulation 比首次建点更积极，因此单独增加一层角度分层：
  // 低 median angle 的两视图边先不参与局部重三角化，新建点也要通过 1 度角度门。
  mapping_options.mapper.triangulation.min_pair_median_triangulation_angle_deg =
      1.0;
  mapping_options.mapper.local_retriangulation_filter
      .min_triangulation_angle_deg = 1.0;
  mapping_options.mapper.local_bundle_adjustment.refine_camera_params =
      options.ba_refine_camera_params;
  mapping_options.mapper.local_bundle_adjustment.refine_camera_param_names =
      options.ba_refine_camera_param_names;
  mapping_options.global_bundle_adjustment.refine_camera_params =
      options.ba_refine_camera_params;
  mapping_options.global_bundle_adjustment.refine_camera_param_names =
      options.ba_refine_camera_param_names;
  mapping_options.max_num_registration_trials_per_image =
      options.max_registration_trials_per_image;
  mapping_options.initialization.max_num_initialization_trials =
      options.max_initialization_trials;
  mapping_options.registration_retry_reprojection_error_multiplier =
      options.registration_retry_reprojection_error_multiplier;
  mapping_options.registration_retry_max_reprojection_error_px =
      options.registration_retry_max_reprojection_error_px;
  const auto mapping_start = std::chrono::steady_clock::now();
  IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, mapping_options, &reconstruction);
  run_stats.incremental_mapping_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - mapping_start).count();
  const auto color_start = std::chrono::steady_clock::now();
  ExtractPointColors(&reconstruction);
  run_stats.color_extraction_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - color_start).count();
  std::filesystem::create_directories(output_dir);
  const auto output_start = std::chrono::steady_clock::now();
  WritePairDiagnostics(output_dir, reconstruction, graph);
  WriteRegistrationDiagnostics(output_dir, summary);
  WriteColmapTextModel(output_dir, reconstruction);
  WritePlyPointCloud(output_dir / "points3D.ply", reconstruction);
  run_stats.output_write_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - output_start).count();
  run_stats.total_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - total_start).count();
  WriteRunSummary(output_dir, reconstruction, summary, run_stats);

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

}  // namespace colite::controllers
