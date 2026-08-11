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
#include <numeric>
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
#include <glog/logging.h>

#include "colite/core/camera.h"
#include "colite/core/parallel.h"
#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/features/aliked_extractor.h"
#include "colite/features/sift_extractor.h"
#include "colite/io/camera_prior.h"
#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/sfm/global_mapper.h"
#include "colite/sfm/similarity_transform.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/learned_matcher.h"
#include "colite/matching/match_graph_consistency.h"
#include "colite/matching/two_view_geometry_batch.h"
#include "colite/matching/weak_pair_rematcher.h"
#include "colite/sfm/incremental_mapper.h"
#include "colite/sfm/reconstruction_manager.h"

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
               << "mapper_type " << run_stats.mapper_type << "\n"
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
               << "camera_mode " << run_stats.camera_sharing_mode << "\n"
               << "num_cameras " << run_stats.num_cameras << "\n"
               << "num_prior_focal_images "
               << run_stats.num_prior_focal_images << "\n"
               << "num_position_prior_images "
               << run_stats.num_position_prior_images << "\n"
               << "position_prior_aligned_models "
               << run_stats.num_position_prior_aligned_models << "\n"
               << "position_prior_initial_mean_error "
               << run_stats.position_prior_initial_mean_error << "\n"
               << "position_prior_final_mean_error "
               << run_stats.position_prior_final_mean_error << "\n"
               << "min_initial_focal_length "
               << run_stats.min_initial_focal_length << "\n"
               << "max_initial_focal_length "
               << run_stats.max_initial_focal_length << "\n"
               << "cpu_threads " << run_stats.cpu_threads << "\n"
               << "matcher_backend " << run_stats.matcher_backend << "\n"
               << "num_extracted_features "
               << run_stats.num_extracted_features << "\n"
               << "num_raw_matches " << run_stats.num_raw_matches << "\n"
               << "num_verified_inliers " << run_stats.num_verified_inliers
               << "\n"
               << "num_verified_inliers_before_consistency "
               << run_stats.num_verified_inliers_before_consistency << "\n"
               << "pair_match_conflicts_removed "
               << run_stats.num_pair_match_conflicts_removed << "\n"
               << "component_match_conflicts_removed "
               << run_stats.num_component_match_conflicts_removed << "\n"
               << "conflicted_match_components "
               << run_stats.num_conflicted_match_components << "\n"
               << "conflicting_match_observations "
               << run_stats.num_conflicting_match_observations << "\n"
               << "sparse_models " << run_stats.num_sparse_models << "\n"
               << "unregistered_images "
               << run_stats.num_unregistered_images << "\n"
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
               << "position_prior_refinement_seconds "
               << run_stats.position_prior_refinement_seconds << "\n"
               << "mapper_initialization_seconds "
               << summary.initialization_seconds << "\n"
               << "mapper_candidate_selection_seconds "
               << summary.candidate_selection_seconds << "\n";
  double registration_seconds = 0.0;
  double track_completion_seconds = 0.0;
  double triangulation_seconds = 0.0;
  double local_retriangulation_seconds = 0.0;
  double local_ba_seconds = 0.0;
  double local_filter_seconds = 0.0;
  for (const IncrementalMapperSummary& registration : summary.registrations) {
    registration_seconds += registration.registration_seconds;
    track_completion_seconds += registration.track_completion_seconds;
    triangulation_seconds += registration.triangulation_seconds;
    local_retriangulation_seconds +=
        registration.local_retriangulation_seconds;
    local_ba_seconds += registration.local_bundle_adjustment_seconds;
    local_filter_seconds += registration.observation_filter_seconds;
  }
  summary_file << "mapper_registration_seconds " << registration_seconds
               << "\n"
               << "mapper_track_completion_seconds "
               << track_completion_seconds << "\n"
               << "mapper_triangulation_seconds " << triangulation_seconds
               << "\n"
               << "mapper_local_retriangulation_seconds "
               << local_retriangulation_seconds << "\n"
               << "mapper_local_ba_seconds " << local_ba_seconds << "\n"
               << "mapper_local_filter_seconds " << local_filter_seconds
               << "\n"
               << "mapper_global_track_completion_seconds "
               << summary.global_track_completion_seconds << "\n"
               << "mapper_pose_guided_track_completion_seconds "
               << summary.pose_guided_track_completion_seconds << "\n"
               << "mapper_global_retriangulation_seconds "
               << summary.global_retriangulation_seconds << "\n"
               << "mapper_global_ba_seconds "
               << summary.global_bundle_adjustment_seconds << "\n"
               << "mapper_global_filter_seconds "
               << summary.global_observation_filter_seconds << "\n"
               << "mapper_final_global_ba_seconds "
               << summary.final_global_bundle_adjustment_seconds << "\n"
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
               << "pose_guided_completed_observations "
               << summary.pose_guided_track_completion
                      .num_completed_observations
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
         "\tfilter_removed_observations\tfilter_removed_points"
         "\tused_structureless\tstructureless_hypotheses"
         "\tstructureless_consistent_hypotheses\n";

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
        << "\t" << registration.observation_filter.num_removed_points << "\t"
        << (registration.used_structureless_registration ? 1 : 0) << "\t"
        << registration.structureless_registration.num_pose_hypotheses << "\t"
        << registration.structureless_registration.num_consistent_hypotheses
        << "\n";
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

void MatchAllImagePairsTyped(
    const std::vector<ImageId>& image_ids,
    const std::map<ImageId, Camera>& cameras_by_image,
    const ReconSfmControllerOptions& options,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph) {
  if (options.matcher_type == FeatureMatcherType::kSiftBruteForce) {
    MatchAllImagePairs(image_ids, base_options, nearby_options, graph);
    return;
  }

  std::vector<std::pair<std::size_t, std::size_t>> pair_indices;
  for (std::size_t i = 0; i < image_ids.size(); ++i) {
    for (std::size_t j = i + 1; j < image_ids.size(); ++j) {
      pair_indices.emplace_back(i, j);
    }
  }
  std::vector<std::vector<FeatureMatch>> pair_matches(pair_indices.size());
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(pair_indices.size())),
      [&](const parallel::Range& range) {
        for (int pair_idx = range.start; pair_idx < range.end; ++pair_idx) {
          const auto [i, j] = pair_indices[static_cast<std::size_t>(pair_idx)];
          const ImageId image_id1 = image_ids[i];
          const ImageId image_id2 = image_ids[j];
          const FeatureSet* features1 = FindFeatures(*graph, image_id1);
          const FeatureSet* features2 = FindFeatures(*graph, image_id2);
          if (features1 == nullptr || features2 == nullptr) {
            throw std::runtime_error("Missing features for exhaustive matching");
          }
          if (options.matcher_type == FeatureMatcherType::kAlikedBruteForce) {
            CosineMatchingOptions learned_options;
            learned_options.min_cosine_similarity =
                static_cast<float>(options.min_cosine_similarity);
            learned_options.max_ratio = static_cast<float>(
                nearby_options.image_index_window > 0 &&
                        j - i <= nearby_options.image_index_window
                    ? nearby_options.max_ratio
                    : base_options.max_ratio);
            learned_options.mutual_check = base_options.mutual_check;
            pair_matches[static_cast<std::size_t>(pair_idx)] =
                MatchAlikedDescriptors(*features1, *features2, learned_options);
          } else {
            LightGlueMatchingOptions learned_options;
            learned_options.matcher_type = options.matcher_type;
            learned_options.model_path = options.learned_matcher_model_path;
            learned_options.min_score =
                static_cast<float>(options.lightglue_min_score);
            learned_options.num_threads = options.learned_matcher_num_threads;
            const Camera& camera1 = cameras_by_image.at(image_id1);
            const Camera& camera2 = cameras_by_image.at(image_id2);
            pair_matches[static_cast<std::size_t>(pair_idx)] =
                MatchFeatureSetsLightGlue(
                    *features1, static_cast<int>(camera1.width),
                    static_cast<int>(camera1.height), *features2,
                    static_cast<int>(camera2.width),
                    static_cast<int>(camera2.height), learned_options);
          }
        }
      });
  for (std::size_t pair_idx = 0; pair_idx < pair_indices.size(); ++pair_idx) {
    const auto [i, j] = pair_indices[pair_idx];
    SetMatches(graph, image_ids[i], image_ids[j], pair_matches[pair_idx]);
  }
}

std::vector<FeatureSet> ExtractFeaturesForImages(
    const std::vector<std::filesystem::path>& image_paths,
    const ReconSfmControllerOptions& options) {
  std::vector<FeatureSet> features(image_paths.size());
  if (image_paths.empty()) {
    return features;
  }

  int num_threads = EffectiveFeatureExtractionThreads(options.feature_num_threads);
  const int inference_num_threads = num_threads;
  const SiftBackend resolved_backend = ResolveSiftBackend(options.feature_backend);
#if defined(COLITE_ENABLE_SIFT_PARALLEL)
  // CPU 优化实现已在尺度层、行带和描述子阶段并行，外层单图调度避免嵌套并行。
  const ScopedParallelThreads feature_thread_scope(
      options.feature_extractor_type != FeatureExtractorType::kSift ||
              resolved_backend == SiftBackend::kCudaSift ||
              resolved_backend == SiftBackend::kCudaVlSift
          ? 1
          : num_threads);
  num_threads = 1;
#else
  if (options.feature_extractor_type != FeatureExtractorType::kSift ||
      resolved_backend == SiftBackend::kCudaSift ||
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
        if (options.feature_extractor_type == FeatureExtractorType::kSift) {
          SiftExtractionOptions sift_options;
          sift_options.max_num_features = options.max_num_features;
          sift_options.backend = options.feature_backend;
          features[*job_idx] = ExtractSiftFeaturesFromFile(
              image_paths[*job_idx], sift_options);
        } else {
          AlikedExtractionOptions aliked_options;
          aliked_options.extractor_type = options.feature_extractor_type;
          aliked_options.model_path = options.aliked_model_path;
          aliked_options.max_num_features = options.max_num_features;
          aliked_options.min_score = static_cast<float>(options.aliked_min_score);
          aliked_options.num_threads = inference_num_threads;
          features[*job_idx] = ExtractAlikedFeaturesFromFile(
              image_paths[*job_idx], aliked_options);
        }
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

std::string ReconFeatureConfigHash(
    const ReconSfmControllerOptions& options,
    const std::vector<std::filesystem::path>& image_paths) {
  std::vector<std::pair<std::string, std::string>> entries = {
      {"stage", "features"},
      {"algorithm_version", "recon_sfm_features_v4_learned_frontend"},
      {"feature_type", std::to_string(static_cast<int>(options.feature_extractor_type))},
      {"feature_backend", std::to_string(static_cast<int>(options.feature_backend))},
      {"max_num_features", std::to_string(options.max_num_features)},
      {"aliked_model_path", options.aliked_model_path.generic_string()},
      {"aliked_min_score", std::to_string(options.aliked_min_score)},
      {"camera_mode",
       std::string(CameraSharingModeName(options.camera_sharing_mode))},
      {"camera_model", std::to_string(static_cast<int>(options.camera_model_id))},
      {"focal_length", std::to_string(options.focal_length)},
      {"default_focal_factor", std::to_string(options.default_focal_length_factor)}};
  if (!options.aliked_model_path.empty() &&
      std::filesystem::exists(options.aliked_model_path)) {
    entries.emplace_back("aliked_model_size", std::to_string(
        std::filesystem::file_size(options.aliked_model_path)));
    entries.emplace_back("aliked_model_mtime", std::to_string(
        std::filesystem::last_write_time(options.aliked_model_path)
            .time_since_epoch().count()));
  }
  for (std::size_t idx = 0; idx < options.camera_params.size(); ++idx) {
    entries.emplace_back("camera_param_" + std::to_string(idx),
                         std::to_string(options.camera_params[idx]));
  }
  for (std::size_t idx = 0; idx < image_paths.size(); ++idx) {
    const std::filesystem::path& path = image_paths[idx];
    const std::string prefix = "image_" + std::to_string(idx) + "_";
    entries.emplace_back(prefix + "path", path.generic_string());
    entries.emplace_back(prefix + "size",
                         std::to_string(std::filesystem::file_size(path)));
    entries.emplace_back(
        prefix + "mtime",
        std::to_string(std::filesystem::last_write_time(path)
                           .time_since_epoch()
                           .count()));
  }
  return StableConfigHash(entries);
}

std::string ReconFrontendConfigHash(
    const ReconSfmControllerOptions& options,
    const std::string& feature_hash) {
  std::vector<std::pair<std::string, std::string>> entries =
      {{"stage", "geometry"},
       {"algorithm_version", "recon_sfm_geometry_v4_learned_frontend"},
       {"feature_hash", feature_hash},
       {"matcher_type", std::to_string(static_cast<int>(options.matcher_type))},
       {"matcher_backend", std::to_string(static_cast<int>(options.matcher_backend))},
       {"learned_model_path", options.learned_matcher_model_path.generic_string()},
       {"min_cosine_similarity", std::to_string(options.min_cosine_similarity)},
       {"lightglue_min_score", std::to_string(options.lightglue_min_score)},
       {"nearby_window", std::to_string(options.nearby_match_window)},
       {"nearby_ratio", std::to_string(options.nearby_match_ratio)},
       {"guided_ratio", std::to_string(options.guided_match_ratio)}};
  if (!options.learned_matcher_model_path.empty() &&
      std::filesystem::exists(options.learned_matcher_model_path)) {
    entries.emplace_back("learned_model_size", std::to_string(
        std::filesystem::file_size(options.learned_matcher_model_path)));
    entries.emplace_back("learned_model_mtime", std::to_string(
        std::filesystem::last_write_time(options.learned_matcher_model_path)
            .time_since_epoch().count()));
  }
  return StableConfigHash(entries);
}

std::string ReconSparseConfigHash(
    const ReconSfmControllerOptions& options,
    const std::string& frontend_hash) {
  return StableConfigHash(
      {{"stage", "recon_sfm"},
       {"algorithm_version", "recon_sfm_resume_multimodel_v6_pose_guided_tracks"},
       {"frontend_hash", frontend_hash},
       {"mapper_type", std::to_string(static_cast<int>(options.mapper_type))},
       {"input_model", options.input_model_path.generic_string()},
       {"input_format", ColmapModelFormatToString(options.input_model_format)},
       {"output_format", ColmapModelFormatToString(options.output_model_format)},
       {"multiple_models", options.multiple_models ? "1" : "0"},
       {"merge_models", options.merge_models ? "1" : "0"},
       {"max_num_models", std::to_string(options.max_num_models)},
       {"min_model_size", std::to_string(options.min_num_images_per_model)},
       {"max_registration_trials",
        std::to_string(options.max_registration_trials_per_image)},
       {"max_initialization_trials",
        std::to_string(options.max_initialization_trials)},
       {"structureless_registration",
        options.use_structureless_registration ? "1" : "0"},
       {"ba_groups",
        std::to_string(static_cast<int>(options.ba_refine_camera_params))},
       {"use_position_priors", options.use_position_priors ? "1" : "0"},
       {"position_prior_stddev",
        std::to_string(options.position_prior_stddev)},
       {"position_prior_loss_scale",
        std::to_string(options.position_prior_loss_scale)},
       {"position_prior_alignment_max_error",
        std::to_string(options.position_prior_alignment_max_error)}});
}

std::vector<std::filesystem::path> ReconFeatureRequiredFiles(
    const std::vector<ImageId>& image_ids) {
  std::vector<std::filesystem::path> files{"cameras.txt", "images.tsv"};
  for (const ImageId image_id : image_ids) {
    files.push_back(std::filesystem::path("features") /
                    (std::to_string(image_id) + ".bin"));
  }
  return files;
}

std::vector<std::filesystem::path> ReconOutputRequiredFiles(
    const ColmapModelFormat format, const std::size_t num_models) {
  std::vector<std::filesystem::path> files{
      "../points3D.ply", "../summary.txt", "../models.txt"};
  if (format == ColmapModelFormat::kText ||
      format == ColmapModelFormat::kTextAndBinary) {
    files.insert(files.end(), {"../cameras.txt", "../images.txt",
                               "../points3D.txt"});
  }
  if (format == ColmapModelFormat::kBinary ||
      format == ColmapModelFormat::kTextAndBinary) {
    files.insert(files.end(), {"../cameras.bin", "../images.bin",
                               "../points3D.bin"});
  }
  for (std::size_t model_idx = 0; model_idx < num_models; ++model_idx) {
    const std::filesystem::path model =
        std::filesystem::path("../models") / std::to_string(model_idx);
    files.push_back(model / "points3D.ply");
    files.push_back(model / "summary.txt");
    if (format == ColmapModelFormat::kText ||
        format == ColmapModelFormat::kTextAndBinary) {
      files.insert(files.end(), {model / "cameras.txt", model / "images.txt",
                                 model / "points3D.txt"});
    }
    if (format == ColmapModelFormat::kBinary ||
        format == ColmapModelFormat::kTextAndBinary) {
      files.insert(files.end(), {model / "cameras.bin", model / "images.bin",
                                 model / "points3D.bin"});
    }
  }
  return files;
}

}  // namespace

int ReconSfmController::Run(std::ostream& out) const {
  const ReconSfmControllerOptions& options = options_;
  const std::filesystem::path image_dir(options.image_path);
  const std::filesystem::path output_dir(options.output_path);
  const std::filesystem::path workspace_dir = output_dir / "workspace";
  const std::vector<std::filesystem::path> image_paths =
      ListImageFiles(image_dir);
  if (image_paths.size() < 2) {
    throw std::runtime_error("Need at least two input images.");
  }
  if (!IsFeatureMatcherCompatible(options.feature_extractor_type,
                                  options.matcher_type)) {
    throw std::invalid_argument(
        "Feature extractor and matcher types are incompatible");
  }
  if (IsAlikedFeature(options.feature_extractor_type) &&
      options.aliked_model_path.empty()) {
    throw std::invalid_argument("ALIKED extraction requires a model path");
  }
  if ((options.matcher_type == FeatureMatcherType::kSiftLightGlue ||
       options.matcher_type == FeatureMatcherType::kAlikedLightGlue) &&
      options.learned_matcher_model_path.empty()) {
    throw std::invalid_argument("LightGlue matching requires a model path");
  }
  if (options.mapper_type == SfmMapperType::kGlobal &&
      !options.input_model_path.empty()) {
    throw std::invalid_argument(
        "Global mapping does not yet support continuing an input model");
  }
  LOG(INFO) << "SfM pipeline: images=" << image_paths.size()
            << " image_path=" << image_dir.string()
            << " output_path=" << output_dir.string()
            << " feature_backend="
            << (options.feature_extractor_type == FeatureExtractorType::kSift
                    ? std::string(SiftBackendName(
                          ResolveSiftBackend(options.feature_backend)))
                    : std::string(FeatureExtractorTypeName(
                          options.feature_extractor_type)))
            << " max_features=" << options.max_num_features
            << " matcher=" << static_cast<int>(options.matcher_backend)
            << " cpu_threads=" << options.feature_num_threads
            << " resume=" << (options.resume ? 1 : 0)
            << " force=" << (options.force ? 1 : 0);
  const std::string feature_hash =
      ReconFeatureConfigHash(options, image_paths);
  const std::string frontend_hash =
      ReconFrontendConfigHash(options, feature_hash);
  const std::string sparse_hash =
      ReconSparseConfigHash(options, frontend_hash);
  if (options.resume && !options.force && options.input_model_path.empty() &&
      WorkspaceCacheIsValid(workspace_dir, "recon_sfm", sparse_hash)) {
    const WorkspaceCacheMetadata metadata =
        ReadWorkspaceCacheMetadata(workspace_dir, "recon_sfm");
    out << "SfM cache hit: 1\n"
        << "Registered images: " << metadata.values.at("registered_images")
        << "\nPoints3D: " << metadata.values.at("points3d") << "\n";
    LOG(INFO) << "SfM cache hit: registered_images="
              << metadata.values.at("registered_images")
              << " points3d=" << metadata.values.at("points3d");
    return 0;
  }

  const int cpu_threads =
      controllers::EffectiveFeatureExtractionThreads(
          options.feature_num_threads);
  const ScopedParallelThreads run_thread_scope(cpu_threads);
  const auto total_start = std::chrono::steady_clock::now();

  SfmCameraSetupOptions camera_options;
  camera_options.sharing_mode = options.camera_sharing_mode;
  camera_options.model_id = options.camera_model_id;
  camera_options.focal_length = options.focal_length;
  camera_options.camera_params = options.camera_params;
  camera_options.image_root = options.image_path;
  camera_options.default_focal_length_factor =
      options.default_focal_length_factor;
  const SfmCameraSetup camera_setup =
      BuildSfmCameraSetup(image_paths, camera_options);

  MatchGraph graph;
  Reconstruction reconstruction;
  for (const Camera& camera : camera_setup.cameras) {
    reconstruction.AddCamera(camera);
  }

  std::vector<ImageId> image_ids;
  image_ids.reserve(image_paths.size());
  for (std::size_t i = 0; i < image_paths.size(); ++i) {
    const ImageId image_id = static_cast<ImageId>(i + 1);
    image_ids.push_back(image_id);
    reconstruction.AddImage(camera_setup.images[i]);
  }
  WriteWorkspaceCameras(workspace_dir, camera_setup.cameras);
  WriteWorkspaceImages(workspace_dir, camera_setup.images);
  if (!options.input_model_path.empty()) {
    reconstruction = ApplyColmapModel(reconstruction,
                                      options.input_model_path,
                                      options.input_model_format);
  }

  double feature_extraction_seconds = 0.0;
  bool feature_cache_hit = false;
  if (options.resume && !options.force &&
      WorkspaceCacheIsValid(workspace_dir, "features", feature_hash)) {
    ReadWorkspaceFeatureSets(workspace_dir, &graph);
    feature_cache_hit = true;
  } else {
    const auto feature_start = std::chrono::steady_clock::now();
    const std::vector<FeatureSet> extracted_features =
        ExtractFeaturesForImages(image_paths, options);
    feature_extraction_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - feature_start).count();
    for (std::size_t i = 0; i < extracted_features.size(); ++i) {
      SetFeatures(&graph, image_ids[i], extracted_features[i]);
      WriteWorkspaceFeatureSet(workspace_dir, image_ids[i],
                               extracted_features[i]);
    }
    WriteWorkspaceCacheMetadata(
        workspace_dir, "features", feature_hash,
        {{"images", std::to_string(image_ids.size())},
         {"features", [&]() {
            std::size_t count = 0;
            for (const FeatureSet& features : extracted_features) {
              count += features.keypoints.size();
            }
            return std::to_string(count);
          }()},
         {"feature_extraction_seconds",
          std::to_string(feature_extraction_seconds)}},
        ReconFeatureRequiredFiles(image_ids));
  }


  ReconSfmRunStats run_stats;
  run_stats.mapper_type = options.mapper_type == SfmMapperType::kGlobal
                              ? "global"
                              : "incremental";
  const SiftBackend resolved_feature_backend =
      ResolveSiftBackend(options.feature_backend);
  run_stats.feature_backend = options.feature_extractor_type ==
                                      FeatureExtractorType::kSift
                                  ? std::string(SiftBackendName(
                                        resolved_feature_backend))
                                  : std::string(FeatureExtractorTypeName(
                                        options.feature_extractor_type));
  run_stats.cuda_sift_available = IsCudaSiftAvailable();
  run_stats.cuda_matcher_available = IsCudaDescriptorMatcherAvailable();
  const InitialFocalLengthEstimate& initial_focal =
      camera_setup.focal_estimates.front();
  if (camera_setup.cameras.size() == 1) {
    run_stats.initial_focal_length =
        camera_setup.cameras.front().MeanFocalLength();
    run_stats.initial_focal_source =
        options.focal_length > 0.0
            ? "user"
            : (camera_setup.cameras.front().has_prior_focal_length
                   ? "shared_exif_median"
                   : "image_size_default");
    run_stats.has_prior_focal_length =
        camera_setup.cameras.front().has_prior_focal_length;
  } else {
    run_stats.initial_focal_length = initial_focal.pixels;
    run_stats.initial_focal_source =
        InitialFocalLengthSourceName(initial_focal.source);
    run_stats.has_prior_focal_length = initial_focal.has_prior;
  }
  run_stats.camera_sharing_mode =
      CameraSharingModeName(options.camera_sharing_mode);
  run_stats.num_cameras = camera_setup.cameras.size();
  run_stats.num_position_prior_images = camera_setup.num_position_priors;
  run_stats.min_initial_focal_length = initial_focal.pixels;
  run_stats.max_initial_focal_length = initial_focal.pixels;
  for (const InitialFocalLengthEstimate& focal :
       camera_setup.focal_estimates) {
    run_stats.min_initial_focal_length =
        std::min(run_stats.min_initial_focal_length, focal.pixels);
    run_stats.max_initial_focal_length =
        std::max(run_stats.max_initial_focal_length, focal.pixels);
    if (focal.has_prior) {
      ++run_stats.num_prior_focal_images;
    }
  }
  run_stats.cpu_threads = cpu_threads;
  run_stats.feature_extraction_seconds = feature_extraction_seconds;
  for (const auto& [image_id, features] : graph.features) {
    (void)image_id;
    run_stats.num_extracted_features += features.keypoints.size();
  }
  out << "Feature cache hit: " << (feature_cache_hit ? 1 : 0) << "\n";

  DescriptorMatchingOptions matching_options;
  matching_options.backend = options.matcher_backend;
  matching_options.matcher_type = options.matcher_type;
  if (options.matcher_type == FeatureMatcherType::kAlikedBruteForce) {
    run_stats.matcher_backend = "aliked_bf";
  } else if (options.matcher_type == FeatureMatcherType::kSiftLightGlue) {
    run_stats.matcher_backend = "sift_lightglue";
  } else if (options.matcher_type == FeatureMatcherType::kAlikedLightGlue) {
    run_stats.matcher_backend = "aliked_lightglue";
  } else switch (options.matcher_backend) {
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
    const Image& image = reconstruction.GetImage(image_id);
    cameras_by_image.emplace(
        image_id, reconstruction.GetCamera(image.camera_id));
  }
  TwoViewGeometryBatchOptions geometry_options;
  geometry_options.keep_degenerate = true;

  WeakPairRematchingOptions weak_pair_options;
  // Guided descriptor re-search applies to every pair with a usable initial
  // geometry. The nearby window only controls first-pass ratio relaxation.
  weak_pair_options.nearby_image_window = 0;
  weak_pair_options.min_inlier_gain = 10;
  weak_pair_options.relaxed_matching = matching_options;
  weak_pair_options.relaxed_matching.max_ratio = options.guided_match_ratio;
  weak_pair_options.relaxed_matching.mutual_check = true;
  weak_pair_options.guided_matching.max_sampson_error_px =
      geometry_options.geometry_options.max_error_px;
  weak_pair_options.verification = geometry_options.geometry_options;
  WeakPairRematchingSummary weak_pair_summary;

  const bool frontend_cache_hit =
      options.resume && !options.force &&
      WorkspaceCacheIsValid(workspace_dir, "geometry", frontend_hash);
  if (frontend_cache_hit) {
    ReadWorkspaceMatches(workspace_dir, &graph);
    ReadWorkspaceVerifiedGeometries(workspace_dir, &graph);
    const WorkspaceCacheMetadata metadata =
        ReadWorkspaceCacheMetadata(workspace_dir, "geometry");
    run_stats.num_raw_matches =
        static_cast<std::size_t>(std::stoull(metadata.values.at("raw_matches")));
  } else if (options.matcher_type == FeatureMatcherType::kSiftBruteForce &&
             matching_options.backend ==
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
    MatchAllImagePairsTyped(image_ids, cameras_by_image, options,
                            matching_options, nearby_options, &graph);
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

    if (options.matcher_type == FeatureMatcherType::kSiftBruteForce) {
      const auto rematching_start = std::chrono::steady_clock::now();
      weak_pair_summary =
          RematchWeakImagePairs(cameras_by_image, weak_pair_options, &graph);
      run_stats.weak_pair_rematching_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - rematching_start).count();
    }
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
  const MatchGraphConsistencySummary consistency_summary =
      EnforceVerifiedMatchGraphConsistency(&graph);
  run_stats.num_verified_inliers_before_consistency =
      consistency_summary.num_input_edges;
  run_stats.num_pair_match_conflicts_removed =
      consistency_summary.num_invalid_edges +
      consistency_summary.num_exact_duplicate_edges +
      consistency_summary.num_pair_conflict_edges;
  run_stats.num_component_match_conflicts_removed =
      consistency_summary.num_cross_component_conflict_edges;
  run_stats.num_conflicted_match_components =
      consistency_summary.num_input_conflicted_components;
  run_stats.num_conflicting_match_observations =
      consistency_summary.num_input_conflicting_observations;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    (void)pair;
    run_stats.num_verified_inliers += geometry.inlier_matches.size();
  }
  if (!frontend_cache_hit) {
    WriteWorkspaceMatches(workspace_dir, graph);
    WriteWorkspaceVerifiedGeometries(workspace_dir, graph);
    WriteWorkspaceCacheMetadata(
        workspace_dir, "matching", frontend_hash,
        {{"raw_matches", std::to_string(run_stats.num_raw_matches)}},
        {"matches/raw_matches.tsv"});
    WriteWorkspaceCacheMetadata(
        workspace_dir, "geometry", frontend_hash,
        {{"raw_matches", std::to_string(run_stats.num_raw_matches)},
         {"verified_inliers",
          std::to_string(run_stats.num_verified_inliers)}},
        {"matches/raw_matches.tsv", "matches/verified_matches.tsv"});
  }
  out << "Frontend cache hit: " << (frontend_cache_hit ? 1 : 0) << "\n";

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
  mapping_options.mapper.use_structureless_registration =
      options.use_structureless_registration;
  mapping_options.registration_retry_reprojection_error_multiplier =
      options.registration_retry_reprojection_error_multiplier;
  mapping_options.registration_retry_max_reprojection_error_px =
      options.registration_retry_max_reprojection_error_px;
  const auto mapping_start = std::chrono::steady_clock::now();
  MultiModelMappingOptions multi_options;
  multi_options.mapping = mapping_options;
  multi_options.max_num_models =
      options.multiple_models ? options.max_num_models : 1;
  multi_options.min_num_images_per_model =
      options.min_num_images_per_model;
  multi_options.merge_models = options.merge_models;
  MultiModelMappingResult mapping_result;
  if (options.mapper_type == SfmMapperType::kGlobal) {
    GlobalMappingOptions global_options;
    global_options.view_graph.min_num_inliers = 15;
    global_options.view_graph.min_median_triangulation_angle_deg = 0.0;
    global_options.triangulation = mapping_options.mapper.triangulation;
    global_options.bundle_adjustment =
        mapping_options.global_bundle_adjustment;
    const GlobalMappingSummary global_summary =
        RunGlobalMapping(graph, global_options, &reconstruction);
    if (!global_summary.success) {
      throw std::runtime_error("Global SfM failed: " +
                               global_summary.message);
    }
    IncrementalMappingSummary compatible_summary;
    compatible_summary.success = true;
    compatible_summary.num_registered_images =
        global_summary.num_registered_images;
    compatible_summary.num_points3d = global_summary.num_points3d;
    compatible_summary.global_track_completion =
        global_summary.triangulation;
    compatible_summary.global_bundle_adjustment =
        global_summary.bundle_adjustment;
    compatible_summary.message = global_summary.message;
    mapping_result.num_unregistered_images =
        reconstruction.Images().size() -
        reconstruction.RegisteredImageIds().size();
    mapping_result.models.push_back(std::move(reconstruction));
    mapping_result.mapping_summaries.push_back(
        std::move(compatible_summary));
  } else {
    mapping_result = RunIncrementalMappingMultiple(
        graph, multi_options, std::move(reconstruction));
  }
  if (mapping_result.models.empty()) {
    throw std::runtime_error("SfM did not produce a valid sparse model");
  }
  std::vector<std::size_t> model_order(mapping_result.models.size());
  std::iota(model_order.begin(), model_order.end(), std::size_t{0});
  std::sort(model_order.begin(), model_order.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              return mapping_result.models[lhs].RegisteredImageIds().size() >
                     mapping_result.models[rhs].RegisteredImageIds().size();
            });
  Reconstruction& primary =
      mapping_result.models[model_order.front()];
  IncrementalMappingSummary& summary =
      mapping_result.mapping_summaries[model_order.front()];
  run_stats.num_sparse_models = mapping_result.models.size();
  run_stats.num_unregistered_images =
      mapping_result.num_unregistered_images;
  run_stats.incremental_mapping_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - mapping_start).count();
  if (options.use_position_priors) {
    const auto prior_start = std::chrono::steady_clock::now();
    double initial_error_sum = 0.0;
    double final_error_sum = 0.0;
    std::size_t error_count = 0;
    for (Reconstruction& model : mapping_result.models) {
      const PositionPriorAlignmentSummary alignment =
          AlignReconstructionToPositionPriors(
              options.position_prior_alignment_max_error, &model);
      if (!alignment.success) {
        LOG(WARNING) << "Position-prior alignment skipped: priors="
                     << alignment.num_priors;
        continue;
      }
      ++run_stats.num_position_prior_aligned_models;
      GlobalBundleAdjustmentOptions prior_ba =
          mapping_options.global_bundle_adjustment;
      prior_ba.gauge_fixing = BundleGaugeFixing::kNone;
      prior_ba.use_position_priors = true;
      prior_ba.position_prior_stddev = options.position_prior_stddev;
      prior_ba.position_prior_loss_scale =
          options.position_prior_loss_scale;
      const GlobalBundleAdjustmentSummary prior_summary =
          AdjustGlobalBundle(prior_ba, &model);
      if (!prior_summary.success) {
        LOG(WARNING) << "Position-prior BA failed: "
                     << prior_summary.message;
        continue;
      }
      initial_error_sum += prior_summary.initial_mean_position_error *
                           prior_summary.num_position_priors;
      final_error_sum += prior_summary.final_mean_position_error *
                         prior_summary.num_position_priors;
      error_count += prior_summary.num_position_priors;
      LOG(INFO) << "Position-prior refinement: priors="
                << prior_summary.num_position_priors
                << " alignment_inliers=" << alignment.num_inliers
                << " scale=" << alignment.scale
                << " initial_error="
                << prior_summary.initial_mean_position_error
                << " final_error="
                << prior_summary.final_mean_position_error;
    }
    if (error_count > 0) {
      run_stats.position_prior_initial_mean_error =
          initial_error_sum / static_cast<double>(error_count);
      run_stats.position_prior_final_mean_error =
          final_error_sum / static_cast<double>(error_count);
    }
    run_stats.position_prior_refinement_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     prior_start)
            .count();
  }
  const auto color_start = std::chrono::steady_clock::now();
  for (Reconstruction& model : mapping_result.models) {
    ExtractPointColors(&model);
  }
  run_stats.color_extraction_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - color_start).count();
  std::filesystem::create_directories(output_dir);
  const auto output_start = std::chrono::steady_clock::now();
  WritePairDiagnostics(output_dir, primary, graph);
  WriteRegistrationDiagnostics(output_dir, summary);
  WriteColmapReconstruction(output_dir, primary,
                            options.output_model_format);
  WritePlyPointCloud(output_dir / "points3D.ply", primary);
  for (std::size_t output_idx = 0; output_idx < model_order.size();
       ++output_idx) {
    const std::filesystem::path model_path =
        output_dir / "models" / std::to_string(output_idx);
    const Reconstruction& model =
        mapping_result.models[model_order[output_idx]];
    WriteColmapReconstruction(model_path, model,
                              options.output_model_format);
    WritePlyPointCloud(model_path / "points3D.ply", model);
  }
  std::ofstream models_file(output_dir / "models.txt");
  models_file << "# MODEL_INDEX REGISTERED_IMAGES POINTS3D PATH\n";
  for (std::size_t output_idx = 0; output_idx < model_order.size();
       ++output_idx) {
    const Reconstruction& model =
        mapping_result.models[model_order[output_idx]];
    models_file << output_idx << " " << model.RegisteredImageIds().size()
                << " " << model.Points3D().size() << " models/"
                << output_idx << "\n";
  }
  run_stats.output_write_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - output_start).count();
  run_stats.total_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - total_start).count();
  WriteRunSummary(output_dir, primary, summary, run_stats);
  for (std::size_t output_idx = 0; output_idx < model_order.size();
       ++output_idx) {
    const std::size_t model_idx = model_order[output_idx];
    WriteRunSummary(output_dir / "models" / std::to_string(output_idx),
                    mapping_result.models[model_idx],
                    mapping_result.mapping_summaries[model_idx], run_stats);
  }
  WriteWorkspaceCacheMetadata(
      workspace_dir, "recon_sfm", sparse_hash,
      {{"registered_images", std::to_string(summary.num_registered_images)},
       {"points3d", std::to_string(summary.num_points3d)},
       {"models", std::to_string(mapping_result.models.size())},
       {"total_seconds", std::to_string(run_stats.total_seconds)}},
      ReconOutputRequiredFiles(options.output_model_format,
                               mapping_result.models.size()));

  out << "Images: " << image_paths.size() << "\n"
      << "Cameras: " << camera_setup.cameras.size() << "\n"
      << "Registered images: " << summary.num_registered_images << "\n"
      << "Points3D: " << summary.num_points3d << "\n"
      << "Feature extraction seconds: " << std::setprecision(6)
      << feature_extraction_seconds << "\n"
      << "Mean reprojection error: "
      << std::setprecision(17)
      << ComputeReconstructionQualityStats(primary)
             .mean_reprojection_error
      << "\n"
      << "Output: " << output_dir.string() << "\n";
  LOG(INFO) << "SfM complete: success=" << (summary.success ? 1 : 0)
            << " registered_images=" << summary.num_registered_images
            << " points3d=" << summary.num_points3d
            << " models=" << mapping_result.models.size()
            << " observations="
            << ComputeReconstructionQualityStats(primary).num_observations
            << " total_seconds=" << run_stats.total_seconds;
  return summary.success ? 0 : 2;
}

}  // namespace colite::controllers
