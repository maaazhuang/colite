#include "colite/core/parallel.h"
#include "colite/features/sift_extractor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

namespace colite {
namespace {

struct OverlapMetrics {
  double matched_percent = 0.0;
  double strict_percent = 0.0;
};

float AngleError(const float lhs, const float rhs) {
  float difference = std::fmod(std::abs(lhs - rhs), 360.0F);
  return std::min(difference, 360.0F - difference);
}

std::uint64_t CellKey(const int x, const int y) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
         static_cast<std::uint32_t>(y);
}

// Keep this metric identical to D:/Code/CXX/Sift/demo/cuda_sift_compare.cpp.
OverlapMetrics CompareCpuGpu(const FeatureSet& cpu, const FeatureSet& gpu) {
  constexpr float kCellSize = 1.5F;
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid;
  grid.reserve(cpu.keypoints.size());
  for (std::size_t index = 0; index < cpu.keypoints.size(); ++index) {
    const FeatureKeypoint& keypoint = cpu.keypoints[index];
    grid[CellKey(static_cast<int>(std::floor(keypoint.x / kCellSize)),
                 static_cast<int>(std::floor(keypoint.y / kCellSize)))]
        .push_back(index);
  }

  std::size_t matched = 0;
  std::size_t strict = 0;
  for (const FeatureKeypoint& query : gpu.keypoints) {
    const FeatureKeypoint* best = nullptr;
    float best_score = std::numeric_limits<float>::infinity();
    const int cell_x = static_cast<int>(std::floor(query.x / kCellSize));
    const int cell_y = static_cast<int>(std::floor(query.y / kCellSize));
    for (int cell_dy = -1; cell_dy <= 1; ++cell_dy) {
      for (int cell_dx = -1; cell_dx <= 1; ++cell_dx) {
        const auto found = grid.find(CellKey(cell_x + cell_dx,
                                             cell_y + cell_dy));
        if (found == grid.end()) continue;
        for (const std::size_t index : found->second) {
          const FeatureKeypoint& candidate = cpu.keypoints[index];
          const float dx = query.x - candidate.x;
          const float dy = query.y - candidate.y;
          if (std::abs(dx) > 1.5F || std::abs(dy) > 1.5F) continue;
          const float scale_error =
              std::abs(query.scale - candidate.scale) /
              std::max(candidate.scale, 1e-6F);
          const float angle_error =
              AngleError(query.orientation, candidate.orientation);
          if (scale_error > 0.08F || angle_error > 8.0F) continue;
          const float score = dx * dx + dy * dy + scale_error +
                              angle_error / 180.0F;
          if (score < best_score) {
            best_score = score;
            best = &candidate;
          }
        }
      }
    }
    if (best == nullptr) continue;
    ++matched;
    const float position = std::hypot(query.x - best->x, query.y - best->y);
    const float scale_error =
        std::abs(query.scale - best->scale) / std::max(best->scale, 1e-6F);
    const float angle_error = AngleError(query.orientation, best->orientation);
    if (position <= 0.05F && scale_error <= 0.001F &&
        angle_error <= 0.1F) {
      ++strict;
    }
  }

  const double denominator = static_cast<double>(
      std::max(cpu.keypoints.size(), gpu.keypoints.size()));
  if (denominator == 0.0) return {};
  return {100.0 * static_cast<double>(matched) / denominator,
          100.0 * static_cast<double>(strict) / denominator};
}

FeatureSet TimedExtract(const cv::Mat& image,
                        const SiftExtractionOptions& options,
                        double* milliseconds) {
  const auto start = std::chrono::steady_clock::now();
  FeatureSet result = ExtractSiftFeatures(image, options);
  *milliseconds = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}

void VerifyCpuGpuOverlap(const SiftBackend cpu_backend,
                         const SiftBackend gpu_backend) {
  if (!IsCudaSiftAvailable()) GTEST_SKIP() << "CUDA SIFT is unavailable.";

  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(image_path)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }
  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(image.empty()) << image_path.string();
  parallel::setNumThreads(20);

  SiftExtractionOptions cpu_options;
  cpu_options.backend = cpu_backend;
  cpu_options.max_num_features = 0;
  cpu_options.use_root_sift = false;
  cpu_options.cuda_num_octaves = 0;
  if (cpu_backend == SiftBackend::kVlSift) {
    // 3.4 / 255 is the normalized equivalent of the Sift project benchmark's
    // VLFeat peak_thresh=3.4 on a [0, 255] float image.
    cpu_options.contrast_threshold = 3.4 / 255.0;
    cpu_options.vl_num_octaves = -1;
    cpu_options.max_num_orientations = 4;
  }
  SiftExtractionOptions gpu_options = cpu_options;
  gpu_options.backend = gpu_backend;

  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  const FeatureSet cpu = TimedExtract(image, cpu_options, &cpu_ms);
  const FeatureSet gpu = TimedExtract(image, gpu_options, &gpu_ms);
  const OverlapMetrics metrics = CompareCpuGpu(cpu, gpu);
  std::cout << SiftBackendName(cpu_backend) << "_cpu_points="
            << cpu.keypoints.size() << " " << SiftBackendName(gpu_backend)
            << "_gpu_points=" << gpu.keypoints.size()
            << " cpu_ms=" << cpu_ms << " gpu_ms=" << gpu_ms
            << " matched_percent=" << metrics.matched_percent
            << " strict_percent=" << metrics.strict_percent << '\n';

  EXPECT_GE(metrics.matched_percent, 99.9);
  EXPECT_GE(metrics.strict_percent, 99.9);
}

TEST(SiftCpuCudaOverlapTest, OpenCvModesOverlapAtLeast999Percent) {
  VerifyCpuGpuOverlap(SiftBackend::kCvSift, SiftBackend::kCudaSift);
}

TEST(SiftCpuCudaOverlapTest, VlFeatModesOverlapAtLeast999Percent) {
  VerifyCpuGpuOverlap(SiftBackend::kVlSift, SiftBackend::kCudaVlSift);
}

TEST(SiftCpuWorkloadTest, ReportsColiteDefault8192OutputWorkload) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(image_path)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }
  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
  ASSERT_FALSE(image.empty()) << image_path.string();
  parallel::setNumThreads(20);

  SiftExtractionOptions options;
  options.max_num_features = 8192;
  options.use_root_sift = false;
  options.backend = SiftBackend::kCvSift;
  double cv_ms = 0.0;
  const FeatureSet cv_features = TimedExtract(image, options, &cv_ms);
  options.backend = SiftBackend::kVlSift;
  double vl_ms = 0.0;
  const FeatureSet vl_features = TimedExtract(image, options, &vl_ms);

  std::cout << "colite_8192_cvsift_points=" << cv_features.keypoints.size()
            << " cv_ms=" << cv_ms
            << " colite_8192_vlsift_points=" << vl_features.keypoints.size()
            << " vl_ms=" << vl_ms << '\n';
  EXPECT_EQ(cv_features.keypoints.size(), std::size_t{8192});
  EXPECT_GE(vl_features.keypoints.size(), std::size_t{8192});
}

}  // namespace
}  // namespace colite
