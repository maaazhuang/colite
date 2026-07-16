#pragma once

#include <cstddef>
#include <filesystem>

namespace colite::controllers {

struct ColmapCompareControllerOptions {
  std::filesystem::path reference_path;
  std::filesystem::path candidate_path;
  std::filesystem::path output_path;
};

struct ColmapCompareControllerSummary {
  std::size_t reference_images = 0;
  std::size_t candidate_images = 0;
  std::size_t common_images = 0;
  std::size_t reference_only_images = 0;
  std::size_t candidate_only_images = 0;
  std::size_t reference_points = 0;
  std::size_t candidate_points = 0;
  std::size_t common_points_by_id = 0;
  std::size_t reference_only_points = 0;
  std::size_t candidate_only_points = 0;
  std::size_t common_pairs = 0;
  bool similarity_alignment_valid = false;
  double similarity_scale = 1.0;
  double mean_rotation_error_deg = -1.0;
  double median_rotation_error_deg = -1.0;
  double max_rotation_error_deg = -1.0;
  double mean_center_error = -1.0;
  double median_center_error = -1.0;
  double max_center_error = -1.0;
  double mean_center_error_over_baseline = -1.0;
  double max_center_error_over_baseline = -1.0;
  double reference_mean_reprojection_error = -1.0;
  double candidate_mean_reprojection_error = -1.0;
  double reference_mean_track_length = 0.0;
  double candidate_mean_track_length = 0.0;
};

class ColmapCompareController {
 public:
  explicit ColmapCompareController(ColmapCompareControllerOptions options);

  ColmapCompareControllerSummary Run() const;

 private:
  ColmapCompareControllerOptions options_;
};

}  // namespace colite::controllers
