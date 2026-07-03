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
};

class ColmapCompareController {
 public:
  explicit ColmapCompareController(ColmapCompareControllerOptions options);

  ColmapCompareControllerSummary Run() const;

 private:
  ColmapCompareControllerOptions options_;
};

}  // namespace colite::controllers
