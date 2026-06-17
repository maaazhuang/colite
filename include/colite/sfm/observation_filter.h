#pragma once

#include <cstddef>
#include <string>

#include "colite/core/reconstruction.h"

namespace colite {

// 局部/增量建图后的坏观测过滤参数。第一版参考 COLMAP 的核心规则：
// 重投影误差过大删观测，track 太短或三角化角过小删 3D 点。
struct ObservationFilterOptions {
  double max_reprojection_error_px = 4.0;
  double min_triangulation_angle_deg = 0.5;
  std::size_t min_track_length = 2;
};

struct ObservationFilterSummary {
  bool success = false;
  std::size_t num_removed_observations = 0;
  std::size_t num_removed_points = 0;
  std::string message;
};

ObservationFilterSummary FilterObservationsAndPoints(
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
