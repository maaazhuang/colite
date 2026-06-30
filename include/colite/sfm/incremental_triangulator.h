#pragma once

#include <cstddef>
#include <string>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// 增量三角化参数。命名参考 COLMAP IncrementalTriangulator::Options，
// 当前版本实现其中 Create 新点的子集，Complete/Merge 后续单独补齐。
struct IncrementalTriangulatorOptions {
  double min_triangulation_angle_deg = 0.5;
  double min_pair_median_triangulation_angle_deg = 0.0;
  double create_max_reprojection_error_px = 4.0;
  // 新建点后追加执行的 track completion / merge 轮数上限，接近 COLMAP 的 complete-merge-create 局部循环。
  std::size_t max_track_update_iterations = 2;
  bool ignore_two_view_tracks = false;
};

struct IncrementalTriangulatorSummary {
  ImageId image_id = kInvalidImageId;
  std::size_t num_created_points = 0;
  std::size_t num_created_observations = 0;
  std::size_t num_completed_observations = 0;
  std::size_t num_merged_points = 0;
  std::size_t num_merged_observations = 0;
  std::size_t num_filtered_observations = 0;
  std::size_t num_filtered_points = 0;
  std::string message;
};

// 为一张已注册图像创建新的 3D 点。
// 函数会遍历该图像和已注册邻居之间的 verified matches，只对两端都尚未关联 3D
// 的观测进行三角化，并将通过几何检查的新 track 写入 Reconstruction。
IncrementalTriangulatorSummary TriangulateRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

// 为一张已注册图像补全已有 3D 点的观测。
// 若目标图像的 2D 点尚未关联 3D，但它通过 verified match 连接到已注册邻居
// 的已有 3D 点，则在重投影误差检查通过后将该观测加入对应 track。
IncrementalTriangulatorSummary CompleteTracksForRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
