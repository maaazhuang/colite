#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// SfM 映射期间 MatchGraph 保持只读。预先构建一次观测对应关系，避免每次局部
// 三角化都重新扫描并分配完整的 verified correspondence graph。
struct IncrementalTriangulatorCache {
  using ObservationKey = std::pair<ImageId, Point2DIdx>;
  struct CorrespondenceAdjacency {
    using Neighbors = std::vector<ObservationKey>;

    void ResizeImage(ImageId image_id, std::size_t num_points2d) {
      by_image[image_id].resize(num_points2d);
    }

    Neighbors* Mutable(const ObservationKey& observation) {
      auto& image = by_image[observation.first];
      if (observation.second >= image.size()) {
        image.resize(observation.second + 1);
      }
      return &image[observation.second];
    }

    const Neighbors* Find(const ObservationKey& observation) const {
      const auto image_it = by_image.find(observation.first);
      if (image_it == by_image.end() ||
          observation.second >= image_it->second.size()) {
        return nullptr;
      }
      return &image_it->second[observation.second];
    }

    std::map<ImageId, std::vector<Neighbors>> by_image;
  };

  explicit IncrementalTriangulatorCache(const MatchGraph& graph);

  CorrespondenceAdjacency correspondence_adjacency;
};

// 增量三角化参数。命名和默认值参考 COLMAP
// IncrementalTriangulator::Options。
struct IncrementalTriangulatorOptions {
  // Match COLMAP's one-hop creation default. Robust creation still keeps only
  // the best geometrically consistent observation from each image when a
  // caller explicitly enables transitive collection.
  std::size_t max_transitivity = 1;
  double min_triangulation_angle_deg = 1.5;
  double min_pair_median_triangulation_angle_deg = 0.0;
  double create_max_angle_error_deg = 2.0;
  double continue_max_angle_error_deg = 2.0;
  double merge_max_reprojection_error_px = 4.0;
  double complete_max_reprojection_error_px = 4.0;
  std::size_t complete_max_transitivity = 6;
  // Final-pose descriptor-guided completion. These conservative defaults
  // require a target feature to agree with at least two existing track
  // descriptors and to lie within two pixels of the projected 3D point.
  double pose_guided_max_reprojection_error_px = 2.0;
  double pose_guided_max_descriptor_angle_rad = 0.301;
  std::size_t pose_guided_min_descriptor_support = 2;
  // 新建点后追加执行的 track completion / merge 轮数上限，接近 COLMAP 的 complete-merge-create 局部循环。
  std::size_t max_track_update_iterations = 5;
  // During incremental mapping, only complete the image being processed so
  // future PnP visibility remains stable. Global completion may enable this to
  // propagate tracks to either endpoint of a verified correspondence.
  bool complete_neighbor_observations = false;
  bool ignore_two_view_tracks = true;
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
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache = nullptr);

// 为一张已注册图像补全已有 3D 点的观测。
// 若目标图像的 2D 点尚未关联 3D，但它通过 verified match 连接到已注册邻居
// 的已有 3D 点，则在重投影误差检查通过后将该观测加入对应 track。
IncrementalTriangulatorSummary CompleteTracksForRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

// COLMAP-style global track refinement: recursively complete tracks through
// the correspondence graph and then merge connected compatible 3D points.
IncrementalTriangulatorSummary CompleteAndMergeAllTracks(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache = nullptr);

// Extend existing tracks after camera poses stabilize. Unlike correspondence
// graph traversal, this searches unassigned features near each projected 3D
// point and requires descriptor agreement with multiple track observations.
IncrementalTriangulatorSummary CompleteTracksByPoseAndDescriptors(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
