#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// 增量 SfM 初始化参数。第一版只负责选择两张初始图像并三角化初始点云，
// 后续图像注册、局部 BA 和全局 BA 会在单独模块中继续推进。
struct IncrementalInitializerOptions {
  std::size_t min_num_inliers = 15;
  std::size_t min_num_triangulated_points = 15;
  double min_triangulation_angle_deg = 0.5;
  double max_reprojection_error_px = 4.0;
};

struct IncrementalInitializerSummary {
  bool success = false;
  ImagePair initial_pair;
  std::size_t num_triangulated_points = 0;
  std::string message;
};

// 从已验证两视图图中选择最适合初始化的图像对。
// 评分优先使用内点数，其次使用中位三角化角，避免选择基线过小的图像对。
std::optional<ImagePair> SelectInitialImagePair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options = {});

// 使用指定图像对初始化 Reconstruction。Reconstruction 需要已经包含对应 Image
// 和 Camera；图像的 points2d 可以为空，函数会从 MatchGraph::features 同步。
IncrementalInitializerSummary InitializeReconstructionFromPair(
    const MatchGraph& graph,
    ImagePair initial_pair,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction);

// 自动选择最佳初始图像对并初始化 Reconstruction。
IncrementalInitializerSummary InitializeReconstructionFromBestPair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
