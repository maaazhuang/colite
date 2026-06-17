#pragma once

#include <cstddef>
#include <string>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// 新图像注册参数。第一版使用 OpenCV solvePnPRansac 作为可靠基线，
// 后续可替换为自研 P3P/EPnP + LORANSAC。
struct ImageRegistrarOptions {
  std::size_t min_num_correspondences = 15;
  std::size_t min_num_inliers = 12;
  int max_num_iterations = 100;
  double max_reprojection_error_px = 4.0;
  double confidence = 0.999;
};

struct ImageRegistrationSummary {
  bool success = false;
  ImageId image_id = kInvalidImageId;
  std::size_t num_correspondences = 0;
  std::size_t num_inliers = 0;
  std::string message;
};

// 使用已注册图像的 2D-3D 观测注册一张新图像。
// 函数会从 MatchGraph::features 同步目标图像 2D 点，并通过 verified_geometries
// 找到与已注册邻居之间的内点匹配，再调用 PnP RANSAC 估计 world -> camera 位姿。
ImageRegistrationSummary RegisterImageWithExistingPoints(
    const MatchGraph& graph,
    ImageId image_id,
    const ImageRegistrarOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
