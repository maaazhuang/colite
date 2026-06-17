#pragma once

#include <vector>

#include "colite/core/camera.h"
#include "colite/core/feature.h"
#include "colite/core/two_view_geometry.h"

namespace colite {

// 两视图几何验证参数。第一版使用 OpenCV 的鲁棒估计作为基线，
// 后续可以在不改变外部接口的前提下替换为自研 RANSAC/LORANSAC。
struct TwoViewGeometryOptions {
  // RANSAC 内点阈值，单位为像素。用于 F/H；E 会换算到归一化平面。
  double max_error_px = 4.0;

  // RANSAC 置信度。
  double confidence = 0.999;

  // 判定几何关系有效所需的最少内点数。
  int min_num_inliers = 15;

  // 是否同时估计单应矩阵。单应矩阵主要用于后续平面/全景退化判断。
  bool estimate_homography = true;
};

// 根据两张图像的特征匹配估计两视图几何关系。
// 输入匹配索引必须指向对应 FeatureSet 的 keypoints；无效索引会抛出异常。
TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1,
    const FeatureSet& features1,
    const Camera& camera2,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& matches,
    const TwoViewGeometryOptions& options = {});

}  // namespace colite
