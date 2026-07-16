#pragma once

#include <vector>

#include "colite/core/camera.h"
#include "colite/core/feature.h"
#include "colite/core/two_view_geometry.h"

namespace colite {

// 两视图几何验证参数。当前实现使用内部 RANSAC + DLT/八点法估计 F/H/E。
struct TwoViewGeometryOptions {
  // RANSAC 内点阈值，单位为像素。用于 F/H；E 会换算到归一化平面。
  double max_error_px = 4.0;

  // RANSAC 置信度。当前保留该参数用于接口兼容，后续 LORANSAC 会使用。
  double confidence = 0.999;

  // 内部 RANSAC 最大迭代次数。
  // COLMAP-style adaptive robust-estimation bounds. The estimator always
  // performs at least min_num_iterations and may continue until confidence is
  // reached or max_num_iterations is exhausted.
  int min_num_iterations = 100;
  int max_num_iterations = 10000;

  // 判定几何关系有效所需的最少内点数。
  int min_num_inliers = 15;

  // E 的内点数需要接近 F 的内点数，才认为标定内参在该图像对上可信。
  double min_E_F_inlier_ratio = 0.95;

  // H 的内点数接近 E/F 时，认为该图像对更可能是平面或纯旋转退化。
  double max_H_inlier_ratio = 0.8;

  // 三角化角低于该阈值时，将 H 退化从平面进一步判定为全景/纯旋转。
  double max_panoramic_triangulation_angle_deg = 0.1;

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
