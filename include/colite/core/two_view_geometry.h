#pragma once

#include <vector>

#include <Eigen/Core>

#include "colite/core/feature.h"
#include "colite/core/pose.h"

namespace colite {

// 两视图几何验证结果，用于初始化、建图连通性判断和相对位姿恢复。
struct TwoViewGeometry {
  enum class Type {
    kUndefined,
    kDegenerate,
    kCalibratedEssential,
    kUncalibratedFundamental,
    kPlanarHomography,
    kPanoramic,
    kMultiple,
  };

  Type type = Type::kUndefined;
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  Pose3 cam2_from_cam1;
  std::vector<FeatureMatch> inlier_matches;
  double median_triangulation_angle = -1.0;
};

}  // namespace colite

