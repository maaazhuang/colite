#pragma once

#include <Eigen/Core>

#include "colite/core/types.h"

namespace colite {

// 图像中的 2D 点。它通常对应一个特征点，也可能已经关联到一个 3D 点。
struct Point2D {
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
  Point2DIdx feature_idx = kInvalidPoint2DIdx;
  Point3DId point3d_id = kInvalidPoint3DId;

  bool HasPoint3D() const {
    return point3d_id != kInvalidPoint3DId;
  }
};

}  // namespace colite

