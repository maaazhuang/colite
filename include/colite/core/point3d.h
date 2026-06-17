#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "colite/core/track.h"
#include "colite/core/types.h"

namespace colite {

// 稀疏重建中的 3D 点，包含空间坐标、颜色、误差和观测 track。
struct Point3D {
  Point3DId id = kInvalidPoint3DId;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Matrix<std::uint8_t, 3, 1> color =
      Eigen::Matrix<std::uint8_t, 3, 1>::Zero();
  double reprojection_error = -1.0;
  Track track;

  bool HasError() const {
    return reprojection_error >= 0.0;
  }
};

}  // namespace colite

