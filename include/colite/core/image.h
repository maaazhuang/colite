#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "colite/core/point2d.h"
#include "colite/core/pose.h"
#include "colite/core/types.h"

namespace colite {

// 输入图像/视图。整体参考 COLMAP Image：图像直接持有 2D 点列表，便于增量 SfM。
struct Image {
  ImageId id = kInvalidImageId;
  CameraId camera_id = kInvalidCameraId;
  std::string name;
  std::string file_path;
  bool registered = false;
  std::optional<Pose3> cam_from_world;
  std::vector<Point2D> points2d;

  bool HasPose() const {
    return cam_from_world.has_value();
  }

  std::size_t NumPoints2D() const {
    return points2d.size();
  }

  std::size_t NumPoints3D() const {
    std::size_t count = 0;
    for (const Point2D& point : points2d) {
      if (point.HasPoint3D()) {
        ++count;
      }
    }
    return count;
  }
};

}  // namespace colite

