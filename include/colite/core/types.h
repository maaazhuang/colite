#pragma once

#include <cstdint>
#include <limits>

namespace colite {

// 核心 ID 类型。保持显式命名，避免在相机、图像、点云之间误传 ID。
using CameraId = std::uint32_t;
using ImageId = std::uint32_t;
using Point2DIdx = std::uint32_t;
using Point3DId = std::uint64_t;

// 无效 ID 哨兵值。使用最大值而不是 0，方便从 0 或 1 开始分配真实 ID。
constexpr CameraId kInvalidCameraId = std::numeric_limits<CameraId>::max();
constexpr ImageId kInvalidImageId = std::numeric_limits<ImageId>::max();
constexpr Point2DIdx kInvalidPoint2DIdx =
    std::numeric_limits<Point2DIdx>::max();
constexpr Point3DId kInvalidPoint3DId =
    std::numeric_limits<Point3DId>::max();

}  // namespace colite

