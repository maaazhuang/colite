#pragma once

#include <vector>

#include "colite/core/types.h"

namespace colite {

// 一个 3D 点在某张图像上的观测位置。
struct TrackElement {
  ImageId image_id = kInvalidImageId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;

  bool operator==(const TrackElement& other) const {
    return image_id == other.image_id && point2d_idx == other.point2d_idx;
  }
};

// 多视图 track：同一个空间点被多张图像中的哪些 2D 点观测到。
struct Track {
  std::vector<TrackElement> elements;

  std::size_t Length() const {
    return elements.size();
  }

  bool Empty() const {
    return elements.empty();
  }

  void AddElement(ImageId image_id, Point2DIdx point2d_idx) {
    elements.push_back({image_id, point2d_idx});
  }
};

}  // namespace colite

