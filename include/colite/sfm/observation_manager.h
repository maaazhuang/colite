#pragma once

#include <optional>

#include <Eigen/Core>

#include "colite/core/reconstruction.h"

namespace colite {

// 集中管理 SfM 中的 2D-3D 观测关系，参考 COLMAP ObservationManager 的职责边界。
// Reconstruction 仍负责底层存储；该类负责所有跨 Image / Point3D 的一致性检查和重定向。
class ObservationManager {
 public:
  explicit ObservationManager(Reconstruction* reconstruction);

  bool AddObservation(Point3DId point3d_id,
                      ImageId image_id,
                      Point2DIdx point2d_idx);

  std::optional<Point3DId> AddPoint3D(const Eigen::Vector3d& xyz,
                                      const Track& track);

  bool MergePoints(Point3DId target_point3d_id,
                   Point3DId source_point3d_id,
                   const Eigen::Vector3d& merged_xyz);

 private:
  bool HasReconstruction() const;
  bool HasValidObservation(ImageId image_id, Point2DIdx point2d_idx) const;
  Reconstruction* reconstruction_ = nullptr;
};

}  // namespace colite
