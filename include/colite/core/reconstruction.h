#pragma once

#include <set>
#include <unordered_map>

#include "colite/core/camera.h"
#include "colite/core/image.h"
#include "colite/core/point3d.h"

namespace colite {

// 一个稀疏重建模型。主体结构参考 COLMAP Reconstruction。
class Reconstruction {
 public:
  const std::unordered_map<CameraId, Camera>& Cameras() const;
  const std::unordered_map<ImageId, Image>& Images() const;
  const std::unordered_map<Point3DId, Point3D>& Points3D() const;
  const std::set<ImageId>& RegisteredImageIds() const;

  bool ExistsCamera(CameraId camera_id) const;
  bool ExistsImage(ImageId image_id) const;
  bool ExistsPoint3D(Point3DId point3d_id) const;
  bool IsImageRegistered(ImageId image_id) const;

  const Camera& GetCamera(CameraId camera_id) const;
  Camera& GetCamera(CameraId camera_id);
  const Image& GetImage(ImageId image_id) const;
  Image& GetImage(ImageId image_id);
  const Point3D& GetPoint3D(Point3DId point3d_id) const;
  Point3D& GetPoint3D(Point3DId point3d_id);

  void AddCamera(Camera camera);
  void AddImage(Image image);

  Point3DId AddPoint3D(const Eigen::Vector3d& xyz, Track track);
  void AddPoint3D(Point3D point3d);
  bool DeleteObservation(Point3DId point3d_id,
                         ImageId image_id,
                         Point2DIdx point2d_idx);
  bool DeletePoint3D(Point3DId point3d_id);

  void RegisterImage(ImageId image_id);
  void DeregisterImage(ImageId image_id);

 private:
  std::unordered_map<CameraId, Camera> cameras_;
  std::unordered_map<ImageId, Image> images_;
  std::unordered_map<Point3DId, Point3D> points3d_;
  std::set<ImageId> registered_image_ids_;
  Point3DId next_point3d_id_ = 1;
};

}  // namespace colite
