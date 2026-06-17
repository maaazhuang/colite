#include "colite/core/reconstruction.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace colite {

const std::unordered_map<CameraId, Camera>& Reconstruction::Cameras() const {
  return cameras_;
}

const std::unordered_map<ImageId, Image>& Reconstruction::Images() const {
  return images_;
}

const std::unordered_map<Point3DId, Point3D>& Reconstruction::Points3D() const {
  return points3d_;
}

const std::set<ImageId>& Reconstruction::RegisteredImageIds() const {
  return registered_image_ids_;
}

bool Reconstruction::ExistsCamera(CameraId camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool Reconstruction::ExistsImage(ImageId image_id) const {
  return images_.find(image_id) != images_.end();
}

bool Reconstruction::ExistsPoint3D(Point3DId point3d_id) const {
  return points3d_.find(point3d_id) != points3d_.end();
}

bool Reconstruction::IsImageRegistered(ImageId image_id) const {
  return registered_image_ids_.find(image_id) != registered_image_ids_.end();
}

const Camera& Reconstruction::GetCamera(CameraId camera_id) const {
  return cameras_.at(camera_id);
}

Camera& Reconstruction::GetCamera(CameraId camera_id) {
  return cameras_.at(camera_id);
}

const Image& Reconstruction::GetImage(ImageId image_id) const {
  return images_.at(image_id);
}

Image& Reconstruction::GetImage(ImageId image_id) {
  return images_.at(image_id);
}

const Point3D& Reconstruction::GetPoint3D(Point3DId point3d_id) const {
  return points3d_.at(point3d_id);
}

Point3D& Reconstruction::GetPoint3D(Point3DId point3d_id) {
  return points3d_.at(point3d_id);
}

void Reconstruction::AddCamera(Camera camera) {
  if (camera.id == kInvalidCameraId) {
    throw std::invalid_argument("Camera id is invalid.");
  }
  if (!camera.VerifyParams()) {
    throw std::invalid_argument("Camera parameters are invalid.");
  }
  cameras_.emplace(camera.id, std::move(camera));
}

void Reconstruction::AddImage(Image image) {
  if (image.id == kInvalidImageId) {
    throw std::invalid_argument("Image id is invalid.");
  }
  if (!ExistsCamera(image.camera_id)) {
    throw std::invalid_argument("Image references a missing camera.");
  }
  if (image.registered) {
    registered_image_ids_.insert(image.id);
  }
  images_.emplace(image.id, std::move(image));
}

Point3DId Reconstruction::AddPoint3D(const Eigen::Vector3d& xyz, Track track) {
  Point3D point3d;
  point3d.id = next_point3d_id_++;
  point3d.xyz = xyz;
  point3d.track = std::move(track);
  AddPoint3D(std::move(point3d));
  return next_point3d_id_ - 1;
}

void Reconstruction::AddPoint3D(Point3D point3d) {
  if (point3d.id == kInvalidPoint3DId) {
    throw std::invalid_argument("Point3D id is invalid.");
  }
  if (ExistsPoint3D(point3d.id)) {
    throw std::invalid_argument("Point3D id already exists.");
  }

  for (const TrackElement& element : point3d.track.elements) {
    Image& image = GetImage(element.image_id);
    if (element.point2d_idx >= image.points2d.size()) {
      throw std::out_of_range("Track element references a missing 2D point.");
    }
    Point2D& point2d = image.points2d[element.point2d_idx];
    if (point2d.HasPoint3D() && point2d.point3d_id != point3d.id) {
      throw std::invalid_argument("2D point is already linked to another 3D point.");
    }
    point2d.point3d_id = point3d.id;
  }

  next_point3d_id_ = std::max(next_point3d_id_, point3d.id + 1);
  points3d_.emplace(point3d.id, std::move(point3d));
}

bool Reconstruction::DeleteObservation(Point3DId point3d_id,
                                       ImageId image_id,
                                       Point2DIdx point2d_idx) {
  if (!ExistsPoint3D(point3d_id)) {
    return false;
  }

  Point3D& point3d = GetPoint3D(point3d_id);
  const auto element_it = std::find(
      point3d.track.elements.begin(), point3d.track.elements.end(),
      TrackElement{image_id, point2d_idx});
  if (element_it == point3d.track.elements.end()) {
    return false;
  }

  if (ExistsImage(image_id)) {
    Image& image = GetImage(image_id);
    if (point2d_idx < image.points2d.size() &&
        image.points2d[point2d_idx].point3d_id == point3d_id) {
      image.points2d[point2d_idx].point3d_id = kInvalidPoint3DId;
    }
  }
  point3d.track.elements.erase(element_it);
  return true;
}

bool Reconstruction::DeletePoint3D(Point3DId point3d_id) {
  const auto point_it = points3d_.find(point3d_id);
  if (point_it == points3d_.end()) {
    return false;
  }

  // 删除整点时先解除所有反向 2D-3D 关联，避免留下悬挂引用。
  const Track track = point_it->second.track;
  for (const TrackElement& element : track.elements) {
    if (!ExistsImage(element.image_id)) {
      continue;
    }
    Image& image = GetImage(element.image_id);
    if (element.point2d_idx < image.points2d.size() &&
        image.points2d[element.point2d_idx].point3d_id == point3d_id) {
      image.points2d[element.point2d_idx].point3d_id = kInvalidPoint3DId;
    }
  }
  points3d_.erase(point_it);
  return true;
}

void Reconstruction::RegisterImage(ImageId image_id) {
  Image& image = GetImage(image_id);
  if (!image.HasPose()) {
    throw std::invalid_argument("Registered image must have a pose.");
  }
  image.registered = true;
  registered_image_ids_.insert(image_id);
}

void Reconstruction::DeregisterImage(ImageId image_id) {
  Image& image = GetImage(image_id);
  image.registered = false;
  registered_image_ids_.erase(image_id);
}

}  // namespace colite
