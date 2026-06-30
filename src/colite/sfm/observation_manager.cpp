#include "colite/sfm/observation_manager.h"

#include <set>
#include <utility>

namespace colite {
namespace {

bool TrackHasObservation(const Track& track,
                         ImageId image_id,
                         Point2DIdx point2d_idx) {
  for (const TrackElement& element : track.elements) {
    if (element.image_id == image_id && element.point2d_idx == point2d_idx) {
      return true;
    }
  }
  return false;
}

}  // namespace

ObservationManager::ObservationManager(Reconstruction* reconstruction)
    : reconstruction_(reconstruction) {}

bool ObservationManager::AddObservation(Point3DId point3d_id,
                                        ImageId image_id,
                                        Point2DIdx point2d_idx) {
  if (!HasReconstruction() || !reconstruction_->ExistsPoint3D(point3d_id) ||
      !HasValidObservation(image_id, point2d_idx)) {
    return false;
  }

  Image& image = reconstruction_->GetImage(image_id);
  Point2D& point2d = image.points2d[point2d_idx];
  if (point2d.HasPoint3D() && point2d.point3d_id != point3d_id) {
    return false;
  }

  Point3D& point3d = reconstruction_->GetPoint3D(point3d_id);
  if (TrackHasObservation(point3d.track, image_id, point2d_idx)) {
    point2d.point3d_id = point3d_id;
    return false;
  }

  point2d.point3d_id = point3d_id;
  point3d.track.AddElement(image_id, point2d_idx);
  return true;
}

std::optional<Point3DId> ObservationManager::AddPoint3D(
    const Eigen::Vector3d& xyz,
    const Track& track) {
  if (!HasReconstruction() || track.elements.empty()) {
    return std::nullopt;
  }

  std::set<ImageId> image_ids;
  for (const TrackElement& element : track.elements) {
    if (!HasValidObservation(element.image_id, element.point2d_idx) ||
        !image_ids.insert(element.image_id).second) {
      return std::nullopt;
    }
    const Image& image = reconstruction_->GetImage(element.image_id);
    const Point2D& point2d = image.points2d[element.point2d_idx];
    if (point2d.HasPoint3D()) {
      return std::nullopt;
    }
  }

  return reconstruction_->AddPoint3D(xyz, track);
}

bool ObservationManager::MergePoints(Point3DId target_point3d_id,
                                     Point3DId source_point3d_id,
                                     const Eigen::Vector3d& merged_xyz) {
  if (!HasReconstruction() || target_point3d_id == source_point3d_id ||
      !reconstruction_->ExistsPoint3D(target_point3d_id) ||
      !reconstruction_->ExistsPoint3D(source_point3d_id)) {
    return false;
  }

  const Point3D& target_point3d =
      reconstruction_->GetPoint3D(target_point3d_id);
  const Point3D& source_point3d =
      reconstruction_->GetPoint3D(source_point3d_id);
  if (TrackHasImageConflict(target_point3d.track, source_point3d.track)) {
    return false;
  }

  const Track source_track = source_point3d.track;
  if (!reconstruction_->DeletePoint3D(source_point3d_id)) {
    return false;
  }

  Point3D& merged_point3d = reconstruction_->GetPoint3D(target_point3d_id);
  merged_point3d.xyz = merged_xyz;
  for (const TrackElement& element : source_track.elements) {
    AddObservation(target_point3d_id, element.image_id, element.point2d_idx);
  }
  return true;
}

bool ObservationManager::HasReconstruction() const {
  return reconstruction_ != nullptr;
}

bool ObservationManager::HasValidObservation(ImageId image_id,
                                             Point2DIdx point2d_idx) const {
  if (!HasReconstruction() || !reconstruction_->ExistsImage(image_id) ||
      point2d_idx == kInvalidPoint2DIdx) {
    return false;
  }
  const Image& image = reconstruction_->GetImage(image_id);
  return point2d_idx < image.points2d.size();
}

bool ObservationManager::TrackHasImageConflict(const Track& track1,
                                               const Track& track2) const {
  for (const TrackElement& element1 : track1.elements) {
    for (const TrackElement& element2 : track2.elements) {
      if (element1.image_id == element2.image_id) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace colite
