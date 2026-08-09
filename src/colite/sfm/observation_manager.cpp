#include "colite/sfm/observation_manager.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include "colite/core/geometry.h"

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

bool TrackHasImage(const Track& track, ImageId image_id) {
  return std::any_of(track.elements.begin(), track.elements.end(),
                     [image_id](const TrackElement& element) {
                       return element.image_id == image_id;
                     });
}

double ObservationReprojectionError(const Reconstruction& reconstruction,
                                    const TrackElement& element,
                                    const Eigen::Vector3d& xyz) {
  if (!reconstruction.ExistsImage(element.image_id)) {
    return std::numeric_limits<double>::infinity();
  }
  const Image& image = reconstruction.GetImage(element.image_id);
  if (!image.HasPose() ||
      !reconstruction.ExistsCamera(image.camera_id) ||
      element.point2d_idx >= image.points2d.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return ReprojectionError(
             reconstruction.GetCamera(image.camera_id),
             *image.cam_from_world, xyz,
             image.points2d[element.point2d_idx].xy)
      .value_or(std::numeric_limits<double>::infinity());
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
  if (TrackHasImage(point3d.track, image_id)) {
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

  const Track target_track =
      reconstruction_->GetPoint3D(target_point3d_id).track;
  const Track source_track =
      reconstruction_->GetPoint3D(source_point3d_id).track;

  struct Candidate {
    TrackElement element;
    bool from_target = false;
    double error = std::numeric_limits<double>::infinity();
  };
  std::map<ImageId, Candidate> selected;
  for (const auto& [track, from_target] :
       {std::pair<const Track*, bool>{&target_track, true},
        std::pair<const Track*, bool>{&source_track, false}}) {
    for (const TrackElement& element : track->elements) {
      Candidate candidate{
          element, from_target,
          ObservationReprojectionError(*reconstruction_, element, merged_xyz)};
      const auto selected_it = selected.find(element.image_id);
      if (selected_it == selected.end() ||
          candidate.error < selected_it->second.error ||
          (candidate.error == selected_it->second.error &&
           candidate.from_target && !selected_it->second.from_target) ||
          (candidate.error == selected_it->second.error &&
           candidate.from_target == selected_it->second.from_target &&
           candidate.element.point2d_idx <
               selected_it->second.element.point2d_idx)) {
        selected[element.image_id] = candidate;
      }
    }
  }

  for (const TrackElement& element : target_track.elements) {
    const auto selected_it = selected.find(element.image_id);
    if (selected_it == selected.end() ||
        selected_it->second.element.point2d_idx != element.point2d_idx ||
        !selected_it->second.from_target) {
      reconstruction_->DeleteObservation(target_point3d_id, element.image_id,
                                         element.point2d_idx);
    }
  }
  if (!reconstruction_->DeletePoint3D(source_point3d_id)) {
    return false;
  }

  Point3D& merged_point3d = reconstruction_->GetPoint3D(target_point3d_id);
  merged_point3d.xyz = merged_xyz;
  for (const auto& [image_id, candidate] : selected) {
    (void)image_id;
    if (!candidate.from_target &&
        !AddObservation(target_point3d_id, candidate.element.image_id,
                        candidate.element.point2d_idx)) {
      return false;
    }
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

}  // namespace colite
