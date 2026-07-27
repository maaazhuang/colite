#include "colite/sfm/observation_filter.h"

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

#include "colite/core/geometry.h"

namespace colite {
namespace {

struct ObservationRef {
  Point3DId point3d_id = kInvalidPoint3DId;
  ImageId image_id = kInvalidImageId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;
};

bool HasUsableObservation(const Reconstruction& reconstruction,
                          Point3DId point3d_id,
                          const TrackElement& element) {
  if (!reconstruction.ExistsImage(element.image_id)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(element.image_id);
  if (!image.registered || !image.HasPose() ||
      !reconstruction.ExistsCamera(image.camera_id) ||
      element.point2d_idx >= image.points2d.size()) {
    return false;
  }
  return image.points2d[element.point2d_idx].point3d_id == point3d_id;
}

bool HasLargeReprojectionError(const Reconstruction& reconstruction,
                               const Point3D& point3d,
                               const TrackElement& element,
                               double max_error_px) {
  const Image& image = reconstruction.GetImage(element.image_id);
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const Point2D& point2d = image.points2d[element.point2d_idx];
  const std::optional<double> error =
      ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                        point2d.xy);
  return !error.has_value() || *error > max_error_px;
}

double MaxTriangulationAngleDegrees(const Reconstruction& reconstruction,
                                    const Point3D& point3d) {
  double max_angle = 0.0;
  for (std::size_t i = 0; i < point3d.track.elements.size(); ++i) {
    const TrackElement& element1 = point3d.track.elements[i];
    if (!HasUsableObservation(reconstruction, point3d.id, element1)) {
      continue;
    }
    const Image& image1 = reconstruction.GetImage(element1.image_id);
    const Camera& camera1 = reconstruction.GetCamera(image1.camera_id);
    const Eigen::Vector3d bearing1 =
        PixelToBearing(camera1, image1.points2d[element1.point2d_idx].xy);

    for (std::size_t j = i + 1; j < point3d.track.elements.size(); ++j) {
      const TrackElement& element2 = point3d.track.elements[j];
      if (!HasUsableObservation(reconstruction, point3d.id, element2)) {
        continue;
      }
      const Image& image2 = reconstruction.GetImage(element2.image_id);
      const Camera& camera2 = reconstruction.GetCamera(image2.camera_id);
      const Eigen::Vector3d bearing2 =
          PixelToBearing(camera2, image2.points2d[element2.point2d_idx].xy);
      max_angle = std::max(max_angle,
                           TriangulationAngleDegrees(*image1.cam_from_world,
                                                     bearing1,
                                                     *image2.cam_from_world,
                                                     bearing2));
    }
  }
  return max_angle;
}

std::vector<Point3DId> CollectPointIds(const Reconstruction& reconstruction) {
  std::vector<Point3DId> point_ids;
  point_ids.reserve(reconstruction.Points3D().size());
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d;
    point_ids.push_back(point3d_id);
  }
  return point_ids;
}

std::vector<Point3DId> CollectPointIdsInImages(
    const Reconstruction& reconstruction,
    const std::set<ImageId>& image_ids) {
  std::set<Point3DId> unique_point_ids;
  for (const ImageId image_id : image_ids) {
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    const Image& image = reconstruction.GetImage(image_id);
    for (const Point2D& point2d : image.points2d) {
      if (point2d.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2d.point3d_id)) {
        unique_point_ids.insert(point2d.point3d_id);
      }
    }
  }
  return {unique_point_ids.begin(), unique_point_ids.end()};
}

ObservationFilterSummary FilterPointIds(
    const ObservationFilterOptions& options,
    const std::vector<Point3DId>& point_ids,
    Reconstruction* reconstruction) {
  ObservationFilterSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  std::vector<ObservationRef> observations_to_delete;
  for (const Point3DId point3d_id : point_ids) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      if (!HasUsableObservation(*reconstruction, point3d_id, element) ||
          HasLargeReprojectionError(*reconstruction, point3d, element,
                                    options.max_reprojection_error_px)) {
        observations_to_delete.push_back(
            {point3d_id, element.image_id, element.point2d_idx});
      }
    }
  }

  for (const ObservationRef& observation : observations_to_delete) {
    if (reconstruction->DeleteObservation(observation.point3d_id,
                                          observation.image_id,
                                          observation.point2d_idx)) {
      ++summary.num_removed_observations;
    }
  }

  std::vector<Point3DId> points_to_delete;
  for (const Point3DId point3d_id : point_ids) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    if (point3d.track.Length() < options.min_track_length ||
        MaxTriangulationAngleDegrees(*reconstruction, point3d) <
            options.min_triangulation_angle_deg) {
      points_to_delete.push_back(point3d_id);
    }
  }

  for (const Point3DId point3d_id : points_to_delete) {
    if (reconstruction->DeletePoint3D(point3d_id)) {
      ++summary.num_removed_points;
    }
  }

  summary.success = true;
  summary.message = "Observation filtering finished";
  return summary;
}

}  // namespace

ObservationFilterSummary FilterObservationsAndPoints(
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    ObservationFilterSummary summary;
    summary.message = "Reconstruction is null";
    return summary;
  }
  return FilterPointIds(options, CollectPointIds(*reconstruction),
                        reconstruction);
}

ObservationFilterSummary FilterObservationsAndPointsInImages(
    const ObservationFilterOptions& options,
    const std::set<ImageId>& image_ids,
    Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    ObservationFilterSummary summary;
    summary.message = "Reconstruction is null";
    return summary;
  }
  return FilterPointIds(options,
                        CollectPointIdsInImages(*reconstruction, image_ids),
                        reconstruction);
}

}  // namespace colite
