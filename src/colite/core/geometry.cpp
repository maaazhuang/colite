#include "colite/core/geometry.h"

#include <algorithm>
#include <cmath>

#include <Eigen/SVD>

#include "colite/core/reconstruction.h"

namespace colite {
namespace {

constexpr double kMinTriangulationSingularGap = 1e-12;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

Eigen::Matrix<double, 3, 4> ProjectionMatrixFromPose(
    const Pose3& cam_from_world) {
  Eigen::Matrix<double, 3, 4> projection;
  projection.leftCols<3>() = cam_from_world.q_cw.toRotationMatrix();
  projection.rightCols<1>() = cam_from_world.t_cw;
  return projection;
}

Eigen::Vector2d NormalizedPointFromPixel(const Camera& camera,
                                         const Eigen::Vector2d& pixel) {
  const Eigen::Vector3d ray = UnprojectPixel(camera, pixel, 1.0);
  return {ray.x() / ray.z(), ray.y() / ray.z()};
}

}  // namespace

Eigen::Vector3d PixelToBearing(const Camera& camera,
                               const Eigen::Vector2d& pixel) {
  return UnprojectPixel(camera, pixel, 1.0).normalized();
}

std::optional<Eigen::Vector2d> ProjectWorldPoint(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world) {
  const Eigen::Vector3d point_camera =
      cam_from_world.TransformPoint(point_world);
  if (point_camera.z() <= 0.0) {
    return std::nullopt;
  }
  return ProjectPoint(camera, point_camera);
}

std::optional<double> ReprojectionError(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world,
    const Eigen::Vector2d& observed_pixel) {
  const std::optional<Eigen::Vector2d> projected =
      ProjectWorldPoint(camera, cam_from_world, point_world);
  if (!projected) {
    return std::nullopt;
  }
  return (*projected - observed_pixel).norm();
}

std::optional<double> Point3DTrackReprojectionError(
    const Reconstruction& reconstruction,
    const Point3D& point3d) {
  double squared_error_sum = 0.0;
  std::size_t num_observations = 0;
  for (const TrackElement& element : point3d.track.elements) {
    if (!reconstruction.ExistsImage(element.image_id)) {
      continue;
    }
    const Image& image = reconstruction.GetImage(element.image_id);
    if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
        element.point2d_idx >= image.points2d.size()) {
      continue;
    }
    const std::optional<double> error = ReprojectionError(
        reconstruction.GetCamera(image.camera_id), *image.cam_from_world,
        point3d.xyz, image.points2d[element.point2d_idx].xy);
    if (!error || !std::isfinite(*error)) {
      continue;
    }
    squared_error_sum += *error * *error;
    ++num_observations;
  }
  if (num_observations == 0) {
    return std::nullopt;
  }
  return std::sqrt(squared_error_sum /
                   static_cast<double>(num_observations));
}

std::optional<Eigen::Vector3d> TriangulatePoint(
    const Camera& camera1,
    const Pose3& cam1_from_world,
    const Eigen::Vector2d& pixel1,
    const Camera& camera2,
    const Pose3& cam2_from_world,
    const Eigen::Vector2d& pixel2) {
  const Eigen::Vector2d x1 = NormalizedPointFromPixel(camera1, pixel1);
  const Eigen::Vector2d x2 = NormalizedPointFromPixel(camera2, pixel2);
  const Eigen::Matrix<double, 3, 4> p1 =
      ProjectionMatrixFromPose(cam1_from_world);
  const Eigen::Matrix<double, 3, 4> p2 =
      ProjectionMatrixFromPose(cam2_from_world);

  Eigen::Matrix4d a;
  a.row(0) = x1.x() * p1.row(2) - p1.row(0);
  a.row(1) = x1.y() * p1.row(2) - p1.row(1);
  a.row(2) = x2.x() * p2.row(2) - p2.row(0);
  a.row(3) = x2.y() * p2.row(2) - p2.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(a, Eigen::ComputeFullV);
  const auto singular_values = svd.singularValues();
  if (singular_values(2) - singular_values(3) < kMinTriangulationSingularGap) {
    return std::nullopt;
  }

  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) < 1e-12) {
    return std::nullopt;
  }
  return homogeneous.hnormalized();
}

double TriangulationAngleDegrees(const Pose3& cam1_from_world,
                                 const Eigen::Vector3d& bearing1,
                                 const Pose3& cam2_from_world,
                                 const Eigen::Vector3d& bearing2) {
  const Eigen::Vector3d ray1_world =
      (cam1_from_world.q_cw.conjugate() * bearing1).normalized();
  const Eigen::Vector3d ray2_world =
      (cam2_from_world.q_cw.conjugate() * bearing2).normalized();
  const double cos_angle =
      std::clamp(ray1_world.dot(ray2_world), -1.0, 1.0);
  return std::acos(cos_angle) * kRadiansToDegrees;
}

}  // namespace colite
