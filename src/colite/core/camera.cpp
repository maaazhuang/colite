#include "colite/core/camera.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace colite {
namespace {

using Names = std::vector<std::string_view>;
using Indices = std::vector<std::size_t>;

const std::vector<CameraModelSpec>& CameraModelSpecs() {
  static const std::vector<CameraModelSpec> specs = {
      {CameraModelId::kSimplePinhole,
       "SIMPLE_PINHOLE",
       Names{"f", "cx", "cy"},
       Indices{0},
       Indices{1, 2},
       Indices{},
       false,
       false},
      {CameraModelId::kPinhole,
       "PINHOLE",
       Names{"fx", "fy", "cx", "cy"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{},
       false,
       true},
      {CameraModelId::kSimpleRadial,
       "SIMPLE_RADIAL",
       Names{"f", "cx", "cy", "k1"},
       Indices{0},
       Indices{1, 2},
       Indices{3},
       false,
       true},
      {CameraModelId::kRadial,
       "RADIAL",
       Names{"f", "cx", "cy", "k1", "k2"},
       Indices{0},
       Indices{1, 2},
       Indices{3, 4},
       false,
       false},
      {CameraModelId::kOpenCV,
       "OPENCV",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7},
       false,
       false},
      {CameraModelId::kOpenCVFisheye,
       "OPENCV_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7},
       true,
       false},
      {CameraModelId::kFullOpenCV,
       "FULL_OPENCV",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11},
       false,
       false},
      {CameraModelId::kFOV,
       "FOV",
       Names{"fx", "fy", "cx", "cy", "omega"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4},
       false,
       false},
      {CameraModelId::kSimpleRadialFisheye,
       "SIMPLE_RADIAL_FISHEYE",
       Names{"f", "cx", "cy", "k1"},
       Indices{0},
       Indices{1, 2},
       Indices{3},
       true,
       false},
      {CameraModelId::kRadialFisheye,
       "RADIAL_FISHEYE",
       Names{"f", "cx", "cy", "k1", "k2"},
       Indices{0},
       Indices{1, 2},
       Indices{3, 4},
       true,
       false},
      {CameraModelId::kThinPrismFisheye,
       "THIN_PRISM_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11},
       true,
       false},
      {CameraModelId::kRadTanThinPrismFisheye,
       "RAD_TAN_THIN_PRISM_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       true,
       false},
  };
  return specs;
}

void RequireValidParams(const Camera& camera) {
  if (!camera.VerifyParams()) {
    throw std::invalid_argument("Camera parameters do not match the model.");
  }
}

void RequirePositiveDepth(const Eigen::Vector3d& point_camera) {
  if (point_camera.z() <= 0.0) {
    throw std::invalid_argument("Point depth must be positive.");
  }
}

Eigen::Vector2d RemoveSimpleRadialDistortion(Eigen::Vector2d distorted,
                                             double k1) {
  // 用固定次数迭代近似去畸变。第一版足够用于反投影和测试，后续可改成更稳健的求解。
  Eigen::Vector2d undistorted = distorted;
  for (int i = 0; i < 8; ++i) {
    const double radius2 = undistorted.squaredNorm();
    const double scale = 1.0 + k1 * radius2;
    if (std::abs(scale) < 1e-12) {
      break;
    }
    undistorted = distorted / scale;
  }
  return undistorted;
}

}  // namespace

CameraParameterGroup operator|(CameraParameterGroup lhs,
                               CameraParameterGroup rhs) {
  return static_cast<CameraParameterGroup>(
      static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

CameraParameterGroup operator&(CameraParameterGroup lhs,
                               CameraParameterGroup rhs) {
  return static_cast<CameraParameterGroup>(
      static_cast<std::uint8_t>(lhs) & static_cast<std::uint8_t>(rhs));
}

bool HasAny(CameraParameterGroup groups, CameraParameterGroup flag) {
  return static_cast<std::uint8_t>(groups & flag) != 0;
}

bool Camera::HasValidId() const {
  return id != kInvalidCameraId;
}

bool Camera::VerifyParams() const {
  if (!ExistsCameraModel(model_id)) {
    return false;
  }
  return params.size() == GetCameraModelSpec(model_id).param_names.size();
}

double Camera::MeanFocalLength() const {
  RequireValidParams(*this);
  const auto& idxs = GetCameraModelSpec(model_id).focal_length_idxs;
  double focal = 0.0;
  for (const std::size_t idx : idxs) {
    focal += params.at(idx);
  }
  return focal / static_cast<double>(idxs.size());
}

Eigen::Matrix3d Camera::CalibrationMatrix() const {
  RequireValidParams(*this);
  const auto& spec = GetCameraModelSpec(model_id);
  const double fx = params.at(spec.focal_length_idxs[0]);
  const double fy = params.at(spec.focal_length_idxs.size() == 1
                                  ? spec.focal_length_idxs[0]
                                  : spec.focal_length_idxs[1]);
  const double cx = params.at(spec.principal_point_idxs[0]);
  const double cy = params.at(spec.principal_point_idxs[1]);

  Eigen::Matrix3d k = Eigen::Matrix3d::Identity();
  k(0, 0) = fx;
  k(1, 1) = fy;
  k(0, 2) = cx;
  k(1, 2) = cy;
  return k;
}

const CameraModelSpec& GetCameraModelSpec(CameraModelId model_id) {
  const auto& specs = CameraModelSpecs();
  const auto it = std::find_if(
      specs.begin(), specs.end(), [model_id](const CameraModelSpec& spec) {
        return spec.id == model_id;
      });
  if (it == specs.end()) {
    throw std::invalid_argument("Unknown camera model.");
  }
  return *it;
}

CameraModelId CameraModelNameToId(std::string_view name) {
  for (const auto& spec : CameraModelSpecs()) {
    if (spec.name == name) {
      return spec.id;
    }
  }
  return CameraModelId::kInvalid;
}

std::string_view CameraModelIdToName(CameraModelId model_id) {
  if (!ExistsCameraModel(model_id)) {
    return "";
  }
  return GetCameraModelSpec(model_id).name;
}

bool ExistsCameraModel(CameraModelId model_id) {
  const auto& specs = CameraModelSpecs();
  return std::any_of(
      specs.begin(), specs.end(), [model_id](const CameraModelSpec& spec) {
        return spec.id == model_id;
      });
}

bool CameraModelProjectionImplemented(CameraModelId model_id) {
  return ExistsCameraModel(model_id) &&
         GetCameraModelSpec(model_id).projection_implemented;
}

std::vector<double> InitializeCameraParams(CameraModelId model_id,
                                           double focal_length,
                                           std::uint32_t width,
                                           std::uint32_t height) {
  const auto& spec = GetCameraModelSpec(model_id);
  std::vector<double> params(spec.param_names.size(), 0.0);
  for (const std::size_t idx : spec.focal_length_idxs) {
    params[idx] = focal_length;
  }
  params[spec.principal_point_idxs[0]] = static_cast<double>(width) / 2.0;
  params[spec.principal_point_idxs[1]] = static_cast<double>(height) / 2.0;
  if (model_id == CameraModelId::kFOV) {
    params[spec.distortion_idxs[0]] = 1e-2;
  }
  return params;
}

std::vector<std::size_t> CameraParameterIndicesForGroups(
    CameraModelId model_id,
    CameraParameterGroup groups) {
  const auto& spec = GetCameraModelSpec(model_id);
  std::vector<std::size_t> indices;
  if (HasAny(groups, CameraParameterGroup::kFocalLength)) {
    indices.insert(indices.end(), spec.focal_length_idxs.begin(),
                   spec.focal_length_idxs.end());
  }
  if (HasAny(groups, CameraParameterGroup::kPrincipalPoint)) {
    indices.insert(indices.end(), spec.principal_point_idxs.begin(),
                   spec.principal_point_idxs.end());
  }
  if (HasAny(groups, CameraParameterGroup::kDistortion)) {
    indices.insert(indices.end(), spec.distortion_idxs.begin(),
                   spec.distortion_idxs.end());
  }
  return indices;
}

Eigen::Vector2d ApplySimpleRadialDistortion(const Eigen::Vector2d& normalized,
                                            double k1) {
  const double radius2 = normalized.squaredNorm();
  return normalized * (1.0 + k1 * radius2);
}

Eigen::Vector2d ProjectPoint(const Camera& camera,
                             const Eigen::Vector3d& point_camera) {
  RequireValidParams(camera);
  RequirePositiveDepth(point_camera);

  if (camera.model_id == CameraModelId::kPinhole) {
    const double fx = camera.params[0];
    const double fy = camera.params[1];
    const double cx = camera.params[2];
    const double cy = camera.params[3];
    return {fx * point_camera.x() / point_camera.z() + cx,
            fy * point_camera.y() / point_camera.z() + cy};
  }

  if (camera.model_id == CameraModelId::kSimpleRadial) {
    const double f = camera.params[0];
    const double cx = camera.params[1];
    const double cy = camera.params[2];
    const double k1 = camera.params[3];
    const Eigen::Vector2d normalized(point_camera.x() / point_camera.z(),
                                     point_camera.y() / point_camera.z());
    const Eigen::Vector2d distorted =
        ApplySimpleRadialDistortion(normalized, k1);
    return {f * distorted.x() + cx, f * distorted.y() + cy};
  }

  throw std::domain_error("Projection is not implemented for this model.");
}

Eigen::Vector3d UnprojectPixel(const Camera& camera,
                               const Eigen::Vector2d& pixel,
                               double depth) {
  RequireValidParams(camera);
  if (depth <= 0.0) {
    throw std::invalid_argument("Depth must be positive.");
  }

  if (camera.model_id == CameraModelId::kPinhole) {
    const double fx = camera.params[0];
    const double fy = camera.params[1];
    const double cx = camera.params[2];
    const double cy = camera.params[3];
    return {((pixel.x() - cx) / fx) * depth,
            ((pixel.y() - cy) / fy) * depth,
            depth};
  }

  if (camera.model_id == CameraModelId::kSimpleRadial) {
    const double f = camera.params[0];
    const double cx = camera.params[1];
    const double cy = camera.params[2];
    const double k1 = camera.params[3];
    const Eigen::Vector2d distorted((pixel.x() - cx) / f,
                                    (pixel.y() - cy) / f);
    const Eigen::Vector2d undistorted =
        RemoveSimpleRadialDistortion(distorted, k1);
    return {undistorted.x() * depth, undistorted.y() * depth, depth};
  }

  throw std::domain_error("Unprojection is not implemented for this model.");
}

}  // namespace colite

