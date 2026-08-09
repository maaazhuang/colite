#include "colite/core/camera.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>

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
       true},
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
       true},
      {CameraModelId::kOpenCV,
       "OPENCV",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7},
       false,
       true},
      {CameraModelId::kOpenCVFisheye,
       "OPENCV_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7},
       true,
       true},
      {CameraModelId::kFullOpenCV,
       "FULL_OPENCV",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11},
       false,
       true},
      {CameraModelId::kFOV,
       "FOV",
       Names{"fx", "fy", "cx", "cy", "omega"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4},
       false,
       true},
      {CameraModelId::kSimpleRadialFisheye,
       "SIMPLE_RADIAL_FISHEYE",
       Names{"f", "cx", "cy", "k1"},
       Indices{0},
       Indices{1, 2},
       Indices{3},
       true,
       true},
      {CameraModelId::kRadialFisheye,
       "RADIAL_FISHEYE",
       Names{"f", "cx", "cy", "k1", "k2"},
       Indices{0},
       Indices{1, 2},
       Indices{3, 4},
       true,
       true},
      {CameraModelId::kThinPrismFisheye,
       "THIN_PRISM_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11},
       true,
       true},
      {CameraModelId::kRadTanThinPrismFisheye,
       "RAD_TAN_THIN_PRISM_FISHEYE",
       Names{"fx", "fy", "cx", "cy", "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3"},
       Indices{0, 1},
       Indices{2, 3},
       Indices{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       true,
       true},
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

Eigen::Vector2d DistortRadial(const Eigen::Vector2d& normalized,
                              double k1,
                              double k2) {
  const double radius2 = normalized.squaredNorm();
  return normalized * (1.0 + k1 * radius2 + k2 * radius2 * radius2);
}

Eigen::Vector2d DistortOpenCV(const Eigen::Vector2d& normalized,
                              double k1,
                              double k2,
                              double p1,
                              double p2) {
  const double x = normalized.x();
  const double y = normalized.y();
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double radius2 = x2 + y2;
  const double radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2;
  return {x * radial + 2.0 * p1 * xy + p2 * (radius2 + 2.0 * x2),
          y * radial + p1 * (radius2 + 2.0 * y2) + 2.0 * p2 * xy};
}

Eigen::Vector2d DistortFullOpenCV(const Eigen::Vector2d& normalized,
                                  const double* extra_params) {
  const double u = normalized.x();
  const double v = normalized.y();
  const double u2 = u * u;
  const double uv = u * v;
  const double v2 = v * v;
  const double r2 = u2 + v2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double radial =
      (1.0 + extra_params[0] * r2 + extra_params[1] * r4 +
       extra_params[4] * r6) /
      (1.0 + extra_params[5] * r2 + extra_params[6] * r4 +
       extra_params[7] * r6);
  return {u * radial + 2.0 * extra_params[2] * uv +
              extra_params[3] * (r2 + 2.0 * u2),
          v * radial + 2.0 * extra_params[3] * uv +
              extra_params[2] * (r2 + 2.0 * v2)};
}

Eigen::Vector2d DistortFov(const Eigen::Vector2d& normalized, double omega) {
  constexpr double kEpsilon = 1e-4;
  const double radius2 = normalized.squaredNorm();
  const double omega2 = omega * omega;
  double factor = 1.0;
  if (omega2 < kEpsilon) {
    factor = (omega2 * radius2) / 3.0 - omega2 / 12.0 + 1.0;
  } else if (radius2 < kEpsilon) {
    const double tan_half_omega = std::tan(omega / 2.0);
    factor =
        (-2.0 * tan_half_omega *
         (4.0 * radius2 * tan_half_omega * tan_half_omega - 3.0)) /
        (3.0 * omega);
  } else {
    const double radius = std::sqrt(radius2);
    factor =
        std::atan(radius * 2.0 * std::tan(omega / 2.0)) / (radius * omega);
  }
  return normalized * factor;
}

Eigen::Vector2d FisheyeFromNormal(const Eigen::Vector2d& normalized) {
  const double radius = normalized.norm();
  if (radius <= std::numeric_limits<double>::epsilon()) {
    return normalized;
  }
  return normalized * (std::atan(radius) / radius);
}

Eigen::Vector2d DistortRadialFisheye(const Eigen::Vector2d& fisheye,
                                     const double* extra_params,
                                     std::size_t num_params) {
  const double theta2 = fisheye.squaredNorm();
  double theta_power = theta2;
  double radial = 0.0;
  for (std::size_t idx = 0; idx < num_params; ++idx) {
    radial += extra_params[idx] * theta_power;
    theta_power *= theta2;
  }
  return fisheye * (1.0 + radial);
}

Eigen::Vector2d DistortThinPrismFisheye(const Eigen::Vector2d& fisheye,
                                        const double* extra_params) {
  const double u = fisheye.x();
  const double v = fisheye.y();
  const double u2 = u * u;
  const double uv = u * v;
  const double v2 = v * v;
  const double r2 = u2 + v2;
  const double r4 = r2 * r2;
  const double r6 = r4 * r2;
  const double r8 = r6 * r2;
  const double radial = extra_params[0] * r2 + extra_params[1] * r4 +
                        extra_params[4] * r6 + extra_params[5] * r8;
  return {u + u * radial + 2.0 * extra_params[2] * uv +
              extra_params[3] * (r2 + 2.0 * u2) + extra_params[6] * r2,
          v + v * radial + 2.0 * extra_params[3] * uv +
              extra_params[2] * (r2 + 2.0 * v2) + extra_params[7] * r2};
}

Eigen::Vector2d DistortRadTanThinPrismFisheye(
    const Eigen::Vector2d& fisheye,
    const double* extra_params) {
  constexpr int kNumRadial = 6;
  const double theta2 = fisheye.squaredNorm();
  double theta_power = 1.0;
  double theta_radial = 1.0;
  for (int idx = 0; idx < kNumRadial; ++idx) {
    theta_power *= theta2;
    theta_radial += extra_params[idx] * theta_power;
  }

  const double x = theta_radial * fisheye.x();
  const double y = theta_radial * fisheye.y();
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double r2 = x2 + y2;
  const double r4 = r2 * r2;
  const double dx_tang = 2.0 * extra_params[7] * xy +
                         extra_params[6] * (r2 + 2.0 * x2);
  const double dy_tang = 2.0 * extra_params[6] * xy +
                         extra_params[7] * (r2 + 2.0 * y2);
  const double dx_tp = extra_params[8] * r2 + extra_params[9] * r4;
  const double dy_tp = extra_params[10] * r2 + extra_params[11] * r4;
  return {x + dx_tang + dx_tp, y + dy_tang + dy_tp};
}

bool ParamsAreNumericallyValid(CameraModelId model_id, const double* params) {
  if (params == nullptr || !ExistsCameraModel(model_id)) {
    return false;
  }
  const CameraModelSpec& spec = GetCameraModelSpec(model_id);
  for (std::size_t idx = 0; idx < spec.param_names.size(); ++idx) {
    if (!std::isfinite(params[idx])) {
      return false;
    }
  }
  for (const std::size_t idx : spec.focal_length_idxs) {
    if (params[idx] <= std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  if (model_id == CameraModelId::kFOV &&
      std::abs(params[spec.distortion_idxs.front()]) >=
          std::acos(-1.0) - 1e-8) {
    return false;
  }
  return true;
}

template <typename TransformFunc>
bool InvertTransform(const Eigen::Vector2d& target,
                     TransformFunc transform,
                     Eigen::Vector2d* inverse) {
  if (inverse == nullptr || !target.allFinite()) {
    return false;
  }

  Eigen::Vector2d estimate = target;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const Eigen::Vector2d transformed = transform(estimate);
    if (!transformed.allFinite()) {
      return false;
    }
    const Eigen::Vector2d residual = transformed - target;
    const double residual_norm = residual.norm();
    if (residual_norm <= 1e-12) {
      *inverse = estimate;
      return true;
    }

    Eigen::Matrix2d jacobian;
    for (int dimension = 0; dimension < 2; ++dimension) {
      const double step =
          1e-6 * std::max(1.0, std::abs(estimate[dimension]));
      Eigen::Vector2d positive = estimate;
      Eigen::Vector2d negative = estimate;
      positive[dimension] += step;
      negative[dimension] -= step;
      const Eigen::Vector2d transformed_positive = transform(positive);
      const Eigen::Vector2d transformed_negative = transform(negative);
      if (!transformed_positive.allFinite() ||
          !transformed_negative.allFinite()) {
        return false;
      }
      jacobian.col(dimension) =
          (transformed_positive - transformed_negative) / (2.0 * step);
    }
    if (!jacobian.allFinite() ||
        std::abs(jacobian.determinant()) <
            64.0 * std::numeric_limits<double>::epsilon()) {
      return false;
    }

    Eigen::Vector2d update = jacobian.partialPivLu().solve(residual);
    if (!update.allFinite()) {
      return false;
    }
    const double trust_radius =
        std::max(0.1, 0.1 * std::max(1.0, estimate.norm()));
    if (update.norm() > trust_radius) {
      update *= trust_radius / update.norm();
    }

    bool accepted = false;
    double step_scale = 1.0;
    for (int line_search = 0; line_search < 12; ++line_search) {
      const Eigen::Vector2d candidate = estimate - step_scale * update;
      const Eigen::Vector2d candidate_value = transform(candidate);
      if (candidate_value.allFinite() &&
          (candidate_value - target).norm() < residual_norm) {
        estimate = candidate;
        accepted = true;
        break;
      }
      step_scale *= 0.5;
    }
    if (!accepted) {
      return false;
    }
    if ((step_scale * update).norm() <= 1e-12) {
      const Eigen::Vector2d final_residual = transform(estimate) - target;
      if (final_residual.allFinite() && final_residual.norm() <= 1e-9) {
        *inverse = estimate;
        return true;
      }
      return false;
    }
  }

  const Eigen::Vector2d final_residual = transform(estimate) - target;
  if (!final_residual.allFinite() || final_residual.norm() > 1e-9) {
    return false;
  }
  *inverse = estimate;
  return true;
}

Eigen::Vector2d ImageToDistortedNormalized(CameraModelId model_id,
                                           const double* params,
                                           const Eigen::Vector2d& pixel) {
  const CameraModelSpec& spec = GetCameraModelSpec(model_id);
  const double fx = params[spec.focal_length_idxs[0]];
  const double fy =
      params[spec.focal_length_idxs.size() == 1
                 ? spec.focal_length_idxs[0]
                 : spec.focal_length_idxs[1]];
  return {(pixel.x() - params[spec.principal_point_idxs[0]]) / fx,
          (pixel.y() - params[spec.principal_point_idxs[1]]) / fy};
}

bool FisheyeToNormal(const Eigen::Vector2d& fisheye,
                     Eigen::Vector2d* normalized) {
  if (normalized == nullptr || !fisheye.allFinite()) {
    return false;
  }
  const double theta = fisheye.norm();
  if (theta <= std::numeric_limits<double>::epsilon()) {
    *normalized = fisheye;
    return true;
  }
  if (theta >= std::acos(-1.0) / 2.0 - 1e-8) {
    return false;
  }
  *normalized = fisheye * (std::tan(theta) / theta);
  return normalized->allFinite();
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
  if (!ExistsCameraModel(model_id) || width == 0 || height == 0 ||
      params.size() != GetCameraModelSpec(model_id).param_names.size()) {
    return false;
  }
  return ParamsAreNumericallyValid(model_id, params.data());
}

bool Camera::HasBogusParams(const double min_focal_length_ratio,
                            const double max_focal_length_ratio,
                            const double max_extra_param) const {
  if (!VerifyParams() || !std::isfinite(min_focal_length_ratio) ||
      !std::isfinite(max_focal_length_ratio) ||
      !std::isfinite(max_extra_param) || min_focal_length_ratio <= 0.0 ||
      max_focal_length_ratio < min_focal_length_ratio ||
      max_extra_param < 0.0) {
    return true;
  }
  const double image_extent = static_cast<double>(std::max(width, height));
  const CameraModelSpec& spec = GetCameraModelSpec(model_id);
  for (const std::size_t idx : spec.focal_length_idxs) {
    const double ratio = params[idx] / image_extent;
    if (ratio < min_focal_length_ratio || ratio > max_focal_length_ratio) {
      return true;
    }
  }
  for (const std::size_t idx : spec.distortion_idxs) {
    if (std::abs(params[idx]) > max_extra_param) {
      return true;
    }
  }
  return false;
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

bool CameraModelUnprojectionImplemented(CameraModelId model_id) {
  return CameraModelProjectionImplemented(model_id);
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

std::vector<std::size_t> CameraParameterIndicesForMask(
    CameraModelId model_id,
    const CameraParameterRefinementMask& mask) {
  const CameraModelSpec& spec = GetCameraModelSpec(model_id);
  std::set<std::size_t> index_set;
  for (const std::size_t idx :
       CameraParameterIndicesForGroups(model_id, mask.groups)) {
    index_set.insert(idx);
  }

  for (const std::string& name : mask.parameter_names) {
    const auto it =
        std::find(spec.param_names.begin(), spec.param_names.end(), name);
    if (it == spec.param_names.end()) {
      throw std::invalid_argument("Unknown camera parameter name: " + name);
    }
    index_set.insert(static_cast<std::size_t>(
        std::distance(spec.param_names.begin(), it)));
  }

  return {index_set.begin(), index_set.end()};
}

Eigen::Vector2d ApplySimpleRadialDistortion(const Eigen::Vector2d& normalized,
                                            double k1) {
  const double radius2 = normalized.squaredNorm();
  return normalized * (1.0 + k1 * radius2);
}

bool ProjectPointWithCameraParams(CameraModelId model_id,
                                  const double* params,
                                  const Eigen::Vector3d& point_camera,
                                  Eigen::Vector2d* projected) {
  if (projected == nullptr || !point_camera.allFinite() ||
      point_camera.z() <= std::numeric_limits<double>::epsilon() ||
      !CameraModelProjectionImplemented(model_id) ||
      !ParamsAreNumericallyValid(model_id, params)) {
    return false;
  }

  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();
  const Eigen::Vector2d normalized(x, y);

  if (model_id == CameraModelId::kSimplePinhole) {
    *projected = {params[0] * x + params[1], params[0] * y + params[2]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kPinhole) {
    *projected = {params[0] * x + params[2], params[1] * y + params[3]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kSimpleRadial) {
    const Eigen::Vector2d distorted =
        ApplySimpleRadialDistortion(normalized, params[3]);
    *projected = {params[0] * distorted.x() + params[1],
                  params[0] * distorted.y() + params[2]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kRadial) {
    const Eigen::Vector2d distorted = DistortRadial(normalized, params[3],
                                                   params[4]);
    *projected = {params[0] * distorted.x() + params[1],
                  params[0] * distorted.y() + params[2]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kOpenCV) {
    const Eigen::Vector2d distorted =
        DistortOpenCV(normalized, params[4], params[5], params[6], params[7]);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kFullOpenCV) {
    const Eigen::Vector2d distorted = DistortFullOpenCV(normalized, &params[4]);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kFOV) {
    const Eigen::Vector2d distorted = DistortFov(normalized, params[4]);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }

  const Eigen::Vector2d fisheye = FisheyeFromNormal(normalized);
  Eigen::Vector2d distorted = fisheye;
  if (model_id == CameraModelId::kSimpleRadialFisheye) {
    distorted = DistortRadialFisheye(fisheye, &params[3], 1);
    *projected = {params[0] * distorted.x() + params[1],
                  params[0] * distorted.y() + params[2]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kRadialFisheye) {
    distorted = DistortRadialFisheye(fisheye, &params[3], 2);
    *projected = {params[0] * distorted.x() + params[1],
                  params[0] * distorted.y() + params[2]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kOpenCVFisheye) {
    distorted = DistortRadialFisheye(fisheye, &params[4], 4);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kThinPrismFisheye) {
    distorted = DistortThinPrismFisheye(fisheye, &params[4]);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }
  if (model_id == CameraModelId::kRadTanThinPrismFisheye) {
    distorted = DistortRadTanThinPrismFisheye(fisheye, &params[4]);
    *projected = {params[0] * distorted.x() + params[2],
                  params[1] * distorted.y() + params[3]};
    return projected->allFinite();
  }
  return false;
}

bool UnprojectPixelWithCameraParams(CameraModelId model_id,
                                    const double* params,
                                    const Eigen::Vector2d& pixel,
                                    const double depth,
                                    Eigen::Vector3d* point_camera) {
  if (point_camera == nullptr || !pixel.allFinite() ||
      !std::isfinite(depth) || depth <= 0.0 ||
      !CameraModelUnprojectionImplemented(model_id) ||
      !ParamsAreNumericallyValid(model_id, params)) {
    return false;
  }

  const Eigen::Vector2d distorted =
      ImageToDistortedNormalized(model_id, params, pixel);
  Eigen::Vector2d model_plane = distorted;
  bool success = true;
  if (model_id == CameraModelId::kSimpleRadial) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return ApplySimpleRadialDistortion(point, params[3]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kRadial) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortRadial(point, params[3], params[4]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kOpenCV) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortOpenCV(point, params[4], params[5], params[6],
                               params[7]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kFullOpenCV) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortFullOpenCV(point, &params[4]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kFOV) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortFov(point, params[4]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kSimpleRadialFisheye) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortRadialFisheye(point, &params[3], 1);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kRadialFisheye) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortRadialFisheye(point, &params[3], 2);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kOpenCVFisheye) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortRadialFisheye(point, &params[4], 4);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kThinPrismFisheye) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortThinPrismFisheye(point, &params[4]);
        },
        &model_plane);
  } else if (model_id == CameraModelId::kRadTanThinPrismFisheye) {
    success = InvertTransform(
        distorted,
        [&](const Eigen::Vector2d& point) {
          return DistortRadTanThinPrismFisheye(point, &params[4]);
        },
        &model_plane);
  }
  if (!success || !model_plane.allFinite()) {
    return false;
  }

  Eigen::Vector2d normalized = model_plane;
  if (GetCameraModelSpec(model_id).is_fisheye &&
      !FisheyeToNormal(model_plane, &normalized)) {
    return false;
  }
  *point_camera =
      Eigen::Vector3d(normalized.x() * depth, normalized.y() * depth, depth);
  return point_camera->allFinite();
}

Eigen::Vector2d ProjectPoint(const Camera& camera,
                             const Eigen::Vector3d& point_camera) {
  RequireValidParams(camera);
  RequirePositiveDepth(point_camera);

  Eigen::Vector2d projected;
  if (ProjectPointWithCameraParams(camera.model_id, camera.params.data(),
                                   point_camera, &projected)) {
    return projected;
  }

  throw std::domain_error("Projection is not implemented for this model.");
}

Eigen::Vector3d UnprojectPixel(const Camera& camera,
                               const Eigen::Vector2d& pixel,
                               double depth) {
  RequireValidParams(camera);
  if (!std::isfinite(depth) || depth <= 0.0) {
    throw std::invalid_argument("Depth must be positive.");
  }
  Eigen::Vector3d point_camera;
  if (!UnprojectPixelWithCameraParams(camera.model_id, camera.params.data(),
                                      pixel, depth, &point_camera)) {
    throw std::domain_error(
        "Pixel cannot be unprojected for this camera model and parameters.");
  }
  return point_camera;
}

}  // namespace colite
