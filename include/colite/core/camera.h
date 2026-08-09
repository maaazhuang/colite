#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "colite/core/types.h"

namespace colite {

// 相机模型 ID。模型族按 COLMAP 组织；当前已实现常用透视模型的投影：
// SIMPLE_PINHOLE、PINHOLE、SIMPLE_RADIAL、RADIAL、OPENCV。
enum class CameraModelId {
  kInvalid = -1,
  kSimplePinhole = 0,          // f, cx, cy
  kPinhole = 1,                // fx, fy, cx, cy
  kSimpleRadial = 2,           // f, cx, cy, k1
  kRadial = 3,                 // f, cx, cy, k1, k2
  kOpenCV = 4,                 // fx, fy, cx, cy, k1, k2, p1, p2
  kOpenCVFisheye = 5,          // fx, fy, cx, cy, k1, k2, k3, k4
  kFullOpenCV = 6,             // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
  kFOV = 7,                    // fx, fy, cx, cy, omega
  kSimpleRadialFisheye = 8,    // f, cx, cy, k1
  kRadialFisheye = 9,          // f, cx, cy, k1, k2
  kThinPrismFisheye = 10,      // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
  kRadTanThinPrismFisheye = 11 // fx, fy, cx, cy, k0..k5, p0, p1, s0..s3
};

// 相机参数组，用于第一版 BA 参数选择逻辑。
// 例如：kFocalLength | kDistortion 表示优化焦距和畸变，不优化主点。
enum class CameraParameterGroup : std::uint8_t {
  kNone = 0,
  kFocalLength = 1 << 0,
  kPrincipalPoint = 1 << 1,
  kDistortion = 1 << 2,
  kAll = (1 << 0) | (1 << 1) | (1 << 2),
};

CameraParameterGroup operator|(CameraParameterGroup lhs,
                               CameraParameterGroup rhs);
CameraParameterGroup operator&(CameraParameterGroup lhs,
                               CameraParameterGroup rhs);
bool HasAny(CameraParameterGroup groups, CameraParameterGroup flag);

// 相机内参优化掩码。groups 用于 COLMAP 风格的焦距/主点/额外参数组，
// parameter_names 用于按参数名细化，例如 {"fx", "k1", "p2"}。
struct CameraParameterRefinementMask {
  CameraParameterGroup groups = CameraParameterGroup::kNone;
  std::set<std::string> parameter_names;
};

// 相机模型规格。所有参数通过 param_names 的顺序存入 Camera::params。
struct CameraModelSpec {
  CameraModelId id = CameraModelId::kInvalid;
  std::string_view name;
  std::vector<std::string_view> param_names;
  std::vector<std::size_t> focal_length_idxs;
  std::vector<std::size_t> principal_point_idxs;
  std::vector<std::size_t> distortion_idxs;
  bool is_fisheye = false;
  bool projection_implemented = false;
};

// 相机内参实体。多个 Image 可以共享同一个 Camera。
struct Camera {
  CameraId id = kInvalidCameraId;
  CameraModelId model_id = CameraModelId::kInvalid;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<double> params;
  bool has_prior_focal_length = false;

  bool HasValidId() const;
  bool VerifyParams() const;
  bool HasBogusParams(double min_focal_length_ratio = 0.1,
                      double max_focal_length_ratio = 10.0,
                      double max_extra_param = 1.0) const;
  double MeanFocalLength() const;
  Eigen::Matrix3d CalibrationMatrix() const;
};

const CameraModelSpec& GetCameraModelSpec(CameraModelId model_id);
CameraModelId CameraModelNameToId(std::string_view name);
std::string_view CameraModelIdToName(CameraModelId model_id);
bool ExistsCameraModel(CameraModelId model_id);
bool CameraModelProjectionImplemented(CameraModelId model_id);
bool CameraModelUnprojectionImplemented(CameraModelId model_id);

// 按模型默认方式初始化参数：焦距使用 focal_length，主点位于图像中心，畸变为 0。
std::vector<double> InitializeCameraParams(CameraModelId model_id,
                                           double focal_length,
                                           std::uint32_t width,
                                           std::uint32_t height);

// 根据模型参数组展开需要优化的参数索引。
std::vector<std::size_t> CameraParameterIndicesForGroups(
    CameraModelId model_id,
    CameraParameterGroup groups);

std::vector<std::size_t> CameraParameterIndicesForMask(
    CameraModelId model_id,
    const CameraParameterRefinementMask& mask);

bool ProjectPointWithCameraParams(CameraModelId model_id,
                                  const double* params,
                                  const Eigen::Vector3d& point_camera,
                                  Eigen::Vector2d* projected);

bool UnprojectPixelWithCameraParams(CameraModelId model_id,
                                    const double* params,
                                    const Eigen::Vector2d& pixel,
                                    double depth,
                                    Eigen::Vector3d* point_camera);

Eigen::Vector2d ProjectPoint(const Camera& camera,
                             const Eigen::Vector3d& point_camera);

Eigen::Vector3d UnprojectPixel(const Camera& camera,
                               const Eigen::Vector2d& pixel,
                               double depth);

Eigen::Vector2d ApplySimpleRadialDistortion(const Eigen::Vector2d& normalized,
                                            double k1);

}  // namespace colite
