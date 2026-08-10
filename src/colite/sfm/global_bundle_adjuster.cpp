#include "colite/sfm/global_bundle_adjuster.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

#include "colite/core/geometry.h"
#include "colite/core/parallel.h"

namespace colite {
namespace {

struct PoseBlock {
  double q_xyzw[4] = {0.0, 0.0, 0.0, 1.0};
  double t[3] = {0.0, 0.0, 0.0};
};

struct PointBlock {
  double xyz[3] = {0.0, 0.0, 0.0};
};

struct Observation {
  ImageId image_id = kInvalidImageId;
  Point3DId point3d_id = kInvalidPoint3DId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
};

struct PositionPriorCost {
  PositionPriorCost(Eigen::Vector3d position, double inverse_stddev)
      : position_(std::move(position)), inverse_stddev_(inverse_stddev) {}

  template <typename T>
  bool operator()(const T* const q_xyzw,
                  const T* const translation,
                  T* residuals) const {
    const Eigen::Quaternion<T> rotation(q_xyzw[3], q_xyzw[0], q_xyzw[1],
                                        q_xyzw[2]);
    const Eigen::Matrix<T, 3, 1> t(translation[0], translation[1],
                                   translation[2]);
    const Eigen::Matrix<T, 3, 1> center = rotation.conjugate() * (-t);
    for (int idx = 0; idx < 3; ++idx) {
      residuals[idx] =
          (center[idx] - T(position_[idx])) * T(inverse_stddev_);
    }
    return true;
  }

  Eigen::Vector3d position_;
  double inverse_stddev_ = 1.0;
};

bool ProjectWithParams(CameraModelId model_id,
                       const double* params,
                       const Eigen::Vector3d& point_camera,
                       Eigen::Vector2d* projected) {
  return ProjectPointWithCameraParams(model_id, params, point_camera,
                                      projected);
}

bool AnalyticProjectionJacobian(CameraModelId model_id,
                                const double* params,
                                const Eigen::Vector3d& point_camera,
                                Eigen::Matrix<double, 2, 3>* jacobian) {
  if (jacobian == nullptr || !point_camera.allFinite() ||
      point_camera.z() <= 0.0) {
    return false;
  }
  const double inverse_z = 1.0 / point_camera.z();
  const double x = point_camera.x() * inverse_z;
  const double y = point_camera.y() * inverse_z;
  Eigen::Matrix2d normalized_jacobian = Eigen::Matrix2d::Zero();
  if (model_id == CameraModelId::kSimplePinhole) {
    normalized_jacobian(0, 0) = params[0];
    normalized_jacobian(1, 1) = params[0];
  } else if (model_id == CameraModelId::kPinhole) {
    normalized_jacobian(0, 0) = params[0];
    normalized_jacobian(1, 1) = params[1];
  } else if (model_id == CameraModelId::kSimpleRadial) {
    const double focal = params[0];
    const double radial = params[3];
    const double xy = 2.0 * radial * x * y;
    const double scale = 1.0 + radial * (x * x + y * y);
    normalized_jacobian(0, 0) =
        focal * (scale + 2.0 * radial * x * x);
    normalized_jacobian(0, 1) = focal * xy;
    normalized_jacobian(1, 0) = focal * xy;
    normalized_jacobian(1, 1) =
        focal * (scale + 2.0 * radial * y * y);
  } else {
    return false;
  }
  Eigen::Matrix<double, 2, 3> perspective_jacobian;
  perspective_jacobian << inverse_z, 0.0, -x * inverse_z,
                          0.0, inverse_z, -y * inverse_z;
  *jacobian = normalized_jacobian * perspective_jacobian;
  return jacobian->allFinite();
}

bool AnalyticCameraParamsJacobian(CameraModelId model_id,
                                  const double* params,
                                  const Eigen::Vector3d& point_camera,
                                  double* jacobian) {
  if (jacobian == nullptr || !point_camera.allFinite() ||
      point_camera.z() <= 0.0) {
    return false;
  }
  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();
  if (model_id == CameraModelId::kSimplePinhole) {
    jacobian[0] = x; jacobian[1] = 1.0; jacobian[2] = 0.0;
    jacobian[3] = y; jacobian[4] = 0.0; jacobian[5] = 1.0;
    return true;
  }
  if (model_id == CameraModelId::kPinhole) {
    jacobian[0] = x;   jacobian[1] = 0.0;
    jacobian[2] = 1.0; jacobian[3] = 0.0;
    jacobian[4] = 0.0; jacobian[5] = y;
    jacobian[6] = 0.0; jacobian[7] = 1.0;
    return true;
  }
  if (model_id == CameraModelId::kSimpleRadial) {
    const double radius2 = x * x + y * y;
    const double scale = 1.0 + params[3] * radius2;
    jacobian[0] = x * scale;
    jacobian[1] = 1.0;
    jacobian[2] = 0.0;
    jacobian[3] = params[0] * x * radius2;
    jacobian[4] = y * scale;
    jacobian[5] = 0.0;
    jacobian[6] = 1.0;
    jacobian[7] = params[0] * y * radius2;
    return true;
  }
  return false;
}

Eigen::Matrix<double, 3, 4> AnalyticQuaternionPointJacobian(
    const double* q_xyzw,
    const Eigen::Vector3d& point_world) {
  Eigen::Vector4d raw(q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]);
  const double norm = raw.norm();
  if (!std::isfinite(norm) || norm <= 0.0) {
    return Eigen::Matrix<double, 3, 4>::Zero();
  }
  const Eigen::Vector4d q = raw / norm;
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double px = point_world.x(), py = point_world.y();
  const double pz = point_world.z();
  Eigen::Matrix<double, 3, 4> unit_jacobian;
  unit_jacobian.col(0) << 2.0 * y * py + 2.0 * z * pz,
      2.0 * y * px - 4.0 * x * py - 2.0 * w * pz,
      2.0 * z * px + 2.0 * w * py - 4.0 * x * pz;
  unit_jacobian.col(1) << -4.0 * y * px + 2.0 * x * py + 2.0 * w * pz,
      2.0 * x * px + 2.0 * z * pz,
      -2.0 * w * px + 2.0 * z * py - 4.0 * y * pz;
  unit_jacobian.col(2) << -4.0 * z * px - 2.0 * w * py + 2.0 * x * pz,
      2.0 * w * px - 4.0 * z * py + 2.0 * y * pz,
      2.0 * x * px + 2.0 * y * py;
  unit_jacobian.col(3) << -2.0 * z * py + 2.0 * y * pz,
      2.0 * z * px - 2.0 * x * pz,
      -2.0 * y * px + 2.0 * x * py;
  const Eigen::Matrix4d normalization_jacobian =
      (Eigen::Matrix4d::Identity() - q * q.transpose()) / norm;
  return unit_jacobian * normalization_jacobian;
}

class ReprojectionCost final : public ceres::CostFunction {
 public:
  ReprojectionCost(CameraModelId model_id,
                   std::size_t num_camera_params,
                   Eigen::Vector2d observed_xy)
      : model_id_(model_id),
        num_camera_params_(num_camera_params),
        observed_xy_(std::move(observed_xy)) {
    set_num_residuals(2);
    mutable_parameter_block_sizes()->push_back(4);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(3);
    mutable_parameter_block_sizes()->push_back(
        static_cast<int>(num_camera_params_));
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    if (!EvaluateResidual(parameters[0], parameters[1], parameters[2],
                          parameters[3], residuals)) {
      return false;
    }
    if (jacobians == nullptr) {
      return true;
    }

    constexpr double kEps = 1e-6;
    if (jacobians[0] != nullptr || jacobians[1] != nullptr ||
        jacobians[2] != nullptr) {
      Eigen::Quaterniond q(parameters[0][3], parameters[0][0],
                           parameters[0][1], parameters[0][2]);
      q.normalize();
      const Eigen::Matrix3d rotation = q.toRotationMatrix();
      const Eigen::Vector3d point_camera =
          rotation * Eigen::Vector3d(parameters[2][0], parameters[2][1],
                                     parameters[2][2]) +
          Eigen::Vector3d(parameters[1][0], parameters[1][1],
                          parameters[1][2]);
      Eigen::Matrix<double, 2, 3> projection_jacobian;
      if (!AnalyticProjectionJacobian(model_id_, parameters[3], point_camera,
                                      &projection_jacobian)) {
        for (int axis = 0; axis < 3; ++axis) {
          Eigen::Vector3d perturbed = point_camera;
          perturbed[axis] += kEps;
          Eigen::Vector2d projected;
          if (!ProjectWithParams(model_id_, parameters[3], perturbed,
                                 &projected)) {
            return false;
          }
          projection_jacobian(0, axis) =
              (projected.x() - (residuals[0] + observed_xy_.x())) / kEps;
          projection_jacobian(1, axis) =
              (projected.y() - (residuals[1] + observed_xy_.y())) / kEps;
        }
      }
      if (jacobians[1] != nullptr) {
        for (int row = 0; row < 2; ++row) {
          for (int col = 0; col < 3; ++col) {
            jacobians[1][row * 3 + col] = projection_jacobian(row, col);
          }
        }
      }
      if (jacobians[2] != nullptr) {
        const Eigen::Matrix<double, 2, 3> point_jacobian =
            projection_jacobian * rotation;
        for (int row = 0; row < 2; ++row) {
          for (int col = 0; col < 3; ++col) {
            jacobians[2][row * 3 + col] = point_jacobian(row, col);
          }
        }
      }
      if (jacobians[0] != nullptr) {
        const Eigen::Vector3d point_world(parameters[2][0], parameters[2][1],
                                          parameters[2][2]);
        const Eigen::Matrix<double, 2, 4> quaternion_jacobian =
            projection_jacobian *
            AnalyticQuaternionPointJacobian(parameters[0], point_world);
        for (int row = 0; row < 2; ++row) {
          for (int col = 0; col < 4; ++col) {
            jacobians[0][row * 4 + col] = quaternion_jacobian(row, col);
          }
        }
      }
    }

    bool analytic_camera_params = false;
    if (jacobians[3] != nullptr) {
      Eigen::Quaterniond q(parameters[0][3], parameters[0][0],
                           parameters[0][1], parameters[0][2]);
      q.normalize();
      const Eigen::Vector3d point_camera =
          q * Eigen::Vector3d(parameters[2][0], parameters[2][1],
                              parameters[2][2]) +
          Eigen::Vector3d(parameters[1][0], parameters[1][1],
                          parameters[1][2]);
      analytic_camera_params = AnalyticCameraParamsJacobian(
          model_id_, parameters[3], point_camera, jacobians[3]);
    }
    const int block_sizes[4] = {4, 3, 3, static_cast<int>(num_camera_params_)};
    const int numeric_blocks[2] = {0, 3};
    for (const int block_idx : numeric_blocks) {
      if (jacobians[block_idx] == nullptr) {
        continue;
      }
      if (block_idx == 0 ||
          (block_idx == 3 && analytic_camera_params)) {
        continue;
      }
      for (int param_idx = 0; param_idx < block_sizes[block_idx]; ++param_idx) {
        std::vector<double> perturbed(
            parameters[block_idx],
            parameters[block_idx] + block_sizes[block_idx]);
        perturbed[param_idx] += kEps;
        double const* plus_parameters[4] = {parameters[0], parameters[1],
                                            parameters[2], parameters[3]};
        plus_parameters[block_idx] = perturbed.data();
        double plus_residuals[2] = {0.0, 0.0};
        if (!EvaluateResidual(plus_parameters[0], plus_parameters[1],
                              plus_parameters[2], plus_parameters[3],
                              plus_residuals)) {
          return false;
        }
        jacobians[block_idx][0 * block_sizes[block_idx] + param_idx] =
            (plus_residuals[0] - residuals[0]) / kEps;
        jacobians[block_idx][1 * block_sizes[block_idx] + param_idx] =
            (plus_residuals[1] - residuals[1]) / kEps;
      }
    }
    return true;
  }

 private:
  bool EvaluateResidual(const double* q_xyzw,
                        const double* t,
                        const double* xyz,
                        const double* camera_params,
                        double* residuals) const {
    Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
    q.normalize();
    const Eigen::Vector3d point_world(xyz[0], xyz[1], xyz[2]);
    const Eigen::Vector3d point_camera =
        q * point_world + Eigen::Vector3d(t[0], t[1], t[2]);
    Eigen::Vector2d projected;
    if (!ProjectWithParams(model_id_, camera_params, point_camera,
                           &projected)) {
      return false;
    }
    residuals[0] = projected.x() - observed_xy_.x();
    residuals[1] = projected.y() - observed_xy_.y();
    return true;
  }

  CameraModelId model_id_ = CameraModelId::kInvalid;
  std::size_t num_camera_params_ = 0;
  Eigen::Vector2d observed_xy_ = Eigen::Vector2d::Zero();
};

std::vector<int> ConstantCameraParameterIndices(
    const Camera& camera,
    CameraParameterGroup refine_groups,
    const std::set<std::string>& refine_param_names) {
  CameraParameterRefinementMask mask;
  mask.groups = refine_groups;
  mask.parameter_names = refine_param_names;
  const std::vector<std::size_t> variable_indices =
      CameraParameterIndicesForMask(camera.model_id, mask);
  std::set<std::size_t> variable_set(variable_indices.begin(),
                                     variable_indices.end());
  std::vector<int> constant_indices;
  for (std::size_t idx = 0; idx < camera.params.size(); ++idx) {
    if (variable_set.find(idx) == variable_set.end()) {
      constant_indices.push_back(static_cast<int>(idx));
    }
  }
  return constant_indices;
}

std::map<ImageId, std::vector<int>> BuildGaugeTranslationConstraints(
    const BundleGaugeFixing gauge_fixing,
    const std::map<ImageId, PoseBlock>& pose_blocks,
    const std::set<ImageId>& constant_pose_image_ids,
    const std::map<ImageId, std::vector<int>>& explicit_constraints,
    std::set<ImageId>* gauge_constant_pose_image_ids) {
  std::map<ImageId, std::vector<int>> constraints = explicit_constraints;
  if (gauge_constant_pose_image_ids == nullptr ||
      gauge_fixing == BundleGaugeFixing::kNone || pose_blocks.empty() ||
      !constant_pose_image_ids.empty()) {
    return constraints;
  }

  auto it = pose_blocks.begin();
  gauge_constant_pose_image_ids->insert(it->first);
  if (gauge_fixing == BundleGaugeFixing::kFirstTwoImages) {
    ++it;
    if (it != pose_blocks.end()) {
      constraints.emplace(it->first, std::vector<int>{0});
    }
  }
  return constraints;
}

PoseBlock MakePoseBlock(const Pose3& pose) {
  PoseBlock block;
  block.q_xyzw[0] = pose.q_cw.x();
  block.q_xyzw[1] = pose.q_cw.y();
  block.q_xyzw[2] = pose.q_cw.z();
  block.q_xyzw[3] = pose.q_cw.w();
  block.t[0] = pose.t_cw.x();
  block.t[1] = pose.t_cw.y();
  block.t[2] = pose.t_cw.z();
  return block;
}

PointBlock MakePointBlock(const Point3D& point3d) {
  PointBlock block;
  block.xyz[0] = point3d.xyz.x();
  block.xyz[1] = point3d.xyz.y();
  block.xyz[2] = point3d.xyz.z();
  return block;
}

void WritePoseBlock(const PoseBlock& block, Pose3* pose) {
  pose->q_cw = Eigen::Quaterniond(block.q_xyzw[3], block.q_xyzw[0],
                                  block.q_xyzw[1], block.q_xyzw[2])
                   .normalized();
  pose->t_cw = Eigen::Vector3d(block.t[0], block.t[1], block.t[2]);
}

void WritePointBlock(const PointBlock& block, Point3D* point3d) {
  point3d->xyz = Eigen::Vector3d(block.xyz[0], block.xyz[1], block.xyz[2]);
}

std::vector<Observation> CollectRegisteredObservations(
    const Reconstruction& reconstruction) {
  std::vector<Observation> observations;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    for (const TrackElement& element : point3d.track.elements) {
      if (!reconstruction.IsImageRegistered(element.image_id) ||
          !reconstruction.ExistsImage(element.image_id)) {
        continue;
      }
      const Image& image = reconstruction.GetImage(element.image_id);
      if (!image.HasPose() || element.point2d_idx >= image.points2d.size()) {
        continue;
      }
      observations.push_back({element.image_id, point3d_id, element.point2d_idx,
                              image.points2d[element.point2d_idx].xy});
    }
  }
  return observations;
}

double MeanReprojectionError(const Reconstruction& reconstruction,
                             const std::vector<Observation>& observations) {
  if (observations.empty()) {
    return -1.0;
  }
  double total_error = 0.0;
  std::size_t num_errors = 0;
  for (const Observation& observation : observations) {
    const Image& image = reconstruction.GetImage(observation.image_id);
    const Camera& camera = reconstruction.GetCamera(image.camera_id);
    const Point3D& point3d = reconstruction.GetPoint3D(observation.point3d_id);
    const auto error = ReprojectionError(camera, *image.cam_from_world,
                                         point3d.xyz, observation.xy);
    if (error.has_value()) {
      total_error += *error;
      ++num_errors;
    }
  }
  return num_errors == 0 ? -1.0 : total_error / static_cast<double>(num_errors);
}

double MeanPositionPriorError(const Reconstruction& reconstruction,
                              std::size_t* num_priors) {
  double total_error = 0.0;
  std::size_t count = 0;
  for (const ImageId image_id : reconstruction.RegisteredImageIds()) {
    const Image& image = reconstruction.GetImage(image_id);
    if (!image.cam_from_world.has_value() ||
        !image.prior_position.has_value()) {
      continue;
    }
    total_error +=
        (image.cam_from_world->CameraCenter() - *image.prior_position).norm();
    ++count;
  }
  if (num_priors != nullptr) *num_priors = count;
  return count == 0 ? -1.0 : total_error / static_cast<double>(count);
}

}  // namespace

GlobalBundleAdjustmentSummary AdjustGlobalBundle(
    const GlobalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction) {
  GlobalBundleAdjustmentSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  const std::vector<Observation> observations =
      CollectRegisteredObservations(*reconstruction);
  summary.num_residuals = observations.size();
  summary.initial_mean_reprojection_error =
      MeanReprojectionError(*reconstruction, observations);
  summary.initial_mean_position_error =
      MeanPositionPriorError(*reconstruction, &summary.num_position_priors);
  if (reconstruction->RegisteredImageIds().empty() ||
      reconstruction->Points3D().empty() || observations.empty()) {
    summary.message = "Global bundle has no observations";
    return summary;
  }

  std::map<ImageId, PoseBlock> pose_blocks;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    const Image& image = reconstruction->GetImage(image_id);
    if (image.HasPose()) {
      pose_blocks.emplace(image_id, MakePoseBlock(*image.cam_from_world));
    }
  }

  std::map<Point3DId, PointBlock> point_blocks;
  for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
    point_blocks.emplace(point3d_id, MakePointBlock(point3d));
  }

  std::map<CameraId, std::vector<double>> camera_blocks;
  for (const auto& [camera_id, camera] : reconstruction->Cameras()) {
    camera_blocks.emplace(camera_id, camera.params);
  }

  ceres::Problem problem;
  std::set<ImageId> constant_pose_image_ids = options.constant_image_ids;
  const std::map<ImageId, std::vector<int>> constant_translation_idxs =
      BuildGaugeTranslationConstraints(
          options.gauge_fixing, pose_blocks, constant_pose_image_ids,
          options.constant_image_translation_idxs,
          &summary.gauge_constant_pose_image_ids);
  constant_pose_image_ids.insert(summary.gauge_constant_pose_image_ids.begin(),
                                 summary.gauge_constant_pose_image_ids.end());
  summary.gauge_constant_translation_idxs = constant_translation_idxs;

  for (auto& [image_id, block] : pose_blocks) {
    problem.AddParameterBlock(
        block.q_xyzw, 4, new ceres::EigenQuaternionParameterization());
    problem.AddParameterBlock(block.t, 3);
    if (!options.refine_poses ||
        constant_pose_image_ids.find(image_id) !=
            constant_pose_image_ids.end()) {
      problem.SetParameterBlockConstant(block.q_xyzw);
      problem.SetParameterBlockConstant(block.t);
    } else {
      ++summary.num_variable_images;
      const auto translation_it = constant_translation_idxs.find(image_id);
      if (translation_it != constant_translation_idxs.end() &&
          !translation_it->second.empty()) {
        problem.SetParameterization(
            block.t,
            new ceres::SubsetParameterization(3, translation_it->second));
        ++summary.num_gauge_constant_translation_images;
      }
    }
  }

  // 若调用方没有指定固定图像，则固定第一张已注册图像消除规范自由度。
  if (options.constant_image_ids.empty() && !pose_blocks.empty()) {
  }

  for (auto& [point3d_id, block] : point_blocks) {
    problem.AddParameterBlock(block.xyz, 3);
    if (!options.refine_points) {
      problem.SetParameterBlockConstant(block.xyz);
    } else {
      (void)point3d_id;
      ++summary.num_variable_points;
    }
  }

  for (auto& [camera_id, params] : camera_blocks) {
    Camera& camera = reconstruction->GetCamera(camera_id);
    problem.AddParameterBlock(params.data(),
                              static_cast<int>(params.size()));
    const CameraModelSpec& spec = GetCameraModelSpec(camera.model_id);
    for (const std::size_t idx : spec.focal_length_idxs) {
      problem.SetParameterLowerBound(
          params.data(), static_cast<int>(idx), 1e-8);
    }
    if (camera.model_id == CameraModelId::kFOV) {
      const int omega_idx = static_cast<int>(spec.distortion_idxs.front());
      const double limit = std::acos(-1.0) - 1e-8;
      problem.SetParameterLowerBound(params.data(), omega_idx, -limit);
      problem.SetParameterUpperBound(params.data(), omega_idx, limit);
    }
    const std::vector<int> constant_indices =
        ConstantCameraParameterIndices(camera, options.refine_camera_params,
                                       options.refine_camera_param_names);
    if (constant_indices.size() == params.size()) {
      problem.SetParameterBlockConstant(params.data());
    } else {
      ++summary.num_variable_cameras;
      if (!constant_indices.empty()) {
        problem.SetParameterization(
            params.data(),
            new ceres::SubsetParameterization(
                static_cast<int>(params.size()), constant_indices));
      }
    }
  }

  for (const Observation& observation : observations) {
    const Image& image = reconstruction->GetImage(observation.image_id);
    const Camera& camera = reconstruction->GetCamera(image.camera_id);
    ceres::LossFunction* loss_function =
        options.loss_scale > 0.0 ? new ceres::SoftLOneLoss(options.loss_scale)
                                 : nullptr;
    problem.AddResidualBlock(
        new ReprojectionCost(camera.model_id, camera.params.size(),
                             observation.xy),
        loss_function,
        pose_blocks.at(observation.image_id).q_xyzw,
        pose_blocks.at(observation.image_id).t,
        point_blocks.at(observation.point3d_id).xyz,
        camera_blocks.at(image.camera_id).data());
  }

  if (options.use_position_priors) {
    if (!std::isfinite(options.position_prior_stddev) ||
        options.position_prior_stddev <= 0.0) {
      summary.message = "Position prior standard deviation must be positive";
      return summary;
    }
    for (const auto& [image_id, block] : pose_blocks) {
      const Image& image = reconstruction->GetImage(image_id);
      if (!image.prior_position.has_value()) continue;
      ceres::LossFunction* loss =
          options.position_prior_loss_scale > 0.0
              ? new ceres::HuberLoss(options.position_prior_loss_scale /
                                     options.position_prior_stddev)
              : nullptr;
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<PositionPriorCost, 3, 4, 3>(
              new PositionPriorCost(*image.prior_position,
                                    1.0 / options.position_prior_stddev)),
          loss, pose_blocks.at(image_id).q_xyzw,
          pose_blocks.at(image_id).t);
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = parallel::getNumThreads();

  ceres::Solver::Summary solver_summary;
  ceres::Solve(solver_options, &problem, &solver_summary);

  for (const auto& [image_id, block] : pose_blocks) {
    Image& image = reconstruction->GetImage(image_id);
    WritePoseBlock(block, &image.cam_from_world.value());
  }
  for (const auto& [point3d_id, block] : point_blocks) {
    WritePointBlock(block, &reconstruction->GetPoint3D(point3d_id));
  }
  for (const auto& [camera_id, params] : camera_blocks) {
    reconstruction->GetCamera(camera_id).params = params;
  }

  summary.final_mean_reprojection_error =
      MeanReprojectionError(*reconstruction, observations);
  summary.final_mean_position_error =
      MeanPositionPriorError(*reconstruction, nullptr);
  summary.success = solver_summary.IsSolutionUsable();
  summary.message = solver_summary.BriefReport();
  return summary;
}

}  // namespace colite
