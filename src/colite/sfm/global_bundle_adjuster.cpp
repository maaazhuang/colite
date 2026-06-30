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

bool ProjectWithParams(CameraModelId model_id,
                       const double* params,
                       const Eigen::Vector3d& point_camera,
                       Eigen::Vector2d* projected) {
  if (point_camera.z() <= 0.0) {
    return false;
  }
  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();

  if (model_id == CameraModelId::kSimplePinhole) {
    *projected = {params[0] * x + params[1], params[0] * y + params[2]};
    return true;
  }
  if (model_id == CameraModelId::kPinhole) {
    *projected = {params[0] * x + params[2], params[1] * y + params[3]};
    return true;
  }
  if (model_id == CameraModelId::kSimpleRadial) {
    const double radius2 = x * x + y * y;
    const double scale = 1.0 + params[3] * radius2;
    *projected = {params[0] * x * scale + params[1],
                  params[0] * y * scale + params[2]};
    return true;
  }
  if (model_id == CameraModelId::kRadial) {
    const double radius2 = x * x + y * y;
    const double scale =
        1.0 + params[3] * radius2 + params[4] * radius2 * radius2;
    *projected = {params[0] * x * scale + params[1],
                  params[0] * y * scale + params[2]};
    return true;
  }
  if (model_id == CameraModelId::kOpenCV) {
    const double x2 = x * x;
    const double y2 = y * y;
    const double xy = x * y;
    const double radius2 = x2 + y2;
    const double radial =
        1.0 + params[4] * radius2 + params[5] * radius2 * radius2;
    const double distorted_x =
        x * radial + 2.0 * params[6] * xy +
        params[7] * (radius2 + 2.0 * x2);
    const double distorted_y =
        y * radial + params[6] * (radius2 + 2.0 * y2) +
        2.0 * params[7] * xy;
    *projected = {params[0] * distorted_x + params[2],
                  params[1] * distorted_y + params[3]};
    return true;
  }
  return false;
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
    const int block_sizes[4] = {4, 3, 3, static_cast<int>(num_camera_params_)};
    for (int block_idx = 0; block_idx < 4; ++block_idx) {
      if (jacobians[block_idx] == nullptr) {
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
    CameraParameterGroup refine_groups) {
  const std::vector<std::size_t> variable_indices =
      CameraParameterIndicesForGroups(camera.model_id, refine_groups);
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
  for (auto& [image_id, block] : pose_blocks) {
    problem.AddParameterBlock(
        block.q_xyzw, 4, new ceres::EigenQuaternionParameterization());
    problem.AddParameterBlock(block.t, 3);
    if (!options.refine_poses ||
        options.constant_image_ids.find(image_id) !=
            options.constant_image_ids.end()) {
      problem.SetParameterBlockConstant(block.q_xyzw);
      problem.SetParameterBlockConstant(block.t);
    } else {
      ++summary.num_variable_images;
    }
  }

  // 若调用方没有指定固定图像，则固定第一张已注册图像消除规范自由度。
  if (options.constant_image_ids.empty() && !pose_blocks.empty()) {
    auto& block = pose_blocks.begin()->second;
    problem.SetParameterBlockConstant(block.q_xyzw);
    problem.SetParameterBlockConstant(block.t);
    if (summary.num_variable_images > 0) {
      --summary.num_variable_images;
    }
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
    const std::vector<int> constant_indices =
        ConstantCameraParameterIndices(camera, options.refine_camera_params);
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

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = 1;

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
  summary.success = solver_summary.IsSolutionUsable();
  summary.message = solver_summary.BriefReport();
  return summary;
}

}  // namespace colite
