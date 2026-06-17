#include "colite/sfm/global_bundle_adjuster.h"

#include <map>
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

struct ReprojectionCost {
  ReprojectionCost(CameraModelId model_id,
                   std::vector<double> params,
                   double observed_x,
                   double observed_y)
      : model_id(model_id),
        params(std::move(params)),
        observed_x(observed_x),
        observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const q_xyzw,
                  const T* const t,
                  const T* const xyz,
                  T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> q(q_xyzw);
    const Eigen::Matrix<T, 3, 1> point_world(xyz[0], xyz[1], xyz[2]);
    const Eigen::Matrix<T, 3, 1> point_camera =
        q * point_world + Eigen::Matrix<T, 3, 1>(t[0], t[1], t[2]);

    const T normalized_x = point_camera.x() / point_camera.z();
    const T normalized_y = point_camera.y() / point_camera.z();

    T projected_x = T(0);
    T projected_y = T(0);
    if (model_id == CameraModelId::kPinhole) {
      projected_x = T(params[0]) * normalized_x + T(params[2]);
      projected_y = T(params[1]) * normalized_y + T(params[3]);
    } else if (model_id == CameraModelId::kSimpleRadial) {
      const T radius2 = normalized_x * normalized_x + normalized_y * normalized_y;
      const T scale = T(1) + T(params[3]) * radius2;
      projected_x = T(params[0]) * normalized_x * scale + T(params[1]);
      projected_y = T(params[0]) * normalized_y * scale + T(params[2]);
    } else {
      return false;
    }

    residuals[0] = projected_x - T(observed_x);
    residuals[1] = projected_y - T(observed_y);
    return true;
  }

  static ceres::CostFunction* Create(const Camera& camera,
                                     const Eigen::Vector2d& observed_xy) {
    return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 4, 3, 3>(
        new ReprojectionCost(camera.model_id, camera.params, observed_xy.x(),
                             observed_xy.y()));
  }

  CameraModelId model_id;
  std::vector<double> params;
  double observed_x = 0.0;
  double observed_y = 0.0;
};

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

  for (const Observation& observation : observations) {
    const Image& image = reconstruction->GetImage(observation.image_id);
    const Camera& camera = reconstruction->GetCamera(image.camera_id);
    ceres::LossFunction* loss_function =
        options.loss_scale > 0.0 ? new ceres::SoftLOneLoss(options.loss_scale)
                                 : nullptr;
    problem.AddResidualBlock(
        ReprojectionCost::Create(camera, observation.xy), loss_function,
        pose_blocks.at(observation.image_id).q_xyzw,
        pose_blocks.at(observation.image_id).t,
        point_blocks.at(observation.point3d_id).xyz);
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

  summary.final_mean_reprojection_error =
      MeanReprojectionError(*reconstruction, observations);
  summary.success = solver_summary.IsSolutionUsable();
  summary.message = solver_summary.BriefReport();
  return summary;
}

}  // namespace colite
