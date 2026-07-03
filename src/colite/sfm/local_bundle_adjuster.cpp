#include "colite/sfm/local_bundle_adjuster.h"

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
  return ProjectPointWithCameraParams(model_id, params, point_camera,
                                      projected);
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
    const Eigen::Vector3d point_camera =
        q * Eigen::Vector3d(xyz[0], xyz[1], xyz[2]) +
        Eigen::Vector3d(t[0], t[1], t[2]);
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

std::vector<std::pair<ImageId, std::size_t>> SortImageScores(
    const std::map<ImageId, std::size_t>& image_scores) {
  std::vector<std::pair<ImageId, std::size_t>> sorted_scores(
      image_scores.begin(), image_scores.end());
  std::sort(sorted_scores.begin(), sorted_scores.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });
  return sorted_scores;
}

std::set<ImageId> CollectLocalImagesBySharedPoints(
    const Reconstruction& reconstruction,
    ImageId root_image_id,
    std::size_t max_num_images) {
  std::set<ImageId> image_ids;
  if (!reconstruction.IsImageRegistered(root_image_id) ||
      max_num_images == 0) {
    return image_ids;
  }

  image_ids.insert(root_image_id);
  if (image_ids.size() >= max_num_images) {
    return image_ids;
  }

  std::map<ImageId, std::size_t> shared_point_counts;
  const Image& root_image = reconstruction.GetImage(root_image_id);
  for (const Point2D& point2d : root_image.points2d) {
    if (!point2d.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2d.point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction.GetPoint3D(point2d.point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      if (element.image_id != root_image_id &&
          reconstruction.IsImageRegistered(element.image_id)) {
        ++shared_point_counts[element.image_id];
      }
    }
  }
  for (const auto& [image_id, count] : SortImageScores(shared_point_counts)) {
    (void)count;
    image_ids.insert(image_id);
    if (image_ids.size() >= max_num_images) {
      break;
    }
  }
  return image_ids;
}

std::set<Point3DId> CollectLocalPoints(const Reconstruction& reconstruction,
                                       const std::set<ImageId>& image_ids) {
  std::set<Point3DId> point_ids;
  for (const ImageId image_id : image_ids) {
    const Image& image = reconstruction.GetImage(image_id);
    for (const Point2D& point2d : image.points2d) {
      if (point2d.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2d.point3d_id)) {
        point_ids.insert(point2d.point3d_id);
      }
    }
  }
  return point_ids;
}

std::set<ImageId> CollectFixedImages(
    const Reconstruction& reconstruction,
    const std::set<ImageId>& local_image_ids,
    const std::set<Point3DId>& point_ids,
    const LocalBundleAdjustmentOptions& options) {
  std::set<ImageId> fixed_image_ids;
  std::map<ImageId, std::size_t> fixed_image_scores;
  for (const Point3DId point3d_id : point_ids) {
    const Point3D& point3d = reconstruction.GetPoint3D(point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      if (local_image_ids.find(element.image_id) != local_image_ids.end() ||
          !reconstruction.IsImageRegistered(element.image_id)) {
        continue;
      }
      ++fixed_image_scores[element.image_id];
    }
  }

  for (const ImageId image_id : options.constant_image_ids) {
    if (fixed_image_scores.find(image_id) != fixed_image_scores.end()) {
      fixed_image_ids.insert(image_id);
      if (fixed_image_ids.size() >= options.max_num_fixed_images) {
        return fixed_image_ids;
      }
    }
  }

  for (const auto& [image_id, count] : SortImageScores(fixed_image_scores)) {
    (void)count;
    fixed_image_ids.insert(image_id);
    if (fixed_image_ids.size() >= options.max_num_fixed_images) {
      break;
    }
  }
  return fixed_image_ids;
}

std::set<ImageId> UnionImageIds(const std::set<ImageId>& local_image_ids,
                                const std::set<ImageId>& fixed_image_ids) {
  std::set<ImageId> image_ids = local_image_ids;
  image_ids.insert(fixed_image_ids.begin(), fixed_image_ids.end());
  return image_ids;
}

std::vector<Observation> CollectObservations(
    const Reconstruction& reconstruction,
    const std::set<ImageId>& image_ids,
    const std::set<Point3DId>& point_ids) {
  std::vector<Observation> observations;
  for (const Point3DId point3d_id : point_ids) {
    const Point3D& point3d = reconstruction.GetPoint3D(point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      if (image_ids.find(element.image_id) == image_ids.end()) {
        continue;
      }
      const Image& image = reconstruction.GetImage(element.image_id);
      if (element.point2d_idx >= image.points2d.size() || !image.HasPose()) {
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

}  // namespace

LocalBundleAdjustmentSummary AdjustLocalBundle(
    ImageId root_image_id,
    const LocalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction) {
  LocalBundleAdjustmentSummary summary;
  summary.root_image_id = root_image_id;

  if (reconstruction == nullptr ||
      !reconstruction->IsImageRegistered(root_image_id)) {
    summary.message = "Root image is not registered";
    return summary;
  }

  const std::set<ImageId> local_image_ids = CollectLocalImagesBySharedPoints(
      *reconstruction, root_image_id, options.max_num_images);
  const std::set<Point3DId> point_ids =
      CollectLocalPoints(*reconstruction, local_image_ids);
  const std::set<ImageId> fixed_image_ids =
      CollectFixedImages(*reconstruction, local_image_ids, point_ids, options);
  const std::set<ImageId> image_ids =
      UnionImageIds(local_image_ids, fixed_image_ids);
  const std::vector<Observation> observations =
      CollectObservations(*reconstruction, image_ids, point_ids);
  summary.local_image_ids = local_image_ids;
  summary.fixed_image_ids = fixed_image_ids;
  summary.num_local_images = local_image_ids.size();
  summary.num_residuals = observations.size();
  summary.initial_mean_reprojection_error =
      MeanReprojectionError(*reconstruction, observations);

  if (image_ids.empty() || point_ids.empty() || observations.empty()) {
    summary.message = "Local bundle has no observations";
    return summary;
  }

  std::map<ImageId, PoseBlock> pose_blocks;
  for (const ImageId image_id : image_ids) {
    const Image& image = reconstruction->GetImage(image_id);
    if (image.HasPose()) {
      pose_blocks.emplace(image_id, MakePoseBlock(*image.cam_from_world));
    }
  }

  std::map<Point3DId, PointBlock> point_blocks;
  for (const Point3DId point3d_id : point_ids) {
    point_blocks.emplace(point3d_id,
                         MakePointBlock(reconstruction->GetPoint3D(point3d_id)));
  }

  std::set<CameraId> camera_ids;
  for (const ImageId image_id : image_ids) {
    camera_ids.insert(reconstruction->GetImage(image_id).camera_id);
  }
  std::map<CameraId, std::vector<double>> camera_blocks;
  for (const CameraId camera_id : camera_ids) {
    camera_blocks.emplace(camera_id, reconstruction->GetCamera(camera_id).params);
  }

  ceres::Problem problem;
  std::set<ImageId> constant_pose_image_ids = fixed_image_ids;
  for (const ImageId image_id : options.constant_image_ids) {
    if (pose_blocks.find(image_id) != pose_blocks.end()) {
      constant_pose_image_ids.insert(image_id);
    }
  }
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
    const bool is_local_image =
        local_image_ids.find(image_id) != local_image_ids.end();
    const bool is_constant_pose =
        !options.refine_poses || !is_local_image ||
        constant_pose_image_ids.find(image_id) != constant_pose_image_ids.end();
    if (is_constant_pose) {
      problem.SetParameterBlockConstant(block.q_xyzw);
      problem.SetParameterBlockConstant(block.t);
      ++summary.num_constant_images;
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

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type = ceres::DENSE_SCHUR;
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
