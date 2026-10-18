#include "colite/sfm/global_mapper.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

#include "colite/sfm/incremental_mapper.h"

namespace colite {
namespace {

void SyncImagePoints2DFromFeatures(const FeatureSet& features, Image* image) {
  if (image->points2d.size() < features.keypoints.size()) {
    image->points2d.resize(features.keypoints.size());
  }
  for (std::size_t idx = 0; idx < features.keypoints.size(); ++idx) {
    image->points2d[idx].xy =
        Eigen::Vector2d(features.keypoints[idx].x, features.keypoints[idx].y);
  }
}

void AccumulateTriangulation(const IncrementalTriangulatorSummary& source,
                             IncrementalTriangulatorSummary* target) {
  target->num_created_points += source.num_created_points;
  target->num_created_observations += source.num_created_observations;
  target->num_completed_observations += source.num_completed_observations;
  target->num_merged_points += source.num_merged_points;
  target->num_merged_observations += source.num_merged_observations;
  target->num_filtered_observations += source.num_filtered_observations;
  target->num_filtered_points += source.num_filtered_points;
}

}  // namespace

GlobalMappingSummary RunGlobalMapping(const MatchGraph& graph,
                                      const GlobalMappingOptions& options,
                                      Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    throw std::invalid_argument("reconstruction must not be null");
  }
  GlobalMappingSummary summary;
  if (!reconstruction->RegisteredImageIds().empty() ||
      !reconstruction->Points3D().empty()) {
    summary.message =
        "Global mapping currently requires an unregistered reconstruction";
    return summary;
  }

  const std::vector<RelativePoseConstraint> constraints =
      BuildRelativePoseConstraints(graph, options.view_graph,
                                   &summary.view_graph);
  if (constraints.empty()) {
    summary.message = "View graph has no usable relative pose constraints";
    return summary;
  }
  summary.anchor_image_id = std::min_element(
      constraints.begin(), constraints.end(),
      [](const RelativePoseConstraint& lhs,
         const RelativePoseConstraint& rhs) {
        return std::min(lhs.image_id1, lhs.image_id2) <
               std::min(rhs.image_id1, rhs.image_id2);
      })->image_id1;
  for (const RelativePoseConstraint& constraint : constraints) {
    summary.anchor_image_id =
        std::min(summary.anchor_image_id,
                 std::min(constraint.image_id1, constraint.image_id2));
  }

  std::map<ImageId, Eigen::Quaterniond> rotations;
  summary.rotation_averaging = EstimateGlobalRotations(
      constraints, summary.anchor_image_id, options.rotation_averaging,
      &rotations);
  if (!summary.rotation_averaging.success) {
    summary.message = "Global rotation averaging failed";
    return summary;
  }
  std::vector<RelativePoseConstraint> positioning_constraints;
  positioning_constraints.reserve(
      summary.rotation_averaging.used_constraint_indices.size());
  for (const std::size_t idx :
       summary.rotation_averaging.used_constraint_indices) {
    positioning_constraints.push_back(constraints.at(idx));
  }
  std::map<ImageId, Eigen::Vector3d> centers;
  summary.positioning = EstimateGlobalCameraCenters(
      positioning_constraints, rotations, summary.anchor_image_id,
      options.positioning, &centers);
  if (!summary.positioning.success) {
    summary.message = "Global camera positioning failed";
    return summary;
  }

  for (const auto& [image_id, center] : centers) {
    if (!reconstruction->ExistsImage(image_id) ||
        graph.features.count(image_id) == 0) {
      continue;
    }
    Image& image = reconstruction->GetImage(image_id);
    SyncImagePoints2DFromFeatures(graph.features.at(image_id), &image);
    Pose3 pose;
    pose.q_cw = rotations.at(image_id).normalized();
    pose.t_cw = -(pose.q_cw * center);
    image.cam_from_world = pose;
    reconstruction->RegisterImage(image_id);
  }
  summary.num_registered_images =
      reconstruction->RegisteredImageIds().size();
  if (summary.num_registered_images < options.min_num_registered_images) {
    summary.message = "Too few globally positioned images";
    return summary;
  }

  const IncrementalTriangulatorCache cache(graph);
  const std::vector<ImageId> registered_ids(
      reconstruction->RegisteredImageIds().begin(),
      reconstruction->RegisteredImageIds().end());
  for (const ImageId image_id : registered_ids) {
    const IncrementalTriangulatorSummary image_summary =
        TriangulateRegisteredImage(graph, image_id, options.triangulation,
                                   reconstruction, &cache);
    AccumulateTriangulation(image_summary, &summary.triangulation);
  }
  IncrementalTriangulatorOptions completion_options = options.triangulation;
  completion_options.complete_neighbor_observations = true;
  const IncrementalTriangulatorSummary completion =
      CompleteAndMergeAllTracks(graph, completion_options, reconstruction,
                                &cache);
  AccumulateTriangulation(completion, &summary.triangulation);
  summary.triangulation.message = "Global tracks triangulated and merged";
  summary.num_points3d = reconstruction->Points3D().size();
  if (summary.num_points3d < options.min_num_points3d) {
    summary.message = "Global mapping triangulated too few points";
    return summary;
  }
  if (options.run_fixed_rotation_bundle_adjustment) {
    GlobalBundleAdjustmentOptions fixed_options =
        options.fixed_rotation_bundle_adjustment;
    fixed_options.refine_poses = true;
    fixed_options.refine_points = true;
    fixed_options.refine_camera_params = CameraParameterGroup::kNone;
    fixed_options.refine_camera_param_names.clear();
    fixed_options.constant_rotation_image_ids.insert(
        registered_ids.begin(), registered_ids.end());
    summary.fixed_rotation_bundle_adjustment =
        AdjustGlobalBundle(fixed_options, reconstruction);
    if (!summary.fixed_rotation_bundle_adjustment.success) {
      summary.message = "Fixed-rotation global bundle adjustment failed: " +
                        summary.fixed_rotation_bundle_adjustment.message;
      return summary;
    }
  }

  if (options.run_retriangulation) {
    summary.retriangulation = RetriangulateAllRegisteredImages(
        graph, options.triangulation, reconstruction, &cache);
    const IncrementalTriangulatorSummary closure =
        CompleteAndMergeAllTracks(graph, completion_options, reconstruction,
                                  &cache);
    AccumulateTriangulation(closure, &summary.retriangulation);
    summary.retriangulation.message =
        "Global retriangulation and track closure completed";
  }

  if (options.run_observation_filter) {
    summary.observation_filter =
        FilterObservationsAndPoints(options.observation_filter,
                                    reconstruction);
    if (!summary.observation_filter.success) {
      summary.message = "Global observation filtering failed: " +
                        summary.observation_filter.message;
      return summary;
    }
    if (reconstruction->Points3D().size() < options.min_num_points3d) {
      summary.message = "Global filtering retained too few points";
      return summary;
    }
  }

  if (options.run_bundle_adjustment) {
    summary.bundle_adjustment =
        AdjustGlobalBundle(options.bundle_adjustment, reconstruction);
    if (!summary.bundle_adjustment.success) {
      summary.message = "Global bundle adjustment failed: " +
                        summary.bundle_adjustment.message;
      return summary;
    }
  }
  summary.num_points3d = reconstruction->Points3D().size();
  summary.success = true;
  summary.message = "Global mapping completed";
  return summary;
}

}  // namespace colite
