#include "colite/sfm/reconstruction_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include "colite/sfm/observation_manager.h"

namespace colite {
namespace {

struct SimilarityTransform {
  double scale = 1.0;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  bool valid = false;

  Eigen::Vector3d TransformPoint(const Eigen::Vector3d& point) const {
    return scale * rotation * point + translation;
  }
};

struct PointPair {
  Point3DId target_id = kInvalidPoint3DId;
  Point3DId source_id = kInvalidPoint3DId;
  std::size_t support = 0;
};

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.type != TwoViewGeometry::Type::kUndefined &&
         geometry.type != TwoViewGeometry::Type::kDegenerate &&
         !geometry.inlier_matches.empty();
}

Reconstruction CloneSchema(const Reconstruction& source) {
  Reconstruction clone;
  for (const auto& [camera_id, camera] : source.Cameras()) {
    (void)camera_id;
    clone.AddCamera(camera);
  }
  for (const auto& [image_id, original] : source.Images()) {
    (void)image_id;
    Image image = original;
    image.registered = false;
    image.cam_from_world.reset();
    for (Point2D& point2d : image.points2d) {
      point2d.point3d_id = kInvalidPoint3DId;
    }
    clone.AddImage(std::move(image));
  }
  return clone;
}

void ValidateAndCompleteSeedFeatures(const MatchGraph& graph,
                                     Reconstruction* reconstruction) {
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    const FeatureSet* features = FindFeatures(graph, image_id);
    if (features == nullptr) {
      throw std::runtime_error(
          "Resume model image has no workspace features: " + image.name);
    }
    if (image.points2d.empty()) {
      image.points2d.resize(features->keypoints.size());
      for (std::size_t idx = 0; idx < features->keypoints.size(); ++idx) {
        image.points2d[idx].xy =
            Eigen::Vector2d(features->keypoints[idx].x,
                            features->keypoints[idx].y);
        image.points2d[idx].feature_idx =
            static_cast<Point2DIdx>(idx);
      }
      continue;
    }
    if (image.points2d.size() != features->keypoints.size()) {
      throw std::runtime_error(
          "Resume model feature count differs from workspace for image: " +
          image.name);
    }
    for (std::size_t idx = 0; idx < features->keypoints.size(); ++idx) {
      const Eigen::Vector2d workspace_xy(features->keypoints[idx].x,
                                         features->keypoints[idx].y);
      if ((image.points2d[idx].xy - workspace_xy).norm() > 1e-3) {
        throw std::runtime_error(
            "Resume model feature ordering differs from workspace for image: " +
            image.name);
      }
      image.points2d[idx].feature_idx =
          static_cast<Point2DIdx>(idx);
    }
  }
}

std::vector<std::set<ImageId>> VerifiedComponents(
    const MatchGraph& graph,
    const std::set<ImageId>& remaining) {
  std::map<ImageId, std::vector<ImageId>> adjacency;
  for (const ImageId image_id : remaining) {
    adjacency[image_id];
  }
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsUsableGeometry(geometry) ||
        remaining.count(pair.image_id1) == 0 ||
        remaining.count(pair.image_id2) == 0) {
      continue;
    }
    adjacency[pair.image_id1].push_back(pair.image_id2);
    adjacency[pair.image_id2].push_back(pair.image_id1);
  }

  std::set<ImageId> visited;
  std::vector<std::set<ImageId>> components;
  for (const auto& [start, neighbors] : adjacency) {
    (void)neighbors;
    if (!visited.insert(start).second) {
      continue;
    }
    std::set<ImageId> component;
    std::vector<ImageId> frontier{start};
    while (!frontier.empty()) {
      const ImageId image_id = frontier.back();
      frontier.pop_back();
      component.insert(image_id);
      for (const ImageId neighbor : adjacency[image_id]) {
        if (visited.insert(neighbor).second) {
          frontier.push_back(neighbor);
        }
      }
    }
    components.push_back(std::move(component));
  }
  std::sort(components.begin(), components.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.size() != rhs.size()) {
                return lhs.size() > rhs.size();
              }
              return *lhs.begin() < *rhs.begin();
            });
  return components;
}

double BoundingBoxDiagonal(const Reconstruction& reconstruction) {
  if (reconstruction.Points3D().empty()) {
    return 0.0;
  }
  Eigen::Vector3d min_point =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector3d max_point =
      Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
  for (const auto& [point_id, point] : reconstruction.Points3D()) {
    (void)point_id;
    min_point = min_point.cwiseMin(point.xyz);
    max_point = max_point.cwiseMax(point.xyz);
  }
  return (max_point - min_point).norm();
}

std::vector<PointPair> CollectPointPairs(const MatchGraph& graph,
                                         const Reconstruction& target,
                                         const Reconstruction& source) {
  std::map<std::pair<Point3DId, Point3DId>, std::size_t> support;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsUsableGeometry(geometry)) {
      continue;
    }
    const bool forward =
        target.IsImageRegistered(pair.image_id1) &&
        source.IsImageRegistered(pair.image_id2);
    const bool backward =
        target.IsImageRegistered(pair.image_id2) &&
        source.IsImageRegistered(pair.image_id1);
    if (!forward && !backward) {
      continue;
    }
    const Image& target_image =
        target.GetImage(forward ? pair.image_id1 : pair.image_id2);
    const Image& source_image =
        source.GetImage(forward ? pair.image_id2 : pair.image_id1);
    for (const FeatureMatch& match : geometry.inlier_matches) {
      const Point2DIdx target_idx = forward ? match.idx1 : match.idx2;
      const Point2DIdx source_idx = forward ? match.idx2 : match.idx1;
      if (target_idx >= target_image.points2d.size() ||
          source_idx >= source_image.points2d.size()) {
        continue;
      }
      const Point2D& target_point = target_image.points2d[target_idx];
      const Point2D& source_point = source_image.points2d[source_idx];
      if (target_point.HasPoint3D() && source_point.HasPoint3D() &&
          target.ExistsPoint3D(target_point.point3d_id) &&
          source.ExistsPoint3D(source_point.point3d_id)) {
        ++support[{target_point.point3d_id, source_point.point3d_id}];
      }
    }
  }

  std::vector<PointPair> candidates;
  candidates.reserve(support.size());
  for (const auto& [ids, count] : support) {
    candidates.push_back({ids.first, ids.second, count});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const PointPair& lhs, const PointPair& rhs) {
              if (lhs.support != rhs.support) {
                return lhs.support > rhs.support;
              }
              return std::tie(lhs.target_id, lhs.source_id) <
                     std::tie(rhs.target_id, rhs.source_id);
            });

  std::set<Point3DId> used_target;
  std::set<Point3DId> used_source;
  std::vector<PointPair> unique;
  for (const PointPair& pair : candidates) {
    if (used_target.insert(pair.target_id).second &&
        used_source.insert(pair.source_id).second) {
      unique.push_back(pair);
    }
  }
  return unique;
}

SimilarityTransform EstimateSimilarity(
    const std::vector<PointPair>& pairs,
    const std::vector<std::size_t>& indices,
    const Reconstruction& target,
    const Reconstruction& source) {
  SimilarityTransform transform;
  if (indices.size() < 3) {
    return transform;
  }
  Eigen::MatrixXd source_points(3, indices.size());
  Eigen::MatrixXd target_points(3, indices.size());
  for (std::size_t column = 0; column < indices.size(); ++column) {
    const PointPair& pair = pairs[indices[column]];
    source_points.col(column) = source.GetPoint3D(pair.source_id).xyz;
    target_points.col(column) = target.GetPoint3D(pair.target_id).xyz;
  }
  const Eigen::Matrix4d matrix =
      Eigen::umeyama(source_points, target_points, true);
  if (!matrix.allFinite()) {
    return transform;
  }
  const Eigen::Matrix3d scaled_rotation = matrix.block<3, 3>(0, 0);
  const double scale = (scaled_rotation.col(0).norm() +
                        scaled_rotation.col(1).norm() +
                        scaled_rotation.col(2).norm()) /
                       3.0;
  if (!(scale > 1e-12) || !std::isfinite(scale)) {
    return transform;
  }
  Eigen::Matrix3d rotation = scaled_rotation / scale;
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
  rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0) {
    return transform;
  }
  transform.scale = scale;
  transform.rotation = rotation;
  transform.translation = matrix.block<3, 1>(0, 3);
  transform.valid = transform.translation.allFinite();
  return transform;
}

std::vector<std::size_t> SimilarityInliers(
    const SimilarityTransform& transform,
    const std::vector<PointPair>& pairs,
    const Reconstruction& target,
    const Reconstruction& source,
    double threshold) {
  std::vector<std::size_t> inliers;
  if (!transform.valid) {
    return inliers;
  }
  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    const Eigen::Vector3d transformed = transform.TransformPoint(
        source.GetPoint3D(pairs[idx].source_id).xyz);
    if ((transformed - target.GetPoint3D(pairs[idx].target_id).xyz).norm() <=
        threshold) {
      inliers.push_back(idx);
    }
  }
  return inliers;
}

SimilarityTransform EstimateRobustSimilarity(
    const std::vector<PointPair>& pairs,
    const ModelMergeOptions& options,
    const Reconstruction& target,
    const Reconstruction& source,
    double threshold,
    std::vector<std::size_t>* best_inliers) {
  SimilarityTransform best;
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  const std::size_t num_hypotheses =
      std::max<std::size_t>(1, options.max_num_hypotheses);
  for (std::size_t hypothesis = 0; hypothesis < num_hypotheses; ++hypothesis) {
    std::set<std::size_t> selected;
    while (selected.size() < 3) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      selected.insert(static_cast<std::size_t>(state % pairs.size()));
    }
    const std::vector<std::size_t> indices(selected.begin(), selected.end());
    const SimilarityTransform candidate =
        EstimateSimilarity(pairs, indices, target, source);
    std::vector<std::size_t> inliers =
        SimilarityInliers(candidate, pairs, target, source, threshold);
    if (inliers.size() > best_inliers->size()) {
      best = candidate;
      *best_inliers = std::move(inliers);
      if (best_inliers->size() == pairs.size()) {
        break;
      }
    }
  }
  if (best_inliers->size() >= 3) {
    best = EstimateSimilarity(pairs, *best_inliers, target, source);
    *best_inliers =
        SimilarityInliers(best, pairs, target, source, threshold);
  }
  return best;
}

Pose3 TransformPose(const Pose3& source_pose,
                    const SimilarityTransform& transform) {
  Pose3 target_pose;
  const Eigen::Matrix3d rotation =
      source_pose.q_cw.toRotationMatrix() * transform.rotation.transpose();
  target_pose.q_cw = Eigen::Quaterniond(rotation).normalized();
  target_pose.t_cw =
      transform.scale * source_pose.t_cw -
      rotation * transform.translation;
  return target_pose;
}

}  // namespace

ModelMergeSummary MergeReconstructions(
    const MatchGraph& graph,
    const ModelMergeOptions& options,
    const Reconstruction& source,
    Reconstruction* target) {
  ModelMergeSummary summary;
  if (target == nullptr) {
    summary.message = "Target reconstruction is null";
    return summary;
  }
  const std::vector<PointPair> pairs =
      CollectPointPairs(graph, *target, source);
  summary.num_correspondences = pairs.size();
  if (pairs.size() < options.min_num_correspondences) {
    summary.message = "Too few verified 3D-3D correspondences";
    return summary;
  }
  const double diagonal = std::max(1e-6, BoundingBoxDiagonal(*target));
  const double threshold =
      std::max(1e-6, options.max_alignment_error_ratio * diagonal);
  std::vector<std::size_t> inliers;
  const SimilarityTransform transform = EstimateRobustSimilarity(
      pairs, options, *target, source, threshold, &inliers);
  summary.num_inliers = inliers.size();
  if (!transform.valid ||
      inliers.size() < options.min_num_correspondences) {
    summary.message = "Robust Sim(3) alignment failed";
    return summary;
  }
  summary.scale = transform.scale;
  for (const std::size_t idx : inliers) {
    summary.max_alignment_error = std::max(
        summary.max_alignment_error,
        (transform.TransformPoint(source.GetPoint3D(pairs[idx].source_id).xyz) -
         target->GetPoint3D(pairs[idx].target_id).xyz)
            .norm());
  }

  std::map<Point3DId, Point3DId> source_to_target;
  for (const std::size_t idx : inliers) {
    source_to_target[pairs[idx].source_id] = pairs[idx].target_id;
  }

  for (const ImageId image_id : source.RegisteredImageIds()) {
    if (target->IsImageRegistered(image_id)) {
      continue;
    }
    const Image& source_image = source.GetImage(image_id);
    if (!target->ExistsImage(image_id) ||
        !source_image.cam_from_world.has_value()) {
      continue;
    }
    Image& target_image = target->GetImage(image_id);
    target_image.points2d = source_image.points2d;
    for (Point2D& point2d : target_image.points2d) {
      point2d.point3d_id = kInvalidPoint3DId;
    }
    target_image.cam_from_world =
        TransformPose(*source_image.cam_from_world, transform);
    target->RegisterImage(image_id);
    ++summary.num_added_images;
  }

  ObservationManager observation_manager(target);
  for (const auto& [source_point_id, source_point] : source.Points3D()) {
    const auto mapped_it = source_to_target.find(source_point_id);
    if (mapped_it != source_to_target.end()) {
      for (const TrackElement& element : source_point.track.elements) {
        if (target->IsImageRegistered(element.image_id)) {
          observation_manager.AddObservation(
              mapped_it->second, element.image_id, element.point2d_idx);
        }
      }
      ++summary.num_merged_points;
      continue;
    }

    Track track;
    std::set<ImageId> track_images;
    for (const TrackElement& element : source_point.track.elements) {
      if (!target->IsImageRegistered(element.image_id) ||
          !track_images.insert(element.image_id).second) {
        continue;
      }
      const Image& image = target->GetImage(element.image_id);
      if (element.point2d_idx < image.points2d.size() &&
          !image.points2d[element.point2d_idx].HasPoint3D()) {
        track.AddElement(element.image_id, element.point2d_idx);
      }
    }
    if (track.Length() < 2) {
      continue;
    }
    const std::optional<Point3DId> point_id =
        observation_manager.AddPoint3D(
            transform.TransformPoint(source_point.xyz), track);
    if (point_id.has_value()) {
      target->GetPoint3D(*point_id).color = source_point.color;
      ++summary.num_added_points;
    }
  }

  summary.success = summary.num_added_images > 0;
  summary.message = summary.success ? "Reconstructions merged"
                                    : "No new registered images to merge";
  return summary;
}

MultiModelMappingResult RunIncrementalMappingMultiple(
    const MatchGraph& graph,
    const MultiModelMappingOptions& options,
    Reconstruction schema_or_seed) {
  MultiModelMappingResult result;
  const Reconstruction schema = CloneSchema(schema_or_seed);
  std::set<ImageId> consumed;

  if (!schema_or_seed.RegisteredImageIds().empty()) {
    ValidateAndCompleteSeedFeatures(graph, &schema_or_seed);
    IncrementalMappingOptions mapping = options.mapping;
    mapping.allowed_image_ids.clear();
    IncrementalMappingSummary summary =
        RunIncrementalMapping(graph, mapping, &schema_or_seed);
    if (summary.success) {
      consumed.insert(schema_or_seed.RegisteredImageIds().begin(),
                      schema_or_seed.RegisteredImageIds().end());
      result.models.push_back(std::move(schema_or_seed));
      result.mapping_summaries.push_back(std::move(summary));
    }
  }

  while (result.models.size() < options.max_num_models) {
    std::set<ImageId> remaining;
    for (const auto& [image_id, image] : schema.Images()) {
      (void)image;
      if (consumed.count(image_id) == 0) {
        remaining.insert(image_id);
      }
    }
    bool made_progress = false;
    for (const std::set<ImageId>& component :
         VerifiedComponents(graph, remaining)) {
      if (component.size() < options.min_num_images_per_model ||
          result.models.size() >= options.max_num_models) {
        continue;
      }
      Reconstruction model = CloneSchema(schema);
      IncrementalMappingOptions mapping = options.mapping;
      mapping.allowed_image_ids = component;
      mapping.initialization.allowed_image_ids = component;
      IncrementalMappingSummary summary =
          RunIncrementalMapping(graph, mapping, &model);
      if (!summary.success ||
          summary.num_registered_images < options.min_num_images_per_model) {
        continue;
      }
      consumed.insert(model.RegisteredImageIds().begin(),
                      model.RegisteredImageIds().end());
      result.models.push_back(std::move(model));
      result.mapping_summaries.push_back(std::move(summary));
      made_progress = true;
    }
    if (!made_progress) {
      break;
    }
  }

  if (options.merge_models && result.models.size() > 1) {
    bool merged_any = true;
    while (merged_any) {
      merged_any = false;
      for (std::size_t target_idx = 0;
           target_idx < result.models.size() && !merged_any; ++target_idx) {
        for (std::size_t source_idx = target_idx + 1;
             source_idx < result.models.size(); ++source_idx) {
          ModelMergeSummary merge = MergeReconstructions(
              graph, options.merging, result.models[source_idx],
              &result.models[target_idx]);
          result.merge_summaries.push_back(merge);
          if (!merge.success) {
            continue;
          }
          IncrementalMappingOptions refinement = options.mapping;
          refinement.allowed_image_ids.clear();
          result.mapping_summaries[target_idx] =
              RunIncrementalMapping(graph, refinement,
                                    &result.models[target_idx]);
          result.models.erase(result.models.begin() + source_idx);
          result.mapping_summaries.erase(
              result.mapping_summaries.begin() + source_idx);
          merged_any = true;
          break;
        }
      }
    }
  }

  std::set<ImageId> registered;
  for (const Reconstruction& model : result.models) {
    registered.insert(model.RegisteredImageIds().begin(),
                      model.RegisteredImageIds().end());
  }
  result.num_unregistered_images =
      schema.Images().size() - registered.size();
  return result;
}

}  // namespace colite
