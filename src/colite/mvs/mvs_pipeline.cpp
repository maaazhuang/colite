#include "colite/mvs/mvs_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

bool ImageUsableForMvs(const Reconstruction& reconstruction, ImageId image_id) {
  if (!reconstruction.ExistsImage(image_id) ||
      !reconstruction.IsImageRegistered(image_id)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(image_id);
  return image.HasPose() && reconstruction.ExistsCamera(image.camera_id);
}

struct SourceImageAccumulator {
  std::size_t num_shared_points = 0;
  double angle_sum_degrees = 0.0;
  std::size_t num_scored_angles = 0;
  std::vector<double> triangulation_angles_degrees;
  double scale_ratio_sum = 0.0;
  std::size_t num_scored_scale_ratios = 0;
  std::unordered_set<std::size_t> occupied_target_grid_cells;
};

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  const double index =
      percentile / 100.0 * static_cast<double>(values.size() - 1);
  const std::size_t left_index =
      static_cast<std::size_t>(std::floor(index));
  const std::size_t right_index =
      static_cast<std::size_t>(std::ceil(index));
  std::nth_element(values.begin(), values.begin() + right_index,
                   values.end());
  const double right = values[right_index];
  if (left_index == right_index) {
    return right;
  }
  const double left =
      *std::max_element(values.begin(), values.begin() + right_index);
  return (static_cast<double>(right_index) - index) * left +
         (index - static_cast<double>(left_index)) * right;
}

double AngleDegreesBetween(const Eigen::Vector3d& lhs,
                           const Eigen::Vector3d& rhs) {
  if (!lhs.allFinite() || !rhs.allFinite() || lhs.squaredNorm() <= 0.0 ||
      rhs.squaredNorm() <= 0.0) {
    return 0.0;
  }
  const double cosine =
      std::max(-1.0, std::min(1.0, lhs.normalized().dot(rhs.normalized())));
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  return std::acos(cosine) * kRadiansToDegrees;
}

double ComputeTrackTriangulationAngleDegrees(
    const Reconstruction& reconstruction,
    ImageId lhs_image_id,
    ImageId rhs_image_id,
    const Eigen::Vector3d& point_world) {
  const Image& lhs_image = reconstruction.GetImage(lhs_image_id);
  const Image& rhs_image = reconstruction.GetImage(rhs_image_id);
  if (!lhs_image.cam_from_world.has_value() ||
      !rhs_image.cam_from_world.has_value()) {
    return 0.0;
  }

  const Eigen::Vector3d lhs_ray =
      point_world - lhs_image.cam_from_world->CameraCenter();
  const Eigen::Vector3d rhs_ray =
      point_world - rhs_image.cam_from_world->CameraCenter();
  return AngleDegreesBetween(lhs_ray, rhs_ray);
}

const Point2D* FindTrackPoint2D(const Reconstruction& reconstruction,
                                ImageId image_id,
                                const Track& track) {
  if (!reconstruction.ExistsImage(image_id)) {
    return nullptr;
  }
  const Image& image = reconstruction.GetImage(image_id);
  for (const TrackElement& element : track.elements) {
    if (element.image_id != image_id ||
        element.point2d_idx >= image.points2d.size()) {
      continue;
    }
    return &image.points2d[element.point2d_idx];
  }
  return nullptr;
}

double ImageFootprintScale(const Reconstruction& reconstruction,
                           ImageId image_id,
                           const Eigen::Vector3d& point_world) {
  const Image& image = reconstruction.GetImage(image_id);
  if (!image.cam_from_world.has_value() ||
      !reconstruction.ExistsCamera(image.camera_id)) {
    return 0.0;
  }
  const Eigen::Vector3d point_camera =
      image.cam_from_world->TransformPoint(point_world);
  if (point_camera.z() <= 0.0) {
    return 0.0;
  }
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  return camera.MeanFocalLength() / point_camera.z();
}

double ComputeTrackScaleRatio(const Reconstruction& reconstruction,
                              ImageId target_image_id,
                              ImageId source_image_id,
                              const Eigen::Vector3d& point_world) {
  const double target_scale =
      ImageFootprintScale(reconstruction, target_image_id, point_world);
  const double source_scale =
      ImageFootprintScale(reconstruction, source_image_id, point_world);
  if (target_scale <= 0.0 || source_scale <= 0.0) {
    return 0.0;
  }
  return std::min(target_scale, source_scale) /
         std::max(target_scale, source_scale);
}

std::size_t TargetCoverageCell(const Camera& camera,
                               const Point2D& point2d,
                               std::size_t grid_size) {
  if (grid_size == 0 || camera.width == 0 || camera.height == 0) {
    return 0;
  }
  const double normalized_x =
      std::max(0.0, std::min(0.999999, point2d.xy.x() /
                                           static_cast<double>(camera.width)));
  const double normalized_y =
      std::max(0.0, std::min(0.999999, point2d.xy.y() /
                                           static_cast<double>(camera.height)));
  const std::size_t col =
      std::min(grid_size - 1,
               static_cast<std::size_t>(std::floor(normalized_x *
                                                   static_cast<double>(grid_size))));
  const std::size_t row =
      std::min(grid_size - 1,
               static_cast<std::size_t>(std::floor(normalized_y *
                                                   static_cast<double>(grid_size))));
  return row * grid_size + col;
}

double ScoreSourceView(double mean_angle_degrees,
                       std::size_t num_shared_points,
                       double mean_scale_ratio,
                       double coverage_score,
                       const MvsPipelineOptions& options) {
  return options.triangulation_angle_weight * mean_angle_degrees +
         options.shared_points_weight *
             static_cast<double>(num_shared_points) +
         options.scale_consistency_weight * mean_scale_ratio +
         options.coverage_weight * coverage_score;
}

}  // namespace

std::vector<SourceImageScore> SelectSourceImages(
    const Reconstruction& reconstruction,
    ImageId target_image_id,
    const MvsPipelineOptions& options) {
  if (!ImageUsableForMvs(reconstruction, target_image_id)) {
    return {};
  }
  const Image& target_image = reconstruction.GetImage(target_image_id);
  const Camera& target_camera = reconstruction.GetCamera(target_image.camera_id);

  std::unordered_map<ImageId, SourceImageAccumulator> source_accumulators;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    bool observes_target = false;
    std::unordered_set<ImageId> source_images;
    for (const TrackElement& element : point3d.track.elements) {
      if (element.image_id == target_image_id) {
        observes_target = true;
      } else if (ImageUsableForMvs(reconstruction, element.image_id)) {
        source_images.insert(element.image_id);
      }
    }
    if (!observes_target) {
      continue;
    }
    for (const ImageId source_image_id : source_images) {
      SourceImageAccumulator& accumulator =
          source_accumulators[source_image_id];
      ++accumulator.num_shared_points;

      const double angle_degrees = ComputeTrackTriangulationAngleDegrees(
          reconstruction, target_image_id, source_image_id, point3d.xyz);
      if (std::isfinite(angle_degrees)) {
        accumulator.triangulation_angles_degrees.push_back(angle_degrees);
      }
      if (angle_degrees >= options.min_triangulation_angle_degrees &&
          angle_degrees <= options.max_triangulation_angle_degrees) {
        accumulator.angle_sum_degrees += angle_degrees;
        ++accumulator.num_scored_angles;
      }

      const double scale_ratio = ComputeTrackScaleRatio(
          reconstruction, target_image_id, source_image_id, point3d.xyz);
      if (scale_ratio > 0.0 && std::isfinite(scale_ratio)) {
        accumulator.scale_ratio_sum += scale_ratio;
        ++accumulator.num_scored_scale_ratios;
      }

      const Point2D* target_point2d =
          FindTrackPoint2D(reconstruction, target_image_id, point3d.track);
      if (target_point2d != nullptr && options.coverage_grid_size > 0) {
        accumulator.occupied_target_grid_cells.insert(TargetCoverageCell(
            target_camera, *target_point2d, options.coverage_grid_size));
      }
    }
  }

  std::vector<SourceImageScore> scores;
  scores.reserve(source_accumulators.size());
  for (const auto& [image_id, accumulator] : source_accumulators) {
    const double angle_percentile =
        Percentile(accumulator.triangulation_angles_degrees,
                   options.triangulation_angle_percentile);
    const bool passes_colmap_angle_gate =
        options.source_selection_mode != SourceSelectionMode::kColmap ||
        angle_percentile >= options.min_triangulation_angle_degrees;
    if (accumulator.num_shared_points >= options.min_num_shared_points &&
        passes_colmap_angle_gate) {
      SourceImageScore score;
      score.image_id = image_id;
      score.num_shared_points = accumulator.num_shared_points;
      score.triangulation_angle_percentile_degrees = angle_percentile;
      if (accumulator.num_scored_angles > 0) {
        score.mean_triangulation_angle_degrees =
            accumulator.angle_sum_degrees /
            static_cast<double>(accumulator.num_scored_angles);
      }
      if (accumulator.num_scored_scale_ratios > 0) {
        score.mean_scale_ratio =
            accumulator.scale_ratio_sum /
            static_cast<double>(accumulator.num_scored_scale_ratios);
      }
      if (options.coverage_grid_size > 0) {
        score.coverage_score =
            static_cast<double>(accumulator.occupied_target_grid_cells.size()) /
            static_cast<double>(options.coverage_grid_size *
                                options.coverage_grid_size);
      }
      if (options.source_selection_mode == SourceSelectionMode::kColmap) {
        score.score = static_cast<double>(score.num_shared_points);
      } else {
        score.score =
            ScoreSourceView(score.mean_triangulation_angle_degrees,
                            score.num_shared_points, score.mean_scale_ratio,
                            score.coverage_score, options);
      }
      scores.push_back(score);
    }
  }
  std::sort(scores.begin(), scores.end(),
            [](const SourceImageScore& lhs, const SourceImageScore& rhs) {
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              if (lhs.num_shared_points != rhs.num_shared_points) {
                return lhs.num_shared_points > rhs.num_shared_points;
              }
              return lhs.image_id < rhs.image_id;
            });
  if (options.max_num_source_images > 0 &&
      scores.size() > options.max_num_source_images) {
    scores.resize(options.max_num_source_images);
  }
  return scores;
}

SparseDepthMap BuildSparseDepthMap(const Reconstruction& reconstruction,
                                   ImageId image_id) {
  SparseDepthMap depth_map;
  depth_map.image_id = image_id;
  if (!ImageUsableForMvs(reconstruction, image_id)) {
    return depth_map;
  }

  const Image& image = reconstruction.GetImage(image_id);
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  depth_map.width = camera.width;
  depth_map.height = camera.height;
  depth_map.samples.reserve(image.NumPoints3D());

  for (const Point2D& point2d : image.points2d) {
    if (!point2d.HasPoint3D() ||
        !reconstruction.ExistsPoint3D(point2d.point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction.GetPoint3D(point2d.point3d_id);
    const Eigen::Vector3d point_camera =
        image.cam_from_world->TransformPoint(point3d.xyz);
    if (point_camera.z() <= 0.0) {
      continue;
    }
    depth_map.samples.push_back({point2d.point3d_id,
                                 point2d.xy.x(),
                                 point2d.xy.y(),
                                 point_camera.z()});
  }

  return depth_map;
}

MvsPipelineSummary RunMvsPipeline(const Reconstruction& reconstruction,
                                  const MvsPipelineOptions& options) {
  MvsPipelineSummary summary;
  summary.images.reserve(reconstruction.RegisteredImageIds().size());

  for (const ImageId image_id : reconstruction.RegisteredImageIds()) {
    if (!ImageUsableForMvs(reconstruction, image_id)) {
      continue;
    }

    ImageMvsSummary image_summary;
    image_summary.image_id = image_id;
    image_summary.source_images =
        SelectSourceImages(reconstruction, image_id, options);
    image_summary.sparse_depth_map =
        BuildSparseDepthMap(reconstruction, image_id);
    summary.num_sparse_depth_samples +=
        image_summary.sparse_depth_map.samples.size();

    std::vector<double> depths;
    depths.reserve(image_summary.sparse_depth_map.samples.size());
    for (const SparseDepthSample& sample :
         image_summary.sparse_depth_map.samples) {
      if (std::isfinite(sample.depth) && sample.depth > 0.0) {
        depths.push_back(sample.depth);
      }
    }
    if (!depths.empty()) {
      // Match COLMAP's automatic per-view PatchMatch search interval:
      // discard the sparse-depth tails and stretch the robust interval by
      // 25%. Using the raw extrema lets a single weak sparse point expand the
      // inverse-depth search by an order of magnitude. Low-parallax
      // hypotheses near that inflated far plane can then appear mutually
      // consistent and survive fusion as distant sheets.
      std::sort(depths.begin(), depths.end());
      constexpr double kMinPercentile = 0.01;
      constexpr double kMaxPercentile = 0.99;
      constexpr double kStretchRatio = 0.25;
      const std::size_t min_index = static_cast<std::size_t>(
          static_cast<double>(depths.size()) * kMinPercentile);
      const std::size_t max_index = std::min(
          depths.size() - 1,
          static_cast<std::size_t>(
              static_cast<double>(depths.size()) * kMaxPercentile));
      image_summary.min_depth =
          depths[min_index] * (1.0 - kStretchRatio);
      image_summary.max_depth =
          depths[max_index] * (1.0 + kStretchRatio);
    }

    summary.images.push_back(std::move(image_summary));
  }

  summary.num_images = summary.images.size();
  summary.success = summary.num_images > 0;
  return summary;
}

}  // namespace colite::mvs
