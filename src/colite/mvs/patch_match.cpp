#include "colite/mvs/patch_match.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

constexpr char kConsistencyGraphBinaryMagic[] =
    "COLITE_CONSISTENCY_GRAPH_BINARY";
constexpr int kConsistencyGraphBinaryVersion = 1;

void RequireValidDepthRange(float depth_min, float depth_max) {
  if (!std::isfinite(depth_min) || !std::isfinite(depth_max) ||
      depth_min <= 0.0F || depth_max < depth_min) {
    throw std::invalid_argument("PatchMatch depth range is invalid.");
  }
}

void RequireConsistencyGraphStream(bool condition,
                                   const std::filesystem::path& path) {
  if (!condition) {
    throw std::runtime_error("Invalid binary consistency graph file: " +
                             path.string());
  }
}

std::pair<float, float> DepthRangeFromSparseMap(
    const SparseDepthMap& sparse_depth_map,
    const PatchMatchInitializationOptions& options) {
  if (!std::isfinite(options.sparse_depth_min_percentile) ||
      !std::isfinite(options.sparse_depth_max_percentile) ||
      !std::isfinite(options.depth_range_padding) ||
      options.sparse_depth_min_percentile < 0.0F ||
      options.sparse_depth_max_percentile > 1.0F ||
      options.sparse_depth_max_percentile <
          options.sparse_depth_min_percentile ||
      options.depth_range_padding < 0.0F) {
    throw std::invalid_argument(
        "PatchMatch sparse depth percentile options are invalid.");
  }

  std::vector<float> depths;
  depths.reserve(sparse_depth_map.samples.size());
  for (const SparseDepthSample& sample : sparse_depth_map.samples) {
    const float depth = static_cast<float>(sample.depth);
    if (!std::isfinite(depth) || depth <= 0.0F) {
      continue;
    }
    depths.push_back(depth);
  }
  if (depths.empty()) {
    throw std::invalid_argument(
        "Sparse depth map has no valid positive depth samples.");
  }
  std::sort(depths.begin(), depths.end());
  const auto percentile_index = [&depths](float percentile) {
    const float clamped = std::max(0.0F, std::min(1.0F, percentile));
    return static_cast<std::size_t>(std::lround(
        clamped * static_cast<float>(depths.size() - 1)));
  };
  float depth_min = depths[percentile_index(options.sparse_depth_min_percentile)];
  float depth_max = depths[percentile_index(options.sparse_depth_max_percentile)];
  if (depth_max < depth_min) {
    std::swap(depth_min, depth_max);
  }
  const float depth_range = depth_max - depth_min;
  depth_min =
      std::max(1e-6F, depth_min - depth_range * options.depth_range_padding);
  depth_max = depth_max + depth_range * options.depth_range_padding;
  return {depth_min, depth_max};
}

PatchMatchCost InvalidCost(const PatchMatchCostOptions& options) {
  PatchMatchCost result;
  result.valid = false;
  result.score = -1.0F;
  result.cost = options.invalid_cost;
  return result;
}

PatchMatchCost ComputeZnccFromSamples(const std::vector<float>& reference_values,
                                      const std::vector<float>& source_values,
                                      const PatchMatchCostOptions& options) {
  if (reference_values.empty() ||
      reference_values.size() != source_values.size()) {
    return InvalidCost(options);
  }

  double reference_mean = 0.0;
  double source_mean = 0.0;
  for (std::size_t i = 0; i < reference_values.size(); ++i) {
    reference_mean += reference_values[i];
    source_mean += source_values[i];
  }
  reference_mean /= static_cast<double>(reference_values.size());
  source_mean /= static_cast<double>(source_values.size());

  double numerator = 0.0;
  double reference_norm2 = 0.0;
  double source_norm2 = 0.0;
  for (std::size_t i = 0; i < reference_values.size(); ++i) {
    const double ref_centered = reference_values[i] - reference_mean;
    const double src_centered = source_values[i] - source_mean;
    numerator += ref_centered * src_centered;
    reference_norm2 += ref_centered * ref_centered;
    source_norm2 += src_centered * src_centered;
  }

  if (reference_norm2 <= options.min_variance ||
      source_norm2 <= options.min_variance) {
    return InvalidCost(options);
  }

  const double denominator = std::sqrt(reference_norm2 * source_norm2);
  const float score = static_cast<float>(
      std::max(-1.0, std::min(1.0, numerator / denominator)));

  PatchMatchCost result;
  result.valid = true;
  result.score = score;
  result.cost = 0.5F * (1.0F - score);
  result.num_samples = reference_values.size();
  return result;
}

void RequireCompatiblePatchMatchMaps(const MvsImage& reference_image,
                                     const DenseDepthMap& depth_map,
                                     const NormalMap& normal_map,
                                     const DenseCostMap& cost_map) {
  if (depth_map.Width() != normal_map.Width() ||
      depth_map.Height() != normal_map.Height() ||
      depth_map.Width() != cost_map.Width() ||
      depth_map.Height() != cost_map.Height()) {
    throw std::invalid_argument(
        "Depth, normal and cost maps must have identical sizes.");
  }
  if (depth_map.Width() != reference_image.Width() ||
      depth_map.Height() != reference_image.Height()) {
    throw std::invalid_argument(
        "Reference image and PatchMatch maps must have identical sizes.");
  }
}

void RequireCompatibleFusionMaps(const MvsImage& reference_image,
                                 const DenseDepthMap& depth_map,
                                 const NormalMap& normal_map,
                                 const DenseCostMap& cost_map,
                                 const ConsistencyGraph* graph) {
  RequireCompatiblePatchMatchMaps(reference_image, depth_map, normal_map,
                                  cost_map);
  if (graph != nullptr &&
      (graph->Width() != depth_map.Width() ||
       graph->Height() != depth_map.Height())) {
    throw std::invalid_argument(
        "Consistency graph and PatchMatch maps must have identical sizes.");
  }
}

float ClampFloat(float value, float min_value, float max_value) {
  return std::max(min_value, std::min(max_value, value));
}

Eigen::Vector3f PerturbNormal(const Eigen::Vector3f& normal,
                              float radius_degrees,
                              std::mt19937* rng) {
  if (rng == nullptr || radius_degrees <= 0.0F || !normal.allFinite() ||
      normal.squaredNorm() <= 0.0F) {
    return normal;
  }

  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
  Eigen::Vector3f delta(distribution(*rng), distribution(*rng),
                        distribution(*rng));
  if (delta.squaredNorm() <= 0.0F) {
    return normal.normalized();
  }

  constexpr float kPi = 3.14159265358979323846F;
  const float radius_radians = radius_degrees * kPi / 180.0F;
  const Eigen::Vector3f candidate =
      normal.normalized() + delta.normalized() * std::sin(radius_radians);
  if (!candidate.allFinite() || candidate.squaredNorm() <= 0.0F) {
    return normal.normalized();
  }
  return candidate.normalized();
}

Eigen::Vector3f SampleNormalInHemisphere(const Eigen::Vector3f& axis,
                                         float max_angle_degrees,
                                         std::mt19937* rng) {
  if (rng == nullptr || !axis.allFinite() || axis.squaredNorm() <= 0.0F) {
    return Eigen::Vector3f(0.0F, 0.0F, -1.0F);
  }
  if (!std::isfinite(max_angle_degrees) || max_angle_degrees < 0.0F ||
      max_angle_degrees > 180.0F) {
    throw std::invalid_argument(
        "PatchMatch initial normal hemisphere angle is invalid.");
  }

  const Eigen::Vector3f normal_axis = axis.normalized();
  if (max_angle_degrees <= 0.0F) {
    return normal_axis;
  }

  constexpr float kPi = 3.14159265358979323846F;
  const float max_angle_radians = max_angle_degrees * kPi / 180.0F;
  const float min_cosine = std::cos(max_angle_radians);
  std::uniform_real_distribution<float> cosine_distribution(min_cosine, 1.0F);
  std::uniform_real_distribution<float> angle_distribution(0.0F, 2.0F * kPi);

  const Eigen::Vector3f helper =
      std::abs(normal_axis.z()) < 0.9F ? Eigen::Vector3f::UnitZ()
                                       : Eigen::Vector3f::UnitX();
  const Eigen::Vector3f tangent_u = normal_axis.cross(helper).normalized();
  const Eigen::Vector3f tangent_v = normal_axis.cross(tangent_u).normalized();

  const float cosine = cosine_distribution(*rng);
  const float sine = std::sqrt(std::max(0.0F, 1.0F - cosine * cosine));
  const float phi = angle_distribution(*rng);
  const Eigen::Vector3f sampled =
      cosine * normal_axis +
      sine * (std::cos(phi) * tangent_u + std::sin(phi) * tangent_v);
  if (!sampled.allFinite() || sampled.squaredNorm() <= 0.0F) {
    return normal_axis;
  }
  return sampled.normalized();
}

std::optional<float> SampleDepthBilinear(const DenseDepthMap& depth_map,
                                         float x,
                                         float y) {
  if (depth_map.Empty() || !std::isfinite(x) || !std::isfinite(y)) {
    return std::nullopt;
  }
  if (x < 0.0F || y < 0.0F ||
      x > static_cast<float>(depth_map.Width() - 1) ||
      y > static_cast<float>(depth_map.Height() - 1)) {
    return std::nullopt;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 =
      std::min(x0 + 1, static_cast<int>(depth_map.Width()) - 1);
  const int y1 =
      std::min(y0 + 1, static_cast<int>(depth_map.Height()) - 1);
  const std::size_t sx0 = static_cast<std::size_t>(x0);
  const std::size_t sx1 = static_cast<std::size_t>(x1);
  const std::size_t sy0 = static_cast<std::size_t>(y0);
  const std::size_t sy1 = static_cast<std::size_t>(y1);
  if (!depth_map.IsValid(sy0, sx0) || !depth_map.IsValid(sy0, sx1) ||
      !depth_map.IsValid(sy1, sx0) || !depth_map.IsValid(sy1, sx1)) {
    return std::nullopt;
  }

  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);
  const float v00 = depth_map.Get(sy0, sx0);
  const float v01 = depth_map.Get(sy0, sx1);
  const float v10 = depth_map.Get(sy1, sx0);
  const float v11 = depth_map.Get(sy1, sx1);
  const float top = (1.0F - dx) * v00 + dx * v01;
  const float bottom = (1.0F - dx) * v10 + dx * v11;
  return (1.0F - dy) * top + dy * bottom;
}

std::optional<float> ProjectPlaneDepthToPixel(const Camera& camera,
                                              const Eigen::Vector2d& plane_xy,
                                              float plane_depth,
                                              const Eigen::Vector3f& normal,
                                              const Eigen::Vector2d& target_xy) {
  if (plane_depth <= 0.0F || !std::isfinite(plane_depth) ||
      !normal.allFinite() || normal.squaredNorm() <= 0.0F) {
    return std::nullopt;
  }
  try {
    const Eigen::Vector3d plane_point =
        UnprojectPixel(camera, plane_xy, static_cast<double>(plane_depth));
    const Eigen::Vector3d target_ray =
        UnprojectPixel(camera, target_xy, 1.0);
    const Eigen::Vector3d normal_d = normal.cast<double>().normalized();
    const double denominator = normal_d.dot(target_ray);
    if (std::abs(denominator) <= 1e-12) {
      return std::nullopt;
    }
    const double depth = normal_d.dot(plane_point) / denominator;
    if (!std::isfinite(depth) || depth <= 0.0) {
      return std::nullopt;
    }
    return static_cast<float>(depth);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

Camera ScaleCameraForPyramidLevel(const Camera& camera,
                                  double scale_x,
                                  double scale_y,
                                  std::size_t width,
                                  std::size_t height) {
  Camera scaled = camera;
  scaled.width = static_cast<std::uint32_t>(width);
  scaled.height = static_cast<std::uint32_t>(height);
  const CameraModelSpec& spec = GetCameraModelSpec(scaled.model_id);
  if (spec.focal_length_idxs.size() >= 2) {
    const std::size_t fx_idx = spec.focal_length_idxs[0];
    const std::size_t fy_idx = spec.focal_length_idxs[1];
    if (fx_idx < scaled.params.size()) {
      scaled.params[fx_idx] *= scale_x;
    }
    if (fy_idx < scaled.params.size()) {
      scaled.params[fy_idx] *= scale_y;
    }
  } else {
    for (const std::size_t idx : spec.focal_length_idxs) {
      if (idx >= scaled.params.size()) {
        continue;
      }
      // 单焦距模型只能存一个焦距，使用 x/y 缩放均值近似各向同性缩放。
      scaled.params[idx] *= 0.5 * (scale_x + scale_y);
    }
  }
  if (spec.principal_point_idxs.size() >= 1 &&
      spec.principal_point_idxs[0] < scaled.params.size()) {
    scaled.params[spec.principal_point_idxs[0]] *= scale_x;
  }
  if (spec.principal_point_idxs.size() >= 2 &&
      spec.principal_point_idxs[1] < scaled.params.size()) {
    scaled.params[spec.principal_point_idxs[1]] *= scale_y;
  }
  return scaled;
}

PatchMatchView ScaleViewForPyramidLevel(const PatchMatchView& view,
                                        double scale_x,
                                        double scale_y,
                                        std::size_t width,
                                        std::size_t height) {
  PatchMatchView scaled = view;
  scaled.camera =
      ScaleCameraForPyramidLevel(view.camera, scale_x, scale_y, width, height);
  return scaled;
}

std::size_t PyramidLevelDivisor(std::size_t num_levels,
                                std::size_t coarse_to_fine_idx) {
  std::size_t divisor = 1;
  for (std::size_t i = coarse_to_fine_idx + 1; i < num_levels; ++i) {
    divisor *= 2;
  }
  return divisor;
}

std::size_t PyramidLevelSize(std::size_t fine_size, std::size_t divisor) {
  return std::max<std::size_t>(
      1, static_cast<std::size_t>(std::lround(
             static_cast<double>(fine_size) / static_cast<double>(divisor))));
}

MvsImage ResizeImageForPyramidLevel(const MvsImage& image,
                                    std::size_t width,
                                    std::size_t height) {
  MvsImage resized = image;
  if (image.Width() == width && image.Height() == height) {
    return resized;
  }
  cv::resize(image.gray, resized.gray,
             cv::Size(static_cast<int>(width), static_cast<int>(height)), 0.0,
             0.0, cv::INTER_AREA);
  if (image.Width() > 0) {
    resized.scale =
        image.scale * static_cast<double>(width) /
        static_cast<double>(image.Width());
  }
  return resized;
}

DenseDepthMap ResizeDepthMapNearest(const DenseDepthMap& source,
                                    std::size_t width,
                                    std::size_t height) {
  DenseDepthMap resized(width, height, source.DepthMin(), source.DepthMax());
  if (source.Empty() || width == 0 || height == 0) {
    return resized;
  }
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t source_row = std::min(
        source.Height() - 1,
        static_cast<std::size_t>(
            std::floor((static_cast<double>(row) + 0.5) *
                       static_cast<double>(source.Height()) /
                       static_cast<double>(height))));
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t source_col = std::min(
          source.Width() - 1,
          static_cast<std::size_t>(
              std::floor((static_cast<double>(col) + 0.5) *
                         static_cast<double>(source.Width()) /
                         static_cast<double>(width))));
      resized.Set(row, col, source.Get(source_row, source_col));
    }
  }
  resized.RecomputeDepthRange();
  if (resized.DepthMin() <= 0.0F || resized.DepthMax() < resized.DepthMin()) {
    resized.SetDepthRange(source.DepthMin(), source.DepthMax());
  }
  return resized;
}

NormalMap ResizeNormalMapNearest(const NormalMap& source,
                                 std::size_t width,
                                 std::size_t height) {
  NormalMap resized(width, height);
  if (source.Empty() || width == 0 || height == 0) {
    return resized;
  }
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t source_row = std::min(
        source.Height() - 1,
        static_cast<std::size_t>(
            std::floor((static_cast<double>(row) + 0.5) *
                       static_cast<double>(source.Height()) /
                       static_cast<double>(height))));
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t source_col = std::min(
          source.Width() - 1,
          static_cast<std::size_t>(
              std::floor((static_cast<double>(col) + 0.5) *
                         static_cast<double>(source.Width()) /
                         static_cast<double>(width))));
      resized.SetNormalized(row, col, source.Get(source_row, source_col));
    }
  }
  return resized;
}

PatchMatchMaps ResizePatchMatchMapsNearest(const PatchMatchMaps& source,
                                           std::size_t width,
                                           std::size_t height) {
  PatchMatchMaps resized;
  resized.depth_map = ResizeDepthMapNearest(source.depth_map, width, height);
  resized.normal_map = ResizeNormalMapNearest(source.normal_map, width, height);
  return resized;
}

struct FusionObservation {
  std::size_t image_idx = 0;
  std::size_t row = 0;
  std::size_t col = 0;
  std::size_t traversal_depth = 0;
  double normal_angle_degrees = 0.0;
  double view_angle_degrees = 0.0;
};

struct FusionPixelData {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Vector3d color = Eigen::Vector3d::Zero();
};

enum class FusionRejectReason {
  kNone,
  kProjection,
  kDepth,
  kMutualConsistency,
  kInvalid,
  kNormal,
  kView,
};

int FusionRejectReasonPriority(FusionRejectReason reason) {
  switch (reason) {
    case FusionRejectReason::kNone:
      return 0;
    case FusionRejectReason::kProjection:
      return 1;
    case FusionRejectReason::kInvalid:
      return 2;
    case FusionRejectReason::kMutualConsistency:
      return 3;
    case FusionRejectReason::kDepth:
      return 4;
    case FusionRejectReason::kNormal:
    case FusionRejectReason::kView:
      return 5;
  }
  return 0;
}

void KeepMoreSpecificRejectReason(FusionRejectReason reason,
                                  FusionRejectReason* current) {
  if (current == nullptr) {
    return;
  }
  if (FusionRejectReasonPriority(reason) >
      FusionRejectReasonPriority(*current)) {
    *current = reason;
  }
}

void AddFusionErrorSample(float value, std::vector<float>* samples) {
  if (samples == nullptr || !std::isfinite(value)) {
    return;
  }
  samples->push_back(value);
}

bool ContainsImageId(const std::vector<ImageId>& image_ids,
                     ImageId image_id) {
  return std::find(image_ids.begin(), image_ids.end(), image_id) !=
         image_ids.end();
}

bool ContainsObservation(const std::vector<FusionObservation>& observations,
                         std::size_t image_idx,
                         std::size_t row,
                         std::size_t col) {
  return std::any_of(observations.begin(), observations.end(),
                     [image_idx, row, col](const FusionObservation& obs) {
                       return obs.image_idx == image_idx && obs.row == row &&
                              obs.col == col;
                     });
}

double AngleDegreesBetween(const Eigen::Vector3d& lhs,
                           const Eigen::Vector3d& rhs) {
  if (!lhs.allFinite() || !rhs.allFinite() || lhs.squaredNorm() <= 0.0 ||
      rhs.squaredNorm() <= 0.0) {
    return 180.0;
  }
  const double cosine =
      std::max(-1.0, std::min(1.0, lhs.normalized().dot(rhs.normalized())));
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  return std::acos(cosine) * kRadiansToDegrees;
}

double ViewAngleDegrees(const PatchMatchView& reference_view,
                        const PatchMatchView& source_view,
                        const Eigen::Vector3d& point_world) {
  const Eigen::Vector3d reference_ray =
      point_world - reference_view.cam_from_world.CameraCenter();
  const Eigen::Vector3d source_ray =
      point_world - source_view.cam_from_world.CameraCenter();
  return AngleDegreesBetween(reference_ray, source_ray);
}

bool IsCompleteFusionPixel(const std::vector<MvsImage>& images,
                           const std::vector<DenseDepthMap>& depth_maps,
                           const std::vector<NormalMap>& normal_maps,
                           const std::vector<DenseCostMap>& cost_maps,
                           const std::vector<ConsistencyGraph>& graphs,
                           const StereoFusionOptions& options,
                           std::size_t image_idx,
                           std::size_t row,
                           std::size_t col) {
  if (image_idx >= depth_maps.size() ||
      !depth_maps[image_idx].IsValid(row, col) ||
      !normal_maps[image_idx].IsValid(row, col) ||
      !cost_maps[image_idx].IsValid(row, col)) {
    return false;
  }
  if (options.require_consistency_graph || !graphs.empty()) {
    if (image_idx >= graphs.size() ||
        graphs[image_idx].Count(row, col) <
            options.min_num_consistent_sources) {
      return false;
    }
  }
  return images[image_idx].Width() == depth_maps[image_idx].Width() &&
         images[image_idx].Height() == depth_maps[image_idx].Height();
}

std::optional<FusionPixelData> MakeFusionPixelData(
    const std::vector<PatchMatchView>& views,
    const std::vector<MvsImage>& images,
    const std::vector<DenseDepthMap>& depth_maps,
    const std::vector<NormalMap>& normal_maps,
    std::size_t image_idx,
    std::size_t row,
    std::size_t col) {
  try {
    const float depth = depth_maps[image_idx].Get(row, col);
    const Eigen::Vector2d xy(static_cast<double>(col),
                             static_cast<double>(row));
    const Eigen::Vector3d point_camera =
        UnprojectPixel(views[image_idx].camera, xy, static_cast<double>(depth));
    const Eigen::Vector3d point_world =
        views[image_idx].cam_from_world.Inverse().TransformPoint(point_camera);

    Eigen::Vector3d normal_world =
        views[image_idx].cam_from_world.q_cw.conjugate() *
        normal_maps[image_idx].Get(row, col).cast<double>();
    if (!normal_world.allFinite() || normal_world.squaredNorm() <= 0.0) {
      return std::nullopt;
    }
    normal_world.normalize();

    const std::optional<float> gray = SampleGrayBilinear(
        images[image_idx], static_cast<float>(col), static_cast<float>(row));
    const float clamped_gray = ClampFloat(gray.value_or(1.0F), 0.0F, 1.0F);
    const double color_value =
        static_cast<double>(std::lround(clamped_gray * 255.0F));

    FusionPixelData data;
    data.xyz = point_world;
    data.normal = normal_world;
    data.color = Eigen::Vector3d(color_value, color_value, color_value);
    return data;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<FusionObservation> ProjectFusionObservationToSource(
    const std::vector<PatchMatchView>& views,
    const std::vector<DenseDepthMap>& depth_maps,
    const std::vector<NormalMap>& normal_maps,
    const std::vector<DenseCostMap>& cost_maps,
    const std::vector<ConsistencyGraph>& graphs,
    const StereoFusionOptions& options,
    const FusionObservation& reference,
    std::size_t source_idx,
    const Eigen::Vector3d& seed_point_world,
    const Eigen::Vector3d& seed_normal_world,
    std::size_t seed_image_idx,
    StereoFusionErrorSamples* error_samples,
    FusionRejectReason* reject_reason) {
  if (reject_reason != nullptr) {
    *reject_reason = FusionRejectReason::kNone;
  }
  try {
    const float reference_depth =
        depth_maps[reference.image_idx].Get(reference.row, reference.col);
    const Eigen::Vector2d reference_xy(static_cast<double>(reference.col),
                                       static_cast<double>(reference.row));
    const Eigen::Vector3d reference_point_camera =
        UnprojectPixel(views[reference.image_idx].camera, reference_xy,
                       static_cast<double>(reference_depth));
    const Eigen::Vector3d point_world =
        views[reference.image_idx].cam_from_world.Inverse().TransformPoint(
            reference_point_camera);
    const Eigen::Vector3d source_point_camera =
        views[source_idx].cam_from_world.TransformPoint(point_world);
    if (source_point_camera.z() <= 0.0) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kProjection;
      }
      return std::nullopt;
    }

    const Eigen::Vector2d source_xy =
        ProjectPoint(views[source_idx].camera, source_point_camera);
    const int rounded_source_col =
        static_cast<int>(std::lround(source_xy.x()));
    const int rounded_source_row =
        static_cast<int>(std::lround(source_xy.y()));
    if (rounded_source_row < 0 || rounded_source_col < 0 ||
        rounded_source_row >=
            static_cast<int>(depth_maps[source_idx].Height()) ||
        rounded_source_col >=
            static_cast<int>(depth_maps[source_idx].Width())) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kProjection;
      }
      return std::nullopt;
    }

    const double view_angle =
        ViewAngleDegrees(views[seed_image_idx], views[source_idx],
                         seed_point_world);
    AddFusionErrorSample(static_cast<float>(view_angle),
                         error_samples == nullptr
                             ? nullptr
                             : &error_samples->view_angles_degrees);
    if (view_angle > static_cast<double>(options.max_view_angle_degrees)) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kView;
      }
      return std::nullopt;
    }

    std::vector<std::pair<double, std::pair<int, int>>> candidates;
    const int search_radius =
        static_cast<int>(options.source_pixel_search_radius);
    for (int row_offset = -search_radius; row_offset <= search_radius;
         ++row_offset) {
      for (int col_offset = -search_radius; col_offset <= search_radius;
           ++col_offset) {
        const int candidate_row = rounded_source_row + row_offset;
        const int candidate_col = rounded_source_col + col_offset;
        if (candidate_row < 0 || candidate_col < 0 ||
            candidate_row >=
                static_cast<int>(depth_maps[source_idx].Height()) ||
            candidate_col >=
                static_cast<int>(depth_maps[source_idx].Width())) {
          continue;
        }
        const Eigen::Vector2d candidate_xy(
            static_cast<double>(candidate_col),
            static_cast<double>(candidate_row));
        candidates.emplace_back((source_xy - candidate_xy).squaredNorm(),
                                std::make_pair(candidate_row, candidate_col));
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& lhs, const auto& rhs) {
                if (lhs.first == rhs.first) {
                  return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
              });

    FusionRejectReason candidate_reject_reason = FusionRejectReason::kNone;
    for (const auto& candidate : candidates) {
      const double reprojection_error = std::sqrt(candidate.first);
      AddFusionErrorSample(static_cast<float>(reprojection_error),
                           error_samples == nullptr
                               ? nullptr
                               : &error_samples->reprojection_errors_pixels);
      if (reprojection_error > options.max_reprojection_error_pixels) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kProjection,
                                     &candidate_reject_reason);
        continue;
      }

      const std::size_t row =
          static_cast<std::size_t>(candidate.second.first);
      const std::size_t col =
          static_cast<std::size_t>(candidate.second.second);
      if (!depth_maps[source_idx].IsValid(row, col) ||
          !normal_maps[source_idx].IsValid(row, col) ||
          !cost_maps[source_idx].IsValid(row, col)) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kInvalid,
                                     &candidate_reject_reason);
        continue;
      }

      // 与几何一致性图保持一致：优先在真实投影位置做双线性深度采样。
      // 若投影落在边界或邻域深度不完整，则回退到候选离散像素深度。
      double source_depth =
          static_cast<double>(depth_maps[source_idx].Get(row, col));
      const std::optional<float> projected_source_depth = SampleDepthBilinear(
          depth_maps[source_idx], static_cast<float>(source_xy.x()),
          static_cast<float>(source_xy.y()));
      if (projected_source_depth.has_value()) {
        source_depth = static_cast<double>(*projected_source_depth);
      }
      const double relative_depth_error =
          std::abs(source_depth - source_point_camera.z()) /
          source_point_camera.z();
      AddFusionErrorSample(static_cast<float>(relative_depth_error),
                           error_samples == nullptr
                               ? nullptr
                               : &error_samples->relative_depth_errors);
      if (relative_depth_error > options.max_relative_depth_error) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kDepth,
                                     &candidate_reject_reason);
        continue;
      }

      if (options.require_mutual_consistency && !graphs.empty() &&
          source_idx < graphs.size() &&
          !ContainsImageId(graphs[source_idx].SourceImageIds(row, col),
                           views[reference.image_idx].image_id)) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kMutualConsistency,
                                     &candidate_reject_reason);
        continue;
      }

      Eigen::Vector3d source_normal_world =
          views[source_idx].cam_from_world.q_cw.conjugate() *
          normal_maps[source_idx].Get(row, col).cast<double>();
      if (!source_normal_world.allFinite() ||
          source_normal_world.squaredNorm() <= 0.0) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kInvalid,
                                     &candidate_reject_reason);
        continue;
      }
      source_normal_world.normalize();

      Eigen::Vector3d reference_normal_world =
          views[reference.image_idx].cam_from_world.q_cw.conjugate() *
          normal_maps[reference.image_idx]
              .Get(reference.row, reference.col)
              .cast<double>();
      if (!reference_normal_world.allFinite() ||
          reference_normal_world.squaredNorm() <= 0.0) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kInvalid,
                                     &candidate_reject_reason);
        continue;
      }
      reference_normal_world.normalize();

      // Track 从 seed 向外传播时，远端观测可能与 seed 法线略有曲率差异。
      // 同时参考 seed 与当前扩展点的法线，避免局部连续表面被过早截断。
      const double normal_angle_to_seed =
          AngleDegreesBetween(seed_normal_world, source_normal_world);
      const double normal_angle_to_reference =
          AngleDegreesBetween(reference_normal_world, source_normal_world);
      const double normal_angle =
          std::min(normal_angle_to_seed, normal_angle_to_reference);
      AddFusionErrorSample(static_cast<float>(normal_angle),
                           error_samples == nullptr
                               ? nullptr
                               : &error_samples->normal_angles_degrees);
      if (normal_angle >
          static_cast<double>(options.max_normal_error_degrees)) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kNormal,
                                     &candidate_reject_reason);
        continue;
      }

      FusionObservation source;
      source.image_idx = source_idx;
      source.row = row;
      source.col = col;
      source.traversal_depth = reference.traversal_depth + 1;
      source.normal_angle_degrees = normal_angle;
      source.view_angle_degrees = view_angle;
      return source;
    }

    if (reject_reason != nullptr) {
      *reject_reason = candidate_reject_reason == FusionRejectReason::kNone
                           ? FusionRejectReason::kProjection
                           : candidate_reject_reason;
    }
    return std::nullopt;
  } catch (const std::exception&) {
    if (reject_reason != nullptr) {
      *reject_reason = FusionRejectReason::kInvalid;
    }
    return std::nullopt;
  }
}

}  // namespace

ConsistencyGraph::ConsistencyGraph(std::size_t width, std::size_t height)
    : width_(width), height_(height), data_(width * height) {}

std::size_t ConsistencyGraph::Width() const {
  return width_;
}

std::size_t ConsistencyGraph::Height() const {
  return height_;
}

std::size_t ConsistencyGraph::NumPixels() const {
  return data_.size();
}

bool ConsistencyGraph::Empty() const {
  return data_.empty();
}

void ConsistencyGraph::Add(std::size_t row,
                           std::size_t col,
                           ImageId source_image_id) {
  data_.at(Offset(row, col)).push_back(source_image_id);
}

const std::vector<ImageId>& ConsistencyGraph::SourceImageIds(
    std::size_t row,
    std::size_t col) const {
  return data_.at(Offset(row, col));
}

std::size_t ConsistencyGraph::Count(std::size_t row, std::size_t col) const {
  return SourceImageIds(row, col).size();
}

std::size_t ConsistencyGraph::NumConsistentPixels(
    std::size_t min_num_sources) const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(),
                    [min_num_sources](const std::vector<ImageId>& sources) {
                      return sources.size() >= min_num_sources;
                    }));
}

std::size_t ConsistencyGraph::Offset(std::size_t row, std::size_t col) const {
  if (row >= height_ || col >= width_) {
    throw std::out_of_range("ConsistencyGraph pixel index is out of range");
  }
  return row * width_ + col;
}

void WriteConsistencyGraphBinary(const std::filesystem::path& path,
                                 const ConsistencyGraph& consistency_graph) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary consistency graph: " +
                             path.string());
  }

  file << kConsistencyGraphBinaryMagic << " "
       << kConsistencyGraphBinaryVersion << "\n"
       << consistency_graph.Width() << " " << consistency_graph.Height()
       << "\n";
  for (std::size_t row = 0; row < consistency_graph.Height(); ++row) {
    for (std::size_t col = 0; col < consistency_graph.Width(); ++col) {
      const std::vector<ImageId>& source_ids =
          consistency_graph.SourceImageIds(row, col);
      const std::uint32_t count =
          static_cast<std::uint32_t>(source_ids.size());
      file.write(reinterpret_cast<const char*>(&count),
                 static_cast<std::streamsize>(sizeof(count)));
      for (const ImageId source_id : source_ids) {
        const std::uint32_t value = static_cast<std::uint32_t>(source_id);
        file.write(reinterpret_cast<const char*>(&value),
                   static_cast<std::streamsize>(sizeof(value)));
      }
    }
  }
  RequireConsistencyGraphStream(static_cast<bool>(file), path);
}

ConsistencyGraph ReadConsistencyGraphBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary consistency graph: " +
                             path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireConsistencyGraphStream(magic == kConsistencyGraphBinaryMagic &&
                                    version ==
                                        kConsistencyGraphBinaryVersion,
                                path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireConsistencyGraphStream(static_cast<bool>(file), path);
  file.get();

  ConsistencyGraph consistency_graph(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::uint32_t count = 0;
      file.read(reinterpret_cast<char*>(&count),
                static_cast<std::streamsize>(sizeof(count)));
      RequireConsistencyGraphStream(static_cast<bool>(file), path);
      for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t source_id = 0;
        file.read(reinterpret_cast<char*>(&source_id),
                  static_cast<std::streamsize>(sizeof(source_id)));
        RequireConsistencyGraphStream(static_cast<bool>(file), path);
        consistency_graph.Add(row, col, static_cast<ImageId>(source_id));
      }
    }
  }
  return consistency_graph;
}

PatchMatchMaps InitializePatchMatchMaps(
    const MvsImage& image,
    float depth_min,
    float depth_max,
    const PatchMatchInitializationOptions& options) {
  if (image.Empty()) {
    throw std::invalid_argument("PatchMatch reference image is empty.");
  }
  RequireValidDepthRange(depth_min, depth_max);

  PatchMatchMaps maps;
  maps.depth_map =
      DenseDepthMap(image.Width(), image.Height(), depth_min, depth_max);
  maps.normal_map = NormalMap(image.Width(), image.Height());

  std::mt19937 rng(options.random_seed);
  std::uniform_real_distribution<float> depth_distribution(depth_min,
                                                           depth_max);
  const Eigen::Vector3f default_normal =
      options.default_normal.squaredNorm() > 0.0F
          ? options.default_normal.normalized()
          : Eigen::Vector3f(0.0F, 0.0F, -1.0F);

  for (std::size_t row = 0; row < image.Height(); ++row) {
    for (std::size_t col = 0; col < image.Width(); ++col) {
      maps.depth_map.Set(row, col, depth_distribution(rng));
      const Eigen::Vector3f initial_normal =
          options.randomize_initial_normals
              ? SampleNormalInHemisphere(
                    default_normal, options.initial_normal_hemisphere_degrees,
                    &rng)
              : default_normal;
      maps.normal_map.Set(row, col, initial_normal);
    }
  }

  return maps;
}

PatchMatchMaps InitializePatchMatchMaps(
    const MvsImage& image,
    const SparseDepthMap& sparse_depth_map,
    const PatchMatchInitializationOptions& options) {
  const auto [depth_min, depth_max] =
      DepthRangeFromSparseMap(sparse_depth_map, options);
  return InitializePatchMatchMaps(image, depth_min, depth_max, options);
}

std::optional<float> SampleGrayBilinear(const MvsImage& image,
                                        float x,
                                        float y) {
  if (image.Empty() || image.gray.type() != CV_32FC1 || !std::isfinite(x) ||
      !std::isfinite(y)) {
    return std::nullopt;
  }
  if (x < 0.0F || y < 0.0F ||
      x > static_cast<float>(image.gray.cols - 1) ||
      y > static_cast<float>(image.gray.rows - 1)) {
    return std::nullopt;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.gray.cols - 1);
  const int y1 = std::min(y0 + 1, image.gray.rows - 1);
  const float dx = x - static_cast<float>(x0);
  const float dy = y - static_cast<float>(y0);

  const float v00 = image.gray.at<float>(y0, x0);
  const float v01 = image.gray.at<float>(y0, x1);
  const float v10 = image.gray.at<float>(y1, x0);
  const float v11 = image.gray.at<float>(y1, x1);
  const float top = (1.0F - dx) * v00 + dx * v01;
  const float bottom = (1.0F - dx) * v10 + dx * v11;
  return (1.0F - dy) * top + dy * bottom;
}

PatchMatchCost ComputePatchZncc(
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    const Eigen::Vector2f& source_xy,
    const PatchMatchCostOptions& options) {
  if (options.window_radius < 0 || options.window_step <= 0) {
    throw std::invalid_argument("Invalid PatchMatch cost window options.");
  }

  std::vector<float> reference_values;
  std::vector<float> source_values;
  const int radius = options.window_radius;
  for (int dy = -radius; dy <= radius; dy += options.window_step) {
    for (int dx = -radius; dx <= radius; dx += options.window_step) {
      const auto ref_sample = SampleGrayBilinear(
          reference_image, reference_xy.x() + static_cast<float>(dx),
          reference_xy.y() + static_cast<float>(dy));
      const auto src_sample = SampleGrayBilinear(
          source_image, source_xy.x() + static_cast<float>(dx),
          source_xy.y() + static_cast<float>(dy));
      if (!ref_sample.has_value() || !src_sample.has_value()) {
        return InvalidCost(options);
      }
      reference_values.push_back(*ref_sample);
      source_values.push_back(*src_sample);
    }
  }

  return ComputeZnccFromSamples(reference_values, source_values, options);
}

std::optional<Eigen::Vector2f> ProjectDepthToSourcePixel(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const Eigen::Vector2f& reference_xy,
    float depth) {
  if (!std::isfinite(depth) || depth <= 0.0F) {
    return std::nullopt;
  }
  try {
    const Eigen::Vector3d point_reference = UnprojectPixel(
        reference_view.camera, reference_xy.cast<double>(),
        static_cast<double>(depth));
    const Eigen::Vector3d point_world =
        reference_view.cam_from_world.Inverse().TransformPoint(point_reference);
    const Eigen::Vector3d point_source =
        source_view.cam_from_world.TransformPoint(point_world);
    if (point_source.z() <= 0.0) {
      return std::nullopt;
    }
    return ProjectPoint(source_view.camera, point_source).cast<float>();
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<Eigen::Vector2f> ProjectPlaneToSourcePixel(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const Eigen::Vector2f& reference_center_xy,
    float center_depth,
    const Eigen::Vector3f& reference_normal,
    const Eigen::Vector2f& sample_xy) {
  if (!std::isfinite(center_depth) || center_depth <= 0.0F ||
      !reference_normal.allFinite() ||
      reference_normal.squaredNorm() <= 0.0F) {
    return std::nullopt;
  }

  try {
    const Eigen::Vector3d center_reference = UnprojectPixel(
        reference_view.camera, reference_center_xy.cast<double>(),
        static_cast<double>(center_depth));
    const Eigen::Vector3d sample_ray = UnprojectPixel(
        reference_view.camera, sample_xy.cast<double>(), 1.0);
    const Eigen::Vector3d normal =
        reference_normal.cast<double>().normalized();
    const double numerator = normal.dot(center_reference);
    const double denominator = normal.dot(sample_ray);
    if (std::abs(denominator) < 1e-12) {
      return std::nullopt;
    }
    const double ray_depth = numerator / denominator;
    if (!std::isfinite(ray_depth) || ray_depth <= 0.0) {
      return std::nullopt;
    }

    const Eigen::Vector3d point_reference = sample_ray * ray_depth;
    const Eigen::Vector3d point_world =
        reference_view.cam_from_world.Inverse().TransformPoint(point_reference);
    const Eigen::Vector3d point_source =
        source_view.cam_from_world.TransformPoint(point_world);
    if (point_source.z() <= 0.0) {
      return std::nullopt;
    }
    return ProjectPoint(source_view.camera, point_source).cast<float>();
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<float> ComputeSourceDepthRelativeError(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const DenseDepthMap& source_depth_map) {
  try {
    const Eigen::Vector3d point_reference =
        UnprojectPixel(reference_view.camera, reference_xy.cast<double>(),
                       static_cast<double>(depth));
    const Eigen::Vector3d point_world =
        reference_view.cam_from_world.Inverse().TransformPoint(point_reference);
    const Eigen::Vector3d point_source =
        source_view.cam_from_world.TransformPoint(point_world);
    if (point_source.z() <= 0.0) {
      return std::nullopt;
    }

    const Eigen::Vector2f source_xy =
        ProjectPoint(source_view.camera, point_source).cast<float>();
    const std::optional<float> source_depth =
        SampleDepthBilinear(source_depth_map, source_xy.x(), source_xy.y());
    if (!source_depth.has_value()) {
      return std::nullopt;
    }

    return static_cast<float>(
        std::abs(static_cast<double>(*source_depth) - point_source.z()) /
        point_source.z());
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool UsePerturbationGeometricFilter(
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  return source_depth_maps != nullptr && !source_depth_maps->empty() &&
         options.geometric_consistency_weight > 0.0F;
}

std::size_t CountGeometricallyConsistentSourceDepths(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>& source_depth_maps) {
  if (source_views.size() != source_depth_maps.size()) {
    throw std::invalid_argument(
        "Source depth maps must match source views for perturbation filter.");
  }
  if (!std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "Invalid PatchMatch perturbation geometric filter options.");
  }

  std::size_t num_consistent_sources = 0;
  for (std::size_t i = 0; i < source_views.size(); ++i) {
    const std::optional<float> relative_depth_error =
        ComputeSourceDepthRelativeError(reference_view, source_views[i],
                                        reference_xy, depth,
                                        source_depth_maps[i]);
    if (relative_depth_error.has_value() &&
        *relative_depth_error <=
            options.geometric_max_relative_depth_error) {
      num_consistent_sources++;
    }
  }
  return num_consistent_sources;
}

PatchMatchCost ComputeDepthHypothesisCost(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const PatchMatchCostOptions& options) {
  const std::optional<Eigen::Vector2f> source_xy =
      ProjectDepthToSourcePixel(reference_view, source_view, reference_xy,
                                depth);
  if (!source_xy.has_value()) {
    return InvalidCost(options);
  }
  return ComputePatchZncc(reference_image, source_image, reference_xy,
                          *source_xy, options);
}

PatchMatchCost ComputePlaneHypothesisCost(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const Eigen::Vector3f& reference_normal,
    const PatchMatchCostOptions& options,
    const DenseDepthMap* source_depth_map) {
  if (options.window_radius < 0 || options.window_step <= 0) {
    throw std::invalid_argument("Invalid PatchMatch cost window options.");
  }
  if (!std::isfinite(options.geometric_consistency_weight) ||
      options.geometric_consistency_weight < 0.0F ||
      !std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "Invalid PatchMatch geometric consistency cost options.");
  }

  std::vector<float> reference_values;
  std::vector<float> source_values;
  const int radius = options.window_radius;
  for (int dy = -radius; dy <= radius; dy += options.window_step) {
    for (int dx = -radius; dx <= radius; dx += options.window_step) {
      const Eigen::Vector2f sample_xy =
          reference_xy + Eigen::Vector2f(static_cast<float>(dx),
                                         static_cast<float>(dy));
      const std::optional<Eigen::Vector2f> source_xy =
          ProjectPlaneToSourcePixel(reference_view, source_view, reference_xy,
                                    depth, reference_normal, sample_xy);
      if (!source_xy.has_value()) {
        return InvalidCost(options);
      }

      const auto ref_sample =
          SampleGrayBilinear(reference_image, sample_xy.x(), sample_xy.y());
      const auto src_sample =
          SampleGrayBilinear(source_image, source_xy->x(), source_xy->y());
      if (!ref_sample.has_value() || !src_sample.has_value()) {
        return InvalidCost(options);
      }
      reference_values.push_back(*ref_sample);
      source_values.push_back(*src_sample);
    }
  }

  PatchMatchCost cost =
      ComputeZnccFromSamples(reference_values, source_values, options);
  if (!cost.valid || source_depth_map == nullptr ||
      options.geometric_consistency_weight <= 0.0F) {
    return cost;
  }

  // 将当前候选深度投影到 source 图，并用上一轮 source depth
  // 给几何不一致的候选增加软惩罚。缺少 source depth 时不强行判坏。
  const std::optional<float> relative_depth_error =
      ComputeSourceDepthRelativeError(reference_view, source_view,
                                      reference_xy, depth, *source_depth_map);
  if (!relative_depth_error.has_value()) {
    return cost;
  }
  cost.cost += options.geometric_consistency_weight *
               std::min(1.0F, *relative_depth_error /
                                   options.geometric_max_relative_depth_error);
  return cost;
}

MultiSourcePatchMatchCost ComputeMultiSourcePlaneHypothesisCost(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const Eigen::Vector3f& reference_normal,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  if (source_views.size() != source_images.size()) {
    throw std::invalid_argument(
        "Source views and source images must have identical sizes.");
  }
  if (source_depth_maps != nullptr &&
      source_depth_maps->size() != source_views.size()) {
    throw std::invalid_argument(
        "Source depth maps must match source views for geometric cost.");
  }

  MultiSourcePatchMatchCost result;
  result.num_total_sources = source_views.size();

  std::vector<float> valid_costs;
  valid_costs.reserve(source_views.size());
  for (std::size_t i = 0; i < source_views.size(); ++i) {
    const DenseDepthMap* source_depth_map =
        source_depth_maps == nullptr ? nullptr : &source_depth_maps->at(i);
    const PatchMatchCost cost =
        ComputePlaneHypothesisCost(reference_view, source_views[i],
                                   reference_image, source_images[i],
                                   reference_xy, depth, reference_normal,
                                   options, source_depth_map);
    if (cost.valid) {
      valid_costs.push_back(cost.cost);
    }
  }

  result.num_valid_sources = valid_costs.size();
  if (valid_costs.size() < options.min_num_valid_sources) {
    result.valid = false;
    result.cost = options.invalid_cost;
    return result;
  }

  std::sort(valid_costs.begin(), valid_costs.end());
  const std::size_t num_used =
      options.max_num_aggregated_costs == 0
          ? valid_costs.size()
          : std::min(valid_costs.size(), options.max_num_aggregated_costs);
  double total_cost = 0.0;
  for (std::size_t i = 0; i < num_used; ++i) {
    total_cost += valid_costs[i];
  }

  result.valid = true;
  result.num_used_sources = num_used;
  result.cost = static_cast<float>(total_cost / static_cast<double>(num_used));
  return result;
}

DenseCostMap ComputeBestCostMap(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const DenseDepthMap& depth_map,
    const NormalMap& normal_map,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  if (depth_map.Width() != normal_map.Width() ||
      depth_map.Height() != normal_map.Height()) {
    throw std::invalid_argument(
        "Depth map and normal map sizes must match for cost evaluation.");
  }
  if (depth_map.Width() != reference_image.Width() ||
      depth_map.Height() != reference_image.Height()) {
    throw std::invalid_argument(
        "Reference image and hypothesis maps must have identical sizes.");
  }

  DenseCostMap cost_map(depth_map.Width(), depth_map.Height());
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      if (!depth_map.IsValid(row, col) || !normal_map.IsValid(row, col)) {
        continue;
      }
      const Eigen::Vector2f reference_xy(static_cast<float>(col),
                                         static_cast<float>(row));
      const MultiSourcePatchMatchCost cost =
          ComputeMultiSourcePlaneHypothesisCost(
              reference_view, source_views, reference_image, source_images,
              reference_xy, depth_map.Get(row, col), normal_map.Get(row, col),
              options, source_depth_maps);
      if (cost.valid) {
        cost_map.Set(row, col, cost.cost);
      }
    }
  }
  return cost_map;
}

PatchMatchSweepSummary RunPatchMatchPropagationSweep(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    PatchMatchSweepDirection direction,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  if (depth_map == nullptr || normal_map == nullptr || cost_map == nullptr) {
    throw std::invalid_argument(
        "PatchMatch propagation requires non-null maps.");
  }
  RequireCompatiblePatchMatchMaps(reference_image, *depth_map, *normal_map,
                                  *cost_map);

  PatchMatchSweepSummary summary;
  const int width = static_cast<int>(depth_map->Width());
  const int height = static_cast<int>(depth_map->Height());

  const auto try_neighbor = [&](int row, int col, int neighbor_row,
                                int neighbor_col) {
    if (neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 ||
        neighbor_col >= width) {
      return;
    }
    const std::size_t r = static_cast<std::size_t>(row);
    const std::size_t c = static_cast<std::size_t>(col);
    const std::size_t nr = static_cast<std::size_t>(neighbor_row);
    const std::size_t nc = static_cast<std::size_t>(neighbor_col);
    if (!depth_map->IsValid(nr, nc) || !normal_map->IsValid(nr, nc) ||
        !cost_map->IsValid(nr, nc)) {
      return;
    }

    summary.num_candidate_evaluations++;
    const Eigen::Vector3f candidate_normal = normal_map->Get(nr, nc);
    const std::optional<float> candidate_depth = ProjectPlaneDepthToPixel(
        reference_view.camera,
        Eigen::Vector2d(static_cast<double>(neighbor_col),
                        static_cast<double>(neighbor_row)),
        depth_map->Get(nr, nc), candidate_normal,
        Eigen::Vector2d(static_cast<double>(col), static_cast<double>(row)));
    if (!candidate_depth.has_value() ||
        *candidate_depth < depth_map->DepthMin() ||
        *candidate_depth > depth_map->DepthMax()) {
      return;
    }
    const Eigen::Vector2f reference_xy(static_cast<float>(col),
                                       static_cast<float>(row));
    const MultiSourcePatchMatchCost candidate_cost =
        ComputeMultiSourcePlaneHypothesisCost(
            reference_view, source_views, reference_image, source_images,
            reference_xy, *candidate_depth, candidate_normal, options,
            source_depth_maps);
    if (!candidate_cost.valid) {
      return;
    }

    const bool current_cost_valid = cost_map->IsValid(r, c);
    if (!current_cost_valid || candidate_cost.cost < cost_map->Get(r, c)) {
      depth_map->Set(r, c, *candidate_depth);
      normal_map->SetNormalized(r, c, candidate_normal);
      cost_map->Set(r, c, candidate_cost.cost);
      summary.num_updates++;
    }
  };

  switch (direction) {
    case PatchMatchSweepDirection::kForward:
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          summary.num_pixels_visited++;
          try_neighbor(row, col, row, col - 1);
          try_neighbor(row, col, row - 1, col);
        }
      }
      break;
    case PatchMatchSweepDirection::kBackward:
      for (int row = height - 1; row >= 0; --row) {
        for (int col = width - 1; col >= 0; --col) {
          summary.num_pixels_visited++;
          try_neighbor(row, col, row, col + 1);
          try_neighbor(row, col, row + 1, col);
        }
      }
      break;
    case PatchMatchSweepDirection::kTopDown:
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          summary.num_pixels_visited++;
          try_neighbor(row, col, row - 1, col);
        }
      }
      break;
    case PatchMatchSweepDirection::kBottomUp:
      for (int row = height - 1; row >= 0; --row) {
        for (int col = 0; col < width; ++col) {
          summary.num_pixels_visited++;
          try_neighbor(row, col, row + 1, col);
        }
      }
      break;
  }

  return summary;
}

PatchMatchSweepSummary RunPatchMatchCheckerboardPropagation(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    int parity,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  if (depth_map == nullptr || normal_map == nullptr || cost_map == nullptr) {
    throw std::invalid_argument(
        "PatchMatch checkerboard propagation requires non-null maps.");
  }
  if (parity != 0 && parity != 1) {
    throw std::invalid_argument("PatchMatch checkerboard parity is invalid.");
  }
  RequireCompatiblePatchMatchMaps(reference_image, *depth_map, *normal_map,
                                  *cost_map);

  PatchMatchSweepSummary summary;
  const int width = static_cast<int>(depth_map->Width());
  const int height = static_cast<int>(depth_map->Height());

  const auto try_neighbor = [&](int row, int col, int neighbor_row,
                                int neighbor_col) {
    if (neighbor_row < 0 || neighbor_row >= height || neighbor_col < 0 ||
        neighbor_col >= width) {
      return;
    }
    const std::size_t r = static_cast<std::size_t>(row);
    const std::size_t c = static_cast<std::size_t>(col);
    const std::size_t nr = static_cast<std::size_t>(neighbor_row);
    const std::size_t nc = static_cast<std::size_t>(neighbor_col);
    if (!depth_map->IsValid(nr, nc) || !normal_map->IsValid(nr, nc) ||
        !cost_map->IsValid(nr, nc)) {
      return;
    }

    summary.num_candidate_evaluations++;
    const Eigen::Vector3f candidate_normal = normal_map->Get(nr, nc);
    const std::optional<float> candidate_depth = ProjectPlaneDepthToPixel(
        reference_view.camera,
        Eigen::Vector2d(static_cast<double>(neighbor_col),
                        static_cast<double>(neighbor_row)),
        depth_map->Get(nr, nc), candidate_normal,
        Eigen::Vector2d(static_cast<double>(col), static_cast<double>(row)));
    if (!candidate_depth.has_value() ||
        *candidate_depth < depth_map->DepthMin() ||
        *candidate_depth > depth_map->DepthMax()) {
      return;
    }
    const Eigen::Vector2f reference_xy(static_cast<float>(col),
                                       static_cast<float>(row));
    const MultiSourcePatchMatchCost candidate_cost =
        ComputeMultiSourcePlaneHypothesisCost(
            reference_view, source_views, reference_image, source_images,
            reference_xy, *candidate_depth, candidate_normal, options,
            source_depth_maps);
    if (!candidate_cost.valid) {
      return;
    }

    const bool current_cost_valid = cost_map->IsValid(r, c);
    if (!current_cost_valid || candidate_cost.cost < cost_map->Get(r, c)) {
      depth_map->Set(r, c, *candidate_depth);
      normal_map->SetNormalized(r, c, candidate_normal);
      cost_map->Set(r, c, candidate_cost.cost);
      summary.num_updates++;
    }
  };

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (((row + col) & 1) != parity) {
        continue;
      }
      summary.num_pixels_visited++;
      // 红黑传播只更新当前颜色格，候选来自四邻域，便于后续并行化。
      try_neighbor(row, col, row, col - 1);
      try_neighbor(row, col, row, col + 1);
      try_neighbor(row, col, row - 1, col);
      try_neighbor(row, col, row + 1, col);
    }
  }

  return summary;
}

PatchMatchPerturbationSummary RunPatchMatchRandomPerturbation(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    const PatchMatchPerturbationOptions& perturbation_options,
    const PatchMatchCostOptions& cost_options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  if (depth_map == nullptr || normal_map == nullptr || cost_map == nullptr) {
    throw std::invalid_argument(
        "PatchMatch perturbation requires non-null maps.");
  }
  RequireCompatiblePatchMatchMaps(reference_image, *depth_map, *normal_map,
                                  *cost_map);
  RequireValidDepthRange(depth_map->DepthMin(), depth_map->DepthMax());
  if (perturbation_options.depth_radius < 0.0F ||
      perturbation_options.normal_radius_degrees < 0.0F ||
      perturbation_options.num_samples_per_pixel == 0) {
    throw std::invalid_argument("PatchMatch perturbation radius is invalid.");
  }

  PatchMatchPerturbationSummary summary;
  std::mt19937 rng(perturbation_options.random_seed);
  std::uniform_real_distribution<float> depth_distribution(
      -perturbation_options.depth_radius, perturbation_options.depth_radius);
  const bool use_geometric_filter =
      UsePerturbationGeometricFilter(cost_options, source_depth_maps);

  for (std::size_t row = 0; row < depth_map->Height(); ++row) {
    for (std::size_t col = 0; col < depth_map->Width(); ++col) {
      summary.num_pixels_visited++;
      if (!depth_map->IsValid(row, col) || !normal_map->IsValid(row, col) ||
          !cost_map->IsValid(row, col)) {
        continue;
      }

      const float current_depth = depth_map->Get(row, col);
      const Eigen::Vector3f current_normal = normal_map->Get(row, col);
      const Eigen::Vector2f reference_xy(static_cast<float>(col),
                                         static_cast<float>(row));

      bool best_valid = false;
      float best_depth = current_depth;
      Eigen::Vector3f best_normal = current_normal;
      float best_cost = cost_map->Get(row, col);
      for (std::size_t sample_idx = 0;
           sample_idx < perturbation_options.num_samples_per_pixel;
           ++sample_idx) {
        const float candidate_depth =
            ClampFloat(current_depth + depth_distribution(rng),
                       depth_map->DepthMin(), depth_map->DepthMax());
        const Eigen::Vector3f candidate_normal =
            PerturbNormal(current_normal,
                          perturbation_options.normal_radius_degrees, &rng);

        summary.num_candidate_evaluations++;
        if (use_geometric_filter) {
          const std::size_t num_consistent_sources =
              CountGeometricallyConsistentSourceDepths(
                  reference_view, source_views, reference_xy, candidate_depth,
                  cost_options, *source_depth_maps);
          if (num_consistent_sources < cost_options.min_num_valid_sources) {
            summary.num_geometric_candidate_rejections++;
            continue;
          }
          summary.num_geometric_candidate_acceptances++;
        }

        const MultiSourcePatchMatchCost candidate_cost =
            ComputeMultiSourcePlaneHypothesisCost(
                reference_view, source_views, reference_image, source_images,
                reference_xy, candidate_depth, candidate_normal, cost_options,
                source_depth_maps);
        if (!candidate_cost.valid || candidate_cost.cost >= best_cost) {
          continue;
        }
        best_valid = true;
        best_depth = candidate_depth;
        best_normal = candidate_normal;
        best_cost = candidate_cost.cost;
      }

      if (!best_valid) {
        continue;
      }

      const double abs_depth_update =
          std::abs(static_cast<double>(best_depth) -
                   static_cast<double>(current_depth));
      const double normal_angle =
          AngleDegreesBetween(current_normal.cast<double>(),
                              best_normal.cast<double>());
      summary.total_abs_depth_update += abs_depth_update;
      summary.max_abs_depth_update =
          std::max(summary.max_abs_depth_update, abs_depth_update);
      summary.total_normal_angle_degrees += normal_angle;
      summary.max_normal_angle_degrees =
          std::max(summary.max_normal_angle_degrees, normal_angle);

      depth_map->Set(row, col, best_depth);
      normal_map->SetNormalized(row, col, best_normal);
      cost_map->Set(row, col, best_cost);
      summary.num_updates++;
    }
  }

  if (summary.num_updates > 0) {
    summary.mean_abs_depth_update =
        summary.total_abs_depth_update /
        static_cast<double>(summary.num_updates);
    summary.mean_normal_angle_degrees =
        summary.total_normal_angle_degrees /
        static_cast<double>(summary.num_updates);
  }
  return summary;
}

PatchMatchNormalRefinementSummary RefineNormalMapFromDepth(
    const PatchMatchView& reference_view,
    const DenseDepthMap& depth_map,
    const PatchMatchNormalRefinementOptions& options,
    NormalMap* normal_map) {
  if (normal_map == nullptr) {
    throw std::invalid_argument("Normal refinement requires a non-null map.");
  }
  if (depth_map.Width() != normal_map->Width() ||
      depth_map.Height() != normal_map->Height()) {
    throw std::invalid_argument(
        "Depth and normal maps must match for normal refinement.");
  }
  if (!std::isfinite(options.max_relative_depth_difference) ||
      options.max_relative_depth_difference < 0.0F) {
    throw std::invalid_argument(
        "Normal refinement depth-difference threshold is invalid.");
  }

  PatchMatchNormalRefinementSummary summary;
  summary.num_pixels = depth_map.NumPixels();
  if (depth_map.Width() < 3 || depth_map.Height() < 3) {
    summary.num_skipped_pixels = summary.num_pixels;
    return summary;
  }

  const auto depth_is_consistent = [&](float center_depth,
                                       std::size_t row,
                                       std::size_t col) {
    if (!depth_map.IsValid(row, col) || center_depth <= 0.0F) {
      return false;
    }
    const float neighbor_depth = depth_map.Get(row, col);
    const float relative_difference =
        std::abs(neighbor_depth - center_depth) / center_depth;
    return relative_difference <= options.max_relative_depth_difference;
  };

  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      if (row == 0 || col == 0 || row + 1 >= depth_map.Height() ||
          col + 1 >= depth_map.Width() || !depth_map.IsValid(row, col) ||
          !normal_map->IsValid(row, col)) {
        summary.num_skipped_pixels++;
        continue;
      }

      const float center_depth = depth_map.Get(row, col);
      if (!depth_is_consistent(center_depth, row, col - 1) ||
          !depth_is_consistent(center_depth, row, col + 1) ||
          !depth_is_consistent(center_depth, row - 1, col) ||
          !depth_is_consistent(center_depth, row + 1, col)) {
        summary.num_skipped_pixels++;
        continue;
      }

      const Eigen::Vector3d left = UnprojectPixel(
          reference_view.camera,
          Eigen::Vector2d(static_cast<double>(col - 1),
                          static_cast<double>(row)),
          static_cast<double>(depth_map.Get(row, col - 1)));
      const Eigen::Vector3d right = UnprojectPixel(
          reference_view.camera,
          Eigen::Vector2d(static_cast<double>(col + 1),
                          static_cast<double>(row)),
          static_cast<double>(depth_map.Get(row, col + 1)));
      const Eigen::Vector3d up = UnprojectPixel(
          reference_view.camera,
          Eigen::Vector2d(static_cast<double>(col),
                          static_cast<double>(row - 1)),
          static_cast<double>(depth_map.Get(row - 1, col)));
      const Eigen::Vector3d down = UnprojectPixel(
          reference_view.camera,
          Eigen::Vector2d(static_cast<double>(col),
                          static_cast<double>(row + 1)),
          static_cast<double>(depth_map.Get(row + 1, col)));

      Eigen::Vector3f refined_normal = (right - left).cross(down - up).cast<float>();
      if (!refined_normal.allFinite() || refined_normal.squaredNorm() <= 0.0F) {
        summary.num_skipped_pixels++;
        continue;
      }
      refined_normal.normalize();

      const Eigen::Vector3f original_normal = normal_map->Get(row, col);
      if (refined_normal.dot(original_normal) < 0.0F) {
        refined_normal = -refined_normal;
      }

      normal_map->SetNormalized(row, col, refined_normal);
      summary.num_refined_pixels++;
    }
  }
  return summary;
}

PatchMatchStereoResult RunPatchMatchStereoForImageSingleScale(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>* source_depth_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options,
    const PatchMatchMaps* initial_maps) {
  if (source_views.size() != source_images.size()) {
    throw std::invalid_argument(
        "Source views and source images must have identical sizes.");
  }
  if (source_depth_maps != nullptr &&
      source_depth_maps->size() != source_views.size()) {
    throw std::invalid_argument(
        "Source depth maps must match source views for PatchMatch refinement.");
  }
  if (options.perturbation_shrink <= 0.0F ||
      options.initial_depth_perturbation_factor < 0.0F ||
      options.min_depth_perturbation < 0.0F ||
      options.min_normal_radius_degrees < 0.0F ||
      !std::isfinite(options.adaptive_perturbation_min_update_ratio) ||
      options.adaptive_perturbation_min_update_ratio < 0.0F ||
      options.adaptive_perturbation_min_update_ratio > 1.0F ||
      options.pyramid_levels == 0) {
    throw std::invalid_argument("PatchMatch stereo options are invalid.");
  }

  PatchMatchStereoResult result;
  if (initial_maps != nullptr) {
    if (initial_maps->depth_map.Width() != reference_image.Width() ||
        initial_maps->depth_map.Height() != reference_image.Height() ||
        initial_maps->normal_map.Width() != reference_image.Width() ||
        initial_maps->normal_map.Height() != reference_image.Height()) {
      throw std::invalid_argument(
          "PatchMatch initial maps must match the reference image size.");
    }
    result.depth_map = initial_maps->depth_map;
    result.normal_map = initial_maps->normal_map;
    result.depth_map.SetDepthRange(depth_min, depth_max);
  } else {
    PatchMatchMaps maps =
        InitializePatchMatchMaps(reference_image, depth_min, depth_max,
                                 options.initialization);
    result.depth_map = std::move(maps.depth_map);
    result.normal_map = std::move(maps.normal_map);
  }
  result.cost_map = ComputeBestCostMap(reference_view, source_views,
                                       reference_image, source_images,
                                       result.depth_map, result.normal_map,
                                       options.cost, source_depth_maps);

  const float depth_range = depth_max - depth_min;
  float depth_radius =
      depth_range * options.initial_depth_perturbation_factor;
  float normal_radius = options.perturbation.normal_radius_degrees;
  for (std::size_t iteration = 0; iteration < options.num_iterations;
       ++iteration) {
    PatchMatchSweepSummary propagation_summary;
    switch (options.propagation_schedule) {
      case PatchMatchPropagationSchedule::kForwardBackward: {
        const PatchMatchSweepSummary forward = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kForward, options.cost,
            source_depth_maps);
        const PatchMatchSweepSummary backward = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kBackward, options.cost,
            source_depth_maps);
        propagation_summary.num_candidate_evaluations +=
            forward.num_candidate_evaluations +
            backward.num_candidate_evaluations;
        propagation_summary.num_updates +=
            forward.num_updates + backward.num_updates;
        result.summary.num_forward_sweeps++;
        result.summary.num_backward_sweeps++;
        break;
      }
      case PatchMatchPropagationSchedule::kFourDirection: {
        const PatchMatchSweepSummary forward = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kForward, options.cost,
            source_depth_maps);
        const PatchMatchSweepSummary backward = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kBackward, options.cost,
            source_depth_maps);
        const PatchMatchSweepSummary top_down = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kTopDown, options.cost,
            source_depth_maps);
        const PatchMatchSweepSummary bottom_up = RunPatchMatchPropagationSweep(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            PatchMatchSweepDirection::kBottomUp, options.cost,
            source_depth_maps);
        propagation_summary.num_candidate_evaluations +=
            forward.num_candidate_evaluations +
            backward.num_candidate_evaluations +
            top_down.num_candidate_evaluations +
            bottom_up.num_candidate_evaluations;
        propagation_summary.num_updates +=
            forward.num_updates + backward.num_updates +
            top_down.num_updates + bottom_up.num_updates;
        result.summary.num_forward_sweeps++;
        result.summary.num_backward_sweeps++;
        result.summary.num_top_down_sweeps++;
        result.summary.num_bottom_up_sweeps++;
        break;
      }
      case PatchMatchPropagationSchedule::kCheckerboard: {
        // 单组红黑只会让候选传播一格；四组红黑让一次用户迭代的
        // 等效访问量接近四方向传播，同时仍保留棋盘格更新结构。
        for (const int parity : {0, 1, 1, 0, 0, 1, 1, 0}) {
          const PatchMatchSweepSummary checkerboard =
              RunPatchMatchCheckerboardPropagation(
                  reference_view, source_views, reference_image,
                  source_images, &result.depth_map, &result.normal_map,
                  &result.cost_map, parity, options.cost, source_depth_maps);
          propagation_summary.num_candidate_evaluations +=
              checkerboard.num_candidate_evaluations;
          propagation_summary.num_updates += checkerboard.num_updates;
          result.summary.num_checkerboard_passes++;
        }
        break;
      }
    }

    PatchMatchPerturbationOptions perturbation = options.perturbation;
    perturbation.random_seed = options.perturbation.random_seed +
                               static_cast<std::uint32_t>(iteration);
    perturbation.depth_radius =
        std::max(options.min_depth_perturbation, depth_radius);
    perturbation.normal_radius_degrees =
        std::max(options.min_normal_radius_degrees, normal_radius);
    result.summary.depth_radii.push_back(perturbation.depth_radius);
    result.summary.normal_radii_degrees.push_back(
        perturbation.normal_radius_degrees);
    const PatchMatchPerturbationSummary perturbation_summary =
        RunPatchMatchRandomPerturbation(
            reference_view, source_views, reference_image, source_images,
            &result.depth_map, &result.normal_map, &result.cost_map,
            perturbation, options.cost, source_depth_maps);

    const std::size_t iteration_candidate_evaluations =
        propagation_summary.num_candidate_evaluations +
        perturbation_summary.num_candidate_evaluations;
    const std::size_t iteration_updates =
        propagation_summary.num_updates + perturbation_summary.num_updates;
    const double update_ratio =
        iteration_candidate_evaluations == 0
            ? 0.0
            : static_cast<double>(iteration_updates) /
                  static_cast<double>(iteration_candidate_evaluations);
    result.summary.update_ratios.push_back(update_ratio);

    result.summary.num_candidate_evaluations +=
        iteration_candidate_evaluations;
    result.summary.num_propagation_updates +=
        propagation_summary.num_updates;
    result.summary.num_perturbation_updates += perturbation_summary.num_updates;
    result.summary.perturbation_total_abs_depth_update +=
        perturbation_summary.total_abs_depth_update;
    result.summary.perturbation_max_abs_depth_update =
        std::max(result.summary.perturbation_max_abs_depth_update,
                 perturbation_summary.max_abs_depth_update);
    result.summary.perturbation_total_normal_angle_degrees +=
        perturbation_summary.total_normal_angle_degrees;
    result.summary.perturbation_max_normal_angle_degrees =
        std::max(result.summary.perturbation_max_normal_angle_degrees,
                 perturbation_summary.max_normal_angle_degrees);
    result.summary.num_perturbation_geometric_candidate_rejections +=
        perturbation_summary.num_geometric_candidate_rejections;
    result.summary.num_perturbation_geometric_candidate_acceptances +=
        perturbation_summary.num_geometric_candidate_acceptances;
    result.summary.num_perturbation_passes++;

    float shrink = options.perturbation_shrink;
    if (update_ratio < options.adaptive_perturbation_min_update_ratio) {
      shrink *= options.perturbation_shrink;
    }
    depth_radius *= shrink;
    normal_radius *= shrink;
  }

  result.summary.num_iterations = options.num_iterations;
  if (result.summary.num_perturbation_updates > 0) {
    result.summary.perturbation_mean_abs_depth_update =
        result.summary.perturbation_total_abs_depth_update /
        static_cast<double>(result.summary.num_perturbation_updates);
    result.summary.perturbation_mean_normal_angle_degrees =
        result.summary.perturbation_total_normal_angle_degrees /
        static_cast<double>(result.summary.num_perturbation_updates);
  }
  if (options.normal_refinement.enabled) {
    const PatchMatchNormalRefinementSummary normal_refinement_summary =
        RefineNormalMapFromDepth(reference_view, result.depth_map,
                                 options.normal_refinement,
                                 &result.normal_map);
    result.summary.num_normal_refinement_pixels =
        normal_refinement_summary.num_pixels;
    result.summary.num_normal_refined_pixels =
        normal_refinement_summary.num_refined_pixels;
    result.summary.num_normal_refinement_skipped_pixels =
        normal_refinement_summary.num_skipped_pixels;
    result.cost_map = ComputeBestCostMap(
        reference_view, source_views, reference_image, source_images,
        result.depth_map, result.normal_map, options.cost, source_depth_maps);
  }
  result.summary.num_valid_cost_pixels = result.cost_map.ValidPixelCount();
  return result;
}

PatchMatchStereoResult RunPatchMatchStereoForImageImpl(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>* source_depth_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options) {
  if (options.pyramid_levels <= 1) {
    PatchMatchStereoResult result = RunPatchMatchStereoForImageSingleScale(
        reference_view, source_views, reference_image, source_images,
        source_depth_maps, depth_min, depth_max, options, nullptr);
    result.summary.num_pyramid_levels = 1;
    return result;
  }

  std::size_t num_levels = options.pyramid_levels;
  while (num_levels > 1) {
    const std::size_t divisor = PyramidLevelDivisor(num_levels, 0);
    const std::size_t width = PyramidLevelSize(reference_image.Width(), divisor);
    const std::size_t height =
        PyramidLevelSize(reference_image.Height(), divisor);
    if (width >= reference_image.Width() && height >= reference_image.Height()) {
      --num_levels;
      continue;
    }
    break;
  }
  if (num_levels <= 1) {
    PatchMatchStereoResult result = RunPatchMatchStereoForImageSingleScale(
        reference_view, source_views, reference_image, source_images,
        source_depth_maps, depth_min, depth_max, options, nullptr);
    result.summary.num_pyramid_levels = 1;
    return result;
  }

  PatchMatchMaps prior_maps;
  bool has_prior = false;
  PatchMatchStereoResult final_result;
  PatchMatchStereoSummary accumulated_summary;
  accumulated_summary.num_pyramid_levels = num_levels;

  for (std::size_t level_idx = 0; level_idx < num_levels; ++level_idx) {
    const std::size_t divisor = PyramidLevelDivisor(num_levels, level_idx);
    const std::size_t reference_width =
        PyramidLevelSize(reference_image.Width(), divisor);
    const std::size_t reference_height =
        PyramidLevelSize(reference_image.Height(), divisor);
    const double reference_scale_x =
        static_cast<double>(reference_width) /
        static_cast<double>(reference_image.Width());
    const double reference_scale_y =
        static_cast<double>(reference_height) /
        static_cast<double>(reference_image.Height());

    const MvsImage level_reference_image =
        ResizeImageForPyramidLevel(reference_image, reference_width,
                                   reference_height);
    const PatchMatchView level_reference_view =
        ScaleViewForPyramidLevel(reference_view, reference_scale_x,
                                 reference_scale_y, reference_width,
                                 reference_height);

    std::vector<MvsImage> level_source_images;
    std::vector<PatchMatchView> level_source_views;
    level_source_images.reserve(source_images.size());
    level_source_views.reserve(source_views.size());
    for (std::size_t i = 0; i < source_images.size(); ++i) {
      const std::size_t source_width =
          PyramidLevelSize(source_images[i].Width(), divisor);
      const std::size_t source_height =
          PyramidLevelSize(source_images[i].Height(), divisor);
      const double source_scale_x =
          static_cast<double>(source_width) /
          static_cast<double>(source_images[i].Width());
      const double source_scale_y =
          static_cast<double>(source_height) /
          static_cast<double>(source_images[i].Height());
      level_source_images.push_back(
          ResizeImageForPyramidLevel(source_images[i], source_width,
                                     source_height));
      level_source_views.push_back(ScaleViewForPyramidLevel(
          source_views[i], source_scale_x, source_scale_y, source_width,
          source_height));
    }

    std::vector<DenseDepthMap> level_source_depth_maps;
    const std::vector<DenseDepthMap>* level_source_depth_maps_ptr = nullptr;
    if (source_depth_maps != nullptr) {
      level_source_depth_maps.reserve(source_depth_maps->size());
      for (std::size_t i = 0; i < source_depth_maps->size(); ++i) {
        level_source_depth_maps.push_back(ResizeDepthMapNearest(
            source_depth_maps->at(i), level_source_images[i].Width(),
            level_source_images[i].Height()));
      }
      level_source_depth_maps_ptr = &level_source_depth_maps;
    }

    PatchMatchMaps level_initial_maps;
    const PatchMatchMaps* level_initial_maps_ptr = nullptr;
    if (has_prior) {
      level_initial_maps = ResizePatchMatchMapsNearest(
          prior_maps, reference_width, reference_height);
      level_initial_maps.depth_map.SetDepthRange(depth_min, depth_max);
      level_initial_maps_ptr = &level_initial_maps;
    }

    PatchMatchStereoResult level_result =
        RunPatchMatchStereoForImageSingleScale(
            level_reference_view, level_source_views, level_reference_image,
            level_source_images, level_source_depth_maps_ptr, depth_min,
            depth_max, options, level_initial_maps_ptr);

    accumulated_summary.num_iterations += level_result.summary.num_iterations;
    accumulated_summary.num_forward_sweeps +=
        level_result.summary.num_forward_sweeps;
    accumulated_summary.num_backward_sweeps +=
        level_result.summary.num_backward_sweeps;
    accumulated_summary.num_top_down_sweeps +=
        level_result.summary.num_top_down_sweeps;
    accumulated_summary.num_bottom_up_sweeps +=
        level_result.summary.num_bottom_up_sweeps;
    accumulated_summary.num_checkerboard_passes +=
        level_result.summary.num_checkerboard_passes;
    accumulated_summary.num_perturbation_passes +=
        level_result.summary.num_perturbation_passes;
    accumulated_summary.num_candidate_evaluations +=
        level_result.summary.num_candidate_evaluations;
    accumulated_summary.num_propagation_updates +=
        level_result.summary.num_propagation_updates;
    accumulated_summary.num_perturbation_updates +=
        level_result.summary.num_perturbation_updates;
    accumulated_summary.perturbation_total_abs_depth_update +=
        level_result.summary.perturbation_total_abs_depth_update;
    accumulated_summary.perturbation_max_abs_depth_update =
        std::max(accumulated_summary.perturbation_max_abs_depth_update,
                 level_result.summary.perturbation_max_abs_depth_update);
    accumulated_summary.perturbation_total_normal_angle_degrees +=
        level_result.summary.perturbation_total_normal_angle_degrees;
    accumulated_summary.perturbation_max_normal_angle_degrees =
        std::max(accumulated_summary.perturbation_max_normal_angle_degrees,
                 level_result.summary.perturbation_max_normal_angle_degrees);
    accumulated_summary.num_perturbation_geometric_candidate_rejections +=
        level_result.summary.num_perturbation_geometric_candidate_rejections;
    accumulated_summary.num_perturbation_geometric_candidate_acceptances +=
        level_result.summary.num_perturbation_geometric_candidate_acceptances;
    accumulated_summary.num_normal_refinement_pixels =
        level_result.summary.num_normal_refinement_pixels;
    accumulated_summary.num_normal_refined_pixels =
        level_result.summary.num_normal_refined_pixels;
    accumulated_summary.num_normal_refinement_skipped_pixels =
        level_result.summary.num_normal_refinement_skipped_pixels;
    accumulated_summary.update_ratios.insert(
        accumulated_summary.update_ratios.end(),
        level_result.summary.update_ratios.begin(),
        level_result.summary.update_ratios.end());
    accumulated_summary.depth_radii.insert(
        accumulated_summary.depth_radii.end(),
        level_result.summary.depth_radii.begin(),
        level_result.summary.depth_radii.end());
    accumulated_summary.normal_radii_degrees.insert(
        accumulated_summary.normal_radii_degrees.end(),
        level_result.summary.normal_radii_degrees.begin(),
        level_result.summary.normal_radii_degrees.end());

    prior_maps.depth_map = level_result.depth_map;
    prior_maps.normal_map = level_result.normal_map;
    has_prior = true;
    final_result = std::move(level_result);
  }

  if (accumulated_summary.num_perturbation_updates > 0) {
    accumulated_summary.perturbation_mean_abs_depth_update =
        accumulated_summary.perturbation_total_abs_depth_update /
        static_cast<double>(accumulated_summary.num_perturbation_updates);
    accumulated_summary.perturbation_mean_normal_angle_degrees =
        accumulated_summary.perturbation_total_normal_angle_degrees /
        static_cast<double>(accumulated_summary.num_perturbation_updates);
  }
  accumulated_summary.num_valid_cost_pixels =
      final_result.cost_map.ValidPixelCount();
  final_result.summary = std::move(accumulated_summary);
  return final_result;
}

PatchMatchStereoResult RunPatchMatchStereoForImage(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options) {
  return RunPatchMatchStereoForImageImpl(reference_view, source_views,
                                         reference_image, source_images,
                                         nullptr, depth_min, depth_max,
                                         options);
}

PatchMatchStereoResult RunPatchMatchStereoForImageWithSourceDepths(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>& source_depth_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options) {
  return RunPatchMatchStereoForImageImpl(
      reference_view, source_views, reference_image, source_images,
      &source_depth_maps, depth_min, depth_max, options);
}

PatchMatchPhotometricStats ComputePatchMatchPhotometricStats(
    const DenseCostMap& cost_map) {
  PatchMatchPhotometricStats stats;
  stats.num_pixels = cost_map.NumPixels();

  std::vector<float> valid_costs;
  valid_costs.reserve(cost_map.NumPixels());
  for (std::size_t row = 0; row < cost_map.Height(); ++row) {
    for (std::size_t col = 0; col < cost_map.Width(); ++col) {
      if (cost_map.IsValid(row, col)) {
        valid_costs.push_back(cost_map.Get(row, col));
      }
    }
  }

  stats.num_valid_cost_pixels = valid_costs.size();
  stats.num_invalid_cost_pixels = stats.num_pixels - valid_costs.size();
  if (stats.num_pixels > 0) {
    stats.valid_ratio = static_cast<double>(stats.num_valid_cost_pixels) /
                        static_cast<double>(stats.num_pixels);
  }
  if (valid_costs.empty()) {
    return stats;
  }

  std::sort(valid_costs.begin(), valid_costs.end());
  double total_cost = 0.0;
  for (const float cost : valid_costs) {
    total_cost += cost;
  }
  stats.mean_cost = total_cost / static_cast<double>(valid_costs.size());
  stats.min_cost = valid_costs.front();
  stats.max_cost = valid_costs.back();
  const std::size_t middle = valid_costs.size() / 2;
  if (valid_costs.size() % 2 == 0) {
    stats.median_cost =
        0.5 * (static_cast<double>(valid_costs[middle - 1]) +
               static_cast<double>(valid_costs[middle]));
  } else {
    stats.median_cost = valid_costs[middle];
  }
  return stats;
}

PatchMatchPhotometricFilterSummary FilterPatchMatchPhotometricResults(
    float max_cost,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map) {
  if (depth_map == nullptr || normal_map == nullptr || cost_map == nullptr) {
    throw std::invalid_argument(
        "Photometric filtering requires non-null maps.");
  }
  if (!std::isfinite(max_cost) || max_cost < 0.0F) {
    throw std::invalid_argument("Photometric filtering max cost is invalid.");
  }
  if (depth_map->Width() != normal_map->Width() ||
      depth_map->Height() != normal_map->Height() ||
      depth_map->Width() != cost_map->Width() ||
      depth_map->Height() != cost_map->Height()) {
    throw std::invalid_argument(
        "Depth, normal and cost maps must have identical sizes.");
  }

  PatchMatchPhotometricFilterSummary summary;
  summary.num_pixels = cost_map->NumPixels();
  for (std::size_t row = 0; row < cost_map->Height(); ++row) {
    for (std::size_t col = 0; col < cost_map->Width(); ++col) {
      const bool keep = depth_map->IsValid(row, col) &&
                        normal_map->IsValid(row, col) &&
                        cost_map->IsValid(row, col) &&
                        cost_map->Get(row, col) <= max_cost;
      if (keep) {
        summary.num_kept_pixels++;
        continue;
      }

      depth_map->Set(row, col, 0.0F);
      normal_map->Set(row, col, Eigen::Vector3f::Zero());
      cost_map->Set(row, col, std::numeric_limits<float>::infinity());
      summary.num_filtered_pixels++;
    }
  }
  return summary;
}

ConsistencyGraph ComputeGeometricConsistencyGraph(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const DenseDepthMap& reference_depth_map,
    const std::vector<DenseDepthMap>& source_depth_maps,
    const PatchMatchGeometricConsistencyOptions& options) {
  if (source_views.size() != source_depth_maps.size()) {
    throw std::invalid_argument(
        "Source views and source depth maps must have identical sizes.");
  }
  if (!std::isfinite(options.max_reprojection_error_pixels) ||
      !std::isfinite(options.max_relative_depth_error) ||
      options.max_reprojection_error_pixels < 0.0F ||
      options.max_relative_depth_error < 0.0F) {
    throw std::invalid_argument(
        "Geometric consistency thresholds are invalid.");
  }

  ConsistencyGraph graph(reference_depth_map.Width(),
                         reference_depth_map.Height());
  for (std::size_t row = 0; row < reference_depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < reference_depth_map.Width(); ++col) {
      if (!reference_depth_map.IsValid(row, col)) {
        continue;
      }
      const float reference_depth = reference_depth_map.Get(row, col);
      const Eigen::Vector2f reference_xy(static_cast<float>(col),
                                         static_cast<float>(row));

      for (std::size_t source_idx = 0; source_idx < source_views.size();
           ++source_idx) {
        const std::optional<Eigen::Vector2f> source_xy =
            ProjectDepthToSourcePixel(reference_view, source_views[source_idx],
                                      reference_xy, reference_depth);
        if (!source_xy.has_value()) {
          continue;
        }
        const std::optional<float> source_depth = SampleDepthBilinear(
            source_depth_maps[source_idx], source_xy->x(), source_xy->y());
        if (!source_depth.has_value()) {
          continue;
        }

        try {
          const Eigen::Vector3d point_source = UnprojectPixel(
              source_views[source_idx].camera, source_xy->cast<double>(),
              static_cast<double>(*source_depth));
          const Eigen::Vector3d point_world =
              source_views[source_idx]
                  .cam_from_world.Inverse()
                  .TransformPoint(point_source);
          const Eigen::Vector3d point_reference =
              reference_view.cam_from_world.TransformPoint(point_world);
          if (point_reference.z() <= 0.0) {
            continue;
          }
          const Eigen::Vector2f reprojected_reference_xy =
              ProjectPoint(reference_view.camera, point_reference)
                  .cast<float>();
          const float reprojection_error =
              (reprojected_reference_xy - reference_xy).norm();
          const float relative_depth_error = static_cast<float>(
              std::abs(point_reference.z() -
                       static_cast<double>(reference_depth)) /
              static_cast<double>(reference_depth));
          if (reprojection_error <= options.max_reprojection_error_pixels &&
              relative_depth_error <= options.max_relative_depth_error) {
            graph.Add(row, col, source_views[source_idx].image_id);
          }
        } catch (const std::exception&) {
          continue;
        }
      }
    }
  }
  return graph;
}

PatchMatchGeometricFilterSummary FilterPatchMatchGeometricResults(
    const ConsistencyGraph& consistency_graph,
    std::size_t min_num_consistent_sources,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map) {
  if (depth_map == nullptr || normal_map == nullptr || cost_map == nullptr) {
    throw std::invalid_argument("Geometric filtering requires non-null maps.");
  }
  if (depth_map->Width() != consistency_graph.Width() ||
      depth_map->Height() != consistency_graph.Height() ||
      normal_map->Width() != consistency_graph.Width() ||
      normal_map->Height() != consistency_graph.Height() ||
      cost_map->Width() != consistency_graph.Width() ||
      cost_map->Height() != consistency_graph.Height()) {
    throw std::invalid_argument(
        "Geometric consistency graph and maps must have identical sizes.");
  }

  PatchMatchGeometricFilterSummary summary;
  summary.num_pixels = consistency_graph.NumPixels();
  for (std::size_t row = 0; row < consistency_graph.Height(); ++row) {
    for (std::size_t col = 0; col < consistency_graph.Width(); ++col) {
      const bool keep = depth_map->IsValid(row, col) &&
                        normal_map->IsValid(row, col) &&
                        cost_map->IsValid(row, col) &&
                        consistency_graph.Count(row, col) >=
                            min_num_consistent_sources;
      if (keep) {
        summary.num_kept_pixels++;
        continue;
      }
      depth_map->Set(row, col, 0.0F);
      normal_map->Set(row, col, Eigen::Vector3f::Zero());
      cost_map->Set(row, col, std::numeric_limits<float>::infinity());
      summary.num_filtered_pixels++;
    }
  }
  return summary;
}

StereoFusionResult FuseStereoDepthMaps(
    const std::vector<PatchMatchView>& views,
    const std::vector<MvsImage>& images,
    const std::vector<DenseDepthMap>& depth_maps,
    const std::vector<NormalMap>& normal_maps,
    const std::vector<DenseCostMap>& cost_maps,
    const std::vector<ConsistencyGraph>& consistency_graphs,
    const StereoFusionOptions& options) {
  if (views.size() != images.size() || views.size() != depth_maps.size() ||
      views.size() != normal_maps.size() || views.size() != cost_maps.size()) {
    throw std::invalid_argument(
        "Stereo fusion inputs must have identical image counts.");
  }
  if (options.require_consistency_graph &&
      consistency_graphs.size() != views.size()) {
    throw std::invalid_argument(
        "Stereo fusion requires one consistency graph per reference image.");
  }
  if (!consistency_graphs.empty() &&
      consistency_graphs.size() != views.size()) {
      throw std::invalid_argument(
        "Stereo fusion consistency graph count must match image count.");
  }
  if (!std::isfinite(options.max_reprojection_error_pixels) ||
      !std::isfinite(options.max_relative_depth_error) ||
      !std::isfinite(options.max_normal_error_degrees) ||
      !std::isfinite(options.max_view_angle_degrees) ||
      options.max_reprojection_error_pixels < 0.0F ||
      options.max_relative_depth_error < 0.0F ||
      options.max_normal_error_degrees < 0.0F ||
      options.max_view_angle_degrees < 0.0F ||
      options.max_normal_error_degrees > 180.0F ||
      options.max_view_angle_degrees > 180.0F ||
      options.min_num_fused_observations == 0 ||
      options.max_num_fused_observations == 0 ||
      options.max_num_fused_observations <
          options.min_num_fused_observations) {
    throw std::invalid_argument("Stereo fusion options are invalid.");
  }

  StereoFusionResult result;
  result.summary.num_input_images = views.size();

  std::unordered_map<ImageId, std::size_t> image_idx_by_id;
  image_idx_by_id.reserve(views.size());
  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    const ConsistencyGraph* graph =
        consistency_graphs.empty() ? nullptr : &consistency_graphs[image_idx];
    RequireCompatibleFusionMaps(images[image_idx], depth_maps[image_idx],
                                normal_maps[image_idx], cost_maps[image_idx],
                                graph);
    image_idx_by_id[views[image_idx].image_id] = image_idx;
  }
  std::map<std::pair<ImageId, ImageId>, StereoFusionPairDiagnostics>
      pair_diagnostics_by_pair;

  std::vector<std::vector<std::uint8_t>> visited(views.size());
  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    visited[image_idx].assign(depth_maps[image_idx].NumPixels(), 0);
  }
  const auto offset_of = [&depth_maps](std::size_t image_idx, std::size_t row,
                                       std::size_t col) {
    return row * depth_maps[image_idx].Width() + col;
  };

  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    for (std::size_t row = 0; row < depth_maps[image_idx].Height(); ++row) {
      for (std::size_t col = 0; col < depth_maps[image_idx].Width(); ++col) {
        if (!depth_maps[image_idx].IsValid(row, col)) {
          continue;
        }
        result.summary.num_depth_pixels++;

        if (visited[image_idx][offset_of(image_idx, row, col)] != 0) {
          continue;
        }
        if (!IsCompleteFusionPixel(images, depth_maps, normal_maps, cost_maps,
                                   consistency_graphs, options, image_idx, row,
                                   col)) {
          result.summary.num_skipped_pixels++;
          continue;
        }
        const std::optional<FusionPixelData> seed_data = MakeFusionPixelData(
            views, images, depth_maps, normal_maps, image_idx, row, col);
        if (!seed_data.has_value()) {
          result.summary.num_skipped_pixels++;
          continue;
        }

        std::vector<FusionObservation> track;
        track.reserve(options.min_num_fused_observations);
        track.push_back({image_idx, row, col, 0});

        for (std::size_t cursor = 0;
             cursor < track.size() &&
             track.size() < options.max_num_fused_observations;
             ++cursor) {
          const FusionObservation current = track[cursor];
          if (current.traversal_depth >= options.max_traversal_depth ||
              consistency_graphs.empty()) {
            continue;
          }
          const std::vector<ImageId>& source_image_ids =
              consistency_graphs[current.image_idx].SourceImageIds(
                  current.row, current.col);
          for (const ImageId source_image_id : source_image_ids) {
            const auto source_it = image_idx_by_id.find(source_image_id);
            if (source_it == image_idx_by_id.end()) {
              continue;
            }
            const std::size_t source_idx = source_it->second;
            StereoFusionPairDiagnostics& pair_diagnostics =
                pair_diagnostics_by_pair[{views[current.image_idx].image_id,
                                          views[source_idx].image_id}];
            pair_diagnostics.reference_image_id =
                views[current.image_idx].image_id;
            pair_diagnostics.source_image_id = views[source_idx].image_id;
            pair_diagnostics.num_candidates++;

            FusionRejectReason reject_reason = FusionRejectReason::kNone;
            const std::optional<FusionObservation> source =
                ProjectFusionObservationToSource(
                    views, depth_maps, normal_maps, cost_maps,
                    consistency_graphs, options, current, source_idx,
                    seed_data->xyz, seed_data->normal, image_idx,
                    &result.error_samples,
                    &reject_reason);
            if (!source.has_value()) {
              switch (reject_reason) {
                case FusionRejectReason::kNormal:
                  result.summary.num_rejected_normal_observations++;
                  pair_diagnostics.num_rejected_normal++;
                  break;
                case FusionRejectReason::kView:
                  result.summary.num_rejected_view_observations++;
                  pair_diagnostics.num_rejected_view++;
                  break;
                case FusionRejectReason::kProjection:
                  pair_diagnostics.num_rejected_projection++;
                  pair_diagnostics.num_rejected_other++;
                  break;
                case FusionRejectReason::kDepth:
                  pair_diagnostics.num_rejected_depth++;
                  pair_diagnostics.num_rejected_other++;
                  break;
                case FusionRejectReason::kMutualConsistency:
                  pair_diagnostics.num_rejected_mutual_consistency++;
                  pair_diagnostics.num_rejected_other++;
                  break;
                case FusionRejectReason::kInvalid:
                case FusionRejectReason::kNone:
                  pair_diagnostics.num_rejected_invalid++;
                  pair_diagnostics.num_rejected_other++;
                  break;
              }
              continue;
            }
            if (visited[source->image_idx][offset_of(source->image_idx,
                                                     source->row,
                                                     source->col)] != 0) {
              pair_diagnostics.num_rejected_visited++;
              pair_diagnostics.num_rejected_other++;
              continue;
            }
            if (ContainsObservation(track, source->image_idx, source->row,
                                    source->col)) {
              pair_diagnostics.num_rejected_duplicate++;
              pair_diagnostics.num_rejected_other++;
              continue;
            }
            if (!IsCompleteFusionPixel(images, depth_maps, normal_maps,
                                       cost_maps, consistency_graphs, options,
                                       source->image_idx, source->row,
                                       source->col)) {
              pair_diagnostics.num_rejected_incomplete++;
              pair_diagnostics.num_rejected_other++;
              continue;
            }
            pair_diagnostics.num_accepted++;
            track.push_back(*source);
            if (track.size() >= options.max_num_fused_observations) {
              break;
            }
          }
        }

        if (track.size() < options.min_num_fused_observations) {
          result.summary.num_rejected_tracks++;
          result.summary.num_skipped_pixels++;
          continue;
        }

        Eigen::Vector3d xyz_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal_sum = Eigen::Vector3d::Zero();
        Eigen::Vector3d color_sum = Eigen::Vector3d::Zero();
        bool track_valid = true;
        for (const FusionObservation& observation : track) {
          const std::optional<FusionPixelData> data = MakeFusionPixelData(
              views, images, depth_maps, normal_maps, observation.image_idx,
              observation.row, observation.col);
          if (!data.has_value()) {
            track_valid = false;
            break;
          }
          xyz_sum += data->xyz;
          normal_sum += data->normal;
          color_sum += data->color;
        }
        if (!track_valid || !normal_sum.allFinite() ||
            normal_sum.squaredNorm() <= 0.0) {
          result.summary.num_rejected_tracks++;
          result.summary.num_skipped_pixels++;
          continue;
        }

        const double inv_track_size =
            1.0 / static_cast<double>(track.size());
        Eigen::Vector3d normal = normal_sum.normalized();
        Eigen::Vector3d color = color_sum * inv_track_size;
        color.x() = std::round(ClampFloat(static_cast<float>(color.x()), 0.0F,
                                          255.0F));
        color.y() = std::round(ClampFloat(static_cast<float>(color.y()), 0.0F,
                                          255.0F));
        color.z() = std::round(ClampFloat(static_cast<float>(color.z()), 0.0F,
                                          255.0F));

        FusedPoint fused_point;
        fused_point.xyz = xyz_sum * inv_track_size;
        fused_point.normal = normal.cast<float>();
        fused_point.color = Eigen::Matrix<std::uint8_t, 3, 1>(
            static_cast<std::uint8_t>(color.x()),
            static_cast<std::uint8_t>(color.y()),
            static_cast<std::uint8_t>(color.z()));
        fused_point.image_id = views[image_idx].image_id;
        fused_point.row = row;
        fused_point.col = col;
        result.points.push_back(fused_point);

        StereoFusionTrack track_diagnostics;
        track_diagnostics.point_idx = result.points.size() - 1;
        track_diagnostics.seed_image_id = views[image_idx].image_id;
        track_diagnostics.seed_row = row;
        track_diagnostics.seed_col = col;
        track_diagnostics.num_observations = track.size();
        std::unordered_set<ImageId> unique_image_ids;
        for (const FusionObservation& observation : track) {
          unique_image_ids.insert(views[observation.image_idx].image_id);
          track_diagnostics.max_traversal_depth = std::max(
              track_diagnostics.max_traversal_depth,
              observation.traversal_depth);
          track_diagnostics.max_normal_angle_degrees = std::max(
              track_diagnostics.max_normal_angle_degrees,
              observation.normal_angle_degrees);
          track_diagnostics.max_view_angle_degrees = std::max(
              track_diagnostics.max_view_angle_degrees,
              observation.view_angle_degrees);
        }
        track_diagnostics.num_unique_images = unique_image_ids.size();
        result.tracks.push_back(track_diagnostics);

        result.summary.num_fusion_tracks++;
        result.summary.num_fused_observations += track.size();

        for (const FusionObservation& observation : track) {
          const std::size_t offset =
              offset_of(observation.image_idx, observation.row,
                        observation.col);
          if (visited[observation.image_idx][offset] == 0) {
            visited[observation.image_idx][offset] = 1;
            result.summary.num_visited_pixels++;
          }
        }
      }
    }
  }

  result.summary.num_fused_points = result.points.size();
  result.pair_diagnostics.reserve(pair_diagnostics_by_pair.size());
  for (const auto& [unused_key, diagnostics] : pair_diagnostics_by_pair) {
    (void)unused_key;
    result.pair_diagnostics.push_back(diagnostics);
  }
  return result;
}

}  // namespace colite::mvs
