#include "colite/mvs/patch_match.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "colite/core/camera.h"
#include "colite/core/parallel.h"
#include "colite/mvs/patch_match_cuda.h"

namespace colite::mvs {

std::string_view PatchMatchBackendToName(PatchMatchBackend backend) {
  switch (backend) {
    case PatchMatchBackend::kAuto:
      return "auto";
    case PatchMatchBackend::kCpu:
      return "cpu";
    case PatchMatchBackend::kCuda:
      return "cuda";
  }
  return "unknown";
}

PatchMatchBackend PatchMatchBackendFromName(std::string_view name) {
  if (name == "auto") {
    return PatchMatchBackend::kAuto;
  }
  if (name == "cpu") {
    return PatchMatchBackend::kCpu;
  }
  if (name == "cuda" || name == "gpu") {
    return PatchMatchBackend::kCuda;
  }
  throw std::invalid_argument("Invalid PatchMatch backend: " +
                              std::string(name));
}

bool IsCudaPatchMatchCompiled() {
#ifdef COLITE_ENABLE_CUDA_MVS
  return true;
#else
  return false;
#endif
}

bool IsCudaPatchMatchAvailable() {
  if (!IsCudaPatchMatchCompiled()) {
    return false;
  }
  std::string unavailable_reason;
  const std::optional<PatchMatchStereoResult> result =
      RunCudaPatchMatchStereo({}, &unavailable_reason);
  (void)result;
  return unavailable_reason.empty();
}

std::string CudaPatchMatchUnavailableReason() {
  if (!IsCudaPatchMatchCompiled()) {
    return "CUDA PatchMatch backend was not compiled in this build.";
  }
  std::string unavailable_reason;
  const std::optional<PatchMatchStereoResult> result =
      RunCudaPatchMatchStereo({}, &unavailable_reason);
  (void)result;
  if (unavailable_reason.empty()) {
    return {};
  }
  return unavailable_reason;
}

namespace {

constexpr char kConsistencyGraphBinaryMagic[] =
    "COLITE_CONSISTENCY_GRAPH_BINARY";
constexpr int kConsistencyGraphBinaryVersion = 2;

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

PatchMatchCost ComputeZnccFromSamples(
    const std::vector<float>& reference_values,
    const std::vector<float>& source_values,
    const std::vector<Eigen::Vector2i>& sample_offsets,
    float reference_center_value,
    const PatchMatchCostOptions& options) {
  if (reference_values.empty() ||
      reference_values.size() != source_values.size() ||
      reference_values.size() != sample_offsets.size()) {
    return InvalidCost(options);
  }

  const double sigma_spatial =
      options.sigma_spatial > 0.0F
          ? static_cast<double>(options.sigma_spatial)
          : static_cast<double>(std::max(1, options.window_radius));
  const double sigma_color = static_cast<double>(options.sigma_color);
  const double spatial_normalization =
      1.0 / (2.0 * sigma_spatial * sigma_spatial);
  const double color_normalization =
      1.0 / (2.0 * sigma_color * sigma_color);

  double weight_sum = 0.0;
  double reference_mean = 0.0;
  double source_mean = 0.0;
  double reference_squared_mean = 0.0;
  double source_squared_mean = 0.0;
  double cross_mean = 0.0;
  for (std::size_t i = 0; i < reference_values.size(); ++i) {
    const double dx = static_cast<double>(sample_offsets[i].x());
    const double dy = static_cast<double>(sample_offsets[i].y());
    const double color_difference =
        static_cast<double>(reference_values[i] - reference_center_value);
    // 与 COLMAP 一致，权重只依赖参考图，从而对所有候选使用同一参考边缘结构。
    const double weight =
        options.bilateral_weighting
            ? std::exp(-(dx * dx + dy * dy) * spatial_normalization -
                       color_difference * color_difference *
                           color_normalization)
            : 1.0;
    const double reference_value = reference_values[i];
    const double source_value = source_values[i];
    weight_sum += weight;
    reference_mean += weight * reference_value;
    source_mean += weight * source_value;
    reference_squared_mean += weight * reference_value * reference_value;
    source_squared_mean += weight * source_value * source_value;
    cross_mean += weight * reference_value * source_value;
  }
  if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
    return InvalidCost(options);
  }
  const double inverse_weight_sum = 1.0 / weight_sum;
  reference_mean *= inverse_weight_sum;
  source_mean *= inverse_weight_sum;
  reference_squared_mean *= inverse_weight_sum;
  source_squared_mean *= inverse_weight_sum;
  cross_mean *= inverse_weight_sum;

  const double reference_variance =
      reference_squared_mean - reference_mean * reference_mean;
  const double source_variance =
      source_squared_mean - source_mean * source_mean;

  if (reference_variance <= options.min_variance ||
      source_variance <= options.min_variance) {
    return InvalidCost(options);
  }

  const double covariance = cross_mean - reference_mean * source_mean;
  const double denominator =
      std::sqrt(reference_variance * source_variance);
  const float score = static_cast<float>(
      std::max(-1.0, std::min(1.0, covariance / denominator)));

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
                                 const DenseCostMap* cost_map,
                                 const ConsistencyGraph* graph) {
  if (depth_map.Width() != normal_map.Width() ||
      depth_map.Height() != normal_map.Height() ||
      depth_map.Width() != reference_image.Width() ||
      depth_map.Height() != reference_image.Height()) {
    throw std::invalid_argument(
        "Reference image, depth map and normal map must have identical sizes.");
  }
  if (cost_map != nullptr &&
      (cost_map->Width() != depth_map.Width() ||
       cost_map->Height() != depth_map.Height())) {
    throw std::invalid_argument(
        "Cost map and fusion maps must have identical sizes.");
  }
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

struct FusionPixelData {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  Eigen::Matrix<std::uint8_t, 3, 1> color =
      Eigen::Matrix<std::uint8_t, 3, 1>(255, 255, 255);
};

struct FusionViewGeometry {
  bool pinhole = false;
  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  Eigen::Matrix3d cam_from_world_rotation = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d world_from_cam_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d cam_from_world_translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d camera_center = Eigen::Vector3d::Zero();
};

FusionViewGeometry MakeFusionViewGeometry(const PatchMatchView& view) {
  FusionViewGeometry geometry;
  geometry.cam_from_world_rotation =
      view.cam_from_world.q_cw.toRotationMatrix();
  geometry.world_from_cam_rotation =
      geometry.cam_from_world_rotation.transpose();
  geometry.cam_from_world_translation = view.cam_from_world.t_cw;
  geometry.camera_center = view.cam_from_world.CameraCenter();
  if (view.camera.model_id == CameraModelId::kPinhole &&
      view.camera.params.size() >= 4) {
    geometry.pinhole = true;
    geometry.fx = view.camera.params[0];
    geometry.fy = view.camera.params[1];
    geometry.cx = view.camera.params[2];
    geometry.cy = view.camera.params[3];
  } else if (view.camera.model_id == CameraModelId::kSimplePinhole &&
             view.camera.params.size() >= 3) {
    geometry.pinhole = true;
    geometry.fx = view.camera.params[0];
    geometry.fy = view.camera.params[0];
    geometry.cx = view.camera.params[1];
    geometry.cy = view.camera.params[2];
  }
  return geometry;
}

struct FusionObservation {
  std::size_t image_idx = 0;
  std::size_t row = 0;
  std::size_t col = 0;
  std::size_t traversal_depth = 0;
  double normal_cosine = 1.0;
  double view_cosine = 1.0;
  FusionPixelData data;
};

struct FusionTaskResult {
  StereoFusionResult result;
  std::vector<StereoFusionPairDiagnostics> pair_diagnostics;
  std::vector<std::uint8_t> pair_diagnostics_seen;
  // A worker processes only one stripe at a time. Reuse its temporary storage
  // across seed pixels instead of allocating one BFS track and nine median
  // vectors for every candidate point.
  std::vector<FusionObservation> track_scratch;
  std::vector<double> median_scratch;
  // Direct weak index avoids a linear hot-set scan and reuses views retained
  // by another worker without entering the persisted-view LRU mutex. Keep a
  // bounded strong hot set so this acceleration does not pin every dense map.
  std::vector<std::weak_ptr<const StereoFusionViewData>> view_index;
  std::vector<
      std::pair<std::size_t,
                std::shared_ptr<const StereoFusionViewData>>>
      view_hot_cache;
};

enum class FusionRejectReason {
  kNone,
  kVisited,
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
    case FusionRejectReason::kVisited:
      return 1;
    case FusionRejectReason::kProjection:
      return 2;
    case FusionRejectReason::kInvalid:
      return 3;
    case FusionRejectReason::kMutualConsistency:
      return 4;
    case FusionRejectReason::kDepth:
      return 5;
    case FusionRejectReason::kNormal:
    case FusionRejectReason::kView:
      return 6;
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
  constexpr std::size_t kMaxSamplesPerWorker = 100000;
  if (samples->size() >= kMaxSamplesPerWorker) {
    return;
  }
  samples->push_back(value);
}

double FusionMedian(std::vector<double>* values) {
  if (values == nullptr || values->empty()) {
    return 0.0;
  }
  // Match COLMAP's percentile implementation without fully sorting every
  // coordinate, normal, and color component of every fusion track.
  const std::size_t right = values->size() / 2;
  std::nth_element(values->begin(), values->begin() + right, values->end());
  const double right_value = (*values)[right];
  if (values->size() % 2 != 0) {
    return right_value;
  }
  const double left_value =
      *std::max_element(values->begin(), values->begin() + right);
  return 0.5 * (left_value + right_value);
}

template <typename ImageIds>
bool ContainsImageId(const ImageIds& image_ids, ImageId image_id) {
  return std::find(image_ids.begin(), image_ids.end(), image_id) !=
         image_ids.end();
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

double CosineBetween(const Eigen::Vector3d& lhs,
                     const Eigen::Vector3d& rhs) {
  if (!lhs.allFinite() || !rhs.allFinite() || lhs.squaredNorm() <= 0.0 ||
      rhs.squaredNorm() <= 0.0) {
    return -1.0;
  }
  return std::max(-1.0, std::min(1.0, lhs.dot(rhs) /
      std::sqrt(lhs.squaredNorm() * rhs.squaredNorm())));
}

double DegreesFromCosine(double cosine) {
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  return std::acos(std::max(-1.0, std::min(1.0, cosine))) *
         kRadiansToDegrees;
}

bool IsCompleteFusionPixel(const StereoFusionViewData& view_data,
                           const StereoFusionOptions& options,
                           std::size_t row,
                           std::size_t col) {
  if (!view_data.depth_map.IsValid(row, col) ||
      !view_data.normal_map.IsValid(row, col) ||
      (options.require_cost_map &&
       !view_data.cost_map.IsValid(row, col))) {
    return false;
  }
  if (!options.prefer_fallback_source_images &&
      (options.require_consistency_graph ||
       !view_data.consistency_graph.Empty())) {
    if (view_data.consistency_graph.Count(row, col) <
        options.min_num_consistent_sources) {
      return false;
    }
  }
  return view_data.image.Width() == view_data.depth_map.Width() &&
         view_data.image.Height() == view_data.depth_map.Height();
}

std::optional<FusionPixelData> MakeFusionPixelData(
    const PatchMatchView& view,
    const FusionViewGeometry& geometry,
    const StereoFusionViewData& view_data,
    std::size_t row,
    std::size_t col,
    const Eigen::Vector3f* precomputed_normal_world = nullptr) {
  try {
    const float depth = view_data.depth_map.Get(row, col);
    Eigen::Vector3d point_camera;
    if (geometry.pinhole) {
      point_camera =
          Eigen::Vector3d((static_cast<double>(col) - geometry.cx) /
                              geometry.fx * depth,
                          (static_cast<double>(row) - geometry.cy) /
                              geometry.fy * depth,
                          depth);
    } else {
      const Eigen::Vector2d xy(static_cast<double>(col),
                               static_cast<double>(row));
      point_camera =
          UnprojectPixel(view.camera, xy, static_cast<double>(depth));
    }
    const Eigen::Vector3d point_world =
        geometry.world_from_cam_rotation * point_camera +
        geometry.camera_center;

    Eigen::Vector3d normal_world;
    if (precomputed_normal_world == nullptr) {
      normal_world = geometry.world_from_cam_rotation *
                     view_data.normal_map.Get(row, col).cast<double>();
    } else {
      normal_world = precomputed_normal_world->cast<double>();
    }
    if (!normal_world.allFinite() || normal_world.squaredNorm() <= 0.0) {
      return std::nullopt;
    }
    const double normal_squared_norm = normal_world.squaredNorm();
    if (std::abs(normal_squared_norm - 1.0) > 1e-4) {
      normal_world /= std::sqrt(normal_squared_norm);
    }

    FusionPixelData data;
    data.xyz = point_world;
    data.normal = normal_world.cast<float>();
    if (!view_data.image.color.empty() &&
        view_data.image.color.type() == CV_8UC3) {
      const cv::Vec3b color = view_data.image.color.at<cv::Vec3b>(
          static_cast<int>(row), static_cast<int>(col));
      data.color = Eigen::Matrix<std::uint8_t, 3, 1>(
          color[0], color[1], color[2]);
    } else {
      const std::optional<float> gray = SampleGrayBilinear(
          view_data.image, static_cast<float>(col), static_cast<float>(row));
      const float clamped_gray =
          ClampFloat(gray.value_or(1.0F), 0.0F, 1.0F);
      const double color_value =
          static_cast<double>(std::lround(clamped_gray * 255.0F));
      const std::uint8_t packed_color =
          static_cast<std::uint8_t>(color_value);
      data.color = Eigen::Matrix<std::uint8_t, 3, 1>(
          packed_color, packed_color, packed_color);
    }
    return data;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<FusionObservation> ProjectFusionObservationToSource(
    const std::vector<PatchMatchView>& views,
    const std::vector<FusionViewGeometry>& geometries,
    const StereoFusionViewData& source_data,
    const StereoFusionOptions& options,
    const FusionObservation& reference,
    std::size_t source_idx,
    const Eigen::Vector3d& reference_point_world,
    const Eigen::Vector3d& seed_point_world,
    const Eigen::Vector3d& seed_normal_world,
    std::size_t seed_image_idx,
    double min_normal_cosine,
    double min_view_cosine,
    const std::atomic<std::uint8_t>* source_visited,
    StereoFusionErrorSamples* error_samples,
    FusionRejectReason* reject_reason) {
  if (reject_reason != nullptr) {
    *reject_reason = FusionRejectReason::kNone;
  }
  try {
    const FusionViewGeometry& source_geometry = geometries[source_idx];
    const Eigen::Vector3d source_point_camera =
        source_geometry.cam_from_world_rotation * reference_point_world +
        source_geometry.cam_from_world_translation;
    if (source_point_camera.z() <= 0.0) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kProjection;
      }
      return std::nullopt;
    }

    const Eigen::Vector2d source_xy =
        source_geometry.pinhole
            ? Eigen::Vector2d(
                  source_geometry.fx * source_point_camera.x() /
                          source_point_camera.z() +
                      source_geometry.cx,
                  source_geometry.fy * source_point_camera.y() /
                          source_point_camera.z() +
                      source_geometry.cy)
            : ProjectPoint(views[source_idx].camera, source_point_camera);
    const Eigen::Vector3d seed_source_point_camera =
        source_geometry.cam_from_world_rotation * seed_point_world +
        source_geometry.cam_from_world_translation;
    if (seed_source_point_camera.z() <= 0.0) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kProjection;
      }
      return std::nullopt;
    }
    const Eigen::Vector2d seed_source_xy =
        source_geometry.pinhole
            ? Eigen::Vector2d(
                  source_geometry.fx * seed_source_point_camera.x() /
                          seed_source_point_camera.z() +
                      source_geometry.cx,
                  source_geometry.fy * seed_source_point_camera.y() /
                          seed_source_point_camera.z() +
                      source_geometry.cy)
            : ProjectPoint(views[source_idx].camera,
                           seed_source_point_camera);
    const int rounded_source_col =
        static_cast<int>(std::lround(source_xy.x()));
    const int rounded_source_row =
        static_cast<int>(std::lround(source_xy.y()));
    if (rounded_source_row < 0 || rounded_source_col < 0 ||
        rounded_source_row >=
            static_cast<int>(source_data.depth_map.Height()) ||
        rounded_source_col >=
            static_cast<int>(source_data.depth_map.Width())) {
      if (reject_reason != nullptr) {
        *reject_reason = FusionRejectReason::kProjection;
      }
      return std::nullopt;
    }

    std::optional<double> view_cosine;
    // Preserve the original rejection/diagnostic ordering when the view-angle
    // gate is active or samples are requested. With the production default of
    // 180 degrees and diagnostics disabled, defer this relatively expensive
    // normalization until a candidate has passed the cheaper checks.
    if (min_view_cosine > -1.0 + 1e-12 || error_samples != nullptr) {
      view_cosine =
          CosineBetween(
              seed_point_world - geometries[seed_image_idx].camera_center,
              seed_point_world - source_geometry.camera_center);
      if (error_samples != nullptr) {
        AddFusionErrorSample(
            static_cast<float>(DegreesFromCosine(*view_cosine)),
            &error_samples->view_angles_degrees);
      }
      if (*view_cosine < min_view_cosine) {
        if (reject_reason != nullptr) {
          *reject_reason = FusionRejectReason::kView;
        }
        return std::nullopt;
      }
    }

    thread_local std::vector<std::pair<double, std::pair<int, int>>>
        candidates;
    candidates.clear();
    const int search_radius =
        static_cast<int>(options.source_pixel_search_radius);
    const int search_diameter = 2 * search_radius + 1;
    candidates.reserve(static_cast<std::size_t>(search_diameter) *
                       static_cast<std::size_t>(search_diameter));
    const auto candidate_less = [](const auto& lhs, const auto& rhs) {
      if (lhs.first == rhs.first) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    };
    const Eigen::Vector2d rounded_source_xy(
        static_cast<double>(rounded_source_col),
        static_cast<double>(rounded_source_row));
    const double rounded_squared_distance =
        (source_xy - rounded_source_xy).squaredNorm();
    const double fractional_col_distance =
        std::abs(source_xy.x() - rounded_source_xy.x());
    const double fractional_row_distance =
        std::abs(source_xy.y() - rounded_source_xy.y());
    const bool has_nearest_tie =
        std::abs(fractional_col_distance - 0.5) <= 1e-12 ||
        std::abs(fractional_row_distance - 0.5) <= 1e-12;
    candidates.emplace_back(
        rounded_squared_distance,
        std::make_pair(rounded_source_row, rounded_source_col));
    bool remaining_candidates_generated = false;
    const auto generate_remaining_candidates = [&]() {
      if (remaining_candidates_generated) {
        return;
      }
      for (int row_offset = -search_radius; row_offset <= search_radius;
           ++row_offset) {
        for (int col_offset = -search_radius; col_offset <= search_radius;
             ++col_offset) {
          if (row_offset == 0 && col_offset == 0) {
            continue;
          }
          const int candidate_row = rounded_source_row + row_offset;
          const int candidate_col = rounded_source_col + col_offset;
          if (candidate_row < 0 || candidate_col < 0 ||
              candidate_row >=
                  static_cast<int>(source_data.depth_map.Height()) ||
              candidate_col >=
                  static_cast<int>(source_data.depth_map.Width())) {
            continue;
          }
          const Eigen::Vector2d candidate_xy(
              static_cast<double>(candidate_col),
              static_cast<double>(candidate_row));
          candidates.emplace_back(
              (source_xy - candidate_xy).squaredNorm(),
              std::make_pair(candidate_row, candidate_col));
        }
      }
      std::sort(candidates.begin() + (has_nearest_tie ? 0 : 1),
                candidates.end(), candidate_less);
      remaining_candidates_generated = true;
    };
    if (has_nearest_tie) {
      generate_remaining_candidates();
    }
    bool projected_source_depth_sampled = false;
    std::optional<float> projected_source_depth;

    FusionRejectReason candidate_reject_reason = FusionRejectReason::kNone;
    for (std::size_t candidate_idx = 0;; ++candidate_idx) {
      if (candidate_idx >= candidates.size()) {
        if (remaining_candidates_generated) {
          break;
        }
        generate_remaining_candidates();
        if (candidate_idx >= candidates.size()) {
          break;
        }
      }
      const auto& candidate = candidates[candidate_idx];
      const Eigen::Vector2d candidate_xy(
          static_cast<double>(candidate.second.second),
          static_cast<double>(candidate.second.first));
      const double seed_squared_reprojection_error =
          (seed_source_xy - candidate_xy).squaredNorm();
      const double max_squared_reprojection_error =
          static_cast<double>(options.max_reprojection_error_pixels) *
          static_cast<double>(options.max_reprojection_error_pixels);
      if (error_samples != nullptr) {
        AddFusionErrorSample(
            static_cast<float>(
                std::sqrt(seed_squared_reprojection_error)),
            &error_samples->reprojection_errors_pixels);
      }
      // Match COLMAP's fusion invariant: source candidates may be discovered
      // by walking from the current cursor, but every accepted observation
      // must still reproject close to the original seed point. Comparing only
      // with the cursor permits small errors to accumulate over graph hops and
      // merges a planar neighborhood into one oversized, bumpy track.
      if (seed_squared_reprojection_error >
          max_squared_reprojection_error) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kProjection,
                                     &candidate_reject_reason);
        continue;
      }

      const std::size_t row =
          static_cast<std::size_t>(candidate.second.first);
      const std::size_t col =
          static_cast<std::size_t>(candidate.second.second);
      if (source_visited != nullptr &&
          source_visited[
              row * source_data.depth_map.Width() + col].load(
                  std::memory_order_relaxed) != 0) {
        if (reject_reason != nullptr) {
          *reject_reason = FusionRejectReason::kVisited;
        }
        return std::nullopt;
      }
      if (!source_data.depth_map.IsValid(row, col) ||
          !source_data.normal_map.IsValid(row, col) ||
          (options.require_cost_map &&
           !source_data.cost_map.IsValid(row, col))) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kInvalid,
                                     &candidate_reject_reason);
        continue;
      }

      // 与几何一致性图保持一致：优先在真实投影位置做双线性深度采样。
      // 若投影落在边界或邻域深度不完整，则回退到候选离散像素深度。
      double source_depth =
          static_cast<double>(source_data.depth_map.Get(row, col));
      if (!projected_source_depth_sampled) {
        projected_source_depth = SampleDepthBilinear(
            source_data.depth_map, static_cast<float>(seed_source_xy.x()),
            static_cast<float>(seed_source_xy.y()));
        projected_source_depth_sampled = true;
      }
      if (projected_source_depth.has_value()) {
        source_depth = static_cast<double>(*projected_source_depth);
      }
      const double relative_depth_error =
          std::abs(seed_source_point_camera.z() - source_depth) /
          source_depth;
      AddFusionErrorSample(static_cast<float>(relative_depth_error),
                           error_samples == nullptr
                               ? nullptr
                               : &error_samples->relative_depth_errors);
      if (relative_depth_error > options.max_relative_depth_error) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kDepth,
                                     &candidate_reject_reason);
        continue;
      }

      if (options.require_mutual_consistency &&
          !source_data.consistency_graph.Empty()) {
        const bool mutually_consistent = ContainsImageId(
            source_data.consistency_graph.SourceImageIds(row, col),
            views[reference.image_idx].image_id);
        if (!mutually_consistent) {
          KeepMoreSpecificRejectReason(
              FusionRejectReason::kMutualConsistency,
              &candidate_reject_reason);
          continue;
        }
      }

      Eigen::Vector3d source_normal_world =
          source_geometry.world_from_cam_rotation *
          source_data.normal_map.Get(row, col).cast<double>();
      const double source_normal_squared_norm =
          source_normal_world.squaredNorm();
      if (!source_normal_world.allFinite() ||
          source_normal_squared_norm <= 0.0) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kInvalid,
                                     &candidate_reject_reason);
        continue;
      }
      if (std::abs(source_normal_squared_norm - 1.0) > 1e-4) {
        source_normal_world /= std::sqrt(source_normal_squared_norm);
      }

      // Match COLMAP's track semantics: every observation is checked against
      // the seed normal. Comparing only with the current BFS cursor permits
      // gradual normal drift and can merge very large curved tracks.
      const double normal_cosine = std::max(
          -1.0, std::min(1.0, seed_normal_world.dot(source_normal_world)));
      if (error_samples != nullptr) {
        AddFusionErrorSample(
            static_cast<float>(DegreesFromCosine(normal_cosine)),
            &error_samples->normal_angles_degrees);
      }
      if (normal_cosine < min_normal_cosine) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kNormal,
                                     &candidate_reject_reason);
        continue;
      }

      if (!view_cosine.has_value()) {
        // The default 180-degree gate cannot reject an observation. Avoid
        // normalizing two viewing rays for every accepted observation when
        // diagnostics are disabled.
        view_cosine = 1.0;
      }
      if (*view_cosine < min_view_cosine) {
        KeepMoreSpecificRejectReason(FusionRejectReason::kView,
                                     &candidate_reject_reason);
        continue;
      }

      FusionObservation source;
      source.image_idx = source_idx;
      source.row = row;
      source.col = col;
      source.traversal_depth = reference.traversal_depth + 1;
      source.normal_cosine = normal_cosine;
      source.view_cosine = *view_cosine;
      source.data.normal = source_normal_world.cast<float>();
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
  return width_ * height_;
}

bool ConsistencyGraph::Empty() const {
  return width_ == 0 || height_ == 0;
}

void ConsistencyGraph::Add(std::size_t row,
                           std::size_t col,
                           ImageId source_image_id) {
  if (IsCompact()) {
    throw std::logic_error("Cannot append to a compact consistency graph.");
  }
  data_.at(Offset(row, col)).push_back(source_image_id);
}

ConsistencyGraph::ImageIdRange ConsistencyGraph::SourceImageIds(
    std::size_t row,
    std::size_t col) const {
  const std::size_t offset = Offset(row, col);
  if (IsCompact()) {
    const std::size_t begin = compact_offsets_[offset];
    const std::size_t end = compact_offsets_[offset + 1];
    if (begin == end && compact_image_ids_.empty()) {
      return ImageIdRange();
    }
    return ImageIdRange(compact_image_ids_.data() + begin, end - begin);
  }
  const std::vector<ImageId>& source_ids = data_.at(offset);
  return ImageIdRange(source_ids.data(), source_ids.size());
}

void ConsistencyGraph::Compact() {
  if (IsCompact()) {
    return;
  }
  compact_offsets_.resize(NumPixels() + 1, 0);
  std::size_t num_image_ids = 0;
  for (std::size_t offset = 0; offset < data_.size(); ++offset) {
    if (num_image_ids > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Consistency graph exceeds CSR32 capacity.");
    }
    compact_offsets_[offset] = static_cast<std::uint32_t>(num_image_ids);
    num_image_ids += data_[offset].size();
  }
  if (num_image_ids > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("Consistency graph exceeds CSR32 capacity.");
  }
  compact_offsets_[NumPixels()] =
      static_cast<std::uint32_t>(num_image_ids);
  compact_image_ids_.reserve(num_image_ids);
  for (const std::vector<ImageId>& source_ids : data_) {
    compact_image_ids_.insert(compact_image_ids_.end(), source_ids.begin(),
                              source_ids.end());
  }
  std::vector<std::vector<ImageId>>().swap(data_);
}

bool ConsistencyGraph::IsCompact() const {
  return !compact_offsets_.empty();
}

std::size_t ConsistencyGraph::Count(std::size_t row, std::size_t col) const {
  if (IsCompact()) {
    const std::size_t offset = Offset(row, col);
    return static_cast<std::size_t>(compact_offsets_[offset + 1] -
                                    compact_offsets_[offset]);
  }
  return SourceImageIds(row, col).size();
}

std::size_t ConsistencyGraph::NumConsistentPixels(
    std::size_t min_num_sources) const {
  std::size_t count = 0;
  for (std::size_t row = 0; row < height_; ++row) {
    for (std::size_t col = 0; col < width_; ++col) {
      count += Count(row, col) >= min_num_sources ? 1 : 0;
    }
  }
  return count;
}

std::vector<int> ConsistencyGraph::ToColmapData(
    const std::vector<ImageId>& image_ids_by_index) const {
  std::unordered_map<ImageId, int> image_index_by_id;
  image_index_by_id.reserve(image_ids_by_index.size());
  for (std::size_t image_idx = 0; image_idx < image_ids_by_index.size();
       ++image_idx) {
    image_index_by_id[image_ids_by_index[image_idx]] =
        static_cast<int>(image_idx);
  }

  std::vector<int> data;
  for (std::size_t row = 0; row < height_; ++row) {
    for (std::size_t col = 0; col < width_; ++col) {
      const ImageIdRange source_ids = SourceImageIds(row, col);
      if (source_ids.empty()) {
        continue;
      }
      data.push_back(static_cast<int>(col));
      data.push_back(static_cast<int>(row));
      data.push_back(static_cast<int>(source_ids.size()));
      for (const ImageId source_id : source_ids) {
        const auto image_it = image_index_by_id.find(source_id);
        if (image_it == image_index_by_id.end()) {
          throw std::invalid_argument(
              "COLMAP consistency graph source image id is not mapped.");
        }
        data.push_back(image_it->second);
      }
    }
  }
  return data;
}

ConsistencyGraph ConsistencyGraph::FromColmapData(
    std::size_t width,
    std::size_t height,
    const std::vector<int>& data,
    const std::vector<ImageId>& image_ids_by_index) {
  ConsistencyGraph graph(width, height);
  for (std::size_t offset = 0; offset < data.size();) {
    if (offset + 3 > data.size()) {
      throw std::runtime_error("Invalid COLMAP consistency graph data.");
    }
    const int col = data[offset++];
    const int row = data[offset++];
    const int num_images = data[offset++];
    if (row < 0 || col < 0 || num_images < 0 ||
        static_cast<std::size_t>(row) >= height ||
        static_cast<std::size_t>(col) >= width ||
        offset + static_cast<std::size_t>(num_images) > data.size()) {
      throw std::runtime_error("Invalid COLMAP consistency graph entry.");
    }
    for (int i = 0; i < num_images; ++i) {
      const int image_idx = data[offset++];
      if (image_idx < 0 ||
          static_cast<std::size_t>(image_idx) >= image_ids_by_index.size()) {
        throw std::runtime_error(
            "Invalid COLMAP consistency graph image index.");
      }
      graph.Add(static_cast<std::size_t>(row), static_cast<std::size_t>(col),
                image_ids_by_index[static_cast<std::size_t>(image_idx)]);
    }
  }
  graph.Compact();
  return graph;
}

ConsistencyGraph ConsistencyGraph::FromCompactData(
    std::size_t width,
    std::size_t height,
    std::vector<std::size_t> offsets,
    std::vector<ImageId> image_ids) {
  if (offsets.size() != width * height + 1 || offsets.empty() ||
      offsets.front() != 0 || offsets.back() != image_ids.size() ||
      !std::is_sorted(offsets.begin(), offsets.end()) ||
      offsets.back() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Invalid compact consistency graph data.");
  }
  std::vector<std::uint32_t> compact_offsets(offsets.size());
  std::transform(offsets.begin(), offsets.end(), compact_offsets.begin(),
                 [](std::size_t offset) {
                   return static_cast<std::uint32_t>(offset);
                 });
  return FromCompactData(width, height, std::move(compact_offsets),
                         std::move(image_ids));
}

ConsistencyGraph ConsistencyGraph::FromCompactData(
    std::size_t width,
    std::size_t height,
    std::vector<std::uint32_t> offsets,
    std::vector<ImageId> image_ids) {
  const std::size_t num_pixels = width * height;
  if (offsets.size() != num_pixels + 1 || offsets.front() != 0 ||
      offsets.back() != image_ids.size() ||
      !std::is_sorted(offsets.begin(), offsets.end())) {
    throw std::invalid_argument("Invalid compact consistency graph data.");
  }
  ConsistencyGraph graph;
  graph.width_ = width;
  graph.height_ = height;
  graph.compact_offsets_ = std::move(offsets);
  graph.compact_image_ids_ = std::move(image_ids);
  return graph;
}

const std::vector<std::uint32_t>& ConsistencyGraph::CompactOffsets() const {
  if (!IsCompact()) {
    throw std::logic_error("Consistency graph is not compact.");
  }
  return compact_offsets_;
}

const std::vector<ImageId>& ConsistencyGraph::CompactImageIds() const {
  if (!IsCompact()) {
    throw std::logic_error("Consistency graph is not compact.");
  }
  return compact_image_ids_;
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
  std::vector<std::uint32_t> generated_offsets;
  std::vector<ImageId> generated_ids;
  const std::vector<std::uint32_t>* graph_offsets_ptr = nullptr;
  const std::vector<ImageId>* graph_ids_ptr = nullptr;
  if (consistency_graph.IsCompact()) {
    graph_offsets_ptr = &consistency_graph.CompactOffsets();
    graph_ids_ptr = &consistency_graph.CompactImageIds();
  } else {
    generated_offsets.resize(consistency_graph.NumPixels() + 1, 0);
    for (std::size_t row = 0; row < consistency_graph.Height(); ++row) {
      for (std::size_t col = 0; col < consistency_graph.Width(); ++col) {
        const std::size_t pixel_offset =
            row * consistency_graph.Width() + col;
        if (generated_ids.size() >
            std::numeric_limits<std::uint32_t>::max()) {
          throw std::overflow_error(
              "Consistency graph exceeds CSR32 capacity.");
        }
        generated_offsets[pixel_offset] =
            static_cast<std::uint32_t>(generated_ids.size());
        const ConsistencyGraph::ImageIdRange ids =
            consistency_graph.SourceImageIds(row, col);
        generated_ids.insert(generated_ids.end(), ids.begin(), ids.end());
      }
    }
    if (generated_ids.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("Consistency graph exceeds CSR32 capacity.");
    }
    generated_offsets.back() =
        static_cast<std::uint32_t>(generated_ids.size());
    graph_offsets_ptr = &generated_offsets;
    graph_ids_ptr = &generated_ids;
  }
  const std::vector<std::uint32_t>& graph_offsets = *graph_offsets_ptr;
  const std::vector<ImageId>& graph_ids = *graph_ids_ptr;
  if (graph_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("Consistency graph is too large for CSR32.");
  }
  const std::uint64_t num_source_ids =
      static_cast<std::uint64_t>(graph_ids.size());
  file.write(reinterpret_cast<const char*>(&num_source_ids),
             static_cast<std::streamsize>(sizeof(num_source_ids)));
  if (!graph_offsets.empty()) {
    file.write(reinterpret_cast<const char*>(graph_offsets.data()),
               static_cast<std::streamsize>(graph_offsets.size() *
                                            sizeof(std::uint32_t)));
  }
  static_assert(sizeof(ImageId) == sizeof(std::uint32_t));
  if (!graph_ids.empty()) {
    file.write(reinterpret_cast<const char*>(graph_ids.data()),
               static_cast<std::streamsize>(graph_ids.size() *
                                            sizeof(ImageId)));
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
  RequireConsistencyGraphStream(
      magic == kConsistencyGraphBinaryMagic &&
          (version == 1 || version == kConsistencyGraphBinaryVersion),
      path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireConsistencyGraphStream(static_cast<bool>(file), path);
  file.get();

  const std::size_t num_pixels = width * height;
  if (version == kConsistencyGraphBinaryVersion) {
    std::uint64_t num_source_ids_u64 = 0;
    file.read(reinterpret_cast<char*>(&num_source_ids_u64),
              static_cast<std::streamsize>(sizeof(num_source_ids_u64)));
    RequireConsistencyGraphStream(
        static_cast<bool>(file) &&
            num_source_ids_u64 <=
                std::numeric_limits<std::uint32_t>::max(),
        path);
    std::vector<std::uint32_t> compact_offsets(num_pixels + 1);
    if (!compact_offsets.empty()) {
      file.read(reinterpret_cast<char*>(compact_offsets.data()),
                static_cast<std::streamsize>(compact_offsets.size() *
                                             sizeof(std::uint32_t)));
      RequireConsistencyGraphStream(static_cast<bool>(file), path);
    }
    std::vector<ImageId> source_ids(
        static_cast<std::size_t>(num_source_ids_u64));
    if (!source_ids.empty()) {
      file.read(reinterpret_cast<char*>(source_ids.data()),
                static_cast<std::streamsize>(source_ids.size() *
                                             sizeof(ImageId)));
      RequireConsistencyGraphStream(static_cast<bool>(file), path);
    }
    RequireConsistencyGraphStream(
        compact_offsets.front() == 0 &&
            compact_offsets.back() == source_ids.size() &&
            std::is_sorted(compact_offsets.begin(), compact_offsets.end()),
        path);
    return ConsistencyGraph::FromCompactData(
        width, height, std::move(compact_offsets), std::move(source_ids));
  }

  const std::streampos data_start = file.tellg();
  file.seekg(0, std::ios::end);
  const std::streamoff data_bytes = file.tellg() - data_start;
  RequireConsistencyGraphStream(
      data_bytes >=
          static_cast<std::streamoff>(num_pixels * sizeof(std::uint32_t)) &&
          data_bytes % static_cast<std::streamoff>(sizeof(std::uint32_t)) == 0,
      path);
  const std::size_t num_source_ids =
      static_cast<std::size_t>(data_bytes) / sizeof(std::uint32_t) -
      num_pixels;
  file.seekg(data_start);
  std::vector<std::uint32_t> words(
      static_cast<std::size_t>(data_bytes) / sizeof(std::uint32_t));
  if (!words.empty()) {
    file.read(reinterpret_cast<char*>(words.data()),
              static_cast<std::streamsize>(data_bytes));
    RequireConsistencyGraphStream(static_cast<bool>(file), path);
  }

  std::vector<std::size_t> offsets(num_pixels + 1, 0);
  std::vector<ImageId> source_ids;
  source_ids.reserve(num_source_ids);
  std::size_t pixel_offset = 0;
  std::size_t word_offset = 0;
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      offsets[pixel_offset++] = source_ids.size();
      RequireConsistencyGraphStream(word_offset < words.size(), path);
      const std::size_t count = words[word_offset++];
      RequireConsistencyGraphStream(
          count <= words.size() - word_offset, path);
      for (std::size_t i = 0; i < count; ++i) {
        source_ids.push_back(
            static_cast<ImageId>(words[word_offset + i]));
      }
      word_offset += count;
    }
  }
  offsets[num_pixels] = source_ids.size();
  RequireConsistencyGraphStream(source_ids.size() == num_source_ids &&
                                    word_offset == words.size(),
                                path);
  return ConsistencyGraph::FromCompactData(
      width, height, std::move(offsets), std::move(source_ids));
}

void WriteConsistencyGraphColmapBinary(
    const std::filesystem::path& path,
    const ConsistencyGraph& consistency_graph,
    const std::vector<ImageId>& image_ids_by_index) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write COLMAP consistency graph: " +
                             path.string());
  }

  file << consistency_graph.Width() << "&" << consistency_graph.Height()
       << "&1&";
  const std::vector<int> colmap_data =
      consistency_graph.ToColmapData(image_ids_by_index);
  for (const int value : colmap_data) {
    file.write(reinterpret_cast<const char*>(&value),
               static_cast<std::streamsize>(sizeof(value)));
  }
  RequireConsistencyGraphStream(static_cast<bool>(file), path);
}

ConsistencyGraph ReadConsistencyGraphColmapBinary(
    const std::filesystem::path& path,
    const std::vector<ImageId>& image_ids_by_index) {
  std::ifstream text_file(path, std::ios::binary);
  if (!text_file) {
    throw std::runtime_error("Failed to read COLMAP consistency graph: " +
                             path.string());
  }
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t depth = 0;
  char delimiter = '\0';
  text_file >> width >> delimiter;
  RequireConsistencyGraphStream(delimiter == '&', path);
  text_file >> height >> delimiter;
  RequireConsistencyGraphStream(delimiter == '&', path);
  text_file >> depth >> delimiter;
  RequireConsistencyGraphStream(delimiter == '&' && width > 0 && height > 0 &&
                                    depth == 1,
                                path);
  const std::streampos data_start = text_file.tellg();
  text_file.seekg(0, std::ios::end);
  const std::streamoff num_bytes = text_file.tellg() - data_start;
  RequireConsistencyGraphStream(num_bytes >= 0 &&
                                    num_bytes %
                                            static_cast<std::streamoff>(
                                                sizeof(int)) ==
                                        0,
                                path);
  std::vector<int> data(static_cast<std::size_t>(num_bytes) / sizeof(int));
  text_file.seekg(data_start);
  if (!data.empty()) {
    text_file.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(num_bytes));
    RequireConsistencyGraphStream(static_cast<bool>(text_file), path);
  }
  return ConsistencyGraph::FromColmapData(width, height, data,
                                          image_ids_by_index);
}

void WritePointsVisibilityColmapBinary(
    const std::filesystem::path& path,
    const std::vector<std::vector<std::uint32_t>>& points_visibility) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write COLMAP points visibility: " +
                             path.string());
  }
  const std::uint64_t num_points =
      static_cast<std::uint64_t>(points_visibility.size());
  file.write(reinterpret_cast<const char*>(&num_points),
             static_cast<std::streamsize>(sizeof(num_points)));
  for (const std::vector<std::uint32_t>& visibility : points_visibility) {
    const std::uint32_t num_visible_images =
        static_cast<std::uint32_t>(visibility.size());
    file.write(reinterpret_cast<const char*>(&num_visible_images),
               static_cast<std::streamsize>(sizeof(num_visible_images)));
    for (const std::uint32_t image_idx : visibility) {
      file.write(reinterpret_cast<const char*>(&image_idx),
                 static_cast<std::streamsize>(sizeof(image_idx)));
    }
  }
  if (!file) {
    throw std::runtime_error("Failed to finish COLMAP points visibility: " +
                             path.string());
  }
}

void WritePointsVisibilityColmapBinary(
    const std::filesystem::path& path,
    const std::vector<std::uint64_t>& points_visibility_masks) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write COLMAP points visibility: " +
                             path.string());
  }
  const std::uint64_t num_points =
      static_cast<std::uint64_t>(points_visibility_masks.size());
  file.write(reinterpret_cast<const char*>(&num_points),
             static_cast<std::streamsize>(sizeof(num_points)));
  for (std::uint64_t visibility_mask : points_visibility_masks) {
    std::uint32_t num_visible_images = 0;
    for (std::uint64_t remaining = visibility_mask; remaining != 0;
         remaining &= remaining - 1) {
      ++num_visible_images;
    }
    file.write(reinterpret_cast<const char*>(&num_visible_images),
               static_cast<std::streamsize>(sizeof(num_visible_images)));
    for (std::uint32_t image_idx = 0; visibility_mask != 0; ++image_idx) {
      if ((visibility_mask & 1U) != 0) {
        file.write(reinterpret_cast<const char*>(&image_idx),
                   static_cast<std::streamsize>(sizeof(image_idx)));
      }
      visibility_mask >>= 1;
    }
  }
  if (!file) {
    throw std::runtime_error("Failed to finish COLMAP points visibility: " +
                             path.string());
  }
}

std::vector<std::vector<std::uint32_t>> ReadPointsVisibilityColmapBinary(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read COLMAP points visibility: " +
                             path.string());
  }
  std::uint64_t num_points = 0;
  file.read(reinterpret_cast<char*>(&num_points),
            static_cast<std::streamsize>(sizeof(num_points)));
  if (!file) {
    throw std::runtime_error("Invalid COLMAP points visibility: " +
                             path.string());
  }
  std::vector<std::vector<std::uint32_t>> points_visibility(
      static_cast<std::size_t>(num_points));
  for (std::uint64_t point_idx = 0; point_idx < num_points; ++point_idx) {
    std::uint32_t num_visible_images = 0;
    file.read(reinterpret_cast<char*>(&num_visible_images),
              static_cast<std::streamsize>(sizeof(num_visible_images)));
    if (!file) {
      throw std::runtime_error("Invalid COLMAP points visibility: " +
                               path.string());
    }
    std::vector<std::uint32_t>& visibility =
        points_visibility[static_cast<std::size_t>(point_idx)];
    visibility.resize(num_visible_images);
    for (std::uint32_t i = 0; i < num_visible_images; ++i) {
      file.read(reinterpret_cast<char*>(&visibility[i]),
                static_cast<std::streamsize>(sizeof(visibility[i])));
      if (!file) {
        throw std::runtime_error("Invalid COLMAP points visibility: " +
                                 path.string());
      }
    }
  }
  return points_visibility;
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

float ComputePatchMatchNccLikelihood(float cost, float ncc_sigma) {
  if (!std::isfinite(cost) || !std::isfinite(ncc_sigma) ||
      ncc_sigma <= 0.0F) {
    throw std::invalid_argument("Invalid PatchMatch NCC likelihood input.");
  }
  constexpr float kPi = 3.14159265358979323846F;
  const float normalization =
      2.0F /
      (std::sqrt(2.0F * kPi) * ncc_sigma *
       std::erf(2.0F / (ncc_sigma * std::sqrt(2.0F))));
  return std::exp(-0.5F * cost * cost / (ncc_sigma * ncc_sigma)) *
         normalization;
}

PatchMatchCost ComputePatchZncc(
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    const Eigen::Vector2f& source_xy,
    const PatchMatchCostOptions& options) {
  if (options.window_radius < 0 || options.window_step <= 0 ||
      !std::isfinite(options.sigma_spatial) ||
      !std::isfinite(options.sigma_color) || options.sigma_color <= 0.0F) {
    throw std::invalid_argument("Invalid PatchMatch cost window options.");
  }

  std::vector<float> reference_values;
  std::vector<float> source_values;
  std::vector<Eigen::Vector2i> sample_offsets;
  const auto reference_center = SampleGrayBilinear(
      reference_image, reference_xy.x(), reference_xy.y());
  if (!reference_center.has_value()) {
    return InvalidCost(options);
  }
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
      sample_offsets.emplace_back(dx, dy);
    }
  }

  return ComputeZnccFromSamples(reference_values, source_values,
                                sample_offsets, *reference_center, options);
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

std::optional<float> ComputeSourceDepthSignedRelativeError(
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

    const Eigen::Vector3d sampled_source_point =
        UnprojectPixel(source_view.camera, source_xy.cast<double>(),
                       static_cast<double>(*source_depth));
    const Eigen::Vector3d sampled_world =
        source_view.cam_from_world.Inverse().TransformPoint(
            sampled_source_point);
    const Eigen::Vector3d returned_reference =
        reference_view.cam_from_world.TransformPoint(sampled_world);
    if (returned_reference.z() <= 0.0) {
      return std::nullopt;
    }
    return static_cast<float>(
        (returned_reference.z() - static_cast<double>(depth)) /
        static_cast<double>(depth));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<float> ComputeSourceDepthReprojectionError(
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
    if (point_source.z() <= 0.0) return std::nullopt;
    const Eigen::Vector2f source_xy =
        ProjectPoint(source_view.camera, point_source).cast<float>();
    const std::optional<float> source_depth =
        SampleDepthBilinear(source_depth_map, source_xy.x(), source_xy.y());
    if (!source_depth.has_value()) return std::nullopt;
    const Eigen::Vector3d sampled_source_point =
        UnprojectPixel(source_view.camera, source_xy.cast<double>(),
                       static_cast<double>(*source_depth));
    const Eigen::Vector3d returned_reference =
        reference_view.cam_from_world.TransformPoint(
            source_view.cam_from_world.Inverse().TransformPoint(
                sampled_source_point));
    if (returned_reference.z() <= 0.0) return std::nullopt;
    return static_cast<float>(
        (ProjectPoint(reference_view.camera, returned_reference) -
         reference_xy.cast<double>()).norm());
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool UseGeometricCandidateFilter(
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>* source_depth_maps) {
  return source_depth_maps != nullptr && !source_depth_maps->empty() &&
         options.geometric_consistency_weight > 0.0F;
}

struct GeometricCandidateStats {
  std::size_t num_consistent_sources = 0;
  std::size_t num_occluded_sources = 0;
  std::size_t num_conflicting_sources = 0;
};

GeometricCandidateStats ComputeGeometricCandidateStats(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>& source_depth_maps) {
  if (source_views.size() != source_depth_maps.size()) {
    throw std::invalid_argument(
        "Source depth maps must match source views for geometric filter.");
  }
  if (!std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "Invalid PatchMatch geometric filter options.");
  }

  GeometricCandidateStats stats;
  for (std::size_t i = 0; i < source_views.size(); ++i) {
    const std::optional<float> signed_relative_depth_error =
        ComputeSourceDepthSignedRelativeError(reference_view, source_views[i],
                                             reference_xy, depth,
                                             source_depth_maps[i]);
    if (!signed_relative_depth_error.has_value()) {
      continue;
    }
    if (std::abs(*signed_relative_depth_error) <=
        options.geometric_max_relative_depth_error) {
      stats.num_consistent_sources++;
    } else if (*signed_relative_depth_error < 0.0F) {
      stats.num_occluded_sources++;
    } else {
      stats.num_conflicting_sources++;
    }
  }
  return stats;
}

bool PassesGeometricCandidateFilter(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const PatchMatchCostOptions& options,
    const std::vector<DenseDepthMap>& source_depth_maps) {
  const GeometricCandidateStats stats = ComputeGeometricCandidateStats(
      reference_view, source_views, reference_xy, depth, options,
      source_depth_maps);
  if (stats.num_consistent_sources < options.min_num_valid_sources) {
    return false;
  }
  if (options.geometric_reject_occluded_candidates &&
      stats.num_occluded_sources > 0) {
    return false;
  }
  if (options.geometric_reject_depth_conflicts &&
      stats.num_conflicting_sources > 0 &&
      stats.num_conflicting_sources >= stats.num_consistent_sources) {
    return false;
  }
  return true;
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
  if (options.window_radius < 0 || options.window_step <= 0 ||
      !std::isfinite(options.sigma_spatial) ||
      !std::isfinite(options.sigma_color) || options.sigma_color <= 0.0F) {
    throw std::invalid_argument("Invalid PatchMatch cost window options.");
  }
  if (!std::isfinite(options.geometric_consistency_weight) ||
      options.geometric_consistency_weight < 0.0F ||
      !std::isfinite(options.geometric_max_reprojection_error) ||
      options.geometric_max_reprojection_error <= 0.0F ||
      !std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "Invalid PatchMatch geometric consistency cost options.");
  }

  std::vector<float> reference_values;
  std::vector<float> source_values;
  std::vector<Eigen::Vector2i> sample_offsets;
  const auto reference_center = SampleGrayBilinear(
      reference_image, reference_xy.x(), reference_xy.y());
  if (!reference_center.has_value()) {
    return InvalidCost(options);
  }
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
      sample_offsets.emplace_back(dx, dy);
    }
  }

  PatchMatchCost cost =
      ComputeZnccFromSamples(reference_values, source_values, sample_offsets,
                             *reference_center, options);
  if (!cost.valid || source_depth_map == nullptr ||
      options.geometric_consistency_weight <= 0.0F) {
    return cost;
  }

  // 将当前候选深度投影到 source 图，并用上一轮 source depth
  // 给几何不一致的候选增加软惩罚。缺少 source depth 时不强行判坏。
  const std::optional<float> reprojection_error =
      ComputeSourceDepthReprojectionError(reference_view, source_view,
                                          reference_xy, depth,
                                          *source_depth_map);
  const float geometric_cost =
      reprojection_error.has_value()
          ? std::min(options.geometric_max_reprojection_error,
                     *reprojection_error)
          : options.geometric_max_reprojection_error;
  // Match COLMAP's public regularizer semantics: our photometric term is
  // half of COLMAP's (1 - ZNCC), so scale the geometric term equally.
  // Invalid source depth receives the truncated maximum penalty.
  cost.cost +=
      0.5F * options.geometric_consistency_weight * geometric_cost;
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
  std::vector<float> geometric_errors;
  geometric_errors.reserve(source_views.size());
  PatchMatchCostOptions source_cost_options = options;
  if (options.robust_geometric_aggregation) {
    source_cost_options.geometric_consistency_weight = 0.0F;
  }
  for (std::size_t i = 0; i < source_views.size(); ++i) {
    const DenseDepthMap* source_depth_map =
        source_depth_maps == nullptr ? nullptr : &source_depth_maps->at(i);
    const PatchMatchCost cost =
        ComputePlaneHypothesisCost(reference_view, source_views[i],
                                   reference_image, source_images[i],
                                   reference_xy, depth, reference_normal,
                                   source_cost_options, source_depth_map);
    if (cost.valid) {
      valid_costs.push_back(cost.cost);
    }
    if (options.robust_geometric_aggregation && source_depth_map != nullptr &&
        options.geometric_consistency_weight > 0.0F) {
      const std::optional<float> reprojection_error =
          ComputeSourceDepthReprojectionError(
              reference_view, source_views[i], reference_xy, depth,
              *source_depth_map);
      if (reprojection_error.has_value()) {
        geometric_errors.push_back(std::min(
            options.geometric_max_reprojection_error, *reprojection_error));
      }
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
  if (options.robust_geometric_aggregation && source_depth_maps != nullptr &&
      options.geometric_consistency_weight > 0.0F) {
    float geometric_cost = options.geometric_max_reprojection_error;
    if (!geometric_errors.empty()) {
      std::sort(geometric_errors.begin(), geometric_errors.end());
      const std::size_t num_geometric_used =
          std::max<std::size_t>(1, (geometric_errors.size() + 1) / 2);
      geometric_cost = std::accumulate(
                           geometric_errors.begin(),
                           geometric_errors.begin() + num_geometric_used,
                           0.0F) /
                       static_cast<float>(num_geometric_used);
    }
    result.cost +=
        0.5F * options.geometric_consistency_weight * geometric_cost;
  }
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
  const bool use_geometric_filter =
      UseGeometricCandidateFilter(options, source_depth_maps);

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
    if (use_geometric_filter) {
      if (!PassesGeometricCandidateFilter(reference_view, source_views,
                                          reference_xy, *candidate_depth,
                                          options, *source_depth_maps)) {
        summary.num_geometric_candidate_rejections++;
        return;
      }
      summary.num_geometric_candidate_acceptances++;
    }
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
  const bool use_geometric_filter =
      UseGeometricCandidateFilter(options, source_depth_maps);

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
    if (use_geometric_filter) {
      if (!PassesGeometricCandidateFilter(reference_view, source_views,
                                          reference_xy, *candidate_depth,
                                          options, *source_depth_maps)) {
        summary.num_geometric_candidate_rejections++;
        return;
      }
      summary.num_geometric_candidate_acceptances++;
    }
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
      UseGeometricCandidateFilter(cost_options, source_depth_maps);

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
          if (!PassesGeometricCandidateFilter(
                  reference_view, source_views, reference_xy, candidate_depth,
                  cost_options, *source_depth_maps)) {
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

std::size_t OrientNormalMapTowardsCamera(
    const PatchMatchView& reference_view,
    NormalMap* normal_map) {
  if (normal_map == nullptr) {
    throw std::invalid_argument(
        "Normal orientation requires a non-null normal map.");
  }
  if (reference_view.camera.width == 0 ||
      reference_view.camera.height == 0) {
    throw std::invalid_argument(
        "Normal orientation requires non-zero camera dimensions.");
  }
  Camera camera = reference_view.camera;
  if (normal_map->Width() != camera.width ||
      normal_map->Height() != camera.height) {
    camera = ScaleCameraForPyramidLevel(
        camera,
        static_cast<double>(normal_map->Width()) /
            static_cast<double>(camera.width),
        static_cast<double>(normal_map->Height()) /
            static_cast<double>(camera.height),
        normal_map->Width(), normal_map->Height());
  }

  const bool pinhole =
      (camera.model_id == CameraModelId::kPinhole &&
       camera.params.size() >= 4) ||
      (camera.model_id == CameraModelId::kSimplePinhole &&
       camera.params.size() >= 3);
  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  if (pinhole) {
    fx = camera.params[0];
    if (camera.model_id == CameraModelId::kPinhole) {
      fy = camera.params[1];
      cx = camera.params[2];
      cy = camera.params[3];
    } else {
      fy = fx;
      cx = camera.params[1];
      cy = camera.params[2];
    }
  }

  std::size_t num_flipped = 0;
  for (std::size_t row = 0; row < normal_map->Height(); ++row) {
    for (std::size_t col = 0; col < normal_map->Width(); ++col) {
      if (!normal_map->IsValid(row, col)) {
        continue;
      }
      const Eigen::Vector3d view_ray =
          pinhole
              ? Eigen::Vector3d(
                    (static_cast<double>(col) - cx) / fx,
                    (static_cast<double>(row) - cy) / fy, 1.0)
              : UnprojectPixel(
                    camera,
                    Eigen::Vector2d(static_cast<double>(col),
                                    static_cast<double>(row)),
                    1.0);
      const Eigen::Vector3f normal = normal_map->Get(row, col);
      if (view_ray.allFinite() &&
          normal.cast<double>().dot(view_ray) > 0.0) {
        normal_map->Set(row, col, -normal);
        ++num_flipped;
      }
    }
  }
  return num_flipped;
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
        propagation_summary.num_geometric_candidate_rejections +=
            forward.num_geometric_candidate_rejections +
            backward.num_geometric_candidate_rejections;
        propagation_summary.num_geometric_candidate_acceptances +=
            forward.num_geometric_candidate_acceptances +
            backward.num_geometric_candidate_acceptances;
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
        propagation_summary.num_geometric_candidate_rejections +=
            forward.num_geometric_candidate_rejections +
            backward.num_geometric_candidate_rejections +
            top_down.num_geometric_candidate_rejections +
            bottom_up.num_geometric_candidate_rejections;
        propagation_summary.num_geometric_candidate_acceptances +=
            forward.num_geometric_candidate_acceptances +
            backward.num_geometric_candidate_acceptances +
            top_down.num_geometric_candidate_acceptances +
            bottom_up.num_geometric_candidate_acceptances;
        result.summary.num_forward_sweeps++;
        result.summary.num_backward_sweeps++;
        result.summary.num_top_down_sweeps++;
        result.summary.num_bottom_up_sweeps++;
        break;
      }
      case PatchMatchPropagationSchedule::kCheckerboard: {
        // 单组红黑只会让候选传播一格；四组红黑让一次用户迭代的
        // 等效访问量接近四方向传播，同时仍保留棋盘格更新结构。
        for (std::size_t round = 0;
             round < options.checkerboard_propagation_rounds; ++round) {
          for (const int parity : {0, 1}) {
          const PatchMatchSweepSummary checkerboard =
              RunPatchMatchCheckerboardPropagation(
                  reference_view, source_views, reference_image,
                  source_images, &result.depth_map, &result.normal_map,
                  &result.cost_map, parity, options.cost, source_depth_maps);
          propagation_summary.num_candidate_evaluations +=
              checkerboard.num_candidate_evaluations;
          propagation_summary.num_updates += checkerboard.num_updates;
          propagation_summary.num_geometric_candidate_rejections +=
              checkerboard.num_geometric_candidate_rejections;
          propagation_summary.num_geometric_candidate_acceptances +=
              checkerboard.num_geometric_candidate_acceptances;
            result.summary.num_checkerboard_passes++;
          }
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
        propagation_summary.num_geometric_candidate_rejections +
        perturbation_summary.num_geometric_candidate_rejections;
    result.summary.num_perturbation_geometric_candidate_acceptances +=
        propagation_summary.num_geometric_candidate_acceptances +
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
    const PatchMatchMaps* initial_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options) {
  if (options.pyramid_levels <= 1) {
    PatchMatchStereoResult result = RunPatchMatchStereoForImageSingleScale(
        reference_view, source_views, reference_image, source_images,
        source_depth_maps, depth_min, depth_max, options, initial_maps);
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
        source_depth_maps, depth_min, depth_max, options, initial_maps);
    result.summary.num_pyramid_levels = 1;
    return result;
  }

  PatchMatchMaps prior_maps;
  bool has_prior = initial_maps != nullptr;
  if (initial_maps != nullptr) {
    prior_maps = *initial_maps;
  }
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

PatchMatchStereoResult RunPatchMatchStereoForImageDispatch(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>* source_depth_maps,
    const PatchMatchMaps* initial_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options,
    CudaPatchMatchWorkspace* cuda_workspace) {
  if (options.backend == PatchMatchBackend::kAuto ||
      options.backend == PatchMatchBackend::kCuda) {
    CudaPatchMatchRequest request;
    request.reference_view = &reference_view;
    request.source_views = &source_views;
    request.reference_image = &reference_image;
    request.source_images = &source_images;
    request.source_depth_maps = source_depth_maps;
    request.initial_maps = initial_maps;
    request.depth_min = depth_min;
    request.depth_max = depth_max;
    request.options = options;
    request.workspace = cuda_workspace;

    std::string unavailable_reason;
    std::optional<PatchMatchStereoResult> cuda_result =
        RunCudaPatchMatchStereo(request, &unavailable_reason);
    if (cuda_result.has_value()) {
      cuda_result->summary.requested_backend = options.backend;
      cuda_result->summary.actual_backend = PatchMatchBackend::kCuda;
      return *cuda_result;
    }
    if (options.backend == PatchMatchBackend::kCuda) {
      if (unavailable_reason.empty()) {
        unavailable_reason = "CUDA PatchMatch backend is unavailable.";
      }
      throw std::runtime_error(unavailable_reason);
    }
  }

  PatchMatchStereoResult cpu_result = RunPatchMatchStereoForImageImpl(
      reference_view, source_views, reference_image, source_images,
      source_depth_maps, initial_maps, depth_min, depth_max, options);
  cpu_result.summary.requested_backend = options.backend;
  cpu_result.summary.actual_backend = PatchMatchBackend::kCpu;
  return cpu_result;
}

PatchMatchStereoResult RunPatchMatchStereoForImage(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options) {
  return RunPatchMatchStereoForImageDispatch(
      reference_view, source_views, reference_image, source_images, nullptr,
      nullptr, depth_min, depth_max, options, nullptr);
}

PatchMatchStereoResult RunPatchMatchStereoForImageWithCudaWorkspace(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    float depth_min,
    float depth_max,
    CudaPatchMatchWorkspace* cuda_workspace,
    const PatchMatchStereoOptions& options) {
  return RunPatchMatchStereoForImageDispatch(
      reference_view, source_views, reference_image, source_images, nullptr,
      nullptr, depth_min, depth_max, options, cuda_workspace);
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
  return RunPatchMatchStereoForImageDispatch(
      reference_view, source_views, reference_image, source_images,
      &source_depth_maps, nullptr, depth_min, depth_max, options, nullptr);
}

PatchMatchStereoResult
RunPatchMatchStereoForImageWithSourceDepthsAndCudaWorkspace(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>& source_depth_maps,
    float depth_min,
    float depth_max,
    CudaPatchMatchWorkspace* cuda_workspace,
    const PatchMatchStereoOptions& options,
    const PatchMatchMaps* initial_maps) {
  return RunPatchMatchStereoForImageDispatch(
      reference_view, source_views, reference_image, source_images,
      &source_depth_maps, initial_maps, depth_min, depth_max, options,
      cuda_workspace);
}

PatchMatchStereoResult
RunPatchMatchStereoForImageWithSourceDepthPointersAndCudaWorkspace(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<const DenseDepthMap*>& source_depth_maps,
    float depth_min,
    float depth_max,
    CudaPatchMatchWorkspace* cuda_workspace,
    const PatchMatchStereoOptions& options,
    const PatchMatchMaps* initial_maps) {
  CudaPatchMatchRequest request;
  request.reference_view = &reference_view;
  request.source_views = &source_views;
  request.reference_image = &reference_image;
  request.source_images = &source_images;
  request.source_depth_map_ptrs = &source_depth_maps;
  request.initial_maps = initial_maps;
  request.depth_min = depth_min;
  request.depth_max = depth_max;
  request.options = options;
  request.workspace = cuda_workspace;
  std::string unavailable_reason;
  std::optional<PatchMatchStereoResult> result =
      RunCudaPatchMatchStereo(request, &unavailable_reason);
  if (!result.has_value()) {
    throw std::runtime_error(
        unavailable_reason.empty()
            ? "CUDA PatchMatch backend is unavailable."
            : unavailable_reason);
  }
  result->summary.requested_backend = options.backend;
  result->summary.actual_backend = PatchMatchBackend::kCuda;
  return std::move(*result);
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
  std::vector<const DenseDepthMap*> source_depth_map_ptrs;
  source_depth_map_ptrs.reserve(source_depth_maps.size());
  for (const DenseDepthMap& source_depth_map : source_depth_maps) {
    source_depth_map_ptrs.push_back(&source_depth_map);
  }
  return ComputeGeometricConsistencyGraph(
      reference_view, source_views, reference_depth_map,
      source_depth_map_ptrs, options);
}

ConsistencyGraph ComputeGeometricConsistencyGraph(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const DenseDepthMap& reference_depth_map,
    const std::vector<const DenseDepthMap*>& source_depth_maps,
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

  const std::size_t width = reference_depth_map.Width();
  const std::size_t height = reference_depth_map.Height();
  const std::size_t num_pixels = width * height;
  std::vector<std::size_t> offsets(num_pixels + 1, 0);
  std::vector<ImageId> consistent_source_ids;
  consistent_source_ids.reserve(
      num_pixels * std::min<std::size_t>(source_views.size(), 4));
  for (std::size_t row = 0; row < reference_depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < reference_depth_map.Width(); ++col) {
      const std::size_t pixel_offset = row * width + col;
      offsets[pixel_offset] = consistent_source_ids.size();
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
        if (source_depth_maps[source_idx] == nullptr) {
          continue;
        }
        const std::optional<float> source_depth = SampleDepthBilinear(
            *source_depth_maps[source_idx], source_xy->x(), source_xy->y());
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
          const bool passes_relative_depth =
              options.max_relative_depth_error <= 0.0F ||
              relative_depth_error <= options.max_relative_depth_error;
          if (reprojection_error <= options.max_reprojection_error_pixels &&
              passes_relative_depth) {
            consistent_source_ids.push_back(
                source_views[source_idx].image_id);
          }
        } catch (const std::exception&) {
          continue;
        }
      }
    }
  }
  offsets[num_pixels] = consistent_source_ids.size();
  return ConsistencyGraph::FromCompactData(
      width, height, std::move(offsets),
      std::move(consistent_source_ids));
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
      views.size() != normal_maps.size() ||
      (options.require_cost_map && views.size() != cost_maps.size()) ||
      (!options.require_cost_map && !cost_maps.empty() &&
       views.size() != cost_maps.size())) {
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

  std::vector<std::shared_ptr<const StereoFusionViewData>> owned_views;
  owned_views.reserve(views.size());
  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    auto view_data = std::make_shared<StereoFusionViewData>();
    view_data->image = images[image_idx];
    view_data->depth_map = depth_maps[image_idx];
    view_data->normal_map = normal_maps[image_idx];
    if (!cost_maps.empty()) {
      view_data->cost_map = cost_maps[image_idx];
    }
    if (!consistency_graphs.empty()) {
      view_data->consistency_graph = consistency_graphs[image_idx];
    }
    owned_views.push_back(std::move(view_data));
  }
  return FuseStereoDepthMapsWithViewCache(
      views,
      [&owned_views](std::size_t image_idx) {
        return owned_views.at(image_idx);
      },
      options);
}

StereoFusionResult FuseStereoDepthMapsWithViewCache(
    const std::vector<PatchMatchView>& views,
    const StereoFusionViewLoader& load_view,
    const StereoFusionOptions& options) {
  if (!load_view) {
    throw std::invalid_argument("Stereo fusion view loader is empty.");
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
  const bool compact_visibility =
      options.compact_visibility && views.size() <= 64;
  std::vector<FusionViewGeometry> geometries;
  geometries.reserve(views.size());
  for (const PatchMatchView& view : views) {
    geometries.push_back(MakeFusionViewGeometry(view));
  }

  std::unordered_map<ImageId, std::size_t> image_idx_by_id;
  image_idx_by_id.reserve(views.size());
  ImageId max_image_id = 0;
  for (const PatchMatchView& view : views) {
    max_image_id = std::max(max_image_id, view.image_id);
  }
  constexpr std::size_t kMaxDenseImageIdLookup = 1U << 20;
  const std::size_t invalid_image_idx =
      std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> dense_image_idx_by_id;
  if (static_cast<std::size_t>(max_image_id) <= kMaxDenseImageIdLookup) {
    dense_image_idx_by_id.assign(
        static_cast<std::size_t>(max_image_id) + 1, invalid_image_idx);
  }
  std::vector<std::size_t> widths(views.size());
  std::vector<std::size_t> heights(views.size());
  std::vector<std::size_t> num_pixels(views.size());
  // Validate in reverse order so the bounded LRU retains the first views
  // consumed by the subsequent fusion loop. Count valid depth pixels in this
  // same pass to avoid loading every persisted map a second time.
  for (std::size_t remaining = views.size(); remaining > 0; --remaining) {
    const std::size_t image_idx = remaining - 1;
    const std::shared_ptr<const StereoFusionViewData> view_data =
        load_view(image_idx);
    if (!view_data) {
      throw std::runtime_error("Stereo fusion view loader returned null.");
    }
    const DenseCostMap* cost_map =
        view_data->cost_map.Empty() ? nullptr : &view_data->cost_map;
    const ConsistencyGraph* graph =
        view_data->consistency_graph.Empty()
            ? nullptr
            : &view_data->consistency_graph;
    RequireCompatibleFusionMaps(
        view_data->image, view_data->depth_map, view_data->normal_map,
        cost_map, graph);
    if (options.require_cost_map && cost_map == nullptr) {
      throw std::invalid_argument("Stereo fusion requires cost maps.");
    }
    if (options.require_consistency_graph && graph == nullptr) {
      throw std::invalid_argument(
          "Stereo fusion requires consistency graphs.");
    }
    widths[image_idx] = view_data->depth_map.Width();
    heights[image_idx] = view_data->depth_map.Height();
    num_pixels[image_idx] = view_data->depth_map.NumPixels();
    result.summary.num_depth_pixels +=
        view_data->depth_map.ValidPixelCount();
    image_idx_by_id[views[image_idx].image_id] = image_idx;
    if (!dense_image_idx_by_id.empty()) {
      dense_image_idx_by_id[views[image_idx].image_id] = image_idx;
    }
  }
  const auto find_image_idx =
      [&dense_image_idx_by_id, &image_idx_by_id,
       invalid_image_idx](ImageId image_id) {
        if (static_cast<std::size_t>(image_id) <
            dense_image_idx_by_id.size()) {
          return dense_image_idx_by_id[image_id];
        }
        const auto it = image_idx_by_id.find(image_id);
        return it == image_idx_by_id.end() ? invalid_image_idx : it->second;
      };
  std::vector<std::unique_ptr<std::atomic<std::uint8_t>[]>> visited(
      views.size());
  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    visited[image_idx] = std::make_unique<std::atomic<std::uint8_t>[]>(
        num_pixels[image_idx]);
    for (std::size_t offset = 0; offset < num_pixels[image_idx]; ++offset) {
      visited[image_idx][offset].store(0, std::memory_order_relaxed);
    }
  }
  const auto offset_of = [&widths](std::size_t image_idx, std::size_t row,
                                   std::size_t col) {
    return row * widths[image_idx] + col;
  };
  const auto claim_pixel = [&visited](std::size_t image_idx,
                                      std::size_t offset) {
    std::uint8_t expected = 0;
    return visited[image_idx][offset].compare_exchange_strong(
        expected, 1, std::memory_order_relaxed, std::memory_order_relaxed);
  };
  constexpr double kPi = 3.14159265358979323846;
  const double min_normal_cosine =
      std::cos(static_cast<double>(options.max_normal_error_degrees) *
               kPi / 180.0);
  const double min_view_cosine =
      std::cos(static_cast<double>(options.max_view_angle_degrees) *
               kPi / 180.0);
  const bool collect_view_angles =
      min_view_cosine > -1.0 + 1e-12 || options.collect_diagnostics;

  const std::size_t num_workers =
      static_cast<std::size_t>(std::max(1, parallel::getNumThreads()));
  std::vector<FusionTaskResult> task_results(num_workers);
  for (FusionTaskResult& task : task_results) {
    if (options.collect_diagnostics) {
      task.pair_diagnostics.resize(views.size() * views.size());
      task.pair_diagnostics_seen.assign(views.size() * views.size(), 0);
    }
    task.track_scratch.reserve(options.max_num_fused_observations);
    task.median_scratch.reserve(std::min<std::size_t>(
        options.max_num_fused_observations,
        std::max<std::size_t>(options.min_num_fused_observations, 64)));
    task.view_index.resize(views.size());
  }

  for (std::size_t image_idx = 0; image_idx < views.size(); ++image_idx) {
    const std::shared_ptr<const StereoFusionViewData> reference_view_data =
        load_view(image_idx);
    const int height = static_cast<int>(heights[image_idx]);
    const int row_stripes = std::max(1, (height + 9) / 10);
    parallel::parallel_for_(
        parallel::Range(0, height),
        [&](const parallel::Range& range) {
          const std::size_t worker_idx = static_cast<std::size_t>(
              std::max(0, parallel::getThreadIndex()));
          FusionTaskResult& task =
              task_results[std::min(worker_idx, task_results.size() - 1)];
          StereoFusionResult& local = task.result;
          const auto load_task_view =
              [&](std::size_t requested_idx)
              -> std::shared_ptr<const StereoFusionViewData> {
            std::shared_ptr<const StereoFusionViewData> loaded =
                task.view_index[requested_idx].lock();
            if (loaded) {
              return loaded;
            }
            loaded = load_view(requested_idx);
            task.view_index[requested_idx] = loaded;
            constexpr std::size_t kWorkerViewCacheCapacity = 4;
            if (task.view_hot_cache.size() >= kWorkerViewCacheCapacity) {
              task.view_hot_cache.erase(task.view_hot_cache.begin());
            }
            task.view_hot_cache.emplace_back(requested_idx, loaded);
            return loaded;
          };
          for (int row_int = range.start; row_int < range.end; ++row_int) {
            const std::size_t row = static_cast<std::size_t>(row_int);
            for (std::size_t col = 0; col < widths[image_idx];
                 ++col) {
              if (!reference_view_data->depth_map.IsValid(row, col)) {
                continue;
              }
              const std::size_t seed_offset =
                  offset_of(image_idx, row, col);
              if (visited[image_idx][seed_offset].load(
                      std::memory_order_relaxed) != 0) {
                continue;
              }
              if (!IsCompleteFusionPixel(
                      *reference_view_data, options, row, col)) {
                local.summary.num_skipped_pixels++;
                continue;
              }
              if (!claim_pixel(image_idx, seed_offset)) {
                continue;
              }
              const std::optional<FusionPixelData> seed_data =
                  MakeFusionPixelData(views[image_idx],
                                      geometries[image_idx],
                                      *reference_view_data, row, col);
              if (!seed_data.has_value()) {
                local.summary.num_skipped_pixels++;
                continue;
              }
              local.summary.num_visited_pixels++;

              std::vector<FusionObservation>& track = task.track_scratch;
              track.clear();
              track.push_back({image_idx, row, col, 0, 1.0, 1.0, *seed_data});

              for (std::size_t cursor = 0;
                   cursor < track.size() &&
                   track.size() < options.max_num_fused_observations;
                   ++cursor) {
                const FusionObservation& current = track[cursor];
                if (current.traversal_depth >= options.max_traversal_depth) {
                  continue;
                }
                const FusionPixelData& current_data = current.data;
                const std::shared_ptr<const StereoFusionViewData>
                    current_view_data =
                        load_task_view(current.image_idx);
                ConsistencyGraph::ImageIdRange source_image_ids;
                if (options.prefer_fallback_source_images ||
                    current_view_data->consistency_graph.Empty()) {
                  source_image_ids = ConsistencyGraph::ImageIdRange(
                      current_view_data->fallback_source_image_ids.data(),
                      current_view_data->fallback_source_image_ids.size());
                } else {
                  source_image_ids =
                      current_view_data->consistency_graph.SourceImageIds(
                          current.row, current.col);
                }
                for (const ImageId source_image_id : source_image_ids) {
                  const std::size_t source_idx =
                      find_image_idx(source_image_id);
                  // Match COLMAP's image-level fusion traversal: once an
                  // earlier reference image has completed, later seeds never
                  // traverse back into it. Its accepted pixels have already
                  // been claimed and retrying them only repeats projection,
                  // cache lookup, and visited checks.
                  if (source_idx == invalid_image_idx ||
                      (options.skip_completed_reference_views &&
                       source_idx < image_idx)) {
                    continue;
                  }
                  const std::size_t pair_offset =
                      current.image_idx * views.size() + source_idx;
                  StereoFusionPairDiagnostics* pair_diagnostics = nullptr;
                  if (options.collect_diagnostics) {
                    pair_diagnostics =
                        &task.pair_diagnostics[pair_offset];
                    task.pair_diagnostics_seen[pair_offset] = 1;
                    pair_diagnostics->reference_image_id =
                        views[current.image_idx].image_id;
                    pair_diagnostics->source_image_id =
                        views[source_idx].image_id;
                    pair_diagnostics->num_candidates++;
                  }

                  FusionRejectReason reject_reason = FusionRejectReason::kNone;
                  const std::shared_ptr<const StereoFusionViewData>
                      source_view_data = load_task_view(source_idx);
                  std::optional<FusionObservation> source =
                      ProjectFusionObservationToSource(
                          views, geometries, *source_view_data, options,
                          current, source_idx,
                          current_data.xyz, seed_data->xyz,
                          seed_data->normal.cast<double>(),
                          image_idx,
                          min_normal_cosine,
                          min_view_cosine,
                          visited[source_idx].get(),
                          options.collect_diagnostics
                              ? &local.error_samples
                              : nullptr,
                          &reject_reason);
                  if (!source.has_value()) {
                    switch (reject_reason) {
                      case FusionRejectReason::kVisited:
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_visited++;
                          pair_diagnostics->num_rejected_other++;
                        }
                        break;
                      case FusionRejectReason::kNormal:
                        local.summary.num_rejected_normal_observations++;
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_normal++;
                        }
                        break;
                      case FusionRejectReason::kView:
                        local.summary.num_rejected_view_observations++;
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_view++;
                        }
                        break;
                      case FusionRejectReason::kProjection:
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_projection++;
                          pair_diagnostics->num_rejected_other++;
                        }
                        break;
                      case FusionRejectReason::kDepth:
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_depth++;
                          pair_diagnostics->num_rejected_other++;
                        }
                        break;
                      case FusionRejectReason::kMutualConsistency:
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_mutual_consistency++;
                          pair_diagnostics->num_rejected_other++;
                        }
                        break;
                      case FusionRejectReason::kInvalid:
                      case FusionRejectReason::kNone:
                        if (pair_diagnostics != nullptr) {
                          pair_diagnostics->num_rejected_invalid++;
                          pair_diagnostics->num_rejected_other++;
                        }
                        break;
                    }
                    continue;
                  }
                  const std::size_t source_offset = offset_of(
                      source->image_idx, source->row, source->col);
                  if (!IsCompleteFusionPixel(
                          *source_view_data, options, source->row,
                          source->col)) {
                    if (pair_diagnostics != nullptr) {
                      pair_diagnostics->num_rejected_incomplete++;
                      pair_diagnostics->num_rejected_other++;
                    }
                    continue;
                  }
                  if (!claim_pixel(source->image_idx, source_offset)) {
                    if (pair_diagnostics != nullptr) {
                      pair_diagnostics->num_rejected_visited++;
                      pair_diagnostics->num_rejected_other++;
                    }
                    continue;
                  }
                  const std::optional<FusionPixelData> source_data =
                      MakeFusionPixelData(
                          views[source->image_idx],
                          geometries[source->image_idx], *source_view_data,
                          source->row, source->col, &source->data.normal);
                  if (!source_data.has_value()) {
                    if (pair_diagnostics != nullptr) {
                      pair_diagnostics->num_rejected_incomplete++;
                      pair_diagnostics->num_rejected_other++;
                    }
                    continue;
                  }
                  if (pair_diagnostics != nullptr) {
                    pair_diagnostics->num_accepted++;
                  }
                  local.summary.num_visited_pixels++;
                  source->data = *source_data;
                  track.push_back(*source);
                  if (track.size() >= options.max_num_fused_observations) {
                    break;
                  }
                }
              }

              if (track.size() < options.min_num_fused_observations) {
                local.summary.num_rejected_tracks++;
                local.summary.num_skipped_pixels++;
                continue;
              }

              Eigen::Vector3d xyz;
              Eigen::Vector3d normal;
              Eigen::Vector3d color;
              const auto median_component =
                  [&](const std::size_t axis, const int value_kind) {
                    task.median_scratch.clear();
                    for (const FusionObservation& observation : track) {
                      const FusionPixelData& data = observation.data;
                      if (value_kind == 0) {
                        task.median_scratch.push_back(
                            data.xyz[static_cast<int>(axis)]);
                      } else if (value_kind == 1) {
                        task.median_scratch.push_back(
                            static_cast<double>(
                                data.normal[static_cast<int>(axis)]));
                      } else {
                        task.median_scratch.push_back(
                            static_cast<double>(
                                data.color[static_cast<int>(axis)]));
                      }
                    }
                    return FusionMedian(&task.median_scratch);
                  };
              for (std::size_t axis = 0; axis < 3; ++axis) {
                xyz[static_cast<int>(axis)] = median_component(axis, 0);
                normal[static_cast<int>(axis)] = median_component(axis, 1);
                color[static_cast<int>(axis)] = median_component(axis, 2);
              }
              if (!xyz.allFinite() || !normal.allFinite() ||
                  normal.squaredNorm() <= 0.0) {
                local.summary.num_rejected_tracks++;
                local.summary.num_skipped_pixels++;
                continue;
              }

              normal.normalize();
              for (int axis = 0; axis < 3; ++axis) {
                color[axis] = std::round(ClampFloat(
                    static_cast<float>(color[axis]), 0.0F, 255.0F));
              }
              FusedPoint fused_point;
              fused_point.xyz = xyz;
              fused_point.normal = normal.cast<float>();
              fused_point.color = Eigen::Matrix<std::uint8_t, 3, 1>(
                  static_cast<std::uint8_t>(color.x()),
                  static_cast<std::uint8_t>(color.y()),
                  static_cast<std::uint8_t>(color.z()));
              fused_point.image_id = views[image_idx].image_id;
              fused_point.row = row;
              fused_point.col = col;
              local.points.push_back(fused_point);

              StereoFusionTrack track_diagnostics;
              track_diagnostics.point_idx = local.points.size() - 1;
              track_diagnostics.seed_image_id = views[image_idx].image_id;
              track_diagnostics.seed_row = row;
              track_diagnostics.seed_col = col;
              track_diagnostics.num_observations = track.size();
              track_diagnostics.view_angle_collected =
                  collect_view_angles;
              std::vector<std::uint32_t> point_visibility;
              if (!compact_visibility) {
                point_visibility.reserve(track.size());
              }
              std::uint64_t point_visibility_mask = 0;
              double min_track_normal_cosine = 1.0;
              double min_track_view_cosine = 1.0;
              for (const FusionObservation& observation : track) {
                if (compact_visibility) {
                  point_visibility_mask |=
                      std::uint64_t{1} << observation.image_idx;
                } else {
                  point_visibility.push_back(
                      static_cast<std::uint32_t>(
                          observation.image_idx));
                }
                track_diagnostics.max_traversal_depth = std::max(
                    track_diagnostics.max_traversal_depth,
                    observation.traversal_depth);
                min_track_normal_cosine =
                    std::min(min_track_normal_cosine,
                             observation.normal_cosine);
                if (collect_view_angles) {
                  min_track_view_cosine =
                      std::min(min_track_view_cosine,
                               observation.view_cosine);
                }
              }
              track_diagnostics.max_normal_angle_degrees =
                  DegreesFromCosine(min_track_normal_cosine);
              if (collect_view_angles) {
                track_diagnostics.max_view_angle_degrees =
                    DegreesFromCosine(min_track_view_cosine);
              }
              if (compact_visibility) {
                std::uint64_t remaining = point_visibility_mask;
                while (remaining != 0) {
                  ++track_diagnostics.num_unique_images;
                  remaining &= remaining - 1;
                }
              } else {
                std::sort(point_visibility.begin(),
                          point_visibility.end());
                point_visibility.erase(
                    std::unique(point_visibility.begin(),
                                point_visibility.end()),
                    point_visibility.end());
                track_diagnostics.num_unique_images =
                    point_visibility.size();
              }
              local.tracks.push_back(track_diagnostics);
              if (compact_visibility) {
                local.points_visibility_masks.push_back(
                    point_visibility_mask);
              } else {
                local.points_visibility.push_back(
                    std::move(point_visibility));
              }
              local.summary.num_fusion_tracks++;
              local.summary.num_fused_observations += track.size();
            }
          }
        },
        static_cast<double>(row_stripes));
  }

  std::vector<StereoFusionPairDiagnostics> pair_diagnostics_matrix(
      views.size() * views.size());
  std::vector<std::uint8_t> pair_diagnostics_seen(
      views.size() * views.size(), 0);
  const auto append_samples = [](const std::vector<float>& source,
                                 std::vector<float>* target) {
    constexpr std::size_t kMaxMergedSamples = 500000;
    const std::size_t count =
        std::min(source.size(), kMaxMergedSamples - target->size());
    target->insert(target->end(), source.begin(), source.begin() + count);
  };
  for (FusionTaskResult& task : task_results) {
    const std::size_t point_offset = result.points.size();
    result.points.insert(result.points.end(), task.result.points.begin(),
                         task.result.points.end());
    for (StereoFusionTrack& track : task.result.tracks) {
      track.point_idx += point_offset;
      result.tracks.push_back(track);
    }
    result.points_visibility.insert(
        result.points_visibility.end(),
        std::make_move_iterator(task.result.points_visibility.begin()),
        std::make_move_iterator(task.result.points_visibility.end()));
    result.points_visibility_masks.insert(
        result.points_visibility_masks.end(),
        task.result.points_visibility_masks.begin(),
        task.result.points_visibility_masks.end());
    const StereoFusionSummary& summary = task.result.summary;
    result.summary.num_fusion_tracks += summary.num_fusion_tracks;
    result.summary.num_fused_observations += summary.num_fused_observations;
    result.summary.num_visited_pixels += summary.num_visited_pixels;
    result.summary.num_rejected_tracks += summary.num_rejected_tracks;
    result.summary.num_rejected_normal_observations +=
        summary.num_rejected_normal_observations;
    result.summary.num_rejected_view_observations +=
        summary.num_rejected_view_observations;
    result.summary.num_skipped_pixels += summary.num_skipped_pixels;
    append_samples(task.result.error_samples.relative_depth_errors,
                   &result.error_samples.relative_depth_errors);
    append_samples(task.result.error_samples.reprojection_errors_pixels,
                   &result.error_samples.reprojection_errors_pixels);
    append_samples(task.result.error_samples.normal_angles_degrees,
                   &result.error_samples.normal_angles_degrees);
    append_samples(task.result.error_samples.view_angles_degrees,
                   &result.error_samples.view_angles_degrees);
    for (std::size_t pair_idx = 0;
         pair_idx < task.pair_diagnostics.size(); ++pair_idx) {
      if (task.pair_diagnostics_seen[pair_idx] == 0) {
        continue;
      }
      pair_diagnostics_seen[pair_idx] = 1;
      StereoFusionPairDiagnostics& target =
          pair_diagnostics_matrix[pair_idx];
      const StereoFusionPairDiagnostics& source =
          task.pair_diagnostics[pair_idx];
      target.reference_image_id = source.reference_image_id;
      target.source_image_id = source.source_image_id;
      target.num_candidates += source.num_candidates;
      target.num_accepted += source.num_accepted;
      target.num_rejected_normal += source.num_rejected_normal;
      target.num_rejected_view += source.num_rejected_view;
      target.num_rejected_projection += source.num_rejected_projection;
      target.num_rejected_depth += source.num_rejected_depth;
      target.num_rejected_mutual_consistency +=
          source.num_rejected_mutual_consistency;
      target.num_rejected_invalid += source.num_rejected_invalid;
      target.num_rejected_visited += source.num_rejected_visited;
      target.num_rejected_duplicate += source.num_rejected_duplicate;
      target.num_rejected_incomplete += source.num_rejected_incomplete;
      target.num_rejected_other += source.num_rejected_other;
    }
  }
  result.summary.num_fused_points = result.points.size();
  result.pair_diagnostics.reserve(views.size() * views.size());
  for (std::size_t pair_idx = 0; pair_idx < pair_diagnostics_matrix.size();
       ++pair_idx) {
    if (pair_diagnostics_seen[pair_idx] != 0) {
      result.pair_diagnostics.push_back(pair_diagnostics_matrix[pair_idx]);
    }
  }
  return result;
}

}  // namespace colite::mvs
