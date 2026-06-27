#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "colite/core/camera.h"
#include "colite/core/pose.h"
#include "colite/mvs/dense_map.h"
#include "colite/mvs/image_workspace.h"
#include "colite/mvs/mvs_pipeline.h"

namespace colite::mvs {

struct PatchMatchInitializationOptions {
  std::uint32_t random_seed = 42;
  float sparse_depth_min_percentile = 0.01F;
  float sparse_depth_max_percentile = 0.99F;
  float depth_range_padding = 0.0F;
  bool randomize_initial_normals = false;
  float initial_normal_hemisphere_degrees = 60.0F;

  // 第一版统一使用朝向相机的初始法线；后续会从局部平面或随机半球采样扩展。
  Eigen::Vector3f default_normal = Eigen::Vector3f(0.0F, 0.0F, -1.0F);
};

struct PatchMatchMaps {
  DenseDepthMap depth_map;
  NormalMap normal_map;
};

struct PatchMatchView {
  ImageId image_id = kInvalidImageId;
  Camera camera;
  Pose3 cam_from_world;
};

struct PatchMatchCostOptions {
  int window_radius = 2;
  int window_step = 1;
  float invalid_cost = 1.0F;
  float min_variance = 1e-8F;
  // 几何一致性软约束权重。为 0 时保持纯光度 PatchMatch 行为。
  float geometric_consistency_weight = 0.0F;
  // source depth 与当前候选深度的相对误差归一化阈值。
  float geometric_max_relative_depth_error = 0.01F;

  // 多源聚合时使用最低 cost 的前 K 个源图。0 表示使用全部有效源图。
  std::size_t max_num_aggregated_costs = 3;
  std::size_t min_num_valid_sources = 1;
};

struct PatchMatchCost {
  bool valid = false;
  float score = -1.0F;
  float cost = 1.0F;
  std::size_t num_samples = 0;
};

struct MultiSourcePatchMatchCost {
  bool valid = false;
  float cost = 1.0F;
  std::size_t num_total_sources = 0;
  std::size_t num_valid_sources = 0;
  std::size_t num_used_sources = 0;
};

enum class PatchMatchSweepDirection {
  kForward,
  kBackward,
  kTopDown,
  kBottomUp,
};

enum class PatchMatchPropagationSchedule {
  kForwardBackward,
  kFourDirection,
  kCheckerboard,
};

struct PatchMatchSweepSummary {
  std::size_t num_pixels_visited = 0;
  std::size_t num_candidate_evaluations = 0;
  std::size_t num_updates = 0;
};

struct PatchMatchPerturbationOptions {
  std::uint32_t random_seed = 42;
  float depth_radius = 0.1F;
  float normal_radius_degrees = 5.0F;
  std::size_t num_samples_per_pixel = 1;
};

struct PatchMatchPerturbationSummary {
  std::size_t num_pixels_visited = 0;
  std::size_t num_candidate_evaluations = 0;
  std::size_t num_updates = 0;
  std::size_t num_geometric_candidate_rejections = 0;
  std::size_t num_geometric_candidate_acceptances = 0;
  double total_abs_depth_update = 0.0;
  double max_abs_depth_update = 0.0;
  double mean_abs_depth_update = 0.0;
  double total_normal_angle_degrees = 0.0;
  double max_normal_angle_degrees = 0.0;
  double mean_normal_angle_degrees = 0.0;
};

struct PatchMatchNormalRefinementOptions {
  bool enabled = false;
  float max_relative_depth_difference = 0.005F;
};

struct PatchMatchNormalRefinementSummary {
  std::size_t num_pixels = 0;
  std::size_t num_refined_pixels = 0;
  std::size_t num_skipped_pixels = 0;
};

struct PatchMatchStereoOptions {
  PatchMatchInitializationOptions initialization;
  PatchMatchCostOptions cost;
  PatchMatchPerturbationOptions perturbation;
  PatchMatchPropagationSchedule propagation_schedule =
      PatchMatchPropagationSchedule::kForwardBackward;
  std::size_t num_iterations = 3;
  float initial_depth_perturbation_factor = 0.5F;
  float perturbation_shrink = 0.5F;
  float min_depth_perturbation = 1e-3F;
  float min_normal_radius_degrees = 0.25F;
  float adaptive_perturbation_min_update_ratio = 0.01F;
  PatchMatchNormalRefinementOptions normal_refinement;
  // 多尺度 PatchMatch 金字塔层数。1 表示保持原有单尺度流程。
  std::size_t pyramid_levels = 1;
};

struct PatchMatchStereoSummary {
  std::size_t num_pyramid_levels = 1;
  std::size_t num_iterations = 0;
  std::size_t num_forward_sweeps = 0;
  std::size_t num_backward_sweeps = 0;
  std::size_t num_top_down_sweeps = 0;
  std::size_t num_bottom_up_sweeps = 0;
  std::size_t num_checkerboard_passes = 0;
  std::size_t num_perturbation_passes = 0;
  std::size_t num_candidate_evaluations = 0;
  std::size_t num_propagation_updates = 0;
  std::size_t num_perturbation_updates = 0;
  double perturbation_total_abs_depth_update = 0.0;
  double perturbation_max_abs_depth_update = 0.0;
  double perturbation_mean_abs_depth_update = 0.0;
  double perturbation_total_normal_angle_degrees = 0.0;
  double perturbation_max_normal_angle_degrees = 0.0;
  double perturbation_mean_normal_angle_degrees = 0.0;
  std::size_t num_perturbation_geometric_candidate_rejections = 0;
  std::size_t num_perturbation_geometric_candidate_acceptances = 0;
  std::size_t num_valid_cost_pixels = 0;
  std::size_t num_normal_refinement_pixels = 0;
  std::size_t num_normal_refined_pixels = 0;
  std::size_t num_normal_refinement_skipped_pixels = 0;
  std::vector<double> update_ratios;
  std::vector<float> depth_radii;
  std::vector<float> normal_radii_degrees;
};

struct PatchMatchStereoResult {
  DenseDepthMap depth_map;
  NormalMap normal_map;
  DenseCostMap cost_map;
  PatchMatchStereoSummary summary;
};

struct PatchMatchPhotometricStats {
  std::size_t num_pixels = 0;
  std::size_t num_valid_cost_pixels = 0;
  std::size_t num_invalid_cost_pixels = 0;
  double valid_ratio = 0.0;
  double mean_cost = 0.0;
  double median_cost = 0.0;
  double min_cost = 0.0;
  double max_cost = 0.0;
};

struct PatchMatchPhotometricFilterSummary {
  std::size_t num_pixels = 0;
  std::size_t num_kept_pixels = 0;
  std::size_t num_filtered_pixels = 0;
};

struct PatchMatchGeometricConsistencyOptions {
  float max_reprojection_error_pixels = 2.0F;
  float max_relative_depth_error = 0.01F;
};

class ConsistencyGraph {
 public:
  ConsistencyGraph() = default;
  ConsistencyGraph(std::size_t width, std::size_t height);

  std::size_t Width() const;
  std::size_t Height() const;
  std::size_t NumPixels() const;
  bool Empty() const;

  void Add(std::size_t row, std::size_t col, ImageId source_image_id);
  const std::vector<ImageId>& SourceImageIds(std::size_t row,
                                             std::size_t col) const;
  std::size_t Count(std::size_t row, std::size_t col) const;
  std::size_t NumConsistentPixels(std::size_t min_num_sources) const;

 private:
  std::size_t Offset(std::size_t row, std::size_t col) const;

  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::vector<std::vector<ImageId>> data_;
};

struct PatchMatchGeometricFilterSummary {
  std::size_t num_pixels = 0;
  std::size_t num_kept_pixels = 0;
  std::size_t num_filtered_pixels = 0;
};

struct FusedPoint {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  Eigen::Matrix<std::uint8_t, 3, 1> color =
      Eigen::Matrix<std::uint8_t, 3, 1>(255, 255, 255);
  ImageId image_id = kInvalidImageId;
  std::size_t row = 0;
  std::size_t col = 0;
};

struct StereoFusionOptions {
  // Fusion 以 consistency graph 作为主要可见性约束。
  std::size_t min_num_consistent_sources = 1;
  // 一个融合 track 至少需要多少个像素观测。设为 2 时更接近 COLMAP 的多视图融合思路。
  std::size_t min_num_fused_observations = 1;
  std::size_t max_num_fused_observations = 100000;
  std::size_t max_traversal_depth = 100;
  float max_reprojection_error_pixels = 2.0F;
  float max_relative_depth_error = 0.01F;
  float max_normal_error_degrees = 10.0F;
  float max_view_angle_degrees = 180.0F;
  // Fusion 时 source 投影点附近搜索有效像素的半径。PatchMatch 的
  // consistency graph 使用双线性深度采样，而 fusion 需要离散像素；
  // 1px 搜索可以减少因为四舍五入像素被过滤导致的单观测 track。
  std::size_t source_pixel_search_radius = 1;
  bool require_mutual_consistency = true;
  bool require_consistency_graph = true;
};

struct StereoFusionSummary {
  std::size_t num_input_images = 0;
  std::size_t num_depth_pixels = 0;
  std::size_t num_fusion_tracks = 0;
  std::size_t num_fused_points = 0;
  std::size_t num_fused_observations = 0;
  std::size_t num_visited_pixels = 0;
  std::size_t num_rejected_tracks = 0;
  std::size_t num_rejected_normal_observations = 0;
  std::size_t num_rejected_view_observations = 0;
  std::size_t num_skipped_pixels = 0;
};

struct StereoFusionTrack {
  std::size_t point_idx = 0;
  ImageId seed_image_id = kInvalidImageId;
  std::size_t seed_row = 0;
  std::size_t seed_col = 0;
  std::size_t num_observations = 0;
  std::size_t num_unique_images = 0;
  std::size_t max_traversal_depth = 0;
  double max_normal_angle_degrees = 0.0;
  double max_view_angle_degrees = 0.0;
};

struct StereoFusionPairDiagnostics {
  ImageId reference_image_id = kInvalidImageId;
  ImageId source_image_id = kInvalidImageId;
  std::size_t num_candidates = 0;
  std::size_t num_accepted = 0;
  std::size_t num_rejected_normal = 0;
  std::size_t num_rejected_view = 0;
  std::size_t num_rejected_projection = 0;
  std::size_t num_rejected_depth = 0;
  std::size_t num_rejected_mutual_consistency = 0;
  std::size_t num_rejected_invalid = 0;
  std::size_t num_rejected_visited = 0;
  std::size_t num_rejected_duplicate = 0;
  std::size_t num_rejected_incomplete = 0;
  std::size_t num_rejected_other = 0;
};

struct StereoFusionErrorSamples {
  std::vector<float> relative_depth_errors;
  std::vector<float> reprojection_errors_pixels;
  std::vector<float> normal_angles_degrees;
  std::vector<float> view_angles_degrees;
};

struct StereoFusionResult {
  std::vector<FusedPoint> points;
  std::vector<StereoFusionTrack> tracks;
  std::vector<StereoFusionPairDiagnostics> pair_diagnostics;
  StereoFusionErrorSamples error_samples;
  StereoFusionSummary summary;
};

PatchMatchMaps InitializePatchMatchMaps(
    const MvsImage& image,
    float depth_min,
    float depth_max,
    const PatchMatchInitializationOptions& options = {});

PatchMatchMaps InitializePatchMatchMaps(
    const MvsImage& image,
    const SparseDepthMap& sparse_depth_map,
    const PatchMatchInitializationOptions& options = {});

std::optional<float> SampleGrayBilinear(const MvsImage& image,
                                        float x,
                                        float y);

PatchMatchCost ComputePatchZncc(
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    const Eigen::Vector2f& source_xy,
    const PatchMatchCostOptions& options = {});

std::optional<Eigen::Vector2f> ProjectDepthToSourcePixel(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const Eigen::Vector2f& reference_xy,
    float depth);

std::optional<Eigen::Vector2f> ProjectPlaneToSourcePixel(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const Eigen::Vector2f& reference_center_xy,
    float center_depth,
    const Eigen::Vector3f& reference_normal,
    const Eigen::Vector2f& sample_xy);

PatchMatchCost ComputeDepthHypothesisCost(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const PatchMatchCostOptions& options = {});

PatchMatchCost ComputePlaneHypothesisCost(
    const PatchMatchView& reference_view,
    const PatchMatchView& source_view,
    const MvsImage& reference_image,
    const MvsImage& source_image,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const Eigen::Vector3f& reference_normal,
    const PatchMatchCostOptions& options = {},
    const DenseDepthMap* source_depth_map = nullptr);

MultiSourcePatchMatchCost ComputeMultiSourcePlaneHypothesisCost(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const Eigen::Vector2f& reference_xy,
    float depth,
    const Eigen::Vector3f& reference_normal,
    const PatchMatchCostOptions& options = {},
    const std::vector<DenseDepthMap>* source_depth_maps = nullptr);

DenseCostMap ComputeBestCostMap(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const DenseDepthMap& depth_map,
    const NormalMap& normal_map,
    const PatchMatchCostOptions& options = {},
    const std::vector<DenseDepthMap>* source_depth_maps = nullptr);

PatchMatchSweepSummary RunPatchMatchPropagationSweep(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    PatchMatchSweepDirection direction,
    const PatchMatchCostOptions& options = {},
    const std::vector<DenseDepthMap>* source_depth_maps = nullptr);

PatchMatchSweepSummary RunPatchMatchCheckerboardPropagation(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    int parity,
    const PatchMatchCostOptions& options = {},
    const std::vector<DenseDepthMap>* source_depth_maps = nullptr);

PatchMatchPerturbationSummary RunPatchMatchRandomPerturbation(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map,
    const PatchMatchPerturbationOptions& perturbation_options = {},
    const PatchMatchCostOptions& cost_options = {},
    const std::vector<DenseDepthMap>* source_depth_maps = nullptr);

PatchMatchNormalRefinementSummary RefineNormalMapFromDepth(
    const PatchMatchView& reference_view,
    const DenseDepthMap& depth_map,
    const PatchMatchNormalRefinementOptions& options,
    NormalMap* normal_map);

PatchMatchStereoResult RunPatchMatchStereoForImage(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options = {});

PatchMatchStereoResult RunPatchMatchStereoForImageWithSourceDepths(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const MvsImage& reference_image,
    const std::vector<MvsImage>& source_images,
    const std::vector<DenseDepthMap>& source_depth_maps,
    float depth_min,
    float depth_max,
    const PatchMatchStereoOptions& options = {});

PatchMatchPhotometricStats ComputePatchMatchPhotometricStats(
    const DenseCostMap& cost_map);

PatchMatchPhotometricFilterSummary FilterPatchMatchPhotometricResults(
    float max_cost,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map);

ConsistencyGraph ComputeGeometricConsistencyGraph(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views,
    const DenseDepthMap& reference_depth_map,
    const std::vector<DenseDepthMap>& source_depth_maps,
    const PatchMatchGeometricConsistencyOptions& options = {});

void WriteConsistencyGraphBinary(const std::filesystem::path& path,
                                 const ConsistencyGraph& consistency_graph);
ConsistencyGraph ReadConsistencyGraphBinary(const std::filesystem::path& path);

PatchMatchGeometricFilterSummary FilterPatchMatchGeometricResults(
    const ConsistencyGraph& consistency_graph,
    std::size_t min_num_consistent_sources,
    DenseDepthMap* depth_map,
    NormalMap* normal_map,
    DenseCostMap* cost_map);

StereoFusionResult FuseStereoDepthMaps(
    const std::vector<PatchMatchView>& views,
    const std::vector<MvsImage>& images,
    const std::vector<DenseDepthMap>& depth_maps,
    const std::vector<NormalMap>& normal_maps,
    const std::vector<DenseCostMap>& cost_maps,
    const std::vector<ConsistencyGraph>& consistency_graphs,
    const StereoFusionOptions& options = {});

}  // namespace colite::mvs
