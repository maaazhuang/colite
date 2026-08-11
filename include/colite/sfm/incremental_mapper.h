#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/sfm/image_registrar.h"
#include "colite/sfm/incremental_initializer.h"
#include "colite/sfm/incremental_triangulator.h"
#include "colite/sfm/local_bundle_adjuster.h"
#include "colite/sfm/observation_filter.h"
#include "colite/sfm/structureless_registrar.h"

namespace colite {

// 注册单张新图像的参数。当前闭环为：
// PnP 注册 -> track 补全 -> 新点三角化 -> 局部 BA -> 坏观测过滤。
struct IncrementalMapperOptions {
  ImageRegistrarOptions registration;
  IncrementalTriangulatorOptions triangulation;
  LocalBundleAdjustmentOptions local_bundle_adjustment;
  ObservationFilterOptions local_retriangulation_filter;
  ObservationFilterOptions observation_filter;
  StructurelessRegistrationOptions structureless_registration;
  bool use_structureless_registration = false;
  bool run_local_bundle_adjustment = true;
  bool run_local_retriangulation_filter = true;
  bool run_observation_filter = true;
};

struct IncrementalMapperSummary {
  bool success = false;
  // 注册轮次，从 1 开始；用于对齐 COLMAP mapper 日志中的注册顺序。
  std::size_t registration_order = 0;
  ImageId image_id = kInvalidImageId;
  // 该图像被选中注册前的候选评分，来自 SelectNextImage 的同一套统计。
  std::size_t candidate_num_visible_points = 0;
  // 可见 3D 点背后的总 track 观测数，用于区分同样可见点数量但 track 支撑不同的候选图像。
  std::size_t candidate_num_visible_observations = 0;
  double candidate_mean_track_length = 0.0;
  std::size_t candidate_num_registered_neighbors = 0;
  std::size_t candidate_num_verified_inliers_to_registered_images = 0;
  double candidate_score = 0.0;
  // 成功注册前该图像已经失败过的次数，用于诊断失败重试调度是否稳定。
  std::size_t registration_attempts_before_success = 0;
  // 本次 PnP 注册实际使用的重投影误差阈值。失败重试时该值可能被动态放宽。
  double registration_max_reprojection_error_px = 0.0;
  ImageRegistrationSummary registration;
  StructurelessRegistrationSummary structureless_registration;
  bool used_structureless_registration = false;
  IncrementalTriangulatorSummary track_completion;
  IncrementalTriangulatorSummary triangulation;
  IncrementalTriangulatorSummary local_retriangulation;
  LocalBundleAdjustmentSummary local_bundle_adjustment;
  ObservationFilterSummary observation_filter;
  double registration_seconds = 0.0;
  double track_completion_seconds = 0.0;
  double triangulation_seconds = 0.0;
  double local_retriangulation_seconds = 0.0;
  double local_bundle_adjustment_seconds = 0.0;
  double observation_filter_seconds = 0.0;
  std::string message;
};

// 完整增量建图参数。initialization 负责初始图像对，mapper 负责后续每张图像。
struct IncrementalMappingOptions {
  IncrementalInitializerOptions initialization;
  IncrementalMapperOptions mapper;
  GlobalBundleAdjustmentOptions global_bundle_adjustment;
  ObservationFilterOptions global_observation_filter;
  std::size_t max_num_registrations = 1000;
  // 单张图像最多尝试注册次数。失败图像会被降级重试，超过上限后跳过，避免调度被坏候选阻塞。
  std::size_t max_num_registration_trials_per_image = 3;
  // 失败重试时按尝试次数动态放宽 PnP RANSAC 重投影阈值：
  // effective = min(max_px, base_px * multiplier ^ previous_failures)。
  double registration_retry_reprojection_error_multiplier = 1.5;
  double registration_retry_max_reprojection_error_px = 8.0;
  bool run_global_track_completion = true;
  bool run_pose_guided_track_completion = true;
  bool run_global_bundle_adjustment = true;
  bool run_global_observation_filter = true;
  bool run_final_global_bundle_adjustment = true;
  // Empty means all images. Registration and initialization are restricted to
  // this set when running one component of a multi-model reconstruction.
  std::set<ImageId> allowed_image_ids;
};

struct ImageRegistrationFailureSummary {
  ImageId image_id = kInvalidImageId;
  std::size_t attempt = 0;
  std::size_t candidate_num_visible_points = 0;
  std::size_t candidate_num_visible_observations = 0;
  double candidate_mean_track_length = 0.0;
  std::size_t candidate_num_registered_neighbors = 0;
  std::size_t candidate_num_verified_inliers_to_registered_images = 0;
  double candidate_score = 0.0;
  double registration_max_reprojection_error_px = 0.0;
  std::size_t registration_num_correspondences = 0;
  std::size_t registration_num_ransac_inliers = 0;
  std::size_t registration_num_inliers = 0;
  std::string message;
};

struct IncrementalMappingSummary {
  bool success = false;
  IncrementalInitializerSummary initialization;
  std::vector<IncrementalMapperSummary> registrations;
  std::vector<ImageRegistrationFailureSummary> failed_registrations;
  IncrementalTriangulatorSummary global_track_completion;
  IncrementalTriangulatorSummary pose_guided_track_completion;
  IncrementalTriangulatorSummary global_retriangulation;
  GlobalBundleAdjustmentSummary global_bundle_adjustment;
  ObservationFilterSummary global_observation_filter;
  GlobalBundleAdjustmentSummary final_global_bundle_adjustment;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
  double initialization_seconds = 0.0;
  double candidate_selection_seconds = 0.0;
  double global_track_completion_seconds = 0.0;
  double pose_guided_track_completion_seconds = 0.0;
  double global_retriangulation_seconds = 0.0;
  double global_bundle_adjustment_seconds = 0.0;
  double global_observation_filter_seconds = 0.0;
  double final_global_bundle_adjustment_seconds = 0.0;
  std::string message;
};

// 注册指定的新图像，并在注册成功后立刻执行局部精化和过滤。
IncrementalMapperSummary RegisterNextImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalMapperOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache = nullptr);

// 对新注册图像及其已注册强邻居执行局部重三角化，补上新图像引入后可见的
// 邻居-邻居 track，作为局部 BA 前的几何增密步骤。
IncrementalTriangulatorSummary RetriangulateLocalNeighborhood(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    const ObservationFilterOptions* local_filter_options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache = nullptr);

// 所有图像完成注册后重新遍历一次已注册图像，在完整位姿图上创建增量阶段
// 遗漏的点。该阶段对应 COLMAP 全局优化前的 Retriangulate。
IncrementalTriangulatorSummary RetriangulateAllRegisteredImages(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache = nullptr);

// 从所有未注册图像中选择下一张最适合注册的图像。
// 分数为“该图像通过 verified matches 能看到的已有 3D 点数量”。
std::optional<ImageId> SelectNextImage(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction);

// 完整增量 SfM 主循环：初始化模型，然后持续选择并注册新图像。
IncrementalMappingSummary RunIncrementalMapping(
    const MatchGraph& graph,
    const IncrementalMappingOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
