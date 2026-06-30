#pragma once

#include <cstddef>
#include <optional>
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

namespace colite {

// 注册单张新图像的参数。当前闭环为：
// PnP 注册 -> track 补全 -> 新点三角化 -> 局部 BA -> 坏观测过滤。
struct IncrementalMapperOptions {
  ImageRegistrarOptions registration;
  IncrementalTriangulatorOptions triangulation;
  LocalBundleAdjustmentOptions local_bundle_adjustment;
  ObservationFilterOptions local_retriangulation_filter;
  ObservationFilterOptions observation_filter;
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
  ImageRegistrationSummary registration;
  IncrementalTriangulatorSummary track_completion;
  IncrementalTriangulatorSummary triangulation;
  IncrementalTriangulatorSummary local_retriangulation;
  LocalBundleAdjustmentSummary local_bundle_adjustment;
  ObservationFilterSummary observation_filter;
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
  bool run_global_track_completion = true;
  bool run_global_bundle_adjustment = true;
  bool run_global_observation_filter = true;
  bool run_final_global_bundle_adjustment = true;
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
  std::string message;
};

struct IncrementalMappingSummary {
  bool success = false;
  IncrementalInitializerSummary initialization;
  std::vector<IncrementalMapperSummary> registrations;
  std::vector<ImageRegistrationFailureSummary> failed_registrations;
  IncrementalTriangulatorSummary global_track_completion;
  GlobalBundleAdjustmentSummary global_bundle_adjustment;
  ObservationFilterSummary global_observation_filter;
  GlobalBundleAdjustmentSummary final_global_bundle_adjustment;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
  std::string message;
};

// 注册指定的新图像，并在注册成功后立刻执行局部精化和过滤。
IncrementalMapperSummary RegisterNextImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalMapperOptions& options,
    Reconstruction* reconstruction);

// 对新注册图像及其已注册强邻居执行局部重三角化，补上新图像引入后可见的
// 邻居-邻居 track，作为局部 BA 前的几何增密步骤。
IncrementalTriangulatorSummary RetriangulateLocalNeighborhood(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    const ObservationFilterOptions* local_filter_options,
    Reconstruction* reconstruction);

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
