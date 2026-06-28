#pragma once

#include <cstddef>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/features/sift_extractor.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite::controllers {

struct ReconSfmControllerOptions {
  // 输入图像目录和输出目录由 exe 层解析，controller 只关心已验证后的配置。
  double focal_length = 0.0;
  int max_num_features = 4096;
  int feature_num_threads = -1;
  SiftBackend feature_backend = SiftBackend::kCvSift;
  bool use_flann = false;
  std::size_t nearby_match_window = 2;
  double nearby_match_ratio = 0.9;
};

struct NearbyPairMatchingOptions {
  // 文件顺序距离不超过该窗口的图像对会做更宽松的近邻匹配；0 表示关闭。
  std::size_t image_index_window = 2;

  // 近邻匹配使用更宽松的 ratio，但仍保留 mutual check。
  double max_ratio = 0.9;
};

struct ReconstructionQualityStats {
  std::size_t num_observations = 0;
  double mean_reprojection_error = -1.0;
  double max_reprojection_error = -1.0;
};

class ReconSfmController {
 public:
  explicit ReconSfmController(ReconSfmControllerOptions options);

  const ReconSfmControllerOptions& Options() const;

  int EffectiveFeatureExtractionThreads() const;

  void MatchAllImagePairs(const std::vector<ImageId>& image_ids,
                          const DescriptorMatchingOptions& base_options,
                          MatchGraph* graph) const;

 private:
  ReconSfmControllerOptions options_;
};

int EffectiveFeatureExtractionThreads(int requested_threads);

ReconstructionQualityStats ComputeReconstructionQualityStats(
    const Reconstruction& reconstruction);

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph);

}  // namespace colite::controllers
