#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::app {

struct ReconSfmCliOptions {
  bool show_help = false;
  bool show_version = false;
  std::string image_path;
  std::string output_path;
  double focal_length = 0.0;
  int max_num_features = 4096;
  bool use_flann = false;
  std::size_t nearby_match_window = 2;
  double nearby_match_ratio = 0.9;
};

struct NearbyPairMatchingOptions {
  // 文件顺序距离不超过该窗口的图像对会做二次匹配；0 表示关闭。
  std::size_t image_index_window = 2;

  // 近邻二次匹配使用更宽松的 ratio，但仍保留 mutual check。
  double max_ratio = 0.9;
};

struct ReconstructionQualityStats {
  std::size_t num_observations = 0;
  double mean_reprojection_error = -1.0;
  double max_reprojection_error = -1.0;
};

int RunReconSfmApp(const std::vector<std::string>& args,
                   std::ostream& out,
                   std::ostream& err);

double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length);

ReconstructionQualityStats ComputeReconstructionQualityStats(
    const Reconstruction& reconstruction);

void WriteRunSummary(const std::filesystem::path& output_path,
                     const Reconstruction& reconstruction,
                     const IncrementalMappingSummary& summary);

void WriteColmapTextModel(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction);

void WritePairDiagnostics(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction,
                          const MatchGraph& graph);

void WritePlyPointCloud(const std::filesystem::path& ply_path,
                        const Reconstruction& reconstruction);

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph);

}  // namespace colite::app
