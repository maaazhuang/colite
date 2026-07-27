#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/features/sift_extractor.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

struct ReconSfmControllerOptions {
  std::filesystem::path image_path;
  std::filesystem::path output_path;
  double focal_length = 0.0;
  int max_num_features = 4096;
  int feature_num_threads = -1;
  SiftBackend feature_backend = SiftBackend::kAuto;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kFlann;
  // Match the standalone exhaustive matcher unless relaxation is explicitly
  // requested by the caller.
  std::size_t nearby_match_window = 0;
  double nearby_match_ratio = 0.8;
  std::size_t max_registration_trials_per_image = 3;
  std::size_t max_initialization_trials = 3;
  double registration_retry_reprojection_error_multiplier = 1.5;
  double registration_retry_max_reprojection_error_px = 8.0;
  CameraParameterGroup ba_refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> ba_refine_camera_param_names;
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

struct ReconSfmRunStats {
  std::string feature_backend;
  std::string matcher_backend;
  bool cuda_sift_available = false;
  bool cuda_matcher_available = false;
  double initial_focal_length = 0.0;
  std::string initial_focal_source;
  bool has_prior_focal_length = false;
  int cpu_threads = 1;
  std::size_t num_extracted_features = 0;
  std::size_t num_raw_matches = 0;
  std::size_t num_verified_inliers = 0;
  std::size_t num_weak_pairs_checked = 0;
  std::size_t num_weak_pairs_rematched = 0;
  std::size_t num_weak_pairs_improved = 0;
  double feature_extraction_seconds = 0.0;
  double frontend_pipeline_seconds = 0.0;
  double matching_seconds = 0.0;
  double geometry_verification_seconds = 0.0;
  double weak_pair_rematching_seconds = 0.0;
  double weak_candidate_generation_work_seconds = 0.0;
  double weak_descriptor_search_work_seconds = 0.0;
  double weak_geometry_verification_work_seconds = 0.0;
  double incremental_mapping_seconds = 0.0;
  double color_extraction_seconds = 0.0;
  double output_write_seconds = 0.0;
  double total_seconds = 0.0;
};

class ReconSfmController {
 public:
  explicit ReconSfmController(ReconSfmControllerOptions options);

  const ReconSfmControllerOptions& Options() const;

  int EffectiveFeatureExtractionThreads() const;

  int Run(std::ostream& out) const;

  void MatchAllImagePairs(const std::vector<ImageId>& image_ids,
                          const DescriptorMatchingOptions& base_options,
                          MatchGraph* graph) const;

 private:
  ReconSfmControllerOptions options_;
};

int EffectiveFeatureExtractionThreads(int requested_threads);

ReconstructionQualityStats ComputeReconstructionQualityStats(
    const Reconstruction& reconstruction);

// Sample every registered observation from its source image and assign the
// multi-view mean RGB value to each sparse point, matching COLMAP's
// ExtractColorsForAllImages convention.
void ExtractPointColors(Reconstruction* reconstruction);

void WriteRunSummary(const std::filesystem::path& output_path,
                     const Reconstruction& reconstruction,
                     const IncrementalMappingSummary& summary,
                     const ReconSfmRunStats& run_stats = {});

void WriteColmapTextModel(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction);

void WritePairDiagnostics(const std::filesystem::path& output_path,
                          const Reconstruction& reconstruction,
                          const MatchGraph& graph);

void WriteRegistrationDiagnostics(
    const std::filesystem::path& output_path,
    const IncrementalMappingSummary& summary);

void WritePlyPointCloud(const std::filesystem::path& ply_path,
                        const Reconstruction& reconstruction);

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph);

}  // namespace colite::controllers
