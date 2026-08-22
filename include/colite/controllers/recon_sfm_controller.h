#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/controllers/sfm_camera_setup.h"
#include "colite/features/aliked_extractor.h"
#include "colite/features/sift_extractor.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

enum class SfmMapperType {
  kIncremental,
  kGlobal,
  kHierarchical,
};

struct ReconSfmControllerOptions {
  std::filesystem::path image_path;
  std::filesystem::path output_path;
  std::filesystem::path input_model_path;
  ColmapModelFormat input_model_format = ColmapModelFormat::kAuto;
  ColmapModelFormat output_model_format =
      ColmapModelFormat::kTextAndBinary;
  bool resume = false;
  bool force = false;
  double focal_length = 0.0;
  CameraSharingMode camera_sharing_mode = CameraSharingMode::kAuto;
  CameraModelId camera_model_id = CameraModelId::kSimpleRadial;
  std::vector<double> camera_params;
  double default_focal_length_factor = 1.2;
  int max_num_features = 4096;
  int feature_num_threads = -1;
  FeatureExtractorType feature_extractor_type = FeatureExtractorType::kSift;
  SiftBackend feature_backend = SiftBackend::kAuto;
  std::filesystem::path aliked_model_path;
  double aliked_min_score = 0.2;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kFlann;
  FeatureMatcherType matcher_type = FeatureMatcherType::kSiftBruteForce;
  int aliked_max_image_size = 1600;
  std::size_t learned_cuda_memory_limit_mb = 8192;
  std::filesystem::path learned_matcher_model_path;
  double min_cosine_similarity = 0.85;
  double lightglue_min_score = 0.1;
  int learned_matcher_num_threads = 1;
  OnnxExecutionProvider learned_execution_provider =
      OnnxExecutionProvider::kAuto;
  int learned_gpu_index = 0;
  SfmMapperType mapper_type = SfmMapperType::kIncremental;
  std::size_t hierarchical_target_cluster_size = 100;
  std::size_t hierarchical_max_cluster_size = 150;
  std::size_t hierarchical_overlap = 10;
  // Match the standalone exhaustive matcher unless relaxation is explicitly
  // requested by the caller.
  std::size_t hierarchical_num_workers = 0;
  std::size_t nearby_match_window = 0;
  double nearby_match_ratio = 0.8;
  double guided_match_ratio = 0.95;
  std::size_t max_registration_trials_per_image = 3;
  std::size_t max_initialization_trials = 3;
  bool use_structureless_registration = false;
  bool multiple_models = true;
  bool merge_models = true;
  std::size_t max_num_models = 50;
  std::size_t min_num_images_per_model = 2;
  double registration_retry_reprojection_error_multiplier = 1.5;
  double registration_retry_max_reprojection_error_px = 8.0;
  CameraParameterGroup ba_refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> ba_refine_camera_param_names;
  bool use_position_priors = false;
  double position_prior_stddev = 5.0;
  double position_prior_loss_scale = 2.0;
  double position_prior_alignment_max_error = 20.0;
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
  std::string mapper_type = "incremental";
  std::string feature_backend;
  std::string matcher_backend;
  bool cuda_sift_available = false;
  bool cuda_matcher_available = false;
  bool onnx_cuda_available = false;
  std::string learned_execution_provider = "cpu";
  double initial_focal_length = 0.0;
  std::string initial_focal_source;
  bool has_prior_focal_length = false;
  std::string camera_sharing_mode;
  std::size_t num_cameras = 0;
  std::size_t num_prior_focal_images = 0;
  std::size_t num_position_prior_images = 0;
  std::size_t num_position_prior_aligned_models = 0;
  double min_initial_focal_length = 0.0;
  double max_initial_focal_length = 0.0;
  int cpu_threads = 1;
  std::size_t num_extracted_features = 0;
  std::size_t num_raw_matches = 0;
  std::size_t num_verified_inliers = 0;
  std::size_t num_verified_inliers_before_consistency = 0;
  std::size_t num_pair_match_conflicts_removed = 0;
  std::size_t num_component_match_conflicts_removed = 0;
  std::size_t num_conflicted_match_components = 0;
  std::size_t num_conflicting_match_observations = 0;
  std::size_t num_sparse_models = 0;
  std::size_t num_unregistered_images = 0;
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
  double position_prior_refinement_seconds = 0.0;
  double position_prior_initial_mean_error = -1.0;
  double position_prior_final_mean_error = -1.0;
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
