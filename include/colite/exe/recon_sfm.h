#pragma once

#include <cstddef>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/controllers/sfm_camera_setup.h"
#include "colite/core/feature.h"
#include "colite/features/sift_extractor.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite::exe {

struct ReconSfmCliOptions {
  bool show_help = false;
  bool show_version = false;
  std::string image_path;
  std::string output_path;
  std::string input_model_path;
  ColmapModelFormat input_model_format = ColmapModelFormat::kAuto;
  ColmapModelFormat output_model_format =
      ColmapModelFormat::kTextAndBinary;
  bool resume = false;
  bool force = false;
  double focal_length = 0.0;
  controllers::CameraSharingMode camera_sharing_mode =
      controllers::CameraSharingMode::kAuto;
  CameraModelId camera_model_id = CameraModelId::kSimpleRadial;
  std::vector<double> camera_params;
  double default_focal_length_factor = 1.2;
  int max_num_features = 4096;
  int feature_num_threads = -1;
  FeatureExtractorType feature_extractor_type = FeatureExtractorType::kSift;
  SiftBackend feature_backend = SiftBackend::kAuto;
  std::string aliked_model_path;
  double aliked_min_score = 0.2;
  DescriptorMatcherBackend matcher_backend = DescriptorMatcherBackend::kFlann;
  FeatureMatcherType matcher_type = FeatureMatcherType::kSiftBruteForce;
  std::string learned_matcher_model_path;
  double min_cosine_similarity = 0.85;
  double lightglue_min_score = 0.1;
  int learned_matcher_num_threads = 1;
  controllers::SfmMapperType mapper_type =
      controllers::SfmMapperType::kIncremental;
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
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = true;
  std::set<std::string> ba_refine_params;
  bool use_position_priors = false;
  double position_prior_stddev = 5.0;
  double position_prior_loss_scale = 2.0;
  double position_prior_alignment_max_error = 20.0;
};

int RunReconSfm(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err);

}  // namespace colite::exe
