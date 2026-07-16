#pragma once

#include <cstddef>
#include <iosfwd>
#include <set>
#include <string>
#include <vector>

#include "colite/features/sift_extractor.h"

namespace colite::exe {

struct ReconSfmCliOptions {
  bool show_help = false;
  bool show_version = false;
  std::string image_path;
  std::string output_path;
  double focal_length = 0.0;
  int max_num_features = 4096;
  int feature_num_threads = -1;
  SiftBackend feature_backend = SiftBackend::kAuto;
  bool use_flann = true;
  std::size_t nearby_match_window = 0;
  double nearby_match_ratio = 0.8;
  std::size_t max_registration_trials_per_image = 3;
  std::size_t max_initialization_trials = 3;
  double registration_retry_reprojection_error_multiplier = 1.5;
  double registration_retry_max_reprojection_error_px = 8.0;
  bool ba_refine_focal_length = false;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = false;
  std::set<std::string> ba_refine_params;
};

int RunReconSfm(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err);

}  // namespace colite::exe
