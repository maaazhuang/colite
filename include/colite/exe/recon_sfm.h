#pragma once

#include <cstddef>
#include <iosfwd>
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
  SiftBackend feature_backend = SiftBackend::kCvSift;
  bool use_flann = false;
  std::size_t nearby_match_window = 2;
  double nearby_match_ratio = 0.9;
  std::size_t max_registration_trials_per_image = 3;
  bool ba_refine_focal_length = false;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = false;
};

int RunReconSfm(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err);

}  // namespace colite::exe
