#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "colite/controllers/sfm_options.h"
#include "colite/controllers/sfm_frontend_options.h"
#include "colite/controllers/sfm_camera_setup.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite::exe {

enum class SfmOptionScope { kFeatures, kMatching, kMapper, kPipeline, kModel };

inline bool ApplySfmFeatureExecutionOption(const std::string& name, const std::string& value,
    controllers::SfmFeatureExecutionOptions* options) {
  auto positive_int = [&] { const int n = controllers::SfmInt(value);
    if (n <= 0) throw std::invalid_argument(name + " must be positive"); return n; };
  auto& sift = options->sift;
  if (name.rfind("--SiftExtraction.", 0) == 0 &&
      name != "--SiftExtraction.normalization" && name != "--SiftExtraction.max_num_features")
    options->custom_sift_geometry = true;
  if (name == "--FeatureExtraction.use_gpu") options->use_gpu = controllers::SfmBool(value);
  else if (name == "--FeatureExtraction.gpu_index") {
    options->gpu_index = controllers::SfmInt(value);
    if (options->gpu_index < -1) throw std::invalid_argument(name + " must be >= -1 (single device)");
  } else if (name == "--FeatureExtraction.max_image_size") options->max_image_size = controllers::SfmInt(value);
  else if (name == "--SiftExtraction.first_octave") {
    const int n = controllers::SfmInt(value);
    if (n != -1 && n != 0) throw std::invalid_argument(name + " currently supports -1 or 0");
    sift.upsample_first_octave = n == -1;
  } else if (name == "--SiftExtraction.num_octaves") sift.vl_num_octaves = positive_int();
  else if (name == "--SiftExtraction.octave_resolution") sift.num_octave_layers = positive_int();
  else if (name == "--SiftExtraction.peak_threshold" || name == "--SiftExtraction.edge_threshold") {
    const double n = controllers::SfmDouble(value);
    if (n <= 0) throw std::invalid_argument(name + " must be positive");
    if (name == "--SiftExtraction.peak_threshold") sift.peak_threshold = n;
    else sift.edge_threshold = n;
  } else if (name == "--SiftExtraction.max_num_orientations") {
    sift.max_num_orientations = positive_int();
    if (sift.max_num_orientations > 4) throw std::invalid_argument(name + " must be <= 4");
  } else if (name == "--SiftExtraction.normalization") {
    if (value != "L1_ROOT" && value != "L2") throw std::invalid_argument(name + " expects L1_ROOT or L2");
    sift.use_root_sift = value == "L1_ROOT";
  } else return false;
  return true;
}

inline bool ApplySfmMatchingExecutionOption(const std::string& name, const std::string& value,
    controllers::SfmMatchingExecutionOptions* options) {
  if (name == "--FeatureMatching.use_gpu") options->use_gpu = controllers::SfmBool(value);
  else if (name == "--FeatureMatching.gpu_index") {
    options->gpu_index = controllers::SfmInt(value);
    if (options->gpu_index < -1) throw std::invalid_argument(name + " must be >= -1 (single device)");
  } else if (name == "--FeatureMatching.num_threads") {
    options->num_threads = controllers::SfmInt(value);
    if (options->num_threads == 0 || options->num_threads < -1) throw std::invalid_argument(name + " must be -1 or positive");
  } else return false;
  return true;
}

// This is the public CLI contract. There is no legacy-name translation layer.
inline std::vector<std::string> NormalizeSfmArguments(
    const std::vector<std::string>& input, SfmOptionScope scope) {
  if (input.empty()) return input;
  std::vector<std::string> expanded{input.front()};
  for (std::size_t i = 1; i < input.size(); ++i) {
    const auto equal = input[i].find('=');
    if (input[i].rfind("--", 0) == 0 && equal != std::string::npos) {
      expanded.push_back(input[i].substr(0, equal));
      expanded.push_back(input[i].substr(equal + 1));
    } else expanded.push_back(input[i]);
  }
  std::set<std::string> seen;
  std::vector<std::string> result{input.front()};
  std::size_t camera_modes = 0;
  std::string feature_type = "sift";
  std::string matcher_type = "SIFT_BRUTEFORCE";
  for (std::size_t i = 1; i < expanded.size(); ++i) {
    std::string name = expanded[i];
    if (name == "-h") name = "--help";
    if (name.rfind("--", 0) != 0) throw std::invalid_argument("Unexpected argument: " + name);
    if (!seen.insert(name).second) throw std::invalid_argument("Duplicate option: " + name);
    const bool no_value = name == "--help" || name == "--version" ||
                          name == "--resume" || name == "--force" || name == "--dry_run";
    std::string value;
    if (!no_value) {
      if (i + 1 == expanded.size() || expanded[i + 1].rfind("--", 0) == 0) {
        throw std::invalid_argument("Missing value for " + name);
      }
      value = expanded[++i];
    }
    if (name == "--ImageReader.single_camera" ||
        name == "--ImageReader.single_camera_per_folder" ||
        name == "--ImageReader.single_camera_per_image") {
      camera_modes += controllers::SfmBool(value) ? 1 : 0;
      if (camera_modes > 1) throw std::invalid_argument("Conflicting ImageReader camera sharing options");
    }
    if (name == "--FeatureExtraction.type") {
      std::transform(value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      feature_type = value;
    }
    if (name == "--FeatureMatching.type") {
      std::transform(value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      matcher_type = value;
    }
    if (name == "--SpatialMatching.max_distance" && controllers::SfmDouble(value) <= 0) {
      throw std::invalid_argument(name + " must be positive; min_num_neighbors is not implemented");
    }
    result.push_back(name);
    if (!no_value) result.push_back(value);
  }
  if (scope == SfmOptionScope::kFeatures || scope == SfmOptionScope::kPipeline) {
    for (const auto& name : seen) {
      const bool sift_option = name.rfind("--SiftExtraction.", 0) == 0;
      const bool aliked_option = name.rfind("--AlikedExtraction.", 0) == 0;
      if ((sift_option && feature_type != "sift") || (aliked_option && feature_type == "sift")) {
        throw std::invalid_argument(name + " does not apply to FeatureExtraction.type " + feature_type);
      }
    }
    if ((seen.count("--AlikedExtraction.n16rot_model_path") && feature_type != "aliked_n16rot") ||
        (seen.count("--AlikedExtraction.n32_model_path") && feature_type != "aliked_n32")) {
      throw std::invalid_argument("ALIKED model path does not match FeatureExtraction.type");
    }
  }
  if (scope == SfmOptionScope::kMatching || scope == SfmOptionScope::kPipeline) {
    for (const auto& name : seen) {
      const bool sift = name.rfind("--SiftMatching.", 0) == 0;
      const bool aliked = name.rfind("--AlikedMatching.", 0) == 0;
      const bool lightglue = name.find(".lightglue_") != std::string::npos;
      if ((sift && matcher_type.rfind("SIFT_", 0) != 0) ||
          (aliked && matcher_type.rfind("ALIKED_", 0) != 0) ||
          ((sift || aliked) && lightglue && matcher_type.find("LIGHTGLUE") == std::string::npos) ||
          ((sift || aliked) && !lightglue && matcher_type.find("BRUTEFORCE") == std::string::npos)) {
        throw std::invalid_argument(name + " does not apply to FeatureMatching.type " + matcher_type);
      }
    }
  }
  return result;
}

inline bool ApplySfmCameraSharingFlag(const std::string& name,
                                      const std::string& value,
                                      controllers::CameraSharingMode* mode) {
  using Mode = controllers::CameraSharingMode;
  Mode selected;
  if (name == "--ImageReader.single_camera") selected = Mode::kSingle;
  else if (name == "--ImageReader.single_camera_per_folder") selected = Mode::kPerFolder;
  else if (name == "--ImageReader.single_camera_per_image") selected = Mode::kPerImage;
  else return false;
  if (controllers::SfmBool(value)) *mode = selected;
  return true;
}

inline FeatureMatcherType ParseSfmMatcherType(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (value == "SIFT_BRUTEFORCE") return FeatureMatcherType::kSiftBruteForce;
  if (value == "SIFT_LIGHTGLUE") return FeatureMatcherType::kSiftLightGlue;
  if (value == "ALIKED_BRUTEFORCE") return FeatureMatcherType::kAlikedBruteForce;
  if (value == "ALIKED_LIGHTGLUE") return FeatureMatcherType::kAlikedLightGlue;
  throw std::invalid_argument("Unsupported FeatureMatching.type: " + value);
}

inline void PrintSfmOptionsHelp(std::ostream& out, SfmOptionScope scope) {
  out << "\nCOLMAP 4.0 options (no legacy aliases):\n";
  if (scope == SfmOptionScope::kFeatures || scope == SfmOptionScope::kPipeline) {
    out << "  --FeatureExtraction.use_gpu <0|1> Default available GPU.\n"
        << "  --FeatureExtraction.gpu_index <index> Single GPU; -1 selects device 0.\n"
        << "  --FeatureExtraction.max_image_size <px> Non-positive: SIFT 3200 / ALIKED 1600.\n"
        << "  --SiftExtraction.first_octave <-1|0> Default -1.\n"
        << "  --SiftExtraction.num_octaves <n> Default 4.\n"
        << "  --SiftExtraction.octave_resolution <n> Default 3.\n"
        << "  --SiftExtraction.peak_threshold <value> Default 0.02/3.\n"
        << "  --SiftExtraction.edge_threshold <value> Default 10.\n"
        << "  --SiftExtraction.max_num_orientations <1..4> Default 2.\n"
        << "  --SiftExtraction.normalization L1_ROOT|L2\n"
        << "  --ImageReader.single_camera <0|1>\n"
        << "  --ImageReader.single_camera_per_folder <0|1>\n"
        << "  --ImageReader.single_camera_per_image <0|1>\n"
        << "  --AlikedExtraction.max_num_features <n>\n"
        << "  --AlikedExtraction.n32_model_path <path>\n";
  }
  if (scope == SfmOptionScope::kMatching || scope == SfmOptionScope::kPipeline) {
    out << "  --FeatureMatching.use_gpu <0|1> Default available GPU for SIFT.\n"
        << "  --FeatureMatching.gpu_index <index> Single GPU; -1 selects device 0.\n"
        << "  --FeatureMatching.num_threads <-1|positive> Matching and geometry thread budget.\n"
        << "  --FeatureMatching.type SIFT_BRUTEFORCE|SIFT_LIGHTGLUE|ALIKED_BRUTEFORCE|ALIKED_LIGHTGLUE\n"
        << "  --AlikedMatching.brute_force_max_ratio <value>\n"
        << "  --AlikedMatching.brute_force_cross_check <0|1>\n"
        << "  --AlikedMatching.lightglue_min_score <value>\n"
        << "  --AlikedMatching.lightglue_model_path <path>\n"
        << "  --TwoViewGeometry.max_error <px>\n"
        << "  --TwoViewGeometry.min_num_inliers <n>\n"
        << "  --TwoViewGeometry.confidence <probability>\n"
        << "  --TwoViewGeometry.max_num_trials <n>\n";
  }
  if (scope == SfmOptionScope::kMatching) {
    out << "  --SiftMatching.max_distance <value> Default 0.7.\n"
        << "  --SiftMatching.cross_check <0|1> Default 1.\n";
  }
  if (scope == SfmOptionScope::kModel) {
    out << "  --BundleAdjustment.refine_focal_length <0|1>\n"
        << "  --BundleAdjustment.refine_principal_point <0|1>\n"
        << "  --BundleAdjustment.refine_extra_params <0|1>\n";
  }
  if (scope == SfmOptionScope::kMapper || scope == SfmOptionScope::kPipeline) {
    for (const char* name : {"ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params", "max_reg_trials", "init_num_trials", "structure_less_registration_fallback",
         "init_min_num_inliers", "init_max_error", "init_min_tri_angle", "abs_pose_max_error",
         "abs_pose_min_num_inliers", "abs_pose_min_inlier_ratio", "filter_max_reproj_error",
         "filter_min_tri_angle", "ba_local_num_images", "ba_local_max_num_iterations",
         "ba_global_max_num_iterations", "tri_max_transitivity", "tri_create_max_angle_error",
         "tri_continue_max_angle_error", "tri_merge_max_reproj_error", "tri_complete_max_reproj_error",
         "tri_complete_max_transitivity", "tri_min_angle", "tri_ignore_two_view_tracks"}) {
      if (scope == SfmOptionScope::kPipeline &&
          (std::string(name).rfind("ba_refine_", 0) == 0 ||
           std::string(name) == "max_reg_trials" || std::string(name) == "init_num_trials" ||
           std::string(name) == "structure_less_registration_fallback")) continue;
      out << "  --Mapper." << name << " <value>\n";
    }
  }
}

}  // namespace colite::exe
