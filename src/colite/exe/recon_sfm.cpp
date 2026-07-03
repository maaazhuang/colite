#include "colite/exe/recon_sfm.h"

#include <exception>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/core/version.h"
#include "colite/features/sift_extractor.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite recon_sfm " << colite::Version() << "\n"
      << "\n"
      << "Usage:\n"
      << "  " << executable << " --help\n"
      << "  " << executable << " --version\n"
      << "  " << executable
      << " --image_path <dir> --output_path <dir> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --focal_length <px>       Initial focal length. If omitted, use "
         "JPEG EXIF 35mm focal prior when available, otherwise "
         "1.2 * max(width, height).\n"
      << "  --max_num_features <n>   SIFT feature limit, default 4096.\n"
      << "  --feature_num_threads <n>\n"
         "                            CPU feature extraction threads, default -1 "
         "uses hardware concurrency.\n"
      << "  --feature_extractor <vlsift|cvsift|cudasift>\n"
         "                            SIFT extraction backend, default cvsift.\n"
      << "  --matcher <bf|flann>     Descriptor matcher backend, default bf.\n"
      << "  --nearby_match_window <n>\n"
         "                            Re-match neighboring image pairs within "
         "this file-order window, default 2; 0 disables it.\n"
      << "  --nearby_match_ratio <r> Relaxed Lowe ratio for neighboring pairs, "
         "default 0.9.\n"
      << "  --max_registration_trials_per_image <n>\n"
         "                            Maximum registration attempts per image, "
         "default 3.\n"
      << "  --max_initialization_trials <n>\n"
         "                            Maximum initial image-pair candidates to "
         "try, default 3.\n"
      << "  --registration_retry_error_multiplier <r>\n"
         "                            PnP reprojection threshold multiplier per "
         "failed retry, default 1.5.\n"
      << "  --registration_retry_max_error <px>\n"
         "                            Maximum relaxed PnP reprojection "
         "threshold, default 8.\n"
      << "  --ba_refine_focal_length <0|1>\n"
         "                            Refine focal length in local/global BA, "
         "default 0.\n"
      << "  --ba_refine_principal_point <0|1>\n"
         "                            Refine principal point in local/global BA, "
         "default 0.\n"
      << "  --ba_refine_extra_params <0|1>\n"
         "                            Refine distortion/extra camera params in "
         "local/global BA, default 0.\n"
      << "  --ba_refine_params <names>\n"
         "                            Comma-separated camera params to refine, "
         "e.g. fx,fy,k1,p2.\n";
}

std::string RequireValue(const std::vector<std::string>& args,
                         std::size_t* index,
                         std::string_view flag) {
  if (*index + 1 >= args.size()) {
    throw std::invalid_argument("Missing value for " + std::string(flag));
  }
  ++(*index);
  return args[*index];
}

bool ParseBoolFlagValue(const std::string& value, std::string_view flag) {
  if (value == "1" || value == "true" || value == "TRUE") {
    return true;
  }
  if (value == "0" || value == "false" || value == "FALSE") {
    return false;
  }
  throw std::invalid_argument("Expected 0/1 for " + std::string(flag));
}

std::set<std::string> ParseCommaSeparatedNames(const std::string& value,
                                               std::string_view flag) {
  std::set<std::string> names;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end =
        comma == std::string::npos ? value.size() : comma;
    const std::string name = value.substr(start, end - start);
    if (name.empty()) {
      throw std::invalid_argument("Empty name in " + std::string(flag));
    }
    names.insert(name);
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return names;
}

ReconSfmCliOptions ParseOptions(const std::vector<std::string>& args) {
  ReconSfmCliOptions options;
  if (args.size() <= 1) {
    options.show_help = true;
    return options;
  }

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
    } else if (arg == "--version") {
      options.show_version = true;
    } else if (arg == "--image_path") {
      options.image_path = RequireValue(args, &i, arg);
    } else if (arg == "--output_path") {
      options.output_path = RequireValue(args, &i, arg);
    } else if (arg == "--focal_length") {
      options.focal_length = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--max_num_features") {
      options.max_num_features = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--feature_num_threads") {
      options.feature_num_threads = std::stoi(RequireValue(args, &i, arg));
    } else if (arg == "--feature_extractor") {
      options.feature_backend =
          SiftBackendFromName(RequireValue(args, &i, arg));
    } else if (arg == "--nearby_match_window") {
      options.nearby_match_window =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--nearby_match_ratio") {
      options.nearby_match_ratio = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--max_registration_trials_per_image") {
      options.max_registration_trials_per_image =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--max_initialization_trials") {
      options.max_initialization_trials =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--registration_retry_error_multiplier") {
      options.registration_retry_reprojection_error_multiplier =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--registration_retry_max_error") {
      options.registration_retry_max_reprojection_error_px =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--ba_refine_focal_length") {
      options.ba_refine_focal_length =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--ba_refine_principal_point") {
      options.ba_refine_principal_point =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--ba_refine_extra_params") {
      options.ba_refine_extra_params =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--ba_refine_params") {
      options.ba_refine_params =
          ParseCommaSeparatedNames(RequireValue(args, &i, arg), arg);
    } else if (arg == "--matcher") {
      const std::string matcher = RequireValue(args, &i, arg);
      if (matcher == "bf") {
        options.use_flann = false;
      } else if (matcher == "flann") {
        options.use_flann = true;
      } else {
        throw std::invalid_argument("Unsupported matcher: " + matcher);
      }
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  return options;
}

void ValidateOptions(const ReconSfmCliOptions& options) {
  if (options.show_help || options.show_version) {
    return;
  }
  if (options.image_path.empty()) {
    throw std::invalid_argument("Missing required argument --image_path");
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path");
  }
  if (options.max_num_features <= 0) {
    throw std::invalid_argument("--max_num_features must be positive");
  }
  if (options.focal_length < 0.0) {
    throw std::invalid_argument("--focal_length must be non-negative");
  }
  if (options.nearby_match_ratio <= 0.0 || options.nearby_match_ratio > 1.0) {
    throw std::invalid_argument("--nearby_match_ratio must be in (0, 1]");
  }
  if (options.max_registration_trials_per_image == 0) {
    throw std::invalid_argument(
        "--max_registration_trials_per_image must be positive");
  }
  if (options.max_initialization_trials == 0) {
    throw std::invalid_argument("--max_initialization_trials must be positive");
  }
  if (options.registration_retry_reprojection_error_multiplier < 1.0) {
    throw std::invalid_argument(
        "--registration_retry_error_multiplier must be >= 1");
  }
  if (options.registration_retry_max_reprojection_error_px <= 0.0) {
    throw std::invalid_argument(
        "--registration_retry_max_error must be positive");
  }
}

controllers::ReconSfmControllerOptions ToControllerOptions(
    const ReconSfmCliOptions& options) {
  controllers::ReconSfmControllerOptions controller_options;
  controller_options.image_path = options.image_path;
  controller_options.output_path = options.output_path;
  controller_options.focal_length = options.focal_length;
  controller_options.max_num_features = options.max_num_features;
  controller_options.feature_num_threads = options.feature_num_threads;
  controller_options.feature_backend = options.feature_backend;
  controller_options.use_flann = options.use_flann;
  controller_options.nearby_match_window = options.nearby_match_window;
  controller_options.nearby_match_ratio = options.nearby_match_ratio;
  controller_options.max_registration_trials_per_image =
      options.max_registration_trials_per_image;
  controller_options.max_initialization_trials =
      options.max_initialization_trials;
  controller_options.registration_retry_reprojection_error_multiplier =
      options.registration_retry_reprojection_error_multiplier;
  controller_options.registration_retry_max_reprojection_error_px =
      options.registration_retry_max_reprojection_error_px;
  if (options.ba_refine_focal_length) {
    controller_options.ba_refine_camera_params =
        controller_options.ba_refine_camera_params |
        CameraParameterGroup::kFocalLength;
  }
  if (options.ba_refine_principal_point) {
    controller_options.ba_refine_camera_params =
        controller_options.ba_refine_camera_params |
        CameraParameterGroup::kPrincipalPoint;
  }
  if (options.ba_refine_extra_params) {
    controller_options.ba_refine_camera_params =
        controller_options.ba_refine_camera_params |
        CameraParameterGroup::kDistortion;
  }
  controller_options.ba_refine_camera_param_names = options.ba_refine_params;
  return controller_options;
}

}  // namespace

int RunReconSfm(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err) {
  const std::string executable = args.empty() ? "recon_sfm" : args.front();

  try {
    const ReconSfmCliOptions options = ParseOptions(args);
    if (options.show_help) {
      PrintHelp(executable, out);
      return 0;
    }
    if (options.show_version) {
      out << colite::Version() << "\n";
      return 0;
    }
    ValidateOptions(options);
    return controllers::ReconSfmController(ToControllerOptions(options)).Run(out);
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return 1;
  }
}

}  // namespace colite::exe
