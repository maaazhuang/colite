#include "colite/exe/recon_sfm.h"

#include <cmath>
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
         "JPEG EXIF 35mm/focal-plane prior when available, otherwise "
         "1.2 * max(width, height); EXIF values are marked as priors.\n"
      << "  --camera_model <name>     Camera model, default SIMPLE_RADIAL.\n"
      << "  --camera_params <csv>     Explicit parameters in model order.\n"
      << "  --camera_mode <auto|single|per_folder|per_image>\n"
         "                            Default auto groups complete EXIF camera "
         "identities; shared/individual are compatibility aliases.\n"
      << "  --default_focal_length_factor <f>\n"
         "                            Missing-EXIF focal factor, default 1.2.\n"
      << "  --max_num_features <n>   SIFT feature limit, default 4096.\n"
      << "  --feature_num_threads <n>\n"
         "                            CPU feature extraction threads, default -1 "
         "uses hardware concurrency.\n"
      << "  --feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>\n"
         "                            Default auto prefers CUDA OpenCV and "
         "falls back to CPU OpenCV.\n"
      << "  --matcher <bf|flann|cuda> Descriptor matcher backend, default flann.\n"
      << "  --nearby_match_window <n>\n"
         "                            Relax and re-match neighboring image pairs "
         "within this file-order window, default 0 disables it.\n"
      << "  --nearby_match_ratio <r> Relaxed Lowe ratio for neighboring pairs, "
         "default 0.8.\n"
      << "  --guided_match_ratio <r> Lowe ratio for geometry-guided matching, "
         "default 0.95; use 0.8 for COLMAP High same-descriptor studies.\n"
      << "  --max_registration_trials_per_image <n>\n"
         "                            Maximum registration attempts per image, "
         "default 3.\n"
      << "  --max_initialization_trials <n>\n"
         "                            Maximum initial image-pair candidates to "
         "try, default 3.\n"
      << "  --input_model_path <dir> Continue from an existing COLMAP model.\n"
      << "  --input_format <auto|txt|bin>\n"
         "                            Input model format, default auto.\n"
      << "  --output_format <txt|bin|both>\n"
         "                            Sparse model format, default both.\n"
      << "  --resume                  Reuse valid feature/frontend/sparse stages.\n"
      << "  --force                   Recompute all SfM stages.\n"
      << "  --multiple_models <0|1>  Reconstruct remaining components, default 1.\n"
      << "  --merge_models <0|1>     Merge Sim(3)-alignable models, default 1.\n"
      << "  --max_num_models <n>     Maximum sparse models, default 50.\n"
      << "  --min_model_size <n>     Minimum registered images per model, default 2.\n"
      << "  --registration_retry_error_multiplier <r>\n"
         "                            PnP reprojection threshold multiplier per "
         "failed retry, default 1.5.\n"
      << "  --registration_retry_max_error <px>\n"
         "                            Maximum relaxed PnP reprojection "
         "threshold, default 8.\n"
      << "  --ba_refine_focal_length <0|1>\n"
         "                            Refine focal length in local/global BA, "
         "default 1.\n"
      << "  --ba_refine_principal_point <0|1>\n"
         "                            Refine principal point in local/global BA, "
         "default 0.\n"
      << "  --ba_refine_extra_params <0|1>\n"
         "                            Refine distortion/extra camera params in "
         "local/global BA, default 1.\n"
      << "  --ba_refine_params <names>\n"
         "                            Comma-separated camera params to refine, "
         "e.g. fx,fy,k1,p2.\n"
      << "  --use_position_priors <0|1>\n"
         "                            Read EXIF GPS, align the model and add "
         "position residuals to final BA; default 0.\n"
      << "  --position_prior_stddev <m>\n"
         "                            Position prior standard deviation, default 5.\n"
      << "  --position_prior_loss_scale <m>\n"
         "                            Huber loss scale, default 2.\n"
      << "  --position_prior_alignment_max_error <m>\n"
         "                            Alignment inlier threshold, default 20.\n";
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
    } else if (arg == "--resume") {
      options.resume = true;
    } else if (arg == "--force") {
      options.force = true;
    } else if (arg == "--image_path") {
      options.image_path = RequireValue(args, &i, arg);
    } else if (arg == "--output_path") {
      options.output_path = RequireValue(args, &i, arg);
    } else if (arg == "--input_model_path") {
      options.input_model_path = RequireValue(args, &i, arg);
    } else if (arg == "--input_format") {
      options.input_model_format =
          ColmapModelFormatFromString(RequireValue(args, &i, arg));
      if (options.input_model_format == ColmapModelFormat::kTextAndBinary) {
        throw std::invalid_argument("--input_format does not accept both");
      }
    } else if (arg == "--output_format") {
      options.output_model_format =
          ColmapModelFormatFromString(RequireValue(args, &i, arg));
      if (options.output_model_format == ColmapModelFormat::kAuto) {
        throw std::invalid_argument("--output_format does not accept auto");
      }
    } else if (arg == "--focal_length") {
      options.focal_length = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--camera_model") {
      options.camera_model_id =
          CameraModelNameToId(RequireValue(args, &i, arg));
      if (!ExistsCameraModel(options.camera_model_id)) {
        throw std::invalid_argument("Unsupported camera model");
      }
    } else if (arg == "--camera_params") {
      options.camera_params =
          controllers::ParseCameraParams(RequireValue(args, &i, arg));
    } else if (arg == "--camera_mode") {
      options.camera_sharing_mode =
          controllers::CameraSharingModeFromName(
              RequireValue(args, &i, arg));
    } else if (arg == "--default_focal_length_factor") {
      options.default_focal_length_factor =
          std::stod(RequireValue(args, &i, arg));
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
    } else if (arg == "--guided_match_ratio") {
      options.guided_match_ratio = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--max_registration_trials_per_image") {
      options.max_registration_trials_per_image =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--max_initialization_trials") {
      options.max_initialization_trials =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--multiple_models") {
      options.multiple_models =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--merge_models") {
      options.merge_models =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--max_num_models") {
      options.max_num_models =
          static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
    } else if (arg == "--min_model_size") {
      options.min_num_images_per_model =
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
    } else if (arg == "--use_position_priors") {
      options.use_position_priors =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--position_prior_stddev") {
      options.position_prior_stddev = std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--position_prior_loss_scale") {
      options.position_prior_loss_scale =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--position_prior_alignment_max_error") {
      options.position_prior_alignment_max_error =
          std::stod(RequireValue(args, &i, arg));
    } else if (arg == "--matcher") {
      const std::string matcher = RequireValue(args, &i, arg);
      if (matcher == "bf") {
        options.matcher_backend = DescriptorMatcherBackend::kBruteForce;
      } else if (matcher == "flann") {
        options.matcher_backend = DescriptorMatcherBackend::kFlann;
      } else if (matcher == "cuda") {
        options.matcher_backend = DescriptorMatcherBackend::kCudaBruteForce;
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
  if (options.resume && options.force) {
    throw std::invalid_argument("--resume and --force are mutually exclusive");
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
  if (!ExistsCameraModel(options.camera_model_id)) {
    throw std::invalid_argument("--camera_model is unsupported");
  }
  if (!std::isfinite(options.default_focal_length_factor) ||
      options.default_focal_length_factor <= 0.0) {
    throw std::invalid_argument(
        "--default_focal_length_factor must be positive");
  }
  if (!options.camera_params.empty() &&
      options.camera_params.size() !=
          GetCameraModelSpec(options.camera_model_id).param_names.size()) {
    throw std::invalid_argument(
        "--camera_params count does not match --camera_model");
  }
  if (options.nearby_match_ratio <= 0.0 || options.nearby_match_ratio > 1.0) {
    throw std::invalid_argument("--nearby_match_ratio must be in (0, 1]");
  }
  if (options.guided_match_ratio <= 0.0 ||
      options.guided_match_ratio > 1.0) {
    throw std::invalid_argument("--guided_match_ratio must be in (0, 1]");
  }
  if (options.max_registration_trials_per_image == 0) {
    throw std::invalid_argument(
        "--max_registration_trials_per_image must be positive");
  }
  if (options.max_initialization_trials == 0) {
    throw std::invalid_argument("--max_initialization_trials must be positive");
  }
  if (options.max_num_models == 0 ||
      options.min_num_images_per_model < 2) {
    throw std::invalid_argument(
        "--max_num_models must be positive and --min_model_size >= 2");
  }
  if (options.registration_retry_reprojection_error_multiplier < 1.0) {
    throw std::invalid_argument(
        "--registration_retry_error_multiplier must be >= 1");
  }
  if (options.registration_retry_max_reprojection_error_px <= 0.0) {
    throw std::invalid_argument(
        "--registration_retry_max_error must be positive");
  }
  if (!std::isfinite(options.position_prior_stddev) ||
      options.position_prior_stddev <= 0.0 ||
      !std::isfinite(options.position_prior_loss_scale) ||
      options.position_prior_loss_scale < 0.0 ||
      !std::isfinite(options.position_prior_alignment_max_error) ||
      options.position_prior_alignment_max_error <= 0.0) {
    throw std::invalid_argument("Invalid position prior configuration");
  }
}

controllers::ReconSfmControllerOptions ToControllerOptions(
    const ReconSfmCliOptions& options) {
  controllers::ReconSfmControllerOptions controller_options;
  controller_options.image_path = options.image_path;
  controller_options.output_path = options.output_path;
  controller_options.input_model_path = options.input_model_path;
  controller_options.input_model_format = options.input_model_format;
  controller_options.output_model_format = options.output_model_format;
  controller_options.resume = options.resume;
  controller_options.force = options.force;
  controller_options.focal_length = options.focal_length;
  controller_options.camera_sharing_mode = options.camera_sharing_mode;
  controller_options.camera_model_id = options.camera_model_id;
  controller_options.camera_params = options.camera_params;
  controller_options.default_focal_length_factor =
      options.default_focal_length_factor;
  controller_options.max_num_features = options.max_num_features;
  controller_options.feature_num_threads = options.feature_num_threads;
  controller_options.feature_backend = options.feature_backend;
  controller_options.matcher_backend = options.matcher_backend;
  controller_options.nearby_match_window = options.nearby_match_window;
  controller_options.nearby_match_ratio = options.nearby_match_ratio;
  controller_options.guided_match_ratio = options.guided_match_ratio;
  controller_options.max_registration_trials_per_image =
      options.max_registration_trials_per_image;
  controller_options.max_initialization_trials =
      options.max_initialization_trials;
  controller_options.multiple_models = options.multiple_models;
  controller_options.merge_models = options.merge_models;
  controller_options.max_num_models = options.max_num_models;
  controller_options.min_num_images_per_model =
      options.min_num_images_per_model;
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
  controller_options.use_position_priors = options.use_position_priors;
  controller_options.position_prior_stddev = options.position_prior_stddev;
  controller_options.position_prior_loss_scale =
      options.position_prior_loss_scale;
  controller_options.position_prior_alignment_max_error =
      options.position_prior_alignment_max_error;
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
