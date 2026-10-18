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
#include "colite/exe/sfm_options.h"
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
      << "  --ImageReader.camera_model <name>     Camera model, default SIMPLE_RADIAL.\n"
      << "  --ImageReader.camera_params <csv>     Explicit parameters in model order.\n"
      << "  --ImageReader.default_focal_length_factor <f>\n"
         "                            Missing-EXIF focal factor, default 1.2.\n"
      << "  --SiftExtraction.max_num_features <n>   SIFT feature limit, default 8192.\n"
      << "  --FeatureExtraction.num_threads <n>\n"
         "                            CPU feature extraction threads, default -1 "
         "uses hardware concurrency.\n"
      << "  --Colite.feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>\n"
         "                            Default auto prefers CUDA VLFeat and "
         "falls back to CPU VLFeat.\n"
      << "  --FeatureExtraction.type <sift|aliked_n16rot|aliked_n32>\n"
         "                            Feature family, default sift.\n"
      << "  --AlikedExtraction.n16rot_model_path <file> Required for ALIKED extraction.\n"
      << "  --AlikedExtraction.min_score <v>    ALIKED score threshold, default 0.2.\n"
      << "  --SiftMatching.lightglue_model_path <file> Required for LightGlue.\n"
      << "  --AlikedMatching.brute_force_min_cossim <v> ALIKED BF threshold, default 0.85.\n"
      << "  --SiftMatching.lightglue_min_score <v> LightGlue threshold, default 0.1.\n"
      << "  --Colite.mapper <incremental|global|hierarchical> SfM mapper, default incremental.\n"
      << "  --Colite.hierarchical_target_cluster_size <n> Core images per cluster, default 100.\n"
      << "  --Colite.hierarchical_max_cluster_size <n> Core plus overlap, default 150.\n"
      << "  --Colite.hierarchical_overlap <n> Boundary images per cluster, default 10.\n"
      << "  --Colite.learned_cuda_memory_limit_mb <n> CUDA arena limit, default 8192.\n"
      << "  --Colite.nearby_match_window <n>\n"
         "                            Relax and re-match neighboring image pairs "
         "within this file-order window, default 0 disables it.\n"
      << "  --Colite.hierarchical_num_workers <n> Local mapper workers; 0 auto, default 0.\n"
      << "  --Colite.nearby_match_ratio <r> Relaxed Lowe ratio for neighboring pairs, "
         "default 0.8.\n"
      << "  --Colite.guided_match_ratio <r> Lowe ratio for geometry-guided matching, "
         "default 0.8; use 0.8 for COLMAP High same-descriptor studies.\n"
      << "  --SiftMatching.max_ratio <r> Initial Lowe ratio, default 0.8.\n"
      << "  --SiftMatching.max_distance <d> Descriptor distance, default 0.7.\n"
      << "  --SiftMatching.cross_check <0|1> Mutual descriptor check, default 1.\n"
      << "  --FeatureMatching.guided_matching <0|1> Guided re-search, default 0.\n"
      << "  --Mapper.max_reg_trials <n>\n"
         "                            Maximum registration attempts per image, "
         "default 3.\n"
      << "  --Mapper.init_num_trials <n>\n"
         "                            Maximum initial image-pair candidates to "
         "try, default 3.\n"
      << "  --Mapper.structure_less_registration_fallback <0|1>\n"
         "                            Use verified multi-neighbor relative-pose "
         "fallback after PnP failure, default 0.\n"
      << "  --input_path <dir> Continue from an existing COLMAP model.\n"
      << "  --Colite.input_format <auto|txt|bin>\n"
         "                            Input model format, default auto.\n"
      << "  --Colite.output_format <txt|bin|both>\n"
         "                            Sparse model format, default both.\n"
      << "  --resume                  Reuse valid feature/frontend/sparse stages.\n"
      << "  --force                   Recompute all SfM stages.\n"
      << "  --Mapper.multiple_models <0|1>  Reconstruct remaining components, default 1.\n"
      << "  --Colite.merge_models <0|1>     Merge Sim(3)-alignable models, default 1.\n"
      << "  --Mapper.max_num_models <n>     Maximum sparse models, default 50.\n"
      << "  --Mapper.min_model_size <n>     Minimum registered images per model, default 2.\n"
      << "  --Colite.registration_retry_error_multiplier <r>\n"
         "                            PnP reprojection threshold multiplier per "
         "failed retry, default 1.5.\n"
      << "  --Colite.registration_retry_max_error <px>\n"
         "                            Maximum relaxed PnP reprojection "
         "threshold, default 8.\n"
      << "  --Mapper.ba_refine_focal_length <0|1>\n"
         "                            Refine focal length in local/global BA, "
         "default 1.\n"
      << "  --Mapper.ba_refine_principal_point <0|1>\n"
         "                            Refine principal point in local/global BA, "
         "default 0.\n"
      << "  --Mapper.ba_refine_extra_params <0|1>\n"
         "                            Refine distortion/extra camera params in "
         "local/global BA, default 1.\n"
      << "  --Colite.use_position_priors <0|1>\n"
         "                            Read EXIF GPS, align the model and add "
         "position residuals to final BA; default 0.\n"
      << "  --Colite.position_prior_stddev <m>\n"
         "                            Position prior standard deviation, default 5.\n"
      << "  --Colite.position_prior_loss_scale <m>\n"
         "                            Huber loss scale, default 2.\n"
      << "  --Colite.position_prior_alignment_max_error <m>\n"
         "                            Alignment inlier threshold, default 20.\n";
  PrintSfmOptionsHelp(out, SfmOptionScope::kPipeline);
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

ReconSfmCliOptions ParseOptions(const std::vector<std::string>& raw_args) {
  const auto args = NormalizeSfmArguments(raw_args, SfmOptionScope::kPipeline);
  ReconSfmCliOptions options;
  bool max_num_features_supplied = false;
  if (args.size() <= 1) {
    options.show_help = true;
    return options;
  }

  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg.rfind("--", 0) == 0 && i + 1 < args.size() &&
        (ApplySfmFeatureExecutionOption(arg, args[i + 1], &options.feature_execution) ||
         ApplySfmMatchingExecutionOption(arg, args[i + 1], &options.matching_execution))) {
      ++i;
    } else if (arg == "--help" || arg == "-h") {
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
    } else if (arg == "--input_path") {
      options.input_model_path = RequireValue(args, &i, arg);
    } else if (arg == "--Colite.input_format") {
      options.input_model_format =
          ColmapModelFormatFromString(RequireValue(args, &i, arg));
      if (options.input_model_format == ColmapModelFormat::kTextAndBinary) {
        throw std::invalid_argument("--Colite.input_format does not accept both");
      }
    } else if (arg == "--Colite.output_format") {
      options.output_model_format =
          ColmapModelFormatFromString(RequireValue(args, &i, arg));
      if (options.output_model_format == ColmapModelFormat::kAuto) {
        throw std::invalid_argument("--Colite.output_format does not accept auto");
      }
    } else if (arg == "--ImageReader.camera_model") {
      options.camera_model_id =
          CameraModelNameToId(RequireValue(args, &i, arg));
      if (!ExistsCameraModel(options.camera_model_id)) {
        throw std::invalid_argument("Unsupported camera model");
      }
    } else if (arg == "--ImageReader.camera_params") {
      options.camera_params =
          controllers::ParseCameraParams(RequireValue(args, &i, arg));
    } else if (arg == "--ImageReader.single_camera" ||
                 arg == "--ImageReader.single_camera_per_folder" ||
                 arg == "--ImageReader.single_camera_per_image") {
      ApplySfmCameraSharingFlag(arg, RequireValue(args, &i, arg), &options.camera_sharing_mode);
    } else if (arg == "--ImageReader.default_focal_length_factor") {
      options.default_focal_length_factor =
          controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--SiftExtraction.max_num_features" || arg == "--AlikedExtraction.max_num_features") {
      options.max_num_features = controllers::SfmInt(RequireValue(args, &i, arg));
      max_num_features_supplied = true;
    } else if (arg == "--FeatureExtraction.num_threads") {
      options.feature_num_threads = controllers::SfmInt(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.feature_extractor") {
      options.feature_backend =
          SiftBackendFromName(RequireValue(args, &i, arg));
    } else if (arg == "--FeatureExtraction.type") {
      const std::string value = RequireValue(args, &i, arg);
      if (value == "sift") {
        options.feature_extractor_type = FeatureExtractorType::kSift;
      } else if (value == "aliked_n16rot") {
        options.feature_extractor_type = FeatureExtractorType::kAlikedN16Rot;
      } else if (value == "aliked_n32") {
        options.feature_extractor_type = FeatureExtractorType::kAlikedN32;
      } else {
        throw std::invalid_argument("Unsupported feature type: " + value);
      }
    } else if (arg == "--AlikedExtraction.n16rot_model_path" || arg == "--AlikedExtraction.n32_model_path") {
      options.aliked_model_path = RequireValue(args, &i, arg);
    } else if (arg == "--AlikedExtraction.min_score") {
      options.aliked_min_score = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.nearby_match_window") {
      options.nearby_match_window =
          static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.nearby_match_ratio") {
      options.nearby_match_ratio = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.guided_match_ratio") {
      options.guided_match_ratio = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Mapper.max_reg_trials") {
      options.max_registration_trials_per_image =
          static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Mapper.init_num_trials") {
      options.max_initialization_trials =
          static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Mapper.structure_less_registration_fallback") {
      options.use_structureless_registration =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Mapper.multiple_models") {
      options.multiple_models =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Colite.merge_models") {
      options.merge_models =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Mapper.max_num_models") {
      options.max_num_models =
          static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Mapper.min_model_size") {
      options.min_num_images_per_model =
          static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.registration_retry_error_multiplier") {
      options.registration_retry_reprojection_error_multiplier =
          controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.registration_retry_max_error") {
      options.registration_retry_max_reprojection_error_px =
          controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Mapper.ba_refine_focal_length") {
      options.ba_refine_focal_length =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Mapper.ba_refine_principal_point") {
      options.ba_refine_principal_point =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Mapper.ba_refine_extra_params") {
      options.ba_refine_extra_params =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Colite.use_position_priors") {
      options.use_position_priors =
          ParseBoolFlagValue(RequireValue(args, &i, arg), arg);
    } else if (arg == "--Colite.position_prior_stddev") {
      options.position_prior_stddev = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.position_prior_loss_scale") {
      options.position_prior_loss_scale =
          controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.position_prior_alignment_max_error") {
      options.position_prior_alignment_max_error =
          controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--SiftMatching.lightglue_model_path" || arg == "--AlikedMatching.lightglue_model_path") {
      options.learned_matcher_model_path = RequireValue(args, &i, arg);
    } else if (arg == "--AlikedMatching.brute_force_min_cossim") {
      options.min_cosine_similarity = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--SiftMatching.lightglue_min_score" || arg == "--AlikedMatching.lightglue_min_score") {
      options.lightglue_min_score = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--Colite.learned_cuda_memory_limit_mb") {
      options.learned_cuda_memory_limit_mb = static_cast<std::size_t>(
          controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.mapper") {
      const std::string mapper = RequireValue(args, &i, arg);
      if (mapper == "incremental") {
        options.mapper_type = controllers::SfmMapperType::kIncremental;
      } else if (mapper == "global") {
        options.mapper_type = controllers::SfmMapperType::kGlobal;
      } else if (mapper == "hierarchical") {
        options.mapper_type = controllers::SfmMapperType::kHierarchical;
      } else {
        throw std::invalid_argument("Unsupported mapper: " + mapper);
      }
    } else if (arg == "--Colite.hierarchical_target_cluster_size") {
      options.hierarchical_target_cluster_size = static_cast<std::size_t>(
          controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.hierarchical_max_cluster_size") {
      options.hierarchical_max_cluster_size = static_cast<std::size_t>(
          controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.hierarchical_overlap") {
      options.hierarchical_overlap = static_cast<std::size_t>(
          controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--Colite.hierarchical_num_workers") {
      options.hierarchical_num_workers = static_cast<std::size_t>(
          controllers::SfmCount(RequireValue(args, &i, arg)));
    } else if (arg == "--FeatureMatching.type") {
      options.matcher_type = ParseSfmMatcherType(RequireValue(args, &i, arg));
    } else if (arg == "--SiftMatching.max_ratio" || arg == "--AlikedMatching.brute_force_max_ratio") {
      options.max_ratio = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--FeatureMatching.guided_matching") {
      options.guided_matching = controllers::SfmBool(RequireValue(args, &i, arg));
    } else if (arg == "--SiftMatching.max_distance") {
      options.max_distance = controllers::SfmDouble(RequireValue(args, &i, arg));
    } else if (arg == "--SiftMatching.cross_check" || arg == "--AlikedMatching.brute_force_cross_check") {
      options.mutual_check = controllers::SfmBool(RequireValue(args, &i, arg));
    } else {
      const std::string value = RequireValue(args, &i, arg);
      if (controllers::ApplySfmGeometryOption(arg, value, &options.two_view_geometry)) continue;
      IncrementalMappingOptions parsed;
      if (controllers::ApplySfmMapperOption(arg, value, &parsed)) {
        options.mapper_options.emplace(arg, value);
        continue;
      }
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.matcher_type == FeatureMatcherType::kAlikedBruteForce &&
      std::find(args.begin(), args.end(), "--AlikedMatching.brute_force_max_ratio") == args.end()) {
    options.max_ratio = 1.0;
  }
  if (!max_num_features_supplied && IsAlikedFeature(options.feature_extractor_type)) {
    options.max_num_features = 2048;
  }
  if (std::find(args.begin(), args.end(), "--Colite.guided_match_ratio") == args.end())
    options.guided_match_ratio = options.max_ratio;
  if (options.matcher_type == FeatureMatcherType::kSiftBruteForce && !options.matching_execution.use_gpu.has_value())
    options.matching_execution.use_gpu = IsCudaDescriptorMatcherAvailable();
  return options;
}

void ValidateOptions(const ReconSfmCliOptions& options) {
  if (options.show_help || options.show_version) {
    return;
  }
  if (options.resume && options.force) {
    throw std::invalid_argument("--resume and --force are mutually exclusive");
  }
  if (options.mapper_type != controllers::SfmMapperType::kIncremental &&
      !options.input_model_path.empty()) {
    throw std::invalid_argument(
        "global/hierarchical mapper does not support --input_path yet");
  }
  if (options.hierarchical_target_cluster_size == 0 ||
      options.hierarchical_max_cluster_size <
          options.hierarchical_target_cluster_size) {
    throw std::invalid_argument("Invalid hierarchical cluster size limits");
  }
  if (options.image_path.empty()) {
    throw std::invalid_argument("Missing required argument --image_path");
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path");
  }
  if (!std::isfinite(options.max_ratio) || options.max_ratio <= 0 || options.max_ratio > 1 ||
      !std::isfinite(options.max_distance) || options.max_distance <= 0) {
    throw std::invalid_argument("Invalid SIFT matching thresholds");
  }
  if (options.mapper_type == controllers::SfmMapperType::kGlobal) {
    for (const auto& [name, value] : options.mapper_options) {
      (void)value;
      if (name.rfind("--Mapper.tri_", 0) != 0 && name != "--Mapper.ba_global_max_num_iterations") {
        throw std::invalid_argument(name + " is not implemented for the global mapper");
      }
    }
  }
  if (options.max_num_features <= 0) {
    throw std::invalid_argument("--SiftExtraction.max_num_features must be positive");
  }
  if (!IsFeatureMatcherCompatible(options.feature_extractor_type,
                                  options.matcher_type)) {
    throw std::invalid_argument(
        "--FeatureExtraction.type is incompatible with --FeatureMatching.type");
  }
  if (IsAlikedFeature(options.feature_extractor_type) &&
      options.aliked_model_path.empty()) {
    throw std::invalid_argument(
        "--AlikedExtraction.n16rot_model_path is required for ALIKED");
  }
  if ((options.matcher_type == FeatureMatcherType::kSiftLightGlue ||
       options.matcher_type == FeatureMatcherType::kAlikedLightGlue) &&
      options.learned_matcher_model_path.empty()) {
    throw std::invalid_argument(
        "--SiftMatching.lightglue_model_path is required for LightGlue");
  }
  if (!std::isfinite(options.aliked_min_score) ||
      options.aliked_min_score < 0.0 ||
      !std::isfinite(options.min_cosine_similarity) ||
      options.min_cosine_similarity < -1.0 ||
      options.min_cosine_similarity > 1.0 ||
      !std::isfinite(options.lightglue_min_score) ||
      options.lightglue_min_score < 0.0 ||
      options.learned_matcher_num_threads <= 0) {
    throw std::invalid_argument("Invalid learned frontend configuration");
  }
  if (options.learned_gpu_index < 0) {
    throw std::invalid_argument("--Colite.learned_gpu_index must be non-negative");
  }
  if (!ExistsCameraModel(options.camera_model_id)) {
    throw std::invalid_argument("--ImageReader.camera_model is unsupported");
  }
  if (options.aliked_max_image_size < 0) {
    throw std::invalid_argument(
        "--Colite.aliked_max_image_size must be non-negative");
  }
  if (!std::isfinite(options.default_focal_length_factor) ||
      options.default_focal_length_factor <= 0.0) {
    throw std::invalid_argument(
        "--ImageReader.default_focal_length_factor must be positive");
  }
  if (!options.camera_params.empty() &&
      options.camera_params.size() !=
          GetCameraModelSpec(options.camera_model_id).param_names.size()) {
    throw std::invalid_argument(
        "--ImageReader.camera_params count does not match --ImageReader.camera_model");
  }
  if (!std::isfinite(options.nearby_match_ratio) || options.nearby_match_ratio <= 0.0 || options.nearby_match_ratio > 1.0) {
    throw std::invalid_argument("--Colite.nearby_match_ratio must be in (0, 1]");
  }
  if (!std::isfinite(options.guided_match_ratio) || options.guided_match_ratio <= 0.0 ||
      options.guided_match_ratio > 1.0) {
    throw std::invalid_argument("--Colite.guided_match_ratio must be in (0, 1]");
  }
  if (options.max_registration_trials_per_image == 0) {
    throw std::invalid_argument(
        "--Mapper.max_reg_trials must be positive");
  }
  if (options.max_initialization_trials == 0) {
    throw std::invalid_argument("--Mapper.init_num_trials must be positive");
  }
  if (options.max_num_models == 0 ||
      options.min_num_images_per_model < 2) {
    throw std::invalid_argument(
        "--Mapper.max_num_models must be positive and --Mapper.min_model_size >= 2");
  }
  if (options.registration_retry_reprojection_error_multiplier < 1.0) {
    throw std::invalid_argument(
        "--Colite.registration_retry_error_multiplier must be >= 1");
  }
  if (options.registration_retry_max_reprojection_error_px <= 0.0) {
    throw std::invalid_argument(
        "--Colite.registration_retry_max_error must be positive");
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
  controller_options.camera_sharing_mode = options.camera_sharing_mode;
  controller_options.camera_model_id = options.camera_model_id;
  controller_options.camera_params = options.camera_params;
  controller_options.default_focal_length_factor =
      options.default_focal_length_factor;
  controller_options.max_num_features = options.max_num_features;
  controller_options.feature_num_threads = options.feature_num_threads;
  controller_options.feature_extractor_type = options.feature_extractor_type;
  controller_options.feature_backend = options.feature_backend;
  controller_options.feature_execution = options.feature_execution;
  controller_options.matching_execution = options.matching_execution;
  controller_options.aliked_model_path = options.aliked_model_path;
  controller_options.aliked_min_score = options.aliked_min_score;
  controller_options.max_ratio = options.max_ratio;
  controller_options.max_distance = options.max_distance;
  controller_options.mutual_check = options.mutual_check;
  controller_options.guided_matching = options.guided_matching;
  controller_options.two_view_geometry = options.two_view_geometry;
  controller_options.mapper_options = options.mapper_options;
  controller_options.matcher_backend = options.matcher_backend;
  controller_options.matcher_type = options.matcher_type;
  controller_options.aliked_max_image_size = options.aliked_max_image_size;
  controller_options.learned_cuda_memory_limit_mb =
      options.learned_cuda_memory_limit_mb;
  controller_options.learned_matcher_model_path =
      options.learned_matcher_model_path;
  controller_options.min_cosine_similarity = options.min_cosine_similarity;
  controller_options.lightglue_min_score = options.lightglue_min_score;
  controller_options.learned_matcher_num_threads =
      options.learned_matcher_num_threads;
  controller_options.learned_execution_provider =
      options.learned_execution_provider;
  controller_options.learned_gpu_index = options.learned_gpu_index;
  controller_options.mapper_type = options.mapper_type;
  controller_options.hierarchical_target_cluster_size =
      options.hierarchical_target_cluster_size;
  controller_options.hierarchical_max_cluster_size =
      options.hierarchical_max_cluster_size;
  controller_options.hierarchical_overlap = options.hierarchical_overlap;
  controller_options.hierarchical_num_workers =
      options.hierarchical_num_workers;
  controller_options.nearby_match_window = options.nearby_match_window;
  controller_options.nearby_match_ratio = options.nearby_match_ratio;
  controller_options.guided_match_ratio = options.guided_match_ratio;
  controller_options.max_registration_trials_per_image =
      options.max_registration_trials_per_image;
  controller_options.max_initialization_trials =
      options.max_initialization_trials;
  controller_options.use_structureless_registration =
      options.use_structureless_registration;
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
  controller_options.use_position_priors = options.use_position_priors;
  controller_options.position_prior_stddev = options.position_prior_stddev;
  controller_options.position_prior_loss_scale =
      options.position_prior_loss_scale;
  controller_options.position_prior_alignment_max_error =
      options.position_prior_alignment_max_error;
  return controller_options;
}

}  // namespace

void ValidateReconSfmArguments(const std::vector<std::string>& args) {
  ValidateOptions(ParseOptions(args));
}

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
