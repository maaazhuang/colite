#include "colite/exe/mapper.h"

#include <cstdlib>
#include <cmath>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/mapper.h"
#include "colite/core/version.h"
#include "colite/exe/sfm_options.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite mapper " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable << " --Colite.workspace_path <dir>\n\n"
      << "Options:\n"
      << "  --Colite.workspace_path <dir>   Colite file workspace directory.\n"
      << "  --input_path <dir> Continue from an existing COLMAP model.\n"
      << "  --Colite.input_format <auto|txt|bin> Input model format, default auto.\n"
      << "  --Colite.output_format <txt|bin|both> Sparse model format, default both.\n"
      << "  --Mapper.multiple_models <0|1>  Reconstruct remaining components, default 1.\n"
      << "  --Colite.merge_models <0|1>     Merge Sim(3)-alignable models, default 1.\n"
      << "  --Mapper.max_num_models <n>     Maximum output models, default 50.\n"
      << "  --Mapper.min_model_size <n>     Minimum registered images per model, default 2.\n"
      << "  --Colite.use_position_priors <0|1> Align to workspace positions and constrain final BA.\n"
      << "  --Colite.position_prior_stddev <m> Default 5.\n"
      << "  --Colite.position_prior_loss_scale <m> Default 2.\n"
      << "  --Colite.position_prior_alignment_max_error <m> Default 20.\n";
  out << "  --Colite.registration_retry_error_multiplier <factor> Default 1.5; 1 keeps fixed PnP thresholds.\n"
      << "  --Colite.registration_retry_max_error <px> Default 8.\n";
  PrintSfmOptionsHelp(out, SfmOptionScope::kMapper);
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

bool ParseBool(const std::string& value, std::string_view flag) {
  if (value == "1" || value == "true") {
    return true;
  }
  if (value == "0" || value == "false") {
    return false;
  }
  throw std::invalid_argument("Expected 0/1 for " + std::string(flag));
}

}  // namespace

int RunMapper(const std::vector<std::string>& raw_args,
              std::ostream& out,
              std::ostream& err) {
  const std::string executable = raw_args.empty() ? "mapper" : raw_args.front();
  try {
    const auto args = NormalizeSfmArguments(raw_args, SfmOptionScope::kMapper);
    controllers::MapperControllerOptions options;
    if (args.size() <= 1) {
      PrintHelp(executable, out);
      return EXIT_SUCCESS;
    }
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& arg = args[i];
      if (arg == "--help" || arg == "-h") {
        PrintHelp(executable, out);
        return EXIT_SUCCESS;
      }
      if (arg == "--Colite.workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
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
      } else if (arg == "--Mapper.multiple_models") {
        options.multiple_models =
            ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--Colite.merge_models") {
        options.merge_models = ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--Mapper.max_num_models") {
        options.max_num_models = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Mapper.min_model_size") {
        options.min_num_images_per_model = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.use_position_priors") {
        options.use_position_priors =
            ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--Colite.position_prior_stddev") {
        options.position_prior_stddev =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.position_prior_loss_scale") {
        options.position_prior_loss_scale =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.position_prior_alignment_max_error") {
        options.position_prior_alignment_max_error =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Mapper.ba_refine_focal_length" || arg == "--Mapper.ba_refine_principal_point" || arg == "--Mapper.ba_refine_extra_params") {
        const CameraParameterGroup group = arg == "--Mapper.ba_refine_focal_length" ? CameraParameterGroup::kFocalLength :
            (arg == "--Mapper.ba_refine_principal_point" ? CameraParameterGroup::kPrincipalPoint : CameraParameterGroup::kDistortion);
        const bool enabled = controllers::SfmBool(RequireValue(args, &i, arg));
        options.ba_refine_camera_params = static_cast<CameraParameterGroup>(enabled ?
            static_cast<int>(options.ba_refine_camera_params) | static_cast<int>(group) :
            static_cast<int>(options.ba_refine_camera_params) & ~static_cast<int>(group));
      } else {
        const std::string value = RequireValue(args, &i, arg);
        IncrementalMappingOptions parsed;
        if (!controllers::ApplySfmMapperOption(arg, value, &parsed)) {
          throw std::invalid_argument("Unknown argument: " + arg);
        }
        options.mapper_options.emplace(arg, value);
      }
    }
    if (options.workspace_path.empty()) {
      throw std::invalid_argument("--Colite.workspace_path is required");
    }
    if (options.max_num_models == 0 ||
        options.min_num_images_per_model < 2) {
      throw std::invalid_argument(
          "--Mapper.max_num_models must be positive and --Mapper.min_model_size >= 2");
    }
    if (!std::isfinite(options.position_prior_stddev) ||
        options.position_prior_stddev <= 0.0 ||
        !std::isfinite(options.position_prior_loss_scale) ||
        options.position_prior_loss_scale < 0.0 ||
        !std::isfinite(options.position_prior_alignment_max_error) ||
        options.position_prior_alignment_max_error <= 0.0) {
      throw std::invalid_argument("Invalid position prior configuration");
    }
    const auto summary = controllers::MapperController(options).Run();
    out << "Registered images: " << summary.num_registered_images << "\n"
        << "Sparse cache hit: " << (summary.cache_hit ? 1 : 0) << "\n"
        << "Sparse models: " << summary.num_models << "\n"
        << "Unregistered images: " << summary.num_unregistered_images << "\n"
        << "Points3D: " << summary.num_points3d << "\n"
        << "Position-prior aligned models: "
        << summary.num_position_prior_aligned_models << "\n"
        << "Final mean position error: "
        << summary.final_mean_position_error << "\n";
    return summary.success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
