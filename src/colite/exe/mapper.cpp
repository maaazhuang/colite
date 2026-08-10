#include "colite/exe/mapper.h"

#include <cstdlib>
#include <cmath>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/mapper.h"
#include "colite/core/version.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite mapper " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable << " --workspace_path <dir>\n\n"
      << "Options:\n"
      << "  --workspace_path <dir>   Colite file workspace directory.\n"
      << "  --input_model_path <dir> Continue from an existing COLMAP model.\n"
      << "  --input_format <auto|txt|bin> Input model format, default auto.\n"
      << "  --output_format <txt|bin|both> Sparse model format, default both.\n"
      << "  --multiple_models <0|1>  Reconstruct remaining components, default 1.\n"
      << "  --merge_models <0|1>     Merge Sim(3)-alignable models, default 1.\n"
      << "  --max_num_models <n>     Maximum output models, default 50.\n"
      << "  --min_model_size <n>     Minimum registered images per model, default 2.\n"
      << "  --use_position_priors <0|1> Align to workspace positions and constrain final BA.\n"
      << "  --position_prior_stddev <m> Default 5.\n"
      << "  --position_prior_loss_scale <m> Default 2.\n"
      << "  --position_prior_alignment_max_error <m> Default 20.\n";
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

int RunMapper(const std::vector<std::string>& args,
              std::ostream& out,
              std::ostream& err) {
  const std::string executable = args.empty() ? "mapper" : args.front();
  try {
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
      if (arg == "--workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
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
      } else if (arg == "--multiple_models") {
        options.multiple_models =
            ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--merge_models") {
        options.merge_models = ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--max_num_models") {
        options.max_num_models = static_cast<std::size_t>(
            std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--min_model_size") {
        options.min_num_images_per_model = static_cast<std::size_t>(
            std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--use_position_priors") {
        options.use_position_priors =
            ParseBool(RequireValue(args, &i, arg), arg);
      } else if (arg == "--position_prior_stddev") {
        options.position_prior_stddev =
            std::stod(RequireValue(args, &i, arg));
      } else if (arg == "--position_prior_loss_scale") {
        options.position_prior_loss_scale =
            std::stod(RequireValue(args, &i, arg));
      } else if (arg == "--position_prior_alignment_max_error") {
        options.position_prior_alignment_max_error =
            std::stod(RequireValue(args, &i, arg));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.workspace_path.empty()) {
      throw std::invalid_argument("--workspace_path is required");
    }
    if (options.max_num_models == 0 ||
        options.min_num_images_per_model < 2) {
      throw std::invalid_argument(
          "--max_num_models must be positive and --min_model_size >= 2");
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
