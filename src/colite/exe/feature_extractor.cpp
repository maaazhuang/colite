#include "colite/exe/feature_extractor.h"

#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/feature_extractor.h"
#include "colite/core/version.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite feature_extractor " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable
      << " --image_path <dir> --workspace_path <dir> [options]\n\n"
      << "Options:\n"
      << "  --focal_length <px>       Initial focal length in pixels.\n"
      << "  --camera_model <name>     Camera model, default SIMPLE_RADIAL.\n"
      << "  --camera_params <csv>     Explicit model parameters in COLMAP order.\n"
      << "  --camera_mode <auto|single|per_folder|per_image>\n"
         "                            Default auto groups complete EXIF camera "
         "identities. Legacy shared/individual aliases remain accepted.\n"
      << "  --default_focal_length_factor <f>\n"
         "                            Missing-EXIF focal factor, default 1.2.\n"
      << "  --max_num_features <n>   Feature limit, default 4096.\n"
      << "  --feature_type <sift|aliked_n16rot|aliked_n32>\n"
      << "                            Feature family, default sift.\n"
      << "  --feature_num_threads <n> CPU feature threads, default -1.\n"
      << "  --feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>\n"
      << "                            Default auto prefers CUDA OpenCV and falls "
         "back to CPU OpenCV.\n"
      << "  --aliked_model_path <file> Required ONNX model for ALIKED.\n"
      << "  --aliked_min_score <s>    ALIKED score threshold, default 0.2.\n";
}

FeatureExtractorType ParseFeatureType(const std::string& value) {
  if (value == "sift") return FeatureExtractorType::kSift;
  if (value == "aliked_n16rot" || value == "aliked-n16rot") {
    return FeatureExtractorType::kAlikedN16Rot;
  }
  if (value == "aliked_n32" || value == "aliked-n32") {
    return FeatureExtractorType::kAlikedN32;
  }
  throw std::invalid_argument("Unsupported feature type: " + value);
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

}  // namespace

int RunFeatureExtractor(const std::vector<std::string>& args,
                        std::ostream& out,
                        std::ostream& err) {
  const std::string executable =
      args.empty() ? "feature_extractor" : args.front();
  try {
    controllers::FeatureExtractorControllerOptions options;
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
      if (arg == "--image_path") {
        options.image_path = RequireValue(args, &i, arg);
      } else if (arg == "--workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
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
        options.num_threads = std::stoi(RequireValue(args, &i, arg));
      } else if (arg == "--feature_type") {
        options.extractor_type = ParseFeatureType(RequireValue(args, &i, arg));
      } else if (arg == "--feature_extractor") {
        options.backend = SiftBackendFromName(RequireValue(args, &i, arg));
      } else if (arg == "--aliked_model_path") {
        options.aliked_model_path = RequireValue(args, &i, arg);
      } else if (arg == "--aliked_min_score") {
        options.aliked_min_score =
            std::stof(RequireValue(args, &i, arg));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.image_path.empty() || options.workspace_path.empty()) {
      throw std::invalid_argument(
          "--image_path and --workspace_path are required");
    }
    if (IsAlikedFeature(options.extractor_type) &&
        options.aliked_model_path.empty()) {
      throw std::invalid_argument(
          "--aliked_model_path is required for ALIKED extraction");
    }
    const auto summary =
        controllers::FeatureExtractorController(options).Run();
    out << "Images: " << summary.num_images << "\n"
        << "Cameras: " << summary.num_cameras << "\n"
        << "Feature cache hit: " << (summary.cache_hit ? 1 : 0) << "\n"
        << "Feature extraction seconds: " << summary.elapsed_seconds << "\n";
    return summary.success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
