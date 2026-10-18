#include "colite/exe/feature_extractor.h"

#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/feature_extractor.h"
#include "colite/core/version.h"
#include "colite/exe/sfm_options.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite feature_extractor " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable
      << " --image_path <dir> --Colite.workspace_path <dir> [options]\n\n"
      << "Options:\n"
      << "  --ImageReader.camera_model <name>     Camera model, default SIMPLE_RADIAL.\n"
      << "  --ImageReader.camera_params <csv>     Explicit model parameters in COLMAP order.\n"
      << "  --ImageReader.default_focal_length_factor <f>\n"
         "                            Missing-EXIF focal factor, default 1.2.\n"
      << "  --SiftExtraction.max_num_features <n>   Feature limit, default 8192.\n"
      << "  --FeatureExtraction.type <sift|aliked_n16rot|aliked_n32>\n"
      << "                            Feature family, default sift.\n"
      << "  --FeatureExtraction.num_threads <n> CPU feature threads, default -1.\n"
      << "  --Colite.feature_extractor <auto|vlsift|cvsift|cuda_opencv|cuda_vlfeat>\n"
      << "                            Default auto prefers CUDA VLFeat and falls "
         "back to CPU VLFeat.\n"
      << "  --AlikedExtraction.n16rot_model_path <file> Required ONNX model for ALIKED.\n"
      << "  --AlikedExtraction.min_score <s>    ALIKED score threshold, default 0.2.\n"
      << "  --Colite.learned_cuda_memory_limit_mb <n> CUDA arena limit, default 8192.\n";
  PrintSfmOptionsHelp(out, SfmOptionScope::kFeatures);
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

int RunFeatureExtractor(const std::vector<std::string>& raw_args,
                        std::ostream& out,
                        std::ostream& err) {
  const std::string executable =
      raw_args.empty() ? "feature_extractor" : raw_args.front();
  try {
    const auto args = NormalizeSfmArguments(raw_args, SfmOptionScope::kFeatures);
    controllers::FeatureExtractorControllerOptions options;
    bool max_num_features_supplied = false;
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
      if (arg.rfind("--", 0) == 0 && i + 1 < args.size() &&
          ApplySfmFeatureExecutionOption(arg, args[i + 1], &options.execution)) {
        ++i;
      } else if (arg == "--image_path") {
        options.image_path = RequireValue(args, &i, arg);
      } else if (arg == "--Colite.workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
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
        options.num_threads = controllers::SfmInt(RequireValue(args, &i, arg));
      } else if (arg == "--FeatureExtraction.type") {
        options.extractor_type = ParseFeatureType(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.feature_extractor") {
        options.backend = SiftBackendFromName(RequireValue(args, &i, arg));
      } else if (arg == "--AlikedExtraction.n16rot_model_path" || arg == "--AlikedExtraction.n32_model_path") {
        options.aliked_model_path = RequireValue(args, &i, arg);
      } else if (arg == "--AlikedExtraction.min_score") {
        options.aliked_min_score =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.learned_cuda_memory_limit_mb") {
        options.learned_cuda_memory_limit_mb = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (!max_num_features_supplied && IsAlikedFeature(options.extractor_type)) {
      options.max_num_features = 2048;
    }
    if (options.max_num_features <= 0) {
      throw std::invalid_argument("Extraction max_num_features must be positive");
    }
    if (options.image_path.empty() || options.workspace_path.empty()) {
      throw std::invalid_argument(
          "--image_path and --Colite.workspace_path are required");
    }
    if (IsAlikedFeature(options.extractor_type) &&
        options.aliked_model_path.empty()) {
      throw std::invalid_argument(
          "--AlikedExtraction.n16rot_model_path is required for ALIKED extraction");
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
