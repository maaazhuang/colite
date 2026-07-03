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
      << "  --max_num_features <n>   SIFT feature limit, default 4096.\n"
      << "  --feature_num_threads <n> CPU feature threads, default -1.\n"
      << "  --feature_extractor <vlsift|cvsift|cudasift>\n";
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
      } else if (arg == "--max_num_features") {
        options.max_num_features = std::stoi(RequireValue(args, &i, arg));
      } else if (arg == "--feature_num_threads") {
        options.num_threads = std::stoi(RequireValue(args, &i, arg));
      } else if (arg == "--feature_extractor") {
        options.backend = SiftBackendFromName(RequireValue(args, &i, arg));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.image_path.empty() || options.workspace_path.empty()) {
      throw std::invalid_argument(
          "--image_path and --workspace_path are required");
    }
    const auto summary =
        controllers::FeatureExtractorController(options).Run();
    out << "Images: " << summary.num_images << "\n"
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
