#include "colite/exe/sequential_matcher.h"

#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/sequential_matcher.h"
#include "colite/core/version.h"

namespace colite::exe {
namespace {

void PrintHelp(std::string_view executable, std::ostream& out) {
  out << "colite sequential_matcher " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable << " --workspace_path <dir> [options]\n\n"
      << "Options:\n"
      << "  --overlap <n>             Match each image with next n images, "
         "default 5.\n"
      << "  --matcher <bf|flann>     Descriptor matcher backend, default flann.\n"
      << "  --max_ratio <r>          Lowe ratio, default 0.8.\n";
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

int RunSequentialMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err) {
  const std::string executable =
      args.empty() ? "sequential_matcher" : args.front();
  try {
    controllers::SequentialMatcherControllerOptions options;
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
      } else if (arg == "--overlap") {
        options.overlap =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--max_ratio") {
        options.max_ratio = std::stod(RequireValue(args, &i, arg));
      } else if (arg == "--matcher") {
        const std::string matcher = RequireValue(args, &i, arg);
        if (matcher == "bf") {
          options.matcher_backend = DescriptorMatcherBackend::kBruteForce;
        } else if (matcher == "flann") {
          options.matcher_backend = DescriptorMatcherBackend::kFlann;
        } else {
          throw std::invalid_argument("Unsupported matcher: " + matcher);
        }
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.workspace_path.empty()) {
      throw std::invalid_argument("--workspace_path is required");
    }
    const auto summary =
        controllers::SequentialMatcherController(options).Run();
    out << "Pairs: " << summary.num_pairs << "\n"
        << "Verified pairs: " << summary.num_verified_pairs << "\n"
        << "Verified inliers: " << summary.num_verified_inliers << "\n";
    return summary.success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
