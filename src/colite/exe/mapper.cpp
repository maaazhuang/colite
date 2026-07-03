#include "colite/exe/mapper.h"

#include <cstdlib>
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
      << "  --workspace_path <dir>   Colite file workspace directory.\n";
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
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.workspace_path.empty()) {
      throw std::invalid_argument("--workspace_path is required");
    }
    const auto summary = controllers::MapperController(options).Run();
    out << "Registered images: " << summary.num_registered_images << "\n"
        << "Sparse cache hit: " << (summary.cache_hit ? 1 : 0) << "\n"
        << "Points3D: " << summary.num_points3d << "\n";
    return summary.success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
