#include "colite/exe/compare_colmap.h"

#include <cstdlib>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "colite/controllers/colmap_compare_controller.h"
#include "colite/core/version.h"

namespace colite::exe {
namespace {

void PrintHelp(const std::string_view executable, std::ostream& out) {
  out << "colite compare_colmap " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable
      << " --reference_path <dir> --candidate_path <dir> --output_path <dir>\n\n"
      << "Options:\n"
      << "  --reference_path <dir>  Reference COLMAP text model directory.\n"
      << "  --candidate_path <dir>  Candidate COLMAP text model directory.\n"
      << "  --output_path <dir>     Directory for fixed TSV difference reports.\n";
}

std::string RequireValue(const std::vector<std::string>& args,
                         std::size_t* index,
                         const std::string_view flag) {
  if (*index + 1 >= args.size()) {
    throw std::invalid_argument("Missing value for " + std::string(flag));
  }
  ++(*index);
  return args[*index];
}

}  // namespace

int RunCompareColmap(const std::vector<std::string>& args,
                     std::ostream& out,
                     std::ostream& err) {
  const std::string executable = args.empty() ? "compare_colmap" : args.front();
  try {
    controllers::ColmapCompareControllerOptions options;
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
      if (arg == "--reference_path") {
        options.reference_path = RequireValue(args, &i, arg);
      } else if (arg == "--candidate_path") {
        options.candidate_path = RequireValue(args, &i, arg);
      } else if (arg == "--output_path") {
        options.output_path = RequireValue(args, &i, arg);
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.reference_path.empty()) {
      throw std::invalid_argument("--reference_path is required");
    }
    if (options.candidate_path.empty()) {
      throw std::invalid_argument("--candidate_path is required");
    }
    if (options.output_path.empty()) {
      throw std::invalid_argument("--output_path is required");
    }

    const auto summary =
        controllers::ColmapCompareController(std::move(options)).Run();
    out << "Reference images: " << summary.reference_images << "\n"
        << "Candidate images: " << summary.candidate_images << "\n"
        << "Common images: " << summary.common_images << "\n"
        << "Common points by id: " << summary.common_points_by_id << "\n"
        << "Common pairs: " << summary.common_pairs << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, err);
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
