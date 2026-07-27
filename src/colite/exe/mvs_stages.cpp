#include "colite/exe/mvs_stages.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "colite/exe/recon_mvs.h"
#include "colite/exe/stage.h"

namespace colite::exe {
namespace {

struct ParsedStageArgs {
  StageOptions stage;
  bool help = false;
  std::unordered_map<std::string, std::string> paths;
  std::vector<std::string> forwarded;
};

bool IsPathOption(const std::string& value) {
  return value == "--workspace_path" || value == "--model_path" ||
         value == "--input_path" || value == "--image_path" ||
         value == "--output_path";
}

ParsedStageArgs Parse(const std::vector<std::string>& args) {
  ParsedStageArgs parsed;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string option = args[i];
    if (option == "--help" || option == "-h") {
      parsed.help = true;
    } else if (option == "--resume") {
      parsed.stage.resume = true;
    } else if (option == "--force") {
      parsed.stage.force = true;
    } else if (option == "--dry_run") {
      parsed.stage.dry_run = true;
    } else if (IsPathOption(option)) {
      if (++i >= args.size()) throw std::invalid_argument("Missing value for " + option);
      parsed.paths[option] = args[i];
    } else {
      parsed.forwarded.push_back(option);
      if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) {
        parsed.forwarded.push_back(args[++i]);
      }
    }
  }
  if (parsed.stage.resume && parsed.stage.force) {
    throw std::invalid_argument("--resume and --force are mutually exclusive");
  }
  return parsed;
}

std::filesystem::path Require(const ParsedStageArgs& args,
                              const std::string& option) {
  const auto it = args.paths.find(option);
  if (it == args.paths.end() || it->second.empty()) {
    throw std::invalid_argument("Missing required option " + option);
  }
  return it->second;
}

std::filesystem::path ModelPath(const ParsedStageArgs& args) {
  if (args.paths.count("--model_path")) return args.paths.at("--model_path");
  if (args.paths.count("--input_path")) return args.paths.at("--input_path");
  if (args.paths.count("--workspace_path")) {
    return std::filesystem::path(args.paths.at("--workspace_path")) / "sparse";
  }
  throw std::invalid_argument("Missing --model_path (or --workspace_path)");
}

std::filesystem::path ImagePath(const ParsedStageArgs& args) {
  if (args.paths.count("--image_path")) return args.paths.at("--image_path");
  if (args.paths.count("--workspace_path")) {
    return std::filesystem::path(args.paths.at("--workspace_path")) / "images";
  }
  throw std::invalid_argument("Missing --image_path (or --workspace_path)");
}

std::filesystem::path OutputPath(const ParsedStageArgs& args) {
  if (args.paths.count("--output_path")) return args.paths.at("--output_path");
  if (args.paths.count("--workspace_path")) return args.paths.at("--workspace_path");
  throw std::invalid_argument("Missing --output_path (or --workspace_path)");
}

std::vector<std::string> BuildReconArgs(const std::string& name,
                                        const std::filesystem::path& model,
                                        const std::filesystem::path& images,
                                        const std::filesystem::path& output,
                                        const ParsedStageArgs& parsed) {
  std::vector<std::string> result{name, "--model_path", model.string(),
                                  "--image_path", images.string(),
                                  "--output_path", output.string()};
  result.insert(result.end(), parsed.forwarded.begin(), parsed.forwarded.end());
  return result;
}

template <typename Function>
int Guard(Function&& function, std::ostream& err) {
  try {
    return function();
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << "\n";
    return 1;
  }
}

}  // namespace

int RunImageUndistorter(const std::vector<std::string>& argv,
                        std::ostream& out, std::ostream& err) {
  return Guard([&] {
    const ParsedStageArgs parsed = Parse(argv);
    if (parsed.help) {
      out << "Usage: image_undistorter --image_path DIR --input_path MODEL "
             "--output_path DIR [undistortion options] [--resume|--force] "
             "[--dry_run]\n";
      return 0;
    }
    const auto model = ModelPath(parsed);
    const auto images = ImagePath(parsed);
    const auto output = OutputPath(parsed);
    auto recon_args = BuildReconArgs(argv.front(), model, images, output, parsed);
    recon_args.push_back("--image_undistorter");
    StageSpec spec{"image_undistorter", output, {model, images},
                   {output / "undistorted" / "images",
                    output / "undistorted" / "sparse"},
                   recon_args};
    return RunStage(spec, parsed.stage,
                    [&] { return RunReconMvs(recon_args, out, err); }, out, err);
  }, err);
}

int RunPatchMatchStereo(const std::vector<std::string>& argv,
                        std::ostream& out, std::ostream& err) {
  return Guard([&] {
    const ParsedStageArgs parsed = Parse(argv);
    if (parsed.help) {
      out << "Usage: patch_match_stereo --workspace_path DIR "
             "[--model_path MODEL --image_path DIR --output_path DIR] "
             "[PatchMatch options] [--resume|--force] [--dry_run]\n";
      return 0;
    }
    const auto model = ModelPath(parsed);
    const auto images = ImagePath(parsed);
    const auto output = OutputPath(parsed);
    auto recon_args = BuildReconArgs(argv.front(), model, images, output, parsed);
    recon_args.push_back("--run_patch_match");
    recon_args.push_back("--check_geometric_consistency");
    auto fingerprint_args = recon_args;
    fingerprint_args.push_back("@implementation=patch_match_v4");
    StageSpec spec{"patch_match_stereo", output, {model, images},
                   {output / "stereo" / "depth_maps",
                    output / "stereo" / "normal_maps"},
                   fingerprint_args};
    return RunStage(spec, parsed.stage,
                    [&] { return RunReconMvs(recon_args, out, err); }, out, err);
  }, err);
}

int RunStereoFusion(const std::vector<std::string>& argv,
                    std::ostream& out, std::ostream& err) {
  return Guard([&] {
    const ParsedStageArgs parsed = Parse(argv);
    if (parsed.help) {
      out << "Usage: stereo_fusion --workspace_path DIR "
             "[fusion options] [--resume|--force] [--dry_run]\n";
      return 0;
    }
    const auto model = ModelPath(parsed);
    const auto images = ImagePath(parsed);
    const auto output = OutputPath(parsed);
    auto recon_args = BuildReconArgs(argv.front(), model, images, output, parsed);
    recon_args.push_back("--fuse_stereo");
    auto fingerprint_args = recon_args;
    fingerprint_args.push_back("@implementation=stereo_fusion_v3");
    StageSpec spec{"stereo_fusion", output,
                   {model, images, output / "stereo"},
                   {output / "dense" / "fused.ply"}, fingerprint_args};
    return RunStage(spec, parsed.stage,
                    [&] { return RunReconMvs(recon_args, out, err); }, out, err);
  }, err);
}

}  // namespace colite::exe
