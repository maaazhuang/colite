#include "colite/exe/command.h"

#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "colite/core/version.h"
#include "colite/exe/compare_colmap.h"
#include "colite/exe/feature_extractor.h"
#include "colite/exe/mapper.h"
#include "colite/exe/model_tools.h"
#include "colite/exe/mvs_stages.h"
#include "colite/exe/mvs_tools.h"
#include "colite/exe/recon_auto.h"
#include "colite/exe/recon_mvs.h"
#include "colite/exe/recon_sfm.h"
#include "colite/exe/sequential_matcher.h"

namespace colite::exe {
namespace {

using CommandRunner = int (*)(const std::vector<std::string>&,
                              std::ostream&,
                              std::ostream&);

struct Command {
  std::string_view name;
  std::string_view description;
  CommandRunner runner = nullptr;
};

constexpr Command kCommands[] = {
    {"feature_extractor", "Extract image features into a file workspace.",
     &RunFeatureExtractor},
    {"exhaustive_matcher", "Match every image pair in a file workspace.",
     &RunExhaustiveMatcher},
    {"sequential_matcher", "Match neighboring images in a file workspace.",
     &RunSequentialMatcher},
    {"spatial_matcher", "Match spatially neighboring images in a file workspace.",
     &RunSpatialMatcher},
    {"vocab_tree_matcher", "Retrieve candidate pairs with visual words.",
     &RunVocabTreeMatcher},
    {"transitive_matcher", "Expand verified paths and match new image pairs.",
     &RunTransitiveMatcher},
    {"vocab_tree_builder", "Build a persistent hierarchical vocabulary tree.",
     &RunVocabTreeBuilder},
    {"mapper", "Run incremental SfM from a file workspace.", &RunMapper},
    {"compare_colmap", "Write fixed COLMAP text model difference reports.",
     &RunCompareColmap},
    {"model_analyzer", "Analyze a COLMAP-compatible sparse model.",
     &RunModelAnalyzer},
    {"model_transformer", "Apply a Sim(3) matrix to a sparse model.",
     &RunModelTransformer},
    {"model_aligner", "Align a sparse model to named XYZ positions.",
     &RunModelAligner},
    {"model_cropper", "Crop sparse points to an axis-aligned box.",
     &RunModelCropper},
    {"model_splitter", "Split a sparse model by track connectivity.",
     &RunModelSplitter},
    {"bundle_adjuster", "Run standalone global bundle adjustment.",
     &RunBundleAdjuster},
    {"point_triangulator", "Triangulate and complete tracks in an existing model.",
     &RunPointTriangulator},
    {"image_registrator", "Register remaining images against an existing model.",
     &RunImageRegistrator},
    {"recon_sfm", "Run the compact end-to-end SfM pipeline.", &RunReconSfm},
    {"recon_mvs", "Run the compact MVS pipeline from a sparse model.",
     &RunReconMvs},
    {"recon_auto", "Run the staged SfM-to-dense reconstruction pipeline.",
     &RunReconAuto},
    {"image_undistorter", "Create the undistorted dense workspace.",
     &RunImageUndistorter},
    {"patch_match_stereo", "Estimate dense depth and normal maps.",
     &RunPatchMatchStereo},
    {"stereo_fusion", "Fuse dense stereo maps into a point cloud.",
     &RunStereoFusion},
    {"poisson_mesher", "Run bundled Poisson surface reconstruction.",
     &RunPoissonMesher},
    {"delaunay_mesher", "Run visibility-aware Delaunay meshing.",
     &RunDelaunayMesher},
    {"mesh_simplifier", "Simplify a triangular PLY mesh.",
     &RunMeshSimplifier},
    {"mesh_texturer", "Bake calibrated images into a mesh texture atlas.",
     &RunMeshTexturer},
};

void PrintHelp(const std::string& executable, std::ostream& out) {
  out << "colite " << colite::Version() << "\n"
      << "\n"
      << "Usage:\n"
      << "  " << executable << " --help\n"
      << "  " << executable << " --version\n"
      << "  " << executable << " <command> [options]\n"
      << "\n"
      << "Available commands:\n";
  for (const Command& command : kCommands) {
    out << "  " << command.name << "\n"
        << "      " << command.description << "\n";
  }
}

const Command* FindCommand(const std::string_view name) {
  for (const Command& command : kCommands) {
    if (command.name == name) {
      return &command;
    }
  }
  return nullptr;
}

std::vector<std::string> MakeSubcommandArgs(
    const std::vector<std::string>& args,
    const std::string& command_name) {
  std::vector<std::string> subcommand_args;
  subcommand_args.reserve(args.size() - 1);
  subcommand_args.push_back(args.front() + " " + command_name);
  for (std::size_t i = 2; i < args.size(); ++i) {
    subcommand_args.push_back(args[i]);
  }
  return subcommand_args;
}

}  // namespace

int RunColite(const std::vector<std::string>& args,
              std::ostream& out,
              std::ostream& err) {
  const std::string executable = args.empty() ? "colite" : args.front();
  if (args.size() <= 1 || args[1] == "--help" || args[1] == "-h") {
    PrintHelp(executable, out);
    return 0;
  }
  if (args[1] == "--version") {
    out << colite::Version() << "\n";
    return 0;
  }

  const std::string& command_name = args[1];
  const Command* command = FindCommand(command_name);
  if (command == nullptr) {
    err << "Unknown command: " << command_name << "\n\n";
    PrintHelp(executable, err);
    return 1;
  }

  return command->runner(MakeSubcommandArgs(args, command_name), out, err);
}

}  // namespace colite::exe
