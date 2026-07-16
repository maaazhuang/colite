#include "colite/exe/sequential_matcher.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/sequential_matcher.h"
#include "colite/core/version.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/vocabulary_tree.h"

namespace colite::exe {
namespace {

std::string_view StrategyName(const MatchingStrategy strategy) {
  switch (strategy) {
    case MatchingStrategy::kExhaustive:
      return "exhaustive";
    case MatchingStrategy::kSequential:
      return "sequential";
    case MatchingStrategy::kSpatial:
      return "spatial";
    case MatchingStrategy::kVocabTree:
      return "vocab_tree";
  }
  return "unknown";
}

MatchingStrategy ParseMatchingStrategy(const std::string& name) {
  if (name == "exhaustive") {
    return MatchingStrategy::kExhaustive;
  }
  if (name == "sequential") {
    return MatchingStrategy::kSequential;
  }
  if (name == "spatial") {
    return MatchingStrategy::kSpatial;
  }
  if (name == "vocab_tree" || name == "vocab-tree" || name == "vocabtree") {
    return MatchingStrategy::kVocabTree;
  }
  throw std::invalid_argument("Unsupported matching type: " + name);
}

void PrintHelp(std::string_view executable,
               MatchingStrategy default_strategy,
               std::ostream& out) {
  const std::string executable_text(executable);
  const std::string command_name =
      std::filesystem::path(executable_text).filename().string();
  const std::string display_name =
      executable_text.rfind("colite ", 0) == 0
          ? executable_text
          : "colite " + command_name;
  out << display_name << " " << colite::Version() << "\n\n"
      << "Usage:\n"
      << "  " << executable << " --workspace_path <dir> [options]\n\n"
      << "Options:\n"
      << "  --matching_type <exhaustive|sequential|spatial|vocab_tree>\n"
      << "                            Pair generation strategy, default "
      << StrategyName(default_strategy) << ".\n"
      << "  --overlap <n>             Match each image with next n images, "
         "default 5.\n"
      << "  --spatial_max_num_neighbors <n>\n"
      << "                            Spatial neighbors per image, default 10.\n"
      << "  --spatial_max_distance <d>\n"
      << "                            Spatial distance limit; <=0 disables it.\n"
      << "  --vocab_tree_num_images <n>\n"
      << "                            Retrieved candidate images per image, "
         "default 50.\n"
      << "  --vocab_tree_num_visual_words <n>\n"
      << "                            Hierarchical visual words, "
         "default 1024.\n"
      << "  --vocab_tree_branching <n> Tree branching factor, default 16.\n"
      << "  --vocab_tree_num_iterations <n> K-means iterations per node, "
         "default 6.\n"
      << "  --vocab_tree_max_num_features <n>\n"
      << "                            Max descriptors per image used to build "
         "retrieval histograms, default 5000.\n"
      << "  --vocab_tree_path <file>  Load a COLITE text vocab tree file instead "
         "of building one in memory.\n"
      << "  --vocab_tree_sequential_overlap <n> Merge next-n temporal pairs; "
         "0 disables, default 2.\n"
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

int RunMatcherCommand(const std::vector<std::string>& args,
                      const MatchingStrategy default_strategy,
                      std::ostream& out,
                      std::ostream& err) {
  const std::string executable = args.empty()
                                     ? std::string("matcher")
                                     : args.front();
  try {
    controllers::SequentialMatcherControllerOptions options;
    options.matching_strategy = default_strategy;
    if (args.size() <= 1) {
      PrintHelp(executable, default_strategy, out);
      return EXIT_SUCCESS;
    }
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& arg = args[i];
      if (arg == "--help" || arg == "-h") {
        PrintHelp(executable, default_strategy, out);
        return EXIT_SUCCESS;
      }
      if (arg == "--workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
      } else if (arg == "--matching_type") {
        options.matching_strategy =
            ParseMatchingStrategy(RequireValue(args, &i, arg));
      } else if (arg == "--overlap") {
        options.overlap =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--spatial_max_num_neighbors") {
        options.spatial_max_num_neighbors =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--spatial_max_distance") {
        options.spatial_max_distance = std::stod(RequireValue(args, &i, arg));
      } else if (arg == "--vocab_tree_num_images") {
        options.vocab_tree_num_images =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--vocab_tree_num_visual_words") {
        options.vocab_tree_num_visual_words =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--vocab_tree_branching") {
        options.vocab_tree_branching =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--vocab_tree_num_iterations") {
        options.vocab_tree_num_iterations =
            std::stoi(RequireValue(args, &i, arg));
      } else if (arg == "--vocab_tree_max_num_features") {
        options.vocab_tree_max_num_features =
            static_cast<std::size_t>(std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--vocab_tree_path") {
        options.vocab_tree_path = RequireValue(args, &i, arg);
      } else if (arg == "--vocab_tree_sequential_overlap") {
        options.vocab_tree_sequential_overlap =
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
        << "Raw match cache hit: "
        << (summary.raw_match_cache_hit ? 1 : 0) << "\n"
        << "Geometry cache hit: " << (summary.geometry_cache_hit ? 1 : 0)
        << "\n"
        << "Verified pairs: " << summary.num_verified_pairs << "\n"
        << "Verified inliers: " << summary.num_verified_inliers << "\n"
        << "Retrieval pairs: "
        << summary.pair_generation.num_retrieval_pairs << "\n"
        << "Sequential guard pairs: "
        << summary.pair_generation.num_sequential_guard_pairs << "\n"
        << "Sequential pairs added: "
        << summary.pair_generation.num_sequential_pairs_added << "\n"
        << "Sequential pair recall: "
        << summary.pair_generation.sequential_pair_recall << "\n"
        << "Candidate graph components: "
        << summary.pair_generation.num_connected_components << "\n"
        << "Candidate graph degree min/mean/max: "
        << summary.pair_generation.min_degree << "/"
        << summary.pair_generation.mean_degree << "/"
        << summary.pair_generation.max_degree << "\n";
    return summary.success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n\n";
    PrintHelp(executable, default_strategy, err);
    return EXIT_FAILURE;
  }
}

}  // namespace

int RunSequentialMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err) {
  return RunMatcherCommand(args, MatchingStrategy::kSequential, out, err);
}

int RunExhaustiveMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err) {
  return RunMatcherCommand(args, MatchingStrategy::kExhaustive, out, err);
}

int RunSpatialMatcher(const std::vector<std::string>& args,
                      std::ostream& out,
                      std::ostream& err) {
  return RunMatcherCommand(args, MatchingStrategy::kSpatial, out, err);
}

int RunVocabTreeMatcher(const std::vector<std::string>& args,
                        std::ostream& out,
                        std::ostream& err) {
  return RunMatcherCommand(args, MatchingStrategy::kVocabTree, out, err);
}

int RunVocabTreeBuilder(const std::vector<std::string>& args,
                        std::ostream& out,
                        std::ostream& err) {
  const std::string executable =
      args.empty() ? "vocab_tree_builder" : args.front();
  try {
    std::filesystem::path workspace_path;
    std::filesystem::path output_path;
    VocabularyTreeBuildOptions options;
    if (args.size() <= 1) {
      out << "Usage:\n  " << executable
          << " --workspace_path <dir> --output_path <file> [options]\n\n"
          << "Options:\n"
          << "  --num_visual_words <n>  Leaf visual words, default 1024.\n"
          << "  --branching <n>         Branching factor, default 16.\n"
          << "  --num_iterations <n>    K-means iterations per node, default 6.\n"
          << "  --max_num_features <n>  Descriptors sampled per image, default 5000.\n";
      return EXIT_SUCCESS;
    }
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& arg = args[i];
      if (arg == "--help" || arg == "-h") {
        std::vector<std::string> help_args{executable};
        return RunVocabTreeBuilder(help_args, out, err);
      }
      if (arg == "--workspace_path") {
        workspace_path = RequireValue(args, &i, arg);
      } else if (arg == "--output_path") {
        output_path = RequireValue(args, &i, arg);
      } else if (arg == "--num_visual_words") {
        options.num_visual_words = static_cast<std::size_t>(
            std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--branching") {
        options.branching = static_cast<std::size_t>(
            std::stoull(RequireValue(args, &i, arg)));
      } else if (arg == "--num_iterations") {
        options.num_iterations = std::stoi(RequireValue(args, &i, arg));
      } else if (arg == "--max_num_features") {
        options.max_num_features_per_image = static_cast<std::size_t>(
            std::stoull(RequireValue(args, &i, arg)));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (workspace_path.empty() || output_path.empty()) {
      throw std::invalid_argument(
          "--workspace_path and --output_path are required");
    }
    const std::vector<Image> images = ReadWorkspaceImages(workspace_path);
    MatchGraph graph;
    ReadWorkspaceFeatureSets(workspace_path, &graph);
    const VocabularyTree tree = BuildVocabularyTree(images, graph, options);
    tree.Write(output_path);
    out << "Vocabulary tree: " << output_path.string() << "\n"
        << "Nodes: " << tree.nodes.size() << "\n"
        << "Visual words: " << tree.NumVisualWords() << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n";
    return EXIT_FAILURE;
  }
}

}  // namespace colite::exe
