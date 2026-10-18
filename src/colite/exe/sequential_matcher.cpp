#include "colite/exe/sequential_matcher.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string_view>

#include "colite/controllers/sequential_matcher.h"
#include "colite/core/version.h"
#include "colite/exe/sfm_options.h"
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
    case MatchingStrategy::kTransitive:
      return "transitive";
  }
  return "unknown";
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
      << "  " << executable << " --Colite.workspace_path <dir> [options]\n\n"
      << "Options:\n"
      << "                            Pair generation strategy, default "
      << StrategyName(default_strategy) << ".\n"
      << "                            Descriptor matcher, default bf.\n"
      << "  --SiftMatching.lightglue_model_path <file> LightGlue ONNX model path.\n"
      << "  --AlikedMatching.brute_force_min_cossim <s> ALIKED BF threshold, default 0.85.\n"
      << "  --SiftMatching.lightglue_min_score <s> LightGlue score threshold, default 0.1.\n"
      << "  --Colite.learned_cuda_memory_limit_mb <n> CUDA arena limit, default 8192.\n"
      << "  --Colite.learned_use_io_binding <0|1> Bind ONNX CUDA I/O, default 1.\n"
      << "  --SiftMatching.max_ratio <r>          Lowe ratio, default 0.8.\n"
      << "  --FeatureMatching.guided_matching <0|1> Re-search descriptors under verified geometry, default 0.\n"
      << "  --Colite.guided_match_ratio <r> Candidate ratio under verified geometry, default 0.8.\n"
      << "  --Colite.guided_min_inlier_gain <n> Minimum guided inlier gain, default 10.\n";
  if (default_strategy == MatchingStrategy::kSequential) {
    out << "  --SequentialMatching.overlap <n> Default 10.\n"
        << "  --SequentialMatching.quadratic_overlap <0|1> Default 1.\n";
  } else if (default_strategy == MatchingStrategy::kSpatial) {
    out << "  --SpatialMatching.max_num_neighbors <n> Default 50.\n"
        << "  --SpatialMatching.max_distance <d> Positive, default 100.\n";
  } else if (default_strategy == MatchingStrategy::kVocabTree) {
    out << "  --VocabTreeMatching.num_images <n> Default 100.\n"
        << "  --Colite.vocab_tree_num_visual_words <n> Default 1024.\n"
        << "  --Colite.vocab_tree_branching <n> Default 16.\n"
        << "  --Colite.vocab_tree_num_iterations <n> Default 6.\n"
        << "  --Colite.vocab_tree_max_num_features <n> Default 5000.\n"
        << "  --Colite.vocab_tree_path <file> COLITE vocabulary file.\n"
        << "  --Colite.vocab_tree_sequential_overlap <n> Default 2.\n";
  } else if (default_strategy == MatchingStrategy::kTransitive) {
    out << "  --Colite.transitive_max_path_length <n> Default 2.\n"
        << "  --Colite.transitive_max_num_neighbors <n> Default 20.\n";
  }
  PrintSfmOptionsHelp(out, SfmOptionScope::kMatching);
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

int RunMatcherCommand(const std::vector<std::string>& raw_args,
                      const MatchingStrategy default_strategy,
                      std::ostream& out,
                      std::ostream& err) {
  const std::string executable = raw_args.empty()
                                     ? std::string("matcher")
                                     : raw_args.front();
  try {
    const auto args = NormalizeSfmArguments(raw_args, SfmOptionScope::kMatching);
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
      if (i + 1 < args.size() && ApplySfmMatchingExecutionOption(arg, args[i + 1], &options.execution)) {
        ++i;
        continue;
      }
      const bool sequential_option = arg.rfind("--SequentialMatching.", 0) == 0;
      const bool spatial_option = arg.rfind("--SpatialMatching.", 0) == 0;
      const bool vocabulary_option = arg.rfind("--VocabTreeMatching.", 0) == 0 ||
                                     arg.rfind("--Colite.vocab_tree_", 0) == 0;
      const bool transitive_option = arg.rfind("--Colite.transitive_", 0) == 0;
      if ((sequential_option && default_strategy != MatchingStrategy::kSequential) ||
          (spatial_option && default_strategy != MatchingStrategy::kSpatial) ||
          (vocabulary_option && default_strategy != MatchingStrategy::kVocabTree) ||
          (transitive_option && default_strategy != MatchingStrategy::kTransitive)) {
        throw std::invalid_argument(arg + " does not apply to " + std::string(StrategyName(default_strategy)));
      }
      if (arg == "--Colite.workspace_path") {
        options.workspace_path = RequireValue(args, &i, arg);
      } else if (arg == "--SequentialMatching.overlap") {
        options.overlap =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--SpatialMatching.max_num_neighbors") {
        options.spatial_max_num_neighbors =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--SpatialMatching.max_distance") {
        options.spatial_max_distance = controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--VocabTreeMatching.num_images") {
        options.vocab_tree_num_images =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.vocab_tree_num_visual_words") {
        options.vocab_tree_num_visual_words =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.vocab_tree_branching") {
        options.vocab_tree_branching =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.vocab_tree_num_iterations") {
        options.vocab_tree_num_iterations =
            controllers::SfmInt(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.vocab_tree_max_num_features") {
        options.vocab_tree_max_num_features =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.vocab_tree_path") {
        options.vocab_tree_path = RequireValue(args, &i, arg);
      } else if (arg == "--Colite.vocab_tree_sequential_overlap") {
        options.vocab_tree_sequential_overlap =
            static_cast<std::size_t>(controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.transitive_max_path_length") {
        options.transitive_max_path_length = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.transitive_max_num_neighbors") {
        options.transitive_max_num_neighbors = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--SiftMatching.max_ratio" || arg == "--AlikedMatching.brute_force_max_ratio") {
        options.max_ratio = controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--FeatureMatching.guided_matching") {
        options.guided_matching =
            controllers::SfmBool(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.guided_match_ratio") {
        options.guided_match_ratio = controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.guided_min_inlier_gain") {
        options.guided_min_inlier_gain = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--SiftMatching.lightglue_model_path" || arg == "--AlikedMatching.lightglue_model_path") {
        options.learned_matcher_model_path = RequireValue(args, &i, arg);
      } else if (arg == "--AlikedMatching.brute_force_min_cossim") {
        options.min_cosine_similarity =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--SiftMatching.lightglue_min_score" || arg == "--AlikedMatching.lightglue_min_score") {
        options.lightglue_min_score =
            controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.learned_cuda_memory_limit_mb") {
        options.learned_cuda_memory_limit_mb = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.learned_use_io_binding") {
        options.learned_use_io_binding =
            controllers::SfmBool(RequireValue(args, &i, arg));
      } else if (arg == "--FeatureMatching.type") {
        options.matcher_type = ParseSfmMatcherType(RequireValue(args, &i, arg));
      } else if (arg == "--SequentialMatching.quadratic_overlap") {
        options.quadratic_overlap = controllers::SfmBool(RequireValue(args, &i, arg));
      } else if (arg == "--SiftMatching.max_distance") {
        options.max_distance = controllers::SfmDouble(RequireValue(args, &i, arg));
      } else if (arg == "--SiftMatching.cross_check" || arg == "--AlikedMatching.brute_force_cross_check") {
        options.mutual_check = controllers::SfmBool(RequireValue(args, &i, arg));
      } else {
        const std::string value = RequireValue(args, &i, arg);
        if (controllers::ApplySfmGeometryOption(arg, value, &options.two_view_geometry)) continue;
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (options.matcher_type == FeatureMatcherType::kAlikedBruteForce &&
        std::find(args.begin(), args.end(), "--AlikedMatching.brute_force_max_ratio") == args.end()) {
      options.max_ratio = 1.0;
    }
    if (std::find(args.begin(), args.end(), "--Colite.guided_match_ratio") == args.end())
      options.guided_match_ratio = options.max_ratio;
    if (!std::isfinite(options.max_ratio) || options.max_ratio <= 0 || options.max_ratio > 1 ||
        !std::isfinite(options.max_distance) || options.max_distance <= 0) {
      throw std::invalid_argument("Invalid SIFT matching thresholds");
    }
    if (options.matcher_type == FeatureMatcherType::kSiftBruteForce && !options.execution.use_gpu.has_value())
      options.execution.use_gpu = IsCudaDescriptorMatcherAvailable();
    if (options.workspace_path.empty()) {
      throw std::invalid_argument("--Colite.workspace_path is required");
    }
    if (options.learned_matcher_num_threads <= 0 ||
        options.learned_gpu_index < 0) {
      throw std::invalid_argument("Invalid learned matcher runtime configuration");
    }
    if (options.guided_match_ratio <= 0.0 ||
        options.guided_match_ratio > 1.0) {
      throw std::invalid_argument("--Colite.guided_match_ratio must be in (0, 1]");
    }
    if ((options.matcher_type == FeatureMatcherType::kSiftLightGlue ||
         options.matcher_type == FeatureMatcherType::kAlikedLightGlue) &&
        options.learned_matcher_model_path.empty()) {
      throw std::invalid_argument(
          "--SiftMatching.lightglue_model_path is required for LightGlue");
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
        << "Guided improved pairs: "
        << summary.guided_matching.num_improved_pairs << "\n"
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

int RunTransitiveMatcher(const std::vector<std::string>& args,
                         std::ostream& out,
                         std::ostream& err) {
  return RunMatcherCommand(args, MatchingStrategy::kTransitive, out, err);
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
          << " --Colite.workspace_path <dir> --output_path <file> [options]\n\n"
          << "Options:\n"
          << "  --Colite.num_visual_words <n>  Leaf visual words, default 1024.\n"
          << "  --Colite.branching <n>         Branching factor, default 16.\n"
          << "  --Colite.num_iterations <n>    K-means iterations per node, default 6.\n"
          << "  --Colite.vocab_tree_max_num_features <n>  Descriptors sampled per image, default 5000.\n";
      return EXIT_SUCCESS;
    }
    for (std::size_t i = 1; i < args.size(); ++i) {
      const std::string& arg = args[i];
      if (arg == "--help" || arg == "-h") {
        std::vector<std::string> help_args{executable};
        return RunVocabTreeBuilder(help_args, out, err);
      }
      if (arg == "--Colite.workspace_path") {
        workspace_path = RequireValue(args, &i, arg);
      } else if (arg == "--output_path") {
        output_path = RequireValue(args, &i, arg);
      } else if (arg == "--Colite.num_visual_words") {
        options.num_visual_words = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.branching") {
        options.branching = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else if (arg == "--Colite.num_iterations") {
        options.num_iterations = controllers::SfmInt(RequireValue(args, &i, arg));
      } else if (arg == "--Colite.vocab_tree_max_num_features") {
        options.max_num_features_per_image = static_cast<std::size_t>(
            controllers::SfmCount(RequireValue(args, &i, arg)));
      } else {
        throw std::invalid_argument("Unknown argument: " + arg);
      }
    }
    if (workspace_path.empty() || output_path.empty()) {
      throw std::invalid_argument(
          "--Colite.workspace_path and --output_path are required");
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
