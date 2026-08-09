#include "colite/controllers/sequential_matcher.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "colite/core/parallel.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/two_view_geometry_batch.h"
#include "colite/matching/match_graph_consistency.h"
#include "colite/matching/weak_pair_rematcher.h"

namespace colite::controllers {
namespace {

std::string MatchingStrategyName(MatchingStrategy strategy) {
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

std::string MatcherBackendName(DescriptorMatcherBackend backend) {
  switch (backend) {
    case DescriptorMatcherBackend::kBruteForce:
      return "bf";
    case DescriptorMatcherBackend::kFlann:
      return "flann";
    case DescriptorMatcherBackend::kCudaBruteForce:
      return "cuda";
  }
  return "unknown";
}

std::string ToConfigValue(double value) {
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string ToConfigValue(std::filesystem::file_time_type value) {
  return std::to_string(value.time_since_epoch().count());
}

std::string CurrentFeatureCacheHash(
    const std::filesystem::path& workspace_path,
    const std::vector<Image>& images) {
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, "features");
  if (!metadata.config_hash.empty() &&
      WorkspaceCacheIsValid(workspace_path, "features",
                            metadata.config_hash)) {
    return metadata.config_hash;
  }

  std::vector<std::pair<std::string, std::string>> entries;
  entries.emplace_back("stage", "feature_files");
  for (const Image& image : images) {
    const std::filesystem::path feature_path =
        WorkspaceFeaturesPath(workspace_path) /
        (std::to_string(image.id) + ".bin");
    entries.emplace_back("feature_" + std::to_string(image.id) + "_path",
                         feature_path.lexically_normal().string());
    if (std::filesystem::exists(feature_path)) {
      entries.emplace_back("feature_" + std::to_string(image.id) + "_size",
                           std::to_string(std::filesystem::file_size(feature_path)));
      entries.emplace_back(
          "feature_" + std::to_string(image.id) + "_mtime",
          ToConfigValue(std::filesystem::last_write_time(feature_path)));
    } else {
      entries.emplace_back("feature_" + std::to_string(image.id) + "_missing",
                           "1");
    }
  }
  return StableConfigHash(entries);
}

std::vector<std::pair<std::string, std::string>> RawMatchCacheEntries(
    const SequentialMatcherControllerOptions& options,
    const std::vector<Image>& images,
    const std::string& feature_hash) {
  return {{"stage", "matches"},
          {"feature_hash", feature_hash},
          {"num_images", std::to_string(images.size())},
          {"matching_strategy", MatchingStrategyName(options.matching_strategy)},
          {"overlap", std::to_string(options.overlap)},
          {"spatial_max_num_neighbors",
           std::to_string(options.spatial_max_num_neighbors)},
          {"spatial_max_distance", ToConfigValue(options.spatial_max_distance)},
          {"vocab_tree_num_images",
           std::to_string(options.vocab_tree_num_images)},
          {"vocab_tree_num_visual_words",
           std::to_string(options.vocab_tree_num_visual_words)},
          {"vocab_tree_branching",
           std::to_string(options.vocab_tree_branching)},
          {"vocab_tree_num_iterations",
           std::to_string(options.vocab_tree_num_iterations)},
          {"vocab_tree_max_num_features",
           std::to_string(options.vocab_tree_max_num_features)},
          {"vocab_tree_sequential_overlap",
           std::to_string(options.vocab_tree_sequential_overlap)},
          {"vocab_tree_path", options.vocab_tree_path.lexically_normal().string()},
          {"matcher_backend", MatcherBackendName(options.matcher_backend)},
          {"max_ratio", ToConfigValue(options.max_ratio)},
          {"mutual_check", options.mutual_check ? "1" : "0"}};
}

std::vector<std::pair<std::string, std::string>> GeometryCacheEntries(
    const SequentialMatcherControllerOptions& options,
    const std::string& raw_match_hash) {
  return {{"stage", "geometry"},
          {"raw_match_hash", raw_match_hash},
          {"guided_matching", options.guided_matching ? "1" : "0"},
          {"guided_match_ratio", ToConfigValue(options.guided_match_ratio)},
          {"guided_min_inlier_gain",
           std::to_string(options.guided_min_inlier_gain)},
          {"keep_degenerate", "1"},
          {"algorithm_version",
           "five_point_adaptive_lo_ransac_consistent_tracks_v4"},
          {"min_num_iterations", "100"},
          {"max_num_iterations", "10000"},
          {"confidence", "0.999"}};
}

std::optional<std::size_t> ReadCacheSizeValue(
    const std::filesystem::path& workspace_path,
    const std::string& stage,
    const std::string& config_hash,
    const std::string& key) {
  if (!WorkspaceCacheIsValid(workspace_path, stage, config_hash)) {
    return std::nullopt;
  }
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, stage);
  const auto it = metadata.values.find(key);
  if (it == metadata.values.end()) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(std::stoull(it->second));
}

}  // namespace

SequentialMatcherController::SequentialMatcherController(
    SequentialMatcherControllerOptions options)
    : options_(std::move(options)) {}

SequentialMatcherControllerSummary SequentialMatcherController::Run() const {
  SequentialMatcherControllerSummary summary;
  const std::vector<Camera> cameras =
      ReadWorkspaceCameras(options_.workspace_path);
  const std::vector<Image> images = ReadWorkspaceImages(options_.workspace_path);
  if (cameras.empty() || images.size() < 2) {
    throw std::runtime_error("Workspace must contain at least two images");
  }

  MatchGraph graph;
  ReadWorkspaceFeatureSets(options_.workspace_path, &graph);

  DescriptorMatchingOptions matching_options;
  matching_options.backend = options_.matcher_backend;
  matching_options.max_ratio = options_.max_ratio;
  matching_options.mutual_check = options_.mutual_check;

  const std::string feature_hash =
      CurrentFeatureCacheHash(options_.workspace_path, images);
  const std::string raw_match_hash =
      StableConfigHash(RawMatchCacheEntries(options_, images, feature_hash));
  const std::string geometry_hash =
      StableConfigHash(GeometryCacheEntries(options_, raw_match_hash));

  MatchingPairGeneratorOptions pair_options;
  pair_options.strategy = options_.matching_strategy;
  pair_options.sequential_overlap = options_.overlap;
  pair_options.spatial_max_num_neighbors = options_.spatial_max_num_neighbors;
  pair_options.spatial_max_distance = options_.spatial_max_distance;
  pair_options.vocab_tree_num_images = options_.vocab_tree_num_images;
  pair_options.vocab_tree_num_visual_words =
      options_.vocab_tree_num_visual_words;
  pair_options.vocab_tree_branching = options_.vocab_tree_branching;
  pair_options.vocab_tree_num_iterations = options_.vocab_tree_num_iterations;
  pair_options.vocab_tree_max_num_features =
      options_.vocab_tree_max_num_features;
  pair_options.vocab_tree_sequential_overlap =
      options_.vocab_tree_sequential_overlap;
  pair_options.vocab_tree_path = options_.vocab_tree_path;

  if (const auto cached_num_pairs = ReadCacheSizeValue(
          options_.workspace_path, "matches", raw_match_hash, "pairs")) {
    ReadWorkspaceMatches(options_.workspace_path, &graph);
    summary.num_pairs = *cached_num_pairs;
    summary.raw_match_cache_hit = true;
    const WorkspaceCacheMetadata metadata =
        ReadWorkspaceCacheMetadata(options_.workspace_path, "matches");
    const auto read_size = [&](const std::string& key) {
      const auto it = metadata.values.find(key);
      return it == metadata.values.end()
                 ? std::size_t{0}
                 : static_cast<std::size_t>(std::stoull(it->second));
    };
    summary.pair_generation.num_retrieval_pairs =
        read_size("retrieval_pairs");
    summary.pair_generation.num_sequential_guard_pairs =
        read_size("sequential_guard_pairs");
    summary.pair_generation.num_sequential_pairs_added =
        read_size("sequential_pairs_added");
    summary.pair_generation.num_connected_components =
        read_size("connected_components");
    summary.pair_generation.min_degree = read_size("min_degree");
    summary.pair_generation.max_degree = read_size("max_degree");
    if (const auto it = metadata.values.find("sequential_pair_recall");
        it != metadata.values.end()) {
      summary.pair_generation.sequential_pair_recall = std::stod(it->second);
    }
    if (const auto it = metadata.values.find("mean_degree");
        it != metadata.values.end()) {
      summary.pair_generation.mean_degree = std::stod(it->second);
    }
  } else {
    const std::vector<ImagePair> pairs =
        GenerateMatchingPairs(images, graph, pair_options,
                              &summary.pair_generation);

    std::map<ImageId, std::size_t> prepared_indices;
    std::vector<std::unique_ptr<PreparedFeatureMatcher>> prepared(images.size());
    for (std::size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
      prepared_indices.emplace(images[image_idx].id, image_idx);
    }
    parallel::parallel_for_(
        parallel::Range(0, static_cast<int>(images.size())),
        [&](const parallel::Range& range) {
          for (int image_idx = range.start; image_idx < range.end; ++image_idx) {
            const FeatureSet* features =
                FindFeatures(graph, images[static_cast<std::size_t>(image_idx)].id);
            if (features == nullptr) {
              throw std::runtime_error("Workspace features are incomplete");
            }
            prepared[static_cast<std::size_t>(image_idx)] =
                std::make_unique<PreparedFeatureMatcher>(
                    *features, matching_options.backend);
          }
        });

    std::vector<std::vector<FeatureMatch>> pair_matches(pairs.size());
    parallel::parallel_for_(
        parallel::Range(0, static_cast<int>(pairs.size())),
        [&](const parallel::Range& range) {
          for (int pair_idx = range.start; pair_idx < range.end; ++pair_idx) {
            const ImagePair& pair = pairs[static_cast<std::size_t>(pair_idx)];
            pair_matches[static_cast<std::size_t>(pair_idx)] =
                MatchPreparedFeatureSets(
                    *prepared.at(prepared_indices.at(pair.image_id1)),
                    *prepared.at(prepared_indices.at(pair.image_id2)),
                    matching_options);
          }
        });
    for (std::size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
      const ImagePair& pair = pairs[pair_idx];
      SetMatches(&graph, pair.image_id1, pair.image_id2,
                 pair_matches[pair_idx]);
    }
    summary.num_pairs = pairs.size();
    WriteWorkspaceMatches(options_.workspace_path, graph);
    WriteWorkspaceCacheMetadata(options_.workspace_path, "matches",
                                raw_match_hash,
                                {{"pairs", std::to_string(summary.num_pairs)},
                                 {"retrieval_pairs", std::to_string(summary.pair_generation.num_retrieval_pairs)},
                                 {"sequential_guard_pairs", std::to_string(summary.pair_generation.num_sequential_guard_pairs)},
                                 {"sequential_pairs_added", std::to_string(summary.pair_generation.num_sequential_pairs_added)},
                                 {"sequential_pair_recall", ToConfigValue(summary.pair_generation.sequential_pair_recall)},
                                 {"connected_components", std::to_string(summary.pair_generation.num_connected_components)},
                                 {"min_degree", std::to_string(summary.pair_generation.min_degree)},
                                 {"max_degree", std::to_string(summary.pair_generation.max_degree)},
                                 {"mean_degree", ToConfigValue(summary.pair_generation.mean_degree)}},
                                {"matches/raw_matches.tsv"});
  }

  std::map<ImageId, Camera> cameras_by_image;
  for (const Image& image : images) {
    const auto camera_it = std::find_if(
        cameras.begin(), cameras.end(),
        [&](const Camera& camera) { return camera.id == image.camera_id; });
    if (camera_it == cameras.end()) {
      throw std::runtime_error("Workspace image references missing camera");
    }
    cameras_by_image.emplace(image.id, *camera_it);
  }
  TwoViewGeometryBatchOptions geometry_options;
  geometry_options.keep_degenerate = true;
  if (WorkspaceCacheIsValid(options_.workspace_path, "geometry",
                            geometry_hash)) {
    ReadWorkspaceVerifiedGeometries(options_.workspace_path, &graph);
    const WorkspaceCacheMetadata metadata =
        ReadWorkspaceCacheMetadata(options_.workspace_path, "geometry");
    summary.num_verified_pairs = static_cast<std::size_t>(
        std::stoull(metadata.values.at("verified_pairs")));
    summary.num_verified_inliers = static_cast<std::size_t>(
        std::stoull(metadata.values.at("verified_inliers")));
    const auto read_guided_size = [&](const std::string& key) {
      const auto it = metadata.values.find(key);
      return it == metadata.values.end()
                 ? std::size_t{0}
                 : static_cast<std::size_t>(std::stoull(it->second));
    };
    summary.guided_matching.num_checked_pairs =
        read_guided_size("guided_checked_pairs");
    summary.guided_matching.num_rematched_pairs =
        read_guided_size("guided_rematched_pairs");
    summary.guided_matching.num_improved_pairs =
        read_guided_size("guided_improved_pairs");
    summary.geometry_cache_hit = true;
  } else {
    VerifyMatchGraphGeometries(cameras_by_image, &graph, geometry_options);
    if (options_.guided_matching) {
      WeakPairRematchingOptions guided_options;
      guided_options.nearby_image_window = 0;
      guided_options.min_inlier_gain = options_.guided_min_inlier_gain;
      guided_options.relaxed_matching = matching_options;
      guided_options.relaxed_matching.max_ratio = options_.guided_match_ratio;
      guided_options.relaxed_matching.mutual_check = true;
      guided_options.guided_matching.max_sampson_error_px =
          geometry_options.geometry_options.max_error_px;
      guided_options.verification = geometry_options.geometry_options;
      summary.guided_matching =
          RematchWeakImagePairs(cameras_by_image, guided_options, &graph);
    }
    EnforceVerifiedMatchGraphConsistency(&graph);
    WriteWorkspaceVerifiedGeometries(options_.workspace_path, graph);

    for (const auto& [pair, geometry] : graph.verified_geometries) {
      (void)pair;
      if (geometry.type != TwoViewGeometry::Type::kUndefined &&
          geometry.type != TwoViewGeometry::Type::kDegenerate &&
          !geometry.inlier_matches.empty()) {
        ++summary.num_verified_pairs;
        summary.num_verified_inliers += geometry.inlier_matches.size();
      }
    }
    WriteWorkspaceCacheMetadata(
        options_.workspace_path, "geometry", geometry_hash,
        {{"verified_pairs", std::to_string(summary.num_verified_pairs)},
         {"verified_inliers", std::to_string(summary.num_verified_inliers)},
         {"guided_checked_pairs", std::to_string(
              summary.guided_matching.num_checked_pairs)},
         {"guided_rematched_pairs", std::to_string(
              summary.guided_matching.num_rematched_pairs)},
         {"guided_improved_pairs", std::to_string(
              summary.guided_matching.num_improved_pairs)}},
        {"matches/verified_matches.tsv"});
  }

  std::filesystem::create_directories(
      WorkspaceMatchesPath(options_.workspace_path));
  std::ofstream summary_file(WorkspaceMatchesPath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "pairs " << summary.num_pairs << "\n"
               << "verified_pairs " << summary.num_verified_pairs << "\n"
               << "verified_inliers " << summary.num_verified_inliers
               << "\nguided_checked_pairs "
               << summary.guided_matching.num_checked_pairs
               << "\nguided_rematched_pairs "
               << summary.guided_matching.num_rematched_pairs
               << "\nguided_improved_pairs "
               << summary.guided_matching.num_improved_pairs
               << "\nretrieval_pairs "
               << summary.pair_generation.num_retrieval_pairs
               << "\nsequential_guard_pairs "
               << summary.pair_generation.num_sequential_guard_pairs
               << "\nsequential_pairs_added "
               << summary.pair_generation.num_sequential_pairs_added
               << "\nsequential_pair_recall "
               << summary.pair_generation.sequential_pair_recall
               << "\nconnected_components "
               << summary.pair_generation.num_connected_components
               << "\nmin_degree " << summary.pair_generation.min_degree
               << "\nmax_degree " << summary.pair_generation.max_degree
               << "\nmean_degree " << summary.pair_generation.mean_degree
               << "\n";

  summary.success = true;
  summary.num_images = images.size();
  return summary;
}

}  // namespace colite::controllers
