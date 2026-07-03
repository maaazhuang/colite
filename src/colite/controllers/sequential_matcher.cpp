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

#include "colite/io/sfm_workspace.h"
#include "colite/matching/two_view_geometry_batch.h"

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
          {"vocab_tree_max_num_features",
           std::to_string(options.vocab_tree_max_num_features)},
          {"vocab_tree_path", options.vocab_tree_path.lexically_normal().string()},
          {"matcher_backend", MatcherBackendName(options.matcher_backend)},
          {"max_ratio", ToConfigValue(options.max_ratio)},
          {"mutual_check", options.mutual_check ? "1" : "0"}};
}

std::vector<std::pair<std::string, std::string>> GeometryCacheEntries(
    const std::string& raw_match_hash) {
  return {{"stage", "geometry"},
          {"raw_match_hash", raw_match_hash},
          {"keep_degenerate", "1"}};
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
      StableConfigHash(GeometryCacheEntries(raw_match_hash));

  MatchingPairGeneratorOptions pair_options;
  pair_options.strategy = options_.matching_strategy;
  pair_options.sequential_overlap = options_.overlap;
  pair_options.spatial_max_num_neighbors = options_.spatial_max_num_neighbors;
  pair_options.spatial_max_distance = options_.spatial_max_distance;
  pair_options.vocab_tree_num_images = options_.vocab_tree_num_images;
  pair_options.vocab_tree_num_visual_words =
      options_.vocab_tree_num_visual_words;
  pair_options.vocab_tree_max_num_features =
      options_.vocab_tree_max_num_features;
  pair_options.vocab_tree_path = options_.vocab_tree_path;

  if (const auto cached_num_pairs = ReadCacheSizeValue(
          options_.workspace_path, "matches", raw_match_hash, "pairs")) {
    ReadWorkspaceMatches(options_.workspace_path, &graph);
    summary.num_pairs = *cached_num_pairs;
    summary.raw_match_cache_hit = true;
  } else {
    const std::vector<ImagePair> pairs =
        GenerateMatchingPairs(images, graph, pair_options);

    for (const ImagePair& pair : pairs) {
      const FeatureSet* features1 = FindFeatures(graph, pair.image_id1);
      const FeatureSet* features2 = FindFeatures(graph, pair.image_id2);
      if (features1 == nullptr || features2 == nullptr) {
        throw std::runtime_error("Workspace features are incomplete");
      }
      SetMatches(&graph, pair.image_id1, pair.image_id2,
                 MatchFeatureSets(*features1, *features2, matching_options));
      ++summary.num_pairs;
    }
    WriteWorkspaceMatches(options_.workspace_path, graph);
    WriteWorkspaceCacheMetadata(options_.workspace_path, "matches",
                                raw_match_hash,
                                {{"pairs", std::to_string(summary.num_pairs)}},
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
    summary.geometry_cache_hit = true;
  } else {
    VerifyMatchGraphGeometries(cameras_by_image, &graph, geometry_options);
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
         {"verified_inliers", std::to_string(summary.num_verified_inliers)}},
        {"matches/verified_matches.tsv"});
  }

  std::filesystem::create_directories(
      WorkspaceMatchesPath(options_.workspace_path));
  std::ofstream summary_file(WorkspaceMatchesPath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "pairs " << summary.num_pairs << "\n"
               << "verified_pairs " << summary.num_verified_pairs << "\n"
               << "verified_inliers " << summary.num_verified_inliers
               << "\n";

  summary.success = true;
  summary.num_images = images.size();
  return summary;
}

}  // namespace colite::controllers
