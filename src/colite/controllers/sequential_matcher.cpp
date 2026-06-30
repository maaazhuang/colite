#include "colite/controllers/sequential_matcher.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

#include "colite/io/sfm_workspace.h"
#include "colite/matching/two_view_geometry_batch.h"

namespace colite::controllers {

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
