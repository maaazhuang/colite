#include "colite/controllers/mapper.h"

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/io/sfm_workspace.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {
namespace {

std::string CurrentStageHash(const std::filesystem::path& workspace_path,
                             const std::string& stage) {
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, stage);
  if (!metadata.config_hash.empty() &&
      WorkspaceCacheIsValid(workspace_path, stage, metadata.config_hash)) {
    return metadata.config_hash;
  }
  return "uncached_" + stage;
}

std::string MapperConfigHash(const std::filesystem::path& workspace_path) {
  return StableConfigHash(
      {{"stage", "sparse"},
       {"geometry_hash", CurrentStageHash(workspace_path, "geometry")},
       {"min_pair_median_triangulation_angle_deg", "1"},
       {"local_retriangulation_min_angle_deg", "1"}});
}

std::optional<MapperControllerSummary> ReadCachedMapperSummary(
    const std::filesystem::path& workspace_path,
    const std::string& config_hash) {
  if (!WorkspaceCacheIsValid(workspace_path, "sparse", config_hash)) {
    return std::nullopt;
  }
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, "sparse");
  MapperControllerSummary summary;
  summary.success = metadata.values.at("success") == "1";
  summary.cache_hit = true;
  summary.num_registered_images = static_cast<std::size_t>(
      std::stoull(metadata.values.at("registered_images")));
  summary.num_points3d =
      static_cast<std::size_t>(std::stoull(metadata.values.at("points3d")));
  return summary;
}

std::vector<std::filesystem::path> SparseRequiredFiles() {
  return {"sparse/cameras.txt",
          "sparse/images.txt",
          "sparse/points3D.txt",
          "sparse/points3D.ply",
          "sparse/summary.txt"};
}

}  // namespace

MapperController::MapperController(MapperControllerOptions options)
    : options_(std::move(options)) {}

MapperControllerSummary MapperController::Run() const {
  const std::string config_hash = MapperConfigHash(options_.workspace_path);
  if (const auto cached_summary =
          ReadCachedMapperSummary(options_.workspace_path, config_hash)) {
    return *cached_summary;
  }

  const std::vector<Camera> cameras =
      ReadWorkspaceCameras(options_.workspace_path);
  const std::vector<Image> images = ReadWorkspaceImages(options_.workspace_path);
  if (cameras.empty() || images.size() < 2) {
    throw std::runtime_error("Workspace must contain cameras and images");
  }

  MatchGraph graph;
  ReadWorkspaceFeatureSets(options_.workspace_path, &graph);
  ReadWorkspaceMatches(options_.workspace_path, &graph);
  ReadWorkspaceVerifiedGeometries(options_.workspace_path, &graph);

  Reconstruction reconstruction;
  for (Camera camera : cameras) {
    reconstruction.AddCamera(std::move(camera));
  }
  for (Image image : images) {
    reconstruction.AddImage(std::move(image));
  }

  IncrementalMappingOptions mapping_options;
  mapping_options.mapper.triangulation.min_pair_median_triangulation_angle_deg =
      1.0;
  mapping_options.mapper.local_retriangulation_filter
      .min_triangulation_angle_deg = 1.0;
  const IncrementalMappingSummary mapping_summary =
      RunIncrementalMapping(graph, mapping_options, &reconstruction);

  WriteWorkspaceSparseReconstruction(options_.workspace_path, reconstruction);
  WriteRegistrationDiagnostics(WorkspaceSparsePath(options_.workspace_path),
                               mapping_summary);
  std::ofstream summary_file(WorkspaceSparsePath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "success " << (mapping_summary.success ? 1 : 0) << "\n"
               << "registered_images "
               << mapping_summary.num_registered_images << "\n"
               << "points3d " << mapping_summary.num_points3d << "\n"
               << "initialization_attempts "
               << mapping_summary.initialization.num_initialization_attempts
               << "\n"
               << "message " << mapping_summary.message << "\n";
  if (!mapping_summary.initialization.message.empty()) {
    summary_file << "initialization_message "
                 << mapping_summary.initialization.message << "\n"
                 << "initialization_pair "
                 << mapping_summary.initialization.initial_pair.image_id1
                 << " "
                 << mapping_summary.initialization.initial_pair.image_id2
                 << "\n"
                 << "initialization_points "
                 << mapping_summary.initialization.num_triangulated_points
                 << "\n";
  }

  MapperControllerSummary summary;
  summary.success = mapping_summary.success;
  summary.num_registered_images = mapping_summary.num_registered_images;
  summary.num_points3d = mapping_summary.num_points3d;
  WriteWorkspaceCacheMetadata(
      options_.workspace_path, "sparse", config_hash,
      {{"success", mapping_summary.success ? "1" : "0"},
       {"registered_images", std::to_string(mapping_summary.num_registered_images)},
       {"points3d", std::to_string(mapping_summary.num_points3d)}},
      SparseRequiredFiles());
  return summary;
}

}  // namespace colite::controllers
