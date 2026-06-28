#include "colite/controllers/mapper.h"

#include <fstream>
#include <stdexcept>

#include "colite/io/sfm_workspace.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

MapperController::MapperController(MapperControllerOptions options)
    : options_(std::move(options)) {}

MapperControllerSummary MapperController::Run() const {
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
  std::ofstream summary_file(WorkspaceSparsePath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "success " << (mapping_summary.success ? 1 : 0) << "\n"
               << "registered_images "
               << mapping_summary.num_registered_images << "\n"
               << "points3d " << mapping_summary.num_points3d << "\n"
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
  return summary;
}

}  // namespace colite::controllers
