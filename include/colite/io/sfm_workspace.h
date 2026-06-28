#pragma once

#include <filesystem>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/image.h"
#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

std::filesystem::path WorkspaceFeaturesPath(
    const std::filesystem::path& workspace_path);
std::filesystem::path WorkspaceMatchesPath(
    const std::filesystem::path& workspace_path);
std::filesystem::path WorkspaceSparsePath(
    const std::filesystem::path& workspace_path);

void WriteWorkspaceCameras(const std::filesystem::path& workspace_path,
                           const std::vector<Camera>& cameras);
std::vector<Camera> ReadWorkspaceCameras(
    const std::filesystem::path& workspace_path);

void WriteWorkspaceImages(const std::filesystem::path& workspace_path,
                          const std::vector<Image>& images);
std::vector<Image> ReadWorkspaceImages(
    const std::filesystem::path& workspace_path);

void WriteWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                              ImageId image_id,
                              const FeatureSet& features);
FeatureSet ReadWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                                   ImageId image_id);
void ReadWorkspaceFeatureSets(const std::filesystem::path& workspace_path,
                              MatchGraph* graph);

void WriteWorkspaceMatches(const std::filesystem::path& workspace_path,
                           const MatchGraph& graph);
void ReadWorkspaceMatches(const std::filesystem::path& workspace_path,
                          MatchGraph* graph);

void WriteWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path,
    const MatchGraph& graph);
void ReadWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path,
    MatchGraph* graph);

void WriteWorkspaceSparseReconstruction(
    const std::filesystem::path& workspace_path,
    const Reconstruction& reconstruction);

}  // namespace colite
