#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <utility>
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

struct WorkspaceCacheMetadata {
  std::string stage;
  std::string config_hash;
  std::map<std::string, std::string> values;
  std::vector<std::filesystem::path> required_files;
};

std::string StableConfigHash(
    const std::vector<std::pair<std::string, std::string>>& entries);

std::filesystem::path WorkspaceCacheMetadataPath(
    const std::filesystem::path& workspace_path,
    const std::string& stage);

void WriteWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage,
    const std::string& config_hash,
    const std::map<std::string, std::string>& values,
    const std::vector<std::filesystem::path>& required_files);

WorkspaceCacheMetadata ReadWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage);

bool WorkspaceCacheIsValid(const std::filesystem::path& workspace_path,
                           const std::string& stage,
                           const std::string& config_hash);

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
