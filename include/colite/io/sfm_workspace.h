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

// COLMAP sparse-model storage. kAuto is valid for reading only and prefers a
// complete binary model when both representations are present.
enum class ColmapModelFormat {
  kAuto,
  kText,
  kBinary,
  kTextAndBinary,
};

ColmapModelFormat ColmapModelFormatFromString(const std::string& value);
std::string ColmapModelFormatToString(ColmapModelFormat format);
ColmapModelFormat DetectColmapModelFormat(
    const std::filesystem::path& model_path);

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
    const Reconstruction& reconstruction,
    ColmapModelFormat format = ColmapModelFormat::kText);

// COLMAP text model I/O shared by SfM resume, multi-model output and MVS.
Reconstruction ReadColmapTextReconstruction(
    const std::filesystem::path& model_path);
Reconstruction ReadColmapBinaryReconstruction(
    const std::filesystem::path& model_path);
Reconstruction ReadColmapReconstruction(
    const std::filesystem::path& model_path,
    ColmapModelFormat format = ColmapModelFormat::kAuto);
void WriteSparseReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction);
void WriteColmapBinaryReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction);
void WriteColmapReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction,
    ColmapModelFormat format);

// Create a reconstruction containing every workspace camera/image. When an
// input model is supplied, its optimized cameras, registered poses, points and
// tracks are overlaid after remapping image IDs by name and validating camera
// compatibility against the workspace.
Reconstruction ReadWorkspaceReconstruction(
    const std::filesystem::path& workspace_path,
    const std::filesystem::path& input_model_path = {});

Reconstruction ApplyColmapTextModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path);
Reconstruction ApplyColmapModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path,
    ColmapModelFormat format = ColmapModelFormat::kAuto);

}  // namespace colite
