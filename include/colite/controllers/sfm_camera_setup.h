#pragma once

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/image.h"
#include "colite/io/camera_prior.h"

namespace colite::controllers {

enum class CameraSharingMode {
  kAuto,
  kSingle,
  kPerFolder,
  kPerImage,
  kShared = kSingle,
  kIndividual = kPerImage,
};

CameraSharingMode CameraSharingModeFromName(std::string_view name);
std::string_view CameraSharingModeName(CameraSharingMode mode);

struct SfmCameraSetupOptions {
  CameraSharingMode sharing_mode = CameraSharingMode::kAuto;
  CameraModelId model_id = CameraModelId::kSimpleRadial;
  double focal_length = 0.0;
  std::vector<double> camera_params;
  std::filesystem::path image_root;
  double default_focal_length_factor = 1.2;
};

struct SfmCameraSetup {
  std::vector<Camera> cameras;
  std::vector<Image> images;
  std::vector<InitialFocalLengthEstimate> focal_estimates;
  std::optional<GeodeticCoordinate> gps_origin;
  std::size_t num_position_priors = 0;
};

std::vector<double> ParseCameraParams(std::string_view value);

SfmCameraSetup BuildSfmCameraSetup(
    const std::vector<std::filesystem::path>& image_paths,
    const SfmCameraSetupOptions& options);

}  // namespace colite::controllers
