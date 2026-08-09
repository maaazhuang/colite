#include "colite/controllers/sfm_camera_setup.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace colite::controllers {
namespace {

struct ImageMetadata {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  InitialFocalLengthEstimate focal;
  ExifCameraMetadata exif;
};

ImageMetadata ReadImageMetadata(const std::filesystem::path& image_path,
                                const SfmCameraSetupOptions& options) {
  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }
  ImageMetadata metadata;
  metadata.width = static_cast<std::uint32_t>(image.cols);
  metadata.height = static_cast<std::uint32_t>(image.rows);
  metadata.focal = EstimateInitialFocalLengthPriorFromImage(
      image_path, options.focal_length, options.default_focal_length_factor);
  metadata.exif = ReadExifCameraMetadata(image_path);
  return metadata;
}

std::filesystem::path CommonImageRoot(
    const std::vector<std::filesystem::path>& image_paths) {
  std::filesystem::path common =
      std::filesystem::absolute(image_paths.front()).parent_path();
  for (std::size_t idx = 1; idx < image_paths.size(); ++idx) {
    const std::filesystem::path parent =
        std::filesystem::absolute(image_paths[idx]).parent_path();
    std::filesystem::path candidate;
    auto common_it = common.begin();
    auto parent_it = parent.begin();
    while (common_it != common.end() && parent_it != parent.end() &&
           *common_it == *parent_it) {
      candidate /= *common_it;
      ++common_it;
      ++parent_it;
    }
    common = candidate;
  }
  return common;
}

std::string RelativeImageName(const std::filesystem::path& image_path,
                              const std::filesystem::path& image_root) {
  std::error_code error;
  const std::filesystem::path relative =
      std::filesystem::relative(image_path, image_root, error);
  if (error || relative.empty() ||
      (!relative.empty() && *relative.begin() == "..")) {
    return image_path.filename().generic_string();
  }
  return relative.generic_string();
}

std::string AutoCameraKey(const ImageMetadata& metadata,
                          std::size_t image_idx) {
  if (!metadata.exif.HasAutoCameraIdentity()) {
    return "image:" + std::to_string(image_idx);
  }
  const double exif_focal = metadata.exif.focal_length_35mm.value_or(
      *metadata.exif.focal_length);
  std::ostringstream key;
  key << "exif:" << *metadata.exif.make << '\n' << *metadata.exif.model
      << '\n' << std::setprecision(17) << exif_focal << '\n'
      << metadata.width << 'x' << metadata.height;
  return key.str();
}

double Median(std::vector<double> values) {
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  const double upper = values[middle];
  if (values.size() % 2 != 0) return upper;
  const double lower =
      *std::max_element(values.begin(), values.begin() + middle);
  return 0.5 * (lower + upper);
}

Camera MakeCamera(CameraId camera_id,
                  const SfmCameraSetupOptions& options,
                  const ImageMetadata& metadata,
                  double focal_length,
                  bool has_prior) {
  Camera camera;
  camera.id = camera_id;
  camera.model_id = options.model_id;
  camera.width = metadata.width;
  camera.height = metadata.height;
  camera.params =
      options.camera_params.empty()
          ? InitializeCameraParams(options.model_id, focal_length,
                                   camera.width, camera.height)
          : options.camera_params;
  camera.has_prior_focal_length =
      has_prior || !options.camera_params.empty();
  if (!camera.VerifyParams()) {
    throw std::invalid_argument(
        "Camera parameters are invalid for model " +
        std::string(CameraModelIdToName(options.model_id)));
  }
  return camera;
}

}  // namespace

CameraSharingMode CameraSharingModeFromName(const std::string_view name) {
  if (name == "auto") return CameraSharingMode::kAuto;
  if (name == "single" || name == "shared") {
    return CameraSharingMode::kSingle;
  }
  if (name == "per_folder") return CameraSharingMode::kPerFolder;
  if (name == "per_image" || name == "individual") {
    return CameraSharingMode::kPerImage;
  }
  throw std::invalid_argument("Unsupported camera mode: " +
                              std::string(name));
}

std::string_view CameraSharingModeName(const CameraSharingMode mode) {
  switch (mode) {
    case CameraSharingMode::kAuto: return "auto";
    case CameraSharingMode::kSingle: return "single";
    case CameraSharingMode::kPerFolder: return "per_folder";
    case CameraSharingMode::kPerImage: return "per_image";
  }
  throw std::invalid_argument("Unknown camera sharing mode");
}

std::vector<double> ParseCameraParams(const std::string_view value) {
  std::vector<double> params;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end =
        comma == std::string_view::npos ? value.size() : comma;
    if (end == start) {
      throw std::invalid_argument("Camera parameter list contains an empty value");
    }
    const std::string token(value.substr(start, end - start));
    std::size_t parsed = 0;
    const double parameter = std::stod(token, &parsed);
    if (parsed != token.size() || !std::isfinite(parameter)) {
      throw std::invalid_argument("Camera parameters must be finite numbers");
    }
    params.push_back(parameter);
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return params;
}

SfmCameraSetup BuildSfmCameraSetup(
    const std::vector<std::filesystem::path>& image_paths,
    const SfmCameraSetupOptions& options) {
  if (image_paths.empty()) {
    throw std::invalid_argument("Cannot create cameras without input images");
  }
  if (!CameraModelProjectionImplemented(options.model_id) ||
      !CameraModelUnprojectionImplemented(options.model_id)) {
    throw std::invalid_argument("Camera model is not fully implemented");
  }
  if (!std::isfinite(options.focal_length) || options.focal_length < 0.0 ||
      !std::isfinite(options.default_focal_length_factor) ||
      options.default_focal_length_factor <= 0.0) {
    throw std::invalid_argument("Invalid focal length configuration");
  }
  const std::size_t expected_params =
      GetCameraModelSpec(options.model_id).param_names.size();
  if (!options.camera_params.empty() &&
      options.camera_params.size() != expected_params) {
    throw std::invalid_argument(
        "--camera_params expects " + std::to_string(expected_params) +
        " values for " + std::string(CameraModelIdToName(options.model_id)));
  }

  std::vector<ImageMetadata> metadata;
  metadata.reserve(image_paths.size());
  SfmCameraSetup setup;
  setup.focal_estimates.reserve(image_paths.size());
  for (const std::filesystem::path& image_path : image_paths) {
    metadata.push_back(ReadImageMetadata(image_path, options));
    setup.focal_estimates.push_back(metadata.back().focal);
  }

  const std::filesystem::path image_root =
      options.image_root.empty() ? CommonImageRoot(image_paths)
                                 : std::filesystem::absolute(options.image_root);
  std::map<std::string, std::vector<std::size_t>> groups;
  for (std::size_t idx = 0; idx < image_paths.size(); ++idx) {
    std::string key;
    switch (options.sharing_mode) {
      case CameraSharingMode::kAuto:
        key = AutoCameraKey(metadata[idx], idx);
        break;
      case CameraSharingMode::kSingle:
        key = "single";
        break;
      case CameraSharingMode::kPerFolder: {
        const std::string name = RelativeImageName(image_paths[idx], image_root);
        key = std::filesystem::path(name).parent_path().generic_string();
        break;
      }
      case CameraSharingMode::kPerImage:
        key = "image:" + std::to_string(idx);
        break;
    }
    groups[key].push_back(idx);
  }

  std::vector<CameraId> camera_ids(image_paths.size(), kInvalidCameraId);
  setup.cameras.reserve(groups.size());
  CameraId next_camera_id = 1;
  for (const auto& [key, indices] : groups) {
    (void)key;
    const ImageMetadata& first = metadata[indices.front()];
    for (const std::size_t idx : indices) {
      if (metadata[idx].width != first.width ||
          metadata[idx].height != first.height) {
        throw std::runtime_error(
            "Shared camera group requires identical image dimensions; use "
            "--camera_mode per_image or separate images by folder");
      }
    }

    double focal = first.focal.pixels;
    bool has_prior = options.focal_length > 0.0;
    if (options.camera_params.empty() && options.focal_length <= 0.0) {
      std::vector<double> prior_focals;
      for (const std::size_t idx : indices) {
        if (metadata[idx].focal.has_prior) {
          prior_focals.push_back(metadata[idx].focal.pixels);
        }
      }
      if (!prior_focals.empty()) {
        focal = Median(prior_focals);
        has_prior = true;
        const auto [minimum, maximum] =
            std::minmax_element(prior_focals.begin(), prior_focals.end());
        if ((*maximum - *minimum) / focal > 0.05) {
          throw std::runtime_error(
              "Camera group contains conflicting EXIF focal priors; use "
              "--camera_mode per_image or provide --camera_params");
        }
      }
    }

    setup.cameras.push_back(
        MakeCamera(next_camera_id, options, first, focal, has_prior));
    for (const std::size_t idx : indices) camera_ids[idx] = next_camera_id;
    ++next_camera_id;
  }

  setup.images.reserve(image_paths.size());
  for (std::size_t idx = 0; idx < image_paths.size(); ++idx) {
    Image image;
    image.id = static_cast<ImageId>(idx + 1);
    image.camera_id = camera_ids[idx];
    image.name = RelativeImageName(image_paths[idx], image_root);
    image.file_path = image_paths[idx].string();
    setup.images.push_back(std::move(image));
  }
  return setup;
}

}  // namespace colite::controllers
