#include "colite/features/sift_extractor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "backends/cuda_sift_extractor.h"
#include "backends/sift_opencv.hpp"
#include "sift_descriptor_conversion.h"

extern "C" {
#include "backends/sift_vlfeat.h"
}

namespace colite {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDefaultOpenCvContrastThreshold = 0.04;

std::string LowerExtension(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext;
}

bool IsSupportedImageExtension(const std::filesystem::path& path) {
  const std::string ext = LowerExtension(path);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
         ext == ".bmp" || ext == ".tif" || ext == ".tiff" ||
         ext == ".webp";
}

cv::Mat ToGrayImage(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }
  if (image.channels() == 1) {
    return image;
  }

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

std::vector<float> GrayToFloatVector(const cv::Mat& gray,
                                     const double scale = 1.0) {
  cv::Mat gray_float;
  gray.convertTo(gray_float, CV_32F, scale);

  std::vector<float> result;
  result.reserve(static_cast<std::size_t>(gray_float.rows * gray_float.cols));
  for (int y = 0; y < gray_float.rows; ++y) {
    const float* row = gray_float.ptr<float>(y);
    result.insert(result.end(), row, row + gray_float.cols);
  }
  return result;
}

double EffectiveVlSiftPeakThreshold(const SiftExtractionOptions& options) {
  // COLMAP CPU SIFT 的 peak_threshold 默认是 0.02 / octave_resolution。
  if (std::abs(options.contrast_threshold -
               kDefaultOpenCvContrastThreshold) < 1e-12) {
    return 0.02 / static_cast<double>(options.num_octave_layers);
  }
  return options.contrast_threshold;
}

void TransformVlFeatDescriptorToUbc(float* descriptor) {
  std::array<float, VL_SIFT_DESCR_LEN> vlfeat_descriptor = {};
  std::copy(descriptor, descriptor + VL_SIFT_DESCR_LEN,
            vlfeat_descriptor.begin());

  const std::array<int, 8> orientation_permutation{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 8; ++k) {
        descriptor[8 * (j + 4 * i) + orientation_permutation[k]] =
            vlfeat_descriptor[8 * (j + 4 * i) + k];
      }
    }
  }
}

struct VlSiftLevelFeatures {
  int dog_level = -1;
  int num_base_keypoints = 0;
  std::vector<FeatureKeypoint> keypoints;
  std::vector<float> descriptors;
};

FeatureSet ExtractCvSiftFeatures(const cv::Mat& gray,
                                 const SiftExtractionOptions& options) {
  const std::vector<float> image = GrayToFloatVector(gray);
  features::backends::Sift sift =
      features::backends::Sift::create(options.max_num_features,
                                       options.num_octave_layers,
                                       options.contrast_threshold,
                                       options.edge_threshold,
                                       options.sigma);
  std::vector<features::backends::KeyPoint> keypoints;
  std::vector<float> descriptors;
  sift.detectAndCompute(image.data(), gray.cols, gray.rows, keypoints,
                        descriptors);

  FeatureSet result;
  result.keypoints.reserve(keypoints.size());
  for (const features::backends::KeyPoint& keypoint : keypoints) {
    FeatureKeypoint converted;
    converted.x = keypoint.x;
    converted.y = keypoint.y;
    converted.scale = keypoint.size;
    converted.orientation = keypoint.angle;
    converted.response = keypoint.response;
    result.keypoints.push_back(converted);
  }
  if (descriptors.size() != keypoints.size() * 128) {
    throw std::runtime_error("SIFT descriptor buffer has unexpected size.");
  }
  result.descriptors = features::ConvertSiftDescriptors(
      descriptors.data(), keypoints.size(), options.use_root_sift);
  return result;
}

FeatureSet ExtractVlSiftFeatures(const cv::Mat& gray,
                                 const SiftExtractionOptions& options) {
  const std::vector<float> image = GrayToFloatVector(gray, 1.0 / 255.0);
  const int first_octave = options.upsample_first_octave ? -1 : 0;
  VlSiftFilt* filter =
      vl_sift_new(gray.cols, gray.rows, options.vl_num_octaves,
                  options.num_octave_layers, first_octave);
  if (filter == nullptr) {
    throw std::runtime_error("Failed to create VL SIFT filter.");
  }
  vl_sift_set_peak_thresh(filter, EffectiveVlSiftPeakThreshold(options));
  vl_sift_set_edge_thresh(filter, options.edge_threshold);

  std::vector<VlSiftLevelFeatures> levels;
  int err = vl_sift_process_first_octave(filter, image.data());
  while (err == VL_ERR_OK) {
    vl_sift_detect(filter);
    const VlSiftKeypoint* keypoints = vl_sift_get_keypoints(filter);
    const int num_keypoints = vl_sift_get_nkeypoints(filter);
    int previous_level = -1;
    VlSiftLevelFeatures* current_level = nullptr;
    for (int i = 0; i < num_keypoints; ++i) {
      if (keypoints[i].is != previous_level) {
        levels.push_back({});
        current_level = &levels.back();
        current_level->dog_level = keypoints[i].is;
        previous_level = keypoints[i].is;
      }
      if (current_level == nullptr) {
        throw std::runtime_error("Invalid VL SIFT level state.");
      }
      ++current_level->num_base_keypoints;

      double angles[4] = {};
      const int num_orientations =
          vl_sift_calc_keypoint_orientations(filter, angles, keypoints + i);
      const int num_used_orientations =
          std::min(num_orientations, options.max_num_orientations);
      for (int orientation_idx = 0; orientation_idx < num_used_orientations;
           ++orientation_idx) {
        FeatureKeypoint converted;
        converted.x = keypoints[i].x + 0.5F;
        converted.y = keypoints[i].y + 0.5F;
        converted.scale = keypoints[i].sigma;
        converted.response = 0.0F;
        converted.orientation =
            static_cast<float>(angles[orientation_idx] * 180.0 / kPi);
        if (converted.orientation < 0.0F) {
          converted.orientation += 360.0F;
        }
        current_level->keypoints.push_back(converted);

        const std::size_t offset = current_level->descriptors.size();
        current_level->descriptors.resize(offset + VL_SIFT_DESCR_LEN);
        float* descriptor = current_level->descriptors.data() + offset;
        vl_sift_calc_keypoint_descriptor(filter, descriptor, keypoints + i,
                                         angles[orientation_idx]);
        TransformVlFeatDescriptorToUbc(descriptor);
      }
    }
    err = vl_sift_process_next_octave(filter);
  }

  vl_sift_delete(filter);

  std::size_t first_level_to_keep = 0;
  if (options.max_num_features > 0) {
    int num_features = 0;
    for (std::size_t reverse_idx = levels.size(); reverse_idx > 0;
         --reverse_idx) {
      const std::size_t level_idx = reverse_idx - 1;
      num_features += levels[level_idx].num_base_keypoints;
      if (num_features > options.max_num_features) {
        first_level_to_keep = level_idx;
        break;
      }
    }
  }

  FeatureSet result;
  std::vector<float> descriptors;
  for (std::size_t level_idx = first_level_to_keep; level_idx < levels.size();
       ++level_idx) {
    result.keypoints.insert(result.keypoints.end(),
                            levels[level_idx].keypoints.begin(),
                            levels[level_idx].keypoints.end());
    descriptors.insert(descriptors.end(), levels[level_idx].descriptors.begin(),
                       levels[level_idx].descriptors.end());
  }
  if (descriptors.size() != result.keypoints.size() * 128) {
    throw std::runtime_error("SIFT descriptor buffer has unexpected size.");
  }
  result.descriptors = features::ConvertSiftDescriptors(
      descriptors.data(), result.keypoints.size(), options.use_root_sift);
  return result;
}

}  // namespace

const char* SiftBackendName(const SiftBackend backend) {
  switch (backend) {
    case SiftBackend::kVlSift:
      return "vlsift";
    case SiftBackend::kCvSift:
      return "cvsift";
    case SiftBackend::kCudaSift:
      return "cudasift";
  }
  return "unknown";
}

SiftBackend SiftBackendFromName(std::string_view name) {
  std::string normalized(name);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });

  if (normalized == "vlsift" || normalized == "vl_sift" ||
      normalized == "vlfeat") {
    return SiftBackend::kVlSift;
  }
  if (normalized == "cvsift" || normalized == "cv_sift") {
    return SiftBackend::kCvSift;
  }
  if (normalized == "cudasift" || normalized == "cuda_sift" ||
      normalized == "cuda") {
    return SiftBackend::kCudaSift;
  }
  throw std::invalid_argument("Unsupported SIFT backend: " + normalized);
}

FeatureSet ExtractSiftFeatures(const cv::Mat& image,
                               const SiftExtractionOptions& options) {
  const cv::Mat gray = ToGrayImage(image);
  switch (options.backend) {
    case SiftBackend::kVlSift:
      return ExtractVlSiftFeatures(gray, options);
    case SiftBackend::kCvSift:
      return ExtractCvSiftFeatures(gray, options);
    case SiftBackend::kCudaSift:
      return ExtractCudaSiftFeatures(gray, options);
  }
  throw std::invalid_argument("Unsupported SIFT backend.");
}

FeatureSet ExtractSiftFeaturesFromFile(const std::filesystem::path& image_path,
                                       const SiftExtractionOptions& options) {
  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }
  return ExtractSiftFeatures(image, options);
}

std::vector<std::filesystem::path> ListImageFiles(
    const std::filesystem::path& image_dir) {
  if (!std::filesystem::is_directory(image_dir)) {
    throw std::invalid_argument("Image path is not a directory.");
  }

  std::vector<std::filesystem::path> image_paths;
  for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    if (entry.is_regular_file() && IsSupportedImageExtension(entry.path())) {
      image_paths.push_back(entry.path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  return image_paths;
}

std::map<std::filesystem::path, FeatureSet> ExtractSiftFeaturesFromDirectory(
    const std::filesystem::path& image_dir,
    const SiftExtractionOptions& options) {
  std::map<std::filesystem::path, FeatureSet> result;
  for (const std::filesystem::path& image_path : ListImageFiles(image_dir)) {
    result.emplace(image_path, ExtractSiftFeaturesFromFile(image_path, options));
  }
  return result;
}

}  // namespace colite
