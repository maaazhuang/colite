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

#include "colite/core/parallel.h"
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
#if defined(COLITE_ENABLE_AVX2)
constexpr bool kUseAvx2 = true;
#else
constexpr bool kUseAvx2 = false;
#endif

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
                                       options.sigma,
                                       false,
                                       kUseAvx2);
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
    vl_sift_update_gradient_for_detected_keypoints(filter);

    std::vector<std::array<double, 4>> angles(
        static_cast<std::size_t>(num_keypoints));
    std::vector<int> orientation_counts(
        static_cast<std::size_t>(num_keypoints), 0);
    parallel::parallel_for_(
        parallel::Range(0, num_keypoints), [&](const parallel::Range& range) {
          for (int i = range.start; i < range.end; ++i) {
            orientation_counts[static_cast<std::size_t>(i)] = std::min(
                vl_sift_calc_keypoint_orientations(
                    filter, angles[static_cast<std::size_t>(i)].data(),
                    keypoints + i),
                options.max_num_orientations);
          }
        });

    const std::size_t first_new_level = levels.size();
    std::vector<std::size_t> keypoint_levels(
        static_cast<std::size_t>(num_keypoints));
    int previous_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (keypoints[i].is != previous_level) {
        levels.push_back({});
        levels.back().dog_level = keypoints[i].is;
        previous_level = keypoints[i].is;
      }
      keypoint_levels[static_cast<std::size_t>(i)] = levels.size() - 1;
      ++levels.back().num_base_keypoints;
    }

    std::vector<std::size_t> level_feature_counts(levels.size(), 0);
    std::vector<std::size_t> keypoint_offsets(
        static_cast<std::size_t>(num_keypoints), 0);
    for (int i = 0; i < num_keypoints; ++i) {
      const std::size_t level = keypoint_levels[static_cast<std::size_t>(i)];
      keypoint_offsets[static_cast<std::size_t>(i)] =
          level_feature_counts[level];
      level_feature_counts[level] += static_cast<std::size_t>(
          orientation_counts[static_cast<std::size_t>(i)]);
    }
    for (std::size_t level = first_new_level; level < levels.size(); ++level) {
      levels[level].keypoints.resize(level_feature_counts[level]);
      levels[level].descriptors.resize(level_feature_counts[level] *
                                        VL_SIFT_DESCR_LEN);
    }

    parallel::parallel_for_(
        parallel::Range(0, num_keypoints), [&](const parallel::Range& range) {
          for (int i = range.start; i < range.end; ++i) {
            const std::size_t keypoint_index = static_cast<std::size_t>(i);
            VlSiftLevelFeatures& level = levels[keypoint_levels[keypoint_index]];
            for (int orientation_idx = 0;
                 orientation_idx < orientation_counts[keypoint_index];
                 ++orientation_idx) {
              const std::size_t output_index =
                  keypoint_offsets[keypoint_index] +
                  static_cast<std::size_t>(orientation_idx);
              const double angle = angles[keypoint_index]
                                           [static_cast<std::size_t>(
                                               orientation_idx)];
              FeatureKeypoint& converted = level.keypoints[output_index];
              converted.x = keypoints[i].x + 0.5F;
              converted.y = keypoints[i].y + 0.5F;
              converted.scale = keypoints[i].sigma;
              converted.response = 0.0F;
              converted.orientation = static_cast<float>(angle * 180.0 / kPi);
              if (converted.orientation < 0.0F) {
                converted.orientation += 360.0F;
              }

              float* descriptor =
                  level.descriptors.data() + output_index * VL_SIFT_DESCR_LEN;
              vl_sift_calc_keypoint_descriptor(filter, descriptor,
                                               keypoints + i, angle);
              TransformVlFeatDescriptorToUbc(descriptor);
            }
          }
        });
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
    case SiftBackend::kAuto:
      return "auto";
    case SiftBackend::kVlSift:
      return "vlsift";
    case SiftBackend::kCvSift:
      return "cvsift";
    case SiftBackend::kCudaSift:
      return "cuda_opencv";
    case SiftBackend::kCudaVlSift:
      return "cuda_vlfeat";
  }
  return "unknown";
}

SiftBackend SiftBackendFromName(std::string_view name) {
  std::string normalized(name);
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });

  if (normalized == "auto") {
    return SiftBackend::kAuto;
  }
  if (normalized == "vlsift" || normalized == "vl_sift" ||
      normalized == "vlfeat") {
    return SiftBackend::kVlSift;
  }
  if (normalized == "cvsift" || normalized == "cv_sift") {
    return SiftBackend::kCvSift;
  }
  if (normalized == "cudasift" || normalized == "cuda_sift" ||
      normalized == "cuda" || normalized == "cuda_opencv") {
    return SiftBackend::kCudaSift;
  }
  if (normalized == "cuda_vlsift" || normalized == "cuda_vl_sift" ||
      normalized == "cuda_vlfeat") {
    return SiftBackend::kCudaVlSift;
  }
  throw std::invalid_argument("Unsupported SIFT backend: " + normalized);
}

SiftBackend ResolveSiftBackend(const SiftBackend requested) {
  if (requested != SiftBackend::kAuto) {
    return requested;
  }
  return IsCudaSiftAvailable() ? SiftBackend::kCudaSift
                               : SiftBackend::kCvSift;
}

FeatureSet ExtractSiftFeatures(const cv::Mat& image,
                               const SiftExtractionOptions& options) {
  const cv::Mat gray = ToGrayImage(image);
  const SiftBackend backend = ResolveSiftBackend(options.backend);
  switch (backend) {
    case SiftBackend::kAuto:
      break;
    case SiftBackend::kVlSift:
      return ExtractVlSiftFeatures(gray, options);
    case SiftBackend::kCvSift:
      return ExtractCvSiftFeatures(gray, options);
    case SiftBackend::kCudaSift:
    case SiftBackend::kCudaVlSift:
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
