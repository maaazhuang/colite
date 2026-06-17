#include "colite/features/sift_extractor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace colite {
namespace {

FeatureKeypoint ConvertKeypoint(const cv::KeyPoint& keypoint,
                                const double coordinate_scale) {
  FeatureKeypoint result;
  result.x = keypoint.pt.x * coordinate_scale;
  result.y = keypoint.pt.y * coordinate_scale;
  result.scale = keypoint.size * coordinate_scale;
  result.orientation = keypoint.angle;
  return result;
}

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

DescriptorMatrix ConvertDescriptors(const cv::Mat& descriptors,
                                    const bool use_root_sift) {
  if (descriptors.empty()) {
    return {};
  }

  cv::Mat float_descriptors;
  if (descriptors.type() == CV_32F) {
    float_descriptors =
        descriptors.isContinuous() ? descriptors : descriptors.clone();
  } else {
    descriptors.convertTo(float_descriptors, CV_32F);
  }

  DescriptorMatrix result(float_descriptors.rows, float_descriptors.cols);
  for (int row = 0; row < float_descriptors.rows; ++row) {
    const float* src = float_descriptors.ptr<float>(row);
    double l1_norm = 0.0;
    for (int col = 0; col < float_descriptors.cols; ++col) {
      const double value = std::max(0.0, static_cast<double>(src[col]));
      result(row, col) = value;
      l1_norm += value;
    }

    if (use_root_sift && l1_norm > 0.0) {
      // RootSIFT：先做 L1 归一化，再逐维开方。这样可弱化大梯度 bin 对
      // 距离排序的支配作用，与 COLMAP 的 l1_root 描述子归一化保持一致。
      const double inv_l1_norm = 1.0 / l1_norm;
      for (int col = 0; col < float_descriptors.cols; ++col) {
        result(row, col) = std::sqrt(result(row, col) * inv_l1_norm);
      }
    }
  }
  return result;
}

}  // namespace

FeatureSet ExtractSiftFeatures(const cv::Mat& image,
                               const SiftExtractionOptions& options) {
  const cv::Mat gray = ToGrayImage(image);
  cv::Mat extraction_image;
  double coordinate_scale = 1.0;
  if (options.upsample_first_octave) {
    // OpenCV SIFT 不暴露 COLMAP 的 first_octave=-1 参数，这里用 2 倍上采样
    // 近似低 octave 检测，再把坐标和尺度还原到原始图像坐标系。
    cv::resize(gray, extraction_image, cv::Size(), 2.0, 2.0,
               cv::INTER_LINEAR);
    coordinate_scale = 0.5;
  } else {
    extraction_image = gray;
  }

  auto sift = cv::SIFT::create(options.max_num_features,
                               options.num_octave_layers,
                               options.contrast_threshold,
                               options.edge_threshold,
                               options.sigma);

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  sift->detectAndCompute(extraction_image, cv::noArray(), keypoints,
                         descriptors);

  FeatureSet result;
  result.keypoints.reserve(keypoints.size());
  for (const cv::KeyPoint& keypoint : keypoints) {
    result.keypoints.push_back(ConvertKeypoint(keypoint, coordinate_scale));
  }

  result.descriptors = ConvertDescriptors(descriptors, options.use_root_sift);
  return result;
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
