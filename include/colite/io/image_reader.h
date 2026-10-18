#pragma once

#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace colite {

cv::Mat ApplyExifOrientation(const cv::Mat& image, int orientation);
// JPEG dimensions are obtained from the header; other formats use a decoded
// grayscale fallback. The size is expressed in the oriented image coordinates.
cv::Size ReadImageSizeWithExifOrientation(
    const std::filesystem::path& image_path);
cv::Mat ReadImageWithExifOrientation(
    const std::filesystem::path& image_path,
    int flags = cv::IMREAD_COLOR);

}  // namespace colite
