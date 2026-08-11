#pragma once

#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace colite {

cv::Mat ApplyExifOrientation(const cv::Mat& image, int orientation);
cv::Mat ReadImageWithExifOrientation(
    const std::filesystem::path& image_path,
    int flags = cv::IMREAD_COLOR);

}  // namespace colite
