#include "colite/io/image_reader.h"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "colite/io/camera_prior.h"

namespace colite {

cv::Mat ApplyExifOrientation(const cv::Mat& image, const int orientation) {
  if (image.empty()) return {};
  cv::Mat result;
  switch (orientation) {
    case 1: return image.clone();
    case 2: cv::flip(image, result, 1); break;
    case 3: cv::rotate(image, result, cv::ROTATE_180); break;
    case 4: cv::flip(image, result, 0); break;
    case 5: cv::transpose(image, result); break;
    case 6: cv::rotate(image, result, cv::ROTATE_90_CLOCKWISE); break;
    case 7:
      cv::transpose(image, result);
      cv::flip(result, result, -1);
      break;
    case 8: cv::rotate(image, result, cv::ROTATE_90_COUNTERCLOCKWISE); break;
    default:
      throw std::invalid_argument("EXIF orientation must be in [1, 8]");
  }
  return result;
}

cv::Mat ReadImageWithExifOrientation(
    const std::filesystem::path& image_path, const int flags) {
  const int read_flags = flags == cv::IMREAD_UNCHANGED
                             ? cv::IMREAD_UNCHANGED
                             : flags | cv::IMREAD_IGNORE_ORIENTATION;
  const cv::Mat encoded = cv::imread(image_path.string(), read_flags);
  if (encoded.empty()) return {};
  return ApplyExifOrientation(encoded, ReadExifOrientation(image_path));
}

}  // namespace colite
