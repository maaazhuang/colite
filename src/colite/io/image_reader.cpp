#include "colite/io/image_reader.h"

#include <stdexcept>
#include <fstream>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "colite/io/camera_prior.h"

namespace colite {

cv::Size ReadImageSizeWithExifOrientation(
    const std::filesystem::path& image_path) {
  cv::Size size;
  std::ifstream file(image_path, std::ios::binary);
  if (file.get() == 0xFF && file.get() == 0xD8) {
    while (file.get() == 0xFF) {
      int marker = file.get();
      while (marker == 0xFF) marker = file.get();
      if (marker < 0 || marker == 0xDA || marker == 0xD9) break;
      if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) continue;
      const int high = file.get(), low = file.get();
      if (high < 0 || low < 0) break;
      const int length = (high << 8) | low;
      if (length < 2) break;
      const bool start_of_frame = marker >= 0xC0 && marker <= 0xCF &&
                                  marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
      if (start_of_frame && length >= 8) {
        unsigned char dimensions[5];
        file.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
        if (file) {
          size.height = (dimensions[1] << 8) | dimensions[2];
          size.width = (dimensions[3] << 8) | dimensions[4];
        }
        break;
      }
      file.seekg(length - 2, std::ios::cur);
    }
  }
  if (size.empty()) {
    size = cv::imread(image_path.string(),
                     cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION).size();
  }
  const int orientation = ReadExifOrientation(image_path);
  if (orientation >= 5 && orientation <= 8) std::swap(size.width, size.height);
  return size;
}

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
  const int orientation = ReadExifOrientation(image_path);
  return orientation == 1 ? encoded : ApplyExifOrientation(encoded, orientation);
}

}  // namespace colite
