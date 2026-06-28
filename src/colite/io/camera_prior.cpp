#include "colite/io/camera_prior.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace colite {
namespace {

constexpr std::uint16_t kExifTagExifIfd = 0x8769;
constexpr std::uint16_t kExifTagFocalLengthIn35mmFilm = 0xA405;

std::uint16_t ReadUInt16(const std::vector<std::uint8_t>& bytes,
                         std::size_t offset,
                         bool little_endian) {
  if (offset + 2 > bytes.size()) {
    throw std::out_of_range("ReadUInt16 offset is outside EXIF data");
  }
  if (little_endian) {
    return static_cast<std::uint16_t>(bytes[offset] |
                                      (bytes[offset + 1] << 8));
  }
  return static_cast<std::uint16_t>((bytes[offset] << 8) | bytes[offset + 1]);
}

std::uint32_t ReadUInt32(const std::vector<std::uint8_t>& bytes,
                         std::size_t offset,
                         bool little_endian) {
  if (offset + 4 > bytes.size()) {
    throw std::out_of_range("ReadUInt32 offset is outside EXIF data");
  }
  if (little_endian) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
  }
  return (static_cast<std::uint32_t>(bytes[offset]) << 24) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
         static_cast<std::uint32_t>(bytes[offset + 3]);
}

std::optional<std::vector<std::uint8_t>> ReadJpegExifTiff(
    const std::filesystem::path& image_path) {
  std::ifstream file(image_path, std::ios::binary);
  if (!file) {
    return std::nullopt;
  }

  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>()};
  if (bytes.size() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8) {
    return std::nullopt;
  }

  std::size_t offset = 2;
  while (offset + 4 <= bytes.size()) {
    if (bytes[offset] != 0xFF) {
      return std::nullopt;
    }
    const std::uint8_t marker = bytes[offset + 1];
    offset += 2;
    if (marker == 0xDA || marker == 0xD9) {
      break;
    }
    if (offset + 2 > bytes.size()) {
      return std::nullopt;
    }
    const std::uint16_t segment_size =
        static_cast<std::uint16_t>((bytes[offset] << 8) | bytes[offset + 1]);
    if (segment_size < 2 || offset + segment_size > bytes.size()) {
      return std::nullopt;
    }

    const std::size_t payload_offset = offset + 2;
    const std::size_t payload_size = segment_size - 2;
    const std::array<std::uint8_t, 6> exif_prefix = {'E', 'x', 'i', 'f', 0, 0};
    if (marker == 0xE1 && payload_size >= exif_prefix.size() &&
        std::equal(exif_prefix.begin(), exif_prefix.end(),
                   bytes.begin() +
                       static_cast<std::ptrdiff_t>(payload_offset))) {
      const std::size_t tiff_offset = payload_offset + exif_prefix.size();
      return std::vector<std::uint8_t>(
          bytes.begin() + static_cast<std::ptrdiff_t>(tiff_offset),
          bytes.begin() +
              static_cast<std::ptrdiff_t>(payload_offset + payload_size));
    }

    offset += segment_size;
  }
  return std::nullopt;
}

std::optional<std::uint32_t> ReadIfdUIntValue(
    const std::vector<std::uint8_t>& tiff,
    std::size_t ifd_offset,
    std::uint16_t desired_tag,
    bool little_endian) {
  if (ifd_offset + 2 > tiff.size()) {
    return std::nullopt;
  }

  const std::uint16_t num_entries =
      ReadUInt16(tiff, ifd_offset, little_endian);
  for (std::uint16_t i = 0; i < num_entries; ++i) {
    const std::size_t entry_offset = ifd_offset + 2 + 12 * i;
    if (entry_offset + 12 > tiff.size()) {
      return std::nullopt;
    }

    const std::uint16_t tag = ReadUInt16(tiff, entry_offset, little_endian);
    if (tag != desired_tag) {
      continue;
    }

    const std::uint16_t type =
        ReadUInt16(tiff, entry_offset + 2, little_endian);
    const std::uint32_t count =
        ReadUInt32(tiff, entry_offset + 4, little_endian);
    if (count != 1) {
      return std::nullopt;
    }

    if (type == 3) {
      return ReadUInt16(tiff, entry_offset + 8, little_endian);
    }
    if (type == 4) {
      return ReadUInt32(tiff, entry_offset + 8, little_endian);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace

std::optional<double> ReadExifFocalLength35mm(
    const std::filesystem::path& image_path) {
  const std::optional<std::vector<std::uint8_t>> tiff =
      ReadJpegExifTiff(image_path);
  if (!tiff.has_value() || tiff->size() < 8) {
    return std::nullopt;
  }

  const bool little_endian = (*tiff)[0] == 'I' && (*tiff)[1] == 'I';
  const bool big_endian = (*tiff)[0] == 'M' && (*tiff)[1] == 'M';
  if (!little_endian && !big_endian) {
    return std::nullopt;
  }
  if (ReadUInt16(*tiff, 2, little_endian) != 42) {
    return std::nullopt;
  }

  const std::uint32_t ifd0_offset = ReadUInt32(*tiff, 4, little_endian);
  const std::optional<std::uint32_t> exif_ifd_offset =
      ReadIfdUIntValue(*tiff, ifd0_offset, kExifTagExifIfd, little_endian);
  if (!exif_ifd_offset.has_value()) {
    return std::nullopt;
  }

  const std::optional<std::uint32_t> focal_35mm =
      ReadIfdUIntValue(*tiff, *exif_ifd_offset,
                       kExifTagFocalLengthIn35mmFilm, little_endian);
  if (!focal_35mm.has_value() || *focal_35mm == 0) {
    return std::nullopt;
  }
  return static_cast<double>(*focal_35mm);
}

double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    const double user_focal_length) {
  if (user_focal_length > 0.0) {
    return user_focal_length;
  }

  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }

  const double max_dimension =
      static_cast<double>(std::max(image.cols, image.rows));
  const std::optional<double> focal_35mm = ReadExifFocalLength35mm(image_path);
  if (focal_35mm.has_value() && *focal_35mm > 0.0) {
    return (*focal_35mm / 35.0) * max_dimension;
  }
  return 1.2 * max_dimension;
}

}  // namespace colite
