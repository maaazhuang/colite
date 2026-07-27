#include "colite/io/camera_prior.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace colite {
namespace {

constexpr std::uint16_t kExifTagExifIfd = 0x8769;
constexpr std::uint16_t kExifTagFocalLengthIn35mmFilm = 0xA405;
constexpr std::uint16_t kExifTagFocalLength = 0x920A;
constexpr std::uint16_t kExifTagFocalPlaneXResolution = 0xA20E;
constexpr std::uint16_t kExifTagFocalPlaneResolutionUnit = 0xA210;
constexpr double kFullFrameDiagonalMillimeters = 43.27;

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

std::optional<double> ReadIfdNumericValue(
    const std::vector<std::uint8_t>& tiff,
    std::size_t ifd_offset,
    std::uint16_t desired_tag,
    bool little_endian) {
  if (ifd_offset + 2 > tiff.size()) return std::nullopt;
  const std::uint16_t num_entries =
      ReadUInt16(tiff, ifd_offset, little_endian);
  for (std::uint16_t i = 0; i < num_entries; ++i) {
    const std::size_t entry_offset = ifd_offset + 2 + 12 * i;
    if (entry_offset + 12 > tiff.size()) return std::nullopt;
    if (ReadUInt16(tiff, entry_offset, little_endian) != desired_tag) continue;
    const std::uint16_t type =
        ReadUInt16(tiff, entry_offset + 2, little_endian);
    const std::uint32_t count =
        ReadUInt32(tiff, entry_offset + 4, little_endian);
    if (count != 1) return std::nullopt;
    if (type == 3) {
      return static_cast<double>(
          ReadUInt16(tiff, entry_offset + 8, little_endian));
    }
    if (type == 4) {
      return static_cast<double>(
          ReadUInt32(tiff, entry_offset + 8, little_endian));
    }
    if (type == 5) {
      const std::size_t value_offset =
          ReadUInt32(tiff, entry_offset + 8, little_endian);
      if (value_offset + 8 > tiff.size()) return std::nullopt;
      const std::uint32_t numerator =
          ReadUInt32(tiff, value_offset, little_endian);
      const std::uint32_t denominator =
          ReadUInt32(tiff, value_offset + 4, little_endian);
      if (denominator == 0) return std::nullopt;
      return static_cast<double>(numerator) / denominator;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

struct ExifFocalMetadata {
  std::optional<double> focal_35mm;
  std::optional<double> focal_mm;
  std::optional<double> focal_plane_x_resolution;
  std::optional<double> focal_plane_resolution_unit;
};

std::optional<ExifFocalMetadata> ReadExifFocalMetadata(
    const std::filesystem::path& image_path) {
  const auto tiff = ReadJpegExifTiff(image_path);
  if (!tiff.has_value() || tiff->size() < 8) return std::nullopt;
  const bool little_endian = (*tiff)[0] == 'I' && (*tiff)[1] == 'I';
  const bool big_endian = (*tiff)[0] == 'M' && (*tiff)[1] == 'M';
  if ((!little_endian && !big_endian) ||
      ReadUInt16(*tiff, 2, little_endian) != 42) {
    return std::nullopt;
  }
  const std::uint32_t ifd0_offset = ReadUInt32(*tiff, 4, little_endian);
  const auto exif_ifd_offset =
      ReadIfdUIntValue(*tiff, ifd0_offset, kExifTagExifIfd, little_endian);
  if (!exif_ifd_offset.has_value()) return std::nullopt;
  ExifFocalMetadata metadata;
  metadata.focal_35mm = ReadIfdNumericValue(
      *tiff, *exif_ifd_offset, kExifTagFocalLengthIn35mmFilm, little_endian);
  metadata.focal_mm = ReadIfdNumericValue(
      *tiff, *exif_ifd_offset, kExifTagFocalLength, little_endian);
  metadata.focal_plane_x_resolution = ReadIfdNumericValue(
      *tiff, *exif_ifd_offset, kExifTagFocalPlaneXResolution, little_endian);
  metadata.focal_plane_resolution_unit = ReadIfdNumericValue(
      *tiff, *exif_ifd_offset, kExifTagFocalPlaneResolutionUnit,
      little_endian);
  return metadata;
}

}  // namespace

std::optional<double> ReadExifFocalLength35mm(
    const std::filesystem::path& image_path) {
  const auto metadata = ReadExifFocalMetadata(image_path);
  if (!metadata.has_value() || !metadata->focal_35mm.has_value() ||
      *metadata->focal_35mm <= 0.0) return std::nullopt;
  return metadata->focal_35mm;
}

InitialFocalLengthEstimate EstimateInitialFocalLengthPriorFromImage(
    const std::filesystem::path& image_path,
    const double user_focal_length) {
  if (user_focal_length > 0.0) {
    return {user_focal_length, InitialFocalLengthSource::kUser, true};
  }

  const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }

  const double max_dimension = static_cast<double>(
      std::max(image.cols, image.rows));
  const auto metadata = ReadExifFocalMetadata(image_path);
  if (metadata.has_value() && metadata->focal_35mm.has_value() &&
      *metadata->focal_35mm > 0.0) {
    const double diagonal = std::hypot(static_cast<double>(image.cols),
                                       static_cast<double>(image.rows));
    return {*metadata->focal_35mm / kFullFrameDiagonalMillimeters * diagonal,
            InitialFocalLengthSource::kExif35mm, true};
  }
  if (metadata.has_value() && metadata->focal_mm.has_value() &&
      metadata->focal_plane_x_resolution.has_value() &&
      metadata->focal_plane_resolution_unit.has_value() &&
      *metadata->focal_mm > 0.0 &&
      *metadata->focal_plane_x_resolution > 0.0) {
    double pixels_per_mm = 0.0;
    switch (static_cast<int>(*metadata->focal_plane_resolution_unit)) {
      case 2: pixels_per_mm = *metadata->focal_plane_x_resolution / 25.4; break;
      case 3: pixels_per_mm = *metadata->focal_plane_x_resolution / 10.0; break;
      case 4: pixels_per_mm = *metadata->focal_plane_x_resolution; break;
      case 5: pixels_per_mm = *metadata->focal_plane_x_resolution * 1000.0; break;
      default: break;
    }
    if (pixels_per_mm > 0.0) {
      return {*metadata->focal_mm * pixels_per_mm,
              InitialFocalLengthSource::kExifFocalPlane, true};
    }
  }
  return {1.2 * max_dimension, InitialFocalLengthSource::kDefaultImageSize,
          false};
}

double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    const double user_focal_length) {
  return EstimateInitialFocalLengthPriorFromImage(image_path,
                                                   user_focal_length).pixels;
}

const char* InitialFocalLengthSourceName(
    const InitialFocalLengthSource source) {
  switch (source) {
    case InitialFocalLengthSource::kUser: return "user";
    case InitialFocalLengthSource::kExif35mm: return "exif_35mm";
    case InitialFocalLengthSource::kExifFocalPlane: return "exif_focal_plane";
    case InitialFocalLengthSource::kDefaultImageSize: return "image_size_default";
  }
  return "unknown";
}

}  // namespace colite
