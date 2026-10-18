#include "colite/io/camera_prior.h"
#include "colite/io/image_reader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <cmath>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace colite {
namespace {

constexpr std::uint16_t kExifTagExifIfd = 0x8769;
constexpr std::uint16_t kExifTagGpsIfd = 0x8825;
constexpr std::uint16_t kExifTagMake = 0x010F;
constexpr std::uint16_t kExifTagModel = 0x0110;
constexpr std::uint16_t kExifTagOrientation = 0x0112;
constexpr std::uint16_t kExifTagFocalLengthIn35mmFilm = 0xA405;
constexpr std::uint16_t kExifTagFocalLength = 0x920A;
constexpr std::uint16_t kExifTagFocalPlaneXResolution = 0xA20E;
constexpr std::uint16_t kExifTagFocalPlaneResolutionUnit = 0xA210;
constexpr std::uint16_t kGpsTagLatitudeRef = 0x0001;
constexpr std::uint16_t kGpsTagLatitude = 0x0002;
constexpr std::uint16_t kGpsTagLongitudeRef = 0x0003;
constexpr std::uint16_t kGpsTagLongitude = 0x0004;
constexpr std::uint16_t kGpsTagAltitudeRef = 0x0005;
constexpr std::uint16_t kGpsTagAltitude = 0x0006;
constexpr double kFullFrameDiagonalMillimeters = 43.27;
constexpr double kWgs84SemiMajorMeters = 6378137.0;
constexpr double kWgs84EccentricitySquared =
    6.6943799901413165e-3;
constexpr double kPi = 3.14159265358979323846;

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

  if (file.get() != 0xFF || file.get() != 0xD8) {
    return std::nullopt;
  }
  // EXIF is a small APP1 header. Never read compressed image pixels just to
  // obtain camera metadata (this function is used several times per image).
  while (file.get() == 0xFF) {
    int marker = file.get();
    while (marker == 0xFF) marker = file.get();
    if (marker == 0xDA || marker == 0xD9) {
      break;
    }
    if (marker < 0) return std::nullopt;
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) continue;
    const int high = file.get(), low = file.get();
    if (high < 0 || low < 0) return std::nullopt;
    const int segment_size = (high << 8) | low;
    if (segment_size < 2) return std::nullopt;
    const std::size_t payload_size = static_cast<std::size_t>(segment_size - 2);
    const std::array<std::uint8_t, 6> exif_prefix = {'E', 'x', 'i', 'f', 0, 0};
    if (marker == 0xE1 && payload_size >= exif_prefix.size()) {
      std::vector<std::uint8_t> payload(payload_size);
      file.read(reinterpret_cast<char*>(payload.data()), payload.size());
      if (!file) return std::nullopt;
      if (std::equal(exif_prefix.begin(), exif_prefix.end(), payload.begin())) {
        return std::vector<std::uint8_t>(payload.begin() + exif_prefix.size(),
                                         payload.end());
      }
    } else {
      file.seekg(static_cast<std::streamoff>(payload_size), std::ios::cur);
    }
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

std::optional<std::string> ReadIfdAsciiValue(
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
    if (type != 2 || count == 0) return std::nullopt;
    const std::size_t value_offset =
        count <= 4 ? entry_offset + 8
                   : ReadUInt32(tiff, entry_offset + 8, little_endian);
    if (value_offset + count > tiff.size()) return std::nullopt;
    std::string value(
        reinterpret_cast<const char*>(tiff.data() + value_offset), count);
    while (!value.empty() &&
           (value.back() == '\0' || value.back() == ' ')) {
      value.pop_back();
    }
    return value.empty() ? std::nullopt
                         : std::optional<std::string>(std::move(value));
  }
  return std::nullopt;
}

std::optional<std::vector<double>> ReadIfdRationalValues(
    const std::vector<std::uint8_t>& tiff,
    const std::size_t ifd_offset,
    const std::uint16_t desired_tag,
    const bool little_endian) {
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
    if (type != 5 || count == 0) return std::nullopt;
    const std::size_t value_offset =
        ReadUInt32(tiff, entry_offset + 8, little_endian);
    if (value_offset + static_cast<std::size_t>(count) * 8 > tiff.size()) {
      return std::nullopt;
    }
    std::vector<double> values;
    values.reserve(count);
    for (std::uint32_t idx = 0; idx < count; ++idx) {
      const std::size_t offset = value_offset + static_cast<std::size_t>(idx) * 8;
      const std::uint32_t numerator =
          ReadUInt32(tiff, offset, little_endian);
      const std::uint32_t denominator =
          ReadUInt32(tiff, offset + 4, little_endian);
      if (denominator == 0) return std::nullopt;
      values.push_back(static_cast<double>(numerator) / denominator);
    }
    return values;
  }
  return std::nullopt;
}

std::optional<std::uint8_t> ReadIfdByteValue(
    const std::vector<std::uint8_t>& tiff,
    const std::size_t ifd_offset,
    const std::uint16_t desired_tag,
    const bool little_endian) {
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
    if (type != 1 || count != 1) return std::nullopt;
    return tiff[entry_offset + 8];
  }
  return std::nullopt;
}

bool ParseTiffHeader(const std::vector<std::uint8_t>& tiff,
                     bool* little_endian,
                     std::uint32_t* ifd0_offset) {
  if (little_endian == nullptr || ifd0_offset == nullptr || tiff.size() < 8) {
    return false;
  }
  const bool little = tiff[0] == 'I' && tiff[1] == 'I';
  const bool big = tiff[0] == 'M' && tiff[1] == 'M';
  if ((!little && !big) || ReadUInt16(tiff, 2, little) != 42) return false;
  *little_endian = little;
  *ifd0_offset = ReadUInt32(tiff, 4, little);
  return *ifd0_offset < tiff.size();
}

struct ExifFocalMetadata {
  std::optional<std::string> make;
  std::optional<std::string> model;
  std::optional<double> focal_35mm;
  std::optional<double> focal_mm;
  std::optional<double> focal_plane_x_resolution;
  std::optional<double> focal_plane_resolution_unit;
};

std::optional<ExifFocalMetadata> ReadExifFocalMetadata(
    const std::filesystem::path& image_path) {
  const auto tiff = ReadJpegExifTiff(image_path);
  if (!tiff.has_value() || tiff->size() < 8) return std::nullopt;
  bool little_endian = false;
  std::uint32_t ifd0_offset = 0;
  if (!ParseTiffHeader(*tiff, &little_endian, &ifd0_offset)) {
    return std::nullopt;
  }
  const auto exif_ifd_offset =
      ReadIfdUIntValue(*tiff, ifd0_offset, kExifTagExifIfd, little_endian);
  if (!exif_ifd_offset.has_value()) return std::nullopt;
  ExifFocalMetadata metadata;
  metadata.make =
      ReadIfdAsciiValue(*tiff, ifd0_offset, kExifTagMake, little_endian);
  metadata.model =
      ReadIfdAsciiValue(*tiff, ifd0_offset, kExifTagModel, little_endian);
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

ExifCameraMetadata ReadExifCameraMetadata(
    const std::filesystem::path& image_path) {
  ExifCameraMetadata result;
  const auto metadata = ReadExifFocalMetadata(image_path);
  if (!metadata.has_value()) return result;
  result.make = metadata->make;
  result.model = metadata->model;
  result.focal_length = metadata->focal_mm;
  result.focal_length_35mm = metadata->focal_35mm;
  return result;
}

int ReadExifOrientation(const std::filesystem::path& image_path) {
  const auto tiff = ReadJpegExifTiff(image_path);
  if (!tiff.has_value()) return 1;
  bool little_endian = false;
  std::uint32_t ifd0_offset = 0;
  if (!ParseTiffHeader(*tiff, &little_endian, &ifd0_offset)) return 1;
  const std::optional<std::uint32_t> orientation =
      ReadIfdUIntValue(*tiff, ifd0_offset, kExifTagOrientation, little_endian);
  if (!orientation.has_value() || *orientation < 1 || *orientation > 8) {
    return 1;
  }
  return static_cast<int>(*orientation);
}

std::optional<GeodeticCoordinate> ReadExifGpsCoordinate(
    const std::filesystem::path& image_path) {
  const auto tiff = ReadJpegExifTiff(image_path);
  if (!tiff.has_value()) return std::nullopt;
  bool little_endian = false;
  std::uint32_t ifd0_offset = 0;
  if (!ParseTiffHeader(*tiff, &little_endian, &ifd0_offset)) {
    return std::nullopt;
  }
  const auto gps_ifd_offset =
      ReadIfdUIntValue(*tiff, ifd0_offset, kExifTagGpsIfd, little_endian);
  if (!gps_ifd_offset.has_value()) return std::nullopt;
  const auto latitude_ref = ReadIfdAsciiValue(
      *tiff, *gps_ifd_offset, kGpsTagLatitudeRef, little_endian);
  const auto longitude_ref = ReadIfdAsciiValue(
      *tiff, *gps_ifd_offset, kGpsTagLongitudeRef, little_endian);
  const auto latitude = ReadIfdRationalValues(
      *tiff, *gps_ifd_offset, kGpsTagLatitude, little_endian);
  const auto longitude = ReadIfdRationalValues(
      *tiff, *gps_ifd_offset, kGpsTagLongitude, little_endian);
  if (!latitude_ref.has_value() || !longitude_ref.has_value() ||
      !latitude.has_value() || latitude->size() != 3 ||
      !longitude.has_value() || longitude->size() != 3) {
    return std::nullopt;
  }
  GeodeticCoordinate result;
  result.latitude_degrees =
      (*latitude)[0] + (*latitude)[1] / 60.0 + (*latitude)[2] / 3600.0;
  result.longitude_degrees =
      (*longitude)[0] + (*longitude)[1] / 60.0 + (*longitude)[2] / 3600.0;
  if (!latitude_ref->empty() &&
      (latitude_ref->front() == 'S' || latitude_ref->front() == 's')) {
    result.latitude_degrees = -result.latitude_degrees;
  }
  if (!longitude_ref->empty() &&
      (longitude_ref->front() == 'W' || longitude_ref->front() == 'w')) {
    result.longitude_degrees = -result.longitude_degrees;
  }
  const auto altitude = ReadIfdRationalValues(
      *tiff, *gps_ifd_offset, kGpsTagAltitude, little_endian);
  if (altitude.has_value() && altitude->size() == 1) {
    result.altitude_meters = altitude->front();
    if (ReadIfdByteValue(*tiff, *gps_ifd_offset, kGpsTagAltitudeRef,
                         little_endian)
            .value_or(0) == 1) {
      result.altitude_meters = -result.altitude_meters;
    }
  }
  if (!std::isfinite(result.latitude_degrees) ||
      !std::isfinite(result.longitude_degrees) ||
      !std::isfinite(result.altitude_meters) ||
      std::abs(result.latitude_degrees) > 90.0 ||
      std::abs(result.longitude_degrees) > 180.0) {
    return std::nullopt;
  }
  return result;
}

Eigen::Vector3d GeodeticToEcef(const GeodeticCoordinate& coordinate) {
  const double latitude = coordinate.latitude_degrees * kPi / 180.0;
  const double longitude = coordinate.longitude_degrees * kPi / 180.0;
  const double sin_latitude = std::sin(latitude);
  const double cos_latitude = std::cos(latitude);
  const double radius = kWgs84SemiMajorMeters /
      std::sqrt(1.0 - kWgs84EccentricitySquared *
                          sin_latitude * sin_latitude);
  return Eigen::Vector3d(
      (radius + coordinate.altitude_meters) * cos_latitude *
          std::cos(longitude),
      (radius + coordinate.altitude_meters) * cos_latitude *
          std::sin(longitude),
      (radius * (1.0 - kWgs84EccentricitySquared) +
       coordinate.altitude_meters) * sin_latitude);
}

Eigen::Vector3d GeodeticToLocalEnu(const GeodeticCoordinate& coordinate,
                                   const GeodeticCoordinate& origin) {
  const Eigen::Vector3d delta =
      GeodeticToEcef(coordinate) - GeodeticToEcef(origin);
  const double latitude = origin.latitude_degrees * kPi / 180.0;
  const double longitude = origin.longitude_degrees * kPi / 180.0;
  const double sin_latitude = std::sin(latitude);
  const double cos_latitude = std::cos(latitude);
  const double sin_longitude = std::sin(longitude);
  const double cos_longitude = std::cos(longitude);
  Eigen::Matrix3d ecef_to_enu;
  ecef_to_enu << -sin_longitude, cos_longitude, 0.0,
      -sin_latitude * cos_longitude, -sin_latitude * sin_longitude,
      cos_latitude,
      cos_latitude * cos_longitude, cos_latitude * sin_longitude,
      sin_latitude;
  return ecef_to_enu * delta;
}

std::optional<double> ReadExifFocalLength35mm(
    const std::filesystem::path& image_path) {
  const auto metadata = ReadExifFocalMetadata(image_path);
  if (!metadata.has_value() || !metadata->focal_35mm.has_value() ||
      *metadata->focal_35mm <= 0.0) return std::nullopt;
  return metadata->focal_35mm;
}

InitialFocalLengthEstimate EstimateInitialFocalLengthPriorFromImage(
    const std::filesystem::path& image_path,
    const double user_focal_length,
    const double default_focal_length_factor) {
  if (!std::isfinite(default_focal_length_factor) ||
      default_focal_length_factor <= 0.0) {
    throw std::invalid_argument(
        "Default focal length factor must be finite and positive");
  }
  if (user_focal_length > 0.0) {
    return {user_focal_length, InitialFocalLengthSource::kUser, true};
  }

  const cv::Size image_size = ReadImageSizeWithExifOrientation(image_path);
  if (image_size.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path.string());
  }

  const double max_dimension = static_cast<double>(
      std::max(image_size.width, image_size.height));
  const auto metadata = ReadExifFocalMetadata(image_path);
  if (metadata.has_value() && metadata->focal_35mm.has_value() &&
      *metadata->focal_35mm > 0.0) {
    const double diagonal = std::hypot(static_cast<double>(image_size.width),
                                       static_cast<double>(image_size.height));
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
  return {default_focal_length_factor * max_dimension,
          InitialFocalLengthSource::kDefaultImageSize,
          false};
}

double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    const double user_focal_length,
    const double default_focal_length_factor) {
  return EstimateInitialFocalLengthPriorFromImage(image_path,
                                                   user_focal_length,
                                                   default_focal_length_factor)
      .pixels;
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
