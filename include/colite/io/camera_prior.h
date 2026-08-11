#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <Eigen/Core>

namespace colite {

enum class InitialFocalLengthSource {
  kUser,
  kExif35mm,
  kExifFocalPlane,
  kDefaultImageSize,
};

struct InitialFocalLengthEstimate {
  double pixels = 0.0;
  InitialFocalLengthSource source =
      InitialFocalLengthSource::kDefaultImageSize;
  bool has_prior = false;
};

struct ExifCameraMetadata {
  std::optional<std::string> make;
  std::optional<std::string> model;
  std::optional<double> focal_length;
  std::optional<double> focal_length_35mm;

  bool HasAutoCameraIdentity() const {
    return make.has_value() && model.has_value() &&
           (focal_length_35mm.has_value() || focal_length.has_value());
  }
};

struct GeodeticCoordinate {
  double latitude_degrees = 0.0;
  double longitude_degrees = 0.0;
  double altitude_meters = 0.0;
};

ExifCameraMetadata ReadExifCameraMetadata(
    const std::filesystem::path& image_path);

// EXIF Orientation value in [1, 8]. Missing or malformed values map to 1.
int ReadExifOrientation(const std::filesystem::path& image_path);

// Reads WGS84 latitude/longitude and optional altitude from JPEG GPS IFD.
// Missing altitude is represented as zero; malformed coordinates return null.
std::optional<GeodeticCoordinate> ReadExifGpsCoordinate(
    const std::filesystem::path& image_path);

// WGS84 conversion helpers. Local ENU is centered at origin and expressed in
// meters, suitable for Image::prior_position and spatial matching/BA.
Eigen::Vector3d GeodeticToEcef(const GeodeticCoordinate& coordinate);
Eigen::Vector3d GeodeticToLocalEnu(const GeodeticCoordinate& coordinate,
                                   const GeodeticCoordinate& origin);

// 从 JPEG EXIF 中读取 35mm 等效焦距。没有 EXIF 或字段不可用时返回空。
std::optional<double> ReadExifFocalLength35mm(
    const std::filesystem::path& image_path);

// 估计初始焦距：
// 1. 用户显式传入焦距时直接使用；
// 2. JPEG EXIF 有 35mm 等效焦距时按 43.27mm 全画幅对角线换算；
// 3. 否则尝试焦距毫米值与焦平面分辨率；
// 4. 仍不可用时使用 1.2 * max(width,height)。
double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length,
    double default_focal_length_factor = 1.2);

InitialFocalLengthEstimate EstimateInitialFocalLengthPriorFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length,
    double default_focal_length_factor = 1.2);

const char* InitialFocalLengthSourceName(InitialFocalLengthSource source);

}  // namespace colite
