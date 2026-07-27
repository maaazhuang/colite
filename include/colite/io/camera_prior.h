#pragma once

#include <filesystem>
#include <optional>

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
    double user_focal_length);

InitialFocalLengthEstimate EstimateInitialFocalLengthPriorFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length);

const char* InitialFocalLengthSourceName(InitialFocalLengthSource source);

}  // namespace colite
