#pragma once

#include <filesystem>
#include <optional>

namespace colite {

// 从 JPEG EXIF 中读取 35mm 等效焦距。没有 EXIF 或字段不可用时返回空。
std::optional<double> ReadExifFocalLength35mm(
    const std::filesystem::path& image_path);

// 估计初始焦距，行为与旧 recon_sfm 保持一致：
// 1. 用户显式传入焦距时直接使用；
// 2. JPEG EXIF 有 35mm 等效焦距时按 f = f35 / 35 * max(width,height)；
// 3. 否则使用 1.2 * max(width,height)。
double EstimateInitialFocalLengthFromImage(
    const std::filesystem::path& image_path,
    double user_focal_length);

}  // namespace colite
