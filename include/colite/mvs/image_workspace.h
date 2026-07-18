#pragma once

#include <cstddef>
#include <filesystem>
#include <list>
#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>

#include "colite/core/reconstruction.h"

namespace colite::mvs {

// MVS 使用的图像缓存项。图像统一保存为 CV_32FC1 灰度图，像素范围为 [0, 1]。
struct MvsImage {
  ImageId image_id = kInvalidImageId;
  std::string name;
  cv::Mat gray;
  // CV_8UC3 in RGB channel order. PatchMatch continues to use gray; fusion
  // samples this image to preserve the source colors in the dense cloud.
  cv::Mat color;
  double scale = 1.0;

  std::size_t Width() const;
  std::size_t Height() const;
  bool Empty() const;
};

struct MvsImageWorkspaceOptions {
  // text model 中只有图像名时，从该目录读取真实图像。
  std::filesystem::path image_path;

  // 0 表示不做容量限制；其它值表示最多缓存的图像数量。
  std::size_t max_cache_entries = 16;

  // 大于 0 时按最长边缩放图像，保持宽高比。
  int max_image_size = -1;
  // Each level halves width and height before applying max_image_size.
  std::size_t resolution_level = 0;
};

// MVS 图像 workspace。它只负责图像加载、尺度变换和 LRU 缓存，不负责深度估计。
class MvsImageWorkspace {
 public:
  MvsImageWorkspace(Reconstruction reconstruction,
                    MvsImageWorkspaceOptions options);

  const MvsImage& GetImage(ImageId image_id);
  bool IsCached(ImageId image_id) const;
  std::size_t NumCachedImages() const;
  void ClearCache();

 private:
  struct CacheEntry {
    MvsImage image;
    std::list<ImageId>::iterator lru_it;
  };

  MvsImage LoadImage(ImageId image_id) const;
  std::filesystem::path ResolveImagePath(const Image& image) const;
  void Touch(ImageId image_id);
  void EnforceCacheLimit();

  Reconstruction reconstruction_;
  MvsImageWorkspaceOptions options_;
  std::list<ImageId> lru_;
  std::unordered_map<ImageId, CacheEntry> cache_;
};

}  // namespace colite::mvs
