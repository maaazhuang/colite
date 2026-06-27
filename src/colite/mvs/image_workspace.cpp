#include "colite/mvs/image_workspace.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace colite::mvs {

std::size_t MvsImage::Width() const {
  return gray.empty() ? 0U : static_cast<std::size_t>(gray.cols);
}

std::size_t MvsImage::Height() const {
  return gray.empty() ? 0U : static_cast<std::size_t>(gray.rows);
}

bool MvsImage::Empty() const {
  return gray.empty();
}

MvsImageWorkspace::MvsImageWorkspace(Reconstruction reconstruction,
                                     MvsImageWorkspaceOptions options)
    : reconstruction_(std::move(reconstruction)), options_(std::move(options)) {}

const MvsImage& MvsImageWorkspace::GetImage(ImageId image_id) {
  const auto cached = cache_.find(image_id);
  if (cached != cache_.end()) {
    Touch(image_id);
    return cached->second.image;
  }

  MvsImage image = LoadImage(image_id);
  lru_.push_front(image_id);
  auto [it, inserted] =
      cache_.emplace(image_id, CacheEntry{std::move(image), lru_.begin()});
  (void)inserted;
  EnforceCacheLimit();
  return it->second.image;
}

bool MvsImageWorkspace::IsCached(ImageId image_id) const {
  return cache_.find(image_id) != cache_.end();
}

std::size_t MvsImageWorkspace::NumCachedImages() const {
  return cache_.size();
}

void MvsImageWorkspace::ClearCache() {
  cache_.clear();
  lru_.clear();
}

MvsImage MvsImageWorkspace::LoadImage(ImageId image_id) const {
  if (!reconstruction_.ExistsImage(image_id) ||
      !reconstruction_.IsImageRegistered(image_id)) {
    throw std::invalid_argument("MVS image must exist and be registered.");
  }
  const Image& image = reconstruction_.GetImage(image_id);
  const std::filesystem::path image_path = ResolveImagePath(image);

  cv::Mat gray_u8 = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
  if (gray_u8.empty()) {
    throw std::runtime_error("Failed to read MVS image: " +
                             image_path.string());
  }

  double scale = 1.0;
  cv::Mat resized_u8 = gray_u8;
  if (options_.max_image_size > 0) {
    const int max_dimension = std::max(gray_u8.cols, gray_u8.rows);
    if (max_dimension > options_.max_image_size) {
      scale = static_cast<double>(options_.max_image_size) /
              static_cast<double>(max_dimension);
      const int new_width =
          std::max(1, static_cast<int>(std::round(gray_u8.cols * scale)));
      const int new_height =
          std::max(1, static_cast<int>(std::round(gray_u8.rows * scale)));
      cv::resize(gray_u8, resized_u8, cv::Size(new_width, new_height),
                 0.0, 0.0, cv::INTER_AREA);
    }
  }

  MvsImage result;
  result.image_id = image_id;
  result.name = image.name;
  result.scale = scale;
  resized_u8.convertTo(result.gray, CV_32FC1, 1.0 / 255.0);
  return result;
}

std::filesystem::path MvsImageWorkspace::ResolveImagePath(
    const Image& image) const {
  if (!image.file_path.empty()) {
    return image.file_path;
  }
  if (options_.image_path.empty()) {
    throw std::invalid_argument("MVS image_path is empty.");
  }
  return options_.image_path / image.name;
}

void MvsImageWorkspace::Touch(ImageId image_id) {
  const auto it = cache_.find(image_id);
  if (it == cache_.end()) {
    return;
  }
  lru_.erase(it->second.lru_it);
  lru_.push_front(image_id);
  it->second.lru_it = lru_.begin();
}

void MvsImageWorkspace::EnforceCacheLimit() {
  if (options_.max_cache_entries == 0) {
    return;
  }
  while (cache_.size() > options_.max_cache_entries && !lru_.empty()) {
    const ImageId evict_image_id = lru_.back();
    lru_.pop_back();
    cache_.erase(evict_image_id);
  }
}

}  // namespace colite::mvs
