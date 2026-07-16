#include "colite/mvs/dense_map.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>

namespace colite::mvs {
namespace {

constexpr char kDepthMapMagic[] = "COLITE_DENSE_DEPTH_MAP_TEXT";
constexpr char kNormalMapMagic[] = "COLITE_NORMAL_MAP_TEXT";
constexpr char kCostMapMagic[] = "COLITE_DENSE_COST_MAP_TEXT";
constexpr char kDepthMapBinaryMagic[] = "COLITE_DEPTH_MAP_BINARY";
constexpr char kNormalMapBinaryMagic[] = "COLITE_NORMAL_MAP_BINARY";
constexpr int kTextMapVersion = 1;
constexpr int kBinaryMapVersion = 1;

bool IsValidDepthValue(float depth) {
  return std::isfinite(depth) && depth > 0.0F;
}

bool IsValidNormalValue(const Eigen::Vector3f& normal) {
  return normal.allFinite() && normal.squaredNorm() > 0.0F;
}

void RequireStream(bool condition, const std::filesystem::path& path) {
  if (!condition) {
    throw std::runtime_error("Invalid dense map file: " + path.string());
  }
}

void RequireBinaryStream(bool condition, const std::filesystem::path& path) {
  if (!condition) {
    throw std::runtime_error("Invalid binary dense map file: " +
                             path.string());
  }
}

}  // namespace

DenseDepthMap::DenseDepthMap(std::size_t width,
                             std::size_t height,
                             float depth_min,
                             float depth_max)
    : width_(width),
      height_(height),
      depth_min_(depth_min),
      depth_max_(depth_max),
      data_(width * height, 0.0F) {}

std::size_t DenseDepthMap::Width() const {
  return width_;
}

std::size_t DenseDepthMap::Height() const {
  return height_;
}

std::size_t DenseDepthMap::NumPixels() const {
  return data_.size();
}

bool DenseDepthMap::Empty() const {
  return data_.empty();
}

float DenseDepthMap::DepthMin() const {
  return depth_min_;
}

float DenseDepthMap::DepthMax() const {
  return depth_max_;
}

void DenseDepthMap::SetDepthRange(float depth_min, float depth_max) {
  depth_min_ = depth_min;
  depth_max_ = depth_max;
}

void DenseDepthMap::Resize(std::size_t width,
                           std::size_t height,
                           float value) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, value);
}

void DenseDepthMap::Fill(float value) {
  std::fill(data_.begin(), data_.end(), value);
}

float DenseDepthMap::Get(std::size_t row, std::size_t col) const {
  return data_.at(Offset(row, col));
}

void DenseDepthMap::Set(std::size_t row, std::size_t col, float depth) {
  data_.at(Offset(row, col)) = depth;
}

bool DenseDepthMap::IsValid(std::size_t row, std::size_t col) const {
  return IsValidDepthValue(Get(row, col));
}

std::size_t DenseDepthMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), IsValidDepthValue));
}

bool DenseDepthMap::RecomputeDepthRange() {
  float min_depth = std::numeric_limits<float>::infinity();
  float max_depth = 0.0F;
  for (const float depth : data_) {
    if (!IsValidDepthValue(depth)) {
      continue;
    }
    min_depth = std::min(min_depth, depth);
    max_depth = std::max(max_depth, depth);
  }
  if (!std::isfinite(min_depth)) {
    depth_min_ = -1.0F;
    depth_max_ = -1.0F;
    return false;
  }
  depth_min_ = min_depth;
  depth_max_ = max_depth;
  return true;
}

const std::vector<float>& DenseDepthMap::Data() const {
  return data_;
}

std::vector<float>& DenseDepthMap::MutableData() {
  return data_;
}

std::size_t DenseDepthMap::Offset(std::size_t row, std::size_t col) const {
  if (row >= height_ || col >= width_) {
    throw std::out_of_range("DenseDepthMap pixel index is out of range");
  }
  return row * width_ + col;
}

NormalMap::NormalMap(std::size_t width, std::size_t height)
    : width_(width),
      height_(height),
      data_(width * height, Eigen::Vector3f::Zero()) {}

std::size_t NormalMap::Width() const {
  return width_;
}

std::size_t NormalMap::Height() const {
  return height_;
}

std::size_t NormalMap::NumPixels() const {
  return data_.size();
}

bool NormalMap::Empty() const {
  return data_.empty();
}

void NormalMap::Resize(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, Eigen::Vector3f::Zero());
}

void NormalMap::Fill(const Eigen::Vector3f& normal) {
  std::fill(data_.begin(), data_.end(), normal);
}

Eigen::Vector3f NormalMap::Get(std::size_t row, std::size_t col) const {
  return data_.at(Offset(row, col));
}

void NormalMap::Set(std::size_t row,
                    std::size_t col,
                    const Eigen::Vector3f& normal) {
  data_.at(Offset(row, col)) = normal;
}

void NormalMap::SetNormalized(std::size_t row,
                              std::size_t col,
                              const Eigen::Vector3f& normal) {
  if (!IsValidNormalValue(normal)) {
    Set(row, col, Eigen::Vector3f::Zero());
    return;
  }
  Set(row, col, normal.normalized());
}

bool NormalMap::IsValid(std::size_t row, std::size_t col) const {
  return IsValidNormalValue(Get(row, col));
}

std::size_t NormalMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), IsValidNormalValue));
}

const std::vector<Eigen::Vector3f>& NormalMap::Data() const {
  return data_;
}

std::vector<Eigen::Vector3f>& NormalMap::MutableData() {
  return data_;
}

std::size_t NormalMap::Offset(std::size_t row, std::size_t col) const {
  if (row >= height_ || col >= width_) {
    throw std::out_of_range("NormalMap pixel index is out of range");
  }
  return row * width_ + col;
}

DenseCostMap::DenseCostMap(std::size_t width,
                           std::size_t height,
                           float invalid_cost)
    : width_(width), height_(height), data_(width * height, invalid_cost) {}

std::size_t DenseCostMap::Width() const {
  return width_;
}

std::size_t DenseCostMap::Height() const {
  return height_;
}

std::size_t DenseCostMap::NumPixels() const {
  return data_.size();
}

bool DenseCostMap::Empty() const {
  return data_.empty();
}

void DenseCostMap::Resize(std::size_t width,
                          std::size_t height,
                          float invalid_cost) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, invalid_cost);
}

void DenseCostMap::Fill(float cost) {
  std::fill(data_.begin(), data_.end(), cost);
}

float DenseCostMap::Get(std::size_t row, std::size_t col) const {
  return data_.at(Offset(row, col));
}

void DenseCostMap::Set(std::size_t row, std::size_t col, float cost) {
  data_.at(Offset(row, col)) = cost;
}

bool DenseCostMap::IsValid(std::size_t row, std::size_t col) const {
  const float cost = Get(row, col);
  return std::isfinite(cost) && cost >= 0.0F;
}

std::size_t DenseCostMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), [](float cost) {
        return std::isfinite(cost) && cost >= 0.0F;
      }));
}

const std::vector<float>& DenseCostMap::Data() const {
  return data_;
}

std::vector<float>& DenseCostMap::MutableData() {
  return data_;
}

std::size_t DenseCostMap::Offset(std::size_t row, std::size_t col) const {
  if (row >= height_ || col >= width_) {
    throw std::out_of_range("DenseCostMap pixel index is out of range");
  }
  return row * width_ + col;
}

void WriteDenseDepthMapText(const std::filesystem::path& path,
                            const DenseDepthMap& depth_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense depth map: " +
                             path.string());
  }

  file << kDepthMapMagic << " " << kTextMapVersion << "\n"
       << depth_map.Width() << " " << depth_map.Height() << "\n"
       << std::setprecision(9) << depth_map.DepthMin() << " "
       << depth_map.DepthMax() << "\n";
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << depth_map.Get(row, col);
    }
    file << "\n";
  }
}

DenseDepthMap ReadDenseDepthMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read dense depth map: " +
                             path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireStream(magic == kDepthMapMagic && version == kTextMapVersion, path);

  std::size_t width = 0;
  std::size_t height = 0;
  float depth_min = -1.0F;
  float depth_max = -1.0F;
  file >> width >> height >> depth_min >> depth_max;
  RequireStream(static_cast<bool>(file), path);

  DenseDepthMap depth_map(width, height, depth_min, depth_max);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      float depth = 0.0F;
      file >> depth;
      RequireStream(static_cast<bool>(file), path);
      depth_map.Set(row, col, depth);
    }
  }
  return depth_map;
}

void WriteDenseDepthMapBinary(const std::filesystem::path& path,
                              const DenseDepthMap& depth_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary dense depth map: " +
                             path.string());
  }

  file << kDepthMapBinaryMagic << " " << kBinaryMapVersion << "\n"
       << depth_map.Width() << " " << depth_map.Height() << " "
       << std::setprecision(9) << depth_map.DepthMin() << " "
       << depth_map.DepthMax() << "\n";
  if (!depth_map.Data().empty()) {
    file.write(reinterpret_cast<const char*>(depth_map.Data().data()),
               static_cast<std::streamsize>(depth_map.Data().size() *
                                            sizeof(float)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

DenseDepthMap ReadDenseDepthMapBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary dense depth map: " +
                             path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireBinaryStream(magic == kDepthMapBinaryMagic &&
                          version == kBinaryMapVersion,
                      path);

  std::size_t width = 0;
  std::size_t height = 0;
  float depth_min = -1.0F;
  float depth_max = -1.0F;
  file >> width >> height >> depth_min >> depth_max;
  RequireBinaryStream(static_cast<bool>(file), path);
  file.get();

  DenseDepthMap depth_map(width, height, depth_min, depth_max);
  if (!depth_map.Data().empty()) {
    std::vector<float> values(depth_map.NumPixels(), 0.0F);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    RequireBinaryStream(static_cast<bool>(file), path);
    for (std::size_t row = 0; row < height; ++row) {
      for (std::size_t col = 0; col < width; ++col) {
        depth_map.Set(row, col, values[row * width + col]);
      }
    }
  }
  return depth_map;
}

void WriteNormalMapText(const std::filesystem::path& path,
                        const NormalMap& normal_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write normal map: " + path.string());
  }

  file << kNormalMapMagic << " " << kTextMapVersion << "\n"
       << normal_map.Width() << " " << normal_map.Height() << "\n";
  for (std::size_t row = 0; row < normal_map.Height(); ++row) {
    for (std::size_t col = 0; col < normal_map.Width(); ++col) {
      const Eigen::Vector3f normal = normal_map.Get(row, col);
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << normal.x() << " " << normal.y() << " "
           << normal.z();
    }
    file << "\n";
  }
}

NormalMap ReadNormalMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read normal map: " + path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireStream(magic == kNormalMapMagic && version == kTextMapVersion, path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireStream(static_cast<bool>(file), path);

  NormalMap normal_map(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      Eigen::Vector3f normal;
      file >> normal.x() >> normal.y() >> normal.z();
      RequireStream(static_cast<bool>(file), path);
      normal_map.Set(row, col, normal);
    }
  }
  return normal_map;
}

void WriteNormalMapBinary(const std::filesystem::path& path,
                          const NormalMap& normal_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary normal map: " +
                             path.string());
  }

  file << kNormalMapBinaryMagic << " " << kBinaryMapVersion << "\n"
       << normal_map.Width() << " " << normal_map.Height() << "\n";
  for (const Eigen::Vector3f& normal : normal_map.Data()) {
    const float xyz[3] = {normal.x(), normal.y(), normal.z()};
    file.write(reinterpret_cast<const char*>(xyz),
               static_cast<std::streamsize>(sizeof(xyz)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

NormalMap ReadNormalMapBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary normal map: " +
                             path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireBinaryStream(magic == kNormalMapBinaryMagic &&
                          version == kBinaryMapVersion,
                      path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireBinaryStream(static_cast<bool>(file), path);
  file.get();

  NormalMap normal_map(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      float xyz[3] = {0.0F, 0.0F, 0.0F};
      file.read(reinterpret_cast<char*>(xyz),
                static_cast<std::streamsize>(sizeof(xyz)));
      RequireBinaryStream(static_cast<bool>(file), path);
      normal_map.Set(row, col, Eigen::Vector3f(xyz[0], xyz[1], xyz[2]));
    }
  }
  return normal_map;
}

void WriteDenseCostMapText(const std::filesystem::path& path,
                           const DenseCostMap& cost_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense cost map: " +
                             path.string());
  }

  file << kCostMapMagic << " " << kTextMapVersion << "\n"
       << cost_map.Width() << " " << cost_map.Height() << "\n";
  for (std::size_t row = 0; row < cost_map.Height(); ++row) {
    for (std::size_t col = 0; col < cost_map.Width(); ++col) {
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << cost_map.Get(row, col);
    }
    file << "\n";
  }
}

DenseCostMap ReadDenseCostMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read dense cost map: " + path.string());
  }
  std::string magic;
  int version = 0;
  std::size_t width = 0;
  std::size_t height = 0;
  file >> magic >> version >> width >> height;
  RequireStream(magic == kCostMapMagic && version == kTextMapVersion, path);
  DenseCostMap cost_map(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::string token;
      file >> token;
      RequireStream(static_cast<bool>(file), path);
      float value = 0.0F;
      try {
        value = std::stof(token);
      } catch (const std::exception&) {
        throw std::runtime_error("Invalid dense map file: " + path.string());
      }
      cost_map.Set(row, col, value);
    }
  }
  return cost_map;
}

}  // namespace colite::mvs
