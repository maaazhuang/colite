#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "colite/core/feature.h"

namespace colite {

// SIFT 特征提取后端。OpenCV 官方 SIFT 直接调用路径已移除；
// CvSift / VlSift / CudaSift 都是项目内接入的实现。
enum class SiftBackend {
  kAuto,
  kVlSift,
  kCvSift,
  // 优化 CUDA 引擎的 OpenCV 语义模式；保留 kCudaSift 名称兼容旧 API。
  kCudaSift,
  // 优化 CUDA 引擎的 VLFeat 语义模式。
  kCudaVlSift,
};

const char* SiftBackendName(SiftBackend backend);
SiftBackend SiftBackendFromName(std::string_view name);
// 将 auto 解析为当前主机实际使用的后端；有可用 CUDA 设备时优先 CUDA OpenCV。
SiftBackend ResolveSiftBackend(SiftBackend requested);

// SIFT 特征提取配置。
struct SiftExtractionOptions {
  SiftBackend backend = SiftBackend::kAuto;
  // 最终输出的特征数量上限。CudaSift 内部会使用更大的候选池，再按响应裁剪。
  int max_num_features = 8192;
  int num_octave_layers = 3;
  double contrast_threshold = 0.04;
  double edge_threshold = 10.0;
  double sigma = 1.6;

  // VLFeat/COLMAP CPU SIFT 专用参数。COLMAP 普通 CPU SIFT 默认固定 4 个
  // octave，并且每个 keypoint 最多保留 2 个方向。
  int vl_num_octaves = 4;
  int max_num_orientations = 2;

  // COLMAP 默认使用 first_octave=-1。该选项用于 VLFeat 后端；CvSift 后端
  // 保留自身 OpenCV-like 行为。
  bool upsample_first_octave = true;

  // COLMAP 默认使用 L1-root 描述子归一化，也就是 RootSIFT。
  bool use_root_sift = true;

  // CudaSift 使用自己的阈值尺度，默认值参考 CudaSift 示例。
  int cuda_device = 0;
  int cuda_num_octaves = 5;
  double cuda_init_blur = 1.0;
  float cuda_threshold = 3.0f;
  float cuda_lowest_scale = 0.0f;
  bool cuda_scale_up = false;
};

// 返回 CudaSift 后端在当前构建和运行时环境下是否可用。
bool IsCudaSiftAvailable();

// 从灰度或彩色图像中提取 SIFT 特征。图像容器仍使用 cv::Mat。
FeatureSet ExtractSiftFeatures(const cv::Mat& image,
                               const SiftExtractionOptions& options = {});

// 从磁盘图像文件读取并提取 SIFT 特征。
FeatureSet ExtractSiftFeaturesFromFile(
    const std::filesystem::path& image_path,
    const SiftExtractionOptions& options = {});

// 批量提取目录下的常见图像文件，结果按文件名路径排序。
std::map<std::filesystem::path, FeatureSet> ExtractSiftFeaturesFromDirectory(
    const std::filesystem::path& image_dir,
    const SiftExtractionOptions& options = {});

// 列出目录下的常见图像文件。该函数不递归。
std::vector<std::filesystem::path> ListImageFiles(
    const std::filesystem::path& image_dir);

}  // namespace colite
