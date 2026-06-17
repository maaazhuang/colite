#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "colite/core/feature.h"

namespace colite {

// SIFT 特征提取配置。第一版只保留 OpenCV SIFT 常用参数，后续再扩展仿射形状等能力。
struct SiftExtractionOptions {
  int max_num_features = 8192;
  int num_octave_layers = 3;
  double contrast_threshold = 0.04;
  double edge_threshold = 10.0;
  double sigma = 1.6;

  // COLMAP 默认使用 first_octave=-1。OpenCV 不直接暴露该参数，因此这里通过
  // 先 2 倍上采样再把 keypoint 坐标缩回原图来近似，提升相邻图像弱纹理区域稳定性。
  bool upsample_first_octave = true;

  // COLMAP 默认使用 L1-root 描述子归一化，也就是 RootSIFT。
  bool use_root_sift = true;
};

// 从灰度或彩色图像中提取 SIFT 特征。
// 返回的描述子为 CV_32F，每一行对应一个 keypoint。
FeatureSet ExtractSiftFeatures(const cv::Mat& image,
                               const SiftExtractionOptions& options = {});

// 从磁盘图像文件读取并提取 SIFT 特征。
FeatureSet ExtractSiftFeaturesFromFile(
    const std::filesystem::path& image_path,
    const SiftExtractionOptions& options = {});

// 批量提取目录下的常见图像文件，结果按文件名路径排序，便于测试和复现。
std::map<std::filesystem::path, FeatureSet> ExtractSiftFeaturesFromDirectory(
    const std::filesystem::path& image_dir,
    const SiftExtractionOptions& options = {});

// 列出目录下的常见图像文件。该函数不递归，第一版保持简单。
std::vector<std::filesystem::path> ListImageFiles(
    const std::filesystem::path& image_dir);

}  // namespace colite
