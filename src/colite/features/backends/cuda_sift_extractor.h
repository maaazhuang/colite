#pragma once

#include <opencv2/core.hpp>

#include "colite/features/sift_extractor.h"

namespace colite {

// 返回当前构建是否包含 CudaSift 后端，并且运行时是否能看到 CUDA 设备。
bool IsCudaSiftAvailable();

// 使用项目内移植的 CudaSift 后端提取 SIFT 特征。
FeatureSet ExtractCudaSiftFeatures(const cv::Mat& image,
                                   const SiftExtractionOptions& options);

}  // namespace colite
