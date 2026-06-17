#pragma once

#include <vector>

#include <Eigen/Core>

#include "colite/core/types.h"

namespace colite {

// 描述子矩阵。每一行是一条描述子，使用行主序便于传给 FLANN 等 C 风格接口。
using DescriptorMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// 图像局部特征点。第一版保留 OpenCV 常用字段，后续可扩展仿射形状。
struct FeatureKeypoint {
  float x = 0.0F;
  float y = 0.0F;
  float scale = 1.0F;
  float orientation = 0.0F;
};

// 两张图像之间的一条特征匹配，idx1/idx2 分别索引两张图像的特征点。
struct FeatureMatch {
  Point2DIdx idx1 = kInvalidPoint2DIdx;
  Point2DIdx idx2 = kInvalidPoint2DIdx;
};

// 单张图像的特征缓存。描述子使用内部矩阵，避免匹配核心依赖 OpenCV 数据结构。
struct FeatureSet {
  std::vector<FeatureKeypoint> keypoints;
  DescriptorMatrix descriptors;
};

}  // namespace colite
