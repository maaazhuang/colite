#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "colite/core/feature.h"

namespace colite {
namespace features {

// 将不同 SIFT 后端的 128 维 float 描述子转换为 colite 内部矩阵。
// use_root_sift=true 时执行 L1 归一化 + 开方，使匹配阶段看到统一的 RootSIFT。
inline DescriptorMatrix ConvertSiftDescriptors(const float* descriptors,
                                                const std::size_t num_rows,
                                                const bool use_root_sift) {
  if (num_rows == 0) {
    return {};
  }
  if (descriptors == nullptr) {
    throw std::runtime_error("SIFT descriptor buffer is null.");
  }

  DescriptorMatrix result(static_cast<Eigen::Index>(num_rows), 128);
  for (std::size_t row = 0; row < num_rows; ++row) {
    double l1_norm = 0.0;
    for (int col = 0; col < 128; ++col) {
      const double value =
          std::max(0.0, static_cast<double>(descriptors[row * 128 + col]));
      result(static_cast<Eigen::Index>(row), col) = static_cast<float>(value);
      l1_norm += value;
    }

    if (use_root_sift && l1_norm > 0.0) {
      const double inv_l1_norm = 1.0 / l1_norm;
      for (int col = 0; col < 128; ++col) {
        result(static_cast<Eigen::Index>(row), col) = static_cast<float>(
            std::sqrt(result(static_cast<Eigen::Index>(row), col) *
                      inv_l1_norm));
      }
    }
  }
  return result;
}

}  // namespace features
}  // namespace colite
