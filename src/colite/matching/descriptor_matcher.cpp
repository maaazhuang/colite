#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <flann/flann.hpp>

namespace colite {
namespace {

// COLMAP 将 SIFT 描述子量化到长度约为 512 的 uint8 空间中，匹配阈值也在该
// 空间下定义。
constexpr int kSqSiftDescriptorNorm = 512 * 512;
constexpr int kFlannNumTrees = 4;
constexpr int kFlannNumLeavesToVisit = 128;

using DescriptorByteMatrix =
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DistanceMatrix =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::uint8_t ClampToByte(const float value) {
  if (!std::isfinite(value) || value <= 0.0F) {
    return 0;
  }
  if (value >= 255.0F) {
    return 255;
  }
  return static_cast<std::uint8_t>(std::round(value));
}

DescriptorByteMatrix QuantizeDescriptors(const DescriptorMatrix& descriptors) {
  DescriptorByteMatrix quantized(descriptors.rows(), descriptors.cols());
  if (descriptors.size() == 0) {
    return quantized;
  }

  bool unit_normalized = true;
  for (Eigen::Index idx = 0; idx < descriptors.size(); ++idx) {
    const float value = descriptors.data()[idx];
    if (std::isfinite(value) && value > 1.0001F) {
      unit_normalized = false;
      break;
    }
  }

  const float scale = unit_normalized ? 512.0F : 1.0F;
  for (Eigen::Index row = 0; row < descriptors.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors.cols(); ++col) {
      quantized(row, col) = ClampToByte(scale * descriptors(row, col));
    }
  }
  return quantized;
}

int DotProduct(const DescriptorByteMatrix& descriptors1,
               const Eigen::Index row1,
               const DescriptorByteMatrix& descriptors2,
               const Eigen::Index row2) {
  int dot_product = 0;
  for (Eigen::Index col = 0; col < descriptors1.cols(); ++col) {
    dot_product += static_cast<int>(descriptors1(row1, col)) *
                   static_cast<int>(descriptors2(row2, col));
  }
  return dot_product;
}

std::vector<int> FindBestMatchesOneWayBruteForce(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const double max_ratio,
    const double max_distance) {
  const float inv_sq_descriptor_norm =
      static_cast<float>(1.0 / kSqSiftDescriptorNorm);

  std::vector<int> matches(static_cast<std::size_t>(query_descriptors.rows()),
                           -1);

  for (Eigen::Index query_idx = 0; query_idx < query_descriptors.rows();
       ++query_idx) {
    int best_train_idx = -1;
    int best_dot_product = -1;
    int second_best_dot_product = -1;

    for (Eigen::Index train_idx = 0; train_idx < train_descriptors.rows();
         ++train_idx) {
      const int dot_product =
          DotProduct(query_descriptors, query_idx, train_descriptors,
                     train_idx);
      if (dot_product > best_dot_product) {
        best_train_idx = static_cast<int>(train_idx);
        second_best_dot_product = best_dot_product;
        best_dot_product = dot_product;
      } else if (dot_product > second_best_dot_product) {
        second_best_dot_product = dot_product;
      }
    }

    if (best_train_idx < 0) {
      continue;
    }

    const float best_distance = std::acos(std::min(
        inv_sq_descriptor_norm * static_cast<float>(best_dot_product), 1.0F));
    if (best_distance > max_distance) {
      continue;
    }

    const float second_best_distance = second_best_dot_product < 0
                                           ? std::numeric_limits<float>::max()
                                           : std::acos(std::min(
                                                 inv_sq_descriptor_norm *
                                                     static_cast<float>(
                                                         second_best_dot_product),
                                                 1.0F));
    // 与 COLMAP 保持一致：best == second_best 时必须拒绝，避免歧义匹配通过。
    if (best_distance >= max_ratio * second_best_distance) {
      continue;
    }

    matches[static_cast<std::size_t>(query_idx)] = best_train_idx;
  }

  return matches;
}

std::vector<int> FindBestMatchesOneWayIndex(const DistanceMatrix& indices,
                                            const DistanceMatrix& l2_dists,
                                            const double max_ratio,
                                            const double max_distance) {
  const int max_l2_dist = static_cast<int>(
      kSqSiftDescriptorNorm * max_distance * max_distance);

  std::vector<int> matches(static_cast<std::size_t>(indices.rows()), -1);
  for (int query_idx = 0; query_idx < indices.rows(); ++query_idx) {
    int best_train_idx = -1;
    int best_l2_dist = std::numeric_limits<int>::max();
    int second_best_l2_dist = std::numeric_limits<int>::max();

    for (int neighbor_idx = 0; neighbor_idx < indices.cols(); ++neighbor_idx) {
      const int train_idx = indices(query_idx, neighbor_idx);
      const int l2_dist = l2_dists(query_idx, neighbor_idx);
      if (l2_dist < best_l2_dist) {
        best_train_idx = train_idx;
        second_best_l2_dist = best_l2_dist;
        best_l2_dist = l2_dist;
      } else if (l2_dist < second_best_l2_dist) {
        second_best_l2_dist = l2_dist;
      }
    }

    if (best_train_idx < 0) {
      continue;
    }
    if (best_l2_dist > max_l2_dist) {
      continue;
    }
    if (std::sqrt(static_cast<float>(best_l2_dist)) >=
        max_ratio * std::sqrt(static_cast<float>(second_best_l2_dist))) {
      continue;
    }

    matches[static_cast<std::size_t>(query_idx)] = best_train_idx;
  }
  return matches;
}

std::vector<int> BestMatchesBruteForce(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const DescriptorMatchingOptions& options) {
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return std::vector<int>(static_cast<std::size_t>(query_descriptors.rows()),
                            -1);
  }
  return FindBestMatchesOneWayBruteForce(query_descriptors, train_descriptors,
                                         options.max_ratio,
                                         options.max_distance);
}

std::vector<int> BestMatchesFlann(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const DescriptorMatchingOptions& options) {
  std::vector<int> best(static_cast<std::size_t>(query_descriptors.rows()), -1);
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return best;
  }

  const int neighbor_count = std::min<int>(2, train_descriptors.rows());
  if (neighbor_count <= 0) {
    return best;
  }

  flann::Matrix<std::uint8_t> train_matrix(
      const_cast<std::uint8_t*>(train_descriptors.data()),
      static_cast<std::size_t>(train_descriptors.rows()),
      static_cast<std::size_t>(train_descriptors.cols()));
  flann::Index<flann::L2<std::uint8_t>> index(
      train_matrix, flann::KDTreeIndexParams(kFlannNumTrees));
  index.buildIndex();

  flann::Matrix<std::uint8_t> query_matrix(
      const_cast<std::uint8_t*>(query_descriptors.data()),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(query_descriptors.cols()));

  DistanceMatrix indices(query_descriptors.rows(), neighbor_count);
  DistanceMatrix l2_dists(query_descriptors.rows(), neighbor_count);
  std::vector<float> l2_dist_storage(
      static_cast<std::size_t>(query_descriptors.rows()) *
      static_cast<std::size_t>(neighbor_count));

  flann::Matrix<int> indices_matrix(
      indices.data(), static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));
  flann::Matrix<float> l2_dist_matrix(
      l2_dist_storage.data(),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));

  index.knnSearch(query_matrix,
                  indices_matrix,
                  l2_dist_matrix,
                  neighbor_count,
                  flann::SearchParams(kFlannNumLeavesToVisit));

  for (int row = 0; row < query_descriptors.rows(); ++row) {
    for (int col = 0; col < neighbor_count; ++col) {
      const std::size_t offset = static_cast<std::size_t>(row) *
                                     static_cast<std::size_t>(neighbor_count) +
                                 static_cast<std::size_t>(col);
      l2_dists(row, col) =
          static_cast<int>(std::round(l2_dist_storage[offset]));
    }
  }

  return FindBestMatchesOneWayIndex(indices, l2_dists, options.max_ratio,
                                    options.max_distance);
}

std::vector<int> BestMatches(const DescriptorByteMatrix& query_descriptors,
                             const DescriptorByteMatrix& train_descriptors,
                             const DescriptorMatchingOptions& options) {
  if (options.backend == DescriptorMatcherBackend::kBruteForce) {
    return BestMatchesBruteForce(query_descriptors, train_descriptors, options);
  }
  if (options.backend == DescriptorMatcherBackend::kFlann) {
    return BestMatchesFlann(query_descriptors, train_descriptors, options);
  }
  throw std::invalid_argument("Unknown descriptor matcher backend.");
}

}  // namespace

std::vector<FeatureMatch> MatchFeatureSets(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const DescriptorMatchingOptions& options) {
  return MatchDescriptors(features1.descriptors, features2.descriptors, options);
}

std::vector<FeatureMatch> MatchDescriptors(
    const DescriptorMatrix& descriptors1,
    const DescriptorMatrix& descriptors2,
    const DescriptorMatchingOptions& options) {
  if (options.max_ratio <= 0.0) {
    throw std::invalid_argument("max_ratio must be positive.");
  }
  if (options.max_distance <= 0.0) {
    throw std::invalid_argument("max_distance must be positive.");
  }
  if (descriptors1.cols() != descriptors2.cols() &&
      descriptors1.size() != 0 && descriptors2.size() != 0) {
    throw std::invalid_argument("Descriptor dimensions do not match.");
  }
  if (descriptors1.size() == 0 || descriptors2.size() == 0) {
    return {};
  }

  const DescriptorByteMatrix quantized1 = QuantizeDescriptors(descriptors1);
  const DescriptorByteMatrix quantized2 = QuantizeDescriptors(descriptors2);

  const std::vector<int> forward = BestMatches(quantized1, quantized2, options);
  std::vector<int> backward;
  if (options.mutual_check) {
    backward = BestMatches(quantized2, quantized1, options);
  }

  std::vector<FeatureMatch> matches;
  for (int idx1 = 0; idx1 < static_cast<int>(forward.size()); ++idx1) {
    const int idx2 = forward[static_cast<std::size_t>(idx1)];
    if (idx2 < 0) {
      continue;
    }
    if (options.mutual_check &&
        backward[static_cast<std::size_t>(idx2)] != idx1) {
      continue;
    }
    matches.push_back({static_cast<Point2DIdx>(idx1),
                       static_cast<Point2DIdx>(idx2)});
  }
  return matches;
}

}  // namespace colite
