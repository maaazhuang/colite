#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <flann/flann.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace colite {
namespace {

cv::Mat WrapAsCvMat(const DescriptorMatrix& descriptors) {
  return cv::Mat(static_cast<int>(descriptors.rows()),
                 static_cast<int>(descriptors.cols()),
                 CV_32F,
                 const_cast<float*>(descriptors.data()));
}

bool PassRatio(double best_distance, double second_distance, double max_ratio) {
  if (second_distance <= std::numeric_limits<double>::epsilon()) {
    return best_distance <= std::numeric_limits<double>::epsilon();
  }
  return best_distance <= max_ratio * second_distance;
}

std::vector<int> BestMatchesOpenCV(const DescriptorMatrix& query_descriptors,
                                   const DescriptorMatrix& train_descriptors,
                                   double max_ratio) {
  std::vector<int> best(static_cast<std::size_t>(query_descriptors.rows()), -1);
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return best;
  }

  const cv::Mat query = WrapAsCvMat(query_descriptors);
  const cv::Mat train = WrapAsCvMat(train_descriptors);
  cv::BFMatcher matcher(cv::NORM_L2, false);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher.knnMatch(query, train, knn_matches, 2);

  for (const std::vector<cv::DMatch>& candidates : knn_matches) {
    if (candidates.size() < 2) {
      continue;
    }
    const cv::DMatch& first = candidates[0];
    const cv::DMatch& second = candidates[1];
    if (PassRatio(first.distance, second.distance, max_ratio)) {
      best[static_cast<std::size_t>(first.queryIdx)] = first.trainIdx;
    }
  }
  return best;
}

std::vector<int> BestMatchesFlann(const DescriptorMatrix& query_descriptors,
                                  const DescriptorMatrix& train_descriptors,
                                  double max_ratio) {
  std::vector<int> best(static_cast<std::size_t>(query_descriptors.rows()), -1);
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return best;
  }

  const int neighbor_count = std::min<int>(2, train_descriptors.rows());
  if (neighbor_count < 2) {
    return best;
  }

  flann::Matrix<float> train_matrix(
      const_cast<float*>(train_descriptors.data()),
      static_cast<std::size_t>(train_descriptors.rows()),
      static_cast<std::size_t>(train_descriptors.cols()));
  flann::Index<flann::L2<float>> index(train_matrix,
                                       flann::KDTreeIndexParams(4));
  index.buildIndex();

  flann::Matrix<float> query_matrix(
      const_cast<float*>(query_descriptors.data()),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(query_descriptors.cols()));
  std::vector<int> indices_storage(
      static_cast<std::size_t>(query_descriptors.rows()) *
      static_cast<std::size_t>(neighbor_count));
  std::vector<float> distances_storage(indices_storage.size());
  flann::Matrix<int> indices_matrix(
      indices_storage.data(),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));
  flann::Matrix<float> distances_matrix(
      distances_storage.data(),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));

  index.knnSearch(query_matrix,
                  indices_matrix,
                  distances_matrix,
                  neighbor_count,
                  flann::SearchParams(128));

  for (int row = 0; row < query_descriptors.rows(); ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) *
                               static_cast<std::size_t>(neighbor_count);
    const float first_distance = distances_storage[offset];
    const float second_distance = distances_storage[offset + 1];
    if (PassRatio(first_distance, second_distance, max_ratio)) {
      best[static_cast<std::size_t>(row)] = indices_storage[offset];
    }
  }
  return best;
}

std::vector<int> BestMatches(const DescriptorMatrix& query_descriptors,
                             const DescriptorMatrix& train_descriptors,
                             const DescriptorMatchingOptions& options) {
  if (options.backend == DescriptorMatcherBackend::kOpenCVBruteForce) {
    return BestMatchesOpenCV(query_descriptors, train_descriptors,
                             options.max_ratio);
  }
  if (options.backend == DescriptorMatcherBackend::kFlann) {
    return BestMatchesFlann(query_descriptors, train_descriptors,
                            options.max_ratio);
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
  if (options.max_ratio <= 0.0 || options.max_ratio > 1.0) {
    throw std::invalid_argument("max_ratio must be in (0, 1].");
  }
  if (descriptors1.cols() != descriptors2.cols() &&
      descriptors1.size() != 0 && descriptors2.size() != 0) {
    throw std::invalid_argument("Descriptor dimensions do not match.");
  }

  if (descriptors1.size() == 0 || descriptors2.size() == 0) {
    return {};
  }

  const std::vector<int> forward = BestMatches(descriptors1, descriptors2,
                                               options);
  std::vector<int> backward;
  if (options.mutual_check) {
    backward = BestMatches(descriptors2, descriptors1, options);
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

