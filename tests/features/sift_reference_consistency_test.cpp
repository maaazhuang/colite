#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#define VlSiftFilt OrigVlSiftFilt
#define VlSiftKeypoint OrigVlSiftKeypoint
#define vl_sift_new orig_vl_sift_new
#define vl_sift_delete orig_vl_sift_delete
#define vl_sift_process_first_octave orig_vl_sift_process_first_octave
#define vl_sift_process_next_octave orig_vl_sift_process_next_octave
#define vl_sift_detect orig_vl_sift_detect
#define vl_sift_calc_keypoint_orientations \
  orig_vl_sift_calc_keypoint_orientations
#define vl_sift_calc_keypoint_descriptor orig_vl_sift_calc_keypoint_descriptor
#define vl_sift_calc_raw_descriptor orig_vl_sift_calc_raw_descriptor
#define vl_sift_keypoint_init orig_vl_sift_keypoint_init
#define vl_sift_get_octave_index orig_vl_sift_get_octave_index
#define vl_sift_get_noctaves orig_vl_sift_get_noctaves
#define vl_sift_get_octave_first orig_vl_sift_get_octave_first
#define vl_sift_get_octave_width orig_vl_sift_get_octave_width
#define vl_sift_get_octave_height orig_vl_sift_get_octave_height
#define vl_sift_get_nlevels orig_vl_sift_get_nlevels
#define vl_sift_get_nkeypoints orig_vl_sift_get_nkeypoints
#define vl_sift_get_peak_thresh orig_vl_sift_get_peak_thresh
#define vl_sift_get_edge_thresh orig_vl_sift_get_edge_thresh
#define vl_sift_get_norm_thresh orig_vl_sift_get_norm_thresh
#define vl_sift_get_magnif orig_vl_sift_get_magnif
#define vl_sift_get_window_size orig_vl_sift_get_window_size
#define vl_sift_get_octave orig_vl_sift_get_octave
#define vl_sift_get_keypoints orig_vl_sift_get_keypoints
#define vl_sift_set_peak_thresh orig_vl_sift_set_peak_thresh
#define vl_sift_set_edge_thresh orig_vl_sift_set_edge_thresh
#define vl_sift_set_norm_thresh orig_vl_sift_set_norm_thresh
#define vl_sift_set_magnif orig_vl_sift_set_magnif
#define vl_sift_set_window_size orig_vl_sift_set_window_size
#include "sift.h"
#undef vl_sift_set_window_size
#undef vl_sift_set_magnif
#undef vl_sift_set_norm_thresh
#undef vl_sift_set_edge_thresh
#undef vl_sift_set_peak_thresh
#undef vl_sift_get_keypoints
#undef vl_sift_get_octave
#undef vl_sift_get_window_size
#undef vl_sift_get_magnif
#undef vl_sift_get_norm_thresh
#undef vl_sift_get_edge_thresh
#undef vl_sift_get_peak_thresh
#undef vl_sift_get_nkeypoints
#undef vl_sift_get_nlevels
#undef vl_sift_get_octave_height
#undef vl_sift_get_octave_width
#undef vl_sift_get_octave_first
#undef vl_sift_get_noctaves
#undef vl_sift_get_octave_index
#undef vl_sift_keypoint_init
#undef vl_sift_calc_raw_descriptor
#undef vl_sift_calc_keypoint_descriptor
#undef vl_sift_calc_keypoint_orientations
#undef vl_sift_detect
#undef vl_sift_process_next_octave
#undef vl_sift_process_first_octave
#undef vl_sift_delete
#undef vl_sift_new
#undef VlSiftKeypoint
#undef VlSiftFilt

extern "C" {
#include "sift_vlfeat.h"
}

namespace colite {
namespace {

constexpr double kVlFeatPositionTolerance = 1.5;
constexpr double kVlFeatScaleRelativeTolerance = 0.08;
constexpr double kVlFeatAngleTolerance = 8.0 * 3.14159265358979323846 / 180.0;
constexpr double kTwoPi = 6.28318530717958647692;

double AngleDistance(double lhs, double rhs) {
  const double delta = std::abs(lhs - rhs);
  return std::min(delta, kTwoPi - std::min(delta, kTwoPi));
}

struct VlFeature {
  int octave = 0;
  int ix = 0;
  int iy = 0;
  int is = 0;
  float x = 0.0f;
  float y = 0.0f;
  float s = 0.0f;
  float sigma = 0.0f;
  double angle = 0.0;
  std::vector<float> descriptor;
};

cv::Mat LoadReferenceGrayImage() {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    ADD_FAILURE() << "Failed to read " << image_path.string();
    return {};
  }

  cv::Mat resized;
  const double scale = 512.0 / static_cast<double>(image.cols);
  cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
  return resized;
}

std::vector<float> MatToFloatVector(const cv::Mat& gray) {
  std::vector<float> result;
  result.reserve(static_cast<size_t>(gray.rows * gray.cols));
  for (int y = 0; y < gray.rows; ++y) {
    const uint8_t* row = gray.ptr<uint8_t>(y);
    for (int x = 0; x < gray.cols; ++x) {
      result.push_back(static_cast<float>(row[x]));
    }
  }
  return result;
}

std::vector<VlFeature> ExtractOriginalVlSift(const cv::Mat& gray) {
  const std::vector<float> image = MatToFloatVector(gray);
  OrigVlSiftFilt* filt =
      orig_vl_sift_new(gray.cols, gray.rows, -1, 3, -1);
  orig_vl_sift_set_peak_thresh(filt, 0.0);
  orig_vl_sift_set_edge_thresh(filt, 10.0);

  std::vector<VlFeature> features;
  int err = orig_vl_sift_process_first_octave(filt, image.data());
  while (err == VL_ERR_OK) {
    orig_vl_sift_detect(filt);
    const OrigVlSiftKeypoint* keypoints = orig_vl_sift_get_keypoints(filt);
    const int num_keypoints = orig_vl_sift_get_nkeypoints(filt);
    for (int i = 0; i < num_keypoints; ++i) {
      double angles[4] = {};
      const int num_angles =
          orig_vl_sift_calc_keypoint_orientations(filt, angles,
                                                  keypoints + i);
      for (int j = 0; j < num_angles; ++j) {
        VlFeature feature;
        feature.octave = keypoints[i].o;
        feature.ix = keypoints[i].ix;
        feature.iy = keypoints[i].iy;
        feature.is = keypoints[i].is;
        feature.x = keypoints[i].x;
        feature.y = keypoints[i].y;
        feature.s = keypoints[i].s;
        feature.sigma = keypoints[i].sigma;
        feature.angle = angles[j];
        feature.descriptor.assign(VL_SIFT_DESCR_LEN, 0.0f);
        orig_vl_sift_calc_keypoint_descriptor(
            filt, feature.descriptor.data(), keypoints + i, angles[j]);
        features.push_back(std::move(feature));
      }
    }
    err = orig_vl_sift_process_next_octave(filt);
  }

  orig_vl_sift_delete(filt);
  return features;
}

std::vector<VlFeature> ExtractModifiedVlSift(const cv::Mat& gray) {
  const std::vector<float> image = MatToFloatVector(gray);
  VlSiftFilt* filt = vl_sift_new(gray.cols, gray.rows, -1, 3, -1);
  vl_sift_set_peak_thresh(filt, 0.0);
  vl_sift_set_edge_thresh(filt, 10.0);

  std::vector<VlFeature> features;
  int err = vl_sift_process_first_octave(filt, image.data());
  while (err == VL_ERR_OK) {
    vl_sift_detect(filt);
    const VlSiftKeypoint* keypoints = vl_sift_get_keypoints(filt);
    const int num_keypoints = vl_sift_get_nkeypoints(filt);
    for (int i = 0; i < num_keypoints; ++i) {
      double angles[4] = {};
      const int num_angles =
          vl_sift_calc_keypoint_orientations(filt, angles, keypoints + i);
      for (int j = 0; j < num_angles; ++j) {
        VlFeature feature;
        feature.octave = keypoints[i].o;
        feature.ix = keypoints[i].ix;
        feature.iy = keypoints[i].iy;
        feature.is = keypoints[i].is;
        feature.x = keypoints[i].x;
        feature.y = keypoints[i].y;
        feature.s = keypoints[i].s;
        feature.sigma = keypoints[i].sigma;
        feature.angle = angles[j];
        feature.descriptor.assign(VL_SIFT_DESCR_LEN, 0.0f);
        vl_sift_calc_keypoint_descriptor(filt, feature.descriptor.data(),
                                         keypoints + i, angles[j]);
        features.push_back(std::move(feature));
      }
    }
    err = vl_sift_process_next_octave(filt);
  }

  vl_sift_delete(filt);
  return features;
}

void SortVlFeatures(std::vector<VlFeature>* features) {
  std::sort(features->begin(), features->end(), [](const auto& a,
                                                   const auto& b) {
    if (std::abs(a.x - b.x) > 1e-6f) return a.x < b.x;
    if (std::abs(a.y - b.y) > 1e-6f) return a.y < b.y;
    if (std::abs(a.sigma - b.sigma) > 1e-6f) return a.sigma < b.sigma;
    return a.angle < b.angle;
  });
}

bool IsSameVlFeatureLocation(const VlFeature& a, const VlFeature& b) {
  const double position = std::hypot(a.x - b.x, a.y - b.y);
  const double scale_relative =
      std::abs(a.sigma - b.sigma) /
      std::max(1e-12, static_cast<double>(a.sigma));
  return a.octave == b.octave && a.ix == b.ix && a.iy == b.iy &&
         a.is == b.is && position <= kVlFeatPositionTolerance &&
         scale_relative <= kVlFeatScaleRelativeTolerance;
}

TEST(SiftReferenceConsistencyTest, VlFeatVariantMatchesOriginal) {
  const cv::Mat image = LoadReferenceGrayImage();
  ASSERT_FALSE(image.empty());

  std::vector<VlFeature> original = ExtractOriginalVlSift(image);
  std::vector<VlFeature> modified = ExtractModifiedVlSift(image);
  SortVlFeatures(&original);
  SortVlFeatures(&modified);

  ASSERT_FALSE(original.empty());
  std::vector<unsigned char> used(modified.size(), 0);
  std::size_t matched = 0;
  double max_descriptor_linf = 0.0;
  double descriptor_l2_sum = 0.0;
  double angle_error_sum = 0.0;
  for (const VlFeature& reference : original) {
    std::size_t best = modified.size();
    double best_error = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < modified.size(); ++i) {
      if (used[i] || !IsSameVlFeatureLocation(reference, modified[i])) continue;
      const double error = std::hypot(reference.x - modified[i].x,
                                      reference.y - modified[i].y) +
                           std::abs(reference.sigma - modified[i].sigma) +
                           AngleDistance(reference.angle, modified[i].angle);
      if (error < best_error) {
        best = i;
        best_error = error;
      }
    }
    if (best == modified.size()) continue;
    used[best] = 1;
    ++matched;
    angle_error_sum += AngleDistance(reference.angle, modified[best].angle);
    double descriptor_l2 = 0.0;
    for (int j = 0; j < VL_SIFT_DESCR_LEN; ++j) {
      const double difference =
          static_cast<double>(modified[best].descriptor[j]) -
          static_cast<double>(reference.descriptor[j]);
      max_descriptor_linf =
          std::max(max_descriptor_linf, std::abs(difference));
      descriptor_l2 += difference * difference;
    }
    descriptor_l2_sum += std::sqrt(descriptor_l2);
  }

  const double match_ratio =
      static_cast<double>(matched) / static_cast<double>(original.size());
  EXPECT_GE(match_ratio, 0.999)
      << "matched=" << matched << " original=" << original.size()
      << " modified=" << modified.size();
  ASSERT_GT(matched, std::size_t{0});
  EXPECT_LE(angle_error_sum / static_cast<double>(matched),
            kVlFeatAngleTolerance);
  EXPECT_LE(descriptor_l2_sum / static_cast<double>(matched), 0.05);
  EXPECT_LE(max_descriptor_linf, 0.25);
}

}  // namespace
}  // namespace colite
