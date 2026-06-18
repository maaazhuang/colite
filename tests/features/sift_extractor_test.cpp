#include "colite/features/sift_extractor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

namespace colite {
namespace {

cv::Mat MakeSyntheticImage() {
  cv::Mat image(256, 256, CV_8UC1, cv::Scalar(0));
  cv::circle(image, cv::Point(64, 64), 28, cv::Scalar(255), -1);
  cv::circle(image, cv::Point(180, 80), 18, cv::Scalar(180), 3);
  cv::rectangle(image, cv::Rect(80, 150, 80, 45), cv::Scalar(220), -1);
  cv::line(image, cv::Point(20, 230), cv::Point(230, 20), cv::Scalar(160), 4);
  return image;
}

TEST(SiftExtractorTest, ExtractsFeaturesFromSyntheticImage) {
  SiftExtractionOptions options;
  options.max_num_features = 256;

  const FeatureSet features = ExtractSiftFeatures(MakeSyntheticImage(), options);

  EXPECT_FALSE(features.keypoints.empty());
  EXPECT_GT(features.descriptors.size(), 0);
  EXPECT_EQ(features.descriptors.rows(),
            static_cast<Eigen::Index>(features.keypoints.size()));
  EXPECT_EQ(features.descriptors.cols(), 128);
}

TEST(SiftExtractorTest, ParsesBackendNames) {
  EXPECT_EQ(SiftBackendFromName("opencv"), SiftBackend::kOpenCV);
  EXPECT_EQ(SiftBackendFromName("vlsift"), SiftBackend::kVlSift);
  EXPECT_EQ(SiftBackendFromName("cvsift"), SiftBackend::kCvSift);
  EXPECT_THROW(SiftBackendFromName("unknown"), std::invalid_argument);
}

TEST(SiftExtractorTest, ExtractsFeaturesWithAllBackends) {
  SiftExtractionOptions options;
  options.max_num_features = 256;
  options.contrast_threshold = 0.0;

  for (const SiftBackend backend :
       {SiftBackend::kOpenCV, SiftBackend::kVlSift, SiftBackend::kCvSift}) {
    options.backend = backend;
    const FeatureSet features =
        ExtractSiftFeatures(MakeSyntheticImage(), options);

    EXPECT_FALSE(features.keypoints.empty()) << SiftBackendName(backend);
    EXPECT_GT(features.descriptors.size(), 0) << SiftBackendName(backend);
    EXPECT_EQ(features.descriptors.rows(),
              static_cast<Eigen::Index>(features.keypoints.size()))
        << SiftBackendName(backend);
    EXPECT_EQ(features.descriptors.cols(), 128) << SiftBackendName(backend);
  }
}

TEST(SiftExtractorTest, VlSiftKeepsCompleteScaleLevelsLikeColmap) {
  SiftExtractionOptions options;
  options.backend = SiftBackend::kVlSift;
  options.max_num_features = 10;
  options.contrast_threshold = 0.0;

  const FeatureSet features = ExtractSiftFeatures(MakeSyntheticImage(), options);

  EXPECT_GT(features.keypoints.size(), static_cast<std::size_t>(10));
  EXPECT_EQ(features.descriptors.rows(),
            static_cast<Eigen::Index>(features.keypoints.size()));
}

TEST(SiftExtractorTest, AppliesRootSiftNormalizationByDefault) {
  SiftExtractionOptions options;
  options.max_num_features = 256;

  const FeatureSet features = ExtractSiftFeatures(MakeSyntheticImage(), options);

  ASSERT_GT(features.descriptors.rows(), 0);
  for (Eigen::Index row = 0; row < features.descriptors.rows(); ++row) {
    EXPECT_NEAR(features.descriptors.row(row).norm(), 1.0, 1e-4);
  }
}

}  // namespace
}  // namespace colite
