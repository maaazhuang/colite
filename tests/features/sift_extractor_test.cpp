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
