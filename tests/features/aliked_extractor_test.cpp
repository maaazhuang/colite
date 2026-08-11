#include "colite/features/aliked_extractor.h"

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(AlikedExtractorTest, PreparesRgbChwAndReplicatePadding) {
  cv::Mat image(1, 2, CV_8UC3);
  image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
  image.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
  const AlikedImageTensor tensor = PrepareAlikedImageTensor(image);
  ASSERT_EQ(tensor.original_width, 2);
  ASSERT_EQ(tensor.original_height, 1);
  ASSERT_EQ(tensor.padded_width, 32);
  ASSERT_EQ(tensor.padded_height, 32);
  const std::size_t plane = 32U * 32U;
  EXPECT_NEAR(tensor.data[0], 30.0F / 255.0F, 1e-7F);
  EXPECT_NEAR(tensor.data[1], 60.0F / 255.0F, 1e-7F);
  EXPECT_NEAR(tensor.data[31], 60.0F / 255.0F, 1e-7F);
  EXPECT_NEAR(tensor.data[plane], 20.0F / 255.0F, 1e-7F);
  EXPECT_NEAR(tensor.data[2 * plane], 10.0F / 255.0F, 1e-7F);
  EXPECT_NEAR(tensor.data[31 * 32 + 31], 60.0F / 255.0F, 1e-7F);
}

TEST(AlikedExtractorTest, RejectsInvalidInputAndOptions) {
  EXPECT_THROW(PrepareAlikedImageTensor(cv::Mat()), std::invalid_argument);
  cv::Mat image(32, 32, CV_8UC3, cv::Scalar::all(0));
  AlikedExtractionOptions options;
  options.extractor_type = FeatureExtractorType::kSift;
  EXPECT_THROW(ExtractAlikedFeatures(image, options), std::invalid_argument);
}

TEST(AlikedExtractorTest, ReportsCompileTimeOnnxAvailability) {
#if defined(COLITE_ENABLE_ONNX)
  EXPECT_TRUE(IsOnnxRuntimeAvailable());
#else
  EXPECT_FALSE(IsOnnxRuntimeAvailable());
#endif
}

}  // namespace
}  // namespace colite
