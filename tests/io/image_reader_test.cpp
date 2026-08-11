#include "colite/io/image_reader.h"

#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

namespace colite {
namespace {

cv::Mat LabeledImage() {
  return (cv::Mat_<std::uint8_t>(2, 3) << 1, 2, 3, 4, 5, 6);
}

TEST(ImageReaderTest, AppliesAllExifOrientations) {
  const cv::Mat image = LabeledImage();
  const cv::Mat orientation2 = ApplyExifOrientation(image, 2);
  EXPECT_EQ(orientation2.at<std::uint8_t>(0, 0), 3);
  EXPECT_EQ(orientation2.at<std::uint8_t>(1, 2), 4);
  const cv::Mat orientation3 = ApplyExifOrientation(image, 3);
  EXPECT_EQ(orientation3.at<std::uint8_t>(0, 0), 6);
  EXPECT_EQ(orientation3.at<std::uint8_t>(1, 2), 1);
  const cv::Mat orientation4 = ApplyExifOrientation(image, 4);
  EXPECT_EQ(orientation4.at<std::uint8_t>(0, 0), 4);
  EXPECT_EQ(orientation4.at<std::uint8_t>(1, 2), 3);
  const cv::Mat orientation5 = ApplyExifOrientation(image, 5);
  ASSERT_EQ(orientation5.rows, 3);
  ASSERT_EQ(orientation5.cols, 2);
  EXPECT_EQ(orientation5.at<std::uint8_t>(0, 1), 4);
  const cv::Mat orientation6 = ApplyExifOrientation(image, 6);
  EXPECT_EQ(orientation6.at<std::uint8_t>(0, 0), 4);
  EXPECT_EQ(orientation6.at<std::uint8_t>(2, 1), 3);
  const cv::Mat orientation7 = ApplyExifOrientation(image, 7);
  EXPECT_EQ(orientation7.at<std::uint8_t>(0, 0), 6);
  EXPECT_EQ(orientation7.at<std::uint8_t>(2, 1), 1);
  const cv::Mat orientation8 = ApplyExifOrientation(image, 8);
  EXPECT_EQ(orientation8.at<std::uint8_t>(0, 0), 3);
  EXPECT_EQ(orientation8.at<std::uint8_t>(2, 1), 4);
}

TEST(ImageReaderTest, RejectsInvalidOrientation) {
  EXPECT_THROW(ApplyExifOrientation(LabeledImage(), 0), std::invalid_argument);
  EXPECT_THROW(ApplyExifOrientation(LabeledImage(), 9), std::invalid_argument);
}

}  // namespace
}  // namespace colite
