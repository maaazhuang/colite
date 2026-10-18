#include "colite/io/image_reader.h"

#include <cstdint>
#include <fstream>
#include <vector>
#include <stdexcept>

#include <gtest/gtest.h>
#include "colite/io/camera_prior.h"

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

TEST(ImageReaderTest, HeaderDimensionsMatchDecodedExifOrientations) {
  const auto root = std::filesystem::temp_directory_path() / "colite_header_dimensions_test";
  std::filesystem::create_directories(root);
  std::vector<unsigned char> encoded;
  ASSERT_TRUE(cv::imencode(".jpg", cv::Mat(9, 17, CV_8UC3, cv::Scalar(20, 40, 80)),
                          encoded, {cv::IMWRITE_JPEG_PROGRESSIVE, 1}));
  for (int orientation = 1; orientation <= 8; ++orientation) {
    // A non-EXIF APP1 followed by an orientation-only little-endian TIFF IFD.
    std::vector<unsigned char> header = {
        0xFF, 0xE1, 0, 8, 'x', 'm', 'p', '0', '0', '0',
        0xFF, 0xFF, 0xE1, 0, 34, 'E', 'x', 'i', 'f', 0, 0,
        'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0,
        0x12, 1, 3, 0, 1, 0, 0, 0,
        static_cast<unsigned char>(orientation), 0, 0, 0, 0, 0, 0, 0};
    const auto path = root / (std::to_string(orientation) + ".jpg");
    {
      std::ofstream file(path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(encoded.data()), 2);
      file.write(reinterpret_cast<const char*>(header.data()), header.size());
      file.write(reinterpret_cast<const char*>(encoded.data() + 2), encoded.size() - 2);
    }
    EXPECT_EQ(ReadExifOrientation(path), orientation);
    const auto expected_size = orientation < 5 ? cv::Size(17, 9) : cv::Size(9, 17);
    EXPECT_EQ(ReadImageSizeWithExifOrientation(path), expected_size);
    EXPECT_EQ(ReadImageWithExifOrientation(path).size(), expected_size);
  }
  const auto png = root / "fallback.png";
  ASSERT_TRUE(cv::imwrite(png.string(), LabeledImage()));
  EXPECT_EQ(ReadImageSizeWithExifOrientation(png), cv::Size(3, 2));
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace colite
