#include "colite/mvs/image_undistorter.h"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace colite::mvs {
namespace {

Camera MakeCamera(CameraModelId model_id) {
  Camera camera;
  camera.id = 1;
  camera.model_id = model_id;
  camera.width = 80;
  camera.height = 60;
  camera.params = InitializeCameraParams(model_id, 60.0, 80, 60);
  for (const std::size_t idx :
       GetCameraModelSpec(model_id).distortion_idxs) {
    camera.params[idx] = model_id == CameraModelId::kFOV ? 0.3 : 0.0005;
  }
  return camera;
}

cv::Mat MakeGradientImage() {
  cv::Mat image(60, 80, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<std::uint8_t>(x * 3),
          static_cast<std::uint8_t>(y * 4), 180);
    }
  }
  return image;
}

TEST(ImageUndistortionOptionsTest, ValidatesParameters) {
  ImageUndistortionOptions options;
  EXPECT_TRUE(options.Check());
  options.blank_pixels = 1.1;
  EXPECT_FALSE(options.Check());
  options.blank_pixels = 0.0;
  options.min_scale = 0.0;
  EXPECT_FALSE(options.Check());
  options.min_scale = 0.2;
  options.max_image_size = 0;
  EXPECT_FALSE(options.Check());
}

TEST(ImageUndistorterTest, SupportsEveryProjectedCameraModel) {
  for (int raw_model = static_cast<int>(CameraModelId::kSimplePinhole);
       raw_model <= static_cast<int>(CameraModelId::kRadTanThinPrismFisheye);
       ++raw_model) {
    const CameraModelId model = static_cast<CameraModelId>(raw_model);
    ASSERT_TRUE(CameraModelProjectionImplemented(model));
    const Camera output = ComputeUndistortedCamera(MakeCamera(model));
    EXPECT_EQ(output.model_id, CameraModelId::kPinhole);
    EXPECT_GT(output.width, 0U);
    EXPECT_GT(output.height, 0U);
    EXPECT_TRUE(output.VerifyParams());
    const Eigen::Vector3d ray(0.1, -0.08, 1.0);
    const Eigen::Vector2d distorted = ProjectPoint(MakeCamera(model), ray);
    const std::optional<Eigen::Vector2d> mapped =
        UndistortPixel(MakeCamera(model), output, distorted);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_TRUE(mapped->isApprox(ProjectPoint(output, ray), 1e-5));
  }
}

TEST(ImageUndistorterTest, PreservesPinholeImageGeometry) {
  const Camera camera = MakeCamera(CameraModelId::kPinhole);
  const cv::Mat source = MakeGradientImage();
  const UndistortedImage result = UndistortImage(camera, source);
  EXPECT_EQ(result.camera.width, camera.width);
  EXPECT_EQ(result.camera.height, camera.height);
  EXPECT_EQ(result.image.size(), source.size());
  EXPECT_GT(result.valid_pixel_ratio, 0.99);
  EXPECT_NEAR(cv::norm(result.image, source, cv::NORM_INF), 0.0, 1.0);
}

TEST(ImageUndistorterTest, RemovesRadialDistortionAndControlsBlankPixels) {
  Camera camera = MakeCamera(CameraModelId::kRadial);
  camera.params[3] = 0.2;
  camera.params[4] = 0.02;
  ImageUndistortionOptions cropped_options;
  cropped_options.blank_pixels = 0.0;
  const UndistortedImage cropped =
      UndistortImage(camera, MakeGradientImage(), cropped_options);
  EXPECT_EQ(cropped.camera.model_id, CameraModelId::kPinhole);
  EXPECT_GT(cropped.valid_pixel_ratio, 0.98);

  ImageUndistortionOptions wide_options = cropped_options;
  wide_options.blank_pixels = 0.2;
  const UndistortedImage wide =
      UndistortImage(camera, MakeGradientImage(), wide_options);
  EXPECT_LE(wide.camera.params[0], cropped.camera.params[0]);
  EXPECT_LE(wide.valid_pixel_ratio, cropped.valid_pixel_ratio);
}

TEST(ImageUndistorterTest, AppliesMaximumImageSize) {
  ImageUndistortionOptions options;
  options.max_image_size = 40;
  const UndistortedImage result =
      UndistortImage(MakeCamera(CameraModelId::kPinhole),
                     MakeGradientImage(), options);
  EXPECT_EQ(std::max(result.camera.width, result.camera.height), 40U);
  EXPECT_EQ(std::max(result.image.cols, result.image.rows), 40);
}

TEST(ImageUndistorterTest, RejectsMismatchedInputImage) {
  EXPECT_THROW(UndistortImage(MakeCamera(CameraModelId::kPinhole),
                              cv::Mat(10, 10, CV_8UC3)),
               std::invalid_argument);
}

}  // namespace
}  // namespace colite::mvs
