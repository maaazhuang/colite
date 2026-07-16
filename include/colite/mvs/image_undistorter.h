#pragma once

#include <optional>

#include <opencv2/core.hpp>

#include "colite/core/camera.h"

namespace colite::mvs {

struct ImageUndistortionOptions {
  // Approximate permitted fraction of output pixels without a source sample.
  double blank_pixels = 0.0;
  double min_scale = 0.2;
  double max_scale = 2.0;
  int max_image_size = -1;

  bool Check() const;
};

struct UndistortedImage {
  Camera camera;
  cv::Mat image;
  cv::Mat valid_mask;
  double valid_pixel_ratio = 0.0;
};

Camera ComputeUndistortedCamera(
    const Camera& camera,
    const ImageUndistortionOptions& options = {});

// Maps a pixel from an arbitrary supported source model into an undistorted
// PINHOLE camera. Uses a small numerical solve and supports all projected
// camera models, including those without a closed-form unprojection.
std::optional<Eigen::Vector2d> UndistortPixel(
    const Camera& source_camera,
    const Camera& undistorted_camera,
    const Eigen::Vector2d& source_pixel);

// Uses inverse mapping from each output PINHOLE ray into the original camera.
// The source may be an 8-bit grayscale or BGR image.
UndistortedImage UndistortImage(
    const Camera& camera,
    const cv::Mat& image,
    const ImageUndistortionOptions& options = {});

}  // namespace colite::mvs
