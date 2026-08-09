// Generic inverse-mapping image undistortion. The camera sizing behavior is
// based on the BSD-licensed COLMAP image undistorter, rewritten around
// COLITE's camera projection API so all implemented camera models are handled.

#include "colite/mvs/image_undistorter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace colite::mvs {
namespace {

Camera MakePinholeCamera(const Camera& source, double focal_scale) {
  const Eigen::Matrix3d calibration = source.CalibrationMatrix();
  Camera output;
  output.id = source.id;
  output.model_id = CameraModelId::kPinhole;
  output.width = source.width;
  output.height = source.height;
  output.params = {calibration(0, 0) * focal_scale,
                   calibration(1, 1) * focal_scale, calibration(0, 2),
                   calibration(1, 2)};
  output.has_prior_focal_length = source.has_prior_focal_length;
  return output;
}

void RescaleCamera(double scale, Camera* camera) {
  const std::uint32_t old_width = camera->width;
  const std::uint32_t old_height = camera->height;
  camera->width = static_cast<std::uint32_t>(
      std::max(1.0, std::round(old_width * scale)));
  camera->height = static_cast<std::uint32_t>(
      std::max(1.0, std::round(old_height * scale)));
  const double scale_x = static_cast<double>(camera->width) / old_width;
  const double scale_y = static_cast<double>(camera->height) / old_height;
  camera->params[0] *= scale_x;
  camera->params[1] *= scale_y;
  camera->params[2] *= scale_x;
  camera->params[3] *= scale_y;
}

}  // namespace

bool ImageUndistortionOptions::Check() const {
  return std::isfinite(blank_pixels) && blank_pixels >= 0.0 &&
         blank_pixels <= 1.0 && std::isfinite(min_scale) &&
         std::isfinite(max_scale) && min_scale > 0.0 &&
         min_scale <= max_scale && max_image_size != 0 &&
         resolution_level <= 30;
}

Camera ComputeUndistortedCamera(const Camera& camera,
                                const ImageUndistortionOptions& options) {
  if (!options.Check() || !camera.VerifyParams() || camera.width == 0 ||
      camera.height == 0 ||
      !CameraModelProjectionImplemented(camera.model_id)) {
    throw std::invalid_argument("Invalid image undistortion input.");
  }

  Camera output = MakePinholeCamera(camera, 1.0);
  const CameraModelSpec& spec = GetCameraModelSpec(camera.model_id);
  if (!spec.distortion_idxs.empty()) {
    double left_min = std::numeric_limits<double>::max();
    double left_max = std::numeric_limits<double>::lowest();
    double right_min = std::numeric_limits<double>::max();
    double right_max = std::numeric_limits<double>::lowest();
    for (std::uint32_t y = 0; y < camera.height; ++y) {
      const auto left = UndistortPixel(
          camera, output, Eigen::Vector2d(0.5, y + 0.5));
      const auto right = UndistortPixel(
          camera, output,
          Eigen::Vector2d(static_cast<double>(camera.width) - 0.5, y + 0.5));
      if (left.has_value()) {
        left_min = std::min(left_min, left->x());
        left_max = std::max(left_max, left->x());
      }
      if (right.has_value()) {
        right_min = std::min(right_min, right->x());
        right_max = std::max(right_max, right->x());
      }
    }

    double top_min = std::numeric_limits<double>::max();
    double top_max = std::numeric_limits<double>::lowest();
    double bottom_min = std::numeric_limits<double>::max();
    double bottom_max = std::numeric_limits<double>::lowest();
    for (std::uint32_t x = 0; x < camera.width; ++x) {
      const auto top = UndistortPixel(
          camera, output, Eigen::Vector2d(x + 0.5, 0.5));
      const auto bottom = UndistortPixel(
          camera, output,
          Eigen::Vector2d(x + 0.5,
                          static_cast<double>(camera.height) - 0.5));
      if (top.has_value()) {
        top_min = std::min(top_min, top->y());
        top_max = std::max(top_max, top->y());
      }
      if (bottom.has_value()) {
        bottom_min = std::min(bottom_min, bottom->y());
        bottom_max = std::max(bottom_max, bottom->y());
      }
    }

    const bool valid_bounds =
        std::isfinite(left_min) && std::isfinite(left_max) &&
        std::isfinite(right_min) && std::isfinite(right_max) &&
        std::isfinite(top_min) && std::isfinite(top_max) &&
        std::isfinite(bottom_min) && std::isfinite(bottom_max);
    if (!valid_bounds) {
      throw std::runtime_error("Failed to determine undistorted boundaries.");
    }

    const double cx = output.params[2];
    const double cy = output.params[3];
    const double right_extent = static_cast<double>(output.width) - 0.5 - cx;
    const double bottom_extent = static_cast<double>(output.height) - 0.5 - cy;
    const double contain_x =
        std::min(cx / (cx - left_min), right_extent / (right_max - cx));
    const double contain_y =
        std::min(cy / (cy - top_min), bottom_extent / (bottom_max - cy));
    const double crop_x =
        std::max(cx / (cx - left_max), right_extent / (right_min - cx));
    const double crop_y =
        std::max(cy / (cy - top_max), bottom_extent / (bottom_min - cy));
    double scale_x = 1.0 / (contain_x * options.blank_pixels +
                            crop_x * (1.0 - options.blank_pixels));
    double scale_y = 1.0 / (contain_y * options.blank_pixels +
                            crop_y * (1.0 - options.blank_pixels));
    scale_x = std::clamp(scale_x, options.min_scale, options.max_scale);
    scale_y = std::clamp(scale_y, options.min_scale, options.max_scale);

    const std::uint32_t original_width = output.width;
    const std::uint32_t original_height = output.height;
    output.width = static_cast<std::uint32_t>(std::max(
        1.0, scale_x * static_cast<double>(original_width)));
    output.height = static_cast<std::uint32_t>(std::max(
        1.0, scale_y * static_cast<double>(original_height)));
    output.params[2] *= static_cast<double>(output.width) / original_width;
    output.params[3] *= static_cast<double>(output.height) / original_height;
  }
  if (options.resolution_level > 0) {
    RescaleCamera(std::ldexp(1.0, -static_cast<int>(options.resolution_level)),
                  &output);
  }
  if (options.max_image_size > 0) {
    const double scale =
        static_cast<double>(options.max_image_size) /
        static_cast<double>(std::max(output.width, output.height));
    if (scale < 1.0) RescaleCamera(scale, &output);
  }
  return output;
}

std::optional<Eigen::Vector2d> UndistortPixel(
    const Camera& source_camera,
    const Camera& undistorted_camera,
    const Eigen::Vector2d& source_pixel) {
  if (!source_pixel.allFinite() ||
      undistorted_camera.model_id != CameraModelId::kPinhole ||
      !source_camera.VerifyParams() || !undistorted_camera.VerifyParams()) {
    return std::nullopt;
  }
  Eigen::Vector3d ray;
  if (!UnprojectPixelWithCameraParams(
          source_camera.model_id, source_camera.params.data(), source_pixel,
          1.0, &ray)) {
    return std::nullopt;
  }
  Eigen::Vector2d output;
  if (!ProjectPointWithCameraParams(
          undistorted_camera.model_id, undistorted_camera.params.data(), ray,
          &output)) {
    return std::nullopt;
  }
  return output;
}

UndistortedImage UndistortImage(const Camera& camera,
                                const cv::Mat& image,
                                const ImageUndistortionOptions& options) {
  if (image.empty() || image.depth() != CV_8U ||
      (image.channels() != 1 && image.channels() != 3) ||
      image.cols != static_cast<int>(camera.width) ||
      image.rows != static_cast<int>(camera.height)) {
    throw std::invalid_argument(
        "Undistortion requires a matching 8-bit grayscale or BGR image.");
  }
  UndistortedImage result;
  result.camera = ComputeUndistortedCamera(camera, options);
  // Match COLMAP's anti-aliasing path: warp at the distorted source
  // resolution first and only then downsample to the requested output size.
  Camera warp_camera = result.camera;
  const double warp_scale_x =
      static_cast<double>(camera.width) / result.camera.width;
  const double warp_scale_y =
      static_cast<double>(camera.height) / result.camera.height;
  warp_camera.width = camera.width;
  warp_camera.height = camera.height;
  warp_camera.params[0] *= warp_scale_x;
  warp_camera.params[1] *= warp_scale_y;
  warp_camera.params[2] *= warp_scale_x;
  warp_camera.params[3] *= warp_scale_y;

  cv::Mat map_x(static_cast<int>(camera.height),
                static_cast<int>(camera.width), CV_32FC1);
  cv::Mat map_y(map_x.size(), CV_32FC1);
  cv::Mat warp_valid_mask(map_x.size(), CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < map_x.rows; ++y) {
    for (int x = 0; x < map_x.cols; ++x) {
      const Eigen::Vector3d ray(
          (x + 0.5 - warp_camera.params[2]) / warp_camera.params[0],
          (y + 0.5 - warp_camera.params[3]) / warp_camera.params[1], 1.0);
      Eigen::Vector2d source_pixel(-1.0, -1.0);
      try {
        source_pixel = ProjectPoint(camera, ray);
      } catch (const std::exception&) {
      }
      // COLMAP camera coordinates use pixel centers at (0.5, 0.5), while
      // OpenCV remap indexes the first sample at (0, 0).
      map_x.at<float>(y, x) = static_cast<float>(source_pixel.x() - 0.5);
      map_y.at<float>(y, x) = static_cast<float>(source_pixel.y() - 0.5);
      if (source_pixel.allFinite() && source_pixel.x() >= 0.5 &&
          source_pixel.y() >= 0.5 && source_pixel.x() <= image.cols - 0.5 &&
          source_pixel.y() <= image.rows - 0.5) {
        warp_valid_mask.at<std::uint8_t>(y, x) = 255;
      }
    }
  }
  cv::Mat warped_image;
  cv::remap(image, warped_image, map_x, map_y, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar::all(0));
  const cv::Size output_size(static_cast<int>(result.camera.width),
                             static_cast<int>(result.camera.height));
  if (warped_image.size() == output_size) {
    result.image = std::move(warped_image);
    result.valid_mask = std::move(warp_valid_mask);
  } else {
    cv::resize(warped_image, result.image, output_size, 0.0, 0.0,
               cv::INTER_AREA);
    cv::resize(warp_valid_mask, result.valid_mask, output_size, 0.0, 0.0,
               cv::INTER_NEAREST);
  }
  const std::size_t valid_pixels =
      static_cast<std::size_t>(cv::countNonZero(result.valid_mask));
  result.valid_pixel_ratio =
      result.valid_mask.total() == 0
          ? 0.0
          : static_cast<double>(valid_pixels) /
                static_cast<double>(result.valid_mask.total());
  return result;
}

}  // namespace colite::mvs
