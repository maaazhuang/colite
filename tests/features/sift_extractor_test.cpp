#include "colite/features/sift_extractor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <tuple>
#include <vector>

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

cv::Mat MakeDenseSyntheticImage() {
  cv::Mat image(384, 384, CV_8UC1, cv::Scalar(0));
  for (int y = 32; y < image.rows - 32; y += 48) {
    for (int x = 32; x < image.cols - 32; x += 48) {
      const int radius = ((x + y) / 48) % 2 == 0 ? 12 : 18;
      cv::circle(image, cv::Point(x, y), radius, cv::Scalar(220), -1);
      cv::line(image, cv::Point(x - 18, y), cv::Point(x + 18, y),
               cv::Scalar(80), 2);
      cv::line(image, cv::Point(x, y - 18), cv::Point(x, y + 18),
               cv::Scalar(80), 2);
    }
  }
  cv::rectangle(image, cv::Rect(42, 268, 130, 44), cv::Scalar(255), -1);
  cv::line(image, cv::Point(20, 350), cv::Point(360, 40), cv::Scalar(160), 5);
  return image;
}

std::vector<FeatureKeypoint> RankedKeypoints(FeatureSet features,
                                             const std::size_t max_count) {
  std::vector<FeatureKeypoint> ranked = std::move(features.keypoints);
  std::stable_sort(ranked.begin(), ranked.end(),
                   [](const FeatureKeypoint& lhs,
                      const FeatureKeypoint& rhs) {
                     return std::tie(rhs.response, lhs.y, lhs.x, lhs.scale,
                                     lhs.orientation) <
                            std::tie(lhs.response, rhs.y, rhs.x, rhs.scale,
                                     rhs.orientation);
                   });
  if (ranked.size() > max_count) {
    ranked.resize(max_count);
  }
  return ranked;
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

TEST(SiftExtractorTest, ListsImagesRecursivelyWithStablePaths) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_recursive_image_listing_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "a");
  std::filesystem::create_directories(root / "b");
  ASSERT_TRUE(cv::imwrite((root / "a" / "same.png").string(),
                          MakeSyntheticImage()));
  ASSERT_TRUE(cv::imwrite((root / "b" / "same.png").string(),
                          MakeSyntheticImage()));
  const std::vector<std::filesystem::path> images = ListImageFiles(root);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(images[0], root / "a" / "same.png");
  EXPECT_EQ(images[1], root / "b" / "same.png");
}

TEST(SiftExtractorTest, CvSiftHandlesKernelWiderThanSmallestPyramidLevel) {
  cv::Mat image(768, 1024, CV_8UC1, cv::Scalar(24));
  for (int y = 48; y < image.rows; y += 96) {
    for (int x = 48; x < image.cols; x += 96) {
      cv::circle(image, cv::Point(x, y), 24,
                 cv::Scalar(((x + y) / 96) % 2 ? 220 : 120), -1);
    }
  }
  cv::line(image, cv::Point(0, image.rows - 1),
           cv::Point(image.cols - 1, 0), cv::Scalar(255), 9);

  SiftExtractionOptions options;
  options.backend = SiftBackend::kCvSift;
  options.max_num_features = 512;
  const FeatureSet features = ExtractSiftFeatures(image, options);

  EXPECT_FALSE(features.keypoints.empty());
  EXPECT_EQ(features.descriptors.rows(),
            static_cast<Eigen::Index>(features.keypoints.size()));
  EXPECT_EQ(features.descriptors.cols(), 128);
}

TEST(SiftExtractorTest, CvSiftReflectBorderHandlesTinyImages) {
  for (const cv::Size size :
       {cv::Size(1, 1), cv::Size(2, 1), cv::Size(1, 2), cv::Size(2, 2),
        cv::Size(3, 2)}) {
    cv::Mat image(size, CV_8UC1);
    cv::randu(image, 0, 255);
    SiftExtractionOptions options;
    options.backend = SiftBackend::kCvSift;
    options.max_num_features = 32;
    EXPECT_NO_THROW({
      const FeatureSet features = ExtractSiftFeatures(image, options);
      EXPECT_EQ(features.descriptors.rows(),
                static_cast<Eigen::Index>(features.keypoints.size()));
      if (!features.keypoints.empty()) {
        EXPECT_EQ(features.descriptors.cols(), 128);
      }
    }) << "size=" << size;
  }
}

TEST(SiftExtractorTest, ParsesBackendNames) {
  EXPECT_EQ(SiftBackendFromName("auto"), SiftBackend::kAuto);
  EXPECT_EQ(SiftBackendFromName("vlsift"), SiftBackend::kVlSift);
  EXPECT_EQ(SiftBackendFromName("cvsift"), SiftBackend::kCvSift);
  EXPECT_EQ(SiftBackendFromName("cudasift"), SiftBackend::kCudaSift);
  EXPECT_EQ(SiftBackendFromName("cuda_sift"), SiftBackend::kCudaSift);
  EXPECT_EQ(SiftBackendFromName("cuda_opencv"), SiftBackend::kCudaSift);
  EXPECT_EQ(SiftBackendFromName("cuda_vlfeat"), SiftBackend::kCudaVlSift);
  EXPECT_THROW(SiftBackendFromName("opencv"), std::invalid_argument);
  EXPECT_THROW(SiftBackendFromName("unknown"), std::invalid_argument);
}

TEST(SiftExtractorTest, ExtractsFeaturesWithAllBackends) {
  SiftExtractionOptions options;
  options.max_num_features = 256;
  options.contrast_threshold = 0.0;

  for (const SiftBackend backend :
       {SiftBackend::kVlSift, SiftBackend::kCvSift}) {
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

TEST(SiftExtractorTest, CudaSiftBackendReportsAvailability) {
  SiftExtractionOptions options;
  options.max_num_features = 256;
  options.contrast_threshold = 0.0;

  if (IsCudaSiftAvailable()) {
    EXPECT_EQ(ResolveSiftBackend(SiftBackend::kAuto),
              SiftBackend::kCudaSift);
    for (const SiftBackend backend :
         {SiftBackend::kCudaSift, SiftBackend::kCudaVlSift}) {
      options.backend = backend;
      const FeatureSet features =
          ExtractSiftFeatures(MakeSyntheticImage(), options);

      EXPECT_FALSE(features.keypoints.empty()) << SiftBackendName(backend);
      EXPECT_EQ(features.descriptors.rows(),
                static_cast<Eigen::Index>(features.keypoints.size()));
      EXPECT_EQ(features.descriptors.cols(), 128);
    }
  } else {
    options.backend = SiftBackend::kCudaSift;
    EXPECT_EQ(ResolveSiftBackend(SiftBackend::kAuto), SiftBackend::kCvSift);
    EXPECT_THROW(
        {
          try {
            (void)ExtractSiftFeatures(MakeSyntheticImage(), options);
          } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("COLITE_ENABLE_CUDA_SIFT"),
                      std::string::npos);
            throw;
          }
        },
        std::runtime_error);
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

TEST(SiftExtractorTest, CudaSiftLimitsFeaturesByResponseRanking) {
  if (!IsCudaSiftAvailable()) {
    GTEST_SKIP() << "CudaSift backend is not available in this build.";
  }

  SiftExtractionOptions broad_options;
  broad_options.backend = SiftBackend::kCudaSift;
  broad_options.max_num_features = 512;
  broad_options.cuda_threshold = 1.0F;

  SiftExtractionOptions limited_options = broad_options;
  limited_options.max_num_features = 64;

  const cv::Mat image = MakeDenseSyntheticImage();
  const FeatureSet broad = ExtractSiftFeatures(image, broad_options);
  const FeatureSet limited = ExtractSiftFeatures(image, limited_options);

  ASSERT_GT(broad.keypoints.size(), limited.keypoints.size());
  ASSERT_EQ(limited.keypoints.size(), std::size_t{64});
  for (const FeatureKeypoint& keypoint : limited.keypoints) {
    EXPECT_GT(keypoint.response, 0.0F);
  }

  const std::vector<FeatureKeypoint> expected =
      RankedKeypoints(broad, limited.keypoints.size());
  ASSERT_EQ(limited.keypoints.size(), expected.size());
  const float min_expected_response = expected.back().response;
  float previous_response = std::numeric_limits<float>::infinity();
  for (const FeatureKeypoint& keypoint : limited.keypoints) {
    EXPECT_LE(keypoint.response, previous_response + 1e-5F);
    EXPECT_GE(keypoint.response, min_expected_response - 1e-5F);
    previous_response = keypoint.response;
  }
}

}  // namespace
}  // namespace colite
