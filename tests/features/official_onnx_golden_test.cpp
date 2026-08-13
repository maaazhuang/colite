#include "colite/features/aliked_extractor.h"
#include "colite/matching/learned_matcher.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

namespace colite {
namespace {

std::filesystem::path OfficialModelDirectory() {
  const char* directory = std::getenv("COLITE_TEST_MODEL_DIR");
  return directory == nullptr ? std::filesystem::path()
                              : std::filesystem::path(directory);
}

std::filesystem::path OfficialModelPath(const char* filename) {
  const std::filesystem::path directory = OfficialModelDirectory();
  if (directory.empty()) return {};
  const std::filesystem::path model = directory / filename;
  return std::filesystem::is_regular_file(model) ? model
                                                 : std::filesystem::path();
}

cv::Mat MakeGoldenImage() {
  cv::Mat image(192, 256, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<unsigned char>((3 * x + 5 * y + (x * y) % 37) % 256),
          static_cast<unsigned char>((7 * x + 2 * y + (x ^ y)) % 256),
          static_cast<unsigned char>((11 * x + 13 * y + (x * y) % 53) % 256));
    }
  }
  for (int y = 20; y < image.rows; y += 38) {
    for (int x = 20; x < image.cols; x += 43) {
      cv::circle(image, cv::Point(x, y), 8 + (x + y) % 7,
                 cv::Scalar(250, 20 + y % 200, 240), 2, cv::LINE_AA);
    }
  }
  cv::rectangle(image, cv::Rect(35, 31, 83, 57), cv::Scalar(10, 245, 30),
                3, cv::LINE_AA);
  cv::line(image, cv::Point(5, 180), cv::Point(245, 9),
           cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return image;
}

std::uint64_t QuantizedFeatureDigest(const FeatureSet& features) {
  std::uint64_t digest = 1469598103934665603ULL;
  const auto mix = [&](const std::int64_t value) {
    digest ^= static_cast<std::uint64_t>(value);
    digest *= 1099511628211ULL;
  };
  for (const FeatureKeypoint& keypoint : features.keypoints) {
    mix(std::llround(keypoint.x * 1000.0F));
    mix(std::llround(keypoint.y * 1000.0F));
    mix(std::llround(keypoint.response * 1000000.0F));
  }
  for (Eigen::Index row = 0; row < features.descriptors.rows(); ++row) {
    for (Eigen::Index col = 0; col < features.descriptors.cols(); col += 7) {
      mix(std::llround(features.descriptors(row, col) * 1000000.0F));
    }
  }
  return digest;
}

std::uint64_t MatchDigest(const std::vector<FeatureMatch>& matches) {
  std::uint64_t digest = 1469598103934665603ULL;
  for (const FeatureMatch& match : matches) {
    digest ^= (static_cast<std::uint64_t>(match.idx1) << 32U) | match.idx2;
    digest *= 1099511628211ULL;
  }
  return digest;
}

FeatureSet MakeSyntheticSiftFeatures() {
  constexpr int kCount = 64;
  constexpr int kDimension = 128;
  FeatureSet features;
  features.extractor_type = FeatureExtractorType::kSift;
  features.descriptor_metric = DescriptorMetric::kSiftAngular;
  features.descriptor_normalization = DescriptorNormalization::kRootL1;
  features.descriptors.resize(kCount, kDimension);
  for (int index = 0; index < kCount; ++index) {
    features.keypoints.push_back(
        {16.0F + 28.0F * (index % 8), 14.0F + 23.0F * (index / 8),
         1.6F + 0.1F * (index % 5), static_cast<float>((17 * index) % 360),
         1.0F});
    for (int dimension = 0; dimension < kDimension; ++dimension) {
      const float value = static_cast<float>(
          1 + ((index * 29 + dimension * 17 + index * dimension * 3) % 251));
      features.descriptors(index, dimension) = std::sqrt(value / 16128.0F);
    }
    features.descriptors.row(index).normalize();
  }
  return features;
}

TEST(OfficialOnnxGoldenTest, AlikedN16RotInference) {
  const std::filesystem::path model = OfficialModelPath("aliked-n16rot.onnx");
  if (!IsOnnxRuntimeAvailable() || model.empty()) {
    GTEST_SKIP() << "ONNX support or official N16Rot model is unavailable";
  }
  AlikedExtractionOptions options;
  options.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  options.model_path = model;
  options.max_num_features = 256;
  options.min_score = 0.2F;
  options.execution_provider = OnnxExecutionProvider::kCpu;
  const FeatureSet features = ExtractAlikedFeatures(MakeGoldenImage(), options);
  std::cerr << "N16 count=" << features.keypoints.size()
            << " dim=" << features.descriptors.cols()
            << " digest=" << QuantizedFeatureDigest(features) << '\n';
  EXPECT_EQ(features.keypoints.size(), 256U);
  EXPECT_EQ(features.descriptors.rows(), 256);
  EXPECT_EQ(features.descriptors.cols(), 128);
  EXPECT_EQ(QuantizedFeatureDigest(features), 7397223274269576188ULL);
}

TEST(OfficialOnnxGoldenTest, AlikedN32Inference) {
  const std::filesystem::path model = OfficialModelPath("aliked-n32.onnx");
  if (!IsOnnxRuntimeAvailable() || model.empty()) {
    GTEST_SKIP() << "ONNX support or official N32 model is unavailable";
  }
  AlikedExtractionOptions options;
  options.extractor_type = FeatureExtractorType::kAlikedN32;
  options.model_path = model;
  options.max_num_features = 256;
  options.min_score = 0.2F;
  options.execution_provider = OnnxExecutionProvider::kCpu;
  const FeatureSet features = ExtractAlikedFeatures(MakeGoldenImage(), options);
  std::cerr << "N32 count=" << features.keypoints.size()
            << " dim=" << features.descriptors.cols()
            << " digest=" << QuantizedFeatureDigest(features) << '\n';
  EXPECT_EQ(features.keypoints.size(), 256U);
  EXPECT_EQ(features.descriptors.rows(), 256);
  EXPECT_EQ(features.descriptors.cols(), 128);
  EXPECT_EQ(QuantizedFeatureDigest(features), 13846976305683541616ULL);
}

TEST(OfficialOnnxGoldenTest, AlikedLightGlueInference) {
  const std::filesystem::path extractor_model =
      OfficialModelPath("aliked-n16rot.onnx");
  const std::filesystem::path matcher_model =
      OfficialModelPath("aliked-lightglue.onnx");
  if (!IsOnnxRuntimeAvailable() || extractor_model.empty() ||
      matcher_model.empty()) {
    GTEST_SKIP() << "ONNX support or official ALIKED models are unavailable";
  }
  AlikedExtractionOptions extraction;
  extraction.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  extraction.model_path = extractor_model;
  extraction.max_num_features = 128;
  extraction.min_score = 0.2F;
  extraction.execution_provider = OnnxExecutionProvider::kCpu;
  const FeatureSet features = ExtractAlikedFeatures(MakeGoldenImage(), extraction);
  LightGlueMatchingOptions matching;
  matching.matcher_type = FeatureMatcherType::kAlikedLightGlue;
  matching.model_path = matcher_model;
  matching.min_score = 0.1F;
  matching.execution_provider = OnnxExecutionProvider::kCpu;
  const std::vector<FeatureMatch> matches = MatchFeatureSetsLightGlue(
      features, 256, 192, features, 256, 192, matching);
  std::cerr << "ALIKED LightGlue count=" << matches.size()
            << " digest=" << MatchDigest(matches) << '\n';
  EXPECT_EQ(matches.size(), 128U);
  EXPECT_EQ(MatchDigest(matches), 3359675601613547907ULL);
  for (std::size_t index = 0; index < matches.size(); ++index) {
    EXPECT_EQ(matches[index].idx1, index);
    EXPECT_EQ(matches[index].idx2, index);
  }
}

TEST(OfficialOnnxGoldenTest, SiftLightGlueInference) {
  const std::filesystem::path model = OfficialModelPath("sift-lightglue.onnx");
  if (!IsOnnxRuntimeAvailable() || model.empty()) {
    GTEST_SKIP() << "ONNX support or official SIFT-LightGlue model is unavailable";
  }
  const FeatureSet features = MakeSyntheticSiftFeatures();
  LightGlueMatchingOptions options;
  options.matcher_type = FeatureMatcherType::kSiftLightGlue;
  options.model_path = model;
  options.min_score = 0.1F;
  options.execution_provider = OnnxExecutionProvider::kCpu;
  const std::vector<FeatureMatch> matches = MatchFeatureSetsLightGlue(
      features, 256, 192, features, 256, 192, options);
  std::cerr << "SIFT LightGlue count=" << matches.size()
            << " digest=" << MatchDigest(matches) << '\n';
  EXPECT_EQ(matches.size(), 64U);
  EXPECT_EQ(MatchDigest(matches), 15057192408810339971ULL);
  for (std::size_t index = 0; index < matches.size(); ++index) {
    EXPECT_EQ(matches[index].idx1, index);
    EXPECT_EQ(matches[index].idx2, index);
  }
}

TEST(OfficialOnnxCudaTest, RunsAlikedAndLightGlueOnCudaProvider) {
  const std::filesystem::path extractor_model =
      OfficialModelPath("aliked-n16rot.onnx");
  const std::filesystem::path matcher_model =
      OfficialModelPath("aliked-lightglue.onnx");
  if (!IsOnnxCudaExecutionProviderAvailable() || extractor_model.empty() ||
      matcher_model.empty()) {
    GTEST_SKIP() << "ONNX CUDA provider or official ALIKED models unavailable";
  }
  AlikedExtractionOptions extraction;
  extraction.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  extraction.model_path = extractor_model;
  extraction.max_num_features = 128;
  extraction.min_score = 0.2F;
  extraction.execution_provider = OnnxExecutionProvider::kCuda;
  extraction.gpu_index = 0;
  const FeatureSet features =
      ExtractAlikedFeatures(MakeGoldenImage(), extraction);
  ASSERT_EQ(features.keypoints.size(), 128U);
  ASSERT_EQ(features.descriptors.cols(), 128);

  LightGlueMatchingOptions matching;
  matching.matcher_type = FeatureMatcherType::kAlikedLightGlue;
  matching.model_path = matcher_model;
  matching.min_score = 0.1F;
  matching.execution_provider = OnnxExecutionProvider::kCuda;
  matching.gpu_index = 0;
  const std::vector<FeatureMatch> matches = MatchFeatureSetsLightGlue(
      features, 256, 192, features, 256, 192, matching);
  ASSERT_EQ(matches.size(), 128U);
  for (std::size_t index = 0; index < matches.size(); ++index) {
    EXPECT_EQ(matches[index].idx1, index);
    EXPECT_EQ(matches[index].idx2, index);
  }
}

}  // namespace
}  // namespace colite
