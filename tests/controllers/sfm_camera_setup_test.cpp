#include <filesystem>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "colite/controllers/sfm_camera_setup.h"

namespace colite::controllers {
namespace {

void WriteImage(const std::filesystem::path& path, int width, int height) {
  std::filesystem::create_directories(path.parent_path());
  const cv::Mat image(height, width, CV_8UC3, cv::Scalar(64, 128, 192));
  ASSERT_TRUE(cv::imwrite(path.string(), image));
}

TEST(SfmCameraSetupTest, IndividualModeSupportsMixedImageDimensions) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_individual_camera_mixed_size_test";
  std::filesystem::remove_all(root);
  const std::filesystem::path image1 = root / "image1.png";
  const std::filesystem::path image2 = root / "image2.png";
  WriteImage(image1, 100, 80);
  WriteImage(image2, 120, 60);

  const SfmCameraSetup setup =
      BuildSfmCameraSetup({image1, image2}, {});

  ASSERT_EQ(setup.cameras.size(), 2U);
  ASSERT_EQ(setup.images.size(), 2U);
  EXPECT_EQ(setup.cameras[0].id, 1U);
  EXPECT_EQ(setup.cameras[0].width, 100U);
  EXPECT_EQ(setup.cameras[0].height, 80U);
  EXPECT_EQ(setup.cameras[1].id, 2U);
  EXPECT_EQ(setup.cameras[1].width, 120U);
  EXPECT_EQ(setup.cameras[1].height, 60U);
  EXPECT_EQ(setup.images[0].camera_id, 1U);
  EXPECT_EQ(setup.images[1].camera_id, 2U);
  EXPECT_DOUBLE_EQ(setup.cameras[0].params[0], 120.0);
  EXPECT_DOUBLE_EQ(setup.cameras[1].params[0], 144.0);
}

TEST(SfmCameraSetupTest, IndividualModeReadsExifForEveryImage) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_individual_camera_exif_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path exif_image =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(exif_image)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }
  const std::filesystem::path no_exif_image = root / "without_exif.png";
  const cv::Mat decoded =
      cv::imread(exif_image.string(), cv::IMREAD_COLOR);
  ASSERT_FALSE(decoded.empty());
  ASSERT_TRUE(cv::imwrite(no_exif_image.string(), decoded));

  const SfmCameraSetup setup =
      BuildSfmCameraSetup({exif_image, no_exif_image}, {});

  ASSERT_EQ(setup.cameras.size(), 2U);
  ASSERT_EQ(setup.focal_estimates.size(), 2U);
  EXPECT_TRUE(setup.cameras[0].has_prior_focal_length);
  EXPECT_FALSE(setup.cameras[1].has_prior_focal_length);
  EXPECT_NEAR(setup.cameras[0].params[0], 2484.86249133349, 1e-6);
  EXPECT_DOUBLE_EQ(setup.cameras[1].params[0],
                   1.2 * static_cast<double>(decoded.cols));

  SfmCameraSetupOptions shared_options;
  shared_options.sharing_mode = CameraSharingMode::kShared;
  const SfmCameraSetup shared_setup =
      BuildSfmCameraSetup({exif_image, no_exif_image}, shared_options);
  ASSERT_EQ(shared_setup.cameras.size(), 1U);
  EXPECT_TRUE(shared_setup.cameras[0].has_prior_focal_length);
  EXPECT_NEAR(shared_setup.cameras[0].params[0], 2484.86249133349, 1e-6);
}

TEST(SfmCameraSetupTest, SharedModeAssignsOneCameraToEqualSizeImages) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_shared_camera_equal_size_test";
  std::filesystem::remove_all(root);
  const std::filesystem::path image1 = root / "image1.png";
  const std::filesystem::path image2 = root / "image2.png";
  WriteImage(image1, 100, 80);
  WriteImage(image2, 100, 80);

  SfmCameraSetupOptions options;
  options.sharing_mode = CameraSharingMode::kShared;
  options.focal_length = 500.0;
  const SfmCameraSetup setup =
      BuildSfmCameraSetup({image1, image2}, options);

  ASSERT_EQ(setup.cameras.size(), 1U);
  ASSERT_EQ(setup.images.size(), 2U);
  EXPECT_EQ(setup.images[0].camera_id, setup.cameras[0].id);
  EXPECT_EQ(setup.images[1].camera_id, setup.cameras[0].id);
  EXPECT_DOUBLE_EQ(setup.cameras[0].params[0], 500.0);
  EXPECT_TRUE(setup.cameras[0].has_prior_focal_length);
}

TEST(SfmCameraSetupTest, SharedModeRejectsMixedImageDimensions) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_shared_camera_mixed_size_test";
  std::filesystem::remove_all(root);
  const std::filesystem::path image1 = root / "image1.png";
  const std::filesystem::path image2 = root / "image2.png";
  WriteImage(image1, 100, 80);
  WriteImage(image2, 120, 60);

  SfmCameraSetupOptions options;
  options.sharing_mode = CameraSharingMode::kShared;
  EXPECT_THROW(BuildSfmCameraSetup({image1, image2}, options),
               std::runtime_error);
}

TEST(SfmCameraSetupTest, ParsesCameraSharingModeNames) {
  EXPECT_EQ(CameraSharingModeFromName("auto"), CameraSharingMode::kAuto);
  EXPECT_EQ(CameraSharingModeFromName("single"), CameraSharingMode::kSingle);
  EXPECT_EQ(CameraSharingModeFromName("per_folder"),
            CameraSharingMode::kPerFolder);
  EXPECT_EQ(CameraSharingModeFromName("per_image"),
            CameraSharingMode::kPerImage);
  EXPECT_EQ(CameraSharingModeFromName("individual"),
            CameraSharingMode::kPerImage);
  EXPECT_EQ(CameraSharingModeFromName("shared"), CameraSharingMode::kSingle);
  EXPECT_EQ(CameraSharingModeName(CameraSharingMode::kIndividual),
            "per_image");
  EXPECT_THROW(CameraSharingModeFromName("folder"), std::invalid_argument);
}

TEST(SfmCameraSetupTest, AutoGroupsCompleteMatchingExifIdentity) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_auto_camera_exif_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path source =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(source)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }
  const std::filesystem::path image1 = root / "image1.JPG";
  const std::filesystem::path image2 = root / "image2.JPG";
  std::filesystem::copy_file(source, image1);
  std::filesystem::copy_file(source, image2);

  SfmCameraSetupOptions options;
  options.image_root = root;
  const SfmCameraSetup setup =
      BuildSfmCameraSetup({image1, image2}, options);

  ASSERT_EQ(setup.cameras.size(), 1U);
  ASSERT_EQ(setup.images.size(), 2U);
  EXPECT_EQ(setup.images[0].camera_id, setup.images[1].camera_id);
  EXPECT_EQ(setup.cameras[0].model_id, CameraModelId::kSimpleRadial);
}

TEST(SfmCameraSetupTest, PerFolderGroupsRelativeSubdirectories) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_per_folder_camera_test";
  std::filesystem::remove_all(root);
  const std::filesystem::path a1 = root / "a" / "same.png";
  const std::filesystem::path a2 = root / "a" / "second.png";
  const std::filesystem::path b1 = root / "b" / "same.png";
  WriteImage(a1, 100, 80);
  WriteImage(a2, 100, 80);
  WriteImage(b1, 120, 60);

  SfmCameraSetupOptions options;
  options.sharing_mode = CameraSharingMode::kPerFolder;
  options.image_root = root;
  const SfmCameraSetup setup = BuildSfmCameraSetup({a1, a2, b1}, options);

  ASSERT_EQ(setup.cameras.size(), 2U);
  EXPECT_EQ(setup.images[0].camera_id, setup.images[1].camera_id);
  EXPECT_NE(setup.images[0].camera_id, setup.images[2].camera_id);
  EXPECT_EQ(setup.images[0].name, "a/same.png");
  EXPECT_EQ(setup.images[2].name, "b/same.png");
}

TEST(SfmCameraSetupTest, AppliesExplicitModelParameters) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_camera_params_test";
  std::filesystem::remove_all(root);
  const std::filesystem::path image = root / "image.png";
  WriteImage(image, 100, 80);

  SfmCameraSetupOptions options;
  options.model_id = CameraModelId::kOpenCV;
  options.camera_params =
      ParseCameraParams("90,91,50,40,0.01,-0.002,0.0001,-0.0002");
  const SfmCameraSetup setup = BuildSfmCameraSetup({image}, options);

  ASSERT_EQ(setup.cameras.size(), 1U);
  EXPECT_EQ(setup.cameras[0].model_id, CameraModelId::kOpenCV);
  EXPECT_EQ(setup.cameras[0].params, options.camera_params);
  EXPECT_TRUE(setup.cameras[0].has_prior_focal_length);
}

TEST(SfmCameraSetupTest, ConvertsGeodeticCoordinatesToLocalEnu) {
  const GeodeticCoordinate origin{46.65, 6.53, 780.0};
  const GeodeticCoordinate east{46.65, 6.531, 780.0};
  const GeodeticCoordinate north{46.651, 6.53, 780.0};
  const Eigen::Vector3d east_enu = GeodeticToLocalEnu(east, origin);
  const Eigen::Vector3d north_enu = GeodeticToLocalEnu(north, origin);

  EXPECT_GT(east_enu.x(), 70.0);
  EXPECT_LT(std::abs(east_enu.y()), 1.0);
  EXPECT_GT(north_enu.y(), 100.0);
  EXPECT_LT(std::abs(north_enu.x()), 1.0);
}

}  // namespace
}  // namespace colite::controllers
