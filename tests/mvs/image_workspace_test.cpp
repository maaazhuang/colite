#include "colite/mvs/image_workspace.h"

#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 4;
  camera.height = 2;
  camera.params = InitializeCameraParams(camera.model_id, 10.0,
                                         camera.width, camera.height);
  return camera;
}

Image MakeImage(ImageId image_id, std::string name) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = std::move(name);
  image.registered = true;
  image.cam_from_world = Pose3{};
  return image;
}

Reconstruction MakeWorkspaceReconstruction() {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera());
  reconstruction.AddImage(MakeImage(1, "image1.png"));
  reconstruction.AddImage(MakeImage(2, "image2.png"));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  return reconstruction;
}

void WriteGrayImage(const std::filesystem::path& path, std::uint8_t value) {
  cv::Mat image(2, 4, CV_8UC1, cv::Scalar(value));
  ASSERT_TRUE(cv::imwrite(path.string(), image));
}

TEST(MvsImageWorkspaceTest, LoadsGrayFloatImageFromRegisteredView) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_mvs_workspace_test";
  std::filesystem::create_directories(root);
  WriteGrayImage(root / "image1.png", 128);
  WriteGrayImage(root / "image2.png", 32);

  MvsImageWorkspaceOptions options;
  options.image_path = root;
  options.max_cache_entries = 2;
  MvsImageWorkspace workspace(MakeWorkspaceReconstruction(), options);

  const MvsImage& image = workspace.GetImage(1);

  EXPECT_EQ(image.image_id, 1U);
  EXPECT_EQ(image.name, "image1.png");
  EXPECT_EQ(image.Width(), 4U);
  EXPECT_EQ(image.Height(), 2U);
  EXPECT_EQ(image.gray.type(), CV_32FC1);
  EXPECT_NEAR(image.gray.at<float>(0, 0), 128.0F / 255.0F, 1e-6F);
  EXPECT_EQ(workspace.NumCachedImages(), 1U);
  EXPECT_TRUE(workspace.IsCached(1));
}

TEST(MvsImageWorkspaceTest, EvictsLeastRecentlyUsedImage) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_mvs_workspace_lru_test";
  std::filesystem::create_directories(root);
  WriteGrayImage(root / "image1.png", 128);
  WriteGrayImage(root / "image2.png", 32);

  MvsImageWorkspaceOptions options;
  options.image_path = root;
  options.max_cache_entries = 1;
  MvsImageWorkspace workspace(MakeWorkspaceReconstruction(), options);

  (void)workspace.GetImage(1);
  EXPECT_TRUE(workspace.IsCached(1));
  (void)workspace.GetImage(2);

  EXPECT_FALSE(workspace.IsCached(1));
  EXPECT_TRUE(workspace.IsCached(2));
  EXPECT_EQ(workspace.NumCachedImages(), 1U);
}

TEST(MvsImageWorkspaceTest, DownsizesLoadedImageByMaxDimension) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_mvs_workspace_resize_test";
  std::filesystem::create_directories(root);
  WriteGrayImage(root / "image1.png", 255);
  WriteGrayImage(root / "image2.png", 32);

  MvsImageWorkspaceOptions options;
  options.image_path = root;
  options.max_cache_entries = 2;
  options.max_image_size = 2;
  MvsImageWorkspace workspace(MakeWorkspaceReconstruction(), options);

  const MvsImage& image = workspace.GetImage(1);

  EXPECT_EQ(image.Width(), 2U);
  EXPECT_EQ(image.Height(), 1U);
  EXPECT_DOUBLE_EQ(image.scale, 0.5);
}

}  // namespace
}  // namespace colite::mvs
