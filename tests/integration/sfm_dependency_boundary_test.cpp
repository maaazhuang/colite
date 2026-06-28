#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

TEST(SfmDependencyBoundaryTest, GeometryEstimationDoesNotUseOpenCvCalib3d) {
  const std::filesystem::path source_dir = COLITE_SOURCE_DIR;
  const std::vector<std::filesystem::path> checked_files = {
      source_dir / "src" / "colite" / "matching" /
          "two_view_geometry_verifier.cpp",
      source_dir / "src" / "colite" / "sfm" / "image_registrar.cpp",
  };

  for (const std::filesystem::path& path : checked_files) {
    const std::string text = ReadTextFile(path);
    EXPECT_EQ(text.find("opencv2/calib3d"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::findFundamentalMat"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::findEssentialMat"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::findHomography"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::recoverPose"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::solvePnPRansac"), std::string::npos)
        << path.string();
  }
}

TEST(SfmDependencyBoundaryTest, ProductionCodeKeepsOnlyOpenCvImageModules) {
  const std::filesystem::path source_dir = COLITE_SOURCE_DIR;
  const std::vector<std::filesystem::path> checked_files = {
      source_dir / "CMakeLists.txt",
      source_dir / "src" / "colite" / "features" / "sift_extractor.cpp",
      source_dir / "include" / "colite" / "features" / "sift_extractor.h",
      source_dir / "src" / "colite" / "exe" / "recon_sfm.cpp",
      source_dir / "include" / "colite" / "exe" / "recon_sfm.h",
  };

  for (const std::filesystem::path& path : checked_files) {
    const std::string text = ReadTextFile(path);
    EXPECT_EQ(text.find("opencv2/features2d"), std::string::npos)
        << path.string();
    EXPECT_EQ(text.find("cv::SIFT::"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("SiftBackend::kOpenCV"), std::string::npos)
        << path.string();
  }

  const std::string cmake = ReadTextFile(source_dir / "CMakeLists.txt");
  EXPECT_EQ(cmake.find("features2d"), std::string::npos);
  EXPECT_EQ(cmake.find("calib3d"), std::string::npos);
}

TEST(SfmDependencyBoundaryTest, SiftBackendsUseColiteOwnedNames) {
  const std::filesystem::path source_dir = COLITE_SOURCE_DIR;
  const std::filesystem::path backend_dir =
      source_dir / "src" / "colite" / "features" / "backends";

  EXPECT_TRUE(std::filesystem::exists(backend_dir / "sift_opencv.hpp"));
  EXPECT_TRUE(std::filesystem::exists(backend_dir / "sift_opencv.cpp"));
  EXPECT_TRUE(std::filesystem::exists(backend_dir / "sift_vlfeat.h"));
  EXPECT_TRUE(std::filesystem::exists(backend_dir / "sift_vlfeat.c"));

  EXPECT_FALSE(std::filesystem::exists(backend_dir / "sift_opencv_v2.hpp"));
  EXPECT_FALSE(std::filesystem::exists(backend_dir / "sift_opencv_v2.cpp"));
  EXPECT_FALSE(std::filesystem::exists(backend_dir / "sift_vlfeat_v2.h"));
  EXPECT_FALSE(std::filesystem::exists(backend_dir / "sift_vlfeat_v2.c"));
  EXPECT_FALSE(std::filesystem::exists(backend_dir / "vl_imconv.h"));
  EXPECT_FALSE(std::filesystem::exists(backend_dir / "vl_imconv.c"));

  const std::vector<std::filesystem::path> checked_files = {
      backend_dir / "sift_opencv.hpp",
      backend_dir / "sift_opencv.cpp",
      backend_dir / "sift_vlfeat.h",
      backend_dir / "sift_vlfeat.c",
      source_dir / "src" / "colite" / "features" / "sift_extractor.cpp",
      source_dir / "src" / "colite" / "CMakeLists.txt",
  };

  for (const std::filesystem::path& path : checked_files) {
    const std::string text = ReadTextFile(path);
    EXPECT_EQ(text.find("V2"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("_v2"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("namespace cv"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("cv::SIFT"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("cv::KeyPoint"), std::string::npos) << path.string();
    EXPECT_EQ(text.find("vl_imconv"), std::string::npos) << path.string();
  }
}

}  // namespace
}  // namespace colite
