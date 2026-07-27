#include "colite/mvs/dense_map.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace colite::mvs {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

TEST(DenseDepthMapTest, StoresDepthValuesAndValidity) {
  DenseDepthMap depth_map(3, 2, 0.5F, 8.0F);

  EXPECT_EQ(depth_map.Width(), 3U);
  EXPECT_EQ(depth_map.Height(), 2U);
  EXPECT_EQ(depth_map.NumPixels(), 6U);
  EXPECT_FLOAT_EQ(depth_map.DepthMin(), 0.5F);
  EXPECT_FLOAT_EQ(depth_map.DepthMax(), 8.0F);
  EXPECT_FALSE(depth_map.IsValid(0, 0));

  depth_map.Set(1, 2, 4.5F);

  EXPECT_TRUE(depth_map.IsValid(1, 2));
  EXPECT_FLOAT_EQ(depth_map.Get(1, 2), 4.5F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 1U);
  EXPECT_THROW(depth_map.Get(2, 0), std::out_of_range);
}

TEST(DenseDepthMapTest, RecomputesRangeFromPositiveFiniteDepths) {
  DenseDepthMap depth_map(2, 2);
  depth_map.Set(0, 0, 3.0F);
  depth_map.Set(0, 1, -1.0F);
  depth_map.Set(1, 0, 5.0F);
  depth_map.Set(1, 1, std::numeric_limits<float>::infinity());

  EXPECT_TRUE(depth_map.RecomputeDepthRange());

  EXPECT_FLOAT_EQ(depth_map.DepthMin(), 3.0F);
  EXPECT_FLOAT_EQ(depth_map.DepthMax(), 5.0F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 2U);
}

TEST(DenseDepthMapTest, MutableDataUpdatesDepthValues) {
  DenseDepthMap depth_map(2, 2);

  std::vector<float>& data = depth_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[3] = 7.5F;

  EXPECT_TRUE(depth_map.IsValid(1, 1));
  EXPECT_FLOAT_EQ(depth_map.Get(1, 1), 7.5F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 1U);
}

TEST(NormalMapTest, StoresNormalizedNormalsAndValidity) {
  NormalMap normal_map(2, 2);

  EXPECT_EQ(normal_map.Width(), 2U);
  EXPECT_EQ(normal_map.Height(), 2U);
  EXPECT_FALSE(normal_map.IsValid(0, 0));

  normal_map.SetNormalized(0, 1, Eigen::Vector3f(0.0F, 0.0F, -2.0F));

  EXPECT_TRUE(normal_map.IsValid(0, 1));
  EXPECT_NEAR(normal_map.Get(0, 1).norm(), 1.0F, 1e-6F);
  EXPECT_FLOAT_EQ(normal_map.Get(0, 1).z(), -1.0F);
  EXPECT_EQ(normal_map.ValidPixelCount(), 1U);
  EXPECT_THROW(normal_map.Set(3, 0, Eigen::Vector3f::UnitX()),
               std::out_of_range);
}

TEST(NormalMapTest, MutableDataUpdatesNormalValues) {
  NormalMap normal_map(2, 2);

  std::vector<Eigen::Vector3f>& data = normal_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[2] = Eigen::Vector3f(0.0F, 0.0F, -1.0F);

  EXPECT_TRUE(normal_map.IsValid(1, 0));
  EXPECT_FLOAT_EQ(normal_map.Get(1, 0).z(), -1.0F);
  EXPECT_EQ(normal_map.ValidPixelCount(), 1U);
}

TEST(DenseCostMapTest, StoresZeroAsValidCostAndInfinityAsInvalid) {
  DenseCostMap cost_map(3, 2);

  EXPECT_EQ(cost_map.Width(), 3U);
  EXPECT_EQ(cost_map.Height(), 2U);
  EXPECT_FALSE(cost_map.IsValid(0, 0));

  cost_map.Set(1, 2, 0.0F);

  EXPECT_TRUE(cost_map.IsValid(1, 2));
  EXPECT_FLOAT_EQ(cost_map.Get(1, 2), 0.0F);
  EXPECT_EQ(cost_map.ValidPixelCount(), 1U);
  EXPECT_THROW(cost_map.Get(2, 0), std::out_of_range);
}

TEST(DenseCostMapTest, MutableDataUpdatesCostValues) {
  DenseCostMap cost_map(2, 2);

  std::vector<float>& data = cost_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[1] = 0.125F;

  EXPECT_TRUE(cost_map.IsValid(0, 1));
  EXPECT_FLOAT_EQ(cost_map.Get(0, 1), 0.125F);
  EXPECT_EQ(cost_map.ValidPixelCount(), 1U);
}

TEST(DenseMapIoTest, RoundTripsDepthAndNormalMapsAsDebugText) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_dense_map_io_test";
  std::filesystem::create_directories(root);

  DenseDepthMap depth_map(2, 1, 0.1F, 10.0F);
  depth_map.Set(0, 0, 2.5F);
  depth_map.Set(0, 1, 4.5F);
  WriteDenseDepthMapText(root / "depth.txt", depth_map);

  const DenseDepthMap loaded_depth =
      ReadDenseDepthMapText(root / "depth.txt");
  EXPECT_EQ(loaded_depth.Width(), 2U);
  EXPECT_EQ(loaded_depth.Height(), 1U);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMin(), 0.1F);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMax(), 10.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 1), 4.5F);

  NormalMap normal_map(1, 2);
  normal_map.SetNormalized(0, 0, Eigen::Vector3f(0.0F, 0.0F, -1.0F));
  normal_map.SetNormalized(1, 0, Eigen::Vector3f(1.0F, 0.0F, 0.0F));
  WriteNormalMapText(root / "normal.txt", normal_map);

  const NormalMap loaded_normal = ReadNormalMapText(root / "normal.txt");
  EXPECT_EQ(loaded_normal.Width(), 1U);
  EXPECT_EQ(loaded_normal.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_normal.Get(0, 0).z(), -1.0F);
  EXPECT_FLOAT_EQ(loaded_normal.Get(1, 0).x(), 1.0F);
}

TEST(DenseMapIoTest, RoundTripsDepthAndNormalMapsAsBinary) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_dense_map_binary_test";
  std::filesystem::create_directories(root);

  DenseDepthMap depth_map(3, 2, 0.5F, 8.0F);
  depth_map.Set(0, 0, 1.5F);
  depth_map.Set(0, 1, 0.0F);
  depth_map.Set(1, 2, 6.25F);
  WriteDenseDepthMapBinary(root / "depth.bin", depth_map);

  const DenseDepthMap loaded_depth =
      ReadDenseDepthMapBinary(root / "depth.bin");
  EXPECT_EQ(loaded_depth.Width(), 3U);
  EXPECT_EQ(loaded_depth.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMin(), 0.5F);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMax(), 8.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 0), 1.5F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 1), 0.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(1, 2), 6.25F);
  EXPECT_EQ(loaded_depth.ValidPixelCount(), 2U);

  NormalMap normal_map(2, 2);
  normal_map.SetNormalized(0, 0, Eigen::Vector3f(0.0F, 0.0F, -1.0F));
  normal_map.SetNormalized(1, 1, Eigen::Vector3f(1.0F, 2.0F, 2.0F));
  WriteNormalMapBinary(root / "normal.bin", normal_map);

  const NormalMap loaded_normal = ReadNormalMapBinary(root / "normal.bin");
  EXPECT_EQ(loaded_normal.Width(), 2U);
  EXPECT_EQ(loaded_normal.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_normal.Get(0, 0).z(), -1.0F);
  EXPECT_NEAR(loaded_normal.Get(1, 1).norm(), 1.0F, 1e-6F);
  EXPECT_FALSE(loaded_normal.IsValid(0, 1));
}

TEST(DenseMapIoTest, WritesCostMapAsDebugText) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_cost_map_io_test";
  std::filesystem::create_directories(root);

  DenseCostMap cost_map(2, 1);
  cost_map.Set(0, 0, 0.25F);
  WriteDenseCostMapText(root / "cost.txt", cost_map);

  const std::string text = ReadTextFile(root / "cost.txt");
  EXPECT_NE(text.find("COLITE_DENSE_COST_MAP_TEXT"), std::string::npos);
  EXPECT_NE(text.find("2 1"), std::string::npos);
  EXPECT_NE(text.find("0.25"), std::string::npos);

  const DenseCostMap loaded = ReadDenseCostMapText(root / "cost.txt");
  EXPECT_EQ(loaded.Width(), 2U);
  EXPECT_EQ(loaded.Height(), 1U);
  EXPECT_FLOAT_EQ(loaded.Get(0, 0), 0.25F);
  EXPECT_FALSE(loaded.IsValid(0, 1));
}

TEST(DenseMapIoTest, RoundTripsCostMapAsBinaryFloatData) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_cost_map_binary_io_test";
  std::filesystem::create_directories(root);

  DenseCostMap cost_map(3, 2);
  cost_map.Set(0, 0, 0.25F);
  cost_map.Set(1, 2, 0.75F);
  WriteDenseCostMapBinary(root / "cost.bin", cost_map);

  const DenseCostMap loaded = ReadDenseCostMapBinary(root / "cost.bin");
  EXPECT_EQ(loaded.Width(), 3U);
  EXPECT_EQ(loaded.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded.Get(0, 0), 0.25F);
  EXPECT_FLOAT_EQ(loaded.Get(1, 2), 0.75F);
  EXPECT_FALSE(loaded.IsValid(0, 1));
}

}  // namespace
}  // namespace colite::mvs
