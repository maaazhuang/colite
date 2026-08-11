#include "colite/io/model_resource.h"

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(ModelResourceTest, ComputesStandardSha256Vector) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "colite_sha256_abc.bin";
  {
    std::ofstream file(path, std::ios::binary);
    file << "abc";
  }
  EXPECT_EQ(ComputeFileSha256(path),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_NO_THROW(ValidateLearnedModelResource(
      path,
      "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"));
  std::filesystem::remove(path);
}

TEST(ModelResourceTest, EnforcesOfficialManifestByFilename) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "aliked-n16rot.onnx";
  {
    std::ofstream file(path, std::ios::binary);
    file << "not an ONNX model";
  }
  EXPECT_THROW(ValidateLearnedModelResource(path), std::runtime_error);
  const LearnedModelResource* resource =
      FindOfficialLearnedModelResource("aliked-n16rot.onnx");
  ASSERT_NE(resource, nullptr);
  EXPECT_EQ(resource->sha256.size(), 64U);
  std::filesystem::remove(path);
}

TEST(ModelResourceTest, AcceptsUnknownExplicitPathWithoutImplicitHash) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "custom-learned-model.onnx";
  {
    std::ofstream file(path, std::ios::binary);
    file << "custom";
  }
  EXPECT_NO_THROW(ValidateLearnedModelResource(path));
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace colite
