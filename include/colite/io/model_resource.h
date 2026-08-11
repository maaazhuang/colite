#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace colite {

struct LearnedModelResource {
  std::string name;
  std::string filename;
  std::string url;
  std::string sha256;
};

const std::vector<LearnedModelResource>& OfficialLearnedModelResources();
const LearnedModelResource* FindOfficialLearnedModelResource(
    std::string_view filename);

std::string ComputeFileSha256(const std::filesystem::path& path);

// An explicit hash takes precedence. If it is empty and the basename matches
// an official resource, the built-in manifest hash is enforced. Unknown
// explicit model paths are accepted after existence/readability checks.
void ValidateLearnedModelResource(
    const std::filesystem::path& path,
    std::string_view expected_sha256 = {});

}  // namespace colite
