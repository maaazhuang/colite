#pragma once

#include <string_view>

namespace colite {

enum class OnnxExecutionProvider {
  kAuto,
  kCpu,
  kCuda,
};

constexpr std::string_view OnnxExecutionProviderName(
    const OnnxExecutionProvider provider) {
  switch (provider) {
    case OnnxExecutionProvider::kAuto:
      return "auto";
    case OnnxExecutionProvider::kCpu:
      return "cpu";
    case OnnxExecutionProvider::kCuda:
      return "cuda";
  }
  return "unknown";
}

}  // namespace colite
