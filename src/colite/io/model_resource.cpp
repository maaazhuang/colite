#include "colite/io/model_resource.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace colite {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t RotateRight(const std::uint32_t value,
                          const std::uint32_t bits) {
  return (value >> bits) | (value << (32U - bits));
}

class Sha256 {
 public:
  void Update(const std::uint8_t* data, const std::size_t size) {
    for (std::size_t idx = 0; idx < size; ++idx) {
      buffer_[buffer_size_++] = data[idx];
      if (buffer_size_ == buffer_.size()) {
        Transform(buffer_.data());
        bit_count_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> Finalize() {
    bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) {
        buffer_[buffer_size_++] = 0;
      }
      Transform(buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) {
      buffer_[buffer_size_++] = 0;
    }
    for (int byte = 7; byte >= 0; --byte) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>(
          (bit_count_ >> (static_cast<unsigned int>(byte) * 8U)) & 0xffU);
    }
    Transform(buffer_.data());
    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[word * 4 + byte] = static_cast<std::uint8_t>(
            state_[word] >> (24U - static_cast<unsigned int>(byte) * 8U));
      }
    }
    return digest;
  }

 private:
  void Transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t idx = 0; idx < 16; ++idx) {
      schedule[idx] = (static_cast<std::uint32_t>(block[idx * 4]) << 24U) |
                      (static_cast<std::uint32_t>(block[idx * 4 + 1]) << 16U) |
                      (static_cast<std::uint32_t>(block[idx * 4 + 2]) << 8U) |
                      static_cast<std::uint32_t>(block[idx * 4 + 3]);
    }
    for (std::size_t idx = 16; idx < schedule.size(); ++idx) {
      const std::uint32_t s0 = RotateRight(schedule[idx - 15], 7) ^
                               RotateRight(schedule[idx - 15], 18) ^
                               (schedule[idx - 15] >> 3U);
      const std::uint32_t s1 = RotateRight(schedule[idx - 2], 17) ^
                               RotateRight(schedule[idx - 2], 19) ^
                               (schedule[idx - 2] >> 10U);
      schedule[idx] = schedule[idx - 16] + s0 + schedule[idx - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t idx = 0; idx < schedule.size(); ++idx) {
      const std::uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
                                 RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 =
          h + sum1 + choice + kRoundConstants[idx] + schedule[idx];
      const std::uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
                                 RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

std::string Lowercase(std::string value) {
  for (char& character : value) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  return value;
}

}  // namespace

const std::vector<LearnedModelResource>& OfficialLearnedModelResources() {
  static const std::vector<LearnedModelResource> resources = {
      {"ALIKED N16Rot", "aliked-n16rot.onnx",
       "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
       "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547"},
      {"ALIKED N32", "aliked-n32.onnx",
       "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n32.onnx",
       "a077728a02d2de1a775c66df6de8cfeb7c6b51ca57572c64c680131c988c8b3c"},
      {"ALIKED LightGlue", "aliked-lightglue.onnx",
       "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
       "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d"},
      {"SIFT LightGlue", "sift-lightglue.onnx",
       "https://github.com/colmap/colmap/releases/download/3.13.0/sift-lightglue.onnx",
       "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e"}};
  return resources;
}

const LearnedModelResource* FindOfficialLearnedModelResource(
    const std::string_view filename) {
  for (const LearnedModelResource& resource : OfficialLearnedModelResources()) {
    if (resource.filename == filename) {
      return &resource;
    }
  }
  return nullptr;
}

std::string ComputeFileSha256(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::invalid_argument("Cannot open model file: " + path.string());
  }
  Sha256 hash;
  std::array<std::uint8_t, 64 * 1024> buffer{};
  while (file) {
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = file.gcount();
    if (count > 0) {
      hash.Update(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  if (!file.eof()) {
    throw std::runtime_error("Failed while reading model file: " +
                             path.string());
  }
  const auto digest = hash.Finalize();
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    stream << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return stream.str();
}

void ValidateLearnedModelResource(
    const std::filesystem::path& path,
    const std::string_view expected_sha256) {
  if (path.empty() || !std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument("Model path is not a regular file: " +
                                path.string());
  }
  std::string expected(expected_sha256);
  if (expected.empty()) {
    const LearnedModelResource* resource =
        FindOfficialLearnedModelResource(path.filename().string());
    if (resource != nullptr) {
      expected = resource->sha256;
    }
  }
  if (expected.empty()) {
    return;
  }
  expected = Lowercase(expected);
  if (expected.size() != 64 ||
      expected.find_first_not_of("0123456789abcdef") != std::string::npos) {
    throw std::invalid_argument("Expected SHA-256 must contain 64 hex digits");
  }
  const std::string actual = ComputeFileSha256(path);
  if (actual != expected) {
    throw std::runtime_error("Model SHA-256 mismatch for " + path.string() +
                             ": expected " + expected + ", got " + actual);
  }
}

}  // namespace colite
