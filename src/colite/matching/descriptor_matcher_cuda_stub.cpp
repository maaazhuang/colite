#include "descriptor_matcher_cuda.h"

#include <stdexcept>

namespace colite::cuda_matcher {

bool IsAvailable() { return false; }

std::unique_ptr<DescriptorBuffer> CreateBuffer(const std::uint8_t*,
                                               int,
                                               int,
                                               const float*) {
  throw std::runtime_error(
      "CUDA descriptor matcher is unavailable; rebuild with "
      "COLITE_ENABLE_CUDA_MATCHER=ON.");
}

std::vector<BestMatch> FindBestMatches(const DescriptorBuffer&,
                                       const DescriptorBuffer&) {
  throw std::runtime_error(
      "CUDA descriptor matcher is unavailable; rebuild with "
      "COLITE_ENABLE_CUDA_MATCHER=ON.");
}

MutualBestMatches FindBestMatchesMutual(const DescriptorBuffer&,
                                        const DescriptorBuffer&) {
  throw std::runtime_error(
      "CUDA descriptor matcher is unavailable; rebuild with "
      "COLITE_ENABLE_CUDA_MATCHER=ON.");
}

std::vector<BestMatch> FindBestMatchesGuided(
    const DescriptorBuffer&,
    const DescriptorBuffer&,
    const std::vector<int>&,
    const std::vector<int>&) {
  throw std::runtime_error(
      "CUDA descriptor matcher is unavailable; rebuild with "
      "COLITE_ENABLE_CUDA_MATCHER=ON.");
}

MutualBestMatches FindBestMatchesGuidedFundamental(
    const DescriptorBuffer&,
    const DescriptorBuffer&,
    const double*,
    double) {
  throw std::runtime_error(
      "CUDA descriptor matcher is unavailable; rebuild with "
      "COLITE_ENABLE_CUDA_MATCHER=ON.");
}

}  // namespace colite::cuda_matcher
