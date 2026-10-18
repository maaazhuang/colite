#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace colite::cuda_matcher {

struct BestMatch {
  int index = -1;
  int best_dot = -1;
  int second_best_dot = -1;
};

struct MutualBestMatches {
  std::vector<BestMatch> forward;
  std::vector<BestMatch> backward;
};

class DescriptorBuffer {
 public:
  virtual ~DescriptorBuffer() = default;
  virtual int Rows() const = 0;
  virtual int Columns() const = 0;
};

bool IsAvailable();
void SetDevice(int device);
std::unique_ptr<DescriptorBuffer> CreateBuffer(const std::uint8_t* descriptors,
                                               int rows,
                                               int columns,
                                               const float* keypoints_xy =
                                                   nullptr);
std::vector<BestMatch> FindBestMatches(const DescriptorBuffer& query,
                                       const DescriptorBuffer& train);
MutualBestMatches FindBestMatchesMutual(const DescriptorBuffer& features1,
                                        const DescriptorBuffer& features2);
std::vector<BestMatch> FindBestMatchesGuided(
    const DescriptorBuffer& query,
    const DescriptorBuffer& train,
    const std::vector<int>& candidate_offsets,
    const std::vector<int>& candidate_indices);
MutualBestMatches FindBestMatchesGuidedFundamental(
    const DescriptorBuffer& features1,
    const DescriptorBuffer& features2,
    const double* fundamental_matrix_col_major,
    double max_sampson_error_px);
}  // namespace colite::cuda_matcher
