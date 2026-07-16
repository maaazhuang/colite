#pragma once

#include <memory>
#include <optional>
#include <string>

#include "colite/mvs/patch_match.h"

namespace colite::mvs {

class CudaPatchMatchWorkspace;

void DestroyCudaPatchMatchWorkspace(CudaPatchMatchWorkspace* workspace);

using CudaPatchMatchWorkspacePtr =
    std::unique_ptr<CudaPatchMatchWorkspace,
                    void (*)(CudaPatchMatchWorkspace*)>;

CudaPatchMatchWorkspacePtr MakeCudaPatchMatchWorkspace();

double WarmupCudaPatchMatch(int cuda_gpu_index,
                            std::string* unavailable_reason);

struct CudaPatchMatchRequest {
  const PatchMatchView* reference_view = nullptr;
  const std::vector<PatchMatchView>* source_views = nullptr;
  const MvsImage* reference_image = nullptr;
  const std::vector<MvsImage>* source_images = nullptr;
  const std::vector<DenseDepthMap>* source_depth_maps = nullptr;
  // Reference-view pixel grid, one probability map per source image.
  // Values outside [0, 1] are clamped by the CUDA backend.
  const std::vector<DenseCostMap>* source_selection_probability_maps = nullptr;
  const PatchMatchMaps* initial_maps = nullptr;
  float depth_min = 0.0F;
  float depth_max = 0.0F;
  PatchMatchStereoOptions options;
  CudaPatchMatchWorkspace* workspace = nullptr;
};

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereo(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason);

}  // namespace colite::mvs
