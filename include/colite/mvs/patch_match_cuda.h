#pragma once

#include <optional>
#include <string>

#include "colite/mvs/patch_match.h"

namespace colite::mvs {

struct CudaPatchMatchRequest {
  const PatchMatchView* reference_view = nullptr;
  const std::vector<PatchMatchView>* source_views = nullptr;
  const MvsImage* reference_image = nullptr;
  const std::vector<MvsImage>* source_images = nullptr;
  const std::vector<DenseDepthMap>* source_depth_maps = nullptr;
  const PatchMatchMaps* initial_maps = nullptr;
  float depth_min = 0.0F;
  float depth_max = 0.0F;
  PatchMatchStereoOptions options;
};

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereo(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason);

}  // namespace colite::mvs
