#include "colite/mvs/patch_match_cuda.h"

namespace colite::mvs {

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereo(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason) {
  (void)request;
  if (unavailable_reason != nullptr) {
#ifdef COLITE_ENABLE_CUDA_MVS
    *unavailable_reason =
        "CUDA PatchMatch backend is compiled, but CUDA kernels are not "
        "implemented in this build.";
#else
    *unavailable_reason =
        "CUDA PatchMatch backend was not compiled in this build.";
#endif
  }
  return std::nullopt;
}

}  // namespace colite::mvs
