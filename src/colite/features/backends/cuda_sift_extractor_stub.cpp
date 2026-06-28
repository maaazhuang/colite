#include "backends/cuda_sift_extractor.h"

#include <stdexcept>

namespace colite {

bool IsCudaSiftAvailable() { return false; }

FeatureSet ExtractCudaSiftFeatures(const cv::Mat&,
                                   const SiftExtractionOptions&) {
  throw std::runtime_error(
      "CudaSift backend is not available in this build. Reconfigure with "
      "-DCOLITE_ENABLE_CUDA_SIFT=ON and a CUDA compiler/toolkit.");
}

}  // namespace colite
