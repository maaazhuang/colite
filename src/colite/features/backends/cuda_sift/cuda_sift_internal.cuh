#pragma once

#include "cuda_sift/cuda_sift.hpp"

#include <cuda_runtime_api.h>

#include <vector>

namespace colite::features::cuda::detail {

// Pipeline allocations are stream ordered so repeated warm runs can reuse the
// CUDA memory pool without forcing a device-wide synchronization on every free.
cudaError_t cudaMalloc(void** pointer, size_t bytes) noexcept;
cudaError_t cudaFree(void* pointer) noexcept;
cudaError_t PrepareDeviceMemoryPool(int device) noexcept;

struct OpenCvModeTraits {
    static constexpr CudaSiftMode mode = CudaSiftMode::OpenCV;
    static constexpr const char* tag = "opencv";
};

struct VlFeatModeTraits {
    static constexpr CudaSiftMode mode = CudaSiftMode::VLFeat;
    static constexpr const char* tag = "vlfeat";
};

struct DevicePyramidOctave {
    float* gaussian = nullptr;
    float* dog = nullptr;
    int width = 0;
    int height = 0;
    int gaussianLayers = 0;
    int dogLayers = 0;
    size_t allocatedBytes = 0;
};

struct PipelineRefinedKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float refinedLayer = 0.0f;
    float size = 0.0f;
    float response = 0.0f;
    int layer = 0;
};

struct PipelineOrientedKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;
    float angle = 0.0f;
    float response = 0.0f;
    union {
        int sourceIndex = 0;
        int octave;
    };
    int layer = 0;
};

template <typename Traits>
class CudaSiftPipeline {
public:
    static const char* Tag() noexcept { return Traits::tag; }

    static CudaSiftResult Run(const CudaSiftConfig& config,
                              int device,
                              const float* image,
                              int width,
                              int height);
    static CudaSiftResult Run(const CudaSiftConfig& config,
                              int device,
                              const std::uint8_t* image,
                              int width,
                              int height);
};

extern template class CudaSiftPipeline<OpenCvModeTraits>;
extern template class CudaSiftPipeline<VlFeatModeTraits>;

template <typename Traits>
void LaunchUpsample2x(const float* deviceSource,
                      float* deviceTemporary,
                      float* deviceOutput,
                      int width,
                      int height,
                      cudaStream_t stream = nullptr);

template <typename Traits>
void LaunchUpsample2x(const std::uint8_t* deviceSource,
                      float* deviceTemporary,
                      float* deviceOutput,
                      int width,
                      int height,
                      cudaStream_t stream = nullptr);

void LaunchConvertToFloat(const std::uint8_t* deviceSource,
                          float* deviceOutput,
                          size_t count,
                          cudaStream_t stream = nullptr);

extern template void LaunchUpsample2x<OpenCvModeTraits>(
    const float*, float*, float*, int, int, cudaStream_t);
extern template void LaunchUpsample2x<VlFeatModeTraits>(
    const float*, float*, float*, int, int, cudaStream_t);
extern template void LaunchUpsample2x<OpenCvModeTraits>(
    const std::uint8_t*, float*, float*, int, int, cudaStream_t);
extern template void LaunchUpsample2x<VlFeatModeTraits>(
    const std::uint8_t*, float*, float*, int, int, cudaStream_t);

template <typename Traits>
DevicePyramidOctave BuildInitialOctaveDevice(const float* deviceInput,
                                              int width,
                                              int height,
                                              int octaveLayers,
                                              float sigma,
                                              int firstOctave);

template <typename Traits>
DevicePyramidOctave BuildNextOctaveDevice(const DevicePyramidOctave& previous,
                                           int octaveLayers,
                                           float sigma);

extern template DevicePyramidOctave BuildInitialOctaveDevice<OpenCvModeTraits>(
    const float*, int, int, int, float, int);
extern template DevicePyramidOctave BuildInitialOctaveDevice<VlFeatModeTraits>(
    const float*, int, int, int, float, int);

template <typename Traits>
DevicePyramidOctave BuildInitialOctaveDevice(const std::uint8_t* deviceInput,
                                              int width,
                                              int height,
                                              int octaveLayers,
                                              float sigma,
                                              int firstOctave);

extern template DevicePyramidOctave BuildInitialOctaveDevice<OpenCvModeTraits>(
    const std::uint8_t*, int, int, int, float, int);
extern template DevicePyramidOctave BuildInitialOctaveDevice<VlFeatModeTraits>(
    const std::uint8_t*, int, int, int, float, int);

extern template DevicePyramidOctave BuildNextOctaveDevice<OpenCvModeTraits>(
    const DevicePyramidOctave&, int, float);
extern template DevicePyramidOctave BuildNextOctaveDevice<VlFeatModeTraits>(
    const DevicePyramidOctave&, int, float);

void ReleaseDevicePyramid(DevicePyramidOctave& octave) noexcept;

template <typename Traits>
PipelineRefinedKeypoint* DetectExtremaDevice(const float* deviceDog,
                                             int width,
                                             int height,
                                             int dogLayers,
                                             int octaveLayers,
                                             float contrastThreshold,
                                             float peakThreshold,
                                             float edgeThreshold,
                                             float sigma,
                                             int octave,
                                             int& outputCount,
                                             size_t& temporaryBytes);

extern template PipelineRefinedKeypoint* DetectExtremaDevice<OpenCvModeTraits>(
    const float*, int, int, int, int, float, float, float, float, int, int&, size_t&);
extern template PipelineRefinedKeypoint* DetectExtremaDevice<VlFeatModeTraits>(
    const float*, int, int, int, int, float, float, float, float, int, int&, size_t&);

template <typename Traits>
PipelineOrientedKeypoint* AssignOrientationsDevice(
    const float* deviceGaussian,
    int width,
    int height,
    int gaussianLayers,
    const PipelineRefinedKeypoint* deviceKeypoints,
    int keypointCount,
    int octave,
    int& outputCount,
    size_t& temporaryBytes);

extern template PipelineOrientedKeypoint* AssignOrientationsDevice<OpenCvModeTraits>(
    const float*, int, int, int, const PipelineRefinedKeypoint*, int, int, int&, size_t&);
extern template PipelineOrientedKeypoint* AssignOrientationsDevice<VlFeatModeTraits>(
    const float*, int, int, int, const PipelineRefinedKeypoint*, int, int, int&, size_t&);

PipelineOrientedKeypoint* RetainBestOrientedDevice(
    const PipelineOrientedKeypoint* input,
    int inputCount,
    int maximumCount,
    size_t& peakTemporaryBytes);

template <typename Traits>
float* ComputeDescriptorsDevice(const float* deviceGaussian,
                                int width,
                                int height,
                                int gaussianLayers,
                                const PipelineOrientedKeypoint* deviceKeypoints,
                                int keypointCount,
                                int octave,
                                size_t& allocatedBytes);

extern template float* ComputeDescriptorsDevice<OpenCvModeTraits>(
    const float*, int, int, int, const PipelineOrientedKeypoint*, int, int, size_t&);
extern template float* ComputeDescriptorsDevice<VlFeatModeTraits>(
    const float*, int, int, int, const PipelineOrientedKeypoint*, int, int, size_t&);

std::vector<float> BuildInitialImageForTest(CudaSiftMode mode,
                                            const float* image,
                                            int width,
                                            int height);
std::vector<float> BuildInitialImageForTest(CudaSiftMode mode,
                                            const std::uint8_t* image,
                                            int width,
                                            int height);

struct PyramidSnapshot {
    int width = 0;
    int height = 0;
    int gaussianLayers = 0;
    int dogLayers = 0;
    std::vector<float> gaussian;
    std::vector<float> dog;
};

struct ExtremaCandidateSnapshot {
    int x = 0;
    int y = 0;
    int layer = 0;
};

struct RefinedKeypointSnapshot {
    float x = 0.0f;
    float y = 0.0f;
    float refinedLayer = 0.0f;
    float size = 0.0f;
    float response = 0.0f;
    int layer = 0;
};

struct ExtremaTestResult {
    std::vector<ExtremaCandidateSnapshot> initialCandidates;
    std::vector<RefinedKeypointSnapshot> keypoints;
};

struct OrientedKeypointSnapshot {
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;
    float angle = 0.0f;
    float response = 0.0f;
    int layer = 0;
    int sourceIndex = 0;
};

PyramidSnapshot BuildFirstOctaveForTest(CudaSiftMode mode,
                                        const float* image,
                                        int width,
                                        int height,
                                        int octaveLayers,
                                        float sigma);
PyramidSnapshot BuildFirstOctaveForTest(CudaSiftMode mode,
                                        const std::uint8_t* image,
                                        int width,
                                        int height,
                                        int octaveLayers,
                                        float sigma);
ExtremaTestResult DetectExtremaForTest(CudaSiftMode mode,
                                        const float* dog,
                                        int width,
                                        int height,
                                        int dogLayers,
                                        int octaveLayers,
                                        float contrastThreshold,
                                        float peakThreshold,
                                        float edgeThreshold,
                                        float sigma,
                                        int octave);

std::vector<OrientedKeypointSnapshot> AssignOrientationsForTest(
    CudaSiftMode mode,
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<RefinedKeypointSnapshot>& keypoints,
    int octave);

std::vector<float> ComputeDescriptorsForTest(
    CudaSiftMode mode,
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<OrientedKeypointSnapshot>& keypoints,
    int octave);

}
