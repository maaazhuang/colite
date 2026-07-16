#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace colite::features::cuda {
namespace detail {

static_assert(sizeof(PipelineOrientedKeypoint) == sizeof(CudaSiftKeypoint));
static_assert(offsetof(PipelineOrientedKeypoint, x) == offsetof(CudaSiftKeypoint, x));
static_assert(offsetof(PipelineOrientedKeypoint, y) == offsetof(CudaSiftKeypoint, y));
static_assert(offsetof(PipelineOrientedKeypoint, size) == offsetof(CudaSiftKeypoint, size));
static_assert(offsetof(PipelineOrientedKeypoint, angle) == offsetof(CudaSiftKeypoint, angle));
static_assert(offsetof(PipelineOrientedKeypoint, response) == offsetof(CudaSiftKeypoint, response));
static_assert(offsetof(PipelineOrientedKeypoint, octave) == offsetof(CudaSiftKeypoint, octave));
static_assert(offsetof(PipelineOrientedKeypoint, layer) == offsetof(CudaSiftKeypoint, layer));

cudaError_t cudaMalloc(void** pointer, size_t bytes) noexcept
{
    return ::cudaMallocAsync(pointer, bytes, nullptr);
}

cudaError_t cudaFree(void* pointer) noexcept
{
    return pointer == nullptr ? cudaSuccess : ::cudaFreeAsync(pointer, nullptr);
}

cudaError_t PrepareDeviceMemoryPool(int device) noexcept
{
    int supported = 0;
    cudaError_t error = ::cudaDeviceGetAttribute(
        &supported, cudaDevAttrMemoryPoolsSupported, device);
    if (error != cudaSuccess || supported == 0) {
        return error == cudaSuccess ? cudaErrorNotSupported : error;
    }
    cudaMemPool_t pool = nullptr;
    error = ::cudaDeviceGetDefaultMemPool(&pool, device);
    if (error != cudaSuccess) return error;
    constexpr uint64_t kPoolRetentionLimit = 2ULL * 1024 * 1024 * 1024;
    uint64_t releaseThreshold = kPoolRetentionLimit;
    return ::cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold);
}

namespace {

using Clock = std::chrono::steady_clock;

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

class EventPair {
public:
    EventPair()
    {
        CheckCuda(cudaEventCreate(&start_), "cudaEventCreate start");
        try {
            CheckCuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
        } catch (...) {
            cudaEventDestroy(start_);
            throw;
        }
    }

    ~EventPair()
    {
        if (stop_) cudaEventDestroy(stop_);
        if (start_) cudaEventDestroy(start_);
    }

    void begin() { CheckCuda(cudaEventRecord(start_), "cudaEventRecord start"); }

    float end()
    {
        CheckCuda(cudaEventRecord(stop_), "cudaEventRecord stop");
        CheckCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize stop");
        float milliseconds = 0.0f;
        CheckCuda(cudaEventElapsedTime(&milliseconds, start_, stop_),
                  "cudaEventElapsedTime");
        return milliseconds;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

struct CudaFreeDeleter {
    template <typename T>
    void operator()(T* pointer) const noexcept
    {
        if (pointer) cudaFree(pointer);
    }
};

struct PendingOctaveOutput {
    int count = 0;
    std::unique_ptr<PipelineOrientedKeypoint, CudaFreeDeleter> keypoints;
    std::unique_ptr<float, CudaFreeDeleter> descriptors;
};

int ComputeOctaveBudget(int limit,
                        int /*alreadySelected*/,
                        int /*octaveIndex*/,
                        int /*octaveCount*/,
                        int /*initialWidth*/,
                        int /*initialHeight*/)
{
    if (limit <= 0) return -1;
    // Keep enough candidates from every octave for an exact global top-K.
    // Area-proportional quotas can discard a strong coarse-scale feature in
    // favor of a weaker fine-scale one before the global responses are known.
    return limit;
}

}

template <typename Traits, typename Input>
CudaSiftResult RunPipeline(const CudaSiftConfig& config,
                           int device,
                           const Input* image,
                           int width,
                           int height)
{
    if (image == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("CUDA SIFT input image must be non-null with positive dimensions");
    }
    CheckCuda(cudaSetDevice(device), "cudaSetDevice");
    CheckCuda(PrepareDeviceMemoryPool(device), "PrepareDeviceMemoryPool");
    const auto endToEndStart = Clock::now();
    CudaSiftResult result;
    const size_t estimatedFeatures = config.maxFeatures > 0
        ? static_cast<size_t>(config.maxFeatures)
        : std::clamp(static_cast<size_t>(width) * height / 256,
                     static_cast<size_t>(8192), static_cast<size_t>(131072));
    result.keypoints.reserve(estimatedFeatures);
    result.descriptors.reserve(estimatedFeatures * 128);
    EventPair events;
    const size_t inputBytes = static_cast<size_t>(width) * height * sizeof(Input);
    Input* deviceInput = nullptr;
    DevicePyramidOctave current;
    DevicePyramidOctave previous;
    std::vector<PendingOctaveOutput> pendingOutputs;
    size_t pendingOutputBytes = 0;
    auto cleanup = [&]() {
        ReleaseDevicePyramid(current);
        ReleaseDevicePyramid(previous);
        if (deviceInput) cudaFree(deviceInput);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputBytes),
                  "cudaMalloc pipeline input");
        result.timing.peakDeviceBytes = inputBytes;
        events.begin();
        CheckCuda(cudaMemcpy(deviceInput, image, inputBytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy pipeline H2D");
        result.timing.h2dMs = events.end();

        const int initialWidth = config.firstOctave < 0 ? width * 2 : width;
        const int initialHeight = config.firstOctave < 0 ? height * 2 : height;
        int octaveCount = std::max(static_cast<int>(
            std::log(static_cast<double>(std::min(initialWidth, initialHeight))) /
            std::log(2.0) - 2.0), 1);
        octaveCount -= config.firstOctave;
        if (config.maxOctaves > 0) octaveCount = std::min(octaveCount, config.maxOctaves);
        int selectedFeatures = 0;

        events.begin();
        for (int octaveIndex = 0; octaveIndex < octaveCount; ++octaveIndex) {
            const int octave = config.firstOctave + octaveIndex;
            if (octaveIndex == 0) {
                current = BuildInitialOctaveDevice<Traits>(
                    deviceInput, width, height, config.octaveLayers,
                    config.sigma, config.firstOctave);
            } else {
                current = BuildNextOctaveDevice<Traits>(
                    previous, config.octaveLayers, config.sigma);
                result.timing.peakDeviceBytes = std::max(
                    result.timing.peakDeviceBytes,
                    inputBytes + previous.allocatedBytes + current.allocatedBytes +
                    pendingOutputBytes +
                    static_cast<size_t>(current.width) * current.height *
                        sizeof(float) * 2);
                ReleaseDevicePyramid(previous);
            }

            int refinedCount = 0;
            size_t extremaBytes = 0;
            std::unique_ptr<PipelineRefinedKeypoint, CudaFreeDeleter> refined(
                DetectExtremaDevice<Traits>(
                current.dog, current.width, current.height, current.dogLayers,
                config.octaveLayers, config.contrastThreshold,
                config.peakThreshold, config.edgeThreshold, config.sigma,
                octave, refinedCount, extremaBytes));
            result.timing.peakDeviceBytes = std::max(
                result.timing.peakDeviceBytes,
                inputBytes + current.allocatedBytes + pendingOutputBytes + extremaBytes);

            int orientedCount = 0;
            size_t orientationBytes = 0;
            std::unique_ptr<PipelineOrientedKeypoint, CudaFreeDeleter> oriented;
            if (refinedCount > 0) {
                oriented.reset(AssignOrientationsDevice<Traits>(
                    current.gaussian, current.width, current.height,
                    current.gaussianLayers, refined.get(), refinedCount, octave,
                    orientedCount, orientationBytes));
            }
            result.timing.peakDeviceBytes = std::max(
                result.timing.peakDeviceBytes,
                inputBytes + current.allocatedBytes + pendingOutputBytes +
                static_cast<size_t>(refinedCount) * sizeof(PipelineRefinedKeypoint) +
                orientationBytes);

            const int featureBudget = ComputeOctaveBudget(
                config.maxFeatures, selectedFeatures, octaveIndex, octaveCount,
                initialWidth, initialHeight);
            if (featureBudget == 0) {
                oriented.reset();
                orientedCount = 0;
            } else if (featureBudget > 0 && orientedCount > featureBudget) {
                size_t limitBytes = 0;
                PipelineOrientedKeypoint* limited = RetainBestOrientedDevice(
                    oriented.get(), orientedCount, featureBudget, limitBytes);
                oriented.reset(limited);
                orientedCount = featureBudget;
                result.timing.peakDeviceBytes = std::max(
                    result.timing.peakDeviceBytes,
                    inputBytes + current.allocatedBytes + pendingOutputBytes + limitBytes);
            }
            selectedFeatures += orientedCount;

            size_t descriptorBytes = 0;
            std::unique_ptr<float, CudaFreeDeleter> descriptors;
            if (orientedCount > 0) {
                descriptors.reset(ComputeDescriptorsDevice<Traits>(
                    current.gaussian, current.width, current.height,
                    current.gaussianLayers, oriented.get(), orientedCount, octave,
                    descriptorBytes));
            }
            result.timing.peakDeviceBytes = std::max(
                result.timing.peakDeviceBytes,
                inputBytes + current.allocatedBytes + pendingOutputBytes +
                static_cast<size_t>(refinedCount) * sizeof(PipelineRefinedKeypoint) +
                static_cast<size_t>(orientedCount) * sizeof(PipelineOrientedKeypoint) +
                descriptorBytes);

            if (orientedCount > 0) {
                pendingOutputBytes +=
                    static_cast<size_t>(orientedCount) *
                    (sizeof(PipelineOrientedKeypoint) + 128 * sizeof(float));
                pendingOutputs.push_back(
                    {orientedCount, std::move(oriented), std::move(descriptors)});
            }
            previous = current;
            current = {};
            if (previous.width < 16 || previous.height < 16) break;
        }
        result.timing.computeMs = events.end();
        ReleaseDevicePyramid(previous);

        size_t totalFeatures = 0;
        for (const auto& output : pendingOutputs) {
            totalFeatures += static_cast<size_t>(output.count);
        }
        result.keypoints.resize(totalFeatures);
        result.descriptors.resize(totalFeatures * 128);
        events.begin();
        size_t outputBase = 0;
        for (const auto& output : pendingOutputs) {
            const size_t count = static_cast<size_t>(output.count);
            CheckCuda(::cudaMemcpyAsync(
                          result.keypoints.data() + outputBase, output.keypoints.get(),
                          count * sizeof(CudaSiftKeypoint),
                          cudaMemcpyDeviceToHost, nullptr),
                      "cudaMemcpyAsync pipeline keypoints D2H");
            CheckCuda(::cudaMemcpyAsync(
                          result.descriptors.data() + outputBase * 128,
                          output.descriptors.get(), count * 128 * sizeof(float),
                          cudaMemcpyDeviceToHost, nullptr),
                      "cudaMemcpyAsync pipeline descriptors D2H");
            outputBase += count;
        }
        result.timing.d2hMs = events.end();
        result.timing.transferEndToEndMs = result.timing.h2dMs +
            result.timing.computeMs + result.timing.d2hMs;
        cudaFree(deviceInput);
        deviceInput = nullptr;
        result.timing.endToEndMs = static_cast<float>(
            std::chrono::duration<double, std::milli>(Clock::now() - endToEndStart).count());
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

template <typename Traits>
CudaSiftResult CudaSiftPipeline<Traits>::Run(const CudaSiftConfig& config,
                                              int device,
                                              const float* image,
                                              int width,
                                              int height)
{
    return RunPipeline<Traits>(config, device, image, width, height);
}

template <typename Traits>
CudaSiftResult CudaSiftPipeline<Traits>::Run(const CudaSiftConfig& config,
                                              int device,
                                              const std::uint8_t* image,
                                              int width,
                                              int height)
{
    return RunPipeline<Traits>(config, device, image, width, height);
}
template class CudaSiftPipeline<OpenCvModeTraits>;
template class CudaSiftPipeline<VlFeatModeTraits>;

const char* CompiledModeTag(CudaSiftMode mode)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return CudaSiftPipeline<OpenCvModeTraits>::Tag();
    case CudaSiftMode::VLFeat:
        return CudaSiftPipeline<VlFeatModeTraits>::Tag();
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}

CudaSift::CudaSift(const CudaSiftConfig& config, int device)
    : config_(config), device_(device)
{
    if (device < 0) {
        throw std::invalid_argument("CUDA device index must be non-negative");
    }
    if (config.firstOctave != -1 && config.firstOctave != 0) {
        throw std::invalid_argument("CUDA SIFT currently supports firstOctave -1 or 0");
    }
    if (config.octaveLayers <= 0) {
        throw std::invalid_argument("CUDA SIFT octaveLayers must be positive");
    }
    if (config.maxFeatures < 0) {
        throw std::invalid_argument("CUDA SIFT maxFeatures must be non-negative");
    }
}

CudaSiftResult CudaSift::detectAndCompute(const float* image, int width, int height) const
{
    switch (config_.mode) {
    case CudaSiftMode::OpenCV:
        return detail::CudaSiftPipeline<detail::OpenCvModeTraits>::Run(
            config_, device_, image, width, height);
    case CudaSiftMode::VLFeat:
        return detail::CudaSiftPipeline<detail::VlFeatModeTraits>::Run(
            config_, device_, image, width, height);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

CudaSiftResult CudaSift::detectAndCompute(const std::uint8_t* image,
                                          int width,
                                          int height) const
{
    switch (config_.mode) {
    case CudaSiftMode::OpenCV:
        return detail::CudaSiftPipeline<detail::OpenCvModeTraits>::Run(
            config_, device_, image, width, height);
    case CudaSiftMode::VLFeat:
        return detail::CudaSiftPipeline<detail::VlFeatModeTraits>::Run(
            config_, device_, image, width, height);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
