#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cfloat>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace colite::features::cuda::detail {
namespace {

constexpr int kOrientationBins = 36;
constexpr int kWarpSize = 32;
constexpr int kMaxOrientations = 4;
constexpr float kPi = 3.14159265358979323846f;

struct DeviceInputKeypoint {
    float x;
    float y;
    float size;
    float response;
    int layer;
};

struct DeviceOrientationSet {
    float angles[kMaxOrientations];
};

using DeviceOrientedKeypoint = PipelineOrientedKeypoint;

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ int WrapBin(int bin)
{
    if (bin < 0) return bin + kOrientationBins;
    if (bin >= kOrientationBins) return bin - kOrientationBins;
    return bin;
}

__device__ __forceinline__ float FastAtan2OpenCv(float y, float x)
{
    constexpr float p1 = 0.9997878412794807f * 57.29577951308232f;
    constexpr float p3 = -0.3258083974640975f * 57.29577951308232f;
    constexpr float p5 = 0.1555786518463281f * 57.29577951308232f;
    constexpr float p7 = -0.04432655554792128f * 57.29577951308232f;
    const float ax = fabsf(x);
    const float ay = fabsf(y);
    float c;
    float angle;
    if (ax >= ay) {
        c = ay / (ax + FLT_EPSILON);
        const float c2 = c * c;
        angle = (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    } else {
        c = ax / (ay + FLT_EPSILON);
        const float c2 = c * c;
        angle = 90.0f - (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c;
    }
    if (x < 0.0f) angle = 180.0f - angle;
    if (y < 0.0f) angle = 360.0f - angle;
    if (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

__device__ __forceinline__ float FastAtan2VlFeat(float y, float x)
{
    const float absY = fabsf(y) + FLT_EPSILON;
    float ratio;
    float angle;
    if (x >= 0.0f) {
        ratio = (x - absY) / (x + absY);
        angle = kPi * 0.25f;
    } else {
        ratio = (x + absY) / (absY - x);
        angle = 3.0f * kPi * 0.25f;
    }
    angle += (0.1821f * ratio * ratio - 0.9675f) * ratio;
    angle = y < 0.0f ? -angle : angle;
    angle += 2.0f * kPi;
    if (angle > 2.0f * kPi) angle -= 2.0f * kPi;
    return angle;
}

__device__ __forceinline__ float FastSqrtVlFeat(float value)
{
    if (value < 1e-8f) return 0.0f;
    const float half = 0.5f * value;
    float estimate = __int_as_float(0x5f3759df - (__float_as_int(value) >> 1));
    estimate *= 1.5f - half * estimate * estimate;
    estimate *= 1.5f - half * estimate * estimate;
    return value * estimate;
}

template <typename Traits>
__device__ __forceinline__ void AccumulateSample(
    const float* image,
    int width,
    int height,
    int x,
    int y,
    int dx,
    int dy,
    float centerX,
    float centerY,
    int radius,
    float windowSigma,
    float* partialHistogram)
{
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;
        const float gx = image[static_cast<size_t>(y) * width + x + 1] -
                         image[static_cast<size_t>(y) * width + x - 1];
        const float gy = image[static_cast<size_t>(y - 1) * width + x] -
                         image[static_cast<size_t>(y + 1) * width + x];
        const float magnitude = sqrtf(gx * gx + gy * gy);
        int bin = static_cast<int>(lroundf(FastAtan2OpenCv(gy, gx) *
                                           kOrientationBins / 360.0f));
        bin = WrapBin(bin);
        const float exponentScale = -1.0f / (2.0f * windowSigma * windowSigma);
        partialHistogram[bin] += expf((dx * dx + dy * dy) * exponentScale) * magnitude;
    } else {
        if (x < 0 || x >= width || y < 0 || y >= height) return;
        const float sampleX = static_cast<float>(x) - centerX;
        const float sampleY = static_cast<float>(y) - centerY;
        const float squaredRadius = sampleX * sampleX + sampleY * sampleY;
        if (squaredRadius >= radius * radius + 0.6f) return;
        const int left = max(x - 1, 0);
        const int right = min(x + 1, width - 1);
        const int up = max(y - 1, 0);
        const int down = min(y + 1, height - 1);
        const float xScale = x == 0 || x == width - 1 ? 1.0f : 0.5f;
        const float yScale = y == 0 || y == height - 1 ? 1.0f : 0.5f;
        const float gx = xScale * (image[static_cast<size_t>(y) * width + right] -
                                   image[static_cast<size_t>(y) * width + left]);
        const float gy = yScale * (image[static_cast<size_t>(down) * width + x] -
                                   image[static_cast<size_t>(up) * width + x]);
        const float magnitude = FastSqrtVlFeat(gx * gx + gy * gy);
        const float floatingBin = kOrientationBins * FastAtan2VlFeat(gy, gx) /
                                  (2.0f * kPi);
        const int bin = static_cast<int>(floorf(floatingBin - 0.5f));
        const float fraction = floatingBin - bin - 0.5f;
        const float weight = expf(-squaredRadius /
                                  (2.0f * windowSigma * windowSigma));
        partialHistogram[WrapBin(bin)] += (1.0f - fraction) * magnitude * weight;
        partialHistogram[WrapBin(bin + 1)] += fraction * magnitude * weight;
    }
}

template <typename Traits>
__global__ void BuildOrientationSetsKernel(const float* gaussian,
                                           const DeviceInputKeypoint* keypoints,
                                           DeviceOrientationSet* orientationSets,
                                           int* orientationCounts,
                                           int keypointCount,
                                           int width,
                                           int height,
                                           int gaussianLayers,
                                           int octave)
{
    __shared__ float partial[kWarpSize][kOrientationBins];
    __shared__ float histogram[kOrientationBins];
    const int keypointIndex = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (keypointIndex >= keypointCount) return;
    for (int bin = 0; bin < kOrientationBins; ++bin) partial[lane][bin] = 0.0f;
    if (lane < kOrientationBins) histogram[lane] = 0.0f;
    if (lane < kMaxOrientations) orientationSets[keypointIndex].angles[lane] = 0.0f;
    if (lane == 0) orientationCounts[keypointIndex] = 0;
    __syncwarp();

    const DeviceInputKeypoint keypoint = keypoints[keypointIndex];
    const float octaveScale = ldexpf(1.0f, octave);
    const float localX = keypoint.x / octaveScale;
    const float localY = keypoint.y / octaveScale;
    const int centerX = std::is_same_v<Traits, OpenCvModeTraits>
        ? static_cast<int>(lroundf(localX))
        : static_cast<int>(localX + 0.5f);
    const int centerY = std::is_same_v<Traits, OpenCvModeTraits>
        ? static_cast<int>(lroundf(localY))
        : static_cast<int>(localY + 0.5f);
    if (keypoint.layer < 0 || keypoint.layer >= gaussianLayers ||
        centerX < 0 || centerX >= width || centerY < 0 || centerY >= height) {
        return;
    }

    const float localSigma = std::is_same_v<Traits, OpenCvModeTraits>
        ? keypoint.size * 0.5f / octaveScale
        : keypoint.size / octaveScale;
    const float windowSigma = 1.5f * localSigma;
    const int radius = std::is_same_v<Traits, OpenCvModeTraits>
        ? static_cast<int>(lroundf(4.5f * localSigma))
        : max(static_cast<int>(floorf(3.0f * windowSigma)), 1);
    const int side = 2 * radius + 1;
    const int sampleCount = side * side;
    const float* image = gaussian + static_cast<size_t>(keypoint.layer) * width * height;
    for (int sample = lane; sample < sampleCount; sample += kWarpSize) {
        const int dy = sample / side - radius;
        const int dx = sample - (dy + radius) * side - radius;
        AccumulateSample<Traits>(image, width, height,
                                 centerX + dx, centerY + dy, dx, dy,
                                 localX, localY, radius, windowSigma,
                                 partial[lane]);
    }
    __syncwarp();

    for (int bin = lane; bin < kOrientationBins; bin += kWarpSize) {
        float sum = 0.0f;
#pragma unroll
        for (int owner = 0; owner < kWarpSize; ++owner) sum += partial[owner][bin];
        histogram[bin] = sum;
    }
    __syncwarp();

    if (lane == 0) {
        float smoothed[kOrientationBins];
        if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
            for (int bin = 0; bin < kOrientationBins; ++bin) {
                smoothed[bin] = (histogram[WrapBin(bin - 2)] +
                                 histogram[WrapBin(bin + 2)]) / 16.0f +
                                (histogram[WrapBin(bin - 1)] +
                                 histogram[WrapBin(bin + 1)]) * 0.25f +
                                histogram[bin] * 0.375f;
            }
        } else {
            for (int bin = 0; bin < kOrientationBins; ++bin) smoothed[bin] = histogram[bin];
            for (int iteration = 0; iteration < 6; ++iteration) {
                float previous = smoothed[kOrientationBins - 1];
                const float first = smoothed[0];
                for (int bin = 0; bin < kOrientationBins - 1; ++bin) {
                    const float value = (previous + smoothed[bin] + smoothed[bin + 1]) / 3.0f;
                    previous = smoothed[bin];
                    smoothed[bin] = value;
                }
                smoothed[kOrientationBins - 1] =
                    (previous + smoothed[kOrientationBins - 1] + first) / 3.0f;
            }
        }

        float maximum = 0.0f;
        for (int bin = 0; bin < kOrientationBins; ++bin) maximum = fmaxf(maximum, smoothed[bin]);
        int count = 0;
        for (int bin = 0; bin < kOrientationBins && count < kMaxOrientations; ++bin) {
            const float left = smoothed[WrapBin(bin - 1)];
            const float center = smoothed[bin];
            const float right = smoothed[WrapBin(bin + 1)];
            const bool peak = std::is_same_v<Traits, OpenCvModeTraits>
                ? center > left && center > right && center >= 0.8f * maximum
                : center > left && center > right && center > 0.8f * maximum;
            if (!peak) continue;
            float offset;
            if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
                offset = 0.5f * (left - right) / (left - 2.0f * center + right);
                float refinedBin = bin + offset;
                if (refinedBin < 0.0f) refinedBin += kOrientationBins;
                if (refinedBin >= kOrientationBins) refinedBin -= kOrientationBins;
                float angle = 360.0f - 360.0f * refinedBin / kOrientationBins;
                if (fabsf(angle - 360.0f) < FLT_EPSILON) angle = 0.0f;
                orientationSets[keypointIndex].angles[count++] = angle;
            } else {
                offset = -0.5f * (right - left) /
                         (right + left - 2.0f * center);
                orientationSets[keypointIndex].angles[count++] =
                    2.0f * kPi * (bin + offset + 0.5f) / kOrientationBins;
            }
        }
        orientationCounts[keypointIndex] = count;
    }
}

__global__ void ScatterOrientationsKernel(const DeviceInputKeypoint* keypoints,
                                           const DeviceOrientationSet* orientationSets,
                                           const int* orientationCounts,
                                           const int* orientationOffsets,
                                           DeviceOrientedKeypoint* output,
                                           int keypointCount,
                                           int outputTag,
                                           bool preserveSourceIndex)
{
    const int sourceIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (sourceIndex >= keypointCount) return;
    const DeviceInputKeypoint source = keypoints[sourceIndex];
    const int outputOffset = orientationOffsets[sourceIndex];
    for (int orientation = 0; orientation < orientationCounts[sourceIndex]; ++orientation) {
        DeviceOrientedKeypoint destination;
        destination.x = source.x;
        destination.y = source.y;
        destination.size = source.size;
        destination.angle = orientationSets[sourceIndex].angles[orientation];
        destination.response = source.response;
        destination.sourceIndex = preserveSourceIndex ? sourceIndex : outputTag;
        destination.layer = source.layer;
        output[outputOffset + orientation] = destination;
    }
}

__global__ void ConvertRefinedKeypointsKernel(
    const PipelineRefinedKeypoint* source,
    DeviceInputKeypoint* destination,
    int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    destination[index] = {source[index].x, source[index].y, source[index].size,
                          source[index].response, source[index].layer};
}

void ExclusiveScan(const int* input, int* output, int count)
{
    void* temporary = nullptr;
    size_t temporaryBytes = 0;
    CheckCuda(cub::DeviceScan::ExclusiveSum(temporary, temporaryBytes,
                                            input, output, count),
              "CUB orientation scan size query");
    CheckCuda(cudaMalloc(&temporary, temporaryBytes),
              "cudaMalloc orientation scan temporary");
    try {
        CheckCuda(cub::DeviceScan::ExclusiveSum(temporary, temporaryBytes,
                                                input, output, count),
                  "CUB orientation exclusive scan");
        CheckCuda(cudaFree(temporary), "cudaFree orientation scan temporary");
    } catch (...) {
        cudaFree(temporary);
        throw;
    }
}

int ReadOutputCount(const int* counts, const int* offsets, int count)
{
    int lastCount = 0;
    int lastOffset = 0;
    CheckCuda(cudaMemcpy(&lastCount, counts + count - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "cudaMemcpy last orientation count");
    CheckCuda(cudaMemcpy(&lastOffset, offsets + count - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "cudaMemcpy last orientation offset");
    return lastCount + lastOffset;
}

template <typename Traits>
std::vector<OrientedKeypointSnapshot> AssignOrientations(
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<RefinedKeypointSnapshot>& keypoints,
    int octave)
{
    if (gaussian == nullptr || width <= 0 || height <= 0 || gaussianLayers <= 0) {
        throw std::invalid_argument("Invalid CUDA SIFT orientation image");
    }
    if (keypoints.empty()) return {};
    const size_t gaussianCount = static_cast<size_t>(width) * height * gaussianLayers;
    const int keypointCount = static_cast<int>(keypoints.size());
    std::vector<DeviceInputKeypoint> hostKeypoints(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        hostKeypoints[i] = {keypoints[i].x, keypoints[i].y, keypoints[i].size,
                            keypoints[i].response, keypoints[i].layer};
    }

    float* deviceGaussian = nullptr;
    DeviceInputKeypoint* deviceKeypoints = nullptr;
    DeviceOrientationSet* deviceOrientationSets = nullptr;
    int* deviceCounts = nullptr;
    int* deviceOffsets = nullptr;
    DeviceOrientedKeypoint* deviceOutput = nullptr;
    auto cleanup = [&]() {
        if (deviceOutput) cudaFree(deviceOutput);
        if (deviceOffsets) cudaFree(deviceOffsets);
        if (deviceCounts) cudaFree(deviceCounts);
        if (deviceOrientationSets) cudaFree(deviceOrientationSets);
        if (deviceKeypoints) cudaFree(deviceKeypoints);
        if (deviceGaussian) cudaFree(deviceGaussian);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceGaussian),
                             gaussianCount * sizeof(float)),
                  "cudaMalloc orientation Gaussian");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceKeypoints),
                             keypoints.size() * sizeof(DeviceInputKeypoint)),
                  "cudaMalloc orientation keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOrientationSets),
                             keypoints.size() * sizeof(DeviceOrientationSet)),
                  "cudaMalloc orientation sets");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceCounts),
                             keypoints.size() * sizeof(int)),
                  "cudaMalloc orientation counts");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOffsets),
                             keypoints.size() * sizeof(int)),
                  "cudaMalloc orientation offsets");
        CheckCuda(cudaMemcpy(deviceGaussian, gaussian,
                             gaussianCount * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy orientation Gaussian");
        CheckCuda(cudaMemcpy(deviceKeypoints, hostKeypoints.data(),
                             keypoints.size() * sizeof(DeviceInputKeypoint),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy orientation keypoints");

        BuildOrientationSetsKernel<Traits><<<keypointCount, kWarpSize>>>(
            deviceGaussian, deviceKeypoints, deviceOrientationSets, deviceCounts,
            keypointCount, width, height, gaussianLayers, octave);
        CheckCuda(cudaGetLastError(), "BuildOrientationSetsKernel launch");
        ExclusiveScan(deviceCounts, deviceOffsets, keypointCount);
        const int outputCount = ReadOutputCount(deviceCounts, deviceOffsets, keypointCount);
        if (outputCount == 0) {
            cleanup();
            return {};
        }
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOutput),
                             static_cast<size_t>(outputCount) *
                                 sizeof(DeviceOrientedKeypoint)),
                  "cudaMalloc oriented keypoints");
        constexpr int scatterBlock = 256;
        ScatterOrientationsKernel<<<(keypointCount + scatterBlock - 1) / scatterBlock,
                                    scatterBlock>>>(
            deviceKeypoints, deviceOrientationSets, deviceCounts, deviceOffsets,
            deviceOutput, keypointCount, 0, true);
        CheckCuda(cudaGetLastError(), "ScatterOrientationsKernel launch");

        std::vector<DeviceOrientedKeypoint> hostOutput(outputCount);
        CheckCuda(cudaMemcpy(hostOutput.data(), deviceOutput,
                             hostOutput.size() * sizeof(DeviceOrientedKeypoint),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy oriented keypoints");
        std::vector<OrientedKeypointSnapshot> result(outputCount);
        for (int i = 0; i < outputCount; ++i) {
            const auto& source = hostOutput[i];
            result[i] = {source.x, source.y, source.size, source.angle,
                         source.response, source.layer, source.sourceIndex};
        }
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}

__global__ void LimitOrientationCountsKernel(int* counts, int count, int maximum) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) counts[index] = min(counts[index], maximum);
}

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
    size_t& temporaryBytes, int maxOrientations)
{
    outputCount = 0;
    temporaryBytes = 0;
    if (deviceGaussian == nullptr || deviceKeypoints == nullptr ||
        width <= 0 || height <= 0 || gaussianLayers <= 0 || keypointCount <= 0) {
        return nullptr;
    }
    DeviceInputKeypoint* inputs = nullptr;
    DeviceOrientationSet* sets = nullptr;
    int* counts = nullptr;
    int* offsets = nullptr;
    DeviceOrientedKeypoint* output = nullptr;
    auto cleanup = [&]() {
        if (offsets) cudaFree(offsets);
        if (counts) cudaFree(counts);
        if (sets) cudaFree(sets);
        if (inputs) cudaFree(inputs);
    };
    try {
        const size_t inputBytes = static_cast<size_t>(keypointCount) *
                                  sizeof(DeviceInputKeypoint);
        const size_t setBytes = static_cast<size_t>(keypointCount) *
                                sizeof(DeviceOrientationSet);
        const size_t countBytes = static_cast<size_t>(keypointCount) * sizeof(int);
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&inputs), inputBytes),
                  "cudaMalloc pipeline orientation inputs");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sets), setBytes),
                  "cudaMalloc pipeline orientation sets");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&counts), countBytes),
                  "cudaMalloc pipeline orientation counts");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&offsets), countBytes),
                  "cudaMalloc pipeline orientation offsets");
        constexpr int conversionBlock = 256;
        ConvertRefinedKeypointsKernel<<<(keypointCount + conversionBlock - 1) /
                                           conversionBlock, conversionBlock>>>(
            deviceKeypoints, inputs, keypointCount);
        CheckCuda(cudaGetLastError(), "pipeline refined conversion launch");
        BuildOrientationSetsKernel<Traits><<<keypointCount, kWarpSize>>>(
            deviceGaussian, inputs, sets, counts, keypointCount,
            width, height, gaussianLayers, octave);
        CheckCuda(cudaGetLastError(), "pipeline orientation histogram launch");
        LimitOrientationCountsKernel<<<(keypointCount + 255) / 256, 256>>>(counts, keypointCount, maxOrientations);
        CheckCuda(cudaGetLastError(), "pipeline orientation count limit");
        ExclusiveScan(counts, offsets, keypointCount);
        outputCount = ReadOutputCount(counts, offsets, keypointCount);
        if (outputCount > 0) {
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&output),
                                 static_cast<size_t>(outputCount) *
                                     sizeof(DeviceOrientedKeypoint)),
                      "cudaMalloc pipeline oriented output");
            ScatterOrientationsKernel<<<(keypointCount + conversionBlock - 1) /
                                            conversionBlock, conversionBlock>>>(
                inputs, sets, counts, offsets, output, keypointCount,
                octave, false);
            CheckCuda(cudaGetLastError(), "pipeline orientation scatter launch");
        }
        temporaryBytes = inputBytes + setBytes + countBytes * 2 +
            static_cast<size_t>(outputCount) * sizeof(DeviceOrientedKeypoint);
        cleanup();
        return output;
    } catch (...) {
        cleanup();
        if (output) cudaFree(output);
        throw;
    }
}

template PipelineOrientedKeypoint* AssignOrientationsDevice<OpenCvModeTraits>(
    const float*, int, int, int, const PipelineRefinedKeypoint*, int, int, int&, size_t&, int);
template PipelineOrientedKeypoint* AssignOrientationsDevice<VlFeatModeTraits>(
    const float*, int, int, int, const PipelineRefinedKeypoint*, int, int, int&, size_t&, int);

std::vector<OrientedKeypointSnapshot> AssignOrientationsForTest(
    CudaSiftMode mode,
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<RefinedKeypointSnapshot>& keypoints,
    int octave)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return AssignOrientations<OpenCvModeTraits>(gaussian, width, height,
                                                    gaussianLayers, keypoints, octave);
    case CudaSiftMode::VLFeat:
        return AssignOrientations<VlFeatModeTraits>(gaussian, width, height,
                                                    gaussianLayers, keypoints, octave);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
