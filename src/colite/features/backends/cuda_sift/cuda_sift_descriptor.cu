#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace colite::features::cuda::detail {
namespace {

constexpr int kSpatialBins = 4;
constexpr int kOrientationBins = 8;
constexpr int kDescriptorLength = 128;
constexpr int kDescriptorThreads = 128;
constexpr int kWarpsPerBlock = kDescriptorThreads / 32;
constexpr float kPi = 3.14159265358979323846f;

using DeviceOrientedKeypoint = PipelineOrientedKeypoint;

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ int FloorToInt(float value)
{
    const int integer = static_cast<int>(value);
    return integer - (integer > value ? 1 : 0);
}

__device__ __forceinline__ int WrapOrientationBin(int bin)
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
    const float absoluteY = fabsf(y) + FLT_EPSILON;
    float ratio;
    float angle;
    if (x >= 0.0f) {
        ratio = (x - absoluteY) / (x + absoluteY);
        angle = kPi * 0.25f;
    } else {
        ratio = (x + absoluteY) / (absoluteY - x);
        angle = 3.0f * kPi * 0.25f;
    }
    angle += (0.1821f * ratio * ratio - 0.9675f) * ratio;
    return y < 0.0f ? -angle : angle;
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

struct DescriptorGeometry {
    float x = 0.0f;
    float y = 0.0f;
    float orientation = 0.0f;
    float spatialScale = 0.0f;
    float inverseSpatialScale = 0.0f;
    float sine = 0.0f;
    float cosine = 0.0f;
    int centerX = 0;
    int centerY = 0;
    int radius = 0;
};

template <typename Traits>
__device__ void InitializeDescriptorGeometry(
    const DeviceOrientedKeypoint& keypoint,
    int octave,
    int width,
    int height,
    DescriptorGeometry& geometry)
{
    const float octaveScale = ldexpf(1.0f, octave);
    geometry.x = keypoint.x / octaveScale;
    geometry.y = keypoint.y / octaveScale;
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        geometry.centerX = static_cast<int>(lroundf(geometry.x));
        geometry.centerY = static_cast<int>(lroundf(geometry.y));
        geometry.orientation = 360.0f - keypoint.angle;
        if (fabsf(geometry.orientation - 360.0f) < FLT_EPSILON) {
            geometry.orientation = 0.0f;
        }
        const float scale = keypoint.size * 0.5f / octaveScale;
        geometry.spatialScale = 3.0f * scale;
        const float angleRadians = geometry.orientation * kPi / 180.0f;
        geometry.cosine = cosf(angleRadians) / geometry.spatialScale;
        geometry.sine = sinf(angleRadians) / geometry.spatialScale;
        geometry.radius = static_cast<int>(lroundf(
            geometry.spatialScale * 1.4142135623730951f *
            (kSpatialBins + 1) * 0.5f));
        geometry.radius = min(geometry.radius, static_cast<int>(sqrtf(
            static_cast<float>(width * width + height * height))));
    } else {
        geometry.centerX = static_cast<int>(geometry.x + 0.5f);
        geometry.centerY = static_cast<int>(geometry.y + 0.5f);
        geometry.spatialScale = keypoint.size / octaveScale * 3.0f + DBL_EPSILON;
        geometry.inverseSpatialScale = 1.0f / geometry.spatialScale;
        geometry.radius = static_cast<int>(floorf(
            1.4142135623730951f * geometry.spatialScale *
            (kSpatialBins + 1) * 0.5f + 0.5f));
        geometry.sine = sinf(keypoint.angle);
        geometry.cosine = cosf(keypoint.angle);
    }
}

__device__ __forceinline__ void AddTrilinear(float* histogram,
                                             int row0,
                                             int column0,
                                             int orientation0,
                                             float rowFraction,
                                             float columnFraction,
                                             float orientationFraction,
                                             float magnitude)
{
    for (int dr = 0; dr < 2; ++dr) {
        const int row = row0 + dr;
        if (row < 0 || row >= kSpatialBins) continue;
        const float rowWeight = dr ? rowFraction : 1.0f - rowFraction;
        for (int dc = 0; dc < 2; ++dc) {
            const int column = column0 + dc;
            if (column < 0 || column >= kSpatialBins) continue;
            const float columnWeight = dc ? columnFraction : 1.0f - columnFraction;
            for (int direction = 0; direction < 2; ++direction) {
                const int orientation = WrapOrientationBin(orientation0 + direction);
                const float orientationWeight = direction
                    ? orientationFraction : 1.0f - orientationFraction;
                const int index = (row * kSpatialBins + column) *
                                  kOrientationBins + orientation;
                atomicAdd(histogram + index,
                          magnitude * rowWeight * columnWeight * orientationWeight);
            }
        }
    }
}

__device__ __forceinline__ float BlockReduceSum(float value, float* warpSums)
{
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) warpSums[warp] = value;
    __syncthreads();

    float blockSum = lane < kWarpsPerBlock ? warpSums[lane] : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            blockSum += __shfl_down_sync(0xffffffffu, blockSum, offset);
        }
        if (lane == 0) warpSums[0] = blockSum;
    }
    __syncthreads();
    return warpSums[0];
}

__device__ void AccumulateOpenCvDescriptor(const float* image,
                                           int width,
                                           int height,
                                           const DeviceOrientedKeypoint& keypoint,
                                           const DescriptorGeometry& geometry,
                                           float* partialHistogram)
{
    const int centerX = geometry.centerX;
    const int centerY = geometry.centerY;
    const float orientation = geometry.orientation;
    const float cosine = geometry.cosine;
    const float sine = geometry.sine;
    const int radius = geometry.radius;
    const int side = 2 * radius + 1;
    const int sampleCount = side * side;

    for (int sample = static_cast<int>(threadIdx.x);
         sample < sampleCount;
         sample += kDescriptorThreads) {
        const int rowOffset = sample / side - radius;
        const int columnOffset = sample - (rowOffset + radius) * side - radius;
        const int x = centerX + columnOffset;
        const int y = centerY + rowOffset;
        const float columnRotated = columnOffset * cosine - rowOffset * sine;
        const float rowRotated = columnOffset * sine + rowOffset * cosine;
        const float rowBin = rowRotated + 1.5f;
        const float columnBin = columnRotated + 1.5f;
        if (rowBin <= -1.0f || rowBin >= kSpatialBins ||
            columnBin <= -1.0f || columnBin >= kSpatialBins ||
            x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
            continue;
        }
        const float gx = image[static_cast<size_t>(y) * width + x + 1] -
                         image[static_cast<size_t>(y) * width + x - 1];
        const float gy = image[static_cast<size_t>(y - 1) * width + x] -
                         image[static_cast<size_t>(y + 1) * width + x];
        const float magnitude = sqrtf(gx * gx + gy * gy) *
            expf(-2.0f * (columnRotated * columnRotated +
                          rowRotated * rowRotated) /
                 (kSpatialBins * kSpatialBins));
        const float orientationBin =
            (FastAtan2OpenCv(gy, gx) - orientation) * kOrientationBins / 360.0f;
        const int row0 = FloorToInt(rowBin);
        const int column0 = FloorToInt(columnBin);
        int orientation0 = FloorToInt(orientationBin);
        const float rowFraction = rowBin - row0;
        const float columnFraction = columnBin - column0;
        const float orientationFraction = orientationBin - orientation0;
        orientation0 = WrapOrientationBin(orientation0);
        AddTrilinear(partialHistogram, row0, column0, orientation0,
                     rowFraction, columnFraction, orientationFraction, magnitude);
    }
}

__device__ void AccumulateVlFeatDescriptor(const float* image,
                                           int width,
                                           int height,
                                           const DeviceOrientedKeypoint& keypoint,
                                           const DescriptorGeometry& geometry,
                                           float* partialHistogram)
{
    const float x = geometry.x;
    const float y = geometry.y;
    const int centerX = geometry.centerX;
    const int centerY = geometry.centerY;
    const int radius = geometry.radius;
    const float sine = geometry.sine;
    const float cosine = geometry.cosine;
    const int side = 2 * radius + 1;
    const int sampleCount = side * side;

    for (int sample = static_cast<int>(threadIdx.x);
         sample < sampleCount;
         sample += kDescriptorThreads) {
        const int rowOffset = sample / side - radius;
        const int columnOffset = sample - (rowOffset + radius) * side - radius;
        const int sampleX = centerX + columnOffset;
        const int sampleY = centerY + rowOffset;
        if (sampleX < 1 || sampleX >= width - 1 ||
            sampleY < 1 || sampleY >= height - 1) {
            continue;
        }
        const float gx = 0.5f *
            (image[static_cast<size_t>(sampleY) * width + sampleX + 1] -
             image[static_cast<size_t>(sampleY) * width + sampleX - 1]);
        const float gy = 0.5f *
            (image[static_cast<size_t>(sampleY + 1) * width + sampleX] -
             image[static_cast<size_t>(sampleY - 1) * width + sampleX]);
        const float magnitude = FastSqrtVlFeat(gx * gx + gy * gy);
        float gradientAngle = FastAtan2VlFeat(gy, gx) + 2.0f * kPi;
        if (gradientAngle > 2.0f * kPi) gradientAngle -= 2.0f * kPi;
        float relativeAngle = gradientAngle - keypoint.angle;
        if (relativeAngle < 0.0f) relativeAngle += 2.0f * kPi;
        else if (relativeAngle > 2.0f * kPi) relativeAngle -= 2.0f * kPi;

        const float dx = static_cast<float>(sampleX) - x;
        const float dy = static_cast<float>(sampleY) - y;
        const float normalizedX = (cosine * dx + sine * dy) *
                                  geometry.inverseSpatialScale;
        const float normalizedY = (-sine * dx + cosine * dy) *
                                  geometry.inverseSpatialScale;
        const float orientationBin = kOrientationBins * relativeAngle /
                                     (2.0f * kPi);
        const float window = expf(-(normalizedX * normalizedX +
                                    normalizedY * normalizedY) / 8.0f);
        const int column0 = FloorToInt(normalizedX - 0.5f);
        const int row0 = FloorToInt(normalizedY - 0.5f);
        int orientation0 = FloorToInt(orientationBin);
        const float columnFraction = normalizedX - (column0 + 0.5f);
        const float rowFraction = normalizedY - (row0 + 0.5f);
        const float orientationFraction = orientationBin - orientation0;
        orientation0 = WrapOrientationBin(orientation0);
        AddTrilinear(partialHistogram, row0 + kSpatialBins / 2,
                     column0 + kSpatialBins / 2, orientation0,
                     rowFraction, columnFraction, orientationFraction,
                     window * magnitude);
    }
}

template <typename Traits>
__global__ void DescriptorKernel(const float* gaussian,
                                 const DeviceOrientedKeypoint* keypoints,
                                 float* descriptors,
                                 int keypointCount,
                                 int width,
                                 int height,
                                 int gaussianLayers,
                                 int octave)
{
    __shared__ float partial[kWarpsPerBlock * kDescriptorLength];
    __shared__ float warpSums[kWarpsPerBlock];
    __shared__ DeviceOrientedKeypoint sharedKeypoint;
    __shared__ DescriptorGeometry geometry;
    const int keypointIndex = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread >> 5;
    if (keypointIndex >= keypointCount) return;
    descriptors[static_cast<size_t>(keypointIndex) * kDescriptorLength + thread] = 0.0f;
    for (int index = thread; index < kWarpsPerBlock * kDescriptorLength;
         index += kDescriptorThreads) {
        partial[index] = 0.0f;
    }
    __syncthreads();

    if (thread == 0) {
        sharedKeypoint = keypoints[keypointIndex];
        InitializeDescriptorGeometry<Traits>(
            sharedKeypoint, octave, width, height, geometry);
    }
    __syncthreads();
    const DeviceOrientedKeypoint& keypoint = sharedKeypoint;
    const bool valid = keypoint.layer >= 0 && keypoint.layer < gaussianLayers &&
                       keypoint.size > 0.0f && geometry.centerX >= 0 &&
                       geometry.centerX < width && geometry.centerY >= 0 &&
                       geometry.centerY < height;
    if (!valid) return;
    const float* image = gaussian + static_cast<size_t>(keypoint.layer) * width * height;
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        AccumulateOpenCvDescriptor(image, width, height, keypoint, geometry,
                                   partial + warp * kDescriptorLength);
    } else {
        AccumulateVlFeatDescriptor(image, width, height, keypoint, geometry,
                                   partial + warp * kDescriptorLength);
    }
    __syncthreads();

    float value = 0.0f;
#pragma unroll
    for (int sourceWarp = 0; sourceWarp < kWarpsPerBlock; ++sourceWarp) {
        value += partial[sourceWarp * kDescriptorLength + thread];
    }
    const float firstSquaredNorm = BlockReduceSum(value * value, warpSums);

    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        const float threshold = sqrtf(firstSquaredNorm) * 0.2f;
        value = fminf(value, threshold);
        const float secondSquaredNorm = BlockReduceSum(value * value, warpSums);
        const float factor = 512.0f /
                             fmaxf(sqrtf(secondSquaredNorm), FLT_EPSILON);
        descriptors[static_cast<size_t>(keypointIndex) * kDescriptorLength + thread] =
            fminf(fmaxf(nearbyintf(value * factor), 0.0f), 255.0f);
    } else {
        const float firstNorm = sqrtf(firstSquaredNorm) + FLT_EPSILON;
        value = fminf(value / firstNorm, 0.2f);
        const float secondSquaredNorm = BlockReduceSum(value * value, warpSums);
        const float secondNorm = sqrtf(secondSquaredNorm) + FLT_EPSILON;
        descriptors[static_cast<size_t>(keypointIndex) * kDescriptorLength + thread] =
            value / secondNorm;
    }
}

template <typename Traits>
std::vector<float> ComputeDescriptors(
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<OrientedKeypointSnapshot>& keypoints,
    int octave)
{
    if (gaussian == nullptr || width <= 0 || height <= 0 || gaussianLayers <= 0) {
        throw std::invalid_argument("Invalid CUDA SIFT descriptor image");
    }
    if (keypoints.empty()) return {};
    const size_t gaussianCount = static_cast<size_t>(width) * height * gaussianLayers;
    std::vector<DeviceOrientedKeypoint> hostKeypoints(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        hostKeypoints[i].x = keypoints[i].x;
        hostKeypoints[i].y = keypoints[i].y;
        hostKeypoints[i].size = keypoints[i].size;
        hostKeypoints[i].angle = keypoints[i].angle;
        hostKeypoints[i].response = keypoints[i].response;
        hostKeypoints[i].sourceIndex = keypoints[i].sourceIndex;
        hostKeypoints[i].layer = keypoints[i].layer;
    }

    float* deviceGaussian = nullptr;
    DeviceOrientedKeypoint* deviceKeypoints = nullptr;
    float* deviceDescriptors = nullptr;
    auto cleanup = [&]() {
        if (deviceDescriptors) cudaFree(deviceDescriptors);
        if (deviceKeypoints) cudaFree(deviceKeypoints);
        if (deviceGaussian) cudaFree(deviceGaussian);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceGaussian),
                             gaussianCount * sizeof(float)),
                  "cudaMalloc descriptor Gaussian");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceKeypoints),
                             hostKeypoints.size() * sizeof(DeviceOrientedKeypoint)),
                  "cudaMalloc descriptor keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDescriptors),
                             hostKeypoints.size() * kDescriptorLength * sizeof(float)),
                  "cudaMalloc descriptors");
        CheckCuda(cudaMemcpy(deviceGaussian, gaussian,
                             gaussianCount * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy descriptor Gaussian");
        CheckCuda(cudaMemcpy(deviceKeypoints, hostKeypoints.data(),
                             hostKeypoints.size() * sizeof(DeviceOrientedKeypoint),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy descriptor keypoints");
        DescriptorKernel<Traits><<<static_cast<int>(keypoints.size()),
                                   kDescriptorThreads>>>(
            deviceGaussian, deviceKeypoints, deviceDescriptors,
            static_cast<int>(keypoints.size()), width, height,
            gaussianLayers, octave);
        CheckCuda(cudaGetLastError(), "DescriptorKernel launch");

        std::vector<float> result(keypoints.size() * kDescriptorLength);
        CheckCuda(cudaMemcpy(result.data(), deviceDescriptors,
                             result.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "cudaMemcpy descriptors");
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}

template <typename Traits>
float* ComputeDescriptorsDevice(const float* deviceGaussian,
                                int width,
                                int height,
                                int gaussianLayers,
                                const PipelineOrientedKeypoint* deviceKeypoints,
                                int keypointCount,
                                int octave,
                                size_t& allocatedBytes)
{
    allocatedBytes = 0;
    if (deviceGaussian == nullptr || deviceKeypoints == nullptr ||
        width <= 0 || height <= 0 || gaussianLayers <= 0 || keypointCount <= 0) {
        return nullptr;
    }
    float* descriptors = nullptr;
    const size_t bytes = static_cast<size_t>(keypointCount) *
                         kDescriptorLength * sizeof(float);
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&descriptors), bytes),
              "cudaMalloc pipeline descriptors");
    try {
        DescriptorKernel<Traits><<<keypointCount, kDescriptorThreads>>>(
            deviceGaussian, deviceKeypoints, descriptors, keypointCount,
            width, height, gaussianLayers, octave);
        CheckCuda(cudaGetLastError(), "pipeline descriptor launch");
        allocatedBytes = bytes;
        return descriptors;
    } catch (...) {
        cudaFree(descriptors);
        throw;
    }
}

template float* ComputeDescriptorsDevice<OpenCvModeTraits>(
    const float*, int, int, int, const PipelineOrientedKeypoint*, int, int, size_t&);
template float* ComputeDescriptorsDevice<VlFeatModeTraits>(
    const float*, int, int, int, const PipelineOrientedKeypoint*, int, int, size_t&);

std::vector<float> ComputeDescriptorsForTest(
    CudaSiftMode mode,
    const float* gaussian,
    int width,
    int height,
    int gaussianLayers,
    const std::vector<OrientedKeypointSnapshot>& keypoints,
    int octave)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return ComputeDescriptors<OpenCvModeTraits>(gaussian, width, height,
                                                    gaussianLayers, keypoints, octave);
    case CudaSiftMode::VLFeat:
        return ComputeDescriptors<VlFeatModeTraits>(gaussian, width, height,
                                                    gaussianLayers, keypoints, octave);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
