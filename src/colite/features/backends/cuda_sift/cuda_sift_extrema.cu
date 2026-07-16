#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace colite::features::cuda::detail {
namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpsPerBlock = kBlockSize / 32;
constexpr int kMaxRefinementSteps = 5;
constexpr int kOpenCvBorder = 5;

struct DeviceCandidates {
    int* x = nullptr;
    int* y = nullptr;
    int* layer = nullptr;
};

using DeviceRefinedKeypoint = PipelineRefinedKeypoint;

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ size_t DogIndex(int layer,
                                           int y,
                                           int x,
                                           int width,
                                           int height)
{
    return (static_cast<size_t>(layer) * height + y) * width + x;
}

__device__ __forceinline__ double SampleDogDouble(const float* dog,
                                                  int width,
                                                  int height,
                                                  int layer,
                                                  int y,
                                                  int x)
{
    return static_cast<double>(dog[DogIndex(layer, y, x, width, height)]);
}

template <typename Traits>
__device__ __forceinline__ bool PassesPreThreshold(float value,
                                                   float contrastThreshold,
                                                   float peakThreshold,
                                                   int octaveLayers)
{
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        const float threshold = floorf(0.5f * contrastThreshold /
                                       static_cast<float>(octaveLayers) * 255.0f);
        return fabsf(value) > threshold;
    } else {
        const float threshold = 0.8f * peakThreshold;
        return value >= threshold || value <= -threshold;
    }
}

template <typename Traits>
__device__ __forceinline__ bool IsScaleSpaceExtremum(const float* dog,
                                                     int x,
                                                     int y,
                                                     int layer,
                                                     int width,
                                                     int height,
                                                     int octaveLayers,
                                                     float contrastThreshold,
                                                     float peakThreshold)
{
    const int border = std::is_same_v<Traits, OpenCvModeTraits> ? kOpenCvBorder : 1;
    if (x < border || x >= width - border || y < border || y >= height - border) {
        return false;
    }

    const float value = dog[DogIndex(layer, y, x, width, height)];
    if (!PassesPreThreshold<Traits>(value, contrastThreshold, peakThreshold,
                                    octaveLayers)) {
        return false;
    }

    for (int ds = -1; ds <= 1; ++ds) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && ds == 0) continue;
                const float neighbor = dog[DogIndex(layer + ds, y + dy, x + dx,
                                                     width, height)];
                if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
                    if (value > 0.0f ? value < neighbor : value > neighbor) return false;
                } else {
                    if (value > 0.0f ? value <= neighbor : value >= neighbor) return false;
                }
            }
        }
    }
    return true;
}

template <typename Traits>
__global__ void BuildExtremaMasksKernel(const float* dog,
                                        unsigned int* warpMasks,
                                        int* blockCounts,
                                        int width,
                                        int height,
                                        int octaveLayers,
                                        float contrastThreshold,
                                        float peakThreshold,
                                        int totalPositions)
{
    __shared__ int warpCounts[kWarpsPerBlock];
    const int global = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int plane = width * height;
    bool accepted = false;
    if (global < totalPositions) {
        const int layer = global / plane + 1;
        const int pixel = global - (layer - 1) * plane;
        const int y = pixel / width;
        const int x = pixel - y * width;
        accepted = IsScaleSpaceExtremum<Traits>(dog, x, y, layer, width, height,
                                                octaveLayers, contrastThreshold,
                                                peakThreshold);
    }

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const unsigned int mask = __ballot_sync(0xffffffffu, accepted);
    if (lane == 0) {
        warpMasks[static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp] = mask;
        warpCounts[warp] = __popc(mask);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int count = 0;
#pragma unroll
        for (int i = 0; i < kWarpsPerBlock; ++i) count += warpCounts[i];
        blockCounts[blockIdx.x] = count;
    }
}

__global__ void ScatterCandidatesKernel(const unsigned int* warpMasks,
                                        const int* blockOffsets,
                                        DeviceCandidates output,
                                        int width,
                                        int height,
                                        int totalPositions)
{
    const int global = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (global >= totalPositions) return;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const unsigned int* blockMasks = warpMasks +
        static_cast<size_t>(blockIdx.x) * kWarpsPerBlock;
    const unsigned int mask = blockMasks[warp];
    if ((mask & (1u << lane)) == 0) return;

    int rank = 0;
#pragma unroll
    for (int i = 0; i < kWarpsPerBlock; ++i) {
        if (i >= warp) break;
        rank += __popc(blockMasks[i]);
    }
    const unsigned int lowerLaneMask = lane == 0 ? 0u : ((1u << lane) - 1u);
    rank += __popc(mask & lowerLaneMask);
    const int outputIndex = blockOffsets[blockIdx.x] + rank;

    const int plane = width * height;
    const int layer = global / plane + 1;
    const int pixel = global - (layer - 1) * plane;
    const int y = pixel / width;
    output.x[outputIndex] = pixel - y * width;
    output.y[outputIndex] = y;
    output.layer[outputIndex] = layer;
}

template <typename T>
__device__ bool Solve3x3(T matrix[9], T rhs[3], T result[3])
{
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        T pivotAbs = fabs(matrix[column * 3 + column]);
        for (int row = column + 1; row < 3; ++row) {
            const T candidate = fabs(matrix[row * 3 + column]);
            if (candidate > pivotAbs) {
                pivotAbs = candidate;
                pivot = row;
            }
        }
        if (pivotAbs < static_cast<T>(1e-10)) return false;
        if (pivot != column) {
            for (int j = column; j < 3; ++j) {
                const T temporary = matrix[column * 3 + j];
                matrix[column * 3 + j] = matrix[pivot * 3 + j];
                matrix[pivot * 3 + j] = temporary;
            }
            const T temporary = rhs[column];
            rhs[column] = rhs[pivot];
            rhs[pivot] = temporary;
        }
        const T inversePivot = static_cast<T>(1) / matrix[column * 3 + column];
        for (int j = column; j < 3; ++j) matrix[column * 3 + j] *= inversePivot;
        rhs[column] *= inversePivot;
        for (int row = column + 1; row < 3; ++row) {
            const T factor = matrix[row * 3 + column];
            for (int j = column; j < 3; ++j) {
                matrix[row * 3 + j] -= factor * matrix[column * 3 + j];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    result[2] = rhs[2];
    result[1] = rhs[1] - matrix[1 * 3 + 2] * result[2];
    result[0] = rhs[0] - matrix[0 * 3 + 1] * result[1] -
                matrix[0 * 3 + 2] * result[2];
    return true;
}

__device__ bool RefineOpenCv(const float* dog,
                             int width,
                             int height,
                             int octaveLayers,
                             float contrastThreshold,
                             float edgeThreshold,
                             float sigma,
                             int octave,
                             int x,
                             int y,
                             int layer,
                             DeviceRefinedKeypoint& output)
{
    constexpr float imageScale = 1.0f / 255.0f;
    constexpr float derivativeScale = 0.5f / 255.0f;
    constexpr float secondDerivativeScale = 1.0f / 255.0f;
    constexpr float crossDerivativeScale = 0.25f / 255.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float offsetScale = 0.0f;
    int currentX = x;
    int currentY = y;
    int currentLayer = layer;
    int iteration = 0;

    for (; iteration < kMaxRefinementSteps; ++iteration) {
        const float center = dog[DogIndex(currentLayer, currentY, currentX, width, height)];
        float derivative[3] = {
            (dog[DogIndex(currentLayer, currentY, currentX + 1, width, height)] -
             dog[DogIndex(currentLayer, currentY, currentX - 1, width, height)]) * derivativeScale,
            (dog[DogIndex(currentLayer, currentY + 1, currentX, width, height)] -
             dog[DogIndex(currentLayer, currentY - 1, currentX, width, height)]) * derivativeScale,
            (dog[DogIndex(currentLayer + 1, currentY, currentX, width, height)] -
             dog[DogIndex(currentLayer - 1, currentY, currentX, width, height)]) * derivativeScale
        };
        const float twiceCenter = 2.0f * center;
        const float dxx = (dog[DogIndex(currentLayer, currentY, currentX + 1, width, height)] +
                           dog[DogIndex(currentLayer, currentY, currentX - 1, width, height)] -
                           twiceCenter) * secondDerivativeScale;
        const float dyy = (dog[DogIndex(currentLayer, currentY + 1, currentX, width, height)] +
                           dog[DogIndex(currentLayer, currentY - 1, currentX, width, height)] -
                           twiceCenter) * secondDerivativeScale;
        const float dss = (dog[DogIndex(currentLayer + 1, currentY, currentX, width, height)] +
                           dog[DogIndex(currentLayer - 1, currentY, currentX, width, height)] -
                           twiceCenter) * secondDerivativeScale;
        const float dxy = (dog[DogIndex(currentLayer, currentY + 1, currentX + 1, width, height)] -
                           dog[DogIndex(currentLayer, currentY - 1, currentX + 1, width, height)] -
                           dog[DogIndex(currentLayer, currentY + 1, currentX - 1, width, height)] +
                           dog[DogIndex(currentLayer, currentY - 1, currentX - 1, width, height)]) * crossDerivativeScale;
        const float dxs = (dog[DogIndex(currentLayer + 1, currentY, currentX + 1, width, height)] -
                           dog[DogIndex(currentLayer + 1, currentY, currentX - 1, width, height)] -
                           dog[DogIndex(currentLayer - 1, currentY, currentX + 1, width, height)] +
                           dog[DogIndex(currentLayer - 1, currentY, currentX - 1, width, height)]) * crossDerivativeScale;
        const float dys = (dog[DogIndex(currentLayer + 1, currentY + 1, currentX, width, height)] -
                           dog[DogIndex(currentLayer + 1, currentY - 1, currentX, width, height)] -
                           dog[DogIndex(currentLayer - 1, currentY + 1, currentX, width, height)] +
                           dog[DogIndex(currentLayer - 1, currentY - 1, currentX, width, height)]) * crossDerivativeScale;
        float hessian[9] = {dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss};
        float solution[3];
        if (!Solve3x3(hessian, derivative, solution)) return false;
        offsetX = -solution[0];
        offsetY = -solution[1];
        offsetScale = -solution[2];
        if (fabsf(offsetX) < 0.5f && fabsf(offsetY) < 0.5f &&
            fabsf(offsetScale) < 0.5f) {
            break;
        }
        currentX += static_cast<int>(lroundf(offsetX));
        currentY += static_cast<int>(lroundf(offsetY));
        currentLayer += static_cast<int>(lroundf(offsetScale));
        if (currentLayer < 1 || currentLayer > octaveLayers ||
            currentX < kOpenCvBorder || currentX >= width - kOpenCvBorder ||
            currentY < kOpenCvBorder || currentY >= height - kOpenCvBorder) {
            return false;
        }
    }
    if (iteration >= kMaxRefinementSteps) return false;

    const float center = dog[DogIndex(currentLayer, currentY, currentX, width, height)];
    const float dx = (dog[DogIndex(currentLayer, currentY, currentX + 1, width, height)] -
                      dog[DogIndex(currentLayer, currentY, currentX - 1, width, height)]) * derivativeScale;
    const float dy = (dog[DogIndex(currentLayer, currentY + 1, currentX, width, height)] -
                      dog[DogIndex(currentLayer, currentY - 1, currentX, width, height)]) * derivativeScale;
    const float ds = (dog[DogIndex(currentLayer + 1, currentY, currentX, width, height)] -
                      dog[DogIndex(currentLayer - 1, currentY, currentX, width, height)]) * derivativeScale;
    const float contrast = center * imageScale +
        0.5f * (dx * offsetX + dy * offsetY + ds * offsetScale);
    if (fabsf(contrast) * octaveLayers < contrastThreshold) return false;

    const float twiceCenter = 2.0f * center;
    const float dxx = (dog[DogIndex(currentLayer, currentY, currentX + 1, width, height)] +
                       dog[DogIndex(currentLayer, currentY, currentX - 1, width, height)] -
                       twiceCenter) * secondDerivativeScale;
    const float dyy = (dog[DogIndex(currentLayer, currentY + 1, currentX, width, height)] +
                       dog[DogIndex(currentLayer, currentY - 1, currentX, width, height)] -
                       twiceCenter) * secondDerivativeScale;
    const float dxy = (dog[DogIndex(currentLayer, currentY + 1, currentX + 1, width, height)] -
                       dog[DogIndex(currentLayer, currentY - 1, currentX + 1, width, height)] -
                       dog[DogIndex(currentLayer, currentY + 1, currentX - 1, width, height)] +
                       dog[DogIndex(currentLayer, currentY - 1, currentX - 1, width, height)]) * crossDerivativeScale;
    const float trace = dxx + dyy;
    const float determinant = dxx * dyy - dxy * dxy;
    const float edgeLimit = (edgeThreshold + 1.0f) * (edgeThreshold + 1.0f);
    if (determinant <= 0.0f || trace * trace * edgeThreshold >= edgeLimit * determinant) {
        return false;
    }

    const float octaveScale = ldexpf(1.0f, octave);
    output.x = (currentX + offsetX) * octaveScale;
    output.y = (currentY + offsetY) * octaveScale;
    output.refinedLayer = currentLayer + offsetScale;
    output.size = sigma * powf(2.0f, output.refinedLayer / octaveLayers) * octaveScale * 2.0f;
    output.response = fabsf(contrast);
    output.layer = currentLayer;
    return true;
}

__device__ bool RefineVlFeat(const float* dog,
                             int width,
                             int height,
                             int dogLayers,
                             int octaveLayers,
                             float peakThreshold,
                             float edgeThreshold,
                             float sigma,
                             int octave,
                             int x,
                             int y,
                             int layer,
                             DeviceRefinedKeypoint& output)
{
    int currentX = x;
    int currentY = y;
    int moveX = 0;
    int moveY = 0;
    double derivativeX = 0.0;
    double derivativeY = 0.0;
    double derivativeScale = 0.0;
    double dxx = 0.0;
    double dyy = 0.0;
    double dss = 0.0;
    double dxy = 0.0;
    double dxs = 0.0;
    double dys = 0.0;
    double offset[3] = {0.0, 0.0, 0.0};

    for (int iteration = 0; iteration < kMaxRefinementSteps; ++iteration) {
        currentX += moveX;
        currentY += moveY;
#define SAMPLE_DOG(dx, dy, ds) \
        SampleDogDouble(dog, width, height, layer + (ds), \
                        currentY + (dy), currentX + (dx))
        derivativeX = 0.5 * (SAMPLE_DOG(1, 0, 0) - SAMPLE_DOG(-1, 0, 0));
        derivativeY = 0.5 * (SAMPLE_DOG(0, 1, 0) - SAMPLE_DOG(0, -1, 0));
        derivativeScale = 0.5 * (SAMPLE_DOG(0, 0, 1) - SAMPLE_DOG(0, 0, -1));
        const double twiceCenter = 2.0 * SAMPLE_DOG(0, 0, 0);
        dxx = SAMPLE_DOG(1, 0, 0) + SAMPLE_DOG(-1, 0, 0) - twiceCenter;
        dyy = SAMPLE_DOG(0, 1, 0) + SAMPLE_DOG(0, -1, 0) - twiceCenter;
        dss = SAMPLE_DOG(0, 0, 1) + SAMPLE_DOG(0, 0, -1) - twiceCenter;
        dxy = 0.25 * (SAMPLE_DOG(1, 1, 0) + SAMPLE_DOG(-1, -1, 0) -
                      SAMPLE_DOG(-1, 1, 0) - SAMPLE_DOG(1, -1, 0));
        dxs = 0.25 * (SAMPLE_DOG(1, 0, 1) + SAMPLE_DOG(-1, 0, -1) -
                      SAMPLE_DOG(-1, 0, 1) - SAMPLE_DOG(1, 0, -1));
        dys = 0.25 * (SAMPLE_DOG(0, 1, 1) + SAMPLE_DOG(0, -1, -1) -
                      SAMPLE_DOG(0, -1, 1) - SAMPLE_DOG(0, 1, -1));
#undef SAMPLE_DOG
        double hessian[9] = {dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss};
        double rhs[3] = {-derivativeX, -derivativeY, -derivativeScale};
        if (!Solve3x3(hessian, rhs, offset)) return false;
        moveX = (offset[0] > 0.6 && currentX < width - 2 ? 1 : 0) +
                (offset[0] < -0.6 && currentX > 1 ? -1 : 0);
        moveY = (offset[1] > 0.6 && currentY < height - 2 ? 1 : 0) +
                (offset[1] < -0.6 && currentY > 1 ? -1 : 0);
        if (moveX == 0 && moveY == 0) break;
    }

    const double center = dog[DogIndex(layer, currentY, currentX, width, height)];
    const double value = center + 0.5 * (derivativeX * offset[0] +
                                         derivativeY * offset[1] +
                                         derivativeScale * offset[2]);
    const double determinant = dxx * dyy - dxy * dxy;
    const double score = determinant == 0.0 ? 1.0e300 :
        (dxx + dyy) * (dxx + dyy) / determinant;
    const double refinedX = currentX + offset[0];
    const double refinedY = currentY + offset[1];
    const double refinedLayer = layer + offset[2];
    const double edgeLimit = (edgeThreshold + 1.0) * (edgeThreshold + 1.0) /
                             edgeThreshold;
    if (fabs(value) <= peakThreshold || score < 0.0 || score >= edgeLimit ||
        fabs(offset[0]) >= 1.5 || fabs(offset[1]) >= 1.5 || fabs(offset[2]) >= 1.5 ||
        refinedX < 0.0 || refinedX > width - 1 ||
        refinedY < 0.0 || refinedY > height - 1 ||
        refinedLayer < 0.0 || refinedLayer > dogLayers - 1) {
        return false;
    }

    const float octaveScale = ldexpf(1.0f, octave);
    output.x = static_cast<float>(refinedX) * octaveScale;
    output.y = static_cast<float>(refinedY) * octaveScale;
    output.refinedLayer = static_cast<float>(refinedLayer);
    output.size = sigma * powf(2.0f, output.refinedLayer / octaveLayers) * octaveScale;
    output.response = static_cast<float>(fabs(value));
    output.layer = layer;
    return true;
}

template <typename Traits>
__global__ void RefineCandidatesKernel(const float* dog,
                                       DeviceCandidates candidates,
                                       DeviceRefinedKeypoint* provisional,
                                       int* valid,
                                       int count,
                                       int width,
                                       int height,
                                       int dogLayers,
                                       int octaveLayers,
                                       float contrastThreshold,
                                       float peakThreshold,
                                       float edgeThreshold,
                                       float sigma,
                                       int octave)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    DeviceRefinedKeypoint refined{};
    bool accepted;
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        accepted = RefineOpenCv(dog, width, height, octaveLayers,
                                contrastThreshold, edgeThreshold, sigma, octave,
                                candidates.x[index], candidates.y[index],
                                candidates.layer[index], refined);
    } else {
        accepted = RefineVlFeat(dog, width, height, dogLayers, octaveLayers,
                                peakThreshold, edgeThreshold, sigma, octave,
                                candidates.x[index], candidates.y[index],
                                candidates.layer[index], refined);
    }
    valid[index] = accepted ? 1 : 0;
    if (accepted) provisional[index] = refined;
}

__global__ void ScatterRefinedKernel(const DeviceRefinedKeypoint* provisional,
                                     const int* valid,
                                     const int* offsets,
                                     DeviceRefinedKeypoint* compact,
                                     int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count && valid[index]) compact[offsets[index]] = provisional[index];
}

int ReadCompactedCount(const int* counts, const int* offsets, int length)
{
    if (length <= 0) return 0;
    int lastCount = 0;
    int lastOffset = 0;
    CheckCuda(cudaMemcpy(&lastCount, counts + length - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "cudaMemcpy last compact count");
    CheckCuda(cudaMemcpy(&lastOffset, offsets + length - 1, sizeof(int),
                         cudaMemcpyDeviceToHost), "cudaMemcpy last compact offset");
    return lastCount + lastOffset;
}

void ExclusiveScan(const int* input, int* output, int count)
{
    void* temporary = nullptr;
    size_t temporaryBytes = 0;
    CheckCuda(cub::DeviceScan::ExclusiveSum(temporary, temporaryBytes,
                                            input, output, count),
              "CUB scan size query");
    CheckCuda(cudaMalloc(&temporary, temporaryBytes), "cudaMalloc CUB scan temporary");
    try {
        CheckCuda(cub::DeviceScan::ExclusiveSum(temporary, temporaryBytes,
                                                input, output, count),
                  "CUB exclusive scan");
        CheckCuda(cudaFree(temporary), "cudaFree CUB scan temporary");
    } catch (...) {
        cudaFree(temporary);
        throw;
    }
}

template <typename Traits>
ExtremaTestResult DetectExtrema(const float* dog,
                                int width,
                                int height,
                                int dogLayers,
                                int octaveLayers,
                                float contrastThreshold,
                                float peakThreshold,
                                float edgeThreshold,
                                float sigma,
                                int octave)
{
    if (dog == nullptr || width <= 0 || height <= 0 || octaveLayers <= 0 ||
        dogLayers < octaveLayers + 2 || edgeThreshold <= 0.0f || sigma <= 0.0f) {
        throw std::invalid_argument("Invalid CUDA SIFT extrema parameters");
    }
    const size_t dogCount = static_cast<size_t>(width) * height * dogLayers;
    const int totalPositions = width * height * octaveLayers;
    const int blockCount = (totalPositions + kBlockSize - 1) / kBlockSize;
    const int warpMaskCount = blockCount * kWarpsPerBlock;

    float* deviceDog = nullptr;
    unsigned int* deviceWarpMasks = nullptr;
    int* deviceBlockCounts = nullptr;
    int* deviceBlockOffsets = nullptr;
    DeviceCandidates candidates;
    DeviceRefinedKeypoint* provisional = nullptr;
    DeviceRefinedKeypoint* compact = nullptr;
    int* valid = nullptr;
    int* validOffsets = nullptr;
    auto cleanup = [&]() {
        if (validOffsets) cudaFree(validOffsets);
        if (valid) cudaFree(valid);
        if (compact) cudaFree(compact);
        if (provisional) cudaFree(provisional);
        if (candidates.layer) cudaFree(candidates.layer);
        if (candidates.y) cudaFree(candidates.y);
        if (candidates.x) cudaFree(candidates.x);
        if (deviceBlockOffsets) cudaFree(deviceBlockOffsets);
        if (deviceBlockCounts) cudaFree(deviceBlockCounts);
        if (deviceWarpMasks) cudaFree(deviceWarpMasks);
        if (deviceDog) cudaFree(deviceDog);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDog), dogCount * sizeof(float)),
                  "cudaMalloc extrema DoG");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceWarpMasks),
                             static_cast<size_t>(warpMaskCount) * sizeof(unsigned int)),
                  "cudaMalloc extrema warp masks");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceBlockCounts),
                             static_cast<size_t>(blockCount) * sizeof(int)),
                  "cudaMalloc extrema block counts");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceBlockOffsets),
                             static_cast<size_t>(blockCount) * sizeof(int)),
                  "cudaMalloc extrema block offsets");
        CheckCuda(cudaMemcpy(deviceDog, dog, dogCount * sizeof(float),
                             cudaMemcpyHostToDevice), "cudaMemcpy extrema DoG");

        BuildExtremaMasksKernel<Traits><<<blockCount, kBlockSize>>>(
            deviceDog, deviceWarpMasks, deviceBlockCounts, width, height,
            octaveLayers, contrastThreshold, peakThreshold, totalPositions);
        CheckCuda(cudaGetLastError(), "BuildExtremaMasksKernel launch");
        ExclusiveScan(deviceBlockCounts, deviceBlockOffsets, blockCount);
        const int candidateCount = ReadCompactedCount(deviceBlockCounts,
                                                      deviceBlockOffsets,
                                                      blockCount);
        ExtremaTestResult result;
        if (candidateCount == 0) {
            cleanup();
            return result;
        }

        const size_t candidateBytes = static_cast<size_t>(candidateCount) * sizeof(int);
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.x), candidateBytes),
                  "cudaMalloc candidate x");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.y), candidateBytes),
                  "cudaMalloc candidate y");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.layer), candidateBytes),
                  "cudaMalloc candidate layer");
        ScatterCandidatesKernel<<<blockCount, kBlockSize>>>(
            deviceWarpMasks, deviceBlockOffsets, candidates,
            width, height, totalPositions);
        CheckCuda(cudaGetLastError(), "ScatterCandidatesKernel launch");

        std::vector<int> hostX(candidateCount);
        std::vector<int> hostY(candidateCount);
        std::vector<int> hostLayer(candidateCount);
        CheckCuda(cudaMemcpy(hostX.data(), candidates.x, candidateBytes,
                             cudaMemcpyDeviceToHost), "cudaMemcpy candidate x");
        CheckCuda(cudaMemcpy(hostY.data(), candidates.y, candidateBytes,
                             cudaMemcpyDeviceToHost), "cudaMemcpy candidate y");
        CheckCuda(cudaMemcpy(hostLayer.data(), candidates.layer, candidateBytes,
                             cudaMemcpyDeviceToHost), "cudaMemcpy candidate layer");
        result.initialCandidates.resize(candidateCount);
        for (int i = 0; i < candidateCount; ++i) {
            result.initialCandidates[i] = {hostX[i], hostY[i], hostLayer[i]};
        }

        const size_t refinedBytes = static_cast<size_t>(candidateCount) *
                                    sizeof(DeviceRefinedKeypoint);
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&provisional), refinedBytes),
                  "cudaMalloc provisional keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&compact), refinedBytes),
                  "cudaMalloc compact keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&valid), candidateBytes),
                  "cudaMalloc keypoint validity");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&validOffsets), candidateBytes),
                  "cudaMalloc keypoint offsets");
        const int refinementBlocks = (candidateCount + kBlockSize - 1) / kBlockSize;
        RefineCandidatesKernel<Traits><<<refinementBlocks, kBlockSize>>>(
            deviceDog, candidates, provisional, valid, candidateCount,
            width, height, dogLayers, octaveLayers, contrastThreshold,
            peakThreshold, edgeThreshold, sigma, octave);
        CheckCuda(cudaGetLastError(), "RefineCandidatesKernel launch");
        ExclusiveScan(valid, validOffsets, candidateCount);
        const int refinedCount = ReadCompactedCount(valid, validOffsets, candidateCount);
        if (refinedCount > 0) {
            ScatterRefinedKernel<<<refinementBlocks, kBlockSize>>>(
                provisional, valid, validOffsets, compact, candidateCount);
            CheckCuda(cudaGetLastError(), "ScatterRefinedKernel launch");
            std::vector<DeviceRefinedKeypoint> hostRefined(refinedCount);
            CheckCuda(cudaMemcpy(hostRefined.data(), compact,
                                 static_cast<size_t>(refinedCount) *
                                     sizeof(DeviceRefinedKeypoint),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy refined keypoints");
            result.keypoints.resize(refinedCount);
            for (int i = 0; i < refinedCount; ++i) {
                const auto& source = hostRefined[i];
                result.keypoints[i] = {source.x, source.y, source.refinedLayer,
                                       source.size, source.response, source.layer};
            }
        }
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}

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
                                             size_t& temporaryBytes)
{
    outputCount = 0;
    temporaryBytes = 0;
    if (deviceDog == nullptr || width <= 0 || height <= 0 || octaveLayers <= 0 ||
        dogLayers < octaveLayers + 2) {
        throw std::invalid_argument("Invalid device extrema parameters");
    }
    const int totalPositions = width * height * octaveLayers;
    const int blockCount = (totalPositions + kBlockSize - 1) / kBlockSize;
    const int warpMaskCount = blockCount * kWarpsPerBlock;
    unsigned int* warpMasks = nullptr;
    int* blockCounts = nullptr;
    int* blockOffsets = nullptr;
    DeviceCandidates candidates;
    DeviceRefinedKeypoint* provisional = nullptr;
    DeviceRefinedKeypoint* compact = nullptr;
    int* valid = nullptr;
    int* validOffsets = nullptr;
    auto cleanup = [&]() {
        if (validOffsets) cudaFree(validOffsets);
        if (valid) cudaFree(valid);
        if (provisional) cudaFree(provisional);
        if (candidates.layer) cudaFree(candidates.layer);
        if (candidates.y) cudaFree(candidates.y);
        if (candidates.x) cudaFree(candidates.x);
        if (blockOffsets) cudaFree(blockOffsets);
        if (blockCounts) cudaFree(blockCounts);
        if (warpMasks) cudaFree(warpMasks);
    };
    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&warpMasks),
                             static_cast<size_t>(warpMaskCount) * sizeof(unsigned int)),
                  "cudaMalloc pipeline extrema masks");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&blockCounts),
                             static_cast<size_t>(blockCount) * sizeof(int)),
                  "cudaMalloc pipeline extrema counts");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&blockOffsets),
                             static_cast<size_t>(blockCount) * sizeof(int)),
                  "cudaMalloc pipeline extrema offsets");
        BuildExtremaMasksKernel<Traits><<<blockCount, kBlockSize>>>(
            deviceDog, warpMasks, blockCounts, width, height, octaveLayers,
            contrastThreshold, peakThreshold, totalPositions);
        CheckCuda(cudaGetLastError(), "pipeline extrema mask launch");
        ExclusiveScan(blockCounts, blockOffsets, blockCount);
        const int candidateCount = ReadCompactedCount(blockCounts, blockOffsets, blockCount);
        if (candidateCount == 0) {
            cleanup();
            return nullptr;
        }
        const size_t candidateBytes = static_cast<size_t>(candidateCount) * sizeof(int);
        const size_t refinedBytes = static_cast<size_t>(candidateCount) *
                                    sizeof(DeviceRefinedKeypoint);
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.x), candidateBytes),
                  "cudaMalloc pipeline candidate x");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.y), candidateBytes),
                  "cudaMalloc pipeline candidate y");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&candidates.layer), candidateBytes),
                  "cudaMalloc pipeline candidate layer");
        ScatterCandidatesKernel<<<blockCount, kBlockSize>>>(
            warpMasks, blockOffsets, candidates, width, height, totalPositions);
        CheckCuda(cudaGetLastError(), "pipeline candidate scatter launch");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&provisional), refinedBytes),
                  "cudaMalloc pipeline provisional keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&compact), refinedBytes),
                  "cudaMalloc pipeline compact keypoints");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&valid), candidateBytes),
                  "cudaMalloc pipeline valid flags");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&validOffsets), candidateBytes),
                  "cudaMalloc pipeline valid offsets");
        const int refinementBlocks = (candidateCount + kBlockSize - 1) / kBlockSize;
        RefineCandidatesKernel<Traits><<<refinementBlocks, kBlockSize>>>(
            deviceDog, candidates, provisional, valid, candidateCount,
            width, height, dogLayers, octaveLayers, contrastThreshold,
            peakThreshold, edgeThreshold, sigma, octave);
        CheckCuda(cudaGetLastError(), "pipeline extrema refinement launch");
        ExclusiveScan(valid, validOffsets, candidateCount);
        outputCount = ReadCompactedCount(valid, validOffsets, candidateCount);
        if (outputCount > 0) {
            ScatterRefinedKernel<<<refinementBlocks, kBlockSize>>>(
                provisional, valid, validOffsets, compact, candidateCount);
            CheckCuda(cudaGetLastError(), "pipeline refined scatter launch");
        }
        temporaryBytes = static_cast<size_t>(warpMaskCount) * sizeof(unsigned int) +
            static_cast<size_t>(2 * blockCount) * sizeof(int) +
            candidateBytes * 5 + refinedBytes * 2;
        cleanup();
        if (outputCount == 0) {
            cudaFree(compact);
            return nullptr;
        }
        return compact;
    } catch (...) {
        cleanup();
        if (compact) cudaFree(compact);
        throw;
    }
}

template PipelineRefinedKeypoint* DetectExtremaDevice<OpenCvModeTraits>(
    const float*, int, int, int, int, float, float, float, float, int, int&, size_t&);
template PipelineRefinedKeypoint* DetectExtremaDevice<VlFeatModeTraits>(
    const float*, int, int, int, int, float, float, float, float, int, int&, size_t&);

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
                                        int octave)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return DetectExtrema<OpenCvModeTraits>(dog, width, height, dogLayers,
                                               octaveLayers, contrastThreshold,
                                               peakThreshold, edgeThreshold,
                                               sigma, octave);
    case CudaSiftMode::VLFeat:
        return DetectExtrema<VlFeatModeTraits>(dog, width, height, dogLayers,
                                               octaveLayers, contrastThreshold,
                                               peakThreshold, edgeThreshold,
                                               sigma, octave);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
