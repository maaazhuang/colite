#include "cuda_sift/cuda_sift_internal.cuh"

#include <cub/device/device_radix_sort.cuh>

#include <stdexcept>
#include <string>

namespace colite::features::cuda::detail {
namespace {

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename Keypoint>
__global__ void ExtractResponseKernel(const Keypoint* keypoints,
                                      float* responses,
                                      int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) responses[index] = keypoints[index].response;
}

template <typename Keypoint>
Keypoint* RetainBestDevice(const Keypoint* input,
                           int inputCount,
                           int maximumCount,
                           size_t& peakTemporaryBytes)
{
    peakTemporaryBytes = 0;
    if (input == nullptr || inputCount <= 0 || maximumCount <= 0 ||
        maximumCount >= inputCount) {
        return nullptr;
    }

    float* responseInput = nullptr;
    float* responseOutput = nullptr;
    Keypoint* sortedOutput = nullptr;
    Keypoint* limitedOutput = nullptr;
    void* cubTemporary = nullptr;
    size_t cubBytes = 0;
    const size_t responseBytes = static_cast<size_t>(inputCount) * sizeof(float);
    const size_t sortedOutputBytes = static_cast<size_t>(inputCount) * sizeof(Keypoint);
    const size_t limitedOutputBytes = static_cast<size_t>(maximumCount) * sizeof(Keypoint);
    auto cleanup = [&]() {
        if (sortedOutput) cudaFree(sortedOutput);
        if (cubTemporary) cudaFree(cubTemporary);
        if (responseOutput) cudaFree(responseOutput);
        if (responseInput) cudaFree(responseInput);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&responseInput), responseBytes),
                  "cudaMalloc Top-K response input");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&responseOutput), responseBytes),
                  "cudaMalloc Top-K response output");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&sortedOutput), sortedOutputBytes),
                  "cudaMalloc Top-K sorted output");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&limitedOutput), limitedOutputBytes),
                  "cudaMalloc Top-K limited output");
        constexpr int block = 256;
        ExtractResponseKernel<<<(inputCount + block - 1) / block, block>>>(
            input, responseInput, inputCount);
        CheckCuda(cudaGetLastError(), "Top-K response extraction launch");
        CheckCuda(cub::DeviceRadixSort::SortPairsDescending(
                      nullptr, cubBytes, responseInput, responseOutput,
                      input, sortedOutput, inputCount),
                  "CUB Top-K size query");
        CheckCuda(cudaMalloc(&cubTemporary, cubBytes),
                  "cudaMalloc CUB Top-K temporary");
        CheckCuda(cub::DeviceRadixSort::SortPairsDescending(
                      cubTemporary, cubBytes, responseInput, responseOutput,
                      input, sortedOutput, inputCount),
                  "CUB Top-K sort");
        CheckCuda(cudaMemcpyAsync(limitedOutput, sortedOutput, limitedOutputBytes,
                                  cudaMemcpyDeviceToDevice, nullptr),
                  "cudaMemcpyAsync Top-K compact output");
        peakTemporaryBytes = responseBytes * 2 + sortedOutputBytes +
                             limitedOutputBytes + cubBytes;
        cleanup();
        return limitedOutput;
    } catch (...) {
        cleanup();
        if (limitedOutput) cudaFree(limitedOutput);
        throw;
    }
}

}

PipelineOrientedKeypoint* RetainBestOrientedDevice(
    const PipelineOrientedKeypoint* input,
    int inputCount,
    int maximumCount,
    size_t& peakTemporaryBytes)
{
    return RetainBestDevice(input, inputCount, maximumCount, peakTemporaryBytes);
}

}
