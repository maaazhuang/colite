#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace colite::features::cuda::detail {
namespace {

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename Traits, typename Input>
__device__ float HorizontalSample(const Input* source, int width, int x)
{
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        float sourceX = (static_cast<float>(x) + 0.5f) * 0.5f - 0.5f;
        int x0 = static_cast<int>(floorf(sourceX));
        float fraction = sourceX - static_cast<float>(x0);
        if (x0 < 0) {
            x0 = 0;
            fraction = 0.0f;
        } else if (x0 >= width - 1) {
            x0 = width - 1;
            fraction = 0.0f;
        }
        const int x1 = min(x0 + 1, width - 1);
        return static_cast<float>(source[x0]) * (1.0f - fraction) +
               static_cast<float>(source[x1]) * fraction;
    } else {
        const int x0 = x >> 1;
        if ((x & 1) == 0) return static_cast<float>(source[x0]);
        const int x1 = min(x0 + 1, width - 1);
        return 0.5f * (static_cast<float>(source[x0]) +
                       static_cast<float>(source[x1]));
    }
}

template <typename Traits, typename Input>
__global__ void UpsampleHorizontal2xKernel(const Input* source,
                                           float* temporary,
                                           int width,
                                           int height)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int doubledWidth = width * 2;
    if (x >= doubledWidth || y >= height) return;

    temporary[static_cast<size_t>(y) * doubledWidth + x] =
        HorizontalSample<Traits>(source + static_cast<size_t>(y) * width, width, x);
}

template <typename Traits>
__device__ float VerticalSample(const float* temporary,
                                int width,
                                int height,
                                int x,
                                int y);

template <>
__device__ float VerticalSample<OpenCvModeTraits>(const float* temporary,
                                                  int width,
                                                  int height,
                                                  int x,
                                                  int y)
{
    float sourceY = (static_cast<float>(y) + 0.5f) * 0.5f - 0.5f;
    int y0 = static_cast<int>(floorf(sourceY));
    float fraction = sourceY - static_cast<float>(y0);
    if (y0 < 0) {
        y0 = 0;
        fraction = 0.0f;
    } else if (y0 >= height - 1) {
        y0 = height - 1;
        fraction = 0.0f;
    }
    const int y1 = min(y0 + 1, height - 1);
    const float a = temporary[static_cast<size_t>(y0) * width + x];
    const float b = temporary[static_cast<size_t>(y1) * width + x];
    return a * (1.0f - fraction) + b * fraction;
}

template <>
__device__ float VerticalSample<VlFeatModeTraits>(const float* temporary,
                                                  int width,
                                                  int height,
                                                  int x,
                                                  int y)
{
    const int y0 = y >> 1;
    const float a = temporary[static_cast<size_t>(y0) * width + x];
    if ((y & 1) == 0) return a;
    const int y1 = min(y0 + 1, height - 1);
    const float b = temporary[static_cast<size_t>(y1) * width + x];
    return 0.5f * (a + b);
}

template <typename Traits>
__global__ void UpsampleVertical2xKernel(const float* temporary,
                                         float* output,
                                         int doubledWidth,
                                         int sourceHeight)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int doubledHeight = sourceHeight * 2;
    if (x >= doubledWidth || y >= doubledHeight) return;

    output[static_cast<size_t>(y) * doubledWidth + x] =
        VerticalSample<Traits>(temporary, doubledWidth, sourceHeight, x, y);
}

}

template <typename Traits, typename Input>
void LaunchUpsample2xImpl(const Input* deviceSource,
                          float* deviceTemporary,
                          float* deviceOutput,
                          int width,
                          int height,
                          cudaStream_t stream)
{
    const int doubledWidth = width * 2;
    const int doubledHeight = height * 2;
    const dim3 block(32, 8);
    const dim3 horizontalGrid((doubledWidth + block.x - 1) / block.x,
                              (height + block.y - 1) / block.y);
    UpsampleHorizontal2xKernel<Traits><<<horizontalGrid, block, 0, stream>>>(
        deviceSource, deviceTemporary, width, height);
    CheckCuda(cudaGetLastError(), "UpsampleHorizontal2xKernel launch");

    const dim3 verticalGrid((doubledWidth + block.x - 1) / block.x,
                            (doubledHeight + block.y - 1) / block.y);
    UpsampleVertical2xKernel<Traits><<<verticalGrid, block, 0, stream>>>(
        deviceTemporary, deviceOutput, doubledWidth, height);
    CheckCuda(cudaGetLastError(), "UpsampleVertical2xKernel launch");
}

template <typename Traits>
void LaunchUpsample2x(const float* deviceSource,
                      float* deviceTemporary,
                      float* deviceOutput,
                      int width,
                      int height,
                      cudaStream_t stream)
{
    LaunchUpsample2xImpl<Traits>(deviceSource, deviceTemporary, deviceOutput,
                                 width, height, stream);
}

template <typename Traits>
void LaunchUpsample2x(const std::uint8_t* deviceSource,
                      float* deviceTemporary,
                      float* deviceOutput,
                      int width,
                      int height,
                      cudaStream_t stream)
{
    LaunchUpsample2xImpl<Traits>(deviceSource, deviceTemporary, deviceOutput,
                                 width, height, stream);
}

__global__ void ConvertToFloatKernel(const std::uint8_t* source,
                                     float* output,
                                     size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<float>(source[index]);
}

void LaunchConvertToFloat(const std::uint8_t* deviceSource,
                          float* deviceOutput,
                          size_t count,
                          cudaStream_t stream)
{
    constexpr int block = 256;
    ConvertToFloatKernel<<<static_cast<unsigned int>((count + block - 1) / block),
                           block, 0, stream>>>(deviceSource, deviceOutput, count);
    CheckCuda(cudaGetLastError(), "ConvertToFloatKernel launch");
}

template void LaunchUpsample2x<OpenCvModeTraits>(
    const float*, float*, float*, int, int, cudaStream_t);
template void LaunchUpsample2x<VlFeatModeTraits>(
    const float*, float*, float*, int, int, cudaStream_t);
template void LaunchUpsample2x<OpenCvModeTraits>(
    const std::uint8_t*, float*, float*, int, int, cudaStream_t);
template void LaunchUpsample2x<VlFeatModeTraits>(
    const std::uint8_t*, float*, float*, int, int, cudaStream_t);

namespace {

template <typename Traits, typename Input>
std::vector<float> BuildInitialImage(const Input* image, int width, int height)
{
    if (image == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("Initial image input must be non-null with positive dimensions");
    }

    const int doubledWidth = width * 2;
    const int doubledHeight = height * 2;
    const size_t inputBytes = static_cast<size_t>(width) * height * sizeof(Input);
    const size_t temporaryBytes = static_cast<size_t>(doubledWidth) * height * sizeof(float);
    const size_t outputBytes = static_cast<size_t>(doubledWidth) * doubledHeight * sizeof(float);

    Input* deviceInput = nullptr;
    float* deviceTemporary = nullptr;
    float* deviceOutput = nullptr;
    auto cleanup = [&]() {
        if (deviceOutput) cudaFree(deviceOutput);
        if (deviceTemporary) cudaFree(deviceTemporary);
        if (deviceInput) cudaFree(deviceInput);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), inputBytes),
                  "cudaMalloc initial input");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceTemporary), temporaryBytes),
                  "cudaMalloc initial temporary");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), outputBytes),
                  "cudaMalloc initial output");
        CheckCuda(cudaMemcpy(deviceInput, image, inputBytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy initial input");

        LaunchUpsample2x<Traits>(deviceInput, deviceTemporary, deviceOutput,
                                width, height);

        std::vector<float> output(static_cast<size_t>(doubledWidth) * doubledHeight);
        CheckCuda(cudaMemcpy(output.data(), deviceOutput, outputBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy initial output");
        cleanup();
        return output;
    } catch (...) {
        cleanup();
        throw;
    }
}

}

std::vector<float> BuildInitialImageForTest(CudaSiftMode mode,
                                            const float* image,
                                            int width,
                                            int height)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return BuildInitialImage<OpenCvModeTraits>(image, width, height);
    case CudaSiftMode::VLFeat:
        return BuildInitialImage<VlFeatModeTraits>(image, width, height);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

std::vector<float> BuildInitialImageForTest(CudaSiftMode mode,
                                            const std::uint8_t* image,
                                            int width,
                                            int height)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return BuildInitialImage<OpenCvModeTraits>(image, width, height);
    case CudaSiftMode::VLFeat:
        return BuildInitialImage<VlFeatModeTraits>(image, width, height);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
