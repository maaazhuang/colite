#include "cuda_sift/cuda_sift_internal.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace colite::features::cuda::detail {
namespace {

constexpr int kMaxKernelSize = 65;

struct GaussianKernelData {
    float coefficients[kMaxKernelSize]{};
    int radius = 0;
};

void CheckCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename Traits>
GaussianKernelData BuildGaussianKernel(float sigma)
{
    GaussianKernelData kernel;
    int radius;
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        radius = (static_cast<int>(std::lround(sigma * 8.0f + 1.0f)) | 1) / 2;
        double sum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double value = std::exp(-0.5 * static_cast<double>(i * i) /
                                          static_cast<double>(sigma * sigma));
            kernel.coefficients[i + radius] = static_cast<float>(value);
            sum += value;
        }
        for (int i = 0; i <= 2 * radius; ++i) {
            kernel.coefficients[i] = static_cast<float>(kernel.coefficients[i] / sum);
        }
    } else {
        radius = std::max(static_cast<int>(std::ceil(4.0f * sigma)), 1);
        float sum = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            const float d = static_cast<float>(i) / sigma;
            const float value = std::exp(-0.5f * d * d);
            kernel.coefficients[i + radius] = value;
            sum += value;
        }
        for (int i = 0; i <= 2 * radius; ++i) kernel.coefficients[i] /= sum;
    }
    if (2 * radius + 1 > kMaxKernelSize) {
        throw std::runtime_error("CUDA SIFT Gaussian kernel exceeds constant-memory capacity");
    }
    kernel.radius = radius;
    return kernel;
}

template <typename Traits>
__device__ int BorderIndex(int x, int length);

template <>
__device__ int BorderIndex<OpenCvModeTraits>(int x, int length)
{
    if (length <= 1) return 0;
    while (x < 0 || x >= length) {
        x = x < 0 ? -x : 2 * length - x - 2;
    }
    return x;
}

template <>
__device__ int BorderIndex<VlFeatModeTraits>(int x, int length)
{
    return max(0, min(x, length - 1));
}

template <typename Traits>
__global__ void GaussianHorizontalKernel(const float* source,
                                         float* output,
                                         const float* dogBase,
                                         float* dogOutput,
                                         GaussianKernelData kernel,
                                         int width,
                                         int height)
{
    const int radius = kernel.radius;
    extern __shared__ float tile[];
    const int tileStart = static_cast<int>(blockIdx.x * blockDim.x);
    const int y = static_cast<int>(blockIdx.y);
    const int tileElements = static_cast<int>(blockDim.x) + 2 * radius;
    const float* row = source + static_cast<size_t>(y) * width;

    for (int local = static_cast<int>(threadIdx.x);
         local < tileElements;
         local += static_cast<int>(blockDim.x)) {
        const int sourceX = BorderIndex<Traits>(tileStart + local - radius, width);
        tile[local] = row[sourceX];
    }
    __syncthreads();

    const int x = tileStart + static_cast<int>(threadIdx.x);
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        sum += tile[static_cast<int>(threadIdx.x) + k + radius] *
               kernel.coefficients[k + radius];
    }
    const size_t index = static_cast<size_t>(y) * width + x;
    output[index] = sum;
    if (dogOutput != nullptr) dogOutput[index] = sum - dogBase[index];
}

template <typename Traits>
__global__ void GaussianVerticalKernel(const float* source,
                                       float* output,
                                       const float* dogBase,
                                       float* dogOutput,
                                       GaussianKernelData kernel,
                                       int width,
                                       int height)
{
    const int radius = kernel.radius;
    extern __shared__ float tile[];
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tileStartY = static_cast<int>(blockIdx.y * blockDim.y);
    const int tileRows = static_cast<int>(blockDim.y) + 2 * radius;
    const int safeX = min(x, width - 1);

    for (int localY = static_cast<int>(threadIdx.y);
         localY < tileRows;
         localY += static_cast<int>(blockDim.y)) {
        const int sourceY = BorderIndex<Traits>(tileStartY + localY - radius, height);
        tile[static_cast<size_t>(localY) * blockDim.x + threadIdx.x] =
            source[static_cast<size_t>(sourceY) * width + safeX];
    }
    __syncthreads();

    const int y = tileStartY + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height) return;
    float sum = 0.0f;
    for (int k = -radius; k <= radius; ++k) {
        const int localY = static_cast<int>(threadIdx.y) + k + radius;
        sum += tile[static_cast<size_t>(localY) * blockDim.x + threadIdx.x] *
               kernel.coefficients[k + radius];
    }
    const size_t index = static_cast<size_t>(y) * width + x;
    output[index] = sum;
    if (dogOutput != nullptr) dogOutput[index] = sum - dogBase[index];
}

template <typename Traits>
__device__ float UpsampleHorizontalU8(const std::uint8_t* source,
                                      int width,
                                      int outputX)
{
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        float sourceX = (static_cast<float>(outputX) + 0.5f) * 0.5f - 0.5f;
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
        const int x0 = outputX >> 1;
        if ((outputX & 1) == 0) return static_cast<float>(source[x0]);
        const int x1 = min(x0 + 1, width - 1);
        return 0.5f * (static_cast<float>(source[x0]) +
                       static_cast<float>(source[x1]));
    }
}

template <typename Traits>
__global__ void FusedUpsampleGaussianHorizontalU8Kernel(
    const std::uint8_t* source,
    float* temporary,
    GaussianKernelData kernel,
    int width,
    int height)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int outputWidth = width * 2;
    if (x >= outputWidth || y >= height) return;
    const std::uint8_t* row = source + static_cast<size_t>(y) * width;
    float sum = 0.0f;
    for (int k = -kernel.radius; k <= kernel.radius; ++k) {
        const int sampleX = BorderIndex<Traits>(x + k, outputWidth);
        sum += UpsampleHorizontalU8<Traits>(row, width, sampleX) *
               kernel.coefficients[k + kernel.radius];
    }
    temporary[static_cast<size_t>(y) * outputWidth + x] = sum;
}

template <typename Traits>
__device__ float UpsampleVerticalFloat(const float* source,
                                       int width,
                                       int height,
                                       int x,
                                       int outputY)
{
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        float sourceY = (static_cast<float>(outputY) + 0.5f) * 0.5f - 0.5f;
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
        const float a = source[static_cast<size_t>(y0) * width + x];
        const float b = source[static_cast<size_t>(y1) * width + x];
        return a * (1.0f - fraction) + b * fraction;
    } else {
        const int y0 = outputY >> 1;
        const float a = source[static_cast<size_t>(y0) * width + x];
        if ((outputY & 1) == 0) return a;
        const int y1 = min(y0 + 1, height - 1);
        const float b = source[static_cast<size_t>(y1) * width + x];
        return 0.5f * (a + b);
    }
}

template <typename Traits>
__global__ void FusedUpsampleGaussianVerticalKernel(
    const float* temporary,
    float* output,
    GaussianKernelData kernel,
    int outputWidth,
    int sourceHeight)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int outputHeight = sourceHeight * 2;
    if (x >= outputWidth || y >= outputHeight) return;
    float sum = 0.0f;
    for (int k = -kernel.radius; k <= kernel.radius; ++k) {
        const int sampleY = BorderIndex<Traits>(y + k, outputHeight);
        sum += UpsampleVerticalFloat<Traits>(
                   temporary, outputWidth, sourceHeight, x, sampleY) *
               kernel.coefficients[k + kernel.radius];
    }
    output[static_cast<size_t>(y) * outputWidth + x] = sum;
}

template <typename Traits>
void FusedUpsampleGaussianU8(const std::uint8_t* source,
                             float* temporary,
                             float* output,
                             int width,
                             int height,
                             float sigma)
{
    const GaussianKernelData kernel = BuildGaussianKernel<Traits>(sigma);
    const int outputWidth = width * 2;
    const int outputHeight = height * 2;
    const dim3 block(32, 8);
    const dim3 horizontalGrid((outputWidth + block.x - 1) / block.x,
                              (height + block.y - 1) / block.y);
    FusedUpsampleGaussianHorizontalU8Kernel<Traits><<<horizontalGrid, block>>>(
        source, temporary, kernel, width, height);
    CheckCuda(cudaGetLastError(), "fused U8 upsample Gaussian horizontal launch");
    const dim3 verticalGrid((outputWidth + block.x - 1) / block.x,
                            (outputHeight + block.y - 1) / block.y);
    FusedUpsampleGaussianVerticalKernel<Traits><<<verticalGrid, block>>>(
        temporary, output, kernel, outputWidth, height);
    CheckCuda(cudaGetLastError(), "fused U8 upsample Gaussian vertical launch");
}

template <typename Traits>
void GaussianBlur(float* output,
                  float* temporary,
                  const float* input,
                  const float* dogBase,
                  float* dogOutput,
                  int width,
                  int height,
                  float sigma)
{
    const GaussianKernelData kernel = BuildGaussianKernel<Traits>(sigma);
    const int radius = kernel.radius;

    const dim3 horizontalBlock(256, 1);
    const dim3 horizontalGrid((width + horizontalBlock.x - 1) / horizontalBlock.x,
                              height);
    const size_t horizontalShared =
        (static_cast<size_t>(horizontalBlock.x) + 2 * radius) * sizeof(float);
    const dim3 verticalBlock(32, 8);
    const dim3 verticalGrid((width + verticalBlock.x - 1) / verticalBlock.x,
                            (height + verticalBlock.y - 1) / verticalBlock.y);
    const size_t verticalShared =
        static_cast<size_t>(verticalBlock.x) * (verticalBlock.y + 2 * radius) * sizeof(float);
    if constexpr (std::is_same_v<Traits, OpenCvModeTraits>) {
        GaussianHorizontalKernel<Traits><<<horizontalGrid, horizontalBlock, horizontalShared>>>(
            input, temporary, nullptr, nullptr, kernel, width, height);
        CheckCuda(cudaGetLastError(), "OpenCV Gaussian horizontal launch");
        GaussianVerticalKernel<Traits><<<verticalGrid, verticalBlock, verticalShared>>>(
            temporary, output, dogBase, dogOutput, kernel, width, height);
        CheckCuda(cudaGetLastError(), "OpenCV Gaussian vertical launch");
    } else {
        GaussianVerticalKernel<Traits><<<verticalGrid, verticalBlock, verticalShared>>>(
            input, temporary, nullptr, nullptr, kernel, width, height);
        CheckCuda(cudaGetLastError(), "VLFeat Gaussian vertical launch");
        GaussianHorizontalKernel<Traits><<<horizontalGrid, horizontalBlock, horizontalShared>>>(
            temporary, output, dogBase, dogOutput, kernel, width, height);
        CheckCuda(cudaGetLastError(), "VLFeat Gaussian horizontal launch");
    }
}

template <typename Traits, typename Input>
PyramidSnapshot BuildFirstOctave(const Input* image,
                                 int width,
                                 int height,
                                 int octaveLayers,
                                 float sigma)
{
    if (image == nullptr || width <= 0 || height <= 0 || octaveLayers <= 0 || sigma <= 0.0f) {
        throw std::invalid_argument("Invalid CUDA SIFT first-octave parameters");
    }

    PyramidSnapshot snapshot;
    snapshot.width = width * 2;
    snapshot.height = height * 2;
    snapshot.gaussianLayers = octaveLayers + 3;
    snapshot.dogLayers = octaveLayers + 2;

    const size_t sourceCount = static_cast<size_t>(width) * height;
    const size_t layerCount = static_cast<size_t>(snapshot.width) * snapshot.height;
    const size_t sourceBytes = sourceCount * sizeof(Input);
    const size_t layerBytes = layerCount * sizeof(float);

    Input* deviceInput = nullptr;
    float* deviceUpsampleTemporary = nullptr;
    float* deviceCurrent = nullptr;
    float* deviceNext = nullptr;
    float* deviceTemporary = nullptr;
    float* deviceDog = nullptr;
    auto cleanup = [&]() {
        if (deviceDog) cudaFree(deviceDog);
        if (deviceTemporary) cudaFree(deviceTemporary);
        if (deviceNext) cudaFree(deviceNext);
        if (deviceCurrent) cudaFree(deviceCurrent);
        if (deviceUpsampleTemporary) cudaFree(deviceUpsampleTemporary);
        if (deviceInput) cudaFree(deviceInput);
    };

    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInput), sourceBytes), "cudaMalloc pyramid input");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceUpsampleTemporary),
                             static_cast<size_t>(snapshot.width) * height * sizeof(float)),
                  "cudaMalloc pyramid upsample temporary");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceCurrent), layerBytes), "cudaMalloc pyramid current");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceNext), layerBytes), "cudaMalloc pyramid next");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceTemporary), layerBytes), "cudaMalloc pyramid temporary");
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDog), layerBytes), "cudaMalloc pyramid dog");
        CheckCuda(cudaMemcpy(deviceInput, image, sourceBytes, cudaMemcpyHostToDevice), "cudaMemcpy pyramid input");

        const float initialSigma = std::sqrt(sigma * sigma - 1.0f);
        if constexpr (std::is_same_v<Input, std::uint8_t>) {
            FusedUpsampleGaussianU8<Traits>(
                deviceInput, deviceUpsampleTemporary, deviceCurrent,
                width, height, initialSigma);
        } else {
            LaunchUpsample2x<Traits>(deviceInput, deviceUpsampleTemporary,
                                    deviceCurrent, width, height);
            GaussianBlur<Traits>(deviceNext, deviceTemporary, deviceCurrent,
                                 nullptr, nullptr,
                                 snapshot.width, snapshot.height, initialSigma);
            std::swap(deviceCurrent, deviceNext);
        }

        snapshot.gaussian.resize(layerCount * snapshot.gaussianLayers);
        snapshot.dog.resize(layerCount * snapshot.dogLayers);
        CheckCuda(cudaMemcpy(snapshot.gaussian.data(), deviceCurrent, layerBytes,
                             cudaMemcpyDeviceToHost), "cudaMemcpy Gaussian layer 0");

        const float scaleStep = std::pow(2.0f, 1.0f / static_cast<float>(octaveLayers));
        for (int layer = 1; layer < snapshot.gaussianLayers; ++layer) {
            const float previousSigma = std::pow(scaleStep, static_cast<float>(layer - 1)) * sigma;
            const float totalSigma = previousSigma * scaleStep;
            const float incrementalSigma = std::sqrt(totalSigma * totalSigma -
                                                     previousSigma * previousSigma);
            GaussianBlur<Traits>(deviceNext, deviceTemporary, deviceCurrent,
                                 deviceCurrent, deviceDog,
                                 snapshot.width, snapshot.height, incrementalSigma);

            CheckCuda(cudaMemcpy(snapshot.dog.data() + layerCount * (layer - 1),
                                 deviceDog, layerBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy DoG layer");
            CheckCuda(cudaMemcpy(snapshot.gaussian.data() + layerCount * layer,
                                 deviceNext, layerBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy Gaussian layer");
            std::swap(deviceCurrent, deviceNext);
        }
        cleanup();
        return snapshot;
    } catch (...) {
        cleanup();
        throw;
    }
}

}

namespace {

__global__ void Downsample2xKernel(const float* source,
                                   float* output,
                                   int sourceWidth,
                                   int outputWidth,
                                   int outputHeight)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= outputWidth || y >= outputHeight) return;
    output[static_cast<size_t>(y) * outputWidth + x] =
        source[static_cast<size_t>(2 * y) * sourceWidth + 2 * x];
}

template <typename Traits>
void BuildRemainingLayers(DevicePyramidOctave& octave,
                          int octaveLayers,
                          float sigma,
                          float* temporary)
{
    const size_t layerCount = static_cast<size_t>(octave.width) * octave.height;
    const float scaleStep = powf(2.0f, 1.0f / octaveLayers);
    for (int layer = 1; layer < octave.gaussianLayers; ++layer) {
        const float previousSigma = powf(scaleStep, static_cast<float>(layer - 1)) * sigma;
        const float totalSigma = previousSigma * scaleStep;
        const float incrementalSigma = sqrtf(totalSigma * totalSigma -
                                              previousSigma * previousSigma);
        float* output = octave.gaussian + layerCount * layer;
        const float* input = octave.gaussian + layerCount * (layer - 1);
        float* dog = octave.dog + layerCount * (layer - 1);
        GaussianBlur<Traits>(output, temporary, input, input, dog,
                             octave.width, octave.height, incrementalSigma);
    }
}

DevicePyramidOctave AllocateOctave(int width, int height, int octaveLayers)
{
    DevicePyramidOctave octave;
    octave.width = width;
    octave.height = height;
    octave.gaussianLayers = octaveLayers + 3;
    octave.dogLayers = octaveLayers + 2;
    const size_t layerBytes = static_cast<size_t>(width) * height * sizeof(float);
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&octave.gaussian),
                         layerBytes * octave.gaussianLayers),
              "cudaMalloc device Gaussian octave");
    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&octave.dog),
                             layerBytes * octave.dogLayers),
                  "cudaMalloc device DoG octave");
    } catch (...) {
        cudaFree(octave.gaussian);
        throw;
    }
    octave.allocatedBytes = layerBytes *
        static_cast<size_t>(octave.gaussianLayers + octave.dogLayers);
    return octave;
}

}

template <typename Traits, typename Input>
DevicePyramidOctave BuildInitialOctaveDeviceImpl(const Input* deviceInput,
                                                  int width,
                                                  int height,
                                                  int octaveLayers,
                                                  float sigma,
                                                  int firstOctave)
{
    if (deviceInput == nullptr || width <= 0 || height <= 0 ||
        octaveLayers <= 0 || sigma <= 0.0f ||
        (firstOctave != -1 && firstOctave != 0)) {
        throw std::invalid_argument("Invalid initial device octave parameters");
    }
    const int outputWidth = firstOctave < 0 ? width * 2 : width;
    const int outputHeight = firstOctave < 0 ? height * 2 : height;
    DevicePyramidOctave octave = AllocateOctave(outputWidth, outputHeight, octaveLayers);
    const size_t layerBytes = static_cast<size_t>(outputWidth) * outputHeight * sizeof(float);
    float* temporary = nullptr;
    float* upsampleBase = nullptr;
    float* upsampleTemporary = nullptr;
    float* convertedBase = nullptr;
    auto cleanup = [&]() {
        if (convertedBase) cudaFree(convertedBase);
        if (upsampleTemporary) cudaFree(upsampleTemporary);
        if (upsampleBase) cudaFree(upsampleBase);
        if (temporary) cudaFree(temporary);
    };
    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&temporary), layerBytes),
                  "cudaMalloc octave blur temporary");
        const float* base = nullptr;
        bool initialLayerReady = false;
        if constexpr (std::is_same_v<Input, float>) base = deviceInput;
        if (firstOctave < 0) {
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&upsampleTemporary),
                                 static_cast<size_t>(outputWidth) * height * sizeof(float)),
                      "cudaMalloc octave upsample temporary");
            if constexpr (std::is_same_v<Input, std::uint8_t>) {
                const float initialSigma = sqrtf(fmaxf(sigma * sigma - 1.0f, 0.01f));
                FusedUpsampleGaussianU8<Traits>(
                    deviceInput, upsampleTemporary, octave.gaussian,
                    width, height, initialSigma);
                initialLayerReady = true;
            } else {
                CheckCuda(cudaMalloc(reinterpret_cast<void**>(&upsampleBase), layerBytes),
                          "cudaMalloc octave upsample base");
                LaunchUpsample2x<Traits>(deviceInput, upsampleTemporary, upsampleBase,
                                        width, height);
                base = upsampleBase;
            }
        } else if constexpr (std::is_same_v<Input, std::uint8_t>) {
            CheckCuda(cudaMalloc(reinterpret_cast<void**>(&convertedBase), layerBytes),
                      "cudaMalloc octave converted U8 base");
            LaunchConvertToFloat(deviceInput, convertedBase,
                                 static_cast<size_t>(width) * height);
            base = convertedBase;
        }
        const float assumedSigma = firstOctave < 0 ? 1.0f : 0.5f;
        const float initialSigma = sqrtf(fmaxf(sigma * sigma -
                                               assumedSigma * assumedSigma, 0.01f));
        if (!initialLayerReady) {
            GaussianBlur<Traits>(octave.gaussian, temporary, base, nullptr, nullptr,
                                 outputWidth, outputHeight, initialSigma);
        }
        BuildRemainingLayers<Traits>(octave, octaveLayers, sigma,
                                     temporary);
        cleanup();
        return octave;
    } catch (...) {
        cleanup();
        ReleaseDevicePyramid(octave);
        throw;
    }
}

template <typename Traits>
DevicePyramidOctave BuildInitialOctaveDevice(const float* deviceInput,
                                              int width,
                                              int height,
                                              int octaveLayers,
                                              float sigma,
                                              int firstOctave)
{
    return BuildInitialOctaveDeviceImpl<Traits>(deviceInput, width, height,
                                                 octaveLayers, sigma, firstOctave);
}

template <typename Traits>
DevicePyramidOctave BuildInitialOctaveDevice(const std::uint8_t* deviceInput,
                                              int width,
                                              int height,
                                              int octaveLayers,
                                              float sigma,
                                              int firstOctave)
{
    return BuildInitialOctaveDeviceImpl<Traits>(deviceInput, width, height,
                                                  octaveLayers, sigma, firstOctave);
}

template <typename Traits>
DevicePyramidOctave BuildNextOctaveDevice(const DevicePyramidOctave& previous,
                                           int octaveLayers,
                                           float sigma)
{
    if (previous.gaussian == nullptr || previous.width < 2 || previous.height < 2 ||
        octaveLayers <= 0 || octaveLayers >= previous.gaussianLayers) {
        throw std::invalid_argument("Invalid next device octave parameters");
    }
    const int width = previous.width / 2;
    const int height = previous.height / 2;
    DevicePyramidOctave octave = AllocateOctave(width, height, octaveLayers);
    const size_t layerBytes = static_cast<size_t>(width) * height * sizeof(float);
    float* temporary = nullptr;
    try {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&temporary), layerBytes),
                  "cudaMalloc next octave temporary");
        const size_t previousLayerCount =
            static_cast<size_t>(previous.width) * previous.height;
        const float* source = previous.gaussian + previousLayerCount * octaveLayers;
        const dim3 block(32, 8);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        Downsample2xKernel<<<grid, block>>>(source, octave.gaussian,
                                           previous.width, width, height);
        CheckCuda(cudaGetLastError(), "Downsample2xKernel launch");
        BuildRemainingLayers<Traits>(octave, octaveLayers, sigma,
                                     temporary);
        cudaFree(temporary);
        return octave;
    } catch (...) {
        if (temporary) cudaFree(temporary);
        ReleaseDevicePyramid(octave);
        throw;
    }
}

void ReleaseDevicePyramid(DevicePyramidOctave& octave) noexcept
{
    if (octave.dog) cudaFree(octave.dog);
    if (octave.gaussian) cudaFree(octave.gaussian);
    octave = {};
}

template DevicePyramidOctave BuildInitialOctaveDevice<OpenCvModeTraits>(
    const float*, int, int, int, float, int);
template DevicePyramidOctave BuildInitialOctaveDevice<VlFeatModeTraits>(
    const float*, int, int, int, float, int);
template DevicePyramidOctave BuildInitialOctaveDevice<OpenCvModeTraits>(
    const std::uint8_t*, int, int, int, float, int);
template DevicePyramidOctave BuildInitialOctaveDevice<VlFeatModeTraits>(
    const std::uint8_t*, int, int, int, float, int);
template DevicePyramidOctave BuildNextOctaveDevice<OpenCvModeTraits>(
    const DevicePyramidOctave&, int, float);
template DevicePyramidOctave BuildNextOctaveDevice<VlFeatModeTraits>(
    const DevicePyramidOctave&, int, float);

PyramidSnapshot BuildFirstOctaveForTest(CudaSiftMode mode,
                                        const float* image,
                                        int width,
                                        int height,
                                        int octaveLayers,
                                        float sigma)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return BuildFirstOctave<OpenCvModeTraits>(image, width, height, octaveLayers, sigma);
    case CudaSiftMode::VLFeat:
        return BuildFirstOctave<VlFeatModeTraits>(image, width, height, octaveLayers, sigma);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

PyramidSnapshot BuildFirstOctaveForTest(CudaSiftMode mode,
                                        const std::uint8_t* image,
                                        int width,
                                        int height,
                                        int octaveLayers,
                                        float sigma)
{
    switch (mode) {
    case CudaSiftMode::OpenCV:
        return BuildFirstOctave<OpenCvModeTraits>(
            image, width, height, octaveLayers, sigma);
    case CudaSiftMode::VLFeat:
        return BuildFirstOctave<VlFeatModeTraits>(
            image, width, height, octaveLayers, sigma);
    }
    throw std::invalid_argument("Unknown CUDA SIFT mode");
}

}
