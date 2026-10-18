#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace colite::features::cuda {

enum class CudaSiftMode {
    OpenCV,
    VLFeat
};

struct CudaSiftConfig {
    CudaSiftMode mode = CudaSiftMode::OpenCV;
    int firstOctave = -1;
    int octaveLayers = 3;
    int maxOctaves = -1;
    float sigma = 1.6f;
    float contrastThreshold = 0.04f;
    float peakThreshold = 3.4f;
    float edgeThreshold = 10.0f;
    int maxFeatures = 0; // Oriented features retained before descriptors; 0 disables the limit.
    int maxOrientations = 4;
};

struct CudaSiftTiming {
    float h2dMs = 0.0f;
    float computeMs = 0.0f;
    float d2hMs = 0.0f;
    float transferEndToEndMs = 0.0f;
    float endToEndMs = 0.0f;
    size_t peakDeviceBytes = 0;
};

struct CudaSiftKeypoint {
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;
    float angle = 0.0f;
    float response = 0.0f;
    int octave = 0;
    int layer = 0;
};

struct CudaSiftResult {
    std::vector<CudaSiftKeypoint> keypoints;
    std::vector<float> descriptors;
    CudaSiftTiming timing;
};

class CudaSift {
public:
    explicit CudaSift(const CudaSiftConfig& config, int device = 0);

    CudaSiftResult detectAndCompute(const float* image, int width, int height) const;
    CudaSiftResult detectAndCompute(const std::uint8_t* image,
                                    int width,
                                    int height) const;

    const CudaSiftConfig& config() const noexcept { return config_; }
    int device() const noexcept { return device_; }

private:
    CudaSiftConfig config_;
    int device_ = 0;
};

namespace detail {

const char* CompiledModeTag(CudaSiftMode mode);

}
}
