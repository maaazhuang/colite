#include "colite/mvs/patch_match_cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace colite::mvs {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxCudaSourceImages = 32;
constexpr int kPoseKOffset = 0;
constexpr int kPoseROffset = 4;
constexpr int kPoseTOffset = 13;
constexpr int kPoseCOffset = 16;
constexpr int kPosePOffset = 19;
constexpr int kPoseInvPOffset = 31;
constexpr int kPoseWeightOffset = 43;
constexpr int kPoseSelectionProbabilityOffset = 44;
constexpr int kPoseSweepCosOffset = 45;
constexpr int kPoseSweepSinOffset = 46;
constexpr int kNumPoseParams = 47;
constexpr int kNumPoseVariants = 3;

struct DeviceCamera {
  float fx = 1.0F;
  float fy = 1.0F;
  float cx = 0.0F;
  float cy = 0.0F;
};

struct DeviceSourceImage {
  int width = 0;
  int height = 0;
};

struct DeviceCostOptions {
  int window_radius = 2;
  int window_step = 1;
  bool bilateral_weighting = false;
  float sigma_spatial = -1.0F;
  float sigma_color = 0.2F;
  float invalid_cost = 1.0F;
  float min_variance = 1e-8F;
  int min_num_valid_sources = 1;
  int max_num_aggregated_costs = 3;
  float geometric_consistency_weight = 0.0F;
  float geometric_max_reprojection_error = 3.0F;
  float geometric_max_relative_depth_error = 0.01F;
  bool geometric_reject_occluded_candidates = true;
  bool geometric_reject_depth_conflicts = true;
  float source_weight_strength = 0.25F;
  float incident_angle_weight = 0.15F;
};

struct DevicePatchMatchOptions {
  int width = 0;
  int height = 0;
  int max_source_width = 0;
  int max_source_height = 0;
  int num_sources = 0;
  int num_pose_variants = 1;
  float depth_min = 0.0F;
  float depth_max = 0.0F;
  DeviceCamera reference_camera;
  DeviceCostOptions cost;
  bool dynamic_selection_probability = false;
  bool directional_hmm = false;
  int num_source_samples = 0;
  float ncc_sigma = 0.6F;
  float previous_selection_probability_weight = 0.5F;
  float min_selection_probability = 0.05F;
  float max_selection_probability = 0.999F;
};

struct DeviceCounters {
  unsigned long long candidate_evaluations = 0ULL;
  unsigned long long plane_propagation_candidates = 0ULL;
  unsigned long long perturbation_candidate_evaluations = 0ULL;
  unsigned long long propagation_updates = 0ULL;
  unsigned long long perturbation_updates = 0ULL;
  unsigned long long source_selection_tests = 0ULL;
  unsigned long long source_selection_acceptances = 0ULL;
  unsigned long long source_selection_rejections = 0ULL;
  unsigned long long source_selection_probability_updates = 0ULL;
  double total_source_selection_probability = 0.0;
  unsigned long long geometric_candidate_rejections = 0ULL;
  unsigned long long geometric_candidate_acceptances = 0ULL;
  double total_abs_depth_update = 0.0;
  double total_normal_angle_degrees = 0.0;
  double max_normal_angle_degrees = 0.0;
};

struct DeviceOutputPixel {
  float depth = 0.0F;
  float normal_x = 0.0F;
  float normal_y = 0.0F;
  float normal_z = 0.0F;
  float cost = 0.0F;
};

struct CachedCudaDeviceCount {
  cudaError_t status = cudaSuccess;
  int count = 0;
  bool queried_runtime = false;
};

CachedCudaDeviceCount GetCachedCudaDeviceCount() {
  static std::once_flag query_once;
  static cudaError_t cached_status = cudaSuccess;
  static int cached_count = 0;
  bool queried_runtime = false;
  std::call_once(query_once, [&]() {
    queried_runtime = true;
    cached_status = cudaGetDeviceCount(&cached_count);
  });
  return {cached_status, cached_count, queried_runtime};
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t size) { Allocate(size); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { MoveFrom(&other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      MoveFrom(&other);
    }
    return *this;
  }
  ~DeviceBuffer() { Release(); }

  void Allocate(std::size_t size) {
    Resize(size);
  }

  bool Resize(std::size_t size) {
    bool allocated = false;
    if (size > capacity_) {
      Release();
      capacity_ = size;
      if (capacity_ > 0) {
        CudaCheck(cudaMalloc(&ptr_, sizeof(T) * capacity_));
      }
      allocated = true;
    }
    size_ = size;
    return allocated;
  }

  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }

  void AllocateExact(std::size_t size) {
    Release();
    size_ = size;
    capacity_ = size;
    if (capacity_ > 0) {
      CudaCheck(cudaMalloc(&ptr_, sizeof(T) * capacity_));
    }
  }

  void CopyFromHost(const std::vector<T>& values) {
    if (values.size() != size_) {
      throw std::runtime_error("CUDA buffer copy size mismatch.");
    }
    if (size_ > 0) {
      CudaCheck(cudaMemcpy(ptr_, values.data(), sizeof(T) * size_,
                           cudaMemcpyHostToDevice));
    }
  }

  void CopyFromHostAsync(const T* host_values, cudaStream_t stream) {
    if (size_ > 0) {
      CudaCheck(cudaMemcpyAsync(ptr_, host_values, sizeof(T) * size_,
                                cudaMemcpyHostToDevice, stream));
    }
  }

  void CopyFromHostValue(const T& value) {
    if (size_ != 1) {
      throw std::runtime_error("CUDA scalar copy size mismatch.");
    }
    CudaCheck(cudaMemcpy(ptr_, &value, sizeof(T), cudaMemcpyHostToDevice));
  }

  std::vector<T> CopyToHost() const {
    std::vector<T> values(size_);
    if (size_ > 0) {
      CudaCheck(cudaMemcpy(values.data(), ptr_, sizeof(T) * size_,
                           cudaMemcpyDeviceToHost));
    }
    return values;
  }

  void CopyToHostAsync(T* host_values, cudaStream_t stream) const {
    if (size_ > 0) {
      CudaCheck(cudaMemcpyAsync(host_values, ptr_, sizeof(T) * size_,
                                cudaMemcpyDeviceToHost, stream));
    }
  }

  T CopyScalarToHost() const {
    if (size_ != 1) {
      throw std::runtime_error("CUDA scalar copy size mismatch.");
    }
    T value{};
    CudaCheck(cudaMemcpy(&value, ptr_, sizeof(T), cudaMemcpyDeviceToHost));
    return value;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

 private:
  static void CudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

  void Release() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  void MoveFrom(DeviceBuffer* other) {
    ptr_ = other->ptr_;
    size_ = other->size_;
    capacity_ = other->capacity_;
    other->ptr_ = nullptr;
    other->size_ = 0;
    other->capacity_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

void CudaCheck(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  ~PinnedHostBuffer() { Release(); }

  bool Resize(std::size_t size) {
    bool allocated = false;
    if (size > capacity_) {
      Release();
      capacity_ = size;
      if (capacity_ > 0) {
        CudaCheck(cudaMallocHost(&ptr_, sizeof(T) * capacity_));
      }
      allocated = true;
    }
    size_ = size;
    return allocated;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  std::size_t size() const { return size_; }
  std::size_t CapacityBytes() const { return sizeof(T) * capacity_; }

  std::vector<T> ToVector() const {
    if (size_ == 0) {
      return {};
    }
    return std::vector<T>(ptr_, ptr_ + size_);
  }

 private:
  void Release() {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
    }
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point start,
                           const SteadyClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

class CudaEventTimer {
 public:
  CudaEventTimer() {
    CudaCheck(cudaEventCreate(&start_));
    CudaCheck(cudaEventCreate(&stop_));
  }
  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;
  CudaEventTimer(CudaEventTimer&& other) noexcept { MoveFrom(&other); }
  ~CudaEventTimer() {
    Release();
  }

  void Start(cudaStream_t stream = 0) { CudaCheck(cudaEventRecord(start_, stream)); }

  void Stop(cudaStream_t stream = 0) { CudaCheck(cudaEventRecord(stop_, stream)); }

  double ElapsedMilliseconds() {
    CudaCheck(cudaEventSynchronize(stop_));
    float elapsed_ms = 0.0F;
    CudaCheck(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return static_cast<double>(elapsed_ms);
  }

  double StopMilliseconds(cudaStream_t stream = 0) {
    Stop(stream);
    return ElapsedMilliseconds();
  }

 private:
  void Release() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
    start_ = nullptr;
    stop_ = nullptr;
  }

  void MoveFrom(CudaEventTimer* other) {
    start_ = other->start_;
    stop_ = other->stop_;
    other->start_ = nullptr;
    other->stop_ = nullptr;
  }

  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

class FloatLayeredTexture {
 public:
  FloatLayeredTexture() = default;
  FloatLayeredTexture(int width,
                      int height,
                      int depth,
                      const std::vector<float>& host_values,
                      cudaTextureFilterMode filter_mode =
                          cudaFilterModeLinear) {
    Create(width, height, depth, host_values, filter_mode);
  }
  FloatLayeredTexture(const FloatLayeredTexture&) = delete;
  FloatLayeredTexture& operator=(const FloatLayeredTexture&) = delete;
  FloatLayeredTexture(FloatLayeredTexture&& other) noexcept {
    MoveFrom(&other);
  }
  FloatLayeredTexture& operator=(FloatLayeredTexture&& other) = delete;
  ~FloatLayeredTexture() { Release(); }

  bool Create(int width,
              int height,
              int depth,
              const std::vector<float>& host_values,
              cudaTextureFilterMode filter_mode = cudaFilterModeLinear) {
    const std::size_t expected_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        static_cast<std::size_t>(depth);
    if (host_values.size() != expected_size) {
      throw std::invalid_argument("CUDA texture host data size mismatch.");
    }
    return CreateFromHostPointer(width, height, depth, host_values.data(),
                                 filter_mode, 0, false);
  }

  bool CreateAsync(int width,
                   int height,
                   int depth,
                   const float* host_values,
                   cudaStream_t stream,
                   cudaTextureFilterMode filter_mode = cudaFilterModeLinear) {
    return CreateFromHostPointer(width, height, depth, host_values, filter_mode,
                                 stream, true);
  }

  cudaTextureObject_t get() const { return texture_; }

 private:
  bool CreateFromHostPointer(int width,
                             int height,
                             int depth,
                             const float* host_values,
                             cudaTextureFilterMode filter_mode,
                             cudaStream_t stream,
                             bool async_copy) {
    if (width <= 0 || height <= 0 || depth <= 0) {
      throw std::invalid_argument("CUDA texture dimensions must be positive.");
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        static_cast<std::size_t>(depth);
    if (expected_size > 0 && host_values == nullptr) {
      throw std::invalid_argument("CUDA texture host data must not be null.");
    }
    const bool needs_allocation =
        array_ == nullptr || texture_ == 0 || width_ != width ||
        height_ != height || depth_ != depth || filter_mode_ != filter_mode;
    if (needs_allocation) {
      Release();
      width_ = width;
      height_ = height;
      depth_ = depth;
      filter_mode_ = filter_mode;

      const cudaChannelFormatDesc channel_desc =
          cudaCreateChannelDesc<float>();
      const cudaExtent extent = make_cudaExtent(
          static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
          static_cast<std::size_t>(depth_));
      CudaCheck(cudaMalloc3DArray(&array_, &channel_desc, extent,
                                  cudaArrayLayered));

      cudaResourceDesc resource_desc;
      std::memset(&resource_desc, 0, sizeof(resource_desc));
      resource_desc.resType = cudaResourceTypeArray;
      resource_desc.res.array.array = array_;

      cudaTextureDesc texture_desc;
      std::memset(&texture_desc, 0, sizeof(texture_desc));
      texture_desc.addressMode[0] = cudaAddressModeBorder;
      texture_desc.addressMode[1] = cudaAddressModeBorder;
      texture_desc.addressMode[2] = cudaAddressModeBorder;
      texture_desc.filterMode = filter_mode_;
      texture_desc.readMode = cudaReadModeElementType;
      texture_desc.normalizedCoords = false;
      CudaCheck(cudaCreateTextureObject(&texture_, &resource_desc,
                                        &texture_desc, nullptr));
    }

    cudaMemcpy3DParms copy_params;
    std::memset(&copy_params, 0, sizeof(copy_params));
    copy_params.extent = make_cudaExtent(
        static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
        static_cast<std::size_t>(depth_));
    copy_params.kind = cudaMemcpyHostToDevice;
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(host_values),
        static_cast<std::size_t>(width_) * sizeof(float),
        static_cast<std::size_t>(width_), static_cast<std::size_t>(height_));
    copy_params.dstArray = array_;
    if (async_copy) {
      CudaCheck(cudaMemcpy3DAsync(&copy_params, stream));
    } else {
      CudaCheck(cudaMemcpy3D(&copy_params));
    }
    return needs_allocation;
  }

  void Release() {
    if (texture_ != 0) {
      cudaDestroyTextureObject(texture_);
    }
    texture_ = 0;
    if (array_ != nullptr) {
      cudaFreeArray(array_);
    }
    array_ = nullptr;
    width_ = 0;
    height_ = 0;
    depth_ = 0;
    filter_mode_ = cudaFilterModeLinear;
  }

  void MoveFrom(FloatLayeredTexture* other) {
    array_ = other->array_;
    texture_ = other->texture_;
    width_ = other->width_;
    height_ = other->height_;
    depth_ = other->depth_;
    filter_mode_ = other->filter_mode_;
    other->array_ = nullptr;
    other->texture_ = 0;
    other->width_ = 0;
    other->height_ = 0;
    other->depth_ = 0;
    other->filter_mode_ = cudaFilterModeLinear;
  }

  cudaArray_t array_ = nullptr;
  cudaTextureObject_t texture_ = 0;
  int width_ = 0;
  int height_ = 0;
  int depth_ = 0;
  cudaTextureFilterMode filter_mode_ = cudaFilterModeLinear;
};

struct CudaPatchMatchWorkspaceStorage {
  FloatLayeredTexture reference_texture;
  FloatLayeredTexture source_images_texture;
  FloatLayeredTexture poses_texture;
  FloatLayeredTexture source_depth_maps_texture;
  FloatLayeredTexture selection_probability_texture;
  DeviceBuffer<DeviceSourceImage> source_images_info;
  DeviceBuffer<float> depth_map;
  DeviceBuffer<float> normal_map;
  DeviceBuffer<float> cost_map;
  DeviceBuffer<float> dynamic_selection_probability_map;
  DeviceBuffer<float> source_selection_emission_map;
  DeviceBuffer<float> source_selection_backward_message_map;
  DeviceBuffer<DeviceCounters> counters;
  DeviceBuffer<DeviceCounters> iteration_counters;
  DeviceBuffer<float> perturbation_radius_state;
  DeviceBuffer<float> iteration_depth_radii;
  DeviceBuffer<float> iteration_normal_radii;
  DeviceBuffer<DeviceOutputPixel> output_pixels;
  PinnedHostBuffer<DeviceCounters> host_iteration_counters;
  PinnedHostBuffer<float> host_iteration_depth_radii;
  PinnedHostBuffer<float> host_iteration_normal_radii;
  PinnedHostBuffer<DeviceOutputPixel> host_output_pixels;
  PinnedHostBuffer<float> host_dynamic_selection_probability_map;
  PinnedHostBuffer<float> host_depth_upload;
  PinnedHostBuffer<float> host_normal_upload;
  PinnedHostBuffer<float> host_reference_texture_upload;
  PinnedHostBuffer<float> host_source_images_texture_upload;
  PinnedHostBuffer<float> host_poses_texture_upload;
  PinnedHostBuffer<float> host_source_depth_maps_texture_upload;
  PinnedHostBuffer<float> host_selection_probability_texture_upload;
};

}  // namespace

class CudaPatchMatchWorkspace {
 public:
  CudaPatchMatchWorkspace() = default;
  CudaPatchMatchWorkspace(const CudaPatchMatchWorkspace&) = delete;
  CudaPatchMatchWorkspace& operator=(const CudaPatchMatchWorkspace&) = delete;
  ~CudaPatchMatchWorkspace() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  cudaStream_t Stream() {
    if (stream_ == nullptr) {
      const cudaError_t status = cudaStreamCreate(&stream_);
      if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
      }
    }
    return stream_;
  }

  CudaPatchMatchWorkspaceStorage storage_;
  cudaStream_t stream_ = nullptr;
};

void DestroyCudaPatchMatchWorkspace(CudaPatchMatchWorkspace* workspace) {
  delete workspace;
}

CudaPatchMatchWorkspacePtr MakeCudaPatchMatchWorkspace() {
  return CudaPatchMatchWorkspacePtr(new CudaPatchMatchWorkspace,
                                    &DestroyCudaPatchMatchWorkspace);
}

namespace {

__device__ float Random01(std::uint32_t seed) {
  std::uint32_t state = seed * 747796405u + 2891336453u;
  state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  state = (state >> 22u) ^ state;
  return static_cast<float>(state & 0x00FFFFFFu) / 16777215.0F;
}

__device__ float Clamp(float value, float min_value, float max_value) {
  return fminf(max_value, fmaxf(min_value, value));
}

__device__ bool IsFinitePositive(float value) {
  return isfinite(value) && value > 0.0F;
}

__device__ float3 Add3(float3 lhs, float3 rhs) {
  return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

__device__ float3 Scale3(float3 value, float scale) {
  return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__device__ float Dot3(float3 lhs, float3 rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__device__ float3 Cross3(float3 lhs, float3 rhs) {
  return make_float3(lhs.y * rhs.z - lhs.z * rhs.y,
                     lhs.z * rhs.x - lhs.x * rhs.z,
                     lhs.x * rhs.y - lhs.y * rhs.x);
}

__device__ float3 Normalize3(float3 value, float3 fallback) {
  const float norm = sqrtf(Dot3(value, value));
  if (!isfinite(norm) || norm <= 1e-8F) {
    return fallback;
  }
  return Scale3(value, 1.0F / norm);
}

__device__ float AngleDegreesBetween(float3 lhs, float3 rhs) {
  const float3 lhs_n = Normalize3(lhs, make_float3(0.0F, 0.0F, 1.0F));
  const float3 rhs_n = Normalize3(rhs, make_float3(0.0F, 0.0F, 1.0F));
  const float cosine = Clamp(Dot3(lhs_n, rhs_n), -1.0F, 1.0F);
  constexpr float kRadiansToDegrees = 57.29577951308232F;
  return acosf(cosine) * kRadiansToDegrees;
}

__device__ float3 PerturbNormal(float3 normal,
                                float radius_degrees,
                                std::uint32_t seed) {
  const float3 normal_n = Normalize3(normal, make_float3(0.0F, 0.0F, 1.0F));
  if (radius_degrees <= 0.0F) {
    return normal_n;
  }
  constexpr float kPi = 3.14159265358979323846F;
  const float radius_radians = radius_degrees * kPi / 180.0F;
  const float3 helper =
      fabsf(normal_n.z) < 0.9F ? make_float3(0.0F, 0.0F, 1.0F)
                               : make_float3(1.0F, 0.0F, 0.0F);
  const float3 tangent_u =
      Normalize3(Cross3(helper, normal_n), make_float3(1.0F, 0.0F, 0.0F));
  const float3 tangent_v = Cross3(normal_n, tangent_u);
  const float angle =
      radius_radians * sqrtf(Random01(seed ^ 0xA511E9B3u));
  const float phi = 2.0F * kPi * Random01(seed ^ 0x63D83595u);
  const float3 tangent_direction =
      Add3(Scale3(tangent_u, cosf(phi)), Scale3(tangent_v, sinf(phi)));
  return Normalize3(Add3(Scale3(normal_n, cosf(angle)),
                         Scale3(tangent_direction, sinf(angle))),
                    normal_n);
}

__device__ void AtomicMaxDouble(double* address, double value) {
  if (!isfinite(value)) {
    return;
  }
  unsigned long long* address_as_ull =
      reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value > __longlong_as_double(old)) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
    if (assumed == old) {
      break;
    }
  }
}

__device__ float SampleLayeredTexture(cudaTextureObject_t texture,
                                      int width,
                                      int height,
                                      int layer,
                                      float x,
                                      float y) {
  if (!isfinite(x) || !isfinite(y) || x < 0.0F || y < 0.0F ||
      x > static_cast<float>(width - 1) ||
      y > static_cast<float>(height - 1)) {
    return NAN;
  }
  return tex2DLayered<float>(texture, x + 0.5F, y + 0.5F, layer);
}

__device__ float SampleReferenceImage(cudaTextureObject_t reference_texture,
                                      const DevicePatchMatchOptions& options,
                                      const float* shared_reference_tile,
                                      int shared_start_idx,
                                      int shared_num_pixels,
                                      int col,
                                      int row) {
  if (col < 0 || row < 0 || col >= options.width || row >= options.height) {
    return NAN;
  }
  const int sample_idx = row * options.width + col;
  if (shared_reference_tile != nullptr &&
      sample_idx >= shared_start_idx &&
      sample_idx < shared_start_idx + shared_num_pixels) {
    return shared_reference_tile[sample_idx - shared_start_idx];
  }
  return SampleLayeredTexture(reference_texture, options.width,
                              options.height, 0,
                              static_cast<float>(col),
                              static_cast<float>(row));
}

__device__ float SampleSourceImage(cudaTextureObject_t source_images_texture,
                                   const DeviceSourceImage& source,
                                   int source_idx,
                                   float x,
                                   float y) {
  if (!isfinite(x) || !isfinite(y) || x < 0.0F || y < 0.0F ||
      x > static_cast<float>(source.width - 1) ||
      y > static_cast<float>(source.height - 1)) {
    return NAN;
  }
  return tex2DLayered<float>(source_images_texture, x + 0.5F, y + 0.5F,
                             source_idx);
}

__device__ float ReadPose(cudaTextureObject_t poses_texture,
                          int source_idx,
                          int variant_idx,
                          int param_idx) {
  return tex2DLayered<float>(poses_texture,
                             static_cast<float>(param_idx) + 0.5F,
                             static_cast<float>(source_idx) + 0.5F,
                             variant_idx);
}

__device__ bool ProjectPlaneToSource(const DevicePatchMatchOptions& options,
                                     cudaTextureObject_t poses_texture,
                                     int source_idx,
                                     int variant_idx,
                                     float center_x,
                                     float center_y,
                                     float sample_x,
                                     float sample_y,
                                     float depth,
                                     const float3& normal,
                                     float* source_x,
                                     float* source_y,
                                     float* source_depth) {
  const DeviceCamera& ref = options.reference_camera;
  const float3 center = make_float3((center_x - ref.cx) / ref.fx * depth,
                                    (center_y - ref.cy) / ref.fy * depth,
                                    depth);
  const float3 ray = make_float3((sample_x - ref.cx) / ref.fx,
                                 (sample_y - ref.cy) / ref.fy, 1.0F);
  const float numerator =
      normal.x * center.x + normal.y * center.y + normal.z * center.z;
  const float denominator =
      normal.x * ray.x + normal.y * ray.y + normal.z * ray.z;
  if (fabsf(denominator) < 1e-8F) {
    return false;
  }
  const float ray_depth = numerator / denominator;
  if (!IsFinitePositive(ray_depth)) {
    return false;
  }
  const float3 point_ref =
      make_float3(ray.x * ray_depth, ray.y * ray_depth, ray.z * ray_depth);
  float rotation[9];
  float translation[3];
  for (int i = 0; i < 9; ++i) {
    rotation[i] =
        ReadPose(poses_texture, source_idx, variant_idx, kPoseROffset + i);
  }
  for (int i = 0; i < 3; ++i) {
    translation[i] =
        ReadPose(poses_texture, source_idx, variant_idx, kPoseTOffset + i);
  }
  const float sx = rotation[0] * point_ref.x + rotation[1] * point_ref.y +
                   rotation[2] * point_ref.z + translation[0];
  const float sy = rotation[3] * point_ref.x + rotation[4] * point_ref.y +
                   rotation[5] * point_ref.z + translation[1];
  const float sz = rotation[6] * point_ref.x + rotation[7] * point_ref.y +
                   rotation[8] * point_ref.z + translation[2];
  if (!IsFinitePositive(sz)) {
    return false;
  }
  const float fx =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseKOffset + 0);
  const float fy =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseKOffset + 1);
  const float cx =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseKOffset + 2);
  const float cy =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseKOffset + 3);
  *source_x = fx * sx / sz + cx;
  *source_y = fy * sy / sz + cy;
  *source_depth = sz;
  return isfinite(*source_x) && isfinite(*source_y);
}

__device__ bool ComputeSourceDepthRoundTripErrors(
    const DevicePatchMatchOptions& options,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t source_depth_maps_texture,
    const DeviceSourceImage& source,
    int source_idx,
    int col,
    int row,
    float depth,
    float* signed_relative_error,
    float* reprojection_error) {
  constexpr int kBasePoseVariant = 0;
  const DeviceCamera& ref = options.reference_camera;
  const float point_ref[3] = {
      (static_cast<float>(col) - ref.cx) / ref.fx * depth,
      (static_cast<float>(row) - ref.cy) / ref.fy * depth,
      depth};

  float p[12];
  for (int i = 0; i < 12; ++i) {
    p[i] = ReadPose(poses_texture, source_idx, kBasePoseVariant,
                    kPosePOffset + i);
  }
  const float forward_z =
      p[8] * point_ref[0] + p[9] * point_ref[1] +
      p[10] * point_ref[2] + p[11];
  if (!IsFinitePositive(forward_z)) {
    return false;
  }
  const float source_x =
      (p[0] * point_ref[0] + p[1] * point_ref[1] +
       p[2] * point_ref[2] + p[3]) /
      forward_z;
  const float source_y =
      (p[4] * point_ref[0] + p[5] * point_ref[1] +
       p[6] * point_ref[2] + p[7]) /
      forward_z;
  if (!isfinite(source_x) || !isfinite(source_y) || source_x < 0.0F ||
      source_y < 0.0F || source_x > static_cast<float>(source.width - 1) ||
      source_y > static_cast<float>(source.height - 1)) {
    return false;
  }

  const float source_depth =
      SampleLayeredTexture(source_depth_maps_texture, options.max_source_width,
                           options.max_source_height, source_idx, source_x,
                           source_y);
  if (!IsFinitePositive(source_depth)) {
    return false;
  }

  float inv_p[12];
  for (int i = 0; i < 12; ++i) {
    inv_p[i] = ReadPose(poses_texture, source_idx, kBasePoseVariant,
                        kPoseInvPOffset + i);
  }
  const float homogeneous_source[4] = {
      source_x * source_depth, source_y * source_depth, source_depth, 1.0F};
  const float backward_z =
      inv_p[8] * homogeneous_source[0] +
      inv_p[9] * homogeneous_source[1] +
      inv_p[10] * homogeneous_source[2] + inv_p[11];
  if (!IsFinitePositive(backward_z)) {
    return false;
  }
  const float backward_x =
      inv_p[0] * homogeneous_source[0] +
      inv_p[1] * homogeneous_source[1] +
      inv_p[2] * homogeneous_source[2] + inv_p[3];
  const float backward_y =
      inv_p[4] * homogeneous_source[0] +
      inv_p[5] * homogeneous_source[1] +
      inv_p[6] * homogeneous_source[2] + inv_p[7];
  const float returned_x = ref.fx * backward_x / backward_z + ref.cx;
  const float returned_y = ref.fy * backward_y / backward_z + ref.cy;
  const float delta_x = returned_x - static_cast<float>(col);
  const float delta_y = returned_y - static_cast<float>(row);
  *signed_relative_error = (backward_z - depth) / depth;
  *reprojection_error = sqrtf(delta_x * delta_x + delta_y * delta_y);
  return isfinite(*signed_relative_error) && isfinite(*reprojection_error);
}

__device__ bool ComputeSourceDepthSignedRelativeError(
    const DevicePatchMatchOptions& options,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t source_depth_maps_texture,
    const DeviceSourceImage& source,
    int source_idx,
    int col,
    int row,
    float depth,
    float* relative_error) {
  float reprojection_error = 0.0F;
  return ComputeSourceDepthRoundTripErrors(
      options, poses_texture, source_depth_maps_texture, source, source_idx,
      col, row, depth, relative_error, &reprojection_error);
}

__device__ bool PassesGeometricCandidateFilter(
    const DevicePatchMatchOptions& options,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t source_depth_maps_texture,
    const DeviceSourceImage* source_images,
    int col,
    int row,
    float depth,
    DeviceCounters* counters) {
  if (source_depth_maps_texture == 0 ||
      options.cost.geometric_consistency_weight <= 0.0F) {
    return true;
  }

  int num_consistent_sources = 0;
  int num_occluded_sources = 0;
  int num_conflicting_sources = 0;
  for (int source_idx = 0; source_idx < options.num_sources; ++source_idx) {
    float signed_relative_error = 0.0F;
    if (!ComputeSourceDepthSignedRelativeError(
            options, poses_texture, source_depth_maps_texture,
            source_images[source_idx], source_idx, col, row, depth,
            &signed_relative_error)) {
      continue;
    }
    if (fabsf(signed_relative_error) <=
        options.cost.geometric_max_relative_depth_error) {
      ++num_consistent_sources;
    } else if (signed_relative_error < 0.0F) {
      ++num_occluded_sources;
    } else {
      ++num_conflicting_sources;
    }
  }

  bool accepted = num_consistent_sources >= options.cost.min_num_valid_sources;
  if (accepted && options.cost.geometric_reject_occluded_candidates &&
      num_occluded_sources > 0) {
    accepted = false;
  }
  if (accepted && options.cost.geometric_reject_depth_conflicts &&
      num_conflicting_sources > 0 &&
      num_conflicting_sources >= num_consistent_sources) {
    accepted = false;
  }
  if (counters != nullptr) {
    atomicAdd(accepted ? &counters->geometric_candidate_acceptances
                       : &counters->geometric_candidate_rejections,
              1ULL);
  }
  return accepted;
}

__device__ float ComputeIncidentAnglePenalty(
    cudaTextureObject_t poses_texture,
    const DevicePatchMatchOptions& options,
    int source_idx,
    float col,
    float row,
    float depth,
    float3 normal) {
  if (options.cost.incident_angle_weight <= 0.0F) {
    return 1.0F;
  }
  const DeviceCamera& ref = options.reference_camera;
  const float3 point =
      make_float3((col - ref.cx) / ref.fx * depth,
                  (row - ref.cy) / ref.fy * depth, depth);
  constexpr int kBasePoseVariant = 0;
  const float3 source_center = make_float3(
      ReadPose(poses_texture, source_idx, kBasePoseVariant, kPoseCOffset + 0),
      ReadPose(poses_texture, source_idx, kBasePoseVariant, kPoseCOffset + 1),
      ReadPose(poses_texture, source_idx, kBasePoseVariant, kPoseCOffset + 2));
  const float3 view_dir = make_float3(source_center.x - point.x,
                                      source_center.y - point.y,
                                      source_center.z - point.z);
  const float view_norm =
      sqrtf(view_dir.x * view_dir.x + view_dir.y * view_dir.y +
            view_dir.z * view_dir.z);
  const float normal_norm =
      sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  if (view_norm <= 1e-8F || normal_norm <= 1e-8F) {
    return 1.0F + options.cost.incident_angle_weight;
  }
  const float cosine =
      fabsf((normal.x * view_dir.x + normal.y * view_dir.y +
             normal.z * view_dir.z) /
            (normal_norm * view_norm));
  return 1.0F + options.cost.incident_angle_weight * (1.0F - cosine);
}

__device__ float GetSourceSelectionProbability(
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t selection_texture,
    const float* dynamic_probabilities,
    int num_pixels,
    int width,
    int source_idx,
    int col,
    int row) {
  constexpr int kBasePoseVariant = 0;
  float probability =
      dynamic_probabilities != nullptr
          ? dynamic_probabilities[source_idx * num_pixels +
                                  row * width + col]
          : selection_texture != 0
          ? tex2DLayered<float>(selection_texture,
                                static_cast<float>(col) + 0.5F,
                                static_cast<float>(row) + 0.5F, source_idx)
          : ReadPose(poses_texture, source_idx, kBasePoseVariant,
                     kPoseSelectionProbabilityOffset);
  if (dynamic_probabilities != nullptr) {
    // 动态概率描述光度可信度，静态 pose 概率继续作为几何/source prior。
    probability *= ReadPose(poses_texture, source_idx, kBasePoseVariant,
                            kPoseSelectionProbabilityOffset);
  }
  return Clamp(probability, 0.0F, 1.0F);
}

__device__ bool ShouldUseSourceForPixel(cudaTextureObject_t poses_texture,
                                        cudaTextureObject_t selection_texture,
                                        const float* dynamic_probabilities,
                                        int num_pixels,
                                        int width,
                                        float dynamic_min_probability,
                                        int num_source_samples,
                                        int source_idx,
                                        int col,
                                        int row,
                                        DeviceCounters* counters) {
  if (counters != nullptr) {
    atomicAdd(&counters->source_selection_tests, 1ULL);
  }
  const float probability = GetSourceSelectionProbability(
      poses_texture, selection_texture, dynamic_probabilities, num_pixels,
      width, source_idx, col, row);
  if (!isfinite(probability) || probability <= 0.0F) {
    if (counters != nullptr) {
      atomicAdd(&counters->source_selection_rejections, 1ULL);
    }
    return false;
  }
  if (num_source_samples > 0) {
    if (counters != nullptr) {
      atomicAdd(&counters->source_selection_acceptances, 1ULL);
    }
    return true;
  }
  if (dynamic_probabilities != nullptr) {
    // checkerboard 路径没有 COLMAP 的固定数量 CDF 抽样；这里采用确定性
    // floor gate，避免独立 Bernoulli 导致有效 source 数随机不足。
    const bool accepted = probability > dynamic_min_probability;
    if (counters != nullptr) {
      atomicAdd(accepted ? &counters->source_selection_acceptances
                         : &counters->source_selection_rejections,
                1ULL);
    }
    return accepted;
  }
  if (probability >= 0.999F) {
    if (counters != nullptr) {
      atomicAdd(&counters->source_selection_acceptances, 1ULL);
    }
    return true;
  }
  const std::uint32_t seed =
      0x9E3779B9u ^
      (static_cast<std::uint32_t>(source_idx + 1) * 2246822519u) ^
      (static_cast<std::uint32_t>(row + 1) * 3266489917u) ^
      (static_cast<std::uint32_t>(col + 1) * 668265263u);
  const bool accepted = Random01(seed) <= probability;
  if (counters != nullptr) {
    atomicAdd(accepted ? &counters->source_selection_acceptances
                       : &counters->source_selection_rejections,
              1ULL);
  }
  return accepted;
}

__device__ float ComputeSourceCostForVariant(
    cudaTextureObject_t reference_texture,
    cudaTextureObject_t source_images_texture,
    cudaTextureObject_t source_depth_maps_texture,
    cudaTextureObject_t poses_texture,
    const float* shared_reference_tile,
    int shared_reference_start_idx,
    int shared_reference_num_pixels,
    const DeviceSourceImage* source_images,
    const DevicePatchMatchOptions& options,
    int source_idx,
    int variant_idx,
    int col,
    int row,
    float depth,
    float3 normal) {
  double weight_sum = 0.0;
  double sum_ref = 0.0;
  double sum_src = 0.0;
  double sum_ref2 = 0.0;
  double sum_src2 = 0.0;
  double sum_cross = 0.0;
  int num_samples = 0;
  const DeviceSourceImage& source = source_images[source_idx];
  float center_source_x = 0.0F;
  float center_source_y = 0.0F;
  float center_source_depth = 0.0F;
  if (!ProjectPlaneToSource(options, poses_texture, source_idx, variant_idx,
                            static_cast<float>(col), static_cast<float>(row),
                            static_cast<float>(col), static_cast<float>(row),
                            depth, normal, &center_source_x, &center_source_y,
                            &center_source_depth)) {
    return options.cost.invalid_cost;
  }
  const float sweep_cos =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseSweepCosOffset);
  const float sweep_sin =
      ReadPose(poses_texture, source_idx, variant_idx, kPoseSweepSinOffset);
  const float reference_center = SampleReferenceImage(
      reference_texture, options, shared_reference_tile,
      shared_reference_start_idx, shared_reference_num_pixels, col, row);
  if (!isfinite(reference_center)) {
    return options.cost.invalid_cost;
  }
  const float sigma_spatial =
      options.cost.sigma_spatial > 0.0F
          ? options.cost.sigma_spatial
          : static_cast<float>(max(1, options.cost.window_radius));
  const float spatial_normalization =
      1.0F / (2.0F * sigma_spatial * sigma_spatial);
  const float color_normalization =
      1.0F / (2.0F * options.cost.sigma_color * options.cost.sigma_color);
  for (int dy = -options.cost.window_radius; dy <= options.cost.window_radius;
       dy += options.cost.window_step) {
    for (int dx = -options.cost.window_radius;
         dx <= options.cost.window_radius; dx += options.cost.window_step) {
      const float sample_x = static_cast<float>(col + dx);
      const float sample_y = static_cast<float>(row + dy);
      const float ref_value = SampleReferenceImage(
          reference_texture, options, shared_reference_tile,
          shared_reference_start_idx, shared_reference_num_pixels, col + dx,
          row + dy);
      float source_x = 0.0F;
      float source_y = 0.0F;
      float projected_source_depth = 0.0F;
      if (!isfinite(ref_value) ||
          !ProjectPlaneToSource(options, poses_texture, source_idx,
                                variant_idx,
                                static_cast<float>(col),
                                static_cast<float>(row), sample_x,
                                sample_y, depth, normal, &source_x, &source_y,
                                &projected_source_depth)) {
        return options.cost.invalid_cost;
      }
      const float delta_x = source_x - center_source_x;
      const float delta_y = source_y - center_source_y;
      source_x = center_source_x + sweep_cos * delta_x - sweep_sin * delta_y;
      source_y = center_source_y + sweep_sin * delta_x + sweep_cos * delta_y;
      const float src_value = SampleSourceImage(
          source_images_texture, source, source_idx, source_x, source_y);
      if (!isfinite(src_value)) {
        return options.cost.invalid_cost;
      }
      const float color_difference = ref_value - reference_center;
      // 权重只由参考图边缘结构决定，保证不同 source/候选之间可比较。
      const float weight =
          options.cost.bilateral_weighting
              ? expf(-static_cast<float>(dx * dx + dy * dy) *
                         spatial_normalization -
                     color_difference * color_difference *
                         color_normalization)
              : 1.0F;
      weight_sum += weight;
      sum_ref += static_cast<double>(weight) * ref_value;
      sum_src += static_cast<double>(weight) * src_value;
      sum_ref2 += static_cast<double>(weight) * ref_value * ref_value;
      sum_src2 += static_cast<double>(weight) * src_value * src_value;
      sum_cross += static_cast<double>(weight) * ref_value * src_value;
      num_samples++;
    }
  }
  if (num_samples <= 0 || !isfinite(weight_sum) || weight_sum <= 0.0) {
    return options.cost.invalid_cost;
  }
  const double inverse_weight_sum = 1.0 / weight_sum;
  sum_ref *= inverse_weight_sum;
  sum_src *= inverse_weight_sum;
  sum_ref2 *= inverse_weight_sum;
  sum_src2 *= inverse_weight_sum;
  sum_cross *= inverse_weight_sum;
  const double ref_var = sum_ref2 - sum_ref * sum_ref;
  const double src_var = sum_src2 - sum_src * sum_src;
  if (ref_var <= options.cost.min_variance ||
      src_var <= options.cost.min_variance) {
    return options.cost.invalid_cost;
  }
  const double covariance = sum_cross - sum_ref * sum_src;
  const double zncc = fmax(-1.0, fmin(1.0, covariance / sqrt(ref_var * src_var)));
  float cost = static_cast<float>(0.5 * (1.0 - zncc));

  if (source_depth_maps_texture != 0 &&
      options.cost.geometric_consistency_weight > 0.0F) {
    float signed_relative_error = 0.0F;
    float reprojection_error = 0.0F;
    if (ComputeSourceDepthRoundTripErrors(
            options, poses_texture, source_depth_maps_texture, source,
            source_idx, col, row, depth, &signed_relative_error,
            &reprojection_error)) {
      cost += options.cost.geometric_consistency_weight *
              fminf(options.cost.geometric_max_reprojection_error,
                    reprojection_error);
    }
  }
  const float source_weight =
      ReadPose(poses_texture, source_idx, 0, kPoseWeightOffset);
  if (isfinite(source_weight) && source_weight > 0.0F &&
      options.cost.source_weight_strength > 0.0F) {
    cost *= 1.0F + options.cost.source_weight_strength *
                       (1.0F - fminf(1.0F, source_weight));
  }
  cost *= ComputeIncidentAnglePenalty(
      poses_texture, options, source_idx, static_cast<float>(col),
      static_cast<float>(row), depth, normal);
  return cost;
}

__device__ float ComputeSourceCost(cudaTextureObject_t reference_texture,
                                   cudaTextureObject_t source_images_texture,
                                   cudaTextureObject_t source_depth_maps_texture,
                                   cudaTextureObject_t poses_texture,
                                   cudaTextureObject_t selection_texture,
                                   const float* dynamic_probabilities,
                                   const float* shared_reference_tile,
                                   int shared_reference_start_idx,
                                   int shared_reference_num_pixels,
                                   const DeviceSourceImage* source_images,
                                   const DevicePatchMatchOptions& options,
                                   int source_idx,
                                   int col,
                                   int row,
                                   float depth,
                                   float3 normal,
                                   DeviceCounters* counters) {
  if (!ShouldUseSourceForPixel(
          poses_texture, selection_texture, dynamic_probabilities,
          options.width * options.height, options.width,
          options.min_selection_probability, options.num_source_samples,
          source_idx, col, row, counters)) {
    return options.cost.invalid_cost;
  }
  float best_cost = options.cost.invalid_cost;
  for (int variant_idx = 0; variant_idx < options.num_pose_variants;
       ++variant_idx) {
    const float cost = ComputeSourceCostForVariant(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, shared_reference_tile, shared_reference_start_idx,
        shared_reference_num_pixels, source_images, options, source_idx,
        variant_idx, col, row, depth, normal);
    if (isfinite(cost) && cost < best_cost) {
      best_cost = cost;
    }
  }
  return best_cost;
}

__device__ bool ComputeAggregatedCost(cudaTextureObject_t reference_texture,
                                      cudaTextureObject_t source_images_texture,
                                      cudaTextureObject_t source_depth_maps_texture,
                                      cudaTextureObject_t poses_texture,
                                      cudaTextureObject_t selection_texture,
                                      const float* dynamic_probabilities,
                                      const float* shared_reference_tile,
                                      int shared_reference_start_idx,
                                      int shared_reference_num_pixels,
                                      const DeviceSourceImage* source_images,
                                      const DevicePatchMatchOptions& options,
                                      int col,
                                      int row,
                                      float depth,
                                      float3 normal,
                                      float* aggregated_cost,
                                      DeviceCounters* counters) {
  float best_costs[kMaxCudaSourceImages];
  float valid_costs[kMaxCudaSourceImages];
  float valid_weights[kMaxCudaSourceImages];
  int num_valid = 0;
  for (int source_idx = 0; source_idx < options.num_sources; ++source_idx) {
    const float source_cost = ComputeSourceCost(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, selection_texture, dynamic_probabilities,
        shared_reference_tile,
        shared_reference_start_idx, shared_reference_num_pixels, source_images,
        options, source_idx, col, row, depth, normal, counters);
    if (!isfinite(source_cost) || source_cost >= options.cost.invalid_cost) {
      continue;
    }
    valid_costs[num_valid] = source_cost;
    valid_weights[num_valid] = GetSourceSelectionProbability(
        poses_texture, selection_texture, dynamic_probabilities,
        options.width * options.height, options.width, source_idx, col, row);
    int insert_idx = num_valid;
    while (insert_idx > 0 && best_costs[insert_idx - 1] > source_cost) {
      if (insert_idx < kMaxCudaSourceImages) {
        best_costs[insert_idx] = best_costs[insert_idx - 1];
      }
      --insert_idx;
    }
    if (insert_idx < kMaxCudaSourceImages) {
      best_costs[insert_idx] = source_cost;
    }
    num_valid = min(num_valid + 1, kMaxCudaSourceImages);
  }
  if (num_valid < options.cost.min_num_valid_sources) {
    *aggregated_cost = options.cost.invalid_cost;
    return false;
  }
  if (options.num_source_samples > 0) {
    float weight_sum = 0.0F;
    for (int i = 0; i < num_valid; ++i) {
      weight_sum += fmaxf(0.0F, valid_weights[i]);
    }
    if (isfinite(weight_sum) && weight_sum > 1e-8F) {
      float sampled_total = 0.0F;
      const std::uint32_t base_seed =
          0xA511E9B3u ^
          (static_cast<std::uint32_t>(row + 1) * 2246822519u) ^
          (static_cast<std::uint32_t>(col + 1) * 3266489917u);
      for (int sample = 0; sample < options.num_source_samples; ++sample) {
        const float target =
            Random01(base_seed + static_cast<std::uint32_t>(sample) *
                                     668265263u) *
            weight_sum;
        float cumulative = 0.0F;
        int selected = num_valid - 1;
        for (int i = 0; i < num_valid; ++i) {
          cumulative += fmaxf(0.0F, valid_weights[i]);
          if (cumulative >= target) {
            selected = i;
            break;
          }
        }
        sampled_total += valid_costs[selected];
      }
      *aggregated_cost =
          sampled_total / static_cast<float>(options.num_source_samples);
      return true;
    }
  }
  const int num_used =
      options.cost.max_num_aggregated_costs <= 0
          ? num_valid
          : min(num_valid, options.cost.max_num_aggregated_costs);
  float total = 0.0F;
  for (int i = 0; i < num_used; ++i) {
    total += best_costs[i];
  }
  *aggregated_cost = total / static_cast<float>(num_used);
  return true;
}

__device__ float ComputeNccLikelihood(float colite_cost, float ncc_sigma) {
  constexpr float kPi = 3.14159265358979323846F;
  const float colmap_cost = 2.0F * Clamp(colite_cost, 0.0F, 1.0F);
  const float normalization =
      2.0F /
      (sqrtf(2.0F * kPi) * ncc_sigma *
       erff(2.0F / (ncc_sigma * sqrtf(2.0F))));
  return expf(-0.5F * colmap_cost * colmap_cost /
              (ncc_sigma * ncc_sigma)) * normalization;
}

__global__ void UpdateSelectionProbabilityKernel(
    cudaTextureObject_t reference_texture,
    cudaTextureObject_t source_images_texture,
    cudaTextureObject_t source_depth_maps_texture,
    cudaTextureObject_t poses_texture,
    const DeviceSourceImage* source_images,
    const float* depth_map,
    const float* normal_map,
    float* dynamic_probabilities,
    float* emissions,
    DevicePatchMatchOptions options,
    DeviceCounters* counters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  const int total = num_pixels * options.num_sources;
  if (idx >= total) {
    return;
  }
  const int source_idx = idx / num_pixels;
  const int pixel_idx = idx - source_idx * num_pixels;
  const int row = pixel_idx / options.width;
  const int col = pixel_idx - row * options.width;
  const float depth = depth_map[pixel_idx];
  const float3 normal =
      make_float3(normal_map[pixel_idx * 3 + 0],
                  normal_map[pixel_idx * 3 + 1],
                  normal_map[pixel_idx * 3 + 2]);

  float best_cost = options.cost.invalid_cost;
  if (IsFinitePositive(depth)) {
    for (int variant_idx = 0; variant_idx < options.num_pose_variants;
         ++variant_idx) {
      const float cost = ComputeSourceCostForVariant(
          reference_texture, source_images_texture, source_depth_maps_texture,
          poses_texture, nullptr, 0, 0, source_images, options, source_idx,
          variant_idx, col, row, depth, normal);
      if (isfinite(cost) && cost < best_cost) {
        best_cost = cost;
      }
    }
  }

  const float prior = Clamp(dynamic_probabilities[idx],
                            options.min_selection_probability,
                            options.max_selection_probability);
  const float emission = ComputeNccLikelihood(best_cost, options.ncc_sigma);
  constexpr float kUniformOutlierLikelihood = 0.5F;
  const float selected = prior * emission;
  const float rejected = (1.0F - prior) * kUniformOutlierLikelihood;
  const float denominator = selected + rejected;
  const float posterior =
      denominator > 1e-12F ? selected / denominator : prior;
  const float updated = Clamp(
      options.previous_selection_probability_weight * prior +
          (1.0F - options.previous_selection_probability_weight) * posterior,
      options.min_selection_probability,
      options.max_selection_probability);
  if (options.directional_hmm && emissions != nullptr) {
    emissions[idx] = emission;
  } else {
    dynamic_probabilities[idx] = updated;
  }
  if (counters != nullptr) {
    atomicAdd(&counters->source_selection_probability_updates, 1ULL);
    atomicAdd(&counters->total_source_selection_probability,
              static_cast<double>(updated));
  }
}

__device__ float HmmForwardMessage(float emission, float previous) {
  constexpr float kUniformProbability = 0.5F;
  constexpr float kNoChangeProbability = 0.99999F;
  constexpr float kChangeProbability = 1.0F - kNoChangeProbability;
  const float rejected =
      (previous * kChangeProbability +
       (1.0F - previous) * kNoChangeProbability) *
      kUniformProbability;
  const float selected =
      (previous * kNoChangeProbability +
       (1.0F - previous) * kChangeProbability) *
      emission;
  const float denominator = rejected + selected;
  return denominator > 1e-12F ? selected / denominator
                              : kUniformProbability;
}

__device__ float HmmBackwardMessage(float emission, float previous) {
  constexpr float kUniformProbability = 0.5F;
  constexpr float kNoChangeProbability = 0.99999F;
  constexpr float kChangeProbability = 1.0F - kNoChangeProbability;
  const float rejected = previous * emission * kChangeProbability +
                         (1.0F - previous) * kUniformProbability *
                             kNoChangeProbability;
  const float selected = previous * emission * kNoChangeProbability +
                         (1.0F - previous) * kUniformProbability *
                             kChangeProbability;
  const float denominator = rejected + selected;
  return denominator > 1e-12F ? selected / denominator
                              : kUniformProbability;
}

__device__ float HmmSelectionProbability(float alpha,
                                         float beta,
                                         float previous,
                                         DevicePatchMatchOptions options) {
  const float rejected = (1.0F - alpha) * (1.0F - beta);
  const float selected = alpha * beta;
  const float denominator = rejected + selected;
  const float current = denominator > 1e-12F ? selected / denominator : 0.5F;
  return Clamp(options.previous_selection_probability_weight * previous +
                   (1.0F - options.previous_selection_probability_weight) *
                       current,
               options.min_selection_probability,
               options.max_selection_probability);
}

__global__ void DirectionalHmmRowsKernel(
    const float* emissions,
    float* backward_messages,
    float* probabilities,
    DevicePatchMatchOptions options) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_lines = options.num_sources * options.height;
  if (line >= num_lines) return;
  const int source_idx = line / options.height;
  const int row = line - source_idx * options.height;
  const int base = source_idx * options.width * options.height +
                   row * options.width;
  float beta = 0.5F;
  for (int col = options.width - 1; col >= 0; --col) {
    const int idx = base + col;
    beta = HmmBackwardMessage(emissions[idx], beta);
    backward_messages[idx] = beta;
  }
  float alpha = 0.5F;
  for (int col = 0; col < options.width; ++col) {
    const int idx = base + col;
    alpha = HmmForwardMessage(emissions[idx], alpha);
    probabilities[idx] = HmmSelectionProbability(
        alpha, backward_messages[idx], probabilities[idx], options);
  }
}

__global__ void DirectionalHmmColumnsKernel(
    const float* emissions,
    float* backward_messages,
    float* probabilities,
    DevicePatchMatchOptions options) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_lines = options.num_sources * options.width;
  if (line >= num_lines) return;
  const int source_idx = line / options.width;
  const int col = line - source_idx * options.width;
  const int source_base = source_idx * options.width * options.height;
  float beta = 0.5F;
  for (int row = options.height - 1; row >= 0; --row) {
    const int idx = source_base + row * options.width + col;
    beta = HmmBackwardMessage(emissions[idx], beta);
    backward_messages[idx] = beta;
  }
  float alpha = 0.5F;
  for (int row = 0; row < options.height; ++row) {
    const int idx = source_base + row * options.width + col;
    alpha = HmmForwardMessage(emissions[idx], alpha);
    probabilities[idx] = HmmSelectionProbability(
        alpha, backward_messages[idx], probabilities[idx], options);
  }
}

__global__ void InitHypothesesKernel(float* depth_map,
                                     float* normal_map,
                                     DevicePatchMatchOptions options,
                                     std::uint32_t seed,
                                     float3 default_normal,
                                     bool randomize_initial_normals,
                                     float initial_normal_radius_degrees) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  if (idx >= num_pixels) {
    return;
  }
  const float depth =
      options.depth_min +
      Random01(seed + static_cast<std::uint32_t>(idx) * 17u) *
          (options.depth_max - options.depth_min);
  depth_map[idx] = depth;
  // 使用像素索引派生独立随机序列，保证 CUDA 初始化在固定 seed 下可复现。
  const float3 normal =
      randomize_initial_normals
          ? PerturbNormal(default_normal, initial_normal_radius_degrees,
                          seed + static_cast<std::uint32_t>(idx) * 131u)
          : default_normal;
  normal_map[idx * 3 + 0] = normal.x;
  normal_map[idx * 3 + 1] = normal.y;
  normal_map[idx * 3 + 2] = normal.z;
}

__global__ void ResetCountersKernel(DeviceCounters* counters) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counters[0] = DeviceCounters{};
  }
}

__global__ void StoreIterationCountersKernel(
    const DeviceCounters* counters,
    DeviceCounters* iteration_counters,
    int iteration_idx) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    iteration_counters[iteration_idx] = counters[0];
  }
}

__global__ void InitPerturbationRadiusStateKernel(float* radius_state,
                                                  float depth_radius,
                                                  float normal_radius) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    radius_state[0] = depth_radius;
    radius_state[1] = normal_radius;
  }
}

__global__ void StorePerturbationRadiiKernel(const float* radius_state,
                                             float* iteration_depth_radii,
                                             float* iteration_normal_radii,
                                             int iteration_idx,
                                             float min_depth_radius,
                                             float min_normal_radius) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    iteration_depth_radii[iteration_idx] =
        fmaxf(min_depth_radius, radius_state[0]);
    iteration_normal_radii[iteration_idx] =
        fmaxf(min_normal_radius, radius_state[1]);
  }
}

__global__ void UpdatePerturbationRadiusStateKernel(
    float* radius_state,
    const DeviceCounters* counters,
    float perturbation_shrink,
    float adaptive_min_update_ratio) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const double candidate_evaluations =
      static_cast<double>(counters[0].candidate_evaluations);
  const double total_updates =
      static_cast<double>(counters[0].propagation_updates +
                          counters[0].perturbation_updates);
  const double update_ratio =
      candidate_evaluations <= 0.0 ? 0.0 : total_updates / candidate_evaluations;
  float shrink = perturbation_shrink;
  if (update_ratio < static_cast<double>(adaptive_min_update_ratio)) {
    shrink *= perturbation_shrink;
  }
  radius_state[0] *= shrink;
  radius_state[1] *= shrink;
}

__global__ void PackOutputMapsKernel(const float* depth_map,
                                     const float* normal_map,
                                     const float* cost_map,
                                     DeviceOutputPixel* output_pixels,
                                     int num_pixels) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pixels) {
    return;
  }
  DeviceOutputPixel output;
  output.depth = depth_map[idx];
  const float3 normal =
      Normalize3(make_float3(normal_map[idx * 3 + 0],
                             normal_map[idx * 3 + 1],
                             normal_map[idx * 3 + 2]),
                 make_float3(0.0F, 0.0F, 0.0F));
  output.normal_x = normal.x;
  output.normal_y = normal.y;
  output.normal_z = normal.z;
  output.cost = cost_map[idx];
  output_pixels[idx] = output;
}

__global__ void EvaluateCostKernel(cudaTextureObject_t reference_texture,
                                   cudaTextureObject_t source_images_texture,
                                   cudaTextureObject_t source_depth_maps_texture,
                                   cudaTextureObject_t poses_texture,
                                   cudaTextureObject_t selection_texture,
                                   const float* dynamic_probabilities,
                                   const DeviceSourceImage* source_images,
                                   const float* depth_map,
                                   const float* normal_map,
                                   float* cost_map,
                                   DevicePatchMatchOptions options) {
  extern __shared__ float shared_reference_tile[];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  const int block_start_idx = blockIdx.x * blockDim.x;
  int shared_num_pixels = num_pixels - block_start_idx;
  if (shared_num_pixels < 0) {
    shared_num_pixels = 0;
  }
  if (shared_num_pixels > blockDim.x) {
    shared_num_pixels = blockDim.x;
  }
  const int row = idx / options.width;
  const int col = idx - row * options.width;
  shared_reference_tile[threadIdx.x] =
      idx < num_pixels
          ? SampleLayeredTexture(reference_texture, options.width,
                                 options.height, 0,
                                 static_cast<float>(col),
                                 static_cast<float>(row))
          : NAN;
  __syncthreads();
  if (idx >= num_pixels) {
    return;
  }
  const float depth = depth_map[idx];
  const float3 normal =
      make_float3(normal_map[idx * 3 + 0], normal_map[idx * 3 + 1],
                  normal_map[idx * 3 + 2]);
  float cost = options.cost.invalid_cost;
  if (IsFinitePositive(depth)) {
    ComputeAggregatedCost(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          selection_texture, dynamic_probabilities,
                          shared_reference_tile,
                          block_start_idx, shared_num_pixels, source_images,
                          options, col, row, depth, normal, &cost, nullptr);
  }
  cost_map[idx] = cost;
}

__device__ bool ProjectNeighborPlaneDepthToPixel(
    const DevicePatchMatchOptions& options,
    int neighbor_col,
    int neighbor_row,
    float neighbor_depth,
    float3 neighbor_normal,
    int target_col,
    int target_row,
    float* target_depth) {
  const DeviceCamera& camera = options.reference_camera;
  if (!IsFinitePositive(neighbor_depth) || target_depth == nullptr) {
    return false;
  }
  const float3 neighbor_ray =
      make_float3((static_cast<float>(neighbor_col) - camera.cx) / camera.fx,
                  (static_cast<float>(neighbor_row) - camera.cy) / camera.fy,
                  1.0F);
  const float3 target_ray =
      make_float3((static_cast<float>(target_col) - camera.cx) / camera.fx,
                  (static_cast<float>(target_row) - camera.cy) / camera.fy,
                  1.0F);
  const float denominator = Dot3(neighbor_normal, target_ray);
  if (!isfinite(denominator) || fabsf(denominator) <= 1e-8F) {
    return false;
  }
  const float numerator =
      Dot3(neighbor_normal, Scale3(neighbor_ray, neighbor_depth));
  *target_depth = numerator / denominator;
  return IsFinitePositive(*target_depth);
}

__device__ void TryUpdateFromNeighbor(cudaTextureObject_t reference_texture,
                                      cudaTextureObject_t source_images_texture,
                                      cudaTextureObject_t source_depth_maps_texture,
                                      cudaTextureObject_t poses_texture,
                                      cudaTextureObject_t selection_texture,
                                      const float* dynamic_probabilities,
                                      const DeviceSourceImage* source_images,
                                      float* depth_map,
                                      float* normal_map,
                                      float* cost_map,
                                      DevicePatchMatchOptions options,
                                      int row,
                                      int col,
                                      int neighbor_row,
                                      int neighbor_col,
                                      DeviceCounters* counters) {
  if (neighbor_row < 0 || neighbor_row >= options.height || neighbor_col < 0 ||
      neighbor_col >= options.width) {
    return;
  }
  const int idx = row * options.width + col;
  const int neighbor_idx = neighbor_row * options.width + neighbor_col;
  const float neighbor_depth = depth_map[neighbor_idx];
  const float3 candidate_normal =
      make_float3(normal_map[neighbor_idx * 3 + 0],
                  normal_map[neighbor_idx * 3 + 1],
                  normal_map[neighbor_idx * 3 + 2]);
  float candidate_depth = 0.0F;
  // 将邻居平面传播到当前像素射线，避免倾斜平面被错误地传播为等深面。
  if (!ProjectNeighborPlaneDepthToPixel(
          options, neighbor_col, neighbor_row, neighbor_depth,
          candidate_normal, col, row, &candidate_depth) ||
      candidate_depth < options.depth_min || candidate_depth > options.depth_max) {
    return;
  }
  atomicAdd(&counters->plane_propagation_candidates, 1ULL);
  atomicAdd(&counters->candidate_evaluations, 1ULL);
  if (!PassesGeometricCandidateFilter(options, poses_texture,
                                      source_depth_maps_texture,
                                      source_images, col, row,
                                      candidate_depth, counters)) {
    return;
  }
  float candidate_cost = options.cost.invalid_cost;
  if (!ComputeAggregatedCost(reference_texture, source_images_texture,
                             source_depth_maps_texture, poses_texture,
                             selection_texture, dynamic_probabilities, nullptr,
                             0, 0, source_images,
                             options, col, row, candidate_depth,
                             candidate_normal, &candidate_cost, counters)) {
    return;
  }
  const float current_cost = cost_map[idx];
  if (!isfinite(current_cost) || candidate_cost < current_cost) {
    depth_map[idx] = candidate_depth;
    normal_map[idx * 3 + 0] = candidate_normal.x;
    normal_map[idx * 3 + 1] = candidate_normal.y;
    normal_map[idx * 3 + 2] = candidate_normal.z;
    cost_map[idx] = candidate_cost;
    atomicAdd(&counters->propagation_updates, 1ULL);
  }
}

__global__ void PropagationKernel(cudaTextureObject_t reference_texture,
                                  cudaTextureObject_t source_images_texture,
                                  cudaTextureObject_t source_depth_maps_texture,
                                  cudaTextureObject_t poses_texture,
                                  cudaTextureObject_t selection_texture,
                                  const float* dynamic_probabilities,
                                  const DeviceSourceImage* source_images,
                                  float* depth_map,
                                  float* normal_map,
                                  float* cost_map,
                                  DevicePatchMatchOptions options,
                                  int direction,
                                  DeviceCounters* counters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  if (idx >= num_pixels) {
    return;
  }
  const int row = idx / options.width;
  const int col = idx - row * options.width;
  if (direction == 0) {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          selection_texture, dynamic_probabilities,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row, col - 1, counters);
  } else if (direction == 1) {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          selection_texture, dynamic_probabilities,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row, col + 1, counters);
  } else if (direction == 2) {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          selection_texture, dynamic_probabilities,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row - 1, col, counters);
  } else {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          selection_texture, dynamic_probabilities,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row + 1, col, counters);
  }
}

__global__ void CheckerboardPropagationKernel(
    cudaTextureObject_t reference_texture,
    cudaTextureObject_t source_images_texture,
    cudaTextureObject_t source_depth_maps_texture,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t selection_texture,
    const float* dynamic_probabilities,
    const DeviceSourceImage* source_images,
    float* depth_map,
    float* normal_map,
    float* cost_map,
    DevicePatchMatchOptions options,
    int parity,
    DeviceCounters* counters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  if (idx >= num_pixels) {
    return;
  }
  const int row = idx / options.width;
  const int col = idx - row * options.width;
  if (((row + col) & 1) != parity) {
    return;
  }
  TryUpdateFromNeighbor(reference_texture, source_images_texture,
                        source_depth_maps_texture, poses_texture,
                        selection_texture, dynamic_probabilities, source_images,
                        depth_map,
                        normal_map, cost_map, options, row, col, row,
                        col - 1, counters);
  TryUpdateFromNeighbor(reference_texture, source_images_texture,
                        source_depth_maps_texture, poses_texture,
                        selection_texture, dynamic_probabilities, source_images,
                        depth_map,
                        normal_map, cost_map, options, row, col, row,
                        col + 1, counters);
  TryUpdateFromNeighbor(reference_texture, source_images_texture,
                        source_depth_maps_texture, poses_texture,
                        selection_texture, dynamic_probabilities, source_images,
                        depth_map,
                        normal_map, cost_map, options, row, col, row - 1,
                        col, counters);
  TryUpdateFromNeighbor(reference_texture, source_images_texture,
                        source_depth_maps_texture, poses_texture,
                        selection_texture, dynamic_probabilities, source_images,
                        depth_map,
                        normal_map, cost_map, options, row, col, row + 1,
                        col, counters);
}

__device__ void TryUpdatePerturbationCandidate(
    cudaTextureObject_t reference_texture,
    cudaTextureObject_t source_images_texture,
    cudaTextureObject_t source_depth_maps_texture,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t selection_texture,
    const float* dynamic_probabilities,
    const DeviceSourceImage* source_images,
    const DevicePatchMatchOptions& options,
    int col,
    int row,
    float candidate_depth,
    float3 candidate_normal,
    DeviceCounters* counters,
    float* best_depth,
    float3* best_normal,
    float* best_cost,
    bool* updated) {
  atomicAdd(&counters->candidate_evaluations, 1ULL);
  atomicAdd(&counters->perturbation_candidate_evaluations, 1ULL);
  if (!PassesGeometricCandidateFilter(
          options, poses_texture, source_depth_maps_texture, source_images,
          col, row, candidate_depth, counters)) {
    return;
  }
  float candidate_cost = options.cost.invalid_cost;
  if (!ComputeAggregatedCost(
          reference_texture, source_images_texture, source_depth_maps_texture,
          poses_texture, selection_texture, dynamic_probabilities, nullptr, 0,
          0, source_images,
          options, col, row, candidate_depth, candidate_normal,
          &candidate_cost, counters)) {
    return;
  }
  if (candidate_cost < *best_cost) {
    *best_cost = candidate_cost;
    *best_depth = candidate_depth;
    *best_normal = candidate_normal;
    *updated = true;
  }
}

__global__ void PerturbationKernel(cudaTextureObject_t reference_texture,
                                   cudaTextureObject_t source_images_texture,
                                   cudaTextureObject_t source_depth_maps_texture,
                                   cudaTextureObject_t poses_texture,
                                   cudaTextureObject_t selection_texture,
                                   const float* dynamic_probabilities,
                                   const DeviceSourceImage* source_images,
                                   float* depth_map,
                                   float* normal_map,
                                   float* cost_map,
                                   DevicePatchMatchOptions options,
                                   std::uint32_t seed,
                                   const float* iteration_depth_radii,
                                   const float* iteration_normal_radii,
                                   int iteration_idx,
                                   int num_samples,
                                   DeviceCounters* counters) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  if (idx >= num_pixels) {
    return;
  }
  const int row = idx / options.width;
  const int col = idx - row * options.width;
  const float current_depth = depth_map[idx];
  if (!IsFinitePositive(current_depth) || !isfinite(cost_map[idx])) {
    return;
  }
  const float3 current_normal =
      Normalize3(make_float3(normal_map[idx * 3 + 0],
                             normal_map[idx * 3 + 1],
                             normal_map[idx * 3 + 2]),
                 make_float3(0.0F, 0.0F, 1.0F));
  float best_depth = current_depth;
  float3 best_normal = current_normal;
  float best_cost = cost_map[idx];
  bool updated = false;
  const float depth_radius = iteration_depth_radii[iteration_idx];
  const float normal_radius_degrees = iteration_normal_radii[iteration_idx];
  for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
    const float unit =
        2.0F * Random01(seed + static_cast<std::uint32_t>(idx) * 31u +
                        static_cast<std::uint32_t>(sample_idx) * 101u) -
        1.0F;
    const float candidate_depth =
        Clamp(current_depth + unit * depth_radius, options.depth_min,
              options.depth_max);
    const float3 candidate_normal = PerturbNormal(
        current_normal, normal_radius_degrees,
        seed + static_cast<std::uint32_t>(idx) * 131u +
            static_cast<std::uint32_t>(sample_idx) * 977u);
    // 与 COLMAP 坐标下降一致，分别评估联合、仅法线和仅深度三种随机候选。
    TryUpdatePerturbationCandidate(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, selection_texture, dynamic_probabilities, source_images,
        options, col, row,
        candidate_depth, candidate_normal, counters, &best_depth,
        &best_normal, &best_cost, &updated);
    TryUpdatePerturbationCandidate(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, selection_texture, dynamic_probabilities, source_images,
        options, col, row,
        current_depth, candidate_normal, counters, &best_depth, &best_normal,
        &best_cost, &updated);
    TryUpdatePerturbationCandidate(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, selection_texture, dynamic_probabilities, source_images,
        options, col, row,
        candidate_depth, current_normal, counters, &best_depth, &best_normal,
        &best_cost, &updated);
  }
  if (updated) {
    depth_map[idx] = best_depth;
    normal_map[idx * 3 + 0] = best_normal.x;
    normal_map[idx * 3 + 1] = best_normal.y;
    normal_map[idx * 3 + 2] = best_normal.z;
    cost_map[idx] = best_cost;
    atomicAdd(&counters->perturbation_updates, 1ULL);
    atomicAdd(&counters->total_abs_depth_update,
              static_cast<double>(fabsf(best_depth - current_depth)));
    const double normal_angle =
        static_cast<double>(AngleDegreesBetween(current_normal, best_normal));
    atomicAdd(&counters->total_normal_angle_degrees, normal_angle);
    AtomicMaxDouble(&counters->max_normal_angle_degrees, normal_angle);
  }
}

DeviceCamera MakeDeviceCamera(const Camera& camera) {
  const Eigen::Matrix3d k = camera.CalibrationMatrix();
  DeviceCamera device;
  device.fx = static_cast<float>(k(0, 0));
  device.fy = static_cast<float>(k(1, 1));
  device.cx = static_cast<float>(k(0, 2));
  device.cy = static_cast<float>(k(1, 2));
  return device;
}

DeviceSourceImage MakeDeviceSourceImage(const MvsImage& source_image) {
  DeviceSourceImage device;
  device.width = static_cast<int>(source_image.Width());
  device.height = static_cast<int>(source_image.Height());
  return device;
}

std::vector<float> PackPoseTextureData(
    const PatchMatchView& reference_view,
    const std::vector<PatchMatchView>& source_views) {
  std::vector<float> pose_data(static_cast<std::size_t>(kNumPoseParams) *
                                   source_views.size() * kNumPoseVariants,
                               0.0F);
  const Eigen::Matrix3d r_ref =
      reference_view.cam_from_world.q_cw.toRotationMatrix();
  const Eigen::Vector3d t_ref = reference_view.cam_from_world.t_cw;
  const float sweep_angles_degrees[kNumPoseVariants] = {0.0F, -7.5F, 7.5F};
  for (std::size_t source_idx = 0; source_idx < source_views.size();
       ++source_idx) {
    const PatchMatchView& source_view = source_views[source_idx];
    const Eigen::Matrix3d k_src = source_view.camera.CalibrationMatrix();
    const Eigen::Matrix3d k_inv_src = k_src.inverse();
    const Eigen::Matrix3d r_src =
        source_view.cam_from_world.q_cw.toRotationMatrix();
    const Eigen::Matrix3d r_rel = r_src * r_ref.transpose();
    const Eigen::Vector3d t_rel =
        source_view.cam_from_world.t_cw - r_rel * t_ref;
    const Eigen::Vector3d center_rel = -r_rel.transpose() * t_rel;
    const double baseline = center_rel.norm();
    const double focal_ratio =
        source_view.camera.MeanFocalLength() /
        std::max(1e-12, reference_view.camera.MeanFocalLength());
    const double scale_score =
        std::exp(-std::abs(std::log(std::max(1e-6, focal_ratio))));
    const double baseline_score =
        baseline <= 1e-8 ? 1.0 : std::min(1.0, baseline / 0.05);
    const float source_weight = static_cast<float>(
        std::clamp(0.5 * scale_score + 0.5 * baseline_score, 0.05, 1.0));
    const float selection_probability =
        source_views.size() <= 2
            ? 1.0F
            : std::clamp(0.35F + 0.65F * source_weight, 0.25F, 1.0F);

    const Eigen::Matrix<double, 3, 4> p =
        (Eigen::Matrix<double, 3, 4>() << k_src * r_rel, k_src * t_rel)
            .finished();
    const Eigen::Matrix3d inv_rotation = r_rel.transpose();
    const Eigen::Vector3d inv_translation = -inv_rotation * t_rel;
    Eigen::Matrix<double, 3, 4> inv_p;
    inv_p.block<3, 3>(0, 0) = inv_rotation * k_inv_src;
    inv_p.col(3) = inv_translation;

    for (int variant_idx = 0; variant_idx < kNumPoseVariants; ++variant_idx) {
      float* pose =
          pose_data.data() +
          (static_cast<std::size_t>(variant_idx) * source_views.size() +
           source_idx) *
              kNumPoseParams;
      pose[kPoseKOffset + 0] = static_cast<float>(k_src(0, 0));
      pose[kPoseKOffset + 1] = static_cast<float>(k_src(1, 1));
      pose[kPoseKOffset + 2] = static_cast<float>(k_src(0, 2));
      pose[kPoseKOffset + 3] = static_cast<float>(k_src(1, 2));
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          pose[kPoseROffset + row * 3 + col] =
              static_cast<float>(r_rel(row, col));
        }
        pose[kPoseTOffset + row] = static_cast<float>(t_rel(row));
        pose[kPoseCOffset + row] = static_cast<float>(center_rel(row));
      }
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
          pose[kPosePOffset + row * 4 + col] =
              static_cast<float>(p(row, col));
          pose[kPoseInvPOffset + row * 4 + col] =
              static_cast<float>(inv_p(row, col));
        }
      }
      constexpr float kPi = 3.14159265358979323846F;
      const float radians = sweep_angles_degrees[variant_idx] * kPi / 180.0F;
      pose[kPoseWeightOffset] = source_weight;
      pose[kPoseSelectionProbabilityOffset] = selection_probability;
      pose[kPoseSweepCosOffset] = std::cos(radians);
      pose[kPoseSweepSinOffset] = std::sin(radians);
    }
  }
  return pose_data;
}

Camera ScaleCameraForCudaPyramidLevel(const Camera& camera,
                                      double scale_x,
                                      double scale_y,
                                      std::size_t width,
                                      std::size_t height) {
  Camera scaled = camera;
  scaled.width = static_cast<std::uint32_t>(width);
  scaled.height = static_cast<std::uint32_t>(height);
  const CameraModelSpec& spec = GetCameraModelSpec(scaled.model_id);
  if (spec.focal_length_idxs.size() >= 2) {
    const std::size_t fx_idx = spec.focal_length_idxs[0];
    const std::size_t fy_idx = spec.focal_length_idxs[1];
    if (fx_idx < scaled.params.size()) {
      scaled.params[fx_idx] *= scale_x;
    }
    if (fy_idx < scaled.params.size()) {
      scaled.params[fy_idx] *= scale_y;
    }
  } else {
    for (const std::size_t idx : spec.focal_length_idxs) {
      if (idx < scaled.params.size()) {
        scaled.params[idx] *= 0.5 * (scale_x + scale_y);
      }
    }
  }
  if (!spec.principal_point_idxs.empty() &&
      spec.principal_point_idxs[0] < scaled.params.size()) {
    scaled.params[spec.principal_point_idxs[0]] *= scale_x;
  }
  if (spec.principal_point_idxs.size() >= 2 &&
      spec.principal_point_idxs[1] < scaled.params.size()) {
    scaled.params[spec.principal_point_idxs[1]] *= scale_y;
  }
  return scaled;
}

PatchMatchView ScaleViewForCudaPyramidLevel(const PatchMatchView& view,
                                            double scale_x,
                                            double scale_y,
                                            std::size_t width,
                                            std::size_t height) {
  PatchMatchView scaled = view;
  scaled.camera = ScaleCameraForCudaPyramidLevel(view.camera, scale_x, scale_y,
                                                width, height);
  return scaled;
}

std::size_t CudaPyramidLevelDivisor(std::size_t num_levels,
                                    std::size_t coarse_to_fine_idx) {
  std::size_t divisor = 1;
  for (std::size_t i = coarse_to_fine_idx + 1; i < num_levels; ++i) {
    divisor *= 2;
  }
  return divisor;
}

std::size_t CudaPyramidLevelSize(std::size_t fine_size,
                                 std::size_t divisor) {
  return std::max<std::size_t>(
      1, static_cast<std::size_t>(std::lround(
             static_cast<double>(fine_size) / static_cast<double>(divisor))));
}

MvsImage ResizeImageForCudaPyramidLevel(const MvsImage& image,
                                        std::size_t width,
                                        std::size_t height) {
  MvsImage resized = image;
  if (image.Width() == width && image.Height() == height) {
    return resized;
  }
  cv::resize(image.gray, resized.gray,
             cv::Size(static_cast<int>(width), static_cast<int>(height)), 0.0,
             0.0, cv::INTER_AREA);
  if (image.Width() > 0) {
    resized.scale =
        image.scale * static_cast<double>(width) /
        static_cast<double>(image.Width());
  }
  return resized;
}

DenseDepthMap ResizeDepthMapNearestForCuda(const DenseDepthMap& source,
                                           std::size_t width,
                                           std::size_t height) {
  DenseDepthMap resized(width, height, source.DepthMin(), source.DepthMax());
  if (source.Empty() || width == 0 || height == 0) {
    return resized;
  }
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t source_row = std::min(
        source.Height() - 1,
        static_cast<std::size_t>(
            std::floor((static_cast<double>(row) + 0.5) *
                       static_cast<double>(source.Height()) /
                       static_cast<double>(height))));
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t source_col = std::min(
          source.Width() - 1,
          static_cast<std::size_t>(
              std::floor((static_cast<double>(col) + 0.5) *
                         static_cast<double>(source.Width()) /
                         static_cast<double>(width))));
      resized.Set(row, col, source.Get(source_row, source_col));
    }
  }
  resized.RecomputeDepthRange();
  if (resized.DepthMin() <= 0.0F || resized.DepthMax() < resized.DepthMin()) {
    resized.SetDepthRange(source.DepthMin(), source.DepthMax());
  }
  return resized;
}

NormalMap ResizeNormalMapNearestForCuda(const NormalMap& source,
                                        std::size_t width,
                                        std::size_t height) {
  NormalMap resized(width, height);
  if (source.Empty() || width == 0 || height == 0) {
    return resized;
  }
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t source_row = std::min(
        source.Height() - 1,
        static_cast<std::size_t>(
            std::floor((static_cast<double>(row) + 0.5) *
                       static_cast<double>(source.Height()) /
                       static_cast<double>(height))));
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t source_col = std::min(
          source.Width() - 1,
          static_cast<std::size_t>(
              std::floor((static_cast<double>(col) + 0.5) *
                         static_cast<double>(source.Width()) /
                         static_cast<double>(width))));
      resized.SetNormalized(row, col, source.Get(source_row, source_col));
    }
  }
  return resized;
}

DenseCostMap ResizeCostMapNearestForCuda(const DenseCostMap& source,
                                         std::size_t width,
                                         std::size_t height) {
  DenseCostMap resized(width, height);
  if (source.Empty() || width == 0 || height == 0) {
    return resized;
  }
  for (std::size_t row = 0; row < height; ++row) {
    const std::size_t source_row = std::min(
        source.Height() - 1,
        static_cast<std::size_t>(
            std::floor((static_cast<double>(row) + 0.5) *
                       static_cast<double>(source.Height()) /
                       static_cast<double>(height))));
    for (std::size_t col = 0; col < width; ++col) {
      const std::size_t source_col = std::min(
          source.Width() - 1,
          static_cast<std::size_t>(
              std::floor((static_cast<double>(col) + 0.5) *
                         static_cast<double>(source.Width()) /
                         static_cast<double>(width))));
      resized.Set(row, col, source.Get(source_row, source_col));
    }
  }
  return resized;
}

PatchMatchMaps ResizePatchMatchMapsNearestForCuda(const PatchMatchMaps& source,
                                                  std::size_t width,
                                                  std::size_t height) {
  PatchMatchMaps resized;
  resized.depth_map = ResizeDepthMapNearestForCuda(source.depth_map, width,
                                                   height);
  resized.normal_map = ResizeNormalMapNearestForCuda(source.normal_map, width,
                                                     height);
  return resized;
}

std::vector<float> CopyMatToVector(const cv::Mat& image) {
  if (image.empty() || image.type() != CV_32FC1) {
    throw std::invalid_argument("CUDA PatchMatch requires CV_32FC1 images.");
  }
  std::vector<float> values(static_cast<std::size_t>(image.rows) *
                            static_cast<std::size_t>(image.cols));
  for (int row = 0; row < image.rows; ++row) {
    const float* src = image.ptr<float>(row);
    std::copy(src, src + image.cols,
              values.begin() + static_cast<std::size_t>(row) * image.cols);
  }
  return values;
}

std::vector<float> PackSourceImages(const std::vector<MvsImage>& images,
                                    int max_width,
                                    int max_height) {
  std::vector<float> packed(static_cast<std::size_t>(images.size()) *
                                max_width * max_height,
                            0.0F);
  for (std::size_t image_idx = 0; image_idx < images.size(); ++image_idx) {
    if (images[image_idx].gray.empty() ||
        images[image_idx].gray.type() != CV_32FC1) {
      throw std::invalid_argument("CUDA PatchMatch requires CV_32FC1 images.");
    }
    float* layer = packed.data() + image_idx * max_width * max_height;
    for (int row = 0; row < images[image_idx].gray.rows; ++row) {
      const float* src = images[image_idx].gray.ptr<float>(row);
      std::copy(src, src + images[image_idx].gray.cols,
                layer + row * max_width);
    }
  }
  return packed;
}

std::vector<float> PackSourceDepthMaps(
    const std::vector<DenseDepthMap>& depth_maps,
    int max_width,
    int max_height) {
  std::vector<float> packed(static_cast<std::size_t>(depth_maps.size()) *
                                max_width * max_height,
                            0.0F);
  for (std::size_t map_idx = 0; map_idx < depth_maps.size(); ++map_idx) {
    float* layer = packed.data() + map_idx * max_width * max_height;
    for (std::size_t row = 0; row < depth_maps[map_idx].Height(); ++row) {
      for (std::size_t col = 0; col < depth_maps[map_idx].Width(); ++col) {
        layer[row * max_width + col] = depth_maps[map_idx].Get(row, col);
      }
    }
  }
  return packed;
}

std::vector<float> PackSelectionProbabilityMaps(
    const std::vector<DenseCostMap>& probability_maps,
    int width,
    int height) {
  std::vector<float> packed(static_cast<std::size_t>(probability_maps.size()) *
                                width * height,
                            0.0F);
  for (std::size_t map_idx = 0; map_idx < probability_maps.size();
       ++map_idx) {
    float* layer = packed.data() + map_idx * width * height;
    for (std::size_t row = 0; row < probability_maps[map_idx].Height();
         ++row) {
      for (std::size_t col = 0; col < probability_maps[map_idx].Width();
           ++col) {
        layer[row * width + col] =
            std::clamp(probability_maps[map_idx].Get(row, col), 0.0F, 1.0F);
      }
    }
  }
  return packed;
}

std::vector<float> CopyDepthMapToVector(const DenseDepthMap& depth_map) {
  std::vector<float> values(depth_map.NumPixels(), 0.0F);
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      values[row * depth_map.Width() + col] = depth_map.Get(row, col);
    }
  }
  return values;
}

std::vector<float> CopyNormalizedNormalMapToVector(
    const NormalMap& normal_map,
    std::size_t* num_normalizations) {
  if (num_normalizations != nullptr) {
    *num_normalizations = 0;
  }
  std::vector<float> values(normal_map.NumPixels() * 3, 0.0F);
  for (std::size_t row = 0; row < normal_map.Height(); ++row) {
    for (std::size_t col = 0; col < normal_map.Width(); ++col) {
      const std::size_t idx = row * normal_map.Width() + col;
      Eigen::Vector3f normal = normal_map.Get(row, col);
      if (!normal.allFinite() || normal.squaredNorm() <= 0.0F) {
        normal = Eigen::Vector3f::Zero();
      } else {
        normal.normalize();
        if (num_normalizations != nullptr) {
          ++(*num_normalizations);
        }
      }
      values[idx * 3 + 0] = normal.x();
      values[idx * 3 + 1] = normal.y();
      values[idx * 3 + 2] = normal.z();
    }
  }
  return values;
}

void ValidateRequest(const CudaPatchMatchRequest& request) {
  if (request.reference_view == nullptr || request.source_views == nullptr ||
      request.reference_image == nullptr || request.source_images == nullptr) {
    throw std::invalid_argument("CUDA PatchMatch request is incomplete.");
  }
  if (request.reference_image->Empty()) {
    throw std::invalid_argument("CUDA PatchMatch reference image is empty.");
  }
  if (request.source_views->size() != request.source_images->size()) {
    throw std::invalid_argument(
        "CUDA PatchMatch source views and images must match.");
  }
  if (request.source_depth_maps != nullptr &&
      request.source_depth_maps->size() != request.source_views->size()) {
    throw std::invalid_argument(
        "CUDA PatchMatch source depth maps must match source views.");
  }
  if (request.source_selection_probability_maps != nullptr &&
      request.source_selection_probability_maps->size() !=
          request.source_views->size()) {
    throw std::invalid_argument(
        "CUDA PatchMatch selection probability maps must match source views.");
  }
  if (request.source_views->empty()) {
    throw std::invalid_argument(
        "CUDA PatchMatch requires at least one source image.");
  }
  if (request.source_views->size() > kMaxCudaSourceImages) {
    throw std::invalid_argument(
        "CUDA PatchMatch currently supports at most 32 source images.");
  }
  if (!std::isfinite(request.depth_min) || !std::isfinite(request.depth_max) ||
      request.depth_min <= 0.0F || request.depth_max < request.depth_min) {
    throw std::invalid_argument("CUDA PatchMatch depth range is invalid.");
  }
  if (request.options.cost.window_radius < 0 ||
      request.options.cost.window_step <= 0 ||
      !std::isfinite(request.options.cost.sigma_spatial) ||
      !std::isfinite(request.options.cost.sigma_color) ||
      request.options.cost.sigma_color <= 0.0F ||
      !std::isfinite(
          request.options.cost.geometric_max_reprojection_error) ||
      request.options.cost.geometric_max_reprojection_error <= 0.0F ||
      request.options.cost.min_num_valid_sources == 0 ||
      request.options.perturbation.num_samples_per_pixel == 0) {
    throw std::invalid_argument("CUDA PatchMatch options are invalid.");
  }
  const PatchMatchSourceSelectionOptions& source_selection =
      request.options.source_selection;
  if (!std::isfinite(source_selection.ncc_sigma) ||
      source_selection.ncc_sigma <= 0.0F ||
      !std::isfinite(source_selection.previous_probability_weight) ||
      source_selection.previous_probability_weight < 0.0F ||
      source_selection.previous_probability_weight > 1.0F ||
      !std::isfinite(source_selection.min_probability) ||
      !std::isfinite(source_selection.max_probability) ||
      source_selection.min_probability < 0.0F ||
      source_selection.max_probability > 1.0F ||
      source_selection.min_probability > source_selection.max_probability) {
    throw std::invalid_argument(
        "CUDA PatchMatch source selection options are invalid.");
  }
  if (request.initial_maps != nullptr &&
      (request.initial_maps->depth_map.Width() !=
           request.reference_image->Width() ||
       request.initial_maps->depth_map.Height() !=
           request.reference_image->Height() ||
       request.initial_maps->normal_map.Width() !=
           request.reference_image->Width() ||
       request.initial_maps->normal_map.Height() !=
           request.reference_image->Height())) {
    throw std::invalid_argument(
        "CUDA PatchMatch initial maps must match the reference image.");
  }
  if (request.source_selection_probability_maps != nullptr) {
    for (const DenseCostMap& map :
         *request.source_selection_probability_maps) {
      if (map.Width() != request.reference_image->Width() ||
          map.Height() != request.reference_image->Height()) {
        throw std::invalid_argument(
            "CUDA PatchMatch selection probability maps must match the "
            "reference image.");
      }
    }
  }
}

void SetUnavailableReason(std::string* reason, const std::string& message) {
  if (reason != nullptr) {
    *reason = message;
  }
}

void AccumulateCudaSummary(PatchMatchStereoSummary* accumulated,
                           const PatchMatchStereoSummary& level) {
  accumulated->num_iterations += level.num_iterations;
  accumulated->num_forward_sweeps += level.num_forward_sweeps;
  accumulated->num_backward_sweeps += level.num_backward_sweeps;
  accumulated->num_top_down_sweeps += level.num_top_down_sweeps;
  accumulated->num_bottom_up_sweeps += level.num_bottom_up_sweeps;
  accumulated->num_checkerboard_passes += level.num_checkerboard_passes;
  accumulated->num_perturbation_passes += level.num_perturbation_passes;
  accumulated->num_candidate_evaluations += level.num_candidate_evaluations;
  accumulated->num_plane_propagation_candidates +=
      level.num_plane_propagation_candidates;
  accumulated->num_perturbation_candidate_evaluations +=
      level.num_perturbation_candidate_evaluations;
  accumulated->num_propagation_updates += level.num_propagation_updates;
  accumulated->num_perturbation_updates += level.num_perturbation_updates;
  accumulated->num_source_selection_tests += level.num_source_selection_tests;
  accumulated->num_source_selection_acceptances +=
      level.num_source_selection_acceptances;
  accumulated->num_source_selection_rejections +=
      level.num_source_selection_rejections;
  accumulated->num_source_selection_probability_updates +=
      level.num_source_selection_probability_updates;
  accumulated->perturbation_total_abs_depth_update +=
      level.perturbation_total_abs_depth_update;
  accumulated->perturbation_max_abs_depth_update =
      std::max(accumulated->perturbation_max_abs_depth_update,
               level.perturbation_max_abs_depth_update);
  accumulated->perturbation_total_normal_angle_degrees +=
      level.perturbation_total_normal_angle_degrees;
  accumulated->perturbation_max_normal_angle_degrees =
      std::max(accumulated->perturbation_max_normal_angle_degrees,
               level.perturbation_max_normal_angle_degrees);
  accumulated->num_perturbation_geometric_candidate_rejections +=
      level.num_perturbation_geometric_candidate_rejections;
  accumulated->num_perturbation_geometric_candidate_acceptances +=
      level.num_perturbation_geometric_candidate_acceptances;
  accumulated->num_normal_refinement_pixels =
      level.num_normal_refinement_pixels;
  accumulated->num_normal_refined_pixels = level.num_normal_refined_pixels;
  accumulated->num_normal_refinement_skipped_pixels =
      level.num_normal_refinement_skipped_pixels;
  accumulated->cuda_upload_time_ms += level.cuda_upload_time_ms;
  accumulated->cuda_initialization_time_ms +=
      level.cuda_initialization_time_ms;
  accumulated->cuda_evaluate_cost_time_ms +=
      level.cuda_evaluate_cost_time_ms;
  accumulated->cuda_propagation_time_ms += level.cuda_propagation_time_ms;
  accumulated->cuda_perturbation_time_ms += level.cuda_perturbation_time_ms;
  accumulated->cuda_download_time_ms += level.cuda_download_time_ms;
  accumulated->cuda_total_time_ms += level.cuda_total_time_ms;
  accumulated->cuda_wrapper_total_time_ms += level.cuda_wrapper_total_time_ms;
  accumulated->cuda_request_validation_time_ms +=
      level.cuda_request_validation_time_ms;
  accumulated->cuda_pyramid_resize_time_ms +=
      level.cuda_pyramid_resize_time_ms;
  accumulated->cuda_host_pack_time_ms += level.cuda_host_pack_time_ms;
  accumulated->cuda_single_scale_dispatch_time_ms +=
      level.cuda_single_scale_dispatch_time_ms;
  accumulated->cuda_result_materialization_time_ms +=
      level.cuda_result_materialization_time_ms;
  accumulated->cuda_output_to_vector_time_ms +=
      level.cuda_output_to_vector_time_ms;
  accumulated->cuda_dense_map_fill_time_ms +=
      level.cuda_dense_map_fill_time_ms;
  accumulated->cuda_normal_refinement_time_ms +=
      level.cuda_normal_refinement_time_ms;
  accumulated->cuda_valid_cost_count_time_ms +=
      level.cuda_valid_cost_count_time_ms;
  accumulated->cuda_host_initial_normal_upload_normalizations +=
      level.cuda_host_initial_normal_upload_normalizations;
  accumulated->cuda_host_output_normalizations +=
      level.cuda_host_output_normalizations;
  accumulated->cuda_device_count_queries += level.cuda_device_count_queries;
  accumulated->cuda_device_count_cache_hits +=
      level.cuda_device_count_cache_hits;
  accumulated->cuda_shared_reference_tile_pixels +=
      level.cuda_shared_reference_tile_pixels;
  accumulated->cuda_workspace_texture_allocations +=
      level.cuda_workspace_texture_allocations;
  accumulated->cuda_workspace_texture_reuses +=
      level.cuda_workspace_texture_reuses;
  accumulated->cuda_workspace_buffer_allocations +=
      level.cuda_workspace_buffer_allocations;
  accumulated->cuda_workspace_buffer_reuses +=
      level.cuda_workspace_buffer_reuses;
  accumulated->cuda_iteration_counter_snapshots +=
      level.cuda_iteration_counter_snapshots;
  accumulated->cuda_iteration_counter_readbacks +=
      level.cuda_iteration_counter_readbacks;
  accumulated->cuda_packed_output_pixels += level.cuda_packed_output_pixels;
  accumulated->cuda_output_map_readbacks += level.cuda_output_map_readbacks;
  accumulated->cuda_async_host_uploads += level.cuda_async_host_uploads;
  accumulated->cuda_async_texture_uploads += level.cuda_async_texture_uploads;
  accumulated->cuda_async_counter_readbacks +=
      level.cuda_async_counter_readbacks;
  accumulated->cuda_async_output_readbacks += level.cuda_async_output_readbacks;
  accumulated->cuda_pinned_host_upload_bytes =
      std::max(accumulated->cuda_pinned_host_upload_bytes,
               level.cuda_pinned_host_upload_bytes);
  accumulated->cuda_pinned_texture_upload_bytes =
      std::max(accumulated->cuda_pinned_texture_upload_bytes,
               level.cuda_pinned_texture_upload_bytes);
  accumulated->cuda_pinned_host_staging_bytes =
      std::max(accumulated->cuda_pinned_host_staging_bytes,
               level.cuda_pinned_host_staging_bytes);
  accumulated->cuda_stream_kernel_launches +=
      level.cuda_stream_kernel_launches;
  accumulated->update_ratios.insert(accumulated->update_ratios.end(),
                                    level.update_ratios.begin(),
                                    level.update_ratios.end());
  accumulated->depth_radii.insert(accumulated->depth_radii.end(),
                                  level.depth_radii.begin(),
                                  level.depth_radii.end());
  accumulated->normal_radii_degrees.insert(
      accumulated->normal_radii_degrees.end(),
      level.normal_radii_degrees.begin(), level.normal_radii_degrees.end());
  accumulated->iteration_pyramid_levels.insert(
      accumulated->iteration_pyramid_levels.end(),
      level.iteration_pyramid_levels.begin(),
      level.iteration_pyramid_levels.end());
  accumulated->iteration_indices.insert(
      accumulated->iteration_indices.end(), level.iteration_indices.begin(),
      level.iteration_indices.end());
  accumulated->iteration_candidate_evaluations.insert(
      accumulated->iteration_candidate_evaluations.end(),
      level.iteration_candidate_evaluations.begin(),
      level.iteration_candidate_evaluations.end());
  accumulated->iteration_propagation_updates.insert(
      accumulated->iteration_propagation_updates.end(),
      level.iteration_propagation_updates.begin(),
      level.iteration_propagation_updates.end());
  accumulated->iteration_perturbation_updates.insert(
      accumulated->iteration_perturbation_updates.end(),
      level.iteration_perturbation_updates.begin(),
      level.iteration_perturbation_updates.end());
  accumulated->iteration_source_selection_tests.insert(
      accumulated->iteration_source_selection_tests.end(),
      level.iteration_source_selection_tests.begin(),
      level.iteration_source_selection_tests.end());
  accumulated->iteration_source_selection_acceptances.insert(
      accumulated->iteration_source_selection_acceptances.end(),
      level.iteration_source_selection_acceptances.begin(),
      level.iteration_source_selection_acceptances.end());
  accumulated->iteration_source_selection_rejections.insert(
      accumulated->iteration_source_selection_rejections.end(),
      level.iteration_source_selection_rejections.begin(),
      level.iteration_source_selection_rejections.end());
  accumulated->iteration_source_selection_probability_means.insert(
      accumulated->iteration_source_selection_probability_means.end(),
      level.iteration_source_selection_probability_means.begin(),
      level.iteration_source_selection_probability_means.end());
  accumulated->final_source_selection_probability_means =
      level.final_source_selection_probability_means;
}

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereoSingleScale(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason,
    CudaPatchMatchWorkspace* workspace) {
  const SteadyClock::time_point wrapper_start = SteadyClock::now();
  if (unavailable_reason != nullptr) {
    unavailable_reason->clear();
  }

  const SteadyClock::time_point validation_start = SteadyClock::now();
  const CachedCudaDeviceCount device_count = GetCachedCudaDeviceCount();
  const std::size_t cuda_device_count_queries =
      device_count.queried_runtime ? 1U : 0U;
  const std::size_t cuda_device_count_cache_hits =
      device_count.queried_runtime ? 0U : 1U;
  if (device_count.status != cudaSuccess || device_count.count <= 0) {
    SetUnavailableReason(unavailable_reason,
                         "CUDA PatchMatch backend found no CUDA devices.");
    return std::nullopt;
  }

  if (request.reference_view == nullptr && request.reference_image == nullptr) {
    return std::nullopt;
  }

  try {
    if (request.options.cuda_gpu_index >= 0) {
      if (request.options.cuda_gpu_index >= device_count.count) {
        SetUnavailableReason(unavailable_reason,
                             "Requested CUDA PatchMatch GPU index is invalid.");
        return std::nullopt;
      }
      CudaCheck(cudaSetDevice(request.options.cuda_gpu_index));
    }

    ValidateRequest(request);
    const double cuda_request_validation_time_ms =
        ElapsedMilliseconds(validation_start, SteadyClock::now());
    CudaPatchMatchWorkspace local_workspace;
    CudaPatchMatchWorkspaceStorage& workspace_storage =
        (workspace == nullptr ? local_workspace.storage_ : workspace->storage_);
    cudaStream_t stream =
        workspace == nullptr ? local_workspace.Stream() : workspace->Stream();
    const SteadyClock::time_point total_start = SteadyClock::now();
    const SteadyClock::time_point upload_start = SteadyClock::now();
    double cuda_upload_time_ms = 0.0;
    double cuda_initialization_time_ms = 0.0;
    double cuda_evaluate_cost_time_ms = 0.0;
    double cuda_propagation_time_ms = 0.0;
    double cuda_perturbation_time_ms = 0.0;
    double cuda_download_time_ms = 0.0;
    double cuda_host_pack_time_ms = 0.0;
    std::size_t cuda_shared_reference_tile_pixels = 0;
    std::size_t cuda_workspace_texture_allocations = 0;
    std::size_t cuda_workspace_texture_reuses = 0;
    std::size_t cuda_workspace_buffer_allocations = 0;
    std::size_t cuda_workspace_buffer_reuses = 0;
    std::size_t cuda_packed_output_pixels = 0;
    std::size_t cuda_output_map_readbacks = 0;
    std::size_t cuda_async_host_uploads = 0;
    std::size_t cuda_async_texture_uploads = 0;
    std::size_t cuda_async_counter_readbacks = 0;
    std::size_t cuda_async_output_readbacks = 0;
    std::size_t cuda_pinned_host_upload_bytes = 0;
    std::size_t cuda_pinned_texture_upload_bytes = 0;
    std::size_t cuda_pinned_host_staging_bytes = 0;
    std::size_t cuda_stream_kernel_launches = 0;
    std::size_t cuda_host_initial_normal_upload_normalizations = 0;
    std::size_t cuda_host_output_normalizations = 0;
    const int width = static_cast<int>(request.reference_image->Width());
    const int height = static_cast<int>(request.reference_image->Height());
    int max_source_width = 0;
    int max_source_height = 0;
    for (const MvsImage& source_image : *request.source_images) {
      max_source_width =
          std::max(max_source_width, static_cast<int>(source_image.Width()));
      max_source_height =
          std::max(max_source_height, static_cast<int>(source_image.Height()));
    }

    DevicePatchMatchOptions device_options;
    device_options.width = width;
    device_options.height = height;
    device_options.max_source_width = max_source_width;
    device_options.max_source_height = max_source_height;
    device_options.num_sources =
        static_cast<int>(request.source_images->size());
    device_options.num_pose_variants = kNumPoseVariants;
    device_options.depth_min = request.depth_min;
    device_options.depth_max = request.depth_max;
    device_options.reference_camera = MakeDeviceCamera(request.reference_view->camera);
    device_options.dynamic_selection_probability =
        request.options.source_selection.dynamic_probability;
    device_options.directional_hmm =
        request.options.source_selection.directional_hmm;
    device_options.num_source_samples = static_cast<int>(
        request.options.source_selection.num_samples);
    device_options.ncc_sigma = request.options.source_selection.ncc_sigma;
    device_options.previous_selection_probability_weight =
        request.options.source_selection.previous_probability_weight;
    device_options.min_selection_probability =
        request.options.source_selection.min_probability;
    device_options.max_selection_probability =
        request.options.source_selection.max_probability;
    device_options.cost.window_radius = request.options.cost.window_radius;
    device_options.cost.window_step = request.options.cost.window_step;
    device_options.cost.bilateral_weighting =
        request.options.cost.bilateral_weighting;
    device_options.cost.sigma_spatial =
        request.options.cost.sigma_spatial > 0.0F
            ? request.options.cost.sigma_spatial
            : static_cast<float>(std::max(1, request.options.cost.window_radius));
    device_options.cost.sigma_color = request.options.cost.sigma_color;
    device_options.cost.invalid_cost = request.options.cost.invalid_cost;
    device_options.cost.min_variance = request.options.cost.min_variance;
    device_options.cost.min_num_valid_sources =
        static_cast<int>(request.options.cost.min_num_valid_sources);
    device_options.cost.max_num_aggregated_costs =
        static_cast<int>(request.options.cost.max_num_aggregated_costs);
    device_options.cost.geometric_consistency_weight =
        request.options.cost.geometric_consistency_weight;
    device_options.cost.geometric_max_reprojection_error =
        request.options.cost.geometric_max_reprojection_error;
    device_options.cost.geometric_max_relative_depth_error =
        request.options.cost.geometric_max_relative_depth_error;
    device_options.cost.geometric_reject_occluded_candidates =
        request.options.cost.geometric_reject_occluded_candidates;
    device_options.cost.geometric_reject_depth_conflicts =
        request.options.cost.geometric_reject_depth_conflicts;

    const SteadyClock::time_point host_pack_start = SteadyClock::now();
    std::vector<DeviceSourceImage> source_images_info;
    source_images_info.reserve(request.source_images->size());
    for (std::size_t i = 0; i < request.source_views->size(); ++i) {
      source_images_info.push_back(
          MakeDeviceSourceImage(request.source_images->at(i)));
    }
    const std::vector<float> pose_texture_data =
        PackPoseTextureData(*request.reference_view, *request.source_views);

    const std::vector<float> reference_image =
        CopyMatToVector(request.reference_image->gray);
    const std::vector<float> source_images =
        PackSourceImages(*request.source_images, max_source_width,
                         max_source_height);
    const bool use_source_depth_maps = request.source_depth_maps != nullptr &&
                                       !request.source_depth_maps->empty() &&
                                       request.options.cost
                                               .geometric_consistency_weight >
                                           0.0F;
    const bool use_selection_probability_maps =
        request.source_selection_probability_maps != nullptr &&
        !request.source_selection_probability_maps->empty();
    const std::vector<float> source_depth_maps =
        use_source_depth_maps
            ? PackSourceDepthMaps(*request.source_depth_maps, max_source_width,
                                  max_source_height)
            : std::vector<float>{};
    const std::vector<float> selection_probability_maps =
        use_selection_probability_maps
            ? PackSelectionProbabilityMaps(
                  *request.source_selection_probability_maps, width, height)
            : std::vector<float>{};
    std::vector<float> dynamic_selection_probabilities;
    if (request.options.source_selection.dynamic_probability) {
      if (use_selection_probability_maps) {
        dynamic_selection_probabilities = selection_probability_maps;
      } else {
        dynamic_selection_probabilities.resize(
            static_cast<std::size_t>(device_options.num_sources) * width *
            height,
            std::clamp(0.5F,
                       request.options.source_selection.min_probability,
                       request.options.source_selection.max_probability));
      }
    }
    cuda_host_pack_time_ms =
        ElapsedMilliseconds(host_pack_start, SteadyClock::now());

    const std::size_t num_pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const auto record_texture_upload = [&](bool allocated) {
      if (allocated) {
        ++cuda_workspace_texture_allocations;
      } else {
        ++cuda_workspace_texture_reuses;
      }
    };
    const auto record_buffer_resize = [&](bool allocated) {
      if (allocated) {
        ++cuda_workspace_buffer_allocations;
      } else {
        ++cuda_workspace_buffer_reuses;
      }
    };
    const auto upload_texture =
        [&](FloatLayeredTexture& texture, PinnedHostBuffer<float>& staging,
            int texture_width, int texture_height, int texture_depth,
            const std::vector<float>& values,
            cudaTextureFilterMode filter_mode = cudaFilterModeLinear) {
          staging.Resize(values.size());
          std::copy(values.begin(), values.end(), staging.data());
          ++cuda_async_texture_uploads;
          cuda_pinned_texture_upload_bytes += sizeof(float) * staging.size();
          return texture.CreateAsync(texture_width, texture_height,
                                     texture_depth, staging.data(), stream,
                                     filter_mode);
        };
    record_texture_upload(upload_texture(
        workspace_storage.reference_texture,
        workspace_storage.host_reference_texture_upload, width, height, 1,
        reference_image));
    record_texture_upload(upload_texture(
        workspace_storage.source_images_texture,
        workspace_storage.host_source_images_texture_upload, max_source_width,
        max_source_height, device_options.num_sources, source_images));
    record_texture_upload(upload_texture(
        workspace_storage.poses_texture,
        workspace_storage.host_poses_texture_upload, kNumPoseParams,
        device_options.num_sources, kNumPoseVariants, pose_texture_data,
        cudaFilterModePoint));
    if (use_source_depth_maps) {
      record_texture_upload(upload_texture(
          workspace_storage.source_depth_maps_texture,
          workspace_storage.host_source_depth_maps_texture_upload,
          max_source_width, max_source_height, device_options.num_sources,
          source_depth_maps));
    }
    if (use_selection_probability_maps) {
      record_texture_upload(upload_texture(
          workspace_storage.selection_probability_texture,
          workspace_storage.host_selection_probability_texture_upload, width,
          height, device_options.num_sources, selection_probability_maps));
    }
    FloatLayeredTexture& reference_texture =
        workspace_storage.reference_texture;
    FloatLayeredTexture& source_images_texture =
        workspace_storage.source_images_texture;
    FloatLayeredTexture& poses_texture = workspace_storage.poses_texture;
    FloatLayeredTexture& source_depth_maps_texture =
        workspace_storage.source_depth_maps_texture;
    FloatLayeredTexture& selection_probability_texture =
        workspace_storage.selection_probability_texture;
    DeviceBuffer<DeviceSourceImage>& d_source_images_info =
        workspace_storage.source_images_info;
    DeviceBuffer<float>& d_depth_map = workspace_storage.depth_map;
    DeviceBuffer<float>& d_normal_map = workspace_storage.normal_map;
    DeviceBuffer<float>& d_cost_map = workspace_storage.cost_map;
    DeviceBuffer<float>& d_dynamic_selection_probability_map =
        workspace_storage.dynamic_selection_probability_map;
    DeviceBuffer<float>& d_source_selection_emission_map =
        workspace_storage.source_selection_emission_map;
    DeviceBuffer<float>& d_source_selection_backward_message_map =
        workspace_storage.source_selection_backward_message_map;
    DeviceBuffer<DeviceCounters>& d_counters = workspace_storage.counters;
    DeviceBuffer<DeviceCounters>& d_iteration_counters =
        workspace_storage.iteration_counters;
    DeviceBuffer<float>& d_perturbation_radius_state =
        workspace_storage.perturbation_radius_state;
    DeviceBuffer<float>& d_iteration_depth_radii =
        workspace_storage.iteration_depth_radii;
    DeviceBuffer<float>& d_iteration_normal_radii =
        workspace_storage.iteration_normal_radii;
    DeviceBuffer<DeviceOutputPixel>& d_output_pixels =
        workspace_storage.output_pixels;
    PinnedHostBuffer<DeviceCounters>& h_iteration_counters =
        workspace_storage.host_iteration_counters;
    PinnedHostBuffer<float>& h_iteration_depth_radii =
        workspace_storage.host_iteration_depth_radii;
    PinnedHostBuffer<float>& h_iteration_normal_radii =
        workspace_storage.host_iteration_normal_radii;
    PinnedHostBuffer<DeviceOutputPixel>& h_output_pixels =
        workspace_storage.host_output_pixels;
    PinnedHostBuffer<float>& h_dynamic_selection_probability_map =
        workspace_storage.host_dynamic_selection_probability_map;
    PinnedHostBuffer<float>& h_depth_upload =
        workspace_storage.host_depth_upload;
    PinnedHostBuffer<float>& h_normal_upload =
        workspace_storage.host_normal_upload;
    record_buffer_resize(d_source_images_info.Resize(source_images_info.size()));
    record_buffer_resize(d_depth_map.Resize(num_pixels));
    record_buffer_resize(d_normal_map.Resize(num_pixels * 3));
    record_buffer_resize(d_cost_map.Resize(num_pixels));
    if (request.options.source_selection.dynamic_probability) {
      record_buffer_resize(d_dynamic_selection_probability_map.Resize(
          dynamic_selection_probabilities.size()));
      if (request.options.source_selection.directional_hmm) {
        record_buffer_resize(d_source_selection_emission_map.Resize(
            dynamic_selection_probabilities.size()));
        record_buffer_resize(d_source_selection_backward_message_map.Resize(
            dynamic_selection_probabilities.size()));
      }
      h_dynamic_selection_probability_map.Resize(
          dynamic_selection_probabilities.size());
    }
    record_buffer_resize(d_counters.Resize(1));
    record_buffer_resize(d_output_pixels.Resize(num_pixels));
    record_buffer_resize(d_perturbation_radius_state.Resize(2));
    if (request.options.num_iterations > 0) {
      record_buffer_resize(
          d_iteration_counters.Resize(request.options.num_iterations));
      record_buffer_resize(
          d_iteration_depth_radii.Resize(request.options.num_iterations));
      record_buffer_resize(
          d_iteration_normal_radii.Resize(request.options.num_iterations));
      h_iteration_counters.Resize(request.options.num_iterations);
      h_iteration_depth_radii.Resize(request.options.num_iterations);
      h_iteration_normal_radii.Resize(request.options.num_iterations);
    }
    h_output_pixels.Resize(num_pixels);
    cuda_pinned_host_staging_bytes +=
        sizeof(DeviceCounters) * h_iteration_counters.size();
    cuda_pinned_host_staging_bytes +=
        sizeof(float) * h_iteration_depth_radii.size();
    cuda_pinned_host_staging_bytes +=
        sizeof(float) * h_iteration_normal_radii.size();
    cuda_pinned_host_staging_bytes +=
        sizeof(DeviceOutputPixel) * h_output_pixels.size();
    cuda_pinned_host_staging_bytes +=
        sizeof(float) * h_dynamic_selection_probability_map.size();
    d_source_images_info.CopyFromHost(source_images_info);
    if (request.options.source_selection.dynamic_probability) {
      std::copy(dynamic_selection_probabilities.begin(),
                dynamic_selection_probabilities.end(),
                h_dynamic_selection_probability_map.data());
      d_dynamic_selection_probability_map.CopyFromHostAsync(
          h_dynamic_selection_probability_map.data(), stream);
      ++cuda_async_host_uploads;
      cuda_pinned_host_upload_bytes +=
          sizeof(float) * h_dynamic_selection_probability_map.size();
    }
    Eigen::Vector3f default_normal =
        request.options.initialization.default_normal.squaredNorm() > 0.0F
            ? request.options.initialization.default_normal.normalized()
            : Eigen::Vector3f(0.0F, 0.0F, -1.0F);
    const float3 cuda_default_normal =
        make_float3(default_normal.x(), default_normal.y(), default_normal.z());
    const int blocks =
        static_cast<int>((num_pixels + kThreadsPerBlock - 1) /
                         kThreadsPerBlock);
    float* dynamic_selection_probabilities_ptr =
        request.options.source_selection.dynamic_probability
            ? d_dynamic_selection_probability_map.get()
            : nullptr;

    if (request.initial_maps != nullptr) {
      h_depth_upload.Resize(num_pixels);
      h_normal_upload.Resize(num_pixels * 3);
      const std::vector<float> depth_upload =
          CopyDepthMapToVector(request.initial_maps->depth_map);
      const std::vector<float> normal_upload =
          CopyNormalizedNormalMapToVector(
              request.initial_maps->normal_map,
              &cuda_host_initial_normal_upload_normalizations);
      std::copy(depth_upload.begin(), depth_upload.end(),
                h_depth_upload.data());
      std::copy(normal_upload.begin(), normal_upload.end(),
                h_normal_upload.data());
      d_depth_map.CopyFromHostAsync(h_depth_upload.data(), stream);
      d_normal_map.CopyFromHostAsync(h_normal_upload.data(), stream);
      cuda_async_host_uploads += 2;
      cuda_pinned_host_upload_bytes += sizeof(float) * h_depth_upload.size();
      cuda_pinned_host_upload_bytes += sizeof(float) * h_normal_upload.size();
      cuda_upload_time_ms = ElapsedMilliseconds(upload_start,
                                                SteadyClock::now());
    } else {
      cuda_upload_time_ms = ElapsedMilliseconds(upload_start,
                                                SteadyClock::now());
      CudaEventTimer init_timer;
      init_timer.Start(stream);
      InitHypothesesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
          d_depth_map.get(), d_normal_map.get(), device_options,
          request.options.initialization.random_seed, cuda_default_normal,
          request.options.initialization.randomize_initial_normals,
          request.options.initialization.initial_normal_hemisphere_degrees);
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
      cuda_initialization_time_ms += init_timer.StopMilliseconds(stream);
    }
    CudaEventTimer evaluate_timer;
    evaluate_timer.Start(stream);
    EvaluateCostKernel<<<blocks, kThreadsPerBlock,
                         kThreadsPerBlock * sizeof(float), stream>>>(
        reference_texture.get(), source_images_texture.get(),
        use_source_depth_maps ? source_depth_maps_texture.get() : 0,
        poses_texture.get(),
        use_selection_probability_maps ? selection_probability_texture.get()
                                       : 0,
        dynamic_selection_probabilities_ptr,
        d_source_images_info.get(), d_depth_map.get(), d_normal_map.get(),
        d_cost_map.get(), device_options);
    ++cuda_stream_kernel_launches;
    CudaCheck(cudaGetLastError());
    cuda_evaluate_cost_time_ms += evaluate_timer.StopMilliseconds(stream);
    cuda_shared_reference_tile_pixels += num_pixels;

    float depth_radius = (request.depth_max - request.depth_min) *
                         request.options.initial_depth_perturbation_factor;
    float normal_radius = request.options.perturbation.normal_radius_degrees;
    InitPerturbationRadiusStateKernel<<<1, 1, 0, stream>>>(
        d_perturbation_radius_state.get(), depth_radius, normal_radius);
    ++cuda_stream_kernel_launches;
    CudaCheck(cudaGetLastError());
    std::vector<DeviceCounters> iteration_counters;
    std::vector<CudaEventTimer> propagation_timers;
    std::vector<CudaEventTimer> perturbation_timers;
    std::vector<float> iteration_depth_radii;
    std::vector<float> iteration_normal_radii;
    iteration_counters.reserve(request.options.num_iterations);
    propagation_timers.reserve(request.options.num_iterations);
    perturbation_timers.reserve(request.options.num_iterations);
    iteration_depth_radii.reserve(request.options.num_iterations);
    iteration_normal_radii.reserve(request.options.num_iterations);
    for (std::size_t iteration = 0; iteration < request.options.num_iterations;
         ++iteration) {
      ResetCountersKernel<<<1, 1, 0, stream>>>(d_counters.get());
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
      const auto run_direction = [&](int direction) {
        PropagationKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            reference_texture.get(), source_images_texture.get(),
            use_source_depth_maps ? source_depth_maps_texture.get() : 0,
            poses_texture.get(),
            use_selection_probability_maps ? selection_probability_texture.get()
                                           : 0,
            dynamic_selection_probabilities_ptr,
            d_source_images_info.get(),
            d_depth_map.get(), d_normal_map.get(), d_cost_map.get(),
            device_options, direction, d_counters.get());
        ++cuda_stream_kernel_launches;
        CudaCheck(cudaGetLastError());
      };
      const auto run_checkerboard = [&](int parity) {
        CheckerboardPropagationKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            reference_texture.get(), source_images_texture.get(),
            use_source_depth_maps ? source_depth_maps_texture.get() : 0,
            poses_texture.get(),
            use_selection_probability_maps ? selection_probability_texture.get()
                                           : 0,
            dynamic_selection_probabilities_ptr,
            d_source_images_info.get(), d_depth_map.get(), d_normal_map.get(),
            d_cost_map.get(), device_options, parity, d_counters.get());
        ++cuda_stream_kernel_launches;
        CudaCheck(cudaGetLastError());
      };

      propagation_timers.emplace_back();
      propagation_timers.back().Start(stream);
      switch (request.options.propagation_schedule) {
        case PatchMatchPropagationSchedule::kForwardBackward:
          run_direction(0);
          run_direction(1);
          break;
        case PatchMatchPropagationSchedule::kFourDirection:
          run_direction(0);
          run_direction(1);
          run_direction(2);
          run_direction(3);
          break;
        case PatchMatchPropagationSchedule::kCheckerboard:
          for (const int parity : {0, 1, 1, 0, 0, 1, 1, 0}) {
            run_checkerboard(parity);
          }
          break;
      }
      propagation_timers.back().Stop(stream);

      StorePerturbationRadiiKernel<<<1, 1, 0, stream>>>(
          d_perturbation_radius_state.get(), d_iteration_depth_radii.get(),
          d_iteration_normal_radii.get(), static_cast<int>(iteration),
          request.options.min_depth_perturbation,
          request.options.min_normal_radius_degrees);
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
      perturbation_timers.emplace_back();
      perturbation_timers.back().Start(stream);
      PerturbationKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
          reference_texture.get(), source_images_texture.get(),
          use_source_depth_maps ? source_depth_maps_texture.get() : 0,
          poses_texture.get(),
          use_selection_probability_maps ? selection_probability_texture.get()
                                         : 0,
          dynamic_selection_probabilities_ptr,
          d_source_images_info.get(), d_depth_map.get(),
          d_normal_map.get(), d_cost_map.get(), device_options,
          request.options.perturbation.random_seed +
              static_cast<std::uint32_t>(iteration),
          d_iteration_depth_radii.get(), d_iteration_normal_radii.get(),
          static_cast<int>(iteration),
          static_cast<int>(request.options.perturbation.num_samples_per_pixel),
          d_counters.get());
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
      perturbation_timers.back().Stop(stream);
      if (request.options.source_selection.dynamic_probability) {
        const std::size_t num_source_pixels =
            num_pixels * static_cast<std::size_t>(device_options.num_sources);
        const int probability_blocks = static_cast<int>(
            (num_source_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock);
        UpdateSelectionProbabilityKernel<<<probability_blocks,
                                           kThreadsPerBlock, 0, stream>>>(
            reference_texture.get(), source_images_texture.get(),
            use_source_depth_maps ? source_depth_maps_texture.get() : 0,
            poses_texture.get(), d_source_images_info.get(), d_depth_map.get(),
            d_normal_map.get(), dynamic_selection_probabilities_ptr,
            request.options.source_selection.directional_hmm
                ? d_source_selection_emission_map.get()
                : nullptr,
            device_options, d_counters.get());
        ++cuda_stream_kernel_launches;
        CudaCheck(cudaGetLastError());
        if (request.options.source_selection.directional_hmm) {
          const int row_lines =
              device_options.num_sources * device_options.height;
          const int row_blocks =
              (row_lines + kThreadsPerBlock - 1) / kThreadsPerBlock;
          DirectionalHmmRowsKernel<<<row_blocks, kThreadsPerBlock, 0, stream>>>(
              d_source_selection_emission_map.get(),
              d_source_selection_backward_message_map.get(),
              dynamic_selection_probabilities_ptr, device_options);
          ++cuda_stream_kernel_launches;
          CudaCheck(cudaGetLastError());
          const int column_lines =
              device_options.num_sources * device_options.width;
          const int column_blocks =
              (column_lines + kThreadsPerBlock - 1) / kThreadsPerBlock;
          DirectionalHmmColumnsKernel<<<column_blocks, kThreadsPerBlock, 0,
                                        stream>>>(
              d_source_selection_emission_map.get(),
              d_source_selection_backward_message_map.get(),
              dynamic_selection_probabilities_ptr, device_options);
          ++cuda_stream_kernel_launches;
          CudaCheck(cudaGetLastError());
        }
      }
      StoreIterationCountersKernel<<<1, 1, 0, stream>>>(
          d_counters.get(), d_iteration_counters.get(),
          static_cast<int>(iteration));
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
      UpdatePerturbationRadiusStateKernel<<<1, 1, 0, stream>>>(
          d_perturbation_radius_state.get(), d_counters.get(),
          request.options.perturbation_shrink,
          request.options.adaptive_perturbation_min_update_ratio);
      ++cuda_stream_kernel_launches;
      CudaCheck(cudaGetLastError());
    }
    const SteadyClock::time_point download_start = SteadyClock::now();
    if (request.options.num_iterations > 0) {
      d_iteration_counters.CopyToHostAsync(h_iteration_counters.data(),
                                           stream);
      d_iteration_depth_radii.CopyToHostAsync(h_iteration_depth_radii.data(),
                                              stream);
      d_iteration_normal_radii.CopyToHostAsync(h_iteration_normal_radii.data(),
                                               stream);
      ++cuda_async_counter_readbacks;
    }
    PackOutputMapsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        d_depth_map.get(), d_normal_map.get(), d_cost_map.get(),
        d_output_pixels.get(), static_cast<int>(num_pixels));
    ++cuda_stream_kernel_launches;
    CudaCheck(cudaGetLastError());
    d_output_pixels.CopyToHostAsync(h_output_pixels.data(), stream);
    if (request.options.source_selection.dynamic_probability) {
      d_dynamic_selection_probability_map.CopyToHostAsync(
          h_dynamic_selection_probability_map.data(), stream);
      ++cuda_async_output_readbacks;
    }
    cuda_packed_output_pixels = h_output_pixels.size();
    cuda_output_map_readbacks = h_output_pixels.size() == 0 ? 0 : 1;
    cuda_async_output_readbacks = h_output_pixels.size() == 0 ? 0 : 1;
    CudaCheck(cudaStreamSynchronize(stream));
    for (CudaEventTimer& timer : propagation_timers) {
      cuda_propagation_time_ms += timer.ElapsedMilliseconds();
    }
    for (CudaEventTimer& timer : perturbation_timers) {
      cuda_perturbation_time_ms += timer.ElapsedMilliseconds();
    }
    if (request.options.num_iterations > 0) {
      iteration_counters = h_iteration_counters.ToVector();
      iteration_depth_radii = h_iteration_depth_radii.ToVector();
      iteration_normal_radii = h_iteration_normal_radii.ToVector();
    }
    cuda_download_time_ms = ElapsedMilliseconds(download_start,
                                                SteadyClock::now());

    const SteadyClock::time_point materialization_start = SteadyClock::now();
    const SteadyClock::time_point output_to_vector_start = SteadyClock::now();
    double cuda_output_to_vector_time_ms =
        ElapsedMilliseconds(output_to_vector_start, SteadyClock::now());
    const SteadyClock::time_point dense_map_fill_start = SteadyClock::now();
    const DeviceOutputPixel* output_values = h_output_pixels.data();
    PatchMatchStereoResult result;
    result.depth_map =
        DenseDepthMap(static_cast<std::size_t>(width),
                      static_cast<std::size_t>(height), request.depth_min,
                      request.depth_max);
    result.normal_map =
        NormalMap(static_cast<std::size_t>(width),
                  static_cast<std::size_t>(height));
    result.cost_map =
        DenseCostMap(static_cast<std::size_t>(width),
                     static_cast<std::size_t>(height));
    std::vector<float>& depth_values = result.depth_map.MutableData();
    std::vector<Eigen::Vector3f>& normal_values =
        result.normal_map.MutableData();
    std::vector<float>& cost_values = result.cost_map.MutableData();
    for (std::size_t idx = 0; idx < num_pixels; ++idx) {
      const DeviceOutputPixel& output = output_values[idx];
      depth_values[idx] = output.depth;
      const Eigen::Vector3f normal(output.normal_x, output.normal_y,
                                   output.normal_z);
      normal_values[idx] =
          normal.allFinite() && normal.squaredNorm() > 0.0F
              ? normal
              : Eigen::Vector3f::Zero();
      cost_values[idx] = output.cost;
    }
    const double cuda_dense_map_fill_time_ms =
        ElapsedMilliseconds(dense_map_fill_start, SteadyClock::now());

    result.summary.requested_backend = request.options.backend;
    result.summary.actual_backend = PatchMatchBackend::kCuda;
    result.summary.num_pyramid_levels = 1;
    result.summary.num_iterations = request.options.num_iterations;
    result.summary.cuda_upload_time_ms = cuda_upload_time_ms;
    result.summary.cuda_initialization_time_ms =
        cuda_initialization_time_ms;
    result.summary.cuda_evaluate_cost_time_ms =
        cuda_evaluate_cost_time_ms;
    result.summary.cuda_propagation_time_ms = cuda_propagation_time_ms;
    result.summary.cuda_perturbation_time_ms = cuda_perturbation_time_ms;
    result.summary.cuda_download_time_ms = cuda_download_time_ms;
    result.summary.cuda_request_validation_time_ms =
        cuda_request_validation_time_ms;
    result.summary.cuda_host_pack_time_ms = cuda_host_pack_time_ms;
    result.summary.cuda_shared_reference_tile_pixels =
        cuda_shared_reference_tile_pixels;
    result.summary.cuda_workspace_texture_allocations =
        cuda_workspace_texture_allocations;
    result.summary.cuda_workspace_texture_reuses =
        cuda_workspace_texture_reuses;
    result.summary.cuda_workspace_buffer_allocations =
        cuda_workspace_buffer_allocations;
    result.summary.cuda_workspace_buffer_reuses =
        cuda_workspace_buffer_reuses;
    result.summary.cuda_iteration_counter_snapshots =
        iteration_counters.size();
    result.summary.cuda_iteration_counter_readbacks =
        iteration_counters.empty() ? 0 : 1;
    result.summary.cuda_packed_output_pixels = cuda_packed_output_pixels;
    result.summary.cuda_output_map_readbacks = cuda_output_map_readbacks;
    result.summary.cuda_async_host_uploads = cuda_async_host_uploads;
    result.summary.cuda_async_texture_uploads = cuda_async_texture_uploads;
    result.summary.cuda_async_counter_readbacks =
        cuda_async_counter_readbacks;
    result.summary.cuda_async_output_readbacks = cuda_async_output_readbacks;
    result.summary.cuda_pinned_host_upload_bytes =
        cuda_pinned_host_upload_bytes;
    result.summary.cuda_pinned_texture_upload_bytes =
        cuda_pinned_texture_upload_bytes;
    result.summary.cuda_pinned_host_staging_bytes =
        cuda_pinned_host_staging_bytes;
    result.summary.cuda_stream_kernel_launches = cuda_stream_kernel_launches;
    switch (request.options.propagation_schedule) {
      case PatchMatchPropagationSchedule::kForwardBackward:
        result.summary.num_forward_sweeps = request.options.num_iterations;
        result.summary.num_backward_sweeps = request.options.num_iterations;
        break;
      case PatchMatchPropagationSchedule::kFourDirection:
        result.summary.num_forward_sweeps = request.options.num_iterations;
        result.summary.num_backward_sweeps = request.options.num_iterations;
        result.summary.num_top_down_sweeps = request.options.num_iterations;
        result.summary.num_bottom_up_sweeps = request.options.num_iterations;
        break;
      case PatchMatchPropagationSchedule::kCheckerboard:
        result.summary.num_checkerboard_passes =
            request.options.num_iterations * 8;
        break;
    }
    result.summary.num_perturbation_passes = request.options.num_iterations;
    for (std::size_t iteration = 0; iteration < iteration_counters.size();
         ++iteration) {
      const DeviceCounters& counters = iteration_counters[iteration];
      const std::size_t candidate_evaluations =
          static_cast<std::size_t>(counters.candidate_evaluations);
      const std::size_t propagation_updates =
          static_cast<std::size_t>(counters.propagation_updates);
      const std::size_t perturbation_updates =
          static_cast<std::size_t>(counters.perturbation_updates);
      result.summary.num_candidate_evaluations += candidate_evaluations;
      result.summary.num_plane_propagation_candidates +=
          static_cast<std::size_t>(counters.plane_propagation_candidates);
      result.summary.num_perturbation_candidate_evaluations +=
          static_cast<std::size_t>(
              counters.perturbation_candidate_evaluations);
      result.summary.num_propagation_updates += propagation_updates;
      result.summary.num_perturbation_updates += perturbation_updates;
      result.summary.num_source_selection_tests +=
          static_cast<std::size_t>(counters.source_selection_tests);
      result.summary.num_source_selection_acceptances +=
          static_cast<std::size_t>(counters.source_selection_acceptances);
      result.summary.num_source_selection_rejections +=
          static_cast<std::size_t>(counters.source_selection_rejections);
      result.summary.num_source_selection_probability_updates +=
          static_cast<std::size_t>(
              counters.source_selection_probability_updates);
      result.summary.num_perturbation_geometric_candidate_rejections +=
          static_cast<std::size_t>(counters.geometric_candidate_rejections);
      result.summary.num_perturbation_geometric_candidate_acceptances +=
          static_cast<std::size_t>(counters.geometric_candidate_acceptances);
      result.summary.perturbation_total_abs_depth_update +=
          counters.total_abs_depth_update;
      result.summary.perturbation_total_normal_angle_degrees +=
          counters.total_normal_angle_degrees;
      result.summary.perturbation_max_normal_angle_degrees =
          std::max(result.summary.perturbation_max_normal_angle_degrees,
                   counters.max_normal_angle_degrees);
      result.summary.iteration_pyramid_levels.push_back(0);
      result.summary.iteration_indices.push_back(iteration);
      result.summary.iteration_candidate_evaluations.push_back(
          candidate_evaluations);
      result.summary.iteration_propagation_updates.push_back(
          propagation_updates);
      result.summary.iteration_perturbation_updates.push_back(
          perturbation_updates);
      result.summary.iteration_source_selection_tests.push_back(
          static_cast<std::size_t>(counters.source_selection_tests));
      result.summary.iteration_source_selection_acceptances.push_back(
          static_cast<std::size_t>(counters.source_selection_acceptances));
      result.summary.iteration_source_selection_rejections.push_back(
          static_cast<std::size_t>(counters.source_selection_rejections));
      result.summary.iteration_source_selection_probability_means.push_back(
          counters.source_selection_probability_updates == 0
              ? 0.0F
              : static_cast<float>(
                    counters.total_source_selection_probability /
                    static_cast<double>(
                        counters.source_selection_probability_updates)));
      const std::size_t total_updates =
          propagation_updates + perturbation_updates;
      result.summary.update_ratios.push_back(
          candidate_evaluations == 0
              ? 0.0
              : static_cast<double>(total_updates) /
                    static_cast<double>(candidate_evaluations));
      result.summary.depth_radii.push_back(iteration_depth_radii[iteration]);
      result.summary.normal_radii_degrees.push_back(
          iteration_normal_radii[iteration]);
    }
    if (request.options.source_selection.dynamic_probability) {
      result.summary.final_source_selection_probability_means.reserve(
          static_cast<std::size_t>(device_options.num_sources));
      for (int source_idx = 0; source_idx < device_options.num_sources;
           ++source_idx) {
        double sum = 0.0;
        const float* source_probabilities =
            h_dynamic_selection_probability_map.data() +
            static_cast<std::size_t>(source_idx) * num_pixels;
        for (std::size_t pixel_idx = 0; pixel_idx < num_pixels; ++pixel_idx) {
          sum += source_probabilities[pixel_idx];
        }
        result.summary.final_source_selection_probability_means.push_back(
            num_pixels == 0
                ? 0.0F
                : static_cast<float>(sum / static_cast<double>(num_pixels)));
      }
    }
    if (result.summary.num_perturbation_updates > 0) {
      result.summary.perturbation_mean_abs_depth_update =
          result.summary.perturbation_total_abs_depth_update /
          static_cast<double>(result.summary.num_perturbation_updates);
      result.summary.perturbation_mean_normal_angle_degrees =
          result.summary.perturbation_total_normal_angle_degrees /
          static_cast<double>(result.summary.num_perturbation_updates);
    }
    const SteadyClock::time_point normal_refinement_start =
        SteadyClock::now();
    if (request.options.normal_refinement.enabled) {
      const PatchMatchNormalRefinementSummary refinement_summary =
          RefineNormalMapFromDepth(*request.reference_view, result.depth_map,
                                   request.options.normal_refinement,
                                   &result.normal_map);
      result.summary.num_normal_refinement_pixels =
          refinement_summary.num_pixels;
      result.summary.num_normal_refined_pixels =
          refinement_summary.num_refined_pixels;
      result.summary.num_normal_refinement_skipped_pixels =
          refinement_summary.num_skipped_pixels;
    }
    result.summary.cuda_normal_refinement_time_ms =
        ElapsedMilliseconds(normal_refinement_start, SteadyClock::now());
    const SteadyClock::time_point valid_cost_count_start =
        SteadyClock::now();
    result.summary.num_valid_cost_pixels = result.cost_map.ValidPixelCount();
    result.summary.cuda_valid_cost_count_time_ms =
        ElapsedMilliseconds(valid_cost_count_start, SteadyClock::now());
    const SteadyClock::time_point wrapper_end = SteadyClock::now();
    result.summary.cuda_result_materialization_time_ms =
        ElapsedMilliseconds(materialization_start, wrapper_end);
    result.summary.cuda_output_to_vector_time_ms =
        cuda_output_to_vector_time_ms;
    result.summary.cuda_dense_map_fill_time_ms =
        cuda_dense_map_fill_time_ms;
    result.summary.cuda_host_initial_normal_upload_normalizations =
        cuda_host_initial_normal_upload_normalizations;
    result.summary.cuda_host_output_normalizations =
        cuda_host_output_normalizations;
    result.summary.cuda_device_count_queries = cuda_device_count_queries;
    result.summary.cuda_device_count_cache_hits =
        cuda_device_count_cache_hits;
    result.summary.cuda_total_time_ms =
        ElapsedMilliseconds(total_start, wrapper_end);
    result.summary.cuda_wrapper_total_time_ms =
        ElapsedMilliseconds(wrapper_start, wrapper_end);
    result.summary.cuda_single_scale_dispatch_time_ms =
        result.summary.cuda_wrapper_total_time_ms;
    return result;
  } catch (const std::exception& error) {
    SetUnavailableReason(unavailable_reason,
                         std::string("CUDA PatchMatch failed: ") +
                             error.what());
    return std::nullopt;
  }
}

}  // namespace

double WarmupCudaPatchMatch(int cuda_gpu_index,
                            std::string* unavailable_reason) {
  if (unavailable_reason != nullptr) {
    unavailable_reason->clear();
  }
  const SteadyClock::time_point start = SteadyClock::now();
  try {
    const CachedCudaDeviceCount device_count = GetCachedCudaDeviceCount();
    if (device_count.status != cudaSuccess || device_count.count <= 0) {
      SetUnavailableReason(unavailable_reason,
                           "CUDA PatchMatch backend found no CUDA devices.");
      return 0.0;
    }
    if (cuda_gpu_index >= 0) {
      if (cuda_gpu_index >= device_count.count) {
        SetUnavailableReason(unavailable_reason,
                             "Requested CUDA PatchMatch GPU index is invalid.");
        return 0.0;
      }
      CudaCheck(cudaSetDevice(cuda_gpu_index));
    }
    CudaCheck(cudaFree(nullptr));
  } catch (const std::exception& error) {
    SetUnavailableReason(unavailable_reason,
                         std::string("CUDA PatchMatch warmup failed: ") +
                             error.what());
    return 0.0;
  }
  return ElapsedMilliseconds(start, SteadyClock::now());
}

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereo(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason) {
  const SteadyClock::time_point wrapper_start = SteadyClock::now();
  if (request.options.pyramid_levels <= 1) {
    std::optional<PatchMatchStereoResult> result =
        RunCudaPatchMatchStereoSingleScale(request, unavailable_reason,
                                          request.workspace);
    if (result.has_value()) {
      result->summary.num_pyramid_levels = 1;
      result->summary.cuda_wrapper_total_time_ms =
          ElapsedMilliseconds(wrapper_start, SteadyClock::now());
    }
    return result;
  }

  try {
    const SteadyClock::time_point validation_start = SteadyClock::now();
    ValidateRequest(request);
    std::size_t cuda_device_count_queries = 0;
    std::size_t cuda_device_count_cache_hits = 0;
    if (request.options.cuda_gpu_index >= 0) {
      const CachedCudaDeviceCount device_count = GetCachedCudaDeviceCount();
      cuda_device_count_queries += device_count.queried_runtime ? 1U : 0U;
      cuda_device_count_cache_hits +=
          device_count.queried_runtime ? 0U : 1U;
      if (device_count.status != cudaSuccess ||
          request.options.cuda_gpu_index >= device_count.count) {
        SetUnavailableReason(unavailable_reason,
                             "Requested CUDA PatchMatch GPU index is invalid.");
        return std::nullopt;
      }
      CudaCheck(cudaSetDevice(request.options.cuda_gpu_index));
    }
    const double cuda_request_validation_time_ms =
        ElapsedMilliseconds(validation_start, SteadyClock::now());
    CudaPatchMatchWorkspace local_workspace;
    CudaPatchMatchWorkspace* workspace =
        request.workspace == nullptr ? &local_workspace : request.workspace;
    std::size_t num_levels = request.options.pyramid_levels;
    while (num_levels > 1) {
      const std::size_t divisor = CudaPyramidLevelDivisor(num_levels, 0);
      const std::size_t width =
          CudaPyramidLevelSize(request.reference_image->Width(), divisor);
      const std::size_t height =
          CudaPyramidLevelSize(request.reference_image->Height(), divisor);
      if (width >= request.reference_image->Width() &&
          height >= request.reference_image->Height()) {
        --num_levels;
        continue;
      }
      break;
    }
    if (num_levels <= 1) {
      CudaPatchMatchRequest single_request = request;
      single_request.options.pyramid_levels = 1;
      std::optional<PatchMatchStereoResult> result =
          RunCudaPatchMatchStereoSingleScale(single_request,
                                             unavailable_reason, workspace);
      if (result.has_value()) {
        result->summary.num_pyramid_levels = 1;
      }
      return result;
    }

    PatchMatchMaps prior_maps;
    bool has_prior = false;
    PatchMatchStereoResult final_result;
    PatchMatchStereoSummary accumulated_summary;
    accumulated_summary.num_pyramid_levels = num_levels;
    accumulated_summary.cuda_request_validation_time_ms +=
        cuda_request_validation_time_ms;
    accumulated_summary.cuda_device_count_queries +=
        cuda_device_count_queries;
    accumulated_summary.cuda_device_count_cache_hits +=
        cuda_device_count_cache_hits;

    for (std::size_t level_idx = 0; level_idx < num_levels; ++level_idx) {
      const SteadyClock::time_point pyramid_resize_start =
          SteadyClock::now();
      const std::size_t divisor =
          CudaPyramidLevelDivisor(num_levels, level_idx);
      const std::size_t reference_width =
          CudaPyramidLevelSize(request.reference_image->Width(), divisor);
      const std::size_t reference_height =
          CudaPyramidLevelSize(request.reference_image->Height(), divisor);
      const double reference_scale_x =
          static_cast<double>(reference_width) /
          static_cast<double>(request.reference_image->Width());
      const double reference_scale_y =
          static_cast<double>(reference_height) /
          static_cast<double>(request.reference_image->Height());

      const MvsImage level_reference_image =
          ResizeImageForCudaPyramidLevel(*request.reference_image,
                                         reference_width, reference_height);
      const PatchMatchView level_reference_view =
          ScaleViewForCudaPyramidLevel(*request.reference_view,
                                       reference_scale_x, reference_scale_y,
                                       reference_width, reference_height);

      std::vector<MvsImage> level_source_images;
      std::vector<PatchMatchView> level_source_views;
      level_source_images.reserve(request.source_images->size());
      level_source_views.reserve(request.source_views->size());
      for (std::size_t i = 0; i < request.source_images->size(); ++i) {
        const std::size_t source_width =
            CudaPyramidLevelSize(request.source_images->at(i).Width(),
                                 divisor);
        const std::size_t source_height =
            CudaPyramidLevelSize(request.source_images->at(i).Height(),
                                 divisor);
        const double source_scale_x =
            static_cast<double>(source_width) /
            static_cast<double>(request.source_images->at(i).Width());
        const double source_scale_y =
            static_cast<double>(source_height) /
            static_cast<double>(request.source_images->at(i).Height());
        level_source_images.push_back(ResizeImageForCudaPyramidLevel(
            request.source_images->at(i), source_width, source_height));
        level_source_views.push_back(ScaleViewForCudaPyramidLevel(
            request.source_views->at(i), source_scale_x, source_scale_y,
            source_width, source_height));
      }

      std::vector<DenseDepthMap> level_source_depth_maps;
      const std::vector<DenseDepthMap>* level_source_depth_maps_ptr = nullptr;
      if (request.source_depth_maps != nullptr) {
        level_source_depth_maps.reserve(request.source_depth_maps->size());
        for (std::size_t i = 0; i < request.source_depth_maps->size(); ++i) {
          level_source_depth_maps.push_back(ResizeDepthMapNearestForCuda(
              request.source_depth_maps->at(i), level_source_images[i].Width(),
              level_source_images[i].Height()));
        }
        level_source_depth_maps_ptr = &level_source_depth_maps;
      }

      std::vector<DenseCostMap> level_selection_probability_maps;
      const std::vector<DenseCostMap>* level_selection_probability_maps_ptr =
          nullptr;
      if (request.source_selection_probability_maps != nullptr) {
        level_selection_probability_maps.reserve(
            request.source_selection_probability_maps->size());
        for (const DenseCostMap& map :
             *request.source_selection_probability_maps) {
          level_selection_probability_maps.push_back(
              ResizeCostMapNearestForCuda(map, reference_width,
                                          reference_height));
        }
        level_selection_probability_maps_ptr =
            &level_selection_probability_maps;
      }

      PatchMatchMaps level_initial_maps;
      const PatchMatchMaps* level_initial_maps_ptr = nullptr;
      if (has_prior) {
        level_initial_maps = ResizePatchMatchMapsNearestForCuda(
            prior_maps, reference_width, reference_height);
        level_initial_maps.depth_map.SetDepthRange(request.depth_min,
                                                  request.depth_max);
        level_initial_maps_ptr = &level_initial_maps;
      }

      CudaPatchMatchRequest level_request;
      level_request.reference_view = &level_reference_view;
      level_request.source_views = &level_source_views;
      level_request.reference_image = &level_reference_image;
      level_request.source_images = &level_source_images;
      level_request.source_depth_maps = level_source_depth_maps_ptr;
      level_request.source_selection_probability_maps =
          level_selection_probability_maps_ptr;
      level_request.initial_maps = level_initial_maps_ptr;
      level_request.depth_min = request.depth_min;
      level_request.depth_max = request.depth_max;
      level_request.options = request.options;
      level_request.options.pyramid_levels = 1;
      level_request.options.initialization.random_seed +=
          static_cast<std::uint32_t>(level_idx * 101U);
      level_request.options.perturbation.random_seed +=
          static_cast<std::uint32_t>(level_idx * 211U);
      accumulated_summary.cuda_pyramid_resize_time_ms +=
          ElapsedMilliseconds(pyramid_resize_start, SteadyClock::now());

      std::optional<PatchMatchStereoResult> level_result =
          RunCudaPatchMatchStereoSingleScale(level_request,
                                             unavailable_reason, workspace);
      if (!level_result.has_value()) {
        return std::nullopt;
      }
      std::fill(level_result->summary.iteration_pyramid_levels.begin(),
                level_result->summary.iteration_pyramid_levels.end(),
                level_idx);
      AccumulateCudaSummary(&accumulated_summary, level_result->summary);
      prior_maps.depth_map = level_result->depth_map;
      prior_maps.normal_map = level_result->normal_map;
      has_prior = true;
      final_result = std::move(*level_result);
    }

    if (accumulated_summary.num_perturbation_updates > 0) {
      accumulated_summary.perturbation_mean_abs_depth_update =
          accumulated_summary.perturbation_total_abs_depth_update /
          static_cast<double>(accumulated_summary.num_perturbation_updates);
      accumulated_summary.perturbation_mean_normal_angle_degrees =
          accumulated_summary.perturbation_total_normal_angle_degrees /
          static_cast<double>(accumulated_summary.num_perturbation_updates);
    }
    accumulated_summary.requested_backend = request.options.backend;
    accumulated_summary.actual_backend = PatchMatchBackend::kCuda;
    accumulated_summary.num_valid_cost_pixels =
        final_result.cost_map.ValidPixelCount();
    accumulated_summary.cuda_wrapper_total_time_ms =
        ElapsedMilliseconds(wrapper_start, SteadyClock::now());
    final_result.summary = std::move(accumulated_summary);
    return final_result;
  } catch (const std::exception& error) {
    SetUnavailableReason(unavailable_reason,
                         std::string("CUDA PatchMatch pyramid failed: ") +
                             error.what());
    return std::nullopt;
  }
}

}  // namespace colite::mvs
