#include "colite/mvs/patch_match_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
  float invalid_cost = 1.0F;
  float min_variance = 1e-8F;
  int min_num_valid_sources = 1;
  int max_num_aggregated_costs = 3;
  float geometric_consistency_weight = 0.0F;
  float geometric_max_relative_depth_error = 0.01F;
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
};

struct DeviceCounters {
  unsigned long long candidate_evaluations = 0ULL;
  unsigned long long propagation_updates = 0ULL;
  unsigned long long perturbation_updates = 0ULL;
  unsigned long long source_selection_tests = 0ULL;
  unsigned long long source_selection_acceptances = 0ULL;
  unsigned long long source_selection_rejections = 0ULL;
  double total_abs_depth_update = 0.0;
  double total_normal_angle_degrees = 0.0;
  double max_normal_angle_degrees = 0.0;
};

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
    Release();
    size_ = size;
    if (size_ > 0) {
      CudaCheck(cudaMalloc(&ptr_, sizeof(T) * size_));
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
  }

  void MoveFrom(DeviceBuffer* other) {
    ptr_ = other->ptr_;
    size_ = other->size_;
    other->ptr_ = nullptr;
    other->size_ = 0;
  }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

void CudaCheck(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

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

  void Create(int width,
              int height,
              int depth,
              const std::vector<float>& host_values,
              cudaTextureFilterMode filter_mode = cudaFilterModeLinear) {
    if (width <= 0 || height <= 0 || depth <= 0) {
      throw std::invalid_argument("CUDA texture dimensions must be positive.");
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
        static_cast<std::size_t>(depth);
    if (host_values.size() != expected_size) {
      throw std::invalid_argument("CUDA texture host data size mismatch.");
    }
    Release();
    width_ = width;
    height_ = height;
    depth_ = depth;

    const cudaChannelFormatDesc channel_desc =
        cudaCreateChannelDesc<float>();
    const cudaExtent extent = make_cudaExtent(
        static_cast<std::size_t>(width_), static_cast<std::size_t>(height_),
        static_cast<std::size_t>(depth_));
    CudaCheck(cudaMalloc3DArray(&array_, &channel_desc, extent,
                                cudaArrayLayered));

    cudaMemcpy3DParms copy_params;
    std::memset(&copy_params, 0, sizeof(copy_params));
    copy_params.extent = extent;
    copy_params.kind = cudaMemcpyHostToDevice;
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(host_values.data()),
        static_cast<std::size_t>(width_) * sizeof(float),
        static_cast<std::size_t>(width_), static_cast<std::size_t>(height_));
    copy_params.dstArray = array_;
    CudaCheck(cudaMemcpy3D(&copy_params));

    cudaResourceDesc resource_desc;
    std::memset(&resource_desc, 0, sizeof(resource_desc));
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = array_;

    cudaTextureDesc texture_desc;
    std::memset(&texture_desc, 0, sizeof(texture_desc));
    texture_desc.addressMode[0] = cudaAddressModeBorder;
    texture_desc.addressMode[1] = cudaAddressModeBorder;
    texture_desc.addressMode[2] = cudaAddressModeBorder;
    texture_desc.filterMode = filter_mode;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    CudaCheck(cudaCreateTextureObject(&texture_, &resource_desc, &texture_desc,
                                      nullptr));
  }

  cudaTextureObject_t get() const { return texture_; }

 private:
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
  }

  void MoveFrom(FloatLayeredTexture* other) {
    array_ = other->array_;
    texture_ = other->texture_;
    width_ = other->width_;
    height_ = other->height_;
    depth_ = other->depth_;
    other->array_ = nullptr;
    other->texture_ = 0;
    other->width_ = 0;
    other->height_ = 0;
    other->depth_ = 0;
  }

  cudaArray_t array_ = nullptr;
  cudaTextureObject_t texture_ = 0;
  int width_ = 0;
  int height_ = 0;
  int depth_ = 0;
};

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

__device__ bool ComputeSourceDepthRelativeError(
    const DevicePatchMatchOptions& options,
    cudaTextureObject_t poses_texture,
    cudaTextureObject_t source_depth_maps_texture,
    const DeviceSourceImage& source,
    int source_idx,
    int col,
    int row,
    float depth,
    float* relative_error) {
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
  *relative_error = fabsf(backward_z - depth) / depth;
  return isfinite(*relative_error);
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

__device__ bool ShouldUseSourceForPixel(cudaTextureObject_t poses_texture,
                                        int source_idx,
                                        int col,
                                        int row,
                                        DeviceCounters* counters) {
  if (counters != nullptr) {
    atomicAdd(&counters->source_selection_tests, 1ULL);
  }
  constexpr int kBasePoseVariant = 0;
  const float probability =
      ReadPose(poses_texture, source_idx, kBasePoseVariant,
               kPoseSelectionProbabilityOffset);
  if (!isfinite(probability) || probability <= 0.0F) {
    if (counters != nullptr) {
      atomicAdd(&counters->source_selection_rejections, 1ULL);
    }
    return false;
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
    const DeviceSourceImage* source_images,
    const DevicePatchMatchOptions& options,
    int source_idx,
    int variant_idx,
    int col,
    int row,
    float depth,
    float3 normal) {
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
  for (int dy = -options.cost.window_radius; dy <= options.cost.window_radius;
       dy += options.cost.window_step) {
    for (int dx = -options.cost.window_radius;
         dx <= options.cost.window_radius; dx += options.cost.window_step) {
      const float sample_x = static_cast<float>(col + dx);
      const float sample_y = static_cast<float>(row + dy);
      const float ref_value =
          SampleLayeredTexture(reference_texture, options.width,
                               options.height, 0, sample_x, sample_y);
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
      sum_ref += ref_value;
      sum_src += src_value;
      sum_ref2 += static_cast<double>(ref_value) * ref_value;
      sum_src2 += static_cast<double>(src_value) * src_value;
      sum_cross += static_cast<double>(ref_value) * src_value;
      num_samples++;
    }
  }
  if (num_samples <= 0) {
    return options.cost.invalid_cost;
  }
  const double inv_n = 1.0 / static_cast<double>(num_samples);
  const double ref_var = sum_ref2 - sum_ref * sum_ref * inv_n;
  const double src_var = sum_src2 - sum_src * sum_src * inv_n;
  if (ref_var <= options.cost.min_variance ||
      src_var <= options.cost.min_variance) {
    return options.cost.invalid_cost;
  }
  const double covariance = sum_cross - sum_ref * sum_src * inv_n;
  const double zncc = fmax(-1.0, fmin(1.0, covariance / sqrt(ref_var * src_var)));
  float cost = static_cast<float>(0.5 * (1.0 - zncc));

  if (source_depth_maps_texture != 0 &&
      options.cost.geometric_consistency_weight > 0.0F) {
    float relative_error = 0.0F;
    if (ComputeSourceDepthRelativeError(
            options, poses_texture, source_depth_maps_texture, source,
            source_idx, col, row, depth, &relative_error)) {
      cost += options.cost.geometric_consistency_weight *
              fminf(1.0F,
                    relative_error /
                        options.cost.geometric_max_relative_depth_error);
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
                                   const DeviceSourceImage* source_images,
                                   const DevicePatchMatchOptions& options,
                                   int source_idx,
                                   int col,
                                   int row,
                                   float depth,
                                   float3 normal,
                                   DeviceCounters* counters) {
  if (!ShouldUseSourceForPixel(poses_texture, source_idx, col, row,
                               counters)) {
    return options.cost.invalid_cost;
  }
  float best_cost = options.cost.invalid_cost;
  for (int variant_idx = 0; variant_idx < options.num_pose_variants;
       ++variant_idx) {
    const float cost = ComputeSourceCostForVariant(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, source_images, options, source_idx, variant_idx, col,
        row, depth, normal);
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
                                      const DeviceSourceImage* source_images,
                                      const DevicePatchMatchOptions& options,
                                      int col,
                                      int row,
                                      float depth,
                                      float3 normal,
                                      float* aggregated_cost,
                                      DeviceCounters* counters) {
  float best_costs[kMaxCudaSourceImages];
  int num_valid = 0;
  for (int source_idx = 0; source_idx < options.num_sources; ++source_idx) {
    const float source_cost = ComputeSourceCost(
        reference_texture, source_images_texture, source_depth_maps_texture,
        poses_texture, source_images, options, source_idx, col, row, depth,
        normal, counters);
    if (!isfinite(source_cost) || source_cost >= options.cost.invalid_cost) {
      continue;
    }
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

__global__ void InitHypothesesKernel(float* depth_map,
                                     float* normal_map,
                                     DevicePatchMatchOptions options,
                                     std::uint32_t seed,
                                     float3 default_normal) {
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
  normal_map[idx * 3 + 0] = default_normal.x;
  normal_map[idx * 3 + 1] = default_normal.y;
  normal_map[idx * 3 + 2] = default_normal.z;
}

__global__ void EvaluateCostKernel(cudaTextureObject_t reference_texture,
                                   cudaTextureObject_t source_images_texture,
                                   cudaTextureObject_t source_depth_maps_texture,
                                   cudaTextureObject_t poses_texture,
                                   const DeviceSourceImage* source_images,
                                   const float* depth_map,
                                   const float* normal_map,
                                   float* cost_map,
                                   DevicePatchMatchOptions options) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pixels = options.width * options.height;
  if (idx >= num_pixels) {
    return;
  }
  const int row = idx / options.width;
  const int col = idx - row * options.width;
  const float depth = depth_map[idx];
  const float3 normal =
      make_float3(normal_map[idx * 3 + 0], normal_map[idx * 3 + 1],
                  normal_map[idx * 3 + 2]);
  float cost = options.cost.invalid_cost;
  if (IsFinitePositive(depth)) {
    ComputeAggregatedCost(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, options, col, row, depth, normal,
                          &cost, nullptr);
  }
  cost_map[idx] = cost;
}

__device__ void TryUpdateFromNeighbor(cudaTextureObject_t reference_texture,
                                      cudaTextureObject_t source_images_texture,
                                      cudaTextureObject_t source_depth_maps_texture,
                                      cudaTextureObject_t poses_texture,
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
  const float candidate_depth = depth_map[neighbor_idx];
  const float3 candidate_normal =
      make_float3(normal_map[neighbor_idx * 3 + 0],
                  normal_map[neighbor_idx * 3 + 1],
                  normal_map[neighbor_idx * 3 + 2]);
  if (!IsFinitePositive(candidate_depth)) {
    return;
  }
  atomicAdd(&counters->candidate_evaluations, 1ULL);
  float candidate_cost = options.cost.invalid_cost;
  if (!ComputeAggregatedCost(reference_texture, source_images_texture,
                             source_depth_maps_texture, poses_texture,
                             source_images, options, col, row,
                             candidate_depth, candidate_normal,
                             &candidate_cost, counters)) {
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
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row, col - 1, counters);
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row - 1, col, counters);
  } else if (direction == 1) {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row, col + 1, counters);
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row + 1, col, counters);
  } else if (direction == 2) {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row - 1, col, counters);
  } else {
    TryUpdateFromNeighbor(reference_texture, source_images_texture,
                          source_depth_maps_texture, poses_texture,
                          source_images, depth_map, normal_map, cost_map,
                          options, row, col, row + 1, col, counters);
  }
}

__global__ void PerturbationKernel(cudaTextureObject_t reference_texture,
                                   cudaTextureObject_t source_images_texture,
                                   cudaTextureObject_t source_depth_maps_texture,
                                   cudaTextureObject_t poses_texture,
                                   const DeviceSourceImage* source_images,
                                   float* depth_map,
                                   float* normal_map,
                                   float* cost_map,
                                   DevicePatchMatchOptions options,
                                   std::uint32_t seed,
                                   float depth_radius,
                                   float normal_radius_degrees,
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
    atomicAdd(&counters->candidate_evaluations, 1ULL);
    float candidate_cost = options.cost.invalid_cost;
    if (!ComputeAggregatedCost(reference_texture, source_images_texture,
                               source_depth_maps_texture, poses_texture,
                               source_images, options, col, row,
                               candidate_depth, candidate_normal,
                               &candidate_cost, counters)) {
      continue;
    }
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_depth = candidate_depth;
      best_normal = candidate_normal;
      updated = true;
    }
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

std::vector<float> CopyDepthMapToVector(const DenseDepthMap& depth_map) {
  std::vector<float> values(depth_map.NumPixels(), 0.0F);
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      values[row * depth_map.Width() + col] = depth_map.Get(row, col);
    }
  }
  return values;
}

std::vector<float> CopyNormalMapToVector(const NormalMap& normal_map) {
  std::vector<float> values(normal_map.NumPixels() * 3, 0.0F);
  for (std::size_t row = 0; row < normal_map.Height(); ++row) {
    for (std::size_t col = 0; col < normal_map.Width(); ++col) {
      const std::size_t idx = row * normal_map.Width() + col;
      const Eigen::Vector3f normal = normal_map.Get(row, col);
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
      request.options.cost.min_num_valid_sources == 0 ||
      request.options.perturbation.num_samples_per_pixel == 0) {
    throw std::invalid_argument("CUDA PatchMatch options are invalid.");
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
  accumulated->num_propagation_updates += level.num_propagation_updates;
  accumulated->num_perturbation_updates += level.num_perturbation_updates;
  accumulated->num_source_selection_tests += level.num_source_selection_tests;
  accumulated->num_source_selection_acceptances +=
      level.num_source_selection_acceptances;
  accumulated->num_source_selection_rejections +=
      level.num_source_selection_rejections;
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
}

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereoSingleScale(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason) {
  if (unavailable_reason != nullptr) {
    unavailable_reason->clear();
  }

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count <= 0) {
    SetUnavailableReason(unavailable_reason,
                         "CUDA PatchMatch backend found no CUDA devices.");
    return std::nullopt;
  }

  if (request.reference_view == nullptr && request.reference_image == nullptr) {
    return std::nullopt;
  }

  try {
    if (request.options.cuda_gpu_index >= 0) {
      if (request.options.cuda_gpu_index >= device_count) {
        SetUnavailableReason(unavailable_reason,
                             "Requested CUDA PatchMatch GPU index is invalid.");
        return std::nullopt;
      }
      CudaCheck(cudaSetDevice(request.options.cuda_gpu_index));
    }

    ValidateRequest(request);
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
    device_options.cost.window_radius = request.options.cost.window_radius;
    device_options.cost.window_step = request.options.cost.window_step;
    device_options.cost.invalid_cost = request.options.cost.invalid_cost;
    device_options.cost.min_variance = request.options.cost.min_variance;
    device_options.cost.min_num_valid_sources =
        static_cast<int>(request.options.cost.min_num_valid_sources);
    device_options.cost.max_num_aggregated_costs =
        static_cast<int>(request.options.cost.max_num_aggregated_costs);
    device_options.cost.geometric_consistency_weight =
        request.options.cost.geometric_consistency_weight;
    device_options.cost.geometric_max_relative_depth_error =
        request.options.cost.geometric_max_relative_depth_error;

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
    const std::vector<float> source_depth_maps =
        use_source_depth_maps
            ? PackSourceDepthMaps(*request.source_depth_maps, max_source_width,
                                  max_source_height)
            : std::vector<float>{};

    const std::size_t num_pixels =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    FloatLayeredTexture reference_texture(width, height, 1, reference_image);
    FloatLayeredTexture source_images_texture(max_source_width,
                                              max_source_height,
                                              device_options.num_sources,
                                              source_images);
    FloatLayeredTexture poses_texture(kNumPoseParams,
                                      device_options.num_sources,
                                      kNumPoseVariants,
                                      pose_texture_data,
                                      cudaFilterModePoint);
    FloatLayeredTexture source_depth_maps_texture;
    if (use_source_depth_maps) {
      source_depth_maps_texture.Create(max_source_width, max_source_height,
                                       device_options.num_sources,
                                       source_depth_maps);
    }
    DeviceBuffer<DeviceSourceImage> d_source_images_info(
        source_images_info.size());
    DeviceBuffer<float> d_depth_map(num_pixels);
    DeviceBuffer<float> d_normal_map(num_pixels * 3);
    DeviceBuffer<float> d_cost_map(num_pixels);
    DeviceBuffer<DeviceCounters> d_counters(1);

    d_source_images_info.CopyFromHost(source_images_info);
    Eigen::Vector3f default_normal =
        request.options.initialization.default_normal.squaredNorm() > 0.0F
            ? request.options.initialization.default_normal.normalized()
            : Eigen::Vector3f(0.0F, 0.0F, -1.0F);
    const float3 cuda_default_normal =
        make_float3(default_normal.x(), default_normal.y(), default_normal.z());
    const int blocks =
        static_cast<int>((num_pixels + kThreadsPerBlock - 1) /
                         kThreadsPerBlock);

    if (request.initial_maps != nullptr) {
      d_depth_map.CopyFromHost(
          CopyDepthMapToVector(request.initial_maps->depth_map));
      d_normal_map.CopyFromHost(
          CopyNormalMapToVector(request.initial_maps->normal_map));
    } else {
      InitHypothesesKernel<<<blocks, kThreadsPerBlock>>>(
          d_depth_map.get(), d_normal_map.get(), device_options,
          request.options.initialization.random_seed, cuda_default_normal);
      CudaCheck(cudaGetLastError());
    }
    EvaluateCostKernel<<<blocks, kThreadsPerBlock>>>(
        reference_texture.get(), source_images_texture.get(),
        use_source_depth_maps ? source_depth_maps_texture.get() : 0,
        poses_texture.get(), d_source_images_info.get(), d_depth_map.get(),
        d_normal_map.get(), d_cost_map.get(), device_options);
    CudaCheck(cudaGetLastError());

    float depth_radius = (request.depth_max - request.depth_min) *
                         request.options.initial_depth_perturbation_factor;
    float normal_radius = request.options.perturbation.normal_radius_degrees;
    std::vector<DeviceCounters> iteration_counters;
    std::vector<float> iteration_depth_radii;
    std::vector<float> iteration_normal_radii;
    iteration_counters.reserve(request.options.num_iterations);
    iteration_depth_radii.reserve(request.options.num_iterations);
    iteration_normal_radii.reserve(request.options.num_iterations);
    for (std::size_t iteration = 0; iteration < request.options.num_iterations;
         ++iteration) {
      d_counters.CopyFromHostValue(DeviceCounters{});
      const auto run_direction = [&](int direction) {
        PropagationKernel<<<blocks, kThreadsPerBlock>>>(
            reference_texture.get(), source_images_texture.get(),
            use_source_depth_maps ? source_depth_maps_texture.get() : 0,
            poses_texture.get(), d_source_images_info.get(),
            d_depth_map.get(), d_normal_map.get(), d_cost_map.get(),
            device_options, direction, d_counters.get());
        CudaCheck(cudaGetLastError());
      };

      switch (request.options.propagation_schedule) {
        case PatchMatchPropagationSchedule::kForwardBackward:
          run_direction(0);
          run_direction(1);
          break;
        case PatchMatchPropagationSchedule::kFourDirection:
        case PatchMatchPropagationSchedule::kCheckerboard:
          run_direction(0);
          run_direction(1);
          run_direction(2);
          run_direction(3);
          break;
      }

      const float perturb_depth_radius =
          std::max(request.options.min_depth_perturbation, depth_radius);
      const float perturb_normal_radius =
          std::max(request.options.min_normal_radius_degrees, normal_radius);
      PerturbationKernel<<<blocks, kThreadsPerBlock>>>(
          reference_texture.get(), source_images_texture.get(),
          use_source_depth_maps ? source_depth_maps_texture.get() : 0,
          poses_texture.get(), d_source_images_info.get(), d_depth_map.get(),
          d_normal_map.get(), d_cost_map.get(), device_options,
          request.options.perturbation.random_seed +
              static_cast<std::uint32_t>(iteration),
          perturb_depth_radius,
          perturb_normal_radius,
          static_cast<int>(request.options.perturbation.num_samples_per_pixel),
          d_counters.get());
      CudaCheck(cudaGetLastError());

      CudaCheck(cudaDeviceSynchronize());
      iteration_counters.push_back(d_counters.CopyScalarToHost());
      iteration_depth_radii.push_back(perturb_depth_radius);
      iteration_normal_radii.push_back(perturb_normal_radius);

      depth_radius *= request.options.perturbation_shrink;
      normal_radius *= request.options.perturbation_shrink;
      (void)normal_radius;
    }
    CudaCheck(cudaDeviceSynchronize());

    const std::vector<float> depth_values = d_depth_map.CopyToHost();
    const std::vector<float> normal_values = d_normal_map.CopyToHost();
    const std::vector<float> cost_values = d_cost_map.CopyToHost();

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
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        const std::size_t idx =
            static_cast<std::size_t>(row) * width + static_cast<std::size_t>(col);
        result.depth_map.Set(row, col, depth_values[idx]);
        result.normal_map.SetNormalized(
            row, col,
            Eigen::Vector3f(normal_values[idx * 3 + 0],
                            normal_values[idx * 3 + 1],
                            normal_values[idx * 3 + 2]));
        result.cost_map.Set(row, col, cost_values[idx]);
      }
    }

    result.summary.requested_backend = request.options.backend;
    result.summary.actual_backend = PatchMatchBackend::kCuda;
    result.summary.num_pyramid_levels = 1;
    result.summary.num_iterations = request.options.num_iterations;
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
            request.options.num_iterations * 4;
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
      result.summary.num_propagation_updates += propagation_updates;
      result.summary.num_perturbation_updates += perturbation_updates;
      result.summary.num_source_selection_tests +=
          static_cast<std::size_t>(counters.source_selection_tests);
      result.summary.num_source_selection_acceptances +=
          static_cast<std::size_t>(counters.source_selection_acceptances);
      result.summary.num_source_selection_rejections +=
          static_cast<std::size_t>(counters.source_selection_rejections);
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
    if (result.summary.num_perturbation_updates > 0) {
      result.summary.perturbation_mean_abs_depth_update =
          result.summary.perturbation_total_abs_depth_update /
          static_cast<double>(result.summary.num_perturbation_updates);
      result.summary.perturbation_mean_normal_angle_degrees =
          result.summary.perturbation_total_normal_angle_degrees /
          static_cast<double>(result.summary.num_perturbation_updates);
    }
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
    result.summary.num_valid_cost_pixels = result.cost_map.ValidPixelCount();
    return result;
  } catch (const std::exception& error) {
    SetUnavailableReason(unavailable_reason,
                         std::string("CUDA PatchMatch failed: ") +
                             error.what());
    return std::nullopt;
  }
}

}  // namespace

std::optional<PatchMatchStereoResult> RunCudaPatchMatchStereo(
    const CudaPatchMatchRequest& request,
    std::string* unavailable_reason) {
  if (request.options.pyramid_levels <= 1) {
    std::optional<PatchMatchStereoResult> result =
        RunCudaPatchMatchStereoSingleScale(request, unavailable_reason);
    if (result.has_value()) {
      result->summary.num_pyramid_levels = 1;
    }
    return result;
  }

  try {
    ValidateRequest(request);
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
                                             unavailable_reason);
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

    for (std::size_t level_idx = 0; level_idx < num_levels; ++level_idx) {
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
      level_request.initial_maps = level_initial_maps_ptr;
      level_request.depth_min = request.depth_min;
      level_request.depth_max = request.depth_max;
      level_request.options = request.options;
      level_request.options.pyramid_levels = 1;
      level_request.options.initialization.random_seed +=
          static_cast<std::uint32_t>(level_idx * 101U);
      level_request.options.perturbation.random_seed +=
          static_cast<std::uint32_t>(level_idx * 211U);

      std::optional<PatchMatchStereoResult> level_result =
          RunCudaPatchMatchStereoSingleScale(level_request,
                                             unavailable_reason);
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
