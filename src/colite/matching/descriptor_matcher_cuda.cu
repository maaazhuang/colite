#include "descriptor_matcher_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <mutex>
#include <map>
#include <sstream>
#include <stdexcept>

namespace colite::cuda_matcher {
namespace {

int& SelectedDevice() { static int device = 0; return device; }

void CheckCuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}

void CheckCublas(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  std::ostringstream message;
  message << operation << " failed with cuBLAS status "
          << static_cast<int>(status);
  throw std::runtime_error(message.str());
}

class CudaDescriptorBuffer final : public DescriptorBuffer {
 public:
  CudaDescriptorBuffer(const std::uint8_t* descriptors,
                       int rows,
                       int columns,
                       const float* keypoints_xy)
      : rows_(rows), columns_(columns) {
    CheckCuda(cudaGetDevice(&device_), "cudaGetDevice buffer");
    if (rows < 0 || columns < 0) {
      throw std::invalid_argument("Descriptor dimensions must be non-negative.");
    }
    try {
      const std::size_t descriptor_bytes =
          static_cast<std::size_t>(rows) * columns;
      if (descriptor_bytes > 0) {
        CheckCuda(cudaMalloc(&data_, descriptor_bytes),
                  "cudaMalloc descriptors");
        CheckCuda(cudaMemcpy(data_, descriptors, descriptor_bytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy descriptors");
        std::vector<std::int8_t> signed_descriptors(descriptor_bytes);
        std::vector<int> descriptor_sums(static_cast<std::size_t>(rows), 0);
        for (int row = 0; row < rows; ++row) {
          for (int column = 0; column < columns; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * columns + column;
            const std::uint8_t value = descriptors[index];
            signed_descriptors[index] =
                static_cast<std::int8_t>(value ^ 0x80U);
            descriptor_sums[static_cast<std::size_t>(row)] += value;
          }
        }
        const std::size_t signed_bytes = static_cast<std::size_t>(PaddedRows()) * columns;
        CheckCuda(cudaMalloc(&signed_data_, signed_bytes),
                  "cudaMalloc signed descriptors");
        CheckCuda(cudaMemset(signed_data_, 0, signed_bytes), "cudaMemset descriptor padding");
        CheckCuda(cudaMalloc(&descriptor_sums_,
                             descriptor_sums.size() * sizeof(int)),
                  "cudaMalloc descriptor sums");
        CheckCuda(cudaMemcpy(signed_data_, signed_descriptors.data(),
                             descriptor_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy signed descriptors");
        CheckCuda(cudaMemcpy(descriptor_sums_, descriptor_sums.data(),
                             descriptor_sums.size() * sizeof(int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy descriptor sums");
        if (keypoints_xy != nullptr) {
          const std::size_t keypoint_bytes =
              static_cast<std::size_t>(rows) * 2 * sizeof(float);
          CheckCuda(cudaMalloc(&keypoints_xy_, keypoint_bytes),
                    "cudaMalloc matcher keypoints");
          CheckCuda(cudaMemcpy(keypoints_xy_, keypoints_xy, keypoint_bytes,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy matcher keypoints");
        }
      }
    } catch (...) {
      cudaFree(data_);
      cudaFree(signed_data_);
      cudaFree(descriptor_sums_);
      cudaFree(keypoints_xy_);
      data_ = nullptr;
      signed_data_ = nullptr;
      descriptor_sums_ = nullptr;
      keypoints_xy_ = nullptr;
      throw;
    }
  }

  ~CudaDescriptorBuffer() override {
    cudaSetDevice(device_);
    if (data_ != nullptr) cudaFree(data_);
    if (signed_data_ != nullptr) cudaFree(signed_data_);
    if (descriptor_sums_ != nullptr) cudaFree(descriptor_sums_);
    if (keypoints_xy_ != nullptr) cudaFree(keypoints_xy_);
  }
  int Rows() const override { return rows_; }
  int PaddedRows() const { return (rows_ + 3) / 4 * 4; }
  int Columns() const override { return columns_; }
  const std::uint8_t* Data() const { return data_; }
  const std::int8_t* SignedData() const { return signed_data_; }
  const int* DescriptorSums() const { return descriptor_sums_; }
  const float* Keypoints() const { return keypoints_xy_; }
  bool HasKeypoints() const { return keypoints_xy_ != nullptr; }

 private:
  int device_ = 0;
  int rows_ = 0;
  int columns_ = 0;
  std::uint8_t* data_ = nullptr;
  std::int8_t* signed_data_ = nullptr;
  int* descriptor_sums_ = nullptr;
  float* keypoints_xy_ = nullptr;
};

template <typename T>
void EnsureDeviceCapacity(T** data,
                          std::size_t* capacity,
                          const std::size_t required,
                          const char* operation) {
  if (required <= *capacity) return;
  T* replacement = nullptr;
  CheckCuda(cudaMalloc(&replacement, required * sizeof(T)), operation);
  if (*data != nullptr) {
    CheckCuda(cudaFree(*data), "cudaFree replaced matcher workspace buffer");
  }
  *data = replacement;
  *capacity = required;
}

class CudaMatcherWorkspace {
 public:
  CudaMatcherWorkspace() {
    CheckCublas(cublasCreate(&cublas_handle_), "cublasCreate");
  }

  ~CudaMatcherWorkspace() {
    cudaFree(dot_matrix_);
    cudaFree(forward_output_);
    cudaFree(backward_output_);
    cudaFree(candidate_offsets_);
    cudaFree(candidate_indices_);
    cudaFree(column_partials_);
    cublasDestroy(cublas_handle_);
  }

  void EnsureMutual(const std::size_t matrix_values,
                    const std::size_t forward_values,
                    const std::size_t backward_values) {
    EnsureDeviceCapacity(&column_partials_, &column_partials_capacity_,
        ((forward_values + 255) / 256) * backward_values, "cudaMalloc column partials");
    EnsureDeviceCapacity(&dot_matrix_, &dot_matrix_capacity_, matrix_values,
                         "cudaMalloc matcher dot matrix workspace");
    EnsureDeviceCapacity(&forward_output_, &forward_output_capacity_,
                         forward_values,
                         "cudaMalloc matcher forward workspace");
    EnsureDeviceCapacity(&backward_output_, &backward_output_capacity_,
                         backward_values,
                         "cudaMalloc matcher backward workspace");
  }

  void EnsureGuided(const std::size_t offset_values,
                    const std::size_t index_values,
                    const std::size_t output_values) {
    EnsureDeviceCapacity(&candidate_offsets_, &candidate_offsets_capacity_,
                         offset_values,
                         "cudaMalloc guided offset workspace");
    EnsureDeviceCapacity(&candidate_indices_, &candidate_indices_capacity_,
                         index_values,
                         "cudaMalloc guided index workspace");
    EnsureDeviceCapacity(&forward_output_, &forward_output_capacity_,
                         output_values,
                         "cudaMalloc guided output workspace");
  }

  cublasHandle_t CublasHandle() const { return cublas_handle_; }
  int* DotMatrix() const { return dot_matrix_; }
  BestMatch* ForwardOutput() const { return forward_output_; }
  BestMatch* BackwardOutput() const { return backward_output_; }
  BestMatch* ColumnPartials() const { return column_partials_; }
  int* CandidateOffsets() const { return candidate_offsets_; }
  int* CandidateIndices() const { return candidate_indices_; }

 private:
  cublasHandle_t cublas_handle_ = nullptr;
  int* dot_matrix_ = nullptr;
  BestMatch* forward_output_ = nullptr;
  BestMatch* backward_output_ = nullptr;
  BestMatch* column_partials_ = nullptr;
  std::size_t column_partials_capacity_ = 0;
  int* candidate_offsets_ = nullptr;
  int* candidate_indices_ = nullptr;
  std::size_t dot_matrix_capacity_ = 0;
  std::size_t forward_output_capacity_ = 0;
  std::size_t backward_output_capacity_ = 0;
  std::size_t candidate_offsets_capacity_ = 0;
  std::size_t candidate_indices_capacity_ = 0;
};

__global__ void FindBestKernel(const std::uint8_t* query,
                               int num_query,
                               const std::uint8_t* train,
                               int num_train,
                               int dimensions,
                               BestMatch* output) {
  constexpr int kWarpsPerBlock = 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int query_idx = blockIdx.x * kWarpsPerBlock + warp;
  if (query_idx >= num_query) return;

  int best_dot = INT_MIN;
  int second_best_dot = INT_MIN;
  int best_idx = -1;
  const std::uint8_t* query_row = query + query_idx * dimensions;
  const int words = dimensions / 4;
  for (int train_idx = 0; train_idx < num_train; ++train_idx) {
    const std::uint8_t* train_row = train + train_idx * dimensions;
    unsigned int dot = 0;
    for (int word = lane; word < words; word += 32) {
      const unsigned int lhs =
          reinterpret_cast<const unsigned int*>(query_row)[word];
      const unsigned int rhs =
          reinterpret_cast<const unsigned int*>(train_row)[word];
      dot = __dp4a(lhs, rhs, dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffU, dot, offset);
    }
    if (lane == 0) {
      const int signed_dot = static_cast<int>(dot);
      if (signed_dot > best_dot) {
        second_best_dot = best_dot;
        best_idx = train_idx;
        best_dot = signed_dot;
      } else if (signed_dot > second_best_dot) {
        second_best_dot = signed_dot;
      }
    }
  }
  if (lane == 0) output[query_idx] = {best_idx, best_dot, second_best_dot};
}

__device__ void InsertBestTwo(const int score, const int index,
                             int* best, int* best_index,
                             int* second, int* second_index) {
  if (index < 0 || index == *best_index || index == *second_index) return;
  if (score > *best || (score == *best && index < *best_index)) {
    *second = *best;
    *second_index = *best_index;
    *best = score;
    *best_index = index;
  } else if (score > *second || (score == *second && index < *second_index)) {
    *second = score;
    *second_index = index;
  }
}

// One warp scans one row with contiguous loads. One thread per row caused
// strided global-memory transactions across the warp and long serial scans.
template <bool Guided>
__global__ void ReduceDotMatrixRowsKernel(const int* dot_matrix,
                                          int num_rows,
                                          int num_columns,
                                          int matrix_stride,
                                          int dimensions,
                                          const int* row_sums,
                                          const int* column_sums,
                                          BestMatch* output) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (row >= num_rows) return;
  int best_idx = -1;
  int second_best_idx = -1;
  int best_dot = INT_MIN;
  int second_best_dot = INT_MIN;
  const int* values =
      dot_matrix + static_cast<std::size_t>(row) * matrix_stride;
  for (int column = lane; column < num_columns; column += 32) {
    int dot = values[column];
    if constexpr (Guided) {
      if (dot < 0) continue;
    } else {
      dot += 128 * column_sums[column];
    }
    InsertBestTwo(dot, column, &best_dot, &best_idx,
                  &second_best_dot, &second_best_idx);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    const int other_best = __shfl_down_sync(0xFFFFFFFF, best_dot, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
    const int other_second = __shfl_down_sync(0xFFFFFFFF, second_best_dot, offset);
    const int other_second_index = __shfl_down_sync(0xFFFFFFFF, second_best_idx, offset);
    if (lane + offset < 32) {
      InsertBestTwo(other_best, other_index, &best_dot, &best_idx,
                    &second_best_dot, &second_best_idx);
      InsertBestTwo(other_second, other_second_index, &best_dot, &best_idx,
                    &second_best_dot, &second_best_idx);
    }
  }
  if (lane == 0) {
    int correction = 0;
    if constexpr (!Guided) correction = 128 * row_sums[row] - dimensions * 128 * 128;
    output[row] = {best_idx, best_idx >= 0 ? best_dot + correction : -1,
                   second_best_idx >= 0 ? second_best_dot + correction : -1};
  }
}

// Split long columns into independent tiles. A single thread scanning an
// entire 8K-16K column leaves too few blocks to hide memory latency.
template <bool Guided>
__global__ void ReduceColumnTilesKernel(const int* dots, int rows, int columns, int matrix_stride,
    const int* row_sums, BestMatch* partials) {
  const int column = blockIdx.x * 32 + threadIdx.x;
  const int lane_row = threadIdx.y;
  int best = INT_MIN, second = INT_MIN, best_index = -1, second_index = -1;
  for (int row = blockIdx.y * 256 + lane_row;
       row < rows && row < (blockIdx.y + 1) * 256; row += 8) {
    if (column < columns) {
      int dot = dots[static_cast<std::size_t>(row) * matrix_stride + column];
      if constexpr (Guided) { if (dot < 0) continue; }
      else dot += 128 * row_sums[row];
      InsertBestTwo(dot, row, &best, &best_index, &second, &second_index);
    }
  }
  __shared__ int scores[8][32], indices[8][32], seconds[8][32], second_indices[8][32];
  scores[lane_row][threadIdx.x] = best;
  indices[lane_row][threadIdx.x] = best_index;
  seconds[lane_row][threadIdx.x] = second;
  second_indices[lane_row][threadIdx.x] = second_index;
  __syncthreads();
  if (lane_row == 0 && column < columns) {
    for (int i = 1; i < 8; ++i) {
      InsertBestTwo(scores[i][threadIdx.x], indices[i][threadIdx.x],
          &best, &best_index, &second, &second_index);
      InsertBestTwo(seconds[i][threadIdx.x], second_indices[i][threadIdx.x],
          &best, &best_index, &second, &second_index);
    }
    partials[static_cast<std::size_t>(blockIdx.y) * columns + column] =
        {best_index, best, second};
  }
}

template <bool Guided>
__global__ void FinishColumnTilesKernel(const BestMatch* partials, int tiles,
    int columns, int dimensions, const int* column_sums, BestMatch* output) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= columns) return;
  int best = INT_MIN, second = INT_MIN, index = -1;
  for (int tile = 0; tile < tiles; ++tile) {
    const auto candidate = partials[static_cast<std::size_t>(tile) * columns + column];
    if (candidate.index < 0) continue;
    if (candidate.best_dot > best ||
        (candidate.best_dot == best && (index < 0 || candidate.index < index))) {
      second = max(second, best);
      best = candidate.best_dot;
      index = candidate.index;
    } else second = max(second, candidate.best_dot);
    second = max(second, candidate.second_best_dot);
  }
  int correction = 0;
  if constexpr (!Guided) correction = 128 * column_sums[column] - dimensions * 128 * 128;
  output[column] = {index, index >= 0 ? best + correction : -1,
                   second != INT_MIN ? second + correction : -1};
}

__global__ void FindBestGuidedKernel(const std::uint8_t* query,
                                     int num_query,
                                     const std::uint8_t* train,
                                     int dimensions,
                                     const int* candidate_offsets,
                                     const int* candidate_indices,
                                     BestMatch* output) {
  constexpr int kWarpsPerBlock = 8;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int query_idx = blockIdx.x * kWarpsPerBlock + warp;
  if (query_idx >= num_query) return;

  int best_dot = -1;
  int second_best_dot = -1;
  int best_idx = -1;
  const std::uint8_t* query_row = query + query_idx * dimensions;
  const int words = dimensions / 4;
  for (int candidate_idx = candidate_offsets[query_idx];
       candidate_idx < candidate_offsets[query_idx + 1]; ++candidate_idx) {
    const int train_idx = candidate_indices[candidate_idx];
    const std::uint8_t* train_row = train + train_idx * dimensions;
    unsigned int dot = 0;
    for (int word = lane; word < words; word += 32) {
      const unsigned int lhs =
          reinterpret_cast<const unsigned int*>(query_row)[word];
      const unsigned int rhs =
          reinterpret_cast<const unsigned int*>(train_row)[word];
      dot = __dp4a(lhs, rhs, dot);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_down_sync(0xffffffffU, dot, offset);
    }
    if (lane == 0) {
      const int signed_dot = static_cast<int>(dot);
      if (signed_dot > best_dot) {
        best_idx = train_idx;
        second_best_dot = best_dot;
        best_dot = signed_dot;
      } else if (signed_dot > second_best_dot) {
        second_best_dot = signed_dot;
      }
    }
  }
  if (lane == 0) output[query_idx] = {best_idx, best_dot, second_best_dot};
}

struct FundamentalMatrixF {
  float data[9];
};

// Tensor-core GEMM supplies the exact integer dots. Geometry is a bandwidth
// pass; do not recompute 128 descriptor components with scalar DP4A here.
__global__ void MaskGuidedDotMatrixKernel(int* dots, int rows, int columns, int matrix_stride,
    const float* keypoints1, const float* keypoints2, const int* sums1,
    const int* sums2, int dimensions, FundamentalMatrixF fundamental,
    float max_squared_error) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (column >= columns || row >= rows) return;
  const float x = keypoints1[2 * row], y = keypoints1[2 * row + 1];
  const float u = keypoints2[2 * column], v = keypoints2[2 * column + 1];
  const float* f = fundamental.data;
  const float fx = f[0] * x + f[1] * y + f[2];
  const float fy = f[3] * x + f[4] * y + f[5];
  const float fz = f[6] * x + f[7] * y + f[8];
  const float ftx = f[0] * u + f[3] * v + f[6];
  const float fty = f[1] * u + f[4] * v + f[7];
  const float residual = u * fx + v * fy + fz;
  const float denominator = fx * fx + fy * fy + ftx * ftx + fty * fty;
  const auto index = static_cast<std::size_t>(row) * matrix_stride + column;
  if (denominator > 0 && isfinite(denominator) &&
      residual * residual <= max_squared_error * denominator) {
    dots[index] += 128 * (sums1[row] + sums2[column]) - dimensions * 128 * 128;
  } else dots[index] = -1;
}

std::mutex& RuntimeMutex() {
  static std::mutex mutex;
  return mutex;
}

CudaMatcherWorkspace& RuntimeWorkspace() {
  static auto* workspaces = new std::map<int, std::unique_ptr<CudaMatcherWorkspace>>();
  auto& workspace = (*workspaces)[SelectedDevice()];
  if (!workspace) workspace = std::make_unique<CudaMatcherWorkspace>();
  return *workspace;
}

}  // namespace

void SetDevice(int device) {
  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(device), "FeatureMatching.gpu_index");
  SelectedDevice() = device;
}

bool IsAvailable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<DescriptorBuffer> CreateBuffer(const std::uint8_t* descriptors,
                                               int rows,
                                               int columns,
                                               const float* keypoints_xy) {
  if (!IsAvailable()) {
    throw std::runtime_error("No CUDA device is available for descriptor matching.");
  }
  if (columns % 4 != 0) {
    throw std::invalid_argument(
        "CUDA descriptor matcher requires dimensions divisible by four.");
  }
  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(SelectedDevice()), "cudaSetDevice matcher");
  return std::make_unique<CudaDescriptorBuffer>(descriptors, rows, columns,
                                                keypoints_xy);
}

std::vector<BestMatch> FindBestMatches(const DescriptorBuffer& query_base,
                                       const DescriptorBuffer& train_base) {
  const auto* query = dynamic_cast<const CudaDescriptorBuffer*>(&query_base);
  const auto* train = dynamic_cast<const CudaDescriptorBuffer*>(&train_base);
  if (query == nullptr || train == nullptr ||
      query->Columns() != train->Columns()) {
    throw std::invalid_argument("Incompatible CUDA descriptor buffers.");
  }
  std::vector<BestMatch> result(static_cast<std::size_t>(query->Rows()));
  if (query->Rows() == 0 || train->Rows() == 0) return result;

  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(SelectedDevice()), "cudaSetDevice matcher");
  CudaMatcherWorkspace& workspace = RuntimeWorkspace();
  workspace.EnsureGuided(/*offset_values=*/0, /*index_values=*/0,
                         result.size());
  BestMatch* device_output = workspace.ForwardOutput();
  const std::size_t bytes = result.size() * sizeof(BestMatch);
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int blocks = (query->Rows() + kWarpsPerBlock - 1) / kWarpsPerBlock;
  FindBestKernel<<<blocks, kThreads>>>(
      query->Data(), query->Rows(), train->Data(), train->Rows(),
      query->Columns(), device_output);
  CheckCuda(cudaGetLastError(), "CUDA descriptor match kernel launch");
  CheckCuda(cudaMemcpy(result.data(), device_output, bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy match output");
  return result;
}

MutualBestMatches FindBestMatchesMutual(
    const DescriptorBuffer& features1_base,
    const DescriptorBuffer& features2_base) {
  const auto* features1 =
      dynamic_cast<const CudaDescriptorBuffer*>(&features1_base);
  const auto* features2 =
      dynamic_cast<const CudaDescriptorBuffer*>(&features2_base);
  if (features1 == nullptr || features2 == nullptr ||
      features1->Columns() != features2->Columns()) {
    throw std::invalid_argument("Incompatible CUDA descriptor buffers.");
  }

  MutualBestMatches result;
  result.forward.resize(static_cast<std::size_t>(features1->Rows()));
  result.backward.resize(static_cast<std::size_t>(features2->Rows()));
  if (features1->Rows() == 0 || features2->Rows() == 0) return result;

  const std::size_t matrix_values =
      static_cast<std::size_t>(features1->PaddedRows()) * features2->PaddedRows();
  const std::size_t forward_bytes =
      result.forward.size() * sizeof(BestMatch);
  const std::size_t backward_bytes =
      result.backward.size() * sizeof(BestMatch);

  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(SelectedDevice()), "cudaSetDevice matcher");
  CudaMatcherWorkspace& workspace = RuntimeWorkspace();
  workspace.EnsureMutual(matrix_values, result.forward.size(),
                         result.backward.size());
  int* device_matrix = workspace.DotMatrix();
  BestMatch* device_forward = workspace.ForwardOutput();
  BestMatch* device_backward = workspace.BackwardOutput();

  const int alpha = 1;
  const int beta = 0;
  CheckCublas(
      cublasGemmEx(
          workspace.CublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
          features2->PaddedRows(), features1->PaddedRows(), features1->Columns(), &alpha,
          features2->SignedData(), CUDA_R_8I, features2->Columns(),
          features1->SignedData(), CUDA_R_8I, features1->Columns(), &beta,
          device_matrix, CUDA_R_32I, features2->PaddedRows(), CUBLAS_COMPUTE_32I,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx mutual descriptor dots");

  constexpr int kReductionThreads = 256;
  ReduceDotMatrixRowsKernel<false><<<
      (features1->Rows() + kReductionThreads / 32 - 1) / (kReductionThreads / 32),
      kReductionThreads>>>(
      device_matrix, features1->Rows(), features2->Rows(), features2->PaddedRows(),
      features1->Columns(), features1->DescriptorSums(),
      features2->DescriptorSums(), device_forward);
  CheckCuda(cudaGetLastError(), "CUDA mutual row reduction kernel launch");
  const int tiles = (features1->Rows() + 255) / 256;
  ReduceColumnTilesKernel<false><<<dim3((features2->Rows() + 31) / 32, tiles), dim3(32, 8)>>>(
      device_matrix, features1->Rows(), features2->Rows(), features2->PaddedRows(), features1->DescriptorSums(), workspace.ColumnPartials());
  FinishColumnTilesKernel<false><<<(features2->Rows() + 255) / 256, 256>>>(
      workspace.ColumnPartials(), tiles, features2->Rows(), features1->Columns(),
      features2->DescriptorSums(), device_backward);
  CheckCuda(cudaGetLastError(), "CUDA mutual column reduction kernel launch");

  CheckCuda(cudaMemcpy(result.forward.data(), device_forward, forward_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy mutual forward output");
  CheckCuda(cudaMemcpy(result.backward.data(), device_backward, backward_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy mutual backward output");
  return result;
}

std::vector<BestMatch> FindBestMatchesGuided(
    const DescriptorBuffer& query_base,
    const DescriptorBuffer& train_base,
    const std::vector<int>& candidate_offsets,
    const std::vector<int>& candidate_indices) {
  const auto* query = dynamic_cast<const CudaDescriptorBuffer*>(&query_base);
  const auto* train = dynamic_cast<const CudaDescriptorBuffer*>(&train_base);
  if (query == nullptr || train == nullptr ||
      query->Columns() != train->Columns()) {
    throw std::invalid_argument("Incompatible CUDA descriptor buffers.");
  }
  if (candidate_offsets.size() !=
          static_cast<std::size_t>(query->Rows()) + 1 ||
      candidate_offsets.empty() || candidate_offsets.front() != 0 ||
      candidate_offsets.back() !=
          static_cast<int>(candidate_indices.size())) {
    throw std::invalid_argument("Invalid guided candidate CSR data.");
  }
  for (std::size_t idx = 1; idx < candidate_offsets.size(); ++idx) {
    if (candidate_offsets[idx] < candidate_offsets[idx - 1]) {
      throw std::invalid_argument(
          "Guided candidate offsets must be non-decreasing.");
    }
  }
  for (const int train_idx : candidate_indices) {
    if (train_idx < 0 || train_idx >= train->Rows()) {
      throw std::invalid_argument("Guided candidate index is out of range.");
    }
  }
  std::vector<BestMatch> result(static_cast<std::size_t>(query->Rows()));
  if (query->Rows() == 0 || train->Rows() == 0 ||
      candidate_indices.empty()) {
    return result;
  }

  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(SelectedDevice()), "cudaSetDevice matcher");
  CudaMatcherWorkspace& workspace = RuntimeWorkspace();
  workspace.EnsureGuided(candidate_offsets.size(), candidate_indices.size(),
                         result.size());
  int* device_offsets = workspace.CandidateOffsets();
  int* device_indices = workspace.CandidateIndices();
  BestMatch* device_output = workspace.ForwardOutput();
  const std::size_t offsets_bytes =
      candidate_offsets.size() * sizeof(int);
  const std::size_t indices_bytes =
      candidate_indices.size() * sizeof(int);
  const std::size_t output_bytes = result.size() * sizeof(BestMatch);
  CheckCuda(cudaMemcpy(device_offsets, candidate_offsets.data(), offsets_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy guided offsets");
  CheckCuda(cudaMemcpy(device_indices, candidate_indices.data(), indices_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy guided indices");
  constexpr int kThreads = 256;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int blocks =
      (query->Rows() + kWarpsPerBlock - 1) / kWarpsPerBlock;
  FindBestGuidedKernel<<<blocks, kThreads>>>(
      query->Data(), query->Rows(), train->Data(), query->Columns(),
      device_offsets, device_indices, device_output);
  CheckCuda(cudaGetLastError(), "CUDA guided descriptor match kernel launch");
  CheckCuda(cudaMemcpy(result.data(), device_output, output_bytes,
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy guided output");
  return result;
}

MutualBestMatches FindBestMatchesGuidedFundamental(
    const DescriptorBuffer& features1_base,
    const DescriptorBuffer& features2_base,
    const double* fundamental_matrix_col_major,
    const double max_sampson_error_px) {
  const auto* features1 =
      dynamic_cast<const CudaDescriptorBuffer*>(&features1_base);
  const auto* features2 =
      dynamic_cast<const CudaDescriptorBuffer*>(&features2_base);
  if (features1 == nullptr || features2 == nullptr ||
      features1->Columns() != features2->Columns() ||
      features1->Columns() != 128 || !features1->HasKeypoints() ||
      !features2->HasKeypoints() || fundamental_matrix_col_major == nullptr ||
      max_sampson_error_px < 0.0) {
    throw std::invalid_argument(
        "Incompatible CUDA buffers or fundamental matrix for guided "
        "matching.");
  }

  MutualBestMatches result;
  result.forward.resize(static_cast<std::size_t>(features1->Rows()));
  result.backward.resize(static_cast<std::size_t>(features2->Rows()));
  if (features1->Rows() == 0 || features2->Rows() == 0) return result;

  FundamentalMatrixF fundamental;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      fundamental.data[row * 3 + column] =
          static_cast<float>(
              fundamental_matrix_col_major[row + column * 3]);
    }
  }

  const std::size_t matrix_values =
      static_cast<std::size_t>(features1->PaddedRows()) * features2->PaddedRows();
  const std::size_t forward_bytes =
      result.forward.size() * sizeof(BestMatch);
  const std::size_t backward_bytes =
      result.backward.size() * sizeof(BestMatch);
  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CheckCuda(cudaSetDevice(SelectedDevice()), "cudaSetDevice matcher");
  CudaMatcherWorkspace& workspace = RuntimeWorkspace();
  workspace.EnsureMutual(matrix_values, result.forward.size(),
                         result.backward.size());

  const int alpha = 1, beta = 0;
  CheckCublas(cublasGemmEx(workspace.CublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
      features2->PaddedRows(), features1->PaddedRows(), features1->Columns(), &alpha,
      features2->SignedData(), CUDA_R_8I, features2->Columns(),
      features1->SignedData(), CUDA_R_8I, features1->Columns(), &beta,
      workspace.DotMatrix(), CUDA_R_32I, features2->PaddedRows(), CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP), "cublasGemmEx guided descriptor dots");
  constexpr int kMaskThreads = 256;
  const dim3 blocks((features2->Rows() + kMaskThreads - 1) / kMaskThreads, features1->Rows());
  MaskGuidedDotMatrixKernel<<<blocks, kMaskThreads>>>(workspace.DotMatrix(),
      features1->Rows(), features2->Rows(), features2->PaddedRows(), features1->Keypoints(), features2->Keypoints(),
      features1->DescriptorSums(), features2->DescriptorSums(), features1->Columns(),
      fundamental, static_cast<float>(max_sampson_error_px * max_sampson_error_px));
  CheckCuda(cudaGetLastError(),
            "CUDA fused guided dot matrix kernel launch");

  constexpr int kReductionThreads = 256;
  ReduceDotMatrixRowsKernel<true><<<
      (features1->Rows() + kReductionThreads / 32 - 1) / (kReductionThreads / 32),
      kReductionThreads>>>(workspace.DotMatrix(), features1->Rows(),
                           features2->Rows(), features2->PaddedRows(), 0, nullptr, nullptr,
                           workspace.ForwardOutput());
  CheckCuda(cudaGetLastError(),
            "CUDA fused guided row reduction kernel launch");
  const int tiles = (features1->Rows() + 255) / 256;
  ReduceColumnTilesKernel<true><<<dim3((features2->Rows() + 31) / 32, tiles), dim3(32, 8)>>>(
      workspace.DotMatrix(), features1->Rows(), features2->Rows(), features2->PaddedRows(), nullptr, workspace.ColumnPartials());
  FinishColumnTilesKernel<true><<<(features2->Rows() + 255) / 256, 256>>>(
      workspace.ColumnPartials(), tiles, features2->Rows(), 0, nullptr, workspace.BackwardOutput());
  CheckCuda(cudaGetLastError(),
            "CUDA fused guided column reduction kernel launch");

  CheckCuda(cudaMemcpy(result.forward.data(), workspace.ForwardOutput(),
                       forward_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy fused guided forward output");
  CheckCuda(cudaMemcpy(result.backward.data(), workspace.BackwardOutput(),
                       backward_bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy fused guided backward output");
  return result;
}

}  // namespace colite::cuda_matcher
