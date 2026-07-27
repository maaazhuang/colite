#include "descriptor_matcher_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace colite::cuda_matcher {
namespace {

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
        CheckCuda(cudaMalloc(&signed_data_, descriptor_bytes),
                  "cudaMalloc signed descriptors");
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
    if (data_ != nullptr) cudaFree(data_);
    if (signed_data_ != nullptr) cudaFree(signed_data_);
    if (descriptor_sums_ != nullptr) cudaFree(descriptor_sums_);
    if (keypoints_xy_ != nullptr) cudaFree(keypoints_xy_);
  }
  int Rows() const override { return rows_; }
  int Columns() const override { return columns_; }
  const std::uint8_t* Data() const { return data_; }
  const std::int8_t* SignedData() const { return signed_data_; }
  const int* DescriptorSums() const { return descriptor_sums_; }
  const float* Keypoints() const { return keypoints_xy_; }
  bool HasKeypoints() const { return keypoints_xy_ != nullptr; }

 private:
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
    cublasDestroy(cublas_handle_);
  }

  void EnsureMutual(const std::size_t matrix_values,
                    const std::size_t forward_values,
                    const std::size_t backward_values) {
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
  int* CandidateOffsets() const { return candidate_offsets_; }
  int* CandidateIndices() const { return candidate_indices_; }

 private:
  cublasHandle_t cublas_handle_ = nullptr;
  int* dot_matrix_ = nullptr;
  BestMatch* forward_output_ = nullptr;
  BestMatch* backward_output_ = nullptr;
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

__global__ void ReduceDotMatrixRowsKernel(const int* dot_matrix,
                                          int num_rows,
                                          int num_columns,
                                          int dimensions,
                                          const int* row_sums,
                                          const int* column_sums,
                                          BestMatch* output) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  int best_idx = -1;
  int second_best_idx = -1;
  int best_dot = INT_MIN;
  int second_best_dot = INT_MIN;
  const int* values =
      dot_matrix + static_cast<std::size_t>(row) * num_columns;
  for (int column = 0; column < num_columns; ++column) {
    const int dot = values[column] + 128 * column_sums[column];
    if (dot > best_dot) {
      second_best_idx = best_idx;
      best_idx = column;
      second_best_dot = best_dot;
      best_dot = dot;
    } else if (dot > second_best_dot) {
      second_best_idx = column;
      second_best_dot = dot;
    }
  }
  const int row_correction = 128 * row_sums[row] - dimensions * 128 * 128;
  best_dot += row_correction;
  second_best_dot =
      second_best_idx >= 0 ? second_best_dot + row_correction : -1;
  output[row] = {best_idx, best_dot, second_best_dot};
}

__global__ void ReduceDotMatrixColumnsKernel(const int* dot_matrix,
                                             int num_rows,
                                             int num_columns,
                                             int dimensions,
                                             const int* row_sums,
                                             const int* column_sums,
                                             BestMatch* output) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= num_columns) return;
  int best_idx = -1;
  int second_best_idx = -1;
  int best_dot = INT_MIN;
  int second_best_dot = INT_MIN;
  for (int row = 0; row < num_rows; ++row) {
    const int dot =
        dot_matrix[static_cast<std::size_t>(row) * num_columns + column] +
        128 * row_sums[row];
    if (dot > best_dot) {
      second_best_idx = best_idx;
      best_idx = row;
      second_best_dot = best_dot;
      best_dot = dot;
    } else if (dot > second_best_dot) {
      second_best_idx = row;
      second_best_dot = dot;
    }
  }
  const int column_correction =
      128 * column_sums[column] - dimensions * 128 * 128;
  best_dot += column_correction;
  second_best_dot =
      second_best_idx >= 0 ? second_best_dot + column_correction : -1;
  output[column] = {best_idx, best_dot, second_best_dot};
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

__global__ void ComputeGuidedDotMatrixKernel(
    const std::uint8_t* features1,
    int num_features1,
    const float* keypoints1,
    const std::uint8_t* features2,
    int num_features2,
    const float* keypoints2,
    int dimensions,
    FundamentalMatrixF fundamental,
    float max_squared_error,
    int* dot_matrix) {
  constexpr int kQueriesPerBlock = 8;
  constexpr int kTrainPerBlock = 128;
  constexpr int kDescriptorWords = 32;
  __shared__ unsigned int query_descriptors[kQueriesPerBlock]
                                                   [kDescriptorWords];
  __shared__ float query_locations[kQueriesPerBlock][2];

  const int query_start = blockIdx.y * kQueriesPerBlock;
  for (int idx = threadIdx.x;
       idx < kQueriesPerBlock * kDescriptorWords;
       idx += blockDim.x) {
    const int local_query = idx / kDescriptorWords;
    const int word = idx % kDescriptorWords;
    const int query_idx = query_start + local_query;
    query_descriptors[local_query][word] =
        query_idx < num_features1
            ? reinterpret_cast<const unsigned int*>(
                  features1 + query_idx * dimensions)[word]
            : 0U;
  }
  if (threadIdx.x < kQueriesPerBlock * 2) {
    const int local_query = threadIdx.x / 2;
    const int coordinate = threadIdx.x % 2;
    const int query_idx = query_start + local_query;
    query_locations[local_query][coordinate] =
        query_idx < num_features1
            ? keypoints1[query_idx * 2 + coordinate]
            : 0.0F;
  }
  __syncthreads();

  const int train_idx = blockIdx.x * kTrainPerBlock + threadIdx.x;
  if (train_idx >= num_features2) return;
  const float train_x = keypoints2[train_idx * 2];
  const float train_y = keypoints2[train_idx * 2 + 1];
  const float* f = fundamental.data;
  const float ftx2_x = f[0] * train_x + f[3] * train_y + f[6];
  const float ftx2_y = f[1] * train_x + f[4] * train_y + f[7];

  bool accepted[kQueriesPerBlock];
  unsigned int dots[kQueriesPerBlock] = {};
#pragma unroll
  for (int local_query = 0; local_query < kQueriesPerBlock; ++local_query) {
    const int query_idx = query_start + local_query;
    const float query_x = query_locations[local_query][0];
    const float query_y = query_locations[local_query][1];
    const float fx1_x = f[0] * query_x + f[1] * query_y + f[2];
    const float fx1_y = f[3] * query_x + f[4] * query_y + f[5];
    const float fx1_z = f[6] * query_x + f[7] * query_y + f[8];
    const float residual = train_x * fx1_x + train_y * fx1_y + fx1_z;
    const float denominator = fx1_x * fx1_x + fx1_y * fx1_y +
                              ftx2_x * ftx2_x + ftx2_y * ftx2_y;
    accepted[local_query] =
        query_idx < num_features1 && denominator > 0.0F &&
        isfinite(denominator) &&
        residual * residual <= max_squared_error * denominator;
  }

  const unsigned int* train_descriptor =
      reinterpret_cast<const unsigned int*>(
          features2 + train_idx * dimensions);
#pragma unroll
  for (int word = 0; word < kDescriptorWords; ++word) {
    const unsigned int rhs = train_descriptor[word];
#pragma unroll
    for (int local_query = 0; local_query < kQueriesPerBlock;
         ++local_query) {
      if (accepted[local_query]) {
        dots[local_query] = __dp4a(
            query_descriptors[local_query][word], rhs, dots[local_query]);
      }
    }
  }

#pragma unroll
  for (int local_query = 0; local_query < kQueriesPerBlock; ++local_query) {
    const int query_idx = query_start + local_query;
    if (query_idx < num_features1) {
      dot_matrix[static_cast<std::size_t>(query_idx) * num_features2 +
                 train_idx] =
          accepted[local_query] ? static_cast<int>(dots[local_query]) : -1;
    }
  }
}

__global__ void ReduceGuidedDotMatrixRowsKernel(const int* dot_matrix,
                                                int num_rows,
                                                int num_columns,
                                                BestMatch* output) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  int best_idx = -1;
  int best_dot = -1;
  int second_best_dot = -1;
  const int* values =
      dot_matrix + static_cast<std::size_t>(row) * num_columns;
  for (int column = 0; column < num_columns; ++column) {
    const int dot = values[column];
    if (dot > best_dot) {
      best_idx = column;
      second_best_dot = best_dot;
      best_dot = dot;
    } else if (dot > second_best_dot) {
      second_best_dot = dot;
    }
  }
  output[row] = {best_idx, best_dot, second_best_dot};
}

__global__ void ReduceGuidedDotMatrixColumnsKernel(const int* dot_matrix,
                                                   int num_rows,
                                                   int num_columns,
                                                   BestMatch* output) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= num_columns) return;
  int best_idx = -1;
  int best_dot = -1;
  int second_best_dot = -1;
  for (int row = 0; row < num_rows; ++row) {
    const int dot =
        dot_matrix[static_cast<std::size_t>(row) * num_columns + column];
    if (dot > best_dot) {
      best_idx = row;
      second_best_dot = best_dot;
      best_dot = dot;
    } else if (dot > second_best_dot) {
      second_best_dot = dot;
    }
  }
  output[column] = {best_idx, best_dot, second_best_dot};
}

std::mutex& RuntimeMutex() {
  static std::mutex mutex;
  return mutex;
}

CudaMatcherWorkspace& RuntimeWorkspace() {
  static CudaMatcherWorkspace* workspace = new CudaMatcherWorkspace();
  return *workspace;
}

}  // namespace

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
      static_cast<std::size_t>(features1->Rows()) * features2->Rows();
  const std::size_t forward_bytes =
      result.forward.size() * sizeof(BestMatch);
  const std::size_t backward_bytes =
      result.backward.size() * sizeof(BestMatch);

  const std::lock_guard<std::mutex> lock(RuntimeMutex());
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
          features2->Rows(), features1->Rows(), features1->Columns(), &alpha,
          features2->SignedData(), CUDA_R_8I, features2->Columns(),
          features1->SignedData(), CUDA_R_8I, features1->Columns(), &beta,
          device_matrix, CUDA_R_32I, features2->Rows(), CUBLAS_COMPUTE_32I,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "cublasGemmEx mutual descriptor dots");

  constexpr int kReductionThreads = 256;
  ReduceDotMatrixRowsKernel<<<
      (features1->Rows() + kReductionThreads - 1) / kReductionThreads,
      kReductionThreads>>>(
      device_matrix, features1->Rows(), features2->Rows(),
      features1->Columns(), features1->DescriptorSums(),
      features2->DescriptorSums(), device_forward);
  CheckCuda(cudaGetLastError(), "CUDA mutual row reduction kernel launch");
  ReduceDotMatrixColumnsKernel<<<
      (features2->Rows() + kReductionThreads - 1) / kReductionThreads,
      kReductionThreads>>>(
      device_matrix, features1->Rows(), features2->Rows(),
      features1->Columns(), features1->DescriptorSums(),
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
      static_cast<std::size_t>(features1->Rows()) * features2->Rows();
  const std::size_t forward_bytes =
      result.forward.size() * sizeof(BestMatch);
  const std::size_t backward_bytes =
      result.backward.size() * sizeof(BestMatch);
  const std::lock_guard<std::mutex> lock(RuntimeMutex());
  CudaMatcherWorkspace& workspace = RuntimeWorkspace();
  workspace.EnsureMutual(matrix_values, result.forward.size(),
                         result.backward.size());

  constexpr int kTrainPerBlock = 128;
  constexpr int kQueriesPerBlock = 8;
  const dim3 blocks(
      (features2->Rows() + kTrainPerBlock - 1) / kTrainPerBlock,
      (features1->Rows() + kQueriesPerBlock - 1) / kQueriesPerBlock);
  ComputeGuidedDotMatrixKernel<<<blocks, kTrainPerBlock>>>(
      features1->Data(), features1->Rows(), features1->Keypoints(),
      features2->Data(), features2->Rows(), features2->Keypoints(),
      features1->Columns(), fundamental,
      static_cast<float>(max_sampson_error_px * max_sampson_error_px),
      workspace.DotMatrix());
  CheckCuda(cudaGetLastError(),
            "CUDA fused guided dot matrix kernel launch");

  constexpr int kReductionThreads = 256;
  ReduceGuidedDotMatrixRowsKernel<<<
      (features1->Rows() + kReductionThreads - 1) / kReductionThreads,
      kReductionThreads>>>(workspace.DotMatrix(), features1->Rows(),
                           features2->Rows(), workspace.ForwardOutput());
  CheckCuda(cudaGetLastError(),
            "CUDA fused guided row reduction kernel launch");
  ReduceGuidedDotMatrixColumnsKernel<<<
      (features2->Rows() + kReductionThreads - 1) / kReductionThreads,
      kReductionThreads>>>(workspace.DotMatrix(), features1->Rows(),
                           features2->Rows(), workspace.BackwardOutput());
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
