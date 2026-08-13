#include "onnx_runtime.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#if defined(COLITE_ENABLE_ONNX)
namespace colite::features {

namespace {

bool HasCudaExecutionProvider() {
  const std::vector<std::string> providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(),
                   "CUDAExecutionProvider") != providers.end();
}

}  // namespace

std::string FormatOnnxShape(const std::vector<std::int64_t>& shape) {
  std::ostringstream stream;
  stream << '[';
  for (std::size_t idx = 0; idx < shape.size(); ++idx) {
    if (idx != 0) stream << ", ";
    stream << shape[idx];
  }
  stream << ']';
  return stream.str();
}

void ValidateOnnxNode(const std::string_view actual_name,
                      const std::string_view expected_name,
                      const std::vector<std::int64_t>& actual_shape,
                      const std::vector<std::int64_t>& expected_shape) {
  if (actual_name != expected_name ||
      actual_shape.size() != expected_shape.size()) {
    throw std::runtime_error(
        "Unexpected ONNX node " + std::string(actual_name) + " " +
        FormatOnnxShape(actual_shape) + "; expected " +
        std::string(expected_name) + " " + FormatOnnxShape(expected_shape));
  }
  for (std::size_t idx = 0; idx < actual_shape.size(); ++idx) {
    if (expected_shape[idx] != -1 &&
        actual_shape[idx] != expected_shape[idx]) {
      throw std::runtime_error(
          "Unexpected ONNX shape for " + std::string(actual_name) + ": " +
          FormatOnnxShape(actual_shape) + "; expected " +
          FormatOnnxShape(expected_shape));
    }
  }
}

OnnxModel::OnnxModel(const std::filesystem::path& model_path,
                     const int num_threads,
                     const OnnxExecutionProvider execution_provider,
                     const int gpu_index)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "colite") {
  if (model_path.empty() || !std::filesystem::is_regular_file(model_path)) {
    throw std::invalid_argument("ONNX model file does not exist: " +
                                model_path.string());
  }
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetIntraOpNumThreads(std::max(1, num_threads));
  session_options_.SetExecutionMode(ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
  const bool cuda_available = HasCudaExecutionProvider();
  if (execution_provider == OnnxExecutionProvider::kCuda && !cuda_available) {
    throw std::runtime_error(
        "ONNX CUDA execution provider was requested but is unavailable");
  }
  if (gpu_index < 0) {
    throw std::invalid_argument("ONNX CUDA GPU index must be non-negative");
  }
  if (cuda_available && execution_provider != OnnxExecutionProvider::kCpu) {
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = gpu_index;
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
    execution_provider_ = OnnxExecutionProvider::kCuda;
  }
#if defined(_WIN32)
  session_ = std::make_unique<Ort::Session>(
      environment_, model_path.wstring().c_str(), session_options_);
#else
  session_ = std::make_unique<Ort::Session>(
      environment_, model_path.string().c_str(), session_options_);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  const std::size_t input_count = session_->GetInputCount();
  input_names_.reserve(input_count);
  input_shapes_.reserve(input_count);
  for (std::size_t idx = 0; idx < input_count; ++idx) {
    const Ort::AllocatedStringPtr name =
        session_->GetInputNameAllocated(idx, allocator);
    input_names_.emplace_back(name.get());
    input_shapes_.push_back(session_->GetInputTypeInfo(idx)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape());
  }
  const std::size_t output_count = session_->GetOutputCount();
  output_names_.reserve(output_count);
  output_shapes_.reserve(output_count);
  for (std::size_t idx = 0; idx < output_count; ++idx) {
    const Ort::AllocatedStringPtr name =
        session_->GetOutputNameAllocated(idx, allocator);
    output_names_.emplace_back(name.get());
    output_shapes_.push_back(session_->GetOutputTypeInfo(idx)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape());
  }
}

std::vector<Ort::Value> OnnxModel::Run(
    const std::vector<Ort::Value>& input_tensors) const {
  if (input_tensors.size() != input_names_.size()) {
    throw std::invalid_argument("ONNX input tensor count does not match model");
  }
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
  input_names.reserve(input_names_.size());
  output_names.reserve(output_names_.size());
  for (const std::string& name : input_names_) input_names.push_back(name.c_str());
  for (const std::string& name : output_names_) output_names.push_back(name.c_str());
  return session_->Run(Ort::RunOptions{nullptr},
                       input_names.data(),
                       input_tensors.data(),
                       input_tensors.size(),
                       output_names.data(),
                       output_names.size());
}

}  // namespace colite::features
#endif
