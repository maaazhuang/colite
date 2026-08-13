#pragma once

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#if defined(COLITE_ENABLE_ONNX)
#include <onnxruntime_cxx_api.h>

#include "colite/features/onnx_options.h"

namespace colite::features {

std::string FormatOnnxShape(const std::vector<std::int64_t>& shape);
void ValidateOnnxNode(std::string_view actual_name,
                      std::string_view expected_name,
                      const std::vector<std::int64_t>& actual_shape,
                      const std::vector<std::int64_t>& expected_shape);

class OnnxModel {
 public:
  OnnxModel(const std::filesystem::path& model_path,
            int num_threads,
            OnnxExecutionProvider execution_provider,
            int gpu_index);

  std::vector<Ort::Value> Run(
      const std::vector<Ort::Value>& input_tensors) const;

  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::vector<std::int64_t>>& InputShapes() const {
    return input_shapes_;
  }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::vector<std::vector<std::int64_t>>& OutputShapes() const {
    return output_shapes_;
  }
  OnnxExecutionProvider ExecutionProvider() const {
    return execution_provider_;
  }

 private:
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<std::vector<std::int64_t>> input_shapes_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<std::int64_t>> output_shapes_;
  OnnxExecutionProvider execution_provider_ = OnnxExecutionProvider::kCpu;
};

}  // namespace colite::features
#endif
