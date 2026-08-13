#pragma once

#include <filesystem>
#include <vector>

#include "colite/core/feature.h"
#include "colite/features/onnx_options.h"

namespace colite {

struct CosineMatchingOptions {
  float min_cosine_similarity = 0.85F;
  float max_ratio = 1.0F;
  bool mutual_check = true;
};

struct LightGlueMatchingOptions {
  FeatureMatcherType matcher_type = FeatureMatcherType::kAlikedLightGlue;
  std::filesystem::path model_path;
  float min_score = 0.1F;
  int num_threads = 1;
  OnnxExecutionProvider execution_provider = OnnxExecutionProvider::kAuto;
  int gpu_index = 0;
};

std::vector<FeatureMatch> MatchAlikedDescriptors(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const CosineMatchingOptions& options = {});

std::vector<FeatureMatch> MatchFeatureSetsLightGlue(
    const FeatureSet& features1,
    int image_width1,
    int image_height1,
    const FeatureSet& features2,
    int image_width2,
    int image_height2,
    const LightGlueMatchingOptions& options);

}  // namespace colite
