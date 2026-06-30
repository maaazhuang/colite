#pragma once

#include <cstddef>
#include <vector>

#include "colite/core/image.h"
#include "colite/core/match_graph.h"

namespace colite {

// 匹配配对策略，命名和职责参考 COLMAP 的 matcher 家族。
enum class MatchingStrategy {
  kExhaustive,
  kSequential,
  kSpatial,
  kVocabTree,
};

struct MatchingPairGeneratorOptions {
  MatchingStrategy strategy = MatchingStrategy::kSequential;

  // sequential：每张图像与文件顺序中的后续 N 张图像配对。
  std::size_t sequential_overlap = 5;

  // spatial：每张图像连接空间距离最近的 N 个邻居；max_distance<=0 表示不限制距离。
  std::size_t spatial_max_num_neighbors = 10;
  double spatial_max_distance = 0.0;

  // vocab tree：用当前 workspace 特征临时训练视觉词，按 TF-IDF 检索候选图像。
  // 这是文件式 workspace 的轻量实现，后续可替换为持久化/导入的 COLMAP vocab tree。
  std::size_t vocab_tree_num_images = 50;
  std::size_t vocab_tree_num_visual_words = 1024;
  std::size_t vocab_tree_max_num_features = 5000;
};

// 按输入图像顺序生成候选图像对。输出中的 ImagePair 始终有序且去重。
std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchingPairGeneratorOptions& options);

std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const MatchingPairGeneratorOptions& options);

}  // namespace colite
