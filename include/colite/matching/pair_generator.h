#pragma once

#include <cstddef>
#include <filesystem>
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
  std::size_t vocab_tree_branching = 16;
  int vocab_tree_num_iterations = 6;
  std::size_t vocab_tree_max_num_features = 5000;
  // Merge nearby temporal pairs into retrieval results. Set to zero to disable.
  std::size_t vocab_tree_sequential_overlap = 2;
  // 可选词树文件。当前支持 COLITE 文本视觉词格式：
  // COLITE_VOCAB_TREE <version> <descriptor_dim>
  // 后续每行一个 visual word descriptor。为空时保留轻量临时训练路径。
  std::filesystem::path vocab_tree_path;
};

struct MatchingPairGeneratorSummary {
  std::size_t num_retrieval_pairs = 0;
  std::size_t num_sequential_guard_pairs = 0;
  std::size_t num_sequential_pairs_added = 0;
  double sequential_pair_recall = 1.0;
  std::size_t num_connected_components = 0;
  std::size_t min_degree = 0;
  std::size_t max_degree = 0;
  double mean_degree = 0.0;
};

// 按输入图像顺序生成候选图像对。输出中的 ImagePair 始终有序且去重。
std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchingPairGeneratorOptions& options);

std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const MatchingPairGeneratorOptions& options,
    MatchingPairGeneratorSummary* summary = nullptr);

}  // namespace colite
