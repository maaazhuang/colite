#pragma once

#include <cstddef>
#include <filesystem>
#include <limits>
#include <vector>

#include "colite/core/image.h"
#include "colite/core/match_graph.h"

namespace colite {

struct VocabularyTreeBuildOptions {
  std::size_t num_visual_words = 1024;
  std::size_t branching = 16;
  int num_iterations = 6;
  std::size_t max_num_features_per_image = 5000;
};

struct VocabularyTreeNode {
  std::vector<float> center;
  std::vector<std::size_t> children;
  std::size_t word_id = std::numeric_limits<std::size_t>::max();

  bool IsLeaf() const { return children.empty(); }
};

class VocabularyTree {
 public:
  std::size_t descriptor_dim = 0;
  std::size_t branching = 0;
  int num_iterations = 0;
  std::vector<VocabularyTreeNode> nodes;

  std::size_t NumVisualWords() const;
  std::size_t Assign(const std::vector<float>& descriptor) const;
  void Write(const std::filesystem::path& path) const;

  static VocabularyTree Read(const std::filesystem::path& path);
};

VocabularyTree BuildVocabularyTree(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const VocabularyTreeBuildOptions& options = {});

}  // namespace colite
