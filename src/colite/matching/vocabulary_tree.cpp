#include "colite/matching/vocabulary_tree.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace colite {
namespace {

using DescriptorVector = std::vector<float>;

double SquaredDistance(const DescriptorVector& lhs,
                       const DescriptorVector& rhs) {
  double distance = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double difference = static_cast<double>(lhs[i]) - rhs[i];
    distance += difference * difference;
  }
  return distance;
}

std::vector<DescriptorVector> CollectSamples(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const std::size_t max_num_features_per_image) {
  std::vector<DescriptorVector> samples;
  for (const Image& image : images) {
    const FeatureSet* features = FindFeatures(graph, image.id);
    if (features == nullptr || features->descriptors.size() == 0) {
      continue;
    }
    const std::size_t rows =
        static_cast<std::size_t>(features->descriptors.rows());
    const std::size_t cols =
        static_cast<std::size_t>(features->descriptors.cols());
    const std::size_t step =
        max_num_features_per_image > 0 && rows > max_num_features_per_image
            ? (rows + max_num_features_per_image - 1) /
                  max_num_features_per_image
            : 1;
    for (std::size_t row = 0; row < rows; row += step) {
      DescriptorVector descriptor(cols);
      for (std::size_t col = 0; col < cols; ++col) {
        descriptor[col] = features->descriptors(
            static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
      }
      samples.push_back(std::move(descriptor));
    }
  }
  return samples;
}

DescriptorVector MeanDescriptor(
    const std::vector<DescriptorVector>& samples,
    const std::vector<std::size_t>& indices) {
  DescriptorVector mean(samples.front().size(), 0.0F);
  for (const std::size_t index : indices) {
    for (std::size_t dim = 0; dim < mean.size(); ++dim) {
      mean[dim] += samples[index][dim];
    }
  }
  const float inverse = 1.0F / static_cast<float>(indices.size());
  for (float& value : mean) {
    value *= inverse;
  }
  return mean;
}

struct ClusterResult {
  std::vector<DescriptorVector> centers;
  std::vector<std::vector<std::size_t>> clusters;
};

ClusterResult ClusterSamples(const std::vector<DescriptorVector>& samples,
                             const std::vector<std::size_t>& indices,
                             const std::size_t num_clusters,
                             const int num_iterations) {
  ClusterResult result;
  result.centers.reserve(num_clusters);
  for (std::size_t i = 0; i < num_clusters; ++i) {
    const std::size_t offset = num_clusters == 1
                                   ? 0
                                   : (i * (indices.size() - 1)) /
                                         (num_clusters - 1);
    result.centers.push_back(samples[indices[offset]]);
  }

  for (int iteration = 0; iteration < std::max(1, num_iterations);
       ++iteration) {
    result.clusters.assign(num_clusters, {});
    for (const std::size_t index : indices) {
      std::size_t best_cluster = 0;
      double best_distance = std::numeric_limits<double>::infinity();
      for (std::size_t cluster = 0; cluster < result.centers.size();
           ++cluster) {
        const double distance =
            SquaredDistance(samples[index], result.centers[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      result.clusters[best_cluster].push_back(index);
    }
    for (std::size_t cluster = 0; cluster < num_clusters; ++cluster) {
      if (!result.clusters[cluster].empty()) {
        result.centers[cluster] =
            MeanDescriptor(samples, result.clusters[cluster]);
      }
    }
  }
  return result;
}

std::size_t BuildNode(const std::vector<DescriptorVector>& samples,
                      const std::vector<std::size_t>& indices,
                      const std::size_t leaf_budget,
                      const VocabularyTreeBuildOptions& options,
                      std::size_t* next_word_id,
                      VocabularyTree* tree) {
  const std::size_t node_id = tree->nodes.size();
  tree->nodes.emplace_back();
  tree->nodes[node_id].center = MeanDescriptor(samples, indices);
  if (leaf_budget <= 1 || indices.size() <= 1) {
    tree->nodes[node_id].word_id = (*next_word_id)++;
    return node_id;
  }

  const std::size_t num_clusters = std::min(
      {std::max<std::size_t>(2, options.branching), leaf_budget,
       indices.size()});
  ClusterResult clusters = ClusterSamples(samples, indices, num_clusters,
                                          options.num_iterations);
  std::size_t nonempty_clusters = 0;
  for (const auto& cluster : clusters.clusters) {
    nonempty_clusters += cluster.empty() ? 0 : 1;
  }
  if (nonempty_clusters <= 1) {
    tree->nodes[node_id].word_id = (*next_word_id)++;
    return node_id;
  }

  std::size_t remaining_budget = leaf_budget;
  std::size_t remaining_clusters = nonempty_clusters;
  for (const auto& cluster : clusters.clusters) {
    if (cluster.empty()) {
      continue;
    }
    const std::size_t child_budget =
        std::max<std::size_t>(1, remaining_budget / remaining_clusters);
    remaining_budget -= child_budget;
    --remaining_clusters;
    const std::size_t child = BuildNode(samples, cluster, child_budget,
                                        options, next_word_id, tree);
    tree->nodes[node_id].children.push_back(child);
  }
  return node_id;
}

}  // namespace

std::size_t VocabularyTree::NumVisualWords() const {
  std::size_t count = 0;
  for (const VocabularyTreeNode& node : nodes) {
    count += node.IsLeaf() ? 1 : 0;
  }
  return count;
}

std::size_t VocabularyTree::Assign(
    const std::vector<float>& descriptor) const {
  if (nodes.empty() || descriptor.size() != descriptor_dim) {
    throw std::invalid_argument("descriptor does not match vocabulary tree");
  }
  std::size_t node_id = 0;
  while (!nodes[node_id].IsLeaf()) {
    std::size_t best_child = nodes[node_id].children.front();
    double best_distance = std::numeric_limits<double>::infinity();
    for (const std::size_t child : nodes[node_id].children) {
      const double distance = SquaredDistance(descriptor, nodes[child].center);
      if (distance < best_distance) {
        best_distance = distance;
        best_child = child;
      }
    }
    node_id = best_child;
  }
  return nodes[node_id].word_id;
}

void VocabularyTree::Write(const std::filesystem::path& path) const {
  if (nodes.empty() || descriptor_dim == 0) {
    throw std::invalid_argument("cannot write an empty vocabulary tree");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("failed to write vocabulary tree: " +
                             path.string());
  }
  file << "COLITE_VOCAB_TREE 2 " << descriptor_dim << " " << branching
       << " " << num_iterations << " " << nodes.size() << " "
       << NumVisualWords() << "\n" << std::setprecision(9);
  for (const VocabularyTreeNode& node : nodes) {
    const long long word_id = node.IsLeaf()
                                  ? static_cast<long long>(node.word_id)
                                  : -1;
    file << word_id << " " << node.children.size();
    for (const std::size_t child : node.children) {
      file << " " << child;
    }
    for (const float value : node.center) {
      file << " " << value;
    }
    file << "\n";
  }
}

VocabularyTree VocabularyTree::Read(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::invalid_argument("failed to open vocab tree file: " +
                                path.string());
  }
  std::string magic;
  int version = 0;
  VocabularyTree tree;
  file >> magic >> version >> tree.descriptor_dim;
  if (magic != "COLITE_VOCAB_TREE" || tree.descriptor_dim == 0) {
    throw std::invalid_argument("unsupported vocab tree header: " +
                                path.string());
  }
  if (version == 1) {
    tree.branching = 0;
    tree.num_iterations = 0;
    tree.nodes.emplace_back();
    DescriptorVector word(tree.descriptor_dim);
    while (file >> word[0]) {
      for (std::size_t dim = 1; dim < tree.descriptor_dim; ++dim) {
        if (!(file >> word[dim])) {
          throw std::invalid_argument("invalid legacy vocab tree: " +
                                      path.string());
        }
      }
      VocabularyTreeNode leaf;
      leaf.center = word;
      leaf.word_id = tree.nodes.size() - 1;
      tree.nodes.push_back(std::move(leaf));
      tree.nodes[0].children.push_back(tree.nodes.size() - 1);
    }
    if (tree.nodes.size() <= 1) {
      throw std::invalid_argument("vocab tree contains no visual words");
    }
    tree.nodes[0].center.assign(tree.descriptor_dim, 0.0F);
    return tree;
  }
  if (version != 2) {
    throw std::invalid_argument("unsupported vocab tree version");
  }
  std::size_t node_count = 0;
  std::size_t word_count = 0;
  file >> tree.branching >> tree.num_iterations >> node_count >> word_count;
  tree.nodes.resize(node_count);
  for (VocabularyTreeNode& node : tree.nodes) {
    long long word_id = -1;
    std::size_t child_count = 0;
    file >> word_id >> child_count;
    node.children.resize(child_count);
    for (std::size_t& child : node.children) {
      file >> child;
      if (child >= node_count) {
        throw std::invalid_argument("vocab tree child index is invalid");
      }
    }
    node.center.resize(tree.descriptor_dim);
    for (float& value : node.center) {
      file >> value;
    }
    if (!file) {
      throw std::invalid_argument("truncated vocab tree file");
    }
    if (word_id >= 0) {
      node.word_id = static_cast<std::size_t>(word_id);
    }
  }
  if (tree.nodes.empty() || tree.NumVisualWords() != word_count) {
    throw std::invalid_argument("vocab tree metadata is inconsistent");
  }
  return tree;
}

VocabularyTree BuildVocabularyTree(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const VocabularyTreeBuildOptions& options) {
  if (options.num_visual_words == 0 || options.branching < 2 ||
      options.num_iterations <= 0) {
    throw std::invalid_argument("invalid vocabulary tree build options");
  }
  const std::vector<DescriptorVector> samples = CollectSamples(
      images, graph, options.max_num_features_per_image);
  if (samples.empty() || samples.front().empty()) {
    throw std::invalid_argument("vocabulary tree requires descriptors");
  }
  std::vector<std::size_t> indices(samples.size());
  std::iota(indices.begin(), indices.end(), 0);
  VocabularyTree tree;
  tree.descriptor_dim = samples.front().size();
  tree.branching = options.branching;
  tree.num_iterations = options.num_iterations;
  std::size_t next_word_id = 0;
  BuildNode(samples, indices,
            std::min(options.num_visual_words, samples.size()), options,
            &next_word_id, &tree);
  return tree;
}

}  // namespace colite
