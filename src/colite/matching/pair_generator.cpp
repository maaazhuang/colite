#include "colite/matching/pair_generator.h"
#include "colite/matching/vocabulary_tree.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace colite {
namespace {

void AddPair(ImageId image_id1,
             ImageId image_id2,
             std::set<ImagePair>* pairs) {
  if (image_id1 == image_id2) {
    return;
  }
  pairs->insert(ImagePair::Ordered(image_id1, image_id2));
}

std::vector<ImagePair> ToVector(const std::set<ImagePair>& pairs) {
  return {pairs.begin(), pairs.end()};
}

std::vector<ImagePair> GenerateExhaustivePairs(
    const std::vector<Image>& images) {
  std::set<ImagePair> pairs;
  for (std::size_t i = 0; i < images.size(); ++i) {
    for (std::size_t j = i + 1; j < images.size(); ++j) {
      AddPair(images[i].id, images[j].id, &pairs);
    }
  }
  return ToVector(pairs);
}

std::vector<ImagePair> GenerateSequentialPairs(
    const std::vector<Image>& images,
    const std::size_t overlap,
    const bool quadratic = false) {
  if (overlap == 0) {
    throw std::invalid_argument("sequential overlap must be positive");
  }
  std::set<ImagePair> pairs;
  for (std::size_t i = 0; i < images.size(); ++i) {
    const std::size_t remaining = images.size() - i - 1;
    if (quadratic) {
      const std::size_t levels = std::min<std::size_t>(
          overlap, std::numeric_limits<std::size_t>::digits);
      for (std::size_t level = 0; level < levels; ++level) {
        const std::size_t offset = std::size_t{1} << level;
        if (offset > remaining) break;
        AddPair(images[i].id, images[i + offset].id, &pairs);
      }
    } else {
      const std::size_t end = i + std::min(overlap, remaining) + 1;
      for (std::size_t j = i + 1; j < end; ++j) {
        AddPair(images[i].id, images[j].id, &pairs);
      }
    }
  }
  return ToVector(pairs);
}

std::vector<ImagePair> GenerateSpatialPairs(
    const std::vector<Image>& images,
    const std::size_t max_num_neighbors,
    const double max_distance) {
  if (max_num_neighbors == 0) {
    throw std::invalid_argument("spatial max_num_neighbors must be positive");
  }
  for (const Image& image : images) {
    if (!image.prior_position.has_value()) {
      throw std::invalid_argument(
          "spatial matching requires every image to have a prior position");
    }
  }

  std::set<ImagePair> pairs;
  for (std::size_t i = 0; i < images.size(); ++i) {
    std::vector<std::pair<double, std::size_t>> neighbors;
    neighbors.reserve(images.size() > 0 ? images.size() - 1 : 0);
    for (std::size_t j = 0; j < images.size(); ++j) {
      if (i == j) {
        continue;
      }
      const double distance =
          (*images[i].prior_position - *images[j].prior_position).norm();
      if (max_distance > 0.0 && distance > max_distance) {
        continue;
      }
      neighbors.emplace_back(distance, j);
    }
    std::sort(neighbors.begin(), neighbors.end(),
              [&](const auto& lhs, const auto& rhs) {
                return std::tie(lhs.first, images[lhs.second].id) <
                       std::tie(rhs.first, images[rhs.second].id);
              });
    const std::size_t num_neighbors =
        std::min(max_num_neighbors, neighbors.size());
    for (std::size_t k = 0; k < num_neighbors; ++k) {
      AddPair(images[i].id, images[neighbors[k].second].id, &pairs);
    }
  }
  return ToVector(pairs);
}

std::vector<ImagePair> GenerateTransitivePairs(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const std::size_t max_path_length,
    const std::size_t max_num_neighbors) {
  if (max_path_length < 2 || max_num_neighbors == 0) {
    throw std::invalid_argument(
        "transitive matching requires path length >= 2 and neighbors > 0");
  }
  std::map<ImageId, std::set<ImageId>> adjacency;
  for (const Image& image : images) {
    adjacency[image.id];
  }
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (geometry.inlier_matches.empty()) {
      continue;
    }
    adjacency[pair.image_id1].insert(pair.image_id2);
    adjacency[pair.image_id2].insert(pair.image_id1);
  }
  std::set<ImagePair> candidates;
  for (const Image& source_image : images) {
    const ImageId source = source_image.id;
    std::map<ImageId, std::size_t> distance{{source, 0}};
    std::vector<ImageId> frontier{source};
    for (std::size_t depth = 1;
         depth <= max_path_length && !frontier.empty(); ++depth) {
      std::vector<ImageId> next;
      for (const ImageId current : frontier) {
        for (const ImageId neighbor : adjacency[current]) {
          if (distance.emplace(neighbor, depth).second) {
            next.push_back(neighbor);
          }
        }
      }
      frontier = std::move(next);
    }
    std::vector<std::pair<std::size_t, ImageId>> ranked;
    for (const auto& [target, path_length] : distance) {
      if (path_length < 2 || target == source ||
          graph.verified_geometries.count(ImagePair::Ordered(source, target)) > 0 ||
          graph.matches.count(ImagePair::Ordered(source, target)) > 0) {
        continue;
      }
      std::size_t common_neighbors = 0;
      for (const ImageId neighbor : adjacency[source]) {
        common_neighbors += adjacency[target].count(neighbor) > 0 ? 1 : 0;
      }
      ranked.emplace_back(common_neighbors, target);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first != rhs.first ? lhs.first > rhs.first
                                    : lhs.second < rhs.second;
    });
    ranked.resize(std::min(max_num_neighbors, ranked.size()));
    for (const auto& [support, target] : ranked) {
      (void)support;
      AddPair(source, target, &candidates);
    }
  }
  return ToVector(candidates);
}

using DescriptorVector = std::vector<float>;

struct DescriptorSample {
  ImageId image_id = kInvalidImageId;
  DescriptorVector values;
};

double SquaredDistance(const DescriptorVector& lhs,
                       const DescriptorVector& rhs) {
  double distance = 0.0;
  const std::size_t size = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < size; ++i) {
    const double diff = static_cast<double>(lhs[i]) - rhs[i];
    distance += diff * diff;
  }
  return distance;
}

std::vector<DescriptorSample> CollectDescriptorSamples(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const std::size_t max_num_features_per_image) {
  std::vector<DescriptorSample> samples;
  for (const Image& image : images) {
    const FeatureSet* features = FindFeatures(graph, image.id);
    if (features == nullptr || features->descriptors.rows() == 0 ||
        features->descriptors.cols() == 0) {
      continue;
    }
    const std::size_t rows =
        static_cast<std::size_t>(features->descriptors.rows());
    const std::size_t cols =
        static_cast<std::size_t>(features->descriptors.cols());
    const std::size_t step =
        max_num_features_per_image > 0 && rows > max_num_features_per_image
            ? (rows + max_num_features_per_image - 1) / max_num_features_per_image
            : 1;
    for (std::size_t row = 0; row < rows; row += step) {
      DescriptorSample sample;
      sample.image_id = image.id;
      sample.values.resize(cols);
      for (std::size_t col = 0; col < cols; ++col) {
        sample.values[col] =
            features->descriptors(static_cast<int>(row), static_cast<int>(col));
      }
      samples.push_back(std::move(sample));
    }
  }
  return samples;
}

[[maybe_unused]] std::vector<DescriptorVector> TrainVisualWords(
    const std::vector<DescriptorSample>& samples,
    const std::size_t num_visual_words) {
  if (samples.empty()) {
    return {};
  }
  const std::size_t num_words =
      std::max<std::size_t>(1, std::min(num_visual_words, samples.size()));
  std::vector<DescriptorVector> words;
  words.reserve(num_words);
  for (std::size_t i = 0; i < num_words; ++i) {
    const std::size_t sample_idx =
        num_words == 1 ? 0 : (i * (samples.size() - 1)) / (num_words - 1);
    words.push_back(samples[sample_idx].values);
  }

  // 小型 deterministic k-means。真实 COLMAP 使用离线词汇树；这里为文件 workspace
  // 提供无需外部 vocab 文件的第一版候选检索能力。
  for (int iteration = 0; iteration < 6; ++iteration) {
    std::vector<DescriptorVector> sums(words.size(),
                                       DescriptorVector(words[0].size(), 0.0F));
    std::vector<std::size_t> counts(words.size(), 0);
    for (const DescriptorSample& sample : samples) {
      std::size_t best_word = 0;
      double best_distance = std::numeric_limits<double>::max();
      for (std::size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
        const double distance = SquaredDistance(sample.values, words[word_idx]);
        if (distance < best_distance) {
          best_distance = distance;
          best_word = word_idx;
        }
      }
      for (std::size_t dim = 0; dim < sample.values.size(); ++dim) {
        sums[best_word][dim] += sample.values[dim];
      }
      ++counts[best_word];
    }
    for (std::size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
      if (counts[word_idx] == 0) {
        continue;
      }
      for (float& value : sums[word_idx]) {
        value /= static_cast<float>(counts[word_idx]);
      }
      words[word_idx] = std::move(sums[word_idx]);
    }
  }
  return words;
}

[[maybe_unused]] std::vector<DescriptorVector> LoadVisualWordsFromTextFile(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::invalid_argument("failed to open vocab tree file: " +
                                path.string());
  }

  std::string magic;
  int version = 0;
  std::size_t descriptor_dim = 0;
  file >> magic >> version >> descriptor_dim;
  if (magic != "COLITE_VOCAB_TREE" || version != 1 || descriptor_dim == 0) {
    throw std::invalid_argument(
        "unsupported vocab tree file header: " + path.string());
  }

  std::vector<DescriptorVector> words;
  std::string line;
  std::getline(file, line);
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream stream(line);
    DescriptorVector word(descriptor_dim, 0.0F);
    for (std::size_t dim = 0; dim < descriptor_dim; ++dim) {
      if (!(stream >> word[dim])) {
        throw std::invalid_argument(
            "vocab tree word has wrong descriptor dimension: " +
            path.string());
      }
    }
    words.push_back(std::move(word));
  }

  if (words.empty()) {
    throw std::invalid_argument("vocab tree file contains no visual words: " +
                                path.string());
  }
  return words;
}

[[maybe_unused]] std::size_t AssignVisualWord(const DescriptorVector& descriptor,
                             const std::vector<DescriptorVector>& words) {
  std::size_t best_word = 0;
  double best_distance = std::numeric_limits<double>::max();
  for (std::size_t word_idx = 0; word_idx < words.size(); ++word_idx) {
    const double distance = SquaredDistance(descriptor, words[word_idx]);
    if (distance < best_distance) {
      best_distance = distance;
      best_word = word_idx;
    }
  }
  return best_word;
}

std::vector<ImagePair> GenerateVocabTreePairs(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const std::size_t num_images,
    const std::size_t num_visual_words,
    const std::size_t branching,
    const int num_iterations,
    const std::size_t max_num_features_per_image,
    const std::size_t sequential_overlap,
    const std::filesystem::path& vocab_tree_path,
    MatchingPairGeneratorSummary* summary) {
  if (num_images == 0) {
    throw std::invalid_argument("vocab_tree num_images must be positive");
  }
  VocabularyTree tree;
  if (vocab_tree_path.empty()) {
    VocabularyTreeBuildOptions build_options;
    build_options.num_visual_words = num_visual_words;
    build_options.branching = branching;
    build_options.num_iterations = num_iterations;
    build_options.max_num_features_per_image = max_num_features_per_image;
    tree = BuildVocabularyTree(images, graph, build_options);
  } else {
    tree = VocabularyTree::Read(vocab_tree_path);
  }
  const std::vector<DescriptorSample> samples =
      CollectDescriptorSamples(images, graph, max_num_features_per_image);
  if (samples.empty()) {
    throw std::invalid_argument(
        "vocab tree matching requires workspace features");
  }

  std::map<ImageId, std::map<std::size_t, double>> histograms;
  for (const DescriptorSample& sample : samples) {
    ++histograms[sample.image_id][tree.Assign(sample.values)];
  }

  std::map<std::size_t, std::size_t> document_frequency;
  for (const auto& [image_id, histogram] : histograms) {
    (void)image_id;
    for (const auto& [word_id, count] : histogram) {
      (void)count;
      ++document_frequency[word_id];
    }
  }

  const double num_docs = static_cast<double>(histograms.size());
  for (auto& [image_id, histogram] : histograms) {
    (void)image_id;
    double norm = 0.0;
    for (auto& [word_id, value] : histogram) {
      const double idf =
          std::log((num_docs + 1.0) /
                   (static_cast<double>(document_frequency[word_id]) + 1.0)) +
          1.0;
      value *= idf;
      norm += value * value;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0) {
      for (auto& [word_id, value] : histogram) {
        (void)word_id;
        value /= norm;
      }
    }
  }

  std::set<ImagePair> pairs;
  for (const Image& image : images) {
    const auto query_it = histograms.find(image.id);
    if (query_it == histograms.end()) {
      continue;
    }
    std::vector<std::pair<double, ImageId>> scored_images;
    for (const Image& other : images) {
      if (other.id == image.id) {
        continue;
      }
      const auto other_it = histograms.find(other.id);
      if (other_it == histograms.end()) {
        continue;
      }
      double score = 0.0;
      const auto* smaller = &query_it->second;
      const auto* larger = &other_it->second;
      if (smaller->size() > larger->size()) {
        std::swap(smaller, larger);
      }
      for (const auto& [word_id, value] : *smaller) {
        const auto value_it = larger->find(word_id);
        if (value_it != larger->end()) {
          score += value * value_it->second;
        }
      }
      if (score > 0.0) {
        scored_images.emplace_back(score, other.id);
      }
    }
    std::sort(scored_images.begin(), scored_images.end(),
              [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                  return lhs.first > rhs.first;
                }
                return lhs.second < rhs.second;
              });
    const std::size_t keep = std::min(num_images, scored_images.size());
    for (std::size_t i = 0; i < keep; ++i) {
      AddPair(image.id, scored_images[i].second, &pairs);
    }
  }
  const std::set<ImagePair> retrieval_pairs = pairs;
  if (summary != nullptr) {
    summary->num_retrieval_pairs = retrieval_pairs.size();
  }
  if (sequential_overlap > 0) {
    const std::vector<ImagePair> guard_pairs =
        GenerateSequentialPairs(images, sequential_overlap);
    std::size_t recalled = 0;
    for (const ImagePair& pair : guard_pairs) {
      if (retrieval_pairs.count(pair) > 0) {
        ++recalled;
      }
      pairs.insert(pair);
    }
    if (summary != nullptr) {
      summary->num_sequential_guard_pairs = guard_pairs.size();
      summary->num_sequential_pairs_added = guard_pairs.size() - recalled;
      summary->sequential_pair_recall =
          guard_pairs.empty()
              ? 1.0
              : static_cast<double>(recalled) / guard_pairs.size();
    }
  }
  return ToVector(pairs);
}

void ComputeGraphSummary(const std::vector<Image>& images,
                         const std::vector<ImagePair>& pairs,
                         MatchingPairGeneratorSummary* summary) {
  if (summary == nullptr) {
    return;
  }
  std::map<ImageId, std::vector<ImageId>> adjacency;
  for (const Image& image : images) {
    adjacency[image.id];
  }
  for (const ImagePair& pair : pairs) {
    adjacency[pair.image_id1].push_back(pair.image_id2);
    adjacency[pair.image_id2].push_back(pair.image_id1);
  }
  if (adjacency.empty()) {
    return;
  }
  summary->min_degree = std::numeric_limits<std::size_t>::max();
  std::size_t degree_sum = 0;
  for (const auto& [image_id, neighbors] : adjacency) {
    (void)image_id;
    summary->min_degree = std::min(summary->min_degree, neighbors.size());
    summary->max_degree = std::max(summary->max_degree, neighbors.size());
    degree_sum += neighbors.size();
  }
  summary->mean_degree = static_cast<double>(degree_sum) / adjacency.size();

  std::set<ImageId> visited;
  for (const auto& [root, neighbors] : adjacency) {
    (void)neighbors;
    if (!visited.insert(root).second) {
      continue;
    }
    ++summary->num_connected_components;
    std::vector<ImageId> stack{root};
    while (!stack.empty()) {
      const ImageId image_id = stack.back();
      stack.pop_back();
      for (const ImageId neighbor : adjacency.at(image_id)) {
        if (visited.insert(neighbor).second) {
          stack.push_back(neighbor);
        }
      }
    }
  }
}

}  // namespace

std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchingPairGeneratorOptions& options) {
  switch (options.strategy) {
    case MatchingStrategy::kExhaustive:
      return GenerateExhaustivePairs(images);
    case MatchingStrategy::kSequential:
      return GenerateSequentialPairs(images, options.sequential_overlap,
                                     options.sequential_quadratic_overlap);
    case MatchingStrategy::kSpatial:
      return GenerateSpatialPairs(images, options.spatial_max_num_neighbors,
                                  options.spatial_max_distance);
    case MatchingStrategy::kVocabTree:
    case MatchingStrategy::kTransitive:
      throw std::invalid_argument(
          "this matching strategy requires a MatchGraph");
  }
  throw std::invalid_argument("unsupported matching strategy");
}

std::vector<ImagePair> GenerateMatchingPairs(
    const std::vector<Image>& images,
    const MatchGraph& graph,
    const MatchingPairGeneratorOptions& options,
    MatchingPairGeneratorSummary* summary) {
  std::vector<ImagePair> pairs;
  if (options.strategy == MatchingStrategy::kVocabTree) {
    pairs = GenerateVocabTreePairs(
        images, graph, options.vocab_tree_num_images,
        options.vocab_tree_num_visual_words, options.vocab_tree_branching,
        options.vocab_tree_num_iterations,
        options.vocab_tree_max_num_features,
        options.vocab_tree_sequential_overlap, options.vocab_tree_path,
        summary);
  } else if (options.strategy == MatchingStrategy::kTransitive) {
    pairs = GenerateTransitivePairs(
        images, graph, options.transitive_max_path_length,
        options.transitive_max_num_neighbors);
  } else {
    pairs = GenerateMatchingPairs(images, options);
  }
  ComputeGraphSummary(images, pairs, summary);
  return pairs;
}

}  // namespace colite
