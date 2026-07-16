#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace colite::mvs {

// Small, header-only adapter around Boost's Boykov-Kolmogorov implementation.
template <typename Value>
class MinSTGraphCut {
 public:
  struct EdgeData;
  using Traits =
      boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
  using EdgeDescriptor = typename Traits::edge_descriptor;
  using VertexSize = typename Traits::vertices_size_type;
  struct EdgeData {
    Value capacity = Value{};
    Value residual = Value{};
    EdgeDescriptor reverse{};
  };
  using Graph = boost::adjacency_list<boost::vecS, boost::vecS,
                                      boost::directedS, std::size_t, EdgeData>;

  explicit MinSTGraphCut(std::size_t num_nodes)
      : source_(num_nodes), sink_(num_nodes + 1), graph_(num_nodes + 2) {}

  void AddNode(std::size_t node, Value source_capacity, Value sink_capacity) {
    CheckNode(node);
    CheckCapacity(source_capacity);
    CheckCapacity(sink_capacity);
    if (source_capacity > Value{}) {
      AddDirectedPair(source_, node, source_capacity, Value{});
    }
    if (sink_capacity > Value{}) {
      AddDirectedPair(node, sink_, sink_capacity, Value{});
    }
  }

  void AddEdge(std::size_t first, std::size_t second, Value capacity,
               Value reverse_capacity) {
    CheckNode(first);
    CheckNode(second);
    CheckCapacity(capacity);
    CheckCapacity(reverse_capacity);
    AddDirectedPair(first, second, capacity, reverse_capacity);
  }

  Value Compute() {
    const std::size_t count = boost::num_vertices(graph_);
    colors_.resize(count);
    std::vector<EdgeDescriptor> predecessors(count);
    std::vector<VertexSize> distances(count);
    return boost::boykov_kolmogorov_max_flow(
        graph_, boost::get(&EdgeData::capacity, graph_),
        boost::get(&EdgeData::residual, graph_),
        boost::get(&EdgeData::reverse, graph_), predecessors.data(),
        colors_.data(), distances.data(),
        boost::get(boost::vertex_index, graph_), source_, sink_);
  }

  bool IsConnectedToSource(std::size_t node) const {
    CheckNode(node);
    if (colors_.empty()) {
      throw std::logic_error("Graph cut has not been computed.");
    }
    return colors_.at(node) != boost::white_color;
  }

 private:
  void CheckNode(std::size_t node) const {
    if (node >= source_) {
      throw std::out_of_range("Graph-cut node index is out of range.");
    }
  }

  static void CheckCapacity(Value capacity) {
    if (capacity < Value{}) {
      throw std::invalid_argument("Graph-cut capacity must be non-negative.");
    }
  }

  void AddDirectedPair(std::size_t first, std::size_t second, Value capacity,
                       Value reverse_capacity) {
    const EdgeDescriptor forward = boost::add_edge(first, second, graph_).first;
    const EdgeDescriptor reverse = boost::add_edge(second, first, graph_).first;
    graph_[forward].capacity = capacity;
    graph_[reverse].capacity = reverse_capacity;
    graph_[forward].reverse = reverse;
    graph_[reverse].reverse = forward;
  }

  const std::size_t source_;
  const std::size_t sink_;
  Graph graph_;
  std::vector<boost::default_color_type> colors_;
};

}  // namespace colite::mvs
