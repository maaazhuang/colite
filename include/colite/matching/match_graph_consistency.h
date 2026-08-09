#pragma once

#include <cstddef>

#include "colite/core/match_graph.h"

namespace colite {

// Statistics for deterministic verified-correspondence normalization.
// A conflict component contains more than one feature observation from at
// least one image and therefore cannot represent a valid SfM track.
struct MatchGraphConsistencySummary {
  std::size_t num_input_edges = 0;
  std::size_t num_invalid_edges = 0;
  std::size_t num_exact_duplicate_edges = 0;
  std::size_t num_pair_conflict_edges = 0;
  std::size_t num_cross_component_conflict_edges = 0;
  std::size_t num_output_edges = 0;
  std::size_t num_input_components = 0;
  std::size_t num_input_conflicted_components = 0;
  std::size_t num_input_conflicting_observations = 0;
  std::size_t num_output_components = 0;
};

// Enforce the invariants expected by an SfM track graph:
//   1. each image pair is one-to-one;
//   2. cross-pair component conflicts are measured for downstream geometric
//      resolution without destructively deleting verified evidence.
//
// Edges are selected deterministically. Exact three-view cycle support is
// preferred first, followed by cross-pair feature support, lower Sampson error,
// and stable image/feature ordering.
// Cross-pair ambiguity is retained for the mapper's pose-aware geometric
// decisions, avoiding destructive pruning in repetitive texture.
MatchGraphConsistencySummary EnforceVerifiedMatchGraphConsistency(
    MatchGraph* graph);

}  // namespace colite
