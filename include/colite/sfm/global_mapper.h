#pragma once

#include <cstddef>
#include <string>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/sfm/global_positioner.h"
#include "colite/sfm/incremental_triangulator.h"
#include "colite/sfm/rotation_averager.h"
#include "colite/sfm/view_graph.h"

namespace colite {

struct GlobalMappingOptions {
  ViewGraphCalibrationOptions view_graph;
  RotationAveragingOptions rotation_averaging;
  GlobalPositioningOptions positioning;
  IncrementalTriangulatorOptions triangulation;
  GlobalBundleAdjustmentOptions bundle_adjustment;
  std::size_t min_num_registered_images = 2;
  std::size_t min_num_points3d = 1;
  bool run_bundle_adjustment = true;
};

struct GlobalMappingSummary {
  bool success = false;
  ImageId anchor_image_id = kInvalidImageId;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
  ViewGraphCalibrationSummary view_graph;
  RotationAveragingSummary rotation_averaging;
  GlobalPositioningSummary positioning;
  IncrementalTriangulatorSummary triangulation;
  GlobalBundleAdjustmentSummary bundle_adjustment;
  std::string message;
};

GlobalMappingSummary RunGlobalMapping(
    const MatchGraph& graph,
    const GlobalMappingOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
