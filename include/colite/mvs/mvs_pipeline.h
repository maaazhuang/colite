#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "colite/core/reconstruction.h"
#include "colite/core/types.h"

namespace colite::mvs {

// MVS 源视图候选。第一版按共享稀疏点数量排序，后续会加入三角角、尺度和视角约束。
struct SourceImageScore {
  ImageId image_id = kInvalidImageId;
  std::size_t num_shared_points = 0;
  double mean_triangulation_angle_degrees = 0.0;
  double mean_scale_ratio = 0.0;
  double coverage_score = 0.0;
  double score = 0.0;
};

// 一张图像上由稀疏 SfM 点投影得到的深度样本。
struct SparseDepthSample {
  Point3DId point3d_id = kInvalidPoint3DId;
  double x = 0.0;
  double y = 0.0;
  double depth = 0.0;
};

// 稀疏深度图容器。当前只保存有效样本，后续 PatchMatch 会扩展为 dense raster。
struct SparseDepthMap {
  ImageId image_id = kInvalidImageId;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<SparseDepthSample> samples;
};

struct ImageMvsSummary {
  ImageId image_id = kInvalidImageId;
  std::vector<SourceImageScore> source_images;
  SparseDepthMap sparse_depth_map;
  double min_depth = 0.0;
  double max_depth = 0.0;
};

struct MvsPipelineOptions {
  std::size_t max_num_source_images = 8;
  std::size_t min_num_shared_points = 1;
  double min_triangulation_angle_degrees = 1.0;
  double max_triangulation_angle_degrees = 30.0;
  double triangulation_angle_weight = 1.0;
  double shared_points_weight = 1.0;
  double scale_consistency_weight = 2.0;
  double coverage_weight = 5.0;
  std::size_t coverage_grid_size = 4;
};

struct MvsPipelineSummary {
  bool success = false;
  std::size_t num_images = 0;
  std::size_t num_sparse_depth_samples = 0;
  std::vector<ImageMvsSummary> images;
};

std::vector<SourceImageScore> SelectSourceImages(
    const Reconstruction& reconstruction,
    ImageId target_image_id,
    const MvsPipelineOptions& options = {});

SparseDepthMap BuildSparseDepthMap(const Reconstruction& reconstruction,
                                   ImageId image_id);

MvsPipelineSummary RunMvsPipeline(
    const Reconstruction& reconstruction,
    const MvsPipelineOptions& options = {});

}  // namespace colite::mvs
