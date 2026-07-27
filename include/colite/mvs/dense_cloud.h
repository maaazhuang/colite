#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include <Eigen/Core>

#include "colite/mvs/patch_match.h"

namespace colite::mvs {

struct DenseCloudFilterOptions {
  std::size_t min_track_observations = 2;
  double max_normal_angle_degrees = 30.0;
  double voxel_size = 0.0;
  double spatial_bounds_quantile = 0.0;
};

struct DenseCloudFilterStats {
  std::size_t num_input_points = 0;
  std::size_t num_filtered_points = 0;
  std::size_t num_removed_by_track_observations = 0;
  std::size_t num_removed_by_spatial_bounds = 0;
  std::size_t num_removed_by_voxel = 0;
};

struct DenseCloudFilterResult {
  std::vector<FusedPoint> points;
  DenseCloudFilterStats stats;
};

struct DenseFusionTrackQualityStats {
  std::size_t num_tracks = 0;
  std::size_t num_single_observation_tracks = 0;
  std::size_t num_multi_observation_tracks = 0;
  std::size_t num_observations = 0;
  std::size_t max_observations = 0;
  double mean_observations = 0.0;
  double multi_observation_track_ratio = 0.0;
  double mean_max_normal_angle_degrees = 0.0;
  double max_normal_angle_degrees = 0.0;
  double mean_max_view_angle_degrees = 0.0;
  double max_view_angle_degrees = 0.0;
  std::size_t num_view_angle_tracks = 0;
};

std::vector<FusedPoint> FilterDenseCloud(
    const std::vector<FusedPoint>& points,
    const std::vector<StereoFusionTrack>& tracks,
    const DenseCloudFilterOptions& options = {});

DenseCloudFilterResult FilterDenseCloudWithStats(
    const std::vector<FusedPoint>& points,
    const std::vector<StereoFusionTrack>& tracks,
    const DenseCloudFilterOptions& options = {});

DenseFusionTrackQualityStats ComputeFusionTrackQualityStats(
    const std::vector<StereoFusionTrack>& tracks);

void WriteFusionTrackQualityStats(
    const std::filesystem::path& path,
    const DenseFusionTrackQualityStats& stats);

struct DenseCloudMeshingOptions {
  double max_edge_length = 0.0;
  double max_normal_angle_degrees = 30.0;
  std::size_t max_candidate_neighbors = 12;
};

struct DenseTriangleMesh {
  std::vector<FusedPoint> vertices;
  std::vector<Eigen::Vector3i> triangles;
};

struct DenseTriangleMeshStats {
  std::size_t num_vertices = 0;
  std::size_t num_triangles = 0;
  double mean_edge_length = 0.0;
  double max_edge_length = 0.0;
  double mean_triangle_area = 0.0;
  double min_triangle_area = 0.0;
  double max_triangle_area = 0.0;
};

DenseTriangleMesh MeshDenseCloud(
    const std::vector<FusedPoint>& points,
    const DenseCloudMeshingOptions& options = {});

DenseTriangleMeshStats ComputeDenseTriangleMeshStats(
    const DenseTriangleMesh& mesh);

void WriteDenseTriangleMeshPly(const std::filesystem::path& path,
                               const DenseTriangleMesh& mesh);

void WriteDenseTriangleMeshStats(const std::filesystem::path& path,
                                 const DenseTriangleMeshStats& stats);

}  // namespace colite::mvs
