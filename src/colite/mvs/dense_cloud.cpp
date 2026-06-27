#include "colite/mvs/dense_cloud.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace colite::mvs {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool IsFinitePoint(const FusedPoint& point) {
  return point.xyz.allFinite() && point.normal.allFinite();
}

double ClampUnit(double value) {
  return std::max(-1.0, std::min(1.0, value));
}

double NormalAngleDegrees(const Eigen::Vector3f& lhs,
                          const Eigen::Vector3f& rhs) {
  if (lhs.squaredNorm() <= 0.0F || rhs.squaredNorm() <= 0.0F) {
    return 180.0;
  }
  const double dot =
      ClampUnit(static_cast<double>(lhs.normalized().dot(rhs.normalized())));
  return std::acos(dot) * 180.0 / kPi;
}

bool TrackPasses(const StereoFusionTrack* track,
                 const DenseCloudFilterOptions& options) {
  if (track == nullptr) {
    return options.min_track_observations <= 1;
  }
  return track->num_observations >= options.min_track_observations &&
         track->max_normal_angle_degrees <= options.max_normal_angle_degrees;
}

std::vector<const StereoFusionTrack*> BuildTrackLookup(
    std::size_t num_points,
    const std::vector<StereoFusionTrack>& tracks) {
  std::vector<const StereoFusionTrack*> lookup(num_points, nullptr);
  for (const StereoFusionTrack& track : tracks) {
    if (track.point_idx < lookup.size()) {
      lookup[track.point_idx] = &track;
    }
  }
  return lookup;
}

struct VoxelKey {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b97f4a7c15ULL +
            (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b97f4a7c15ULL +
            (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct TriangleKey {
  int a = 0;
  int b = 0;
  int c = 0;

  bool operator==(const TriangleKey& other) const {
    return a == other.a && b == other.b && c == other.c;
  }
};

struct TriangleKeyHash {
  std::size_t operator()(const TriangleKey& key) const {
    std::size_t seed = std::hash<int>{}(key.a);
    seed ^= std::hash<int>{}(key.b) + 0x9e3779b97f4a7c15ULL +
            (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<int>{}(key.c) + 0x9e3779b97f4a7c15ULL +
            (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

VoxelKey MakeVoxelKey(const Eigen::Vector3d& xyz, double voxel_size) {
  return {static_cast<std::int64_t>(std::floor(xyz.x() / voxel_size)),
          static_cast<std::int64_t>(std::floor(xyz.y() / voxel_size)),
          static_cast<std::int64_t>(std::floor(xyz.z() / voxel_size))};
}

double SquaredDistance(const FusedPoint& lhs, const FusedPoint& rhs) {
  return (lhs.xyz - rhs.xyz).squaredNorm();
}

bool EdgeLengthsPass(const FusedPoint& a,
                     const FusedPoint& b,
                     const FusedPoint& c,
                     double max_edge_length) {
  if (max_edge_length <= 0.0) {
    return true;
  }
  const double max_squared = max_edge_length * max_edge_length;
  return SquaredDistance(a, b) <= max_squared &&
         SquaredDistance(a, c) <= max_squared &&
         SquaredDistance(b, c) <= max_squared;
}

bool TriangleNormalsPass(const FusedPoint& a,
                         const FusedPoint& b,
                         const FusedPoint& c,
                         double max_normal_angle_degrees) {
  return NormalAngleDegrees(a.normal, b.normal) <= max_normal_angle_degrees &&
         NormalAngleDegrees(a.normal, c.normal) <= max_normal_angle_degrees &&
         NormalAngleDegrees(b.normal, c.normal) <= max_normal_angle_degrees;
}

double TriangleAreaSquared(const FusedPoint& a,
                           const FusedPoint& b,
                           const FusedPoint& c) {
  return (b.xyz - a.xyz).cross(c.xyz - a.xyz).squaredNorm() * 0.25;
}

std::vector<std::size_t> FindNearestNeighborIndices(
    const std::vector<FusedPoint>& points,
    std::size_t seed_idx,
    std::size_t max_neighbors) {
  std::vector<std::pair<double, std::size_t>> distances;
  distances.reserve(points.size() - 1);
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    if (idx == seed_idx) {
      continue;
    }
    distances.emplace_back(SquaredDistance(points[seed_idx], points[idx]), idx);
  }
  std::sort(distances.begin(), distances.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.first == rhs.first) {
                return lhs.second < rhs.second;
              }
              return lhs.first < rhs.first;
            });
  const std::size_t num_neighbors =
      std::min(max_neighbors, distances.size());
  std::vector<std::size_t> neighbors;
  neighbors.reserve(num_neighbors);
  for (std::size_t i = 0; i < num_neighbors; ++i) {
    neighbors.push_back(distances[i].second);
  }
  return neighbors;
}

TriangleKey MakeTriangleKey(std::size_t i, std::size_t j, std::size_t k) {
  std::array<int, 3> indices = {static_cast<int>(i), static_cast<int>(j),
                                static_cast<int>(k)};
  std::sort(indices.begin(), indices.end());
  return {indices[0], indices[1], indices[2]};
}

}  // namespace

std::vector<FusedPoint> FilterDenseCloud(
    const std::vector<FusedPoint>& points,
    const std::vector<StereoFusionTrack>& tracks,
    const DenseCloudFilterOptions& options) {
  return FilterDenseCloudWithStats(points, tracks, options).points;
}

DenseCloudFilterResult FilterDenseCloudWithStats(
    const std::vector<FusedPoint>& points,
    const std::vector<StereoFusionTrack>& tracks,
    const DenseCloudFilterOptions& options) {
  if (options.min_track_observations == 0) {
    throw std::invalid_argument(
        "Dense cloud minimum track observations must be positive.");
  }
  if (!std::isfinite(options.max_normal_angle_degrees) ||
      options.max_normal_angle_degrees < 0.0 ||
      options.max_normal_angle_degrees > 180.0) {
    throw std::invalid_argument(
        "Dense cloud max normal angle must be in [0, 180].");
  }
  if (!std::isfinite(options.voxel_size) || options.voxel_size < 0.0) {
    throw std::invalid_argument("Dense cloud voxel size must be non-negative.");
  }

  DenseCloudFilterResult result;
  result.stats.num_input_points = points.size();
  const std::vector<const StereoFusionTrack*> track_lookup =
      BuildTrackLookup(points.size(), tracks);
  std::unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
  for (std::size_t point_idx = 0; point_idx < points.size(); ++point_idx) {
    const FusedPoint& point = points[point_idx];
    if (!IsFinitePoint(point) ||
        !TrackPasses(track_lookup[point_idx], options)) {
      ++result.stats.num_removed_by_track_observations;
      continue;
    }
    if (options.voxel_size > 0.0) {
      const VoxelKey voxel_key = MakeVoxelKey(point.xyz, options.voxel_size);
      if (!occupied_voxels.insert(voxel_key).second) {
        ++result.stats.num_removed_by_voxel;
        continue;
      }
    }
    result.points.push_back(point);
  }
  result.stats.num_filtered_points = result.points.size();
  return result;
}

DenseFusionTrackQualityStats ComputeFusionTrackQualityStats(
    const std::vector<StereoFusionTrack>& tracks) {
  DenseFusionTrackQualityStats stats;
  stats.num_tracks = tracks.size();
  if (tracks.empty()) {
    return stats;
  }

  double normal_angle_sum = 0.0;
  double view_angle_sum = 0.0;
  for (const StereoFusionTrack& track : tracks) {
    stats.num_observations += track.num_observations;
    stats.max_observations =
        std::max(stats.max_observations, track.num_observations);
    if (track.num_observations == 1) {
      ++stats.num_single_observation_tracks;
    } else if (track.num_observations > 1) {
      ++stats.num_multi_observation_tracks;
    }
    normal_angle_sum += track.max_normal_angle_degrees;
    stats.max_normal_angle_degrees =
        std::max(stats.max_normal_angle_degrees,
                 track.max_normal_angle_degrees);
    view_angle_sum += track.max_view_angle_degrees;
    stats.max_view_angle_degrees =
        std::max(stats.max_view_angle_degrees, track.max_view_angle_degrees);
  }

  const double num_tracks = static_cast<double>(stats.num_tracks);
  stats.mean_observations =
      static_cast<double>(stats.num_observations) / num_tracks;
  stats.multi_observation_track_ratio =
      static_cast<double>(stats.num_multi_observation_tracks) / num_tracks;
  stats.mean_max_normal_angle_degrees = normal_angle_sum / num_tracks;
  stats.mean_max_view_angle_degrees = view_angle_sum / num_tracks;
  return stats;
}

void WriteFusionTrackQualityStats(
    const std::filesystem::path& path,
    const DenseFusionTrackQualityStats& stats) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion track quality stats: " +
                             path.string());
  }

  file << std::setprecision(17)
       << "num_tracks " << stats.num_tracks << "\n"
       << "num_single_observation_tracks "
       << stats.num_single_observation_tracks << "\n"
       << "num_multi_observation_tracks "
       << stats.num_multi_observation_tracks << "\n"
       << "num_observations " << stats.num_observations << "\n"
       << "max_observations " << stats.max_observations << "\n"
       << "mean_observations " << stats.mean_observations << "\n"
       << "multi_observation_track_ratio "
       << stats.multi_observation_track_ratio << "\n"
       << "mean_max_normal_angle_degrees "
       << stats.mean_max_normal_angle_degrees << "\n"
       << "max_normal_angle_degrees " << stats.max_normal_angle_degrees
       << "\n"
       << "mean_max_view_angle_degrees "
       << stats.mean_max_view_angle_degrees << "\n"
       << "max_view_angle_degrees " << stats.max_view_angle_degrees << "\n";
}

DenseTriangleMesh MeshDenseCloud(
    const std::vector<FusedPoint>& points,
    const DenseCloudMeshingOptions& options) {
  if (!std::isfinite(options.max_edge_length) ||
      options.max_edge_length < 0.0) {
    throw std::invalid_argument("Mesh max edge length must be non-negative.");
  }
  if (!std::isfinite(options.max_normal_angle_degrees) ||
      options.max_normal_angle_degrees < 0.0 ||
      options.max_normal_angle_degrees > 180.0) {
    throw std::invalid_argument("Mesh max normal angle must be in [0, 180].");
  }
  if (options.max_candidate_neighbors < 2) {
    throw std::invalid_argument(
        "Mesh max candidate neighbors must be at least 2.");
  }

  DenseTriangleMesh mesh;
  mesh.vertices = points;
  if (points.size() < 3) {
    return mesh;
  }

  std::unordered_set<TriangleKey, TriangleKeyHash> emitted;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const std::vector<std::size_t> neighbors = FindNearestNeighborIndices(
        points, i, options.max_candidate_neighbors);
    for (std::size_t neighbor_a = 0; neighbor_a + 1 < neighbors.size();
         ++neighbor_a) {
      for (std::size_t neighbor_b = neighbor_a + 1;
           neighbor_b < neighbors.size(); ++neighbor_b) {
        const std::size_t j = neighbors[neighbor_a];
        const std::size_t k = neighbors[neighbor_b];
        const TriangleKey key = MakeTriangleKey(i, j, k);
        if (!emitted.insert(key).second) {
          continue;
        }
        const FusedPoint& a = points[static_cast<std::size_t>(key.a)];
        const FusedPoint& b = points[static_cast<std::size_t>(key.b)];
        const FusedPoint& c = points[static_cast<std::size_t>(key.c)];
        if (!IsFinitePoint(a) || !IsFinitePoint(b) || !IsFinitePoint(c)) {
          continue;
        }
        if (TriangleAreaSquared(a, b, c) <=
            std::numeric_limits<double>::epsilon()) {
          continue;
        }
        if (!EdgeLengthsPass(a, b, c, options.max_edge_length)) {
          continue;
        }
        if (!TriangleNormalsPass(a, b, c,
                                 options.max_normal_angle_degrees)) {
          continue;
        }
        mesh.triangles.emplace_back(key.a, key.b, key.c);
      }
    }
  }
  return mesh;
}

DenseTriangleMeshStats ComputeDenseTriangleMeshStats(
    const DenseTriangleMesh& mesh) {
  DenseTriangleMeshStats stats;
  stats.num_vertices = mesh.vertices.size();
  stats.num_triangles = mesh.triangles.size();
  if (mesh.triangles.empty()) {
    return stats;
  }

  double edge_length_sum = 0.0;
  std::size_t num_valid_edges = 0;
  double area_sum = 0.0;
  std::size_t num_valid_areas = 0;
  stats.min_triangle_area = std::numeric_limits<double>::infinity();

  const auto valid_index = [&mesh](int index) {
    return index >= 0 &&
           static_cast<std::size_t>(index) < mesh.vertices.size();
  };
  const auto edge_length = [&mesh](int lhs, int rhs) {
    return (mesh.vertices[static_cast<std::size_t>(lhs)].xyz -
            mesh.vertices[static_cast<std::size_t>(rhs)].xyz)
        .norm();
  };

  for (const Eigen::Vector3i& triangle : mesh.triangles) {
    if (!valid_index(triangle.x()) || !valid_index(triangle.y()) ||
        !valid_index(triangle.z())) {
      continue;
    }

    const double ab = edge_length(triangle.x(), triangle.y());
    const double ac = edge_length(triangle.x(), triangle.z());
    const double bc = edge_length(triangle.y(), triangle.z());
    edge_length_sum += ab + ac + bc;
    stats.max_edge_length = std::max(stats.max_edge_length, ab);
    stats.max_edge_length = std::max(stats.max_edge_length, ac);
    stats.max_edge_length = std::max(stats.max_edge_length, bc);
    num_valid_edges += 3;

    const FusedPoint& a =
        mesh.vertices[static_cast<std::size_t>(triangle.x())];
    const FusedPoint& b =
        mesh.vertices[static_cast<std::size_t>(triangle.y())];
    const FusedPoint& c =
        mesh.vertices[static_cast<std::size_t>(triangle.z())];
    const double area = std::sqrt(TriangleAreaSquared(a, b, c));
    area_sum += area;
    stats.min_triangle_area = std::min(stats.min_triangle_area, area);
    stats.max_triangle_area = std::max(stats.max_triangle_area, area);
    ++num_valid_areas;
  }

  if (num_valid_edges > 0) {
    stats.mean_edge_length =
        edge_length_sum / static_cast<double>(num_valid_edges);
  }
  if (num_valid_areas > 0) {
    stats.mean_triangle_area =
        area_sum / static_cast<double>(num_valid_areas);
  } else {
    stats.min_triangle_area = 0.0;
  }
  if (!std::isfinite(stats.min_triangle_area)) {
    stats.min_triangle_area = 0.0;
  }
  return stats;
}

void WriteDenseTriangleMeshPly(const std::filesystem::path& path,
                               const DenseTriangleMesh& mesh) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense triangle mesh: " +
                             path.string());
  }

  file << "ply\n"
       << "format ascii 1.0\n"
       << "element vertex " << mesh.vertices.size() << "\n"
       << "property double x\n"
       << "property double y\n"
       << "property double z\n"
       << "property float nx\n"
       << "property float ny\n"
       << "property float nz\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "element face " << mesh.triangles.size() << "\n"
       << "property list uchar int vertex_indices\n"
       << "end_header\n";
  file << std::setprecision(17);
  for (const FusedPoint& point : mesh.vertices) {
    file << point.xyz.x() << " " << point.xyz.y() << " " << point.xyz.z()
         << " " << point.normal.x() << " " << point.normal.y() << " "
         << point.normal.z() << " " << static_cast<int>(point.color.x())
         << " " << static_cast<int>(point.color.y()) << " "
         << static_cast<int>(point.color.z()) << "\n";
  }
  for (const Eigen::Vector3i& triangle : mesh.triangles) {
    file << "3 " << triangle.x() << " " << triangle.y() << " "
         << triangle.z() << "\n";
  }
}

void WriteDenseTriangleMeshStats(const std::filesystem::path& path,
                                 const DenseTriangleMeshStats& stats) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense triangle mesh stats: " +
                             path.string());
  }

  file << std::setprecision(17)
       << "num_vertices " << stats.num_vertices << "\n"
       << "num_triangles " << stats.num_triangles << "\n"
       << "mean_edge_length " << stats.mean_edge_length << "\n"
       << "max_edge_length " << stats.max_edge_length << "\n"
       << "mean_triangle_area " << stats.mean_triangle_area << "\n"
       << "min_triangle_area " << stats.min_triangle_area << "\n"
       << "max_triangle_area " << stats.max_triangle_area << "\n";
}

}  // namespace colite::mvs
