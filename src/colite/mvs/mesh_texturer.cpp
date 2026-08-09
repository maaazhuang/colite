// View selection and atlas construction follow the BSD-licensed COLMAP 4.0
// mesh texturer at an algorithmic level. This implementation is rewritten for
// COLITE's camera, image, and mesh types and uses software BVH ray occlusion.

#include "colite/mvs/mesh_texturer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "colite/core/geometry.h"

namespace colite::mvs {
namespace {

using FaceAdjacency = std::vector<std::vector<std::size_t>>;

struct Projection {
  std::array<Eigen::Vector2f, 3> pixels;
  std::array<float, 3> inverse_depths{};
  bool valid = false;
};

struct FaceRegion {
  int view_id = -1;
  std::vector<std::size_t> face_ids;
  std::vector<std::array<Eigen::Vector2f, 3>> projections;
  Eigen::Vector2f minimum = Eigen::Vector2f::Zero();
  Eigen::Vector2f maximum = Eigen::Vector2f::Zero();
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
};

struct OcclusionBvhNode {
  Eigen::Vector3d minimum = Eigen::Vector3d::Zero();
  Eigen::Vector3d maximum = Eigen::Vector3d::Zero();
  int left = -1;
  int right = -1;
  std::size_t begin = 0;
  std::size_t end = 0;
};

class TriangleOcclusionBvh {
 public:
  explicit TriangleOcclusionBvh(const DenseTriangleMesh& mesh)
      : mesh_(mesh) {
    face_ids_.reserve(mesh.triangles.size());
    for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
      const Eigen::Vector3i& face = mesh.triangles[face_id];
      const Eigen::Vector3d cross =
          (mesh.vertices[face.y()].xyz - mesh.vertices[face.x()].xyz)
              .cross(mesh.vertices[face.z()].xyz -
                     mesh.vertices[face.x()].xyz);
      if (cross.squaredNorm() > 1e-20) {
        face_ids_.push_back(face_id);
      }
    }
    if (!face_ids_.empty()) {
      Build(0, face_ids_.size());
    }
  }

  bool IsOccluded(const Eigen::Vector3d& origin,
                  const Eigen::Vector3d& target,
                  std::size_t excluded_face,
                  int target_vertex,
                  double relative_tolerance) const {
    constexpr double kEndpointEpsilon = 1e-4;
    const Eigen::Vector3d direction = target - origin;
    const double distance = direction.norm();
    if (nodes_.empty() || !direction.allFinite() ||
        distance <= kEndpointEpsilon) {
      return false;
    }
    // A noisy reconstructed surface often contains neighboring triangles a
    // fraction of a percent in front of the target. Treating every such hit
    // as an occluder leaves isolated untextured triangles on otherwise
    // continuous roofs and ground planes. Keep true foreground occluders, but
    // accept intersections inside the configured endpoint-relative band.
    const double maximum_t =
        1.0 - std::max(kEndpointEpsilon / distance, relative_tolerance);
    std::vector<int> stack{0};
    while (!stack.empty()) {
      const int node_idx = stack.back();
      stack.pop_back();
      const OcclusionBvhNode& node = nodes_[node_idx];
      if (!IntersectsBox(origin, direction, maximum_t, node)) {
        continue;
      }
      if (node.left >= 0) {
        stack.push_back(node.left);
        stack.push_back(node.right);
        continue;
      }
      for (std::size_t index = node.begin; index < node.end; ++index) {
        const std::size_t face_id = face_ids_[index];
        if (face_id == excluded_face) {
          continue;
        }
        const Eigen::Vector3i& face = mesh_.triangles[face_id];
        if (face.x() == target_vertex || face.y() == target_vertex ||
            face.z() == target_vertex) {
          // Incident faces only meet this segment at its endpoint. Ignoring
          // them prevents floating-point Moller-Trumbore noise from turning a
          // shared vertex into a self-occluder.
          continue;
        }
        if (IntersectsTriangle(
                origin, direction, maximum_t,
                mesh_.vertices[face.x()].xyz,
                mesh_.vertices[face.y()].xyz,
                mesh_.vertices[face.z()].xyz)) {
          return true;
        }
      }
    }
    return false;
  }

 private:
  int Build(std::size_t begin, std::size_t end) {
    OcclusionBvhNode node;
    node.begin = begin;
    node.end = end;
    node.minimum = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::infinity());
    node.maximum = Eigen::Vector3d::Constant(
        -std::numeric_limits<double>::infinity());
    Eigen::Vector3d centroid_minimum = node.minimum;
    Eigen::Vector3d centroid_maximum = node.maximum;
    for (std::size_t index = begin; index < end; ++index) {
      const Eigen::Vector3i& face = mesh_.triangles[face_ids_[index]];
      Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
      for (int corner = 0; corner < 3; ++corner) {
        const Eigen::Vector3d& point = mesh_.vertices[face[corner]].xyz;
        node.minimum = node.minimum.cwiseMin(point);
        node.maximum = node.maximum.cwiseMax(point);
        centroid += point;
      }
      centroid /= 3.0;
      centroid_minimum = centroid_minimum.cwiseMin(centroid);
      centroid_maximum = centroid_maximum.cwiseMax(centroid);
    }
    const int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    constexpr std::size_t kLeafSize = 8;
    if (end - begin <= kLeafSize) {
      return node_idx;
    }
    Eigen::Index axis = 0;
    (centroid_maximum - centroid_minimum).maxCoeff(&axis);
    const std::size_t middle = begin + (end - begin) / 2;
    std::nth_element(
        face_ids_.begin() + static_cast<std::ptrdiff_t>(begin),
        face_ids_.begin() + static_cast<std::ptrdiff_t>(middle),
        face_ids_.begin() + static_cast<std::ptrdiff_t>(end),
        [&](std::size_t lhs, std::size_t rhs) {
          const Eigen::Vector3i& left_face = mesh_.triangles[lhs];
          const Eigen::Vector3i& right_face = mesh_.triangles[rhs];
          double left = 0.0;
          double right = 0.0;
          for (int corner = 0; corner < 3; ++corner) {
            left += mesh_.vertices[left_face[corner]].xyz[axis];
            right += mesh_.vertices[right_face[corner]].xyz[axis];
          }
          return left < right;
        });
    const int left = Build(begin, middle);
    const int right = Build(middle, end);
    nodes_[node_idx].left = left;
    nodes_[node_idx].right = right;
    return node_idx;
  }

  static bool IntersectsBox(const Eigen::Vector3d& origin,
                            const Eigen::Vector3d& direction,
                            double maximum_t,
                            const OcclusionBvhNode& node) {
    double minimum_t = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      if (std::abs(direction[axis]) <= 1e-15) {
        if (origin[axis] < node.minimum[axis] ||
            origin[axis] > node.maximum[axis]) {
          return false;
        }
        continue;
      }
      const double inverse = 1.0 / direction[axis];
      double first = (node.minimum[axis] - origin[axis]) * inverse;
      double second = (node.maximum[axis] - origin[axis]) * inverse;
      if (first > second) {
        std::swap(first, second);
      }
      minimum_t = std::max(minimum_t, first);
      maximum_t = std::min(maximum_t, second);
      if (minimum_t > maximum_t) {
        return false;
      }
    }
    return true;
  }

  static bool IntersectsTriangle(const Eigen::Vector3d& origin,
                                 const Eigen::Vector3d& direction,
                                 double maximum_t,
                                 const Eigen::Vector3d& first,
                                 const Eigen::Vector3d& second,
                                 const Eigen::Vector3d& third) {
    constexpr double kEpsilon = 1e-12;
    const Eigen::Vector3d edge1 = second - first;
    const Eigen::Vector3d edge2 = third - first;
    const Eigen::Vector3d p = direction.cross(edge2);
    const double determinant = edge1.dot(p);
    if (std::abs(determinant) <= kEpsilon) {
      return false;
    }
    const double inverse_determinant = 1.0 / determinant;
    const Eigen::Vector3d t = origin - first;
    const double u = t.dot(p) * inverse_determinant;
    if (u < -kEpsilon || u > 1.0 + kEpsilon) {
      return false;
    }
    const Eigen::Vector3d q = t.cross(edge1);
    const double v = direction.dot(q) * inverse_determinant;
    if (v < -kEpsilon || u + v > 1.0 + kEpsilon) {
      return false;
    }
    const double ray_t = edge2.dot(q) * inverse_determinant;
    return ray_t > kEpsilon && ray_t < maximum_t;
  }

  const DenseTriangleMesh& mesh_;
  std::vector<std::size_t> face_ids_;
  std::vector<OcclusionBvhNode> nodes_;
};

std::uint64_t EdgeKey(int first, int second) {
  if (first > second) std::swap(first, second);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(first)) << 32) |
         static_cast<std::uint32_t>(second);
}

FaceAdjacency BuildAdjacency(const DenseTriangleMesh& mesh) {
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> edge_faces;
  for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
    const Eigen::Vector3i& face = mesh.triangles[face_id];
    for (int edge = 0; edge < 3; ++edge) {
      edge_faces[EdgeKey(face[edge], face[(edge + 1) % 3])].push_back(face_id);
    }
  }
  FaceAdjacency adjacency(mesh.triangles.size());
  for (const auto& [key, faces] : edge_faces) {
    (void)key;
    if (faces.size() == 2) {
      adjacency[faces[0]].push_back(faces[1]);
      adjacency[faces[1]].push_back(faces[0]);
    }
  }
  return adjacency;
}

Projection ProjectFace(const DenseTriangleMesh& mesh,
                       std::size_t face_id,
                       const PatchMatchView& view) {
  Projection result;
  const Eigen::Vector3i& face = mesh.triangles[face_id];
  for (int corner = 0; corner < 3; ++corner) {
    const Eigen::Vector3d camera_point =
        view.cam_from_world.TransformPoint(mesh.vertices[face[corner]].xyz);
    if (!camera_point.allFinite() || camera_point.z() <= 1e-10) return result;
    try {
      result.pixels[corner] =
          ProjectPoint(view.camera, camera_point).cast<float>();
    } catch (const std::exception&) {
      return result;
    }
    result.inverse_depths[corner] =
        static_cast<float>(1.0 / camera_point.z());
  }
  result.valid = true;
  return result;
}

bool IsFaceProjectable(const DenseTriangleMesh& mesh,
                       std::size_t face_id,
                       const PatchMatchView& view,
                       const MeshTexturingOptions& options) {
  const Projection projection = ProjectFace(mesh, face_id, view);
  if (!projection.valid) return false;
  int visible_vertices = 0;
  for (const Eigen::Vector2f& pixel : projection.pixels) {
    if (pixel.x() >= 0.0F && pixel.y() >= 0.0F &&
        pixel.x() < static_cast<float>(view.camera.width) - 1.0F &&
        pixel.y() < static_cast<float>(view.camera.height) - 1.0F) {
      ++visible_vertices;
    }
  }
  if (visible_vertices < options.min_visible_vertices) return false;

  const Eigen::Vector3i& face = mesh.triangles[face_id];
  const Eigen::Vector3d first = mesh.vertices[face.x()].xyz;
  const Eigen::Vector3d second = mesh.vertices[face.y()].xyz;
  const Eigen::Vector3d third = mesh.vertices[face.z()].xyz;
  Eigen::Vector3d normal = (second - first).cross(third - first);
  if (normal.squaredNorm() <= 1e-24) return false;
  normal.normalize();
  const Eigen::Vector3d centroid = (first + second + third) / 3.0;
  const Eigen::Vector3d view_direction =
      (view.cam_from_world.CameraCenter() - centroid).normalized();
  return normal.dot(view_direction) >= options.min_cos_normal_angle;
}

Eigen::Vector3f Barycentric(const Eigen::Vector2f& point,
                            const std::array<Eigen::Vector2f, 3>& triangle) {
  const Eigen::Vector2f v0 = triangle[1] - triangle[0];
  const Eigen::Vector2f v1 = triangle[2] - triangle[0];
  const Eigen::Vector2f v2 = point - triangle[0];
  const float denominator = v0.x() * v1.y() - v1.x() * v0.y();
  if (std::abs(denominator) < 1e-12F) {
    return Eigen::Vector3f(-1.0F, -1.0F, -1.0F);
  }
  const float second = (v2.x() * v1.y() - v1.x() * v2.y()) / denominator;
  const float third = (v0.x() * v2.y() - v2.x() * v0.y()) / denominator;
  return Eigen::Vector3f(1.0F - second - third, second, third);
}

std::vector<int> SelectViews(const DenseTriangleMesh& mesh,
                             const std::vector<MeshTextureView>& views,
                             const FaceAdjacency& adjacency,
                             const MeshTexturingOptions& options) {
  std::vector<std::vector<double>> scores(
      mesh.triangles.size(), std::vector<double>(views.size(), -1.0));
  const TriangleOcclusionBvh occlusion_bvh(mesh);
  // COLMAP's CGAL any-intersection query is permissive when it happens to
  // return the target face, while a fully strict nearest-hit test rejects
  // legitimate faces on slightly noisy reconstructed planes. Keep the primary
  // endpoint band tightly bounded to 0.5%; the wider configured tolerance is
  // reserved for the topology-supported hole recovery below.
  const double primary_occlusion_tolerance =
      std::min(options.occlusion_relative_tolerance, 0.005);
  const auto process_view = [&](std::size_t view_id) {
    const PatchMatchView& view = views[view_id].geometry;
    const int source_width = static_cast<int>(view.camera.width);
    const int source_height = static_cast<int>(view.camera.height);
    std::vector<Projection> projections(mesh.triangles.size());
    for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
      projections[face_id] = ProjectFace(mesh, face_id, view);
    }
    const Eigen::Vector3d camera_center = view.cam_from_world.CameraCenter();
    for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
      const Projection& projection = projections[face_id];
      if (!projection.valid) continue;
      int visible_vertices = 0;
      for (const Eigen::Vector2f& pixel : projection.pixels) {
        if (pixel.x() >= 0.0F && pixel.y() >= 0.0F &&
            pixel.x() < source_width - 1.0F &&
            pixel.y() < source_height - 1.0F) {
          ++visible_vertices;
        }
      }
      if (visible_vertices < options.min_visible_vertices) continue;
      const Eigen::Vector3i& face = mesh.triangles[face_id];
      const Eigen::Vector3d first = mesh.vertices[face.x()].xyz;
      const Eigen::Vector3d second = mesh.vertices[face.y()].xyz;
      const Eigen::Vector3d third = mesh.vertices[face.z()].xyz;
      Eigen::Vector3d normal = (second - first).cross(third - first);
      if (normal.squaredNorm() <= 1e-24) continue;
      normal.normalize();
      const Eigen::Vector3d centroid = (first + second + third) / 3.0;
      const Eigen::Vector3d view_direction =
          (camera_center - centroid).normalized();
      const double cosine = normal.dot(view_direction);
      if (cosine < options.min_cos_normal_angle) continue;
      bool occluded = false;
      for (int corner = 0; corner < 3; ++corner) {
        if (occlusion_bvh.IsOccluded(camera_center,
                                     mesh.vertices[face[corner]].xyz,
                                     face_id, face[corner],
                                     primary_occlusion_tolerance)) {
          occluded = true;
          break;
        }
      }
      if (occluded) continue;
      const Eigen::Vector2f edge1 = projection.pixels[1] - projection.pixels[0];
      const Eigen::Vector2f edge2 = projection.pixels[2] - projection.pixels[0];
      const double area = std::abs(edge1.x() * edge2.y() -
                                   edge1.y() * edge2.x());
      scores[face_id][view_id] = area;
    }
  };
  if (options.num_threads == 1 || views.size() < 2) {
    for (std::size_t view_id = 0; view_id < views.size(); ++view_id) {
      process_view(view_id);
    }
  } else {
    cv::parallel_for_(cv::Range(0, static_cast<int>(views.size())),
                      [&](const cv::Range& range) {
      for (int view_id = range.start; view_id < range.end; ++view_id) {
        process_view(static_cast<std::size_t>(view_id));
      }
    });
  }
  std::vector<int> selected(mesh.triangles.size(), -1);
  for (std::size_t face_id = 0; face_id < selected.size(); ++face_id) {
    for (std::size_t view_id = 0; view_id < views.size(); ++view_id) {
      if (scores[face_id][view_id] >
          (selected[face_id] < 0 ? -1.0 : scores[face_id][selected[face_id]])) {
        selected[face_id] = static_cast<int>(view_id);
      }
    }
  }
  for (int iteration = 0;
       iteration < options.view_selection_smoothing_iterations; ++iteration) {
    std::vector<int> updated = selected;
    for (std::size_t face_id = 0; face_id < selected.size(); ++face_id) {
      std::unordered_map<int, int> counts;
      for (const std::size_t neighbor : adjacency[face_id]) {
        if (selected[neighbor] >= 0) ++counts[selected[neighbor]];
      }
      int best = selected[face_id];
      int best_count = best >= 0 ? counts[best] : 0;
      for (const auto& [view_id, count] : counts) {
        if (count > best_count && scores[face_id][view_id] >= 0.0) {
          best = view_id;
          best_count = count;
        }
      }
      updated[face_id] = best;
    }
    selected = std::move(updated);
  }

  // Local majority smoothing still leaves one-to-few-face view islands when
  // projected area oscillates over a nearly planar surface. Every such island
  // becomes an independent atlas chart and can appear as a triangular texture
  // fragment. Merge a small component only if a neighboring view is valid for
  // every face, retains at least half of the aggregate projected area, and is
  // supported across a near-coplanar boundary. This preserves actual roof
  // ridges and avoids replacing a good view with an occluded one.
  if (options.view_selection_min_component_size > 0) {
    constexpr double kMinimumNormalCosine = 0.94;  // about 20 degrees
    constexpr double kMinimumRelativeProjectedArea = 0.5;
    std::vector<Eigen::Vector3d> face_normals(
        mesh.triangles.size(), Eigen::Vector3d::Zero());
    for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
      const Eigen::Vector3i& face = mesh.triangles[face_id];
      Eigen::Vector3d normal =
          (mesh.vertices[face.y()].xyz - mesh.vertices[face.x()].xyz)
              .cross(mesh.vertices[face.z()].xyz -
                     mesh.vertices[face.x()].xyz);
      if (normal.squaredNorm() > 1e-24) {
        face_normals[face_id] = normal.normalized();
      }
    }
    for (int pass = 0; pass < 4; ++pass) {
      std::vector<bool> visited(selected.size(), false);
      bool changed = false;
      for (std::size_t seed = 0; seed < selected.size(); ++seed) {
        if (visited[seed] || selected[seed] < 0) continue;
        const int source_view = selected[seed];
        std::vector<std::size_t> component;
        std::queue<std::size_t> queue;
        queue.push(seed);
        visited[seed] = true;
        while (!queue.empty()) {
          const std::size_t face_id = queue.front();
          queue.pop();
          component.push_back(face_id);
          for (const std::size_t neighbor : adjacency[face_id]) {
            if (!visited[neighbor] && selected[neighbor] == source_view) {
              visited[neighbor] = true;
              queue.push(neighbor);
            }
          }
        }
        if (component.size() >= static_cast<std::size_t>(
                                    options.view_selection_min_component_size)) {
          continue;
        }

        std::unordered_map<int, int> boundary_support;
        for (const std::size_t face_id : component) {
          for (const std::size_t neighbor : adjacency[face_id]) {
            const int candidate = selected[neighbor];
            if (candidate < 0 || candidate == source_view ||
                face_normals[face_id].dot(face_normals[neighbor]) <
                    kMinimumNormalCosine) {
              continue;
            }
            ++boundary_support[candidate];
          }
        }

        double source_area = 0.0;
        for (const std::size_t face_id : component) {
          source_area += scores[face_id][source_view];
        }
        int best_view = -1;
        int best_support = 1;
        double best_relative_area = 0.0;
        for (const auto& [candidate, support] : boundary_support) {
          if (support < 2) continue;
          double candidate_area = 0.0;
          bool valid_for_component = true;
          for (const std::size_t face_id : component) {
            if (scores[face_id][candidate] < 0.0) {
              valid_for_component = false;
              break;
            }
            candidate_area += scores[face_id][candidate];
          }
          if (!valid_for_component) continue;
          const double relative_area =
              candidate_area / std::max(source_area, 1e-12);
          if (relative_area < kMinimumRelativeProjectedArea) continue;
          if (support > best_support ||
              (support == best_support &&
               relative_area > best_relative_area)) {
            best_view = candidate;
            best_support = support;
            best_relative_area = relative_area;
          }
        }
        if (best_view < 0) continue;
        for (const std::size_t face_id : component) {
          selected[face_id] = best_view;
        }
        changed = true;
      }
      if (!changed) break;
    }
  }

  // Strict ray tests can leave isolated, collapsed-UV triangles on a noisy
  // but continuous reconstructed plane. Fill only holes supported by at least
  // two textured edge-adjacent faces. Their labels form the candidate set;
  // choose the valid projection with the largest image area while retaining
  // image-boundary and front-facing checks. Requiring two supporting faces
  // prevents expansion from a single silhouette or isolated back surface.
  constexpr int kMaximumSupportedHoleFillIterations = 64;
  for (int iteration = 0;
       iteration < kMaximumSupportedHoleFillIterations; ++iteration) {
    std::vector<int> updated = selected;
    bool changed = false;
    for (std::size_t face_id = 0; face_id < selected.size(); ++face_id) {
      if (selected[face_id] >= 0) continue;
      std::unordered_map<int, int> candidate_views;
      for (const std::size_t neighbor : adjacency[face_id]) {
        if (selected[neighbor] < 0) continue;
        const Eigen::Vector3i& face = mesh.triangles[face_id];
        const Eigen::Vector3i& neighbor_face = mesh.triangles[neighbor];
        Eigen::Vector3d normal =
            (mesh.vertices[face.y()].xyz - mesh.vertices[face.x()].xyz)
                .cross(mesh.vertices[face.z()].xyz -
                       mesh.vertices[face.x()].xyz);
        Eigen::Vector3d neighbor_normal =
            (mesh.vertices[neighbor_face.y()].xyz -
             mesh.vertices[neighbor_face.x()].xyz)
                .cross(mesh.vertices[neighbor_face.z()].xyz -
                       mesh.vertices[neighbor_face.x()].xyz);
        if (normal.squaredNorm() <= 1e-24 ||
            neighbor_normal.squaredNorm() <= 1e-24 ||
            normal.normalized().dot(neighbor_normal.normalized()) < 0.94) {
          continue;
        }
        ++candidate_views[selected[neighbor]];
      }
      int best_view = -1;
      double best_area = -1.0;
      for (const auto& [view_id, support] : candidate_views) {
        if (support < 2) continue;
        if (!IsFaceProjectable(mesh, face_id, views[view_id].geometry,
                               options)) {
          continue;
        }
        const Eigen::Vector3i& face = mesh.triangles[face_id];
        const Eigen::Vector3d camera_center =
            views[view_id].geometry.cam_from_world.CameraCenter();
        bool occluded = false;
        for (int corner = 0; corner < 3; ++corner) {
          if (occlusion_bvh.IsOccluded(
                  camera_center, mesh.vertices[face[corner]].xyz, face_id,
                  face[corner], options.occlusion_relative_tolerance)) {
            occluded = true;
            break;
          }
        }
        if (occluded) continue;
        const Projection projection =
            ProjectFace(mesh, face_id, views[view_id].geometry);
        const Eigen::Vector2f first =
            projection.pixels[1] - projection.pixels[0];
        const Eigen::Vector2f second =
            projection.pixels[2] - projection.pixels[0];
        const double area =
            std::abs(first.x() * second.y() - first.y() * second.x());
        if (area > best_area) {
          best_view = view_id;
          best_area = area;
        }
      }
      if (best_view < 0) continue;
      updated[face_id] = best_view;
      changed = true;
    }
    selected = std::move(updated);
    if (!changed) break;
  }
  return selected;
}

std::vector<FaceRegion> BuildRegions(
    const DenseTriangleMesh& mesh,
    const std::vector<MeshTextureView>& views,
    const std::vector<int>& selected,
    const FaceAdjacency& adjacency) {
  std::vector<bool> visited(mesh.triangles.size(), false);
  std::vector<FaceRegion> regions;
  for (std::size_t seed = 0; seed < selected.size(); ++seed) {
    if (visited[seed] || selected[seed] < 0) continue;
    FaceRegion region;
    region.view_id = selected[seed];
    std::queue<std::size_t> queue;
    queue.push(seed);
    visited[seed] = true;
    while (!queue.empty()) {
      const std::size_t face_id = queue.front();
      queue.pop();
      region.face_ids.push_back(face_id);
      for (const std::size_t neighbor : adjacency[face_id]) {
        if (!visited[neighbor] && selected[neighbor] == region.view_id) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }
    region.minimum = Eigen::Vector2f::Constant(
        std::numeric_limits<float>::infinity());
    region.maximum = Eigen::Vector2f::Constant(
        -std::numeric_limits<float>::infinity());
    for (const std::size_t face_id : region.face_ids) {
      const Projection projection =
          ProjectFace(mesh, face_id, views[region.view_id].geometry);
      region.projections.push_back(projection.pixels);
      for (const Eigen::Vector2f& pixel : projection.pixels) {
        region.minimum = region.minimum.cwiseMin(pixel);
        region.maximum = region.maximum.cwiseMax(pixel);
      }
    }
    regions.push_back(std::move(region));
  }
  return regions;
}

void AppendVertexColorFallbackRegions(
    const std::vector<int>& selected,
    std::vector<FaceRegion>* regions) {
  constexpr float kFallbackTriangleSize = 4.0F;
  for (std::size_t face_id = 0; face_id < selected.size(); ++face_id) {
    if (selected[face_id] >= 0) continue;
    FaceRegion region;
    region.view_id = -1;
    region.face_ids.push_back(face_id);
    region.projections.push_back(
        {Eigen::Vector2f(0.0F, 0.0F),
         Eigen::Vector2f(kFallbackTriangleSize, 0.0F),
         Eigen::Vector2f(0.0F, kFallbackTriangleSize)});
    region.minimum = Eigen::Vector2f::Zero();
    region.maximum =
        Eigen::Vector2f::Constant(kFallbackTriangleSize);
    regions->push_back(std::move(region));
  }
}

bool PackRegions(std::vector<FaceRegion>* regions,
                 int atlas_size,
                 int padding,
                 double scale) {
  std::vector<std::size_t> order(regions->size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  for (FaceRegion& region : *regions) {
    region.width = std::max(
        1, static_cast<int>(std::ceil((region.maximum.x() - region.minimum.x()) *
                                     scale)) +
               1);
    region.height = std::max(
        1, static_cast<int>(std::ceil((region.maximum.y() - region.minimum.y()) *
                                     scale)) +
               1);
  }
  std::sort(order.begin(), order.end(), [&](std::size_t first,
                                             std::size_t second) {
    return (*regions)[first].height > (*regions)[second].height;
  });
  int x = 2 * padding + 1;
  int y = 0;
  int shelf_height = 2 * padding + 1;
  for (const std::size_t region_id : order) {
    FaceRegion& region = (*regions)[region_id];
    const int packed_width = region.width + 2 * padding;
    const int packed_height = region.height + 2 * padding;
    if (packed_width > atlas_size || packed_height > atlas_size) return false;
    if (x + packed_width > atlas_size) {
      y += shelf_height;
      x = 0;
      shelf_height = 0;
    }
    if (y + packed_height > atlas_size) return false;
    region.x = x + padding;
    region.y = y + padding;
    x += packed_width;
    shelf_height = std::max(shelf_height, packed_height);
  }
  return true;
}

cv::Vec3b BilinearSample(const cv::Mat& image, const Eigen::Vector2f& pixel) {
  const float x = std::clamp(pixel.x(), 0.0F,
                             static_cast<float>(image.cols - 1));
  const float y = std::clamp(pixel.y(), 0.0F,
                             static_cast<float>(image.rows - 1));
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, image.cols - 1);
  const int y1 = std::min(y0 + 1, image.rows - 1);
  const float dx = x - x0;
  const float dy = y - y0;
  const auto color = [&](int px, int py) {
    if (image.channels() == 1) {
      const std::uint8_t value = image.at<std::uint8_t>(py, px);
      return cv::Vec3f(value, value, value);
    }
    return cv::Vec3f(image.at<cv::Vec3b>(py, px));
  };
  const cv::Vec3f value =
      (1.0F - dy) * ((1.0F - dx) * color(x0, y0) + dx * color(x1, y0)) +
      dy * ((1.0F - dx) * color(x0, y1) + dx * color(x1, y1));
  return cv::Vec3b(cv::saturate_cast<std::uint8_t>(value[0]),
                   cv::saturate_cast<std::uint8_t>(value[1]),
                   cv::saturate_cast<std::uint8_t>(value[2]));
}

std::vector<cv::Vec3d> ComputeViewColorOffsets(
    const DenseTriangleMesh& mesh,
    const std::vector<MeshTextureView>& views,
    const std::vector<int>& selected,
    const FaceAdjacency& adjacency,
    double regularization) {
  const int num_views = static_cast<int>(views.size());
  std::vector<cv::Vec3d> offsets(views.size(), cv::Vec3d(0.0, 0.0, 0.0));
  if (num_views < 2) return offsets;

  std::vector<Eigen::Triplet<double>> triplets;
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(num_views, 3);
  for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
    const int left_view = selected[face_id];
    if (left_view < 0) continue;
    for (const std::size_t neighbor : adjacency[face_id]) {
      if (neighbor <= face_id) continue;
      const int right_view = selected[neighbor];
      if (right_view < 0 || right_view == left_view) continue;
      const Eigen::Vector3i& left = mesh.triangles[face_id];
      const Eigen::Vector3i& right = mesh.triangles[neighbor];
      for (int lc = 0; lc < 3; ++lc) {
        const int vertex_id = left[lc];
        bool shared = false;
        for (int rc = 0; rc < 3; ++rc) shared |= right[rc] == vertex_id;
        if (!shared) continue;
        const Eigen::Vector3d& point = mesh.vertices[vertex_id].xyz;
        const Eigen::Vector3d left_cam =
            views[left_view].geometry.cam_from_world.TransformPoint(point);
        const Eigen::Vector3d right_cam =
            views[right_view].geometry.cam_from_world.TransformPoint(point);
        if (left_cam.z() <= 0.0 || right_cam.z() <= 0.0) continue;
        const Eigen::Vector2f left_pixel =
            ProjectPoint(views[left_view].geometry.camera, left_cam).cast<float>();
        const Eigen::Vector2f right_pixel =
            ProjectPoint(views[right_view].geometry.camera, right_cam).cast<float>();
        if (left_pixel.x() < 0.0F || left_pixel.y() < 0.0F ||
            right_pixel.x() < 0.0F || right_pixel.y() < 0.0F ||
            left_pixel.x() >= views[left_view].image.cols - 1 ||
            left_pixel.y() >= views[left_view].image.rows - 1 ||
            right_pixel.x() >= views[right_view].image.cols - 1 ||
            right_pixel.y() >= views[right_view].image.rows - 1) {
          continue;
        }
        const cv::Vec3b left_color =
            BilinearSample(views[left_view].image, left_pixel);
        const cv::Vec3b right_color =
            BilinearSample(views[right_view].image, right_pixel);
        triplets.emplace_back(left_view, left_view, 1.0);
        triplets.emplace_back(right_view, right_view, 1.0);
        triplets.emplace_back(left_view, right_view, -1.0);
        triplets.emplace_back(right_view, left_view, -1.0);
        for (int channel = 0; channel < 3; ++channel) {
          const double difference =
              static_cast<double>(right_color[channel]) - left_color[channel];
          rhs(left_view, channel) += difference;
          rhs(right_view, channel) -= difference;
        }
      }
    }
  }
  if (triplets.empty()) return offsets;
  for (int view_id = 0; view_id < num_views; ++view_id) {
    triplets.emplace_back(view_id, view_id, regularization);
  }
  Eigen::SparseMatrix<double> system(num_views, num_views);
  system.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(system);
  if (solver.info() != Eigen::Success) return offsets;
  const Eigen::MatrixXd solution = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !solution.allFinite()) return offsets;
  for (int view_id = 0; view_id < num_views; ++view_id) {
    offsets[view_id] = cv::Vec3d(solution(view_id, 0), solution(view_id, 1),
                                solution(view_id, 2));
  }
  return offsets;
}

std::vector<std::array<cv::Vec3d, 3>> ComputeLocalSeamColorOffsets(
    const DenseTriangleMesh& mesh,
    const std::vector<MeshTextureView>& views,
    const std::vector<int>& selected,
    const FaceAdjacency& adjacency,
    const std::vector<FaceRegion>& regions,
    const std::vector<cv::Vec3d>& view_offsets,
    double regularization) {
  std::vector<std::array<cv::Vec3d, 3>> face_offsets(
      mesh.triangles.size(),
      {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0),
       cv::Vec3d(0.0, 0.0, 0.0)});
  if (views.size() < 2 || regions.empty()) return face_offsets;

  std::vector<int> face_to_region(mesh.triangles.size(), -1);
  for (std::size_t region_id = 0; region_id < regions.size(); ++region_id) {
    if (regions[region_id].view_id < 0) continue;
    for (const std::size_t face_id : regions[region_id].face_ids) {
      face_to_region[face_id] = static_cast<int>(region_id);
    }
  }

  struct SeamEquation {
    std::size_t left_variable = 0;
    std::size_t right_variable = 0;
    cv::Vec3d difference = cv::Vec3d(0.0, 0.0, 0.0);
  };
  std::unordered_map<std::uint64_t, std::size_t> variable_by_key;
  std::vector<SeamEquation> equations;
  const auto variable_for = [&](int region_id, int vertex_id) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(region_id))
         << 32) |
        static_cast<std::uint32_t>(vertex_id);
    const auto [it, inserted] =
        variable_by_key.emplace(key, variable_by_key.size());
    (void)inserted;
    return it->second;
  };
  const auto sample_vertex = [&](int view_id, int vertex_id,
                                 cv::Vec3d* color) {
    const MeshTextureView& view = views[view_id];
    const Eigen::Vector3d camera_point =
        view.geometry.cam_from_world.TransformPoint(
            mesh.vertices[vertex_id].xyz);
    if (!camera_point.allFinite() || camera_point.z() <= 0.0) return false;
    const Eigen::Vector2f pixel =
        ProjectPoint(view.geometry.camera, camera_point).cast<float>();
    if (pixel.x() < 0.0F || pixel.y() < 0.0F ||
        pixel.x() >= view.image.cols - 1 ||
        pixel.y() >= view.image.rows - 1) {
      return false;
    }
    const cv::Vec3b sampled = BilinearSample(view.image, pixel);
    *color = cv::Vec3d(sampled[0], sampled[1], sampled[2]) +
             view_offsets[view_id];
    return true;
  };

  for (std::size_t left_face = 0; left_face < mesh.triangles.size();
       ++left_face) {
    const int left_view = selected[left_face];
    const int left_region = face_to_region[left_face];
    if (left_view < 0 || left_region < 0) continue;
    for (const std::size_t right_face : adjacency[left_face]) {
      if (right_face <= left_face) continue;
      const int right_view = selected[right_face];
      const int right_region = face_to_region[right_face];
      if (right_view < 0 || right_view == left_view || right_region < 0) {
        continue;
      }
      const Eigen::Vector3i& left = mesh.triangles[left_face];
      const Eigen::Vector3i& right = mesh.triangles[right_face];
      for (int left_corner = 0; left_corner < 3; ++left_corner) {
        const int vertex_id = left[left_corner];
        bool shared = false;
        for (int right_corner = 0; right_corner < 3; ++right_corner) {
          shared |= right[right_corner] == vertex_id;
        }
        if (!shared) continue;
        cv::Vec3d left_color;
        cv::Vec3d right_color;
        if (!sample_vertex(left_view, vertex_id, &left_color) ||
            !sample_vertex(right_view, vertex_id, &right_color)) {
          continue;
        }
        equations.push_back(
            {variable_for(left_region, vertex_id),
             variable_for(right_region, vertex_id),
             right_color - left_color});
      }
    }
  }
  if (equations.empty()) return face_offsets;

  const std::size_t num_variables = variable_by_key.size();
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(equations.size() * 4 + num_variables);
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(num_variables, 3);
  for (const SeamEquation& equation : equations) {
    const Eigen::Index left =
        static_cast<Eigen::Index>(equation.left_variable);
    const Eigen::Index right =
        static_cast<Eigen::Index>(equation.right_variable);
    triplets.emplace_back(left, left, 1.0);
    triplets.emplace_back(right, right, 1.0);
    triplets.emplace_back(left, right, -1.0);
    triplets.emplace_back(right, left, -1.0);
    for (int channel = 0; channel < 3; ++channel) {
      rhs(left, channel) += equation.difference[channel];
      rhs(right, channel) -= equation.difference[channel];
    }
  }
  for (std::size_t variable = 0; variable < num_variables; ++variable) {
    triplets.emplace_back(static_cast<Eigen::Index>(variable),
                          static_cast<Eigen::Index>(variable),
                          regularization);
  }
  Eigen::SparseMatrix<double> system(
      static_cast<Eigen::Index>(num_variables),
      static_cast<Eigen::Index>(num_variables));
  system.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(system);
  if (solver.info() != Eigen::Success) return face_offsets;
  const Eigen::MatrixXd solution = solver.solve(rhs);
  if (solver.info() != Eigen::Success || !solution.allFinite()) {
    return face_offsets;
  }

  constexpr double kMaximumLocalOffset = 64.0;
  for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
    const int region_id = face_to_region[face_id];
    if (region_id < 0) continue;
    for (int corner = 0; corner < 3; ++corner) {
      const int vertex_id = mesh.triangles[face_id][corner];
      const std::uint64_t key =
          (static_cast<std::uint64_t>(
               static_cast<std::uint32_t>(region_id))
           << 32) |
          static_cast<std::uint32_t>(vertex_id);
      const auto variable_it = variable_by_key.find(key);
      if (variable_it == variable_by_key.end()) continue;
      const Eigen::Index variable =
          static_cast<Eigen::Index>(variable_it->second);
      for (int channel = 0; channel < 3; ++channel) {
        face_offsets[face_id][corner][channel] =
            std::clamp(solution(variable, channel), -kMaximumLocalOffset,
                       kMaximumLocalOffset);
      }
    }
  }
  return face_offsets;
}

}  // namespace

bool MeshTexturingOptions::Check() const {
  return atlas_size >= 16 && max_atlas_size >= atlas_size &&
         atlas_patch_padding >= 0 && inpaint_radius >= 0 &&
         atlas_patch_padding * 2 + 1 < atlas_size &&
         std::isfinite(texture_scale) && texture_scale > 0.0 &&
         std::isfinite(min_cos_normal_angle) && min_cos_normal_angle >= -1.0 &&
         min_cos_normal_angle <= 1.0 &&
         min_visible_vertices >= 1 && min_visible_vertices <= 3 &&
         view_selection_smoothing_iterations >= 0 &&
         view_selection_min_component_size >= 0 &&
         visibility_resolution >= 16 &&
         std::isfinite(occlusion_relative_tolerance) &&
         occlusion_relative_tolerance >= 0.0 &&
         std::isfinite(color_correction_regularization) &&
         color_correction_regularization > 0.0 &&
         (num_threads == -1 || num_threads > 0);
}

TexturedTriangleMesh TextureMesh(const DenseTriangleMesh& mesh,
                                 const std::vector<MeshTextureView>& views,
                                 const MeshTexturingOptions& options) {
  if (!options.Check()) {
    throw std::invalid_argument("Invalid mesh texturing options.");
  }
  for (const Eigen::Vector3i& face : mesh.triangles) {
    if (face.minCoeff() < 0 ||
        face.maxCoeff() >= static_cast<int>(mesh.vertices.size()) ||
        face.x() == face.y() || face.y() == face.z() ||
        face.z() == face.x()) {
      throw std::invalid_argument(
          "Texture input contains an invalid triangle.");
    }
  }
  for (const MeshTextureView& view : views) {
    if (view.image.empty() || view.image.depth() != CV_8U ||
        (view.image.channels() != 1 && view.image.channels() != 3)) {
      throw std::invalid_argument("Texture views require 8-bit images.");
    }
    if (view.geometry.camera.width == 0 ||
        view.geometry.camera.height == 0 ||
        view.image.cols != static_cast<int>(view.geometry.camera.width) ||
        view.image.rows != static_cast<int>(view.geometry.camera.height)) {
      throw std::invalid_argument(
          "Texture image and camera dimensions must match.");
    }
  }
  TexturedTriangleMesh result;
  result.mesh = mesh;
  result.face_uvs.assign(mesh.triangles.size() * 3,
                         Eigen::Vector2f::Zero());
  const FaceAdjacency adjacency = BuildAdjacency(mesh);
  result.face_view_ids = SelectViews(mesh, views, adjacency, options);
  std::vector<FaceRegion> regions =
      BuildRegions(mesh, views, result.face_view_ids, adjacency);
  if (options.use_vertex_color_fallback) {
    AppendVertexColorFallbackRegions(result.face_view_ids, &regions);
  }
  double atlas_scale = options.texture_scale;
  int atlas_size = options.atlas_size;
  while (!PackRegions(&regions, atlas_size, options.atlas_patch_padding,
                      atlas_scale)) {
    if (atlas_size <= options.max_atlas_size / 2) {
      atlas_size *= 2;
    } else {
      atlas_scale *= 0.8;
      if (atlas_scale < 1e-4) {
        throw std::runtime_error("Texture regions do not fit into the atlas.");
      }
    }
  }
  int atlas_height = 1;
  for (const FaceRegion& region : regions) {
    atlas_height = std::max(
        atlas_height,
        region.y + region.height + options.atlas_patch_padding);
  }
  atlas_height = std::min(atlas_height, atlas_size);
  result.texture_atlas = cv::Mat(atlas_height, atlas_size,
                                 CV_8UC3, cv::Scalar(127, 127, 127));
  cv::Mat mask(atlas_height, atlas_size, CV_8UC1,
               cv::Scalar(0));
  // Pixel (0, 0) is outside every packed region and retains the neutral atlas
  // background. Its center remains valid even when there are no regions and
  // the atlas is only one pixel high.
  const float fallback_u = 0.5F / atlas_size;
  const float fallback_v = 1.0F - 0.5F / atlas_height;
  for (Eigen::Vector2f& uv : result.face_uvs) {
    uv = Eigen::Vector2f(fallback_u, fallback_v);
  }
  const std::vector<cv::Vec3d> view_color_offsets =
      options.apply_color_correction
          ? ComputeViewColorOffsets(mesh, views, result.face_view_ids,
                                    adjacency,
                                    options.color_correction_regularization)
          : std::vector<cv::Vec3d>(views.size(), cv::Vec3d(0.0, 0.0, 0.0));
  const std::vector<std::array<cv::Vec3d, 3>> local_seam_offsets =
      options.apply_color_correction
          ? ComputeLocalSeamColorOffsets(
                mesh, views, result.face_view_ids, adjacency, regions,
                view_color_offsets,
                options.color_correction_regularization)
          : std::vector<std::array<cv::Vec3d, 3>>(
                mesh.triangles.size(),
                {cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0),
                 cv::Vec3d(0.0, 0.0, 0.0)});
  for (const FaceRegion& region : regions) {
    const cv::Mat* source =
        region.view_id >= 0 ? &views[region.view_id].image : nullptr;
    for (std::size_t local_face = 0; local_face < region.face_ids.size();
         ++local_face) {
      const std::size_t face_id = region.face_ids[local_face];
      const Eigen::Vector3i& mesh_face = mesh.triangles[face_id];
      std::array<Eigen::Vector2f, 3> atlas_triangle;
      for (int corner = 0; corner < 3; ++corner) {
        atlas_triangle[corner] =
            (region.projections[local_face][corner] - region.minimum) *
                static_cast<float>(atlas_scale) +
            Eigen::Vector2f(region.x, region.y);
        result.face_uvs[face_id * 3 + corner] = Eigen::Vector2f(
            atlas_triangle[corner].x() / atlas_size,
            1.0F - atlas_triangle[corner].y() / atlas_height);
      }
      const int min_x = std::max(0, static_cast<int>(std::floor(std::min(
                                    {atlas_triangle[0].x(), atlas_triangle[1].x(),
                                     atlas_triangle[2].x()}))));
      const int max_x = std::min(
          atlas_size - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_triangle[0].x(), atlas_triangle[1].x(),
               atlas_triangle[2].x()}))));
      const int min_y = std::max(0, static_cast<int>(std::floor(std::min(
                                    {atlas_triangle[0].y(), atlas_triangle[1].y(),
                                     atlas_triangle[2].y()}))));
      const int max_y = std::min(
          atlas_height - 1,
          static_cast<int>(std::ceil(std::max(
              {atlas_triangle[0].y(), atlas_triangle[1].y(),
               atlas_triangle[2].y()}))));
      for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
          const Eigen::Vector3f weights = Barycentric(
              Eigen::Vector2f(x + 0.5F, y + 0.5F), atlas_triangle);
          if (weights.minCoeff() < -1e-4F) continue;
          const Eigen::Vector2f source_pixel =
              weights.x() * region.projections[local_face][0] +
              weights.y() * region.projections[local_face][1] +
              weights.z() * region.projections[local_face][2];
          cv::Vec3b sampled;
          cv::Vec3d offset(0.0, 0.0, 0.0);
          if (source != nullptr) {
            if (source_pixel.x() < 0.0F || source_pixel.y() < 0.0F ||
                source_pixel.x() >= source->cols - 1 ||
                source_pixel.y() >= source->rows - 1) {
              continue;
            }
            sampled = BilinearSample(*source, source_pixel);
            offset = view_color_offsets[region.view_id];
            for (int corner = 0; corner < 3; ++corner) {
              offset += static_cast<double>(weights[corner]) *
                        local_seam_offsets[face_id][corner];
            }
          } else {
            const Eigen::Vector3f color =
                weights.x() *
                    mesh.vertices[mesh_face[0]].color.cast<float>() +
                weights.y() *
                    mesh.vertices[mesh_face[1]].color.cast<float>() +
                weights.z() *
                    mesh.vertices[mesh_face[2]].color.cast<float>();
            sampled = cv::Vec3b(
                cv::saturate_cast<std::uint8_t>(color.z()),
                cv::saturate_cast<std::uint8_t>(color.y()),
                cv::saturate_cast<std::uint8_t>(color.x()));
          }
          result.texture_atlas.at<cv::Vec3b>(y, x) = cv::Vec3b(
              cv::saturate_cast<std::uint8_t>(sampled[0] + offset[0]),
              cv::saturate_cast<std::uint8_t>(sampled[1] + offset[1]),
              cv::saturate_cast<std::uint8_t>(sampled[2] + offset[2]));
          mask.at<std::uint8_t>(y, x) = 255;
        }
      }
    }
  }
  for (int iteration = 0; iteration < options.inpaint_radius;
       ++iteration) {
    cv::Mat dilated_atlas;
    cv::Mat dilated_mask;
    cv::dilate(result.texture_atlas, dilated_atlas, cv::Mat());
    cv::dilate(mask, dilated_mask, cv::Mat());
    dilated_atlas.copyTo(result.texture_atlas, dilated_mask - mask);
    mask = dilated_mask;
  }
  return result;
}

void WriteTexturedTriangleMesh(
    const std::filesystem::path& output_directory,
    const TexturedTriangleMesh& textured_mesh) {
  if (textured_mesh.face_uvs.size() !=
      textured_mesh.mesh.triangles.size() * 3) {
    throw std::invalid_argument("Textured mesh has invalid UV coordinates.");
  }
  std::filesystem::create_directories(output_directory);
  if (!cv::imwrite((output_directory / "texture.png").string(),
                   textured_mesh.texture_atlas)) {
    throw std::runtime_error("Failed to write texture atlas.");
  }
  std::ofstream material(output_directory / "mesh.mtl");
  std::ofstream object(output_directory / "mesh.obj");
  if (!material || !object) {
    throw std::runtime_error("Failed to write textured OBJ/MTL.");
  }
  material << "newmtl colite_texture\nKd 1 1 1\nmap_Kd texture.png\n";
  object << std::setprecision(17);
  object << "mtllib mesh.mtl\n";
  for (const FusedPoint& vertex : textured_mesh.mesh.vertices) {
    object << "v " << vertex.xyz.x() << ' ' << vertex.xyz.y() << ' '
           << vertex.xyz.z() << '\n';
  }
  for (const FusedPoint& vertex : textured_mesh.mesh.vertices) {
    object << "vn " << vertex.normal.x() << ' ' << vertex.normal.y() << ' '
           << vertex.normal.z() << '\n';
  }
  for (const Eigen::Vector2f& uv : textured_mesh.face_uvs) {
    object << "vt " << uv.x() << ' ' << uv.y() << '\n';
  }
  object << "usemtl colite_texture\n";
  for (std::size_t face_id = 0;
       face_id < textured_mesh.mesh.triangles.size(); ++face_id) {
    const Eigen::Vector3i& face = textured_mesh.mesh.triangles[face_id];
    object << "f";
    for (int corner = 0; corner < 3; ++corner) {
      const int vertex_index = face[corner] + 1;
      const std::size_t uv_index = face_id * 3 + corner + 1;
      object << ' ' << vertex_index << '/' << uv_index << '/' << vertex_index;
    }
    object << '\n';
  }
}

}  // namespace colite::mvs
