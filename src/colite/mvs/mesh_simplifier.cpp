#include "colite/mvs/meshing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace colite::mvs {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinAreaSquared = 1e-24;

struct VertexState {
  FusedPoint point;
  Eigen::Matrix4d quadric = Eigen::Matrix4d::Zero();
  std::vector<int> faces;
  std::uint64_t version = 0;
  bool active = true;
};

struct FaceState {
  std::array<int, 3> vertices{};
  bool active = true;
};

struct CollapseCandidate {
  double error = 0.0;
  int first = -1;
  int second = -1;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  std::uint64_t first_version = 0;
  std::uint64_t second_version = 0;
};

struct GreaterError {
  bool operator()(const CollapseCandidate& lhs,
                  const CollapseCandidate& rhs) const {
    return lhs.error > rhs.error;
  }
};

std::array<int, 2> OrderedEdge(int first, int second) {
  return first < second ? std::array<int, 2>{first, second}
                        : std::array<int, 2>{second, first};
}

bool FaceContains(const FaceState& face, int vertex) {
  return face.vertices[0] == vertex || face.vertices[1] == vertex ||
         face.vertices[2] == vertex;
}

std::vector<int> ActiveNeighbors(int vertex,
                                 const std::vector<VertexState>& vertices,
                                 const std::vector<FaceState>& faces) {
  std::unordered_set<int> unique;
  for (const int face_id : vertices[vertex].faces) {
    if (face_id < 0 || static_cast<std::size_t>(face_id) >= faces.size() ||
        !faces[face_id].active || !FaceContains(faces[face_id], vertex)) {
      continue;
    }
    for (const int other : faces[face_id].vertices) {
      if (other != vertex && vertices[other].active) {
        unique.insert(other);
      }
    }
  }
  return {unique.begin(), unique.end()};
}

int ActiveEdgeFaceCount(int first,
                        int second,
                        const std::vector<VertexState>& vertices,
                        const std::vector<FaceState>& faces) {
  int count = 0;
  for (const int face_id : vertices[first].faces) {
    if (face_id >= 0 && static_cast<std::size_t>(face_id) < faces.size() &&
        faces[face_id].active && FaceContains(faces[face_id], first) &&
        FaceContains(faces[face_id], second)) {
      ++count;
    }
  }
  return count;
}

bool IsBoundaryVertex(int vertex,
                      const std::vector<VertexState>& vertices,
                      const std::vector<FaceState>& faces) {
  for (const int neighbor : ActiveNeighbors(vertex, vertices, faces)) {
    if (ActiveEdgeFaceCount(vertex, neighbor, vertices, faces) == 1) {
      return true;
    }
  }
  return false;
}

double QuadricError(const Eigen::Matrix4d& quadric,
                    const Eigen::Vector3d& position) {
  const Eigen::Vector4d homogeneous(position.x(), position.y(), position.z(),
                                    1.0);
  return std::max(0.0, homogeneous.dot(quadric * homogeneous));
}

CollapseCandidate MakeCandidate(int first,
                                int second,
                                const std::vector<VertexState>& vertices) {
  const auto edge = OrderedEdge(first, second);
  first = edge[0];
  second = edge[1];
  const Eigen::Matrix4d quadric =
      vertices[first].quadric + vertices[second].quadric;
  std::array<Eigen::Vector3d, 4> positions = {
      vertices[first].point.xyz, vertices[second].point.xyz,
      0.5 * (vertices[first].point.xyz + vertices[second].point.xyz),
      Eigen::Vector3d::Zero()};
  std::size_t num_positions = 3;
  const Eigen::Matrix3d system = quadric.topLeftCorner<3, 3>();
  if (std::abs(system.determinant()) > 1e-12) {
    const Eigen::Vector3d unconstrained =
        system.ldlt().solve(-quadric.topRightCorner<3, 1>());
    const Eigen::Vector3d edge_direction =
        vertices[second].point.xyz - vertices[first].point.xyz;
    if (unconstrained.allFinite() &&
        edge_direction.squaredNorm() > std::numeric_limits<double>::epsilon()) {
      // The unconstrained QEM minimizer can be arbitrarily far from the
      // collapsed edge for nearly singular local quadrics. Keep the optimal
      // candidate on the edge segment so simplification cannot expand the
      // mesh beyond its local one-ring envelope.
      const double t = std::clamp(
          (unconstrained - vertices[first].point.xyz).dot(edge_direction) /
              edge_direction.squaredNorm(),
          0.0, 1.0);
      positions[num_positions++] =
          vertices[first].point.xyz + t * edge_direction;
    }
  }
  CollapseCandidate candidate;
  candidate.first = first;
  candidate.second = second;
  candidate.first_version = vertices[first].version;
  candidate.second_version = vertices[second].version;
  candidate.error = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < num_positions; ++i) {
    if (!positions[i].allFinite()) {
      continue;
    }
    const double error = QuadricError(quadric, positions[i]);
    if (error < candidate.error) {
      candidate.error = error;
      candidate.position = positions[i];
    }
  }
  return candidate;
}

bool SatisfiesLinkCondition(int first,
                            int second,
                            const std::vector<VertexState>& vertices,
                            const std::vector<FaceState>& faces) {
  const int edge_faces = ActiveEdgeFaceCount(first, second, vertices, faces);
  if (edge_faces < 1 || edge_faces > 2) {
    return false;
  }
  const std::vector<int> first_neighbors =
      ActiveNeighbors(first, vertices, faces);
  const std::vector<int> second_neighbors =
      ActiveNeighbors(second, vertices, faces);
  std::unordered_set<int> first_set(first_neighbors.begin(),
                                    first_neighbors.end());
  int common = 0;
  for (const int neighbor : second_neighbors) {
    if (neighbor != first && neighbor != second &&
        first_set.count(neighbor) != 0) {
      ++common;
    }
  }
  return common == edge_faces;
}

bool PreservesFaceNormals(const CollapseCandidate& candidate,
                          double minimum_normal_dot,
                          const std::vector<VertexState>& vertices,
                          const std::vector<FaceState>& faces) {
  std::unordered_set<int> checked_faces;
  for (const int endpoint : {candidate.first, candidate.second}) {
    for (const int face_id : vertices[endpoint].faces) {
      if (!checked_faces.insert(face_id).second || face_id < 0 ||
          static_cast<std::size_t>(face_id) >= faces.size() ||
          !faces[face_id].active) {
        continue;
      }
      const FaceState& face = faces[face_id];
      if (FaceContains(face, candidate.first) &&
          FaceContains(face, candidate.second)) {
        continue;
      }
      std::array<Eigen::Vector3d, 3> old_points;
      std::array<Eigen::Vector3d, 3> new_points;
      for (int corner = 0; corner < 3; ++corner) {
        old_points[corner] = vertices[face.vertices[corner]].point.xyz;
        new_points[corner] =
            face.vertices[corner] == candidate.first ||
                    face.vertices[corner] == candidate.second
                ? candidate.position
                : old_points[corner];
      }
      const Eigen::Vector3d old_cross =
          (old_points[1] - old_points[0]).cross(old_points[2] - old_points[0]);
      const Eigen::Vector3d new_cross =
          (new_points[1] - new_points[0]).cross(new_points[2] - new_points[0]);
      if (new_cross.squaredNorm() <= kMinAreaSquared ||
          old_cross.squaredNorm() <= kMinAreaSquared ||
          old_cross.normalized().dot(new_cross.normalized()) <
              minimum_normal_dot) {
        return false;
      }
    }
  }
  return true;
}

bool CanCollapse(const CollapseCandidate& candidate,
                 const MeshSimplificationOptions& options,
                 const std::vector<VertexState>& vertices,
                 const std::vector<FaceState>& faces) {
  if (!vertices[candidate.first].active ||
      !vertices[candidate.second].active ||
      candidate.first_version != vertices[candidate.first].version ||
      candidate.second_version != vertices[candidate.second].version ||
      !SatisfiesLinkCondition(candidate.first, candidate.second, vertices,
                              faces)) {
    return false;
  }
  if (options.preserve_boundary) {
    const bool first_boundary =
        IsBoundaryVertex(candidate.first, vertices, faces);
    const bool second_boundary =
        IsBoundaryVertex(candidate.second, vertices, faces);
    const bool boundary_edge = ActiveEdgeFaceCount(
                                   candidate.first, candidate.second, vertices,
                                   faces) == 1;
    if (first_boundary != second_boundary ||
        (first_boundary && !boundary_edge)) {
      return false;
    }
  }
  const double minimum_normal_dot = std::cos(
      options.max_normal_change_degrees * kPi / 180.0);
  return PreservesFaceNormals(candidate, minimum_normal_dot, vertices, faces);
}

std::size_t Collapse(const CollapseCandidate& candidate,
                     std::vector<VertexState>* vertices,
                     std::vector<FaceState>* faces) {
  VertexState& kept = (*vertices)[candidate.first];
  VertexState& removed = (*vertices)[candidate.second];
  kept.point.xyz = candidate.position;
  const Eigen::Vector3f combined_normal =
      kept.point.normal + removed.point.normal;
  kept.point.normal = combined_normal.squaredNorm() > 0.0F
                          ? combined_normal.normalized()
                          : Eigen::Vector3f::Zero();
  for (int channel = 0; channel < 3; ++channel) {
    kept.point.color[channel] = static_cast<std::uint8_t>(
        (static_cast<unsigned int>(kept.point.color[channel]) +
         static_cast<unsigned int>(removed.point.color[channel])) /
        2U);
  }
  kept.quadric += removed.quadric;
  ++kept.version;
  removed.active = false;
  ++removed.version;

  kept.faces.insert(kept.faces.end(), removed.faces.begin(),
                    removed.faces.end());
  std::sort(kept.faces.begin(), kept.faces.end());
  kept.faces.erase(std::unique(kept.faces.begin(), kept.faces.end()),
                   kept.faces.end());
  std::size_t removed_faces = 0;
  for (const int face_id : kept.faces) {
    FaceState& face = (*faces)[face_id];
    if (!face.active) {
      continue;
    }
    for (int& vertex : face.vertices) {
      if (vertex == candidate.second) {
        vertex = candidate.first;
      }
    }
    if (face.vertices[0] == face.vertices[1] ||
        face.vertices[1] == face.vertices[2] ||
        face.vertices[2] == face.vertices[0]) {
      face.active = false;
      ++removed_faces;
    }
  }
  return removed_faces;
}

void RecomputeNormals(DenseTriangleMesh* mesh) {
  for (FusedPoint& vertex : mesh->vertices) {
    vertex.normal.setZero();
  }
  for (const Eigen::Vector3i& triangle : mesh->triangles) {
    const Eigen::Vector3d cross =
        (mesh->vertices[triangle.y()].xyz -
         mesh->vertices[triangle.x()].xyz)
            .cross(mesh->vertices[triangle.z()].xyz -
                   mesh->vertices[triangle.x()].xyz);
    for (int corner = 0; corner < 3; ++corner) {
      mesh->vertices[triangle[corner]].normal += cross.cast<float>();
    }
  }
  for (FusedPoint& vertex : mesh->vertices) {
    if (vertex.normal.squaredNorm() > 0.0F) {
      vertex.normal.normalize();
    }
  }
}

}  // namespace

bool MeshSimplificationOptions::Check() const {
  return std::isfinite(target_ratio) && target_ratio > 0.0 &&
         target_ratio <= 1.0 && std::isfinite(max_error) && max_error >= 0.0 &&
         std::isfinite(max_normal_change_degrees) &&
         max_normal_change_degrees >= 0.0 &&
         max_normal_change_degrees <= 180.0;
}

DenseTriangleMesh SimplifyMesh(const DenseTriangleMesh& mesh,
                               const MeshSimplificationOptions& options) {
  if (!options.Check()) {
    throw std::invalid_argument("Invalid mesh simplification options.");
  }
  std::vector<VertexState> vertices(mesh.vertices.size());
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    vertices[i].point = mesh.vertices[i];
  }
  std::vector<FaceState> faces;
  faces.reserve(mesh.triangles.size());
  for (const Eigen::Vector3i& triangle : mesh.triangles) {
    if (triangle.minCoeff() < 0 ||
        triangle.maxCoeff() >= static_cast<int>(vertices.size()) ||
        triangle.x() == triangle.y() || triangle.y() == triangle.z() ||
        triangle.z() == triangle.x()) {
      continue;
    }
    const Eigen::Vector3d& first = vertices[triangle.x()].point.xyz;
    const Eigen::Vector3d& second = vertices[triangle.y()].point.xyz;
    const Eigen::Vector3d& third = vertices[triangle.z()].point.xyz;
    Eigen::Vector3d normal = (second - first).cross(third - first);
    if (normal.squaredNorm() <= kMinAreaSquared) {
      continue;
    }
    normal.normalize();
    const Eigen::Vector4d plane(normal.x(), normal.y(), normal.z(),
                                -normal.dot(first));
    const Eigen::Matrix4d quadric = plane * plane.transpose();
    const int face_id = static_cast<int>(faces.size());
    faces.push_back({{triangle.x(), triangle.y(), triangle.z()}, true});
    for (int corner = 0; corner < 3; ++corner) {
      vertices[triangle[corner]].quadric += quadric;
      vertices[triangle[corner]].faces.push_back(face_id);
    }
  }

  const std::size_t initial_faces = faces.size();
  const std::size_t target_faces = std::min(
      initial_faces,
      options.target_num_triangles > 0
          ? options.target_num_triangles
          : static_cast<std::size_t>(std::floor(
                static_cast<double>(initial_faces) * options.target_ratio)));
  std::priority_queue<CollapseCandidate, std::vector<CollapseCandidate>,
                      GreaterError>
      queue;
  std::set<std::array<int, 2>> initial_edges;
  for (const FaceState& face : faces) {
    for (int edge = 0; edge < 3; ++edge) {
      initial_edges.insert(
          OrderedEdge(face.vertices[edge], face.vertices[(edge + 1) % 3]));
    }
  }
  for (const auto& edge : initial_edges) {
    queue.push(MakeCandidate(edge[0], edge[1], vertices));
  }

  std::size_t active_faces = initial_faces;
  while (active_faces > target_faces && !queue.empty()) {
    const CollapseCandidate candidate = queue.top();
    queue.pop();
    if (options.max_error > 0.0 && candidate.error > options.max_error) {
      break;
    }
    if (!CanCollapse(candidate, options, vertices, faces)) {
      continue;
    }
    active_faces -= Collapse(candidate, &vertices, &faces);
    for (const int neighbor :
         ActiveNeighbors(candidate.first, vertices, faces)) {
      queue.push(MakeCandidate(candidate.first, neighbor, vertices));
    }
  }

  DenseTriangleMesh result;
  std::vector<int> new_indices(vertices.size(), -1);
  std::set<std::array<int, 3>> unique_faces;
  for (const FaceState& face : faces) {
    if (!face.active) {
      continue;
    }
    std::array<int, 3> canonical = face.vertices;
    std::sort(canonical.begin(), canonical.end());
    if (!unique_faces.insert(canonical).second) {
      continue;
    }
    Eigen::Vector3i triangle;
    for (int corner = 0; corner < 3; ++corner) {
      const int old_index = face.vertices[corner];
      if (new_indices[old_index] < 0) {
        new_indices[old_index] = static_cast<int>(result.vertices.size());
        result.vertices.push_back(vertices[old_index].point);
      }
      triangle[corner] = new_indices[old_index];
    }
    result.triangles.push_back(triangle);
  }
  RecomputeNormals(&result);
  return result;
}

DenseTriangleMesh ReadDenseTriangleMeshPly(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read triangle mesh: " + path.string());
  }
  std::string line;
  if (!std::getline(file, line)) {
    throw std::runtime_error("Invalid PLY header: " + path.string());
  }
  if (!line.empty() && line.back() == '\r') {
    line.pop_back();
  }
  if (line != "ply") {
    throw std::runtime_error("Invalid PLY header: " + path.string());
  }
  std::size_t num_vertices = 0;
  std::size_t num_faces = 0;
  struct Property {
    bool list = false;
    std::string count_type;
    std::string value_type;
    std::string name;
  };
  std::vector<Property> vertex_properties;
  std::vector<Property> face_properties;
  std::string element;
  bool ascii = false;
  bool binary_little_endian = false;
  bool binary_big_endian = false;
  bool end_header = false;
  while (std::getline(file, line)) {
    std::istringstream stream(line);
    std::string keyword;
    stream >> keyword;
    if (keyword == "format") {
      std::string format;
      stream >> format;
      ascii = format == "ascii";
      binary_little_endian = format == "binary_little_endian";
      binary_big_endian = format == "binary_big_endian";
    } else if (keyword == "element") {
      std::size_t count = 0;
      stream >> element >> count;
      if (element == "vertex") {
        num_vertices = count;
      } else if (element == "face") {
        num_faces = count;
      }
    } else if (keyword == "property" &&
               (element == "vertex" || element == "face")) {
      Property property;
      stream >> property.value_type;
      if (property.value_type == "list") {
        property.list = true;
        stream >> property.count_type >> property.value_type >> property.name;
      } else {
        stream >> property.name;
      }
      (element == "vertex" ? vertex_properties : face_properties)
          .push_back(std::move(property));
    } else if (keyword == "end_header") {
      end_header = true;
      break;
    }
  }
  if ((!ascii && !binary_little_endian && !binary_big_endian) || !end_header) {
    throw std::runtime_error("Unsupported PLY encoding.");
  }
  const auto type_size = [](const std::string& type) -> std::size_t {
    if (type == "char" || type == "int8" || type == "uchar" ||
        type == "uint8") return 1;
    if (type == "short" || type == "int16" || type == "ushort" ||
        type == "uint16") return 2;
    if (type == "int" || type == "int32" || type == "uint" ||
        type == "uint32" || type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64" || type == "int64" ||
        type == "uint64") return 8;
    throw std::runtime_error("Unsupported PLY property type: " + type);
  };
  const bool swap_bytes = binary_big_endian;
  const auto read_binary = [&](const std::string& type) -> double {
    std::array<unsigned char, 8> bytes{};
    const std::size_t size = type_size(type);
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(size));
    if (!file) throw std::runtime_error("Unexpected end of binary PLY data.");
    if (swap_bytes) std::reverse(bytes.begin(), bytes.begin() + size);
    if (type == "char" || type == "int8") {
      std::int8_t value = 0; std::memcpy(&value, bytes.data(), 1); return value;
    }
    if (type == "uchar" || type == "uint8") return bytes[0];
    if (type == "short" || type == "int16") {
      std::int16_t value = 0; std::memcpy(&value, bytes.data(), 2); return value;
    }
    if (type == "ushort" || type == "uint16") {
      std::uint16_t value = 0; std::memcpy(&value, bytes.data(), 2); return value;
    }
    if (type == "int" || type == "int32") {
      std::int32_t value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "uint" || type == "uint32") {
      std::uint32_t value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "float" || type == "float32") {
      float value = 0; std::memcpy(&value, bytes.data(), 4); return value;
    }
    if (type == "double" || type == "float64") {
      double value = 0; std::memcpy(&value, bytes.data(), 8); return value;
    }
    if (type == "int64") {
      std::int64_t value = 0; std::memcpy(&value, bytes.data(), 8);
      return static_cast<double>(value);
    }
    std::uint64_t value = 0; std::memcpy(&value, bytes.data(), 8);
    return static_cast<double>(value);
  };
  DenseTriangleMesh mesh;
  mesh.vertices.resize(num_vertices);
  for (FusedPoint& vertex : mesh.vertices) {
    std::istringstream stream;
    if (ascii) {
      if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of PLY vertex data.");
      }
      stream.str(line);
    }
    for (const Property& property : vertex_properties) {
      if (property.list) {
        throw std::runtime_error("Unsupported vertex list property in PLY.");
      }
      double value = 0.0;
      if (ascii) {
        if (!(stream >> value)) throw std::runtime_error("Invalid PLY vertex data.");
      } else {
        value = read_binary(property.value_type);
      }
      if (property.name == "x") vertex.xyz.x() = value;
      else if (property.name == "y") vertex.xyz.y() = value;
      else if (property.name == "z") vertex.xyz.z() = value;
      else if (property.name == "nx") vertex.normal.x() = static_cast<float>(value);
      else if (property.name == "ny") vertex.normal.y() = static_cast<float>(value);
      else if (property.name == "nz") vertex.normal.z() = static_cast<float>(value);
      else if (property.name == "red")
        vertex.color.x() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
      else if (property.name == "green")
        vertex.color.y() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
      else if (property.name == "blue")
        vertex.color.z() = static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 255.0));
    }
  }
  mesh.triangles.reserve(num_faces);
  for (std::size_t i = 0; i < num_faces; ++i) {
    std::istringstream stream;
    if (ascii) {
      if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of PLY face data.");
      }
      stream.str(line);
    }
    std::vector<int> indices;
    for (const Property& property : face_properties) {
      if (!property.list) {
        double ignored = 0.0;
        if (ascii) stream >> ignored;
        else ignored = read_binary(property.value_type);
        (void)ignored;
        continue;
      }
      std::size_t count = 0;
      if (ascii) stream >> count;
      else count = static_cast<std::size_t>(read_binary(property.count_type));
      std::vector<int> values(count);
      for (int& value : values) {
        if (ascii) stream >> value;
        else value = static_cast<int>(read_binary(property.value_type));
      }
      if ((property.name == "vertex_indices" ||
           property.name == "vertex_index") && indices.empty()) {
        indices = std::move(values);
      }
    }
    if (indices.size() == 3 && (!ascii || stream)) {
      mesh.triangles.emplace_back(indices[0], indices[1], indices[2]);
    }
  }
  return mesh;
}

}  // namespace colite::mvs
