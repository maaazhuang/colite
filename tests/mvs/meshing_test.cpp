#include "colite/mvs/meshing.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

namespace colite::mvs {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<FusedPoint> MakeSpherePoints(std::size_t count) {
  std::vector<FusedPoint> points;
  points.reserve(count);
  const double golden_angle = kPi * (3.0 - std::sqrt(5.0));
  for (std::size_t i = 0; i < count; ++i) {
    const double y =
        1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(count);
    const double radius = std::sqrt(std::max(0.0, 1.0 - y * y));
    const double angle = golden_angle * static_cast<double>(i);
    FusedPoint point;
    point.xyz =
        Eigen::Vector3d(radius * std::cos(angle), y, radius * std::sin(angle));
    point.normal = point.xyz.cast<float>().normalized();
    point.color = Eigen::Matrix<std::uint8_t, 3, 1>(128, 180, 220);
    points.push_back(point);
  }
  return points;
}

void WriteOrientedPointCloudPly(const std::filesystem::path& path,
                                const std::vector<FusedPoint>& points) {
  std::ofstream file(path);
  ASSERT_TRUE(file.is_open());
  file << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
       << "property double x\nproperty double y\nproperty double z\n"
       << "property float nx\nproperty float ny\nproperty float nz\n"
       << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
       << "end_header\n";
  for (const FusedPoint& point : points) {
    file << point.xyz.x() << ' ' << point.xyz.y() << ' ' << point.xyz.z() << ' '
         << point.normal.x() << ' ' << point.normal.y() << ' '
         << point.normal.z() << ' ' << static_cast<int>(point.color.x()) << ' '
         << static_cast<int>(point.color.y()) << ' '
         << static_cast<int>(point.color.z()) << '\n';
  }
}

DenseTriangleMesh MakeGridMesh(int side) {
  DenseTriangleMesh mesh;
  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      FusedPoint point;
      point.xyz = Eigen::Vector3d(x, y, 0.05 * std::sin(0.4 * x) *
                                           std::sin(0.4 * y));
      point.normal = Eigen::Vector3f::UnitZ();
      point.color = Eigen::Matrix<std::uint8_t, 3, 1>(80, 120, 160);
      mesh.vertices.push_back(point);
    }
  }
  for (int y = 0; y + 1 < side; ++y) {
    for (int x = 0; x + 1 < side; ++x) {
      const int a = y * side + x;
      const int b = a + 1;
      const int c = a + side;
      const int d = c + 1;
      mesh.triangles.emplace_back(a, b, d);
      mesh.triangles.emplace_back(a, d, c);
    }
  }
  return mesh;
}

TEST(PoissonMeshingOptionsTest, ValidatesParameters) {
  PoissonMeshingOptions options;
  EXPECT_TRUE(options.Check());
  options.depth = 0;
  EXPECT_FALSE(options.Check());
  options.depth = 8;
  options.trim = -1.0;
  EXPECT_FALSE(options.Check());
  options.trim = 0.0;
  options.point_weight = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(options.Check());
}

TEST(PoissonMeshingTest, RejectsMissingInput) {
  const std::filesystem::path missing = std::filesystem::temp_directory_path() /
                                        "colite_missing_poisson_input.ply";
  EXPECT_THROW(PoissonMeshing({}, missing, missing.parent_path() / "out.ply"),
               std::invalid_argument);
}

TEST(PoissonMeshingTest, ReconstructsSyntheticOrientedSphere) {
  if (!IsPoissonMeshingAvailable()) {
    GTEST_SKIP() << "Poisson meshing is disabled in this build.";
  }
  const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                     "colite_poisson_meshing_execution_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path input = root / "sphere.ply";
  const std::filesystem::path output = root / "mesh.ply";
  WriteOrientedPointCloudPly(input, MakeSpherePoints(256));

  PoissonMeshingOptions options;
  options.depth = 5;
  options.color = false;
  options.trim = 0.0;
  options.num_threads = 1;
  EXPECT_TRUE(PoissonMeshing(options, input, output));
  ASSERT_TRUE(std::filesystem::exists(output));
  EXPECT_GT(std::filesystem::file_size(output), 100U);
  const DenseTriangleMesh mesh = ReadDenseTriangleMeshPly(output);
  ASSERT_FALSE(mesh.triangles.empty());
  MeshSimplificationOptions simplification_options;
  simplification_options.target_ratio = 0.8;
  const DenseTriangleMesh simplified =
      SimplifyMesh(mesh, simplification_options);
  EXPECT_LT(simplified.triangles.size(), mesh.triangles.size());
  std::filesystem::remove_all(root);
}

TEST(DelaunayMeshingOptionsTest, ValidatesParameters) {
  DelaunayMeshingOptions options;
  EXPECT_TRUE(options.Check());
  options.max_depth_dist = 1.01;
  EXPECT_FALSE(options.Check());
  options.max_depth_dist = 0.05;
  options.visibility_sigma = 0.0;
  EXPECT_FALSE(options.Check());
  options.visibility_sigma = 3.0;
  options.max_side_length_percentile = 101.0;
  EXPECT_FALSE(options.Check());
}

TEST(DelaunayMeshingTest, RejectsMismatchedVisibility) {
  std::vector<FusedPoint> points(1);
  points.front().xyz = Eigen::Vector3d::Zero();
  if (IsDelaunayMeshingAvailable()) {
    EXPECT_THROW(DelaunayMeshing({}, points, {}, {}), std::invalid_argument);
  } else {
    EXPECT_THROW(DelaunayMeshing({}, points, {}, {}), std::runtime_error);
  }
}

TEST(DelaunayMeshingTest, ReconstructsSyntheticVisibleSphere) {
  if (!IsDelaunayMeshingAvailable()) {
    GTEST_SKIP() << "CGAL Delaunay meshing is disabled in this build.";
  }
  const std::vector<FusedPoint> points = MakeSpherePoints(48);
  const std::array<Eigen::Vector3d, 6> centers = {
      Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(-3.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 3.0, 0.0), Eigen::Vector3d(0.0, -3.0, 0.0),
      Eigen::Vector3d(0.0, 0.0, 3.0), Eigen::Vector3d(0.0, 0.0, -3.0)};
  std::vector<PatchMatchView> views;
  for (std::size_t i = 0; i < centers.size(); ++i) {
    PatchMatchView view;
    view.image_id = static_cast<ImageId>(i + 1);
    view.cam_from_world.t_cw = -centers[i];
    views.push_back(view);
  }
  std::vector<std::vector<std::uint32_t>> visibility(points.size());
  for (std::vector<std::uint32_t>& visible_views : visibility) {
    for (std::size_t i = 0; i < views.size(); ++i) {
      visible_views.push_back(static_cast<std::uint32_t>(i));
    }
  }

  DelaunayMeshingOptions options;
  options.max_proj_dist = 0.0;
  options.max_side_length_factor = 1000.0;
  const DenseTriangleMesh mesh =
      DelaunayMeshing(options, points, visibility, views);
  EXPECT_FALSE(mesh.vertices.empty());
  EXPECT_FALSE(mesh.triangles.empty());
}

TEST(MeshSimplificationOptionsTest, ValidatesParameters) {
  MeshSimplificationOptions options;
  EXPECT_TRUE(options.Check());
  options.target_ratio = 0.0;
  EXPECT_FALSE(options.Check());
  options.target_ratio = 0.5;
  options.max_error = -1.0;
  EXPECT_FALSE(options.Check());
  options.max_error = 0.0;
  options.max_normal_change_degrees = 181.0;
  EXPECT_FALSE(options.Check());
}

TEST(MeshSimplificationTest, ReducesGridWithValidTriangles) {
  const DenseTriangleMesh mesh = MakeGridMesh(12);
  MeshSimplificationOptions options;
  options.target_ratio = 0.35;
  const DenseTriangleMesh simplified = SimplifyMesh(mesh, options);
  EXPECT_LT(simplified.triangles.size(), mesh.triangles.size());
  EXPECT_LE(simplified.triangles.size(),
            static_cast<std::size_t>(mesh.triangles.size() * 0.40));
  EXPECT_FALSE(simplified.vertices.empty());
  for (const Eigen::Vector3i& triangle : simplified.triangles) {
    EXPECT_GE(triangle.minCoeff(), 0);
    EXPECT_LT(triangle.maxCoeff(),
              static_cast<int>(simplified.vertices.size()));
    EXPECT_EQ(std::set<int>({triangle.x(), triangle.y(), triangle.z()}).size(),
              3U);
    const Eigen::Vector3d cross =
        (simplified.vertices[triangle.y()].xyz -
         simplified.vertices[triangle.x()].xyz)
            .cross(simplified.vertices[triangle.z()].xyz -
                   simplified.vertices[triangle.x()].xyz);
    EXPECT_GT(cross.squaredNorm(), 0.0);
  }
}

TEST(MeshSimplificationTest, CollapsePositionsStayInsideInputBounds) {
  const DenseTriangleMesh mesh = MakeGridMesh(14);
  MeshSimplificationOptions options;
  options.target_ratio = 0.2;
  const DenseTriangleMesh simplified = SimplifyMesh(mesh, options);

  Eigen::Vector3d minimum = mesh.vertices.front().xyz;
  Eigen::Vector3d maximum = mesh.vertices.front().xyz;
  for (const FusedPoint& vertex : mesh.vertices) {
    minimum = minimum.cwiseMin(vertex.xyz);
    maximum = maximum.cwiseMax(vertex.xyz);
  }
  for (const FusedPoint& vertex : simplified.vertices) {
    EXPECT_TRUE((vertex.xyz.array() >= minimum.array() - 1e-12).all());
    EXPECT_TRUE((vertex.xyz.array() <= maximum.array() + 1e-12).all());
  }
}

TEST(MeshSimplificationTest, HonorsExplicitTargetAndErrorLimit) {
  const DenseTriangleMesh mesh = MakeGridMesh(10);
  MeshSimplificationOptions options;
  options.target_ratio = 0.9;
  options.target_num_triangles = 40;
  const DenseTriangleMesh targeted = SimplifyMesh(mesh, options);
  EXPECT_LE(targeted.triangles.size(), 42U);

  options.max_error = 1e-30;
  const DenseTriangleMesh limited = SimplifyMesh(mesh, options);
  EXPECT_GE(limited.triangles.size(), targeted.triangles.size());
}

TEST(MeshSimplificationTest, ReadsWrittenAsciiPly) {
  const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                     "colite_mesh_simplifier_ply_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path path = root / "grid.ply";
  const DenseTriangleMesh expected = MakeGridMesh(4);
  WriteDenseTriangleMeshPly(path, expected);
  const DenseTriangleMesh actual = ReadDenseTriangleMeshPly(path);
  EXPECT_EQ(actual.vertices.size(), expected.vertices.size());
  EXPECT_EQ(actual.triangles.size(), expected.triangles.size());
  EXPECT_TRUE(
      actual.vertices.front().xyz.isApprox(expected.vertices.front().xyz));
  std::filesystem::remove_all(root);
}

TEST(MeshSimplificationTest, ReadsBinaryLittleEndianPly) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     "colite_binary_mesh_test.ply";
  std::ofstream file(path, std::ios::binary);
  file << "ply\nformat binary_little_endian 1.0\n"
       << "element vertex 3\nproperty float x\nproperty float y\n"
       << "property float z\nelement face 1\n"
       << "property list uchar int vertex_indices\nend_header\n";
  for (const std::array<float, 3>& point :
       {std::array<float, 3>{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}) {
    file.write(reinterpret_cast<const char*>(point.data()), sizeof(point));
  }
  const std::uint8_t count = 3;
  const std::array<std::int32_t, 3> indices = {0, 1, 2};
  file.write(reinterpret_cast<const char*>(&count), sizeof(count));
  file.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
  file.close();
  const DenseTriangleMesh mesh = ReadDenseTriangleMeshPly(path);
  ASSERT_EQ(mesh.vertices.size(), 3U);
  ASSERT_EQ(mesh.triangles.size(), 1U);
  EXPECT_EQ(mesh.triangles.front(), Eigen::Vector3i(0, 1, 2));
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace colite::mvs
