#include "colite/mvs/mesh_texturer.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include "colite/core/camera.h"

namespace colite::mvs {
namespace {

DenseTriangleMesh MakeTexturingPlane() {
  DenseTriangleMesh mesh;
  for (const Eigen::Vector3d& position :
       {Eigen::Vector3d(-1.0, -1.0, 3.0),
        Eigen::Vector3d(-1.0, 1.0, 3.0),
        Eigen::Vector3d(1.0, -1.0, 3.0),
        Eigen::Vector3d(1.0, 1.0, 3.0)}) {
    FusedPoint vertex;
    vertex.xyz = position;
    vertex.normal = -Eigen::Vector3f::UnitZ();
    vertex.color = Eigen::Matrix<std::uint8_t, 3, 1>(127, 127, 127);
    mesh.vertices.push_back(vertex);
  }
  mesh.triangles.emplace_back(0, 1, 2);
  mesh.triangles.emplace_back(2, 1, 3);
  return mesh;
}

MeshTextureView MakeTextureView() {
  MeshTextureView view;
  view.geometry.image_id = 1;
  view.geometry.camera.id = 1;
  view.geometry.camera.model_id = CameraModelId::kPinhole;
  view.geometry.camera.width = 64;
  view.geometry.camera.height = 64;
  view.geometry.camera.params = InitializeCameraParams(
      CameraModelId::kPinhole, 48.0, 64, 64);
  view.image = cv::Mat(64, 64, CV_8UC3);
  for (int y = 0; y < view.image.rows; ++y) {
    for (int x = 0; x < view.image.cols; ++x) {
      view.image.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<std::uint8_t>(x * 4),
          static_cast<std::uint8_t>(y * 4), 220);
    }
  }
  return view;
}

DenseTriangleMesh MakeTwoViewSeamPlane() {
  DenseTriangleMesh mesh;
  for (const Eigen::Vector3d& position :
       {Eigen::Vector3d(0.0, -1.0, 3.0),
        Eigen::Vector3d(0.0, 1.0, 3.0),
        Eigen::Vector3d(-1.9, 0.0, 3.0),
        Eigen::Vector3d(1.9, 0.0, 3.0)}) {
    FusedPoint vertex;
    vertex.xyz = position;
    vertex.normal = -Eigen::Vector3f::UnitZ();
    vertex.color =
        Eigen::Matrix<std::uint8_t, 3, 1>(127, 127, 127);
    mesh.vertices.push_back(vertex);
  }
  mesh.triangles.emplace_back(0, 2, 1);
  mesh.triangles.emplace_back(0, 1, 3);
  return mesh;
}

MeshTextureView MakeShiftedGradientView(double camera_center_x,
                                        bool reverse_gradient) {
  MeshTextureView view = MakeTextureView();
  view.geometry.cam_from_world.t_cw =
      Eigen::Vector3d(-camera_center_x, 0.0, 0.0);
  for (int y = 0; y < view.image.rows; ++y) {
    const int gradient = reverse_gradient ? 255 - y * 4 : y * 4;
    for (int x = 0; x < view.image.cols; ++x) {
      view.image.at<cv::Vec3b>(y, x) =
          cv::Vec3b(static_cast<std::uint8_t>(gradient),
                    static_cast<std::uint8_t>(gradient),
                    static_cast<std::uint8_t>(gradient));
    }
  }
  return view;
}

cv::Vec3d SampleAtlas(const TexturedTriangleMesh& textured,
                      std::size_t face_id,
                      const Eigen::Vector3f& weights) {
  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
  for (int corner = 0; corner < 3; ++corner) {
    uv += weights[corner] * textured.face_uvs[face_id * 3 + corner];
  }
  cv::Mat sample;
  cv::getRectSubPix(
      textured.texture_atlas, cv::Size(1, 1),
      cv::Point2f(uv.x() * textured.texture_atlas.cols,
                  (1.0F - uv.y()) * textured.texture_atlas.rows),
      sample);
  const cv::Vec3b color = sample.at<cv::Vec3b>(0, 0);
  return cv::Vec3d(color[0], color[1], color[2]);
}

TEST(MeshTexturingOptionsTest, ValidatesParameters) {
  MeshTexturingOptions options;
  EXPECT_TRUE(options.Check());
  options.atlas_size = 8;
  EXPECT_FALSE(options.Check());
  options.atlas_size = 64;
  options.texture_scale = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(options.Check());
  options.texture_scale = 1.0;
  options.min_cos_normal_angle = 1.1;
  EXPECT_FALSE(options.Check());
  options.min_cos_normal_angle = 0.1;
  options.view_selection_min_component_size = -1;
  EXPECT_FALSE(options.Check());
}

TEST(MeshTexturingTest, SelectsViewAndBakesAtlas) {
  MeshTexturingOptions options;
  options.atlas_size = 64;
  options.visibility_resolution = 64;
  const TexturedTriangleMesh textured =
      TextureMesh(MakeTexturingPlane(), {MakeTextureView()}, options);
  ASSERT_EQ(textured.face_view_ids.size(), 2U);
  EXPECT_EQ(textured.face_view_ids[0], 0);
  EXPECT_EQ(textured.face_view_ids[1], 0);
  ASSERT_EQ(textured.face_uvs.size(), 6U);
  for (const Eigen::Vector2f& uv : textured.face_uvs) {
    EXPECT_GE(uv.minCoeff(), 0.0F);
    EXPECT_LE(uv.maxCoeff(), 1.0F);
  }
  EXPECT_GT(textured.texture_atlas.rows, 0);
  EXPECT_LE(textured.texture_atlas.rows, 64);
  EXPECT_EQ(textured.texture_atlas.cols, 64);
  cv::Mat difference;
  cv::absdiff(textured.texture_atlas, cv::Scalar(127, 127, 127), difference);
  EXPECT_GT(cv::countNonZero(difference.reshape(1)), 0);
}

TEST(MeshTexturingTest, RejectsMismatchedImageDimensions) {
  MeshTextureView view = MakeTextureView();
  view.image = cv::Mat(32, 64, CV_8UC3);
  EXPECT_THROW(TextureMesh(MakeTexturingPlane(), {view}),
               std::invalid_argument);
}

TEST(MeshTexturingTest, RejectsOccludedBackPlane) {
  DenseTriangleMesh mesh = MakeTexturingPlane();
  DenseTriangleMesh back = MakeTexturingPlane();
  for (FusedPoint& vertex : back.vertices) vertex.xyz.z() = 4.0;
  const int offset = static_cast<int>(mesh.vertices.size());
  mesh.vertices.insert(mesh.vertices.end(), back.vertices.begin(),
                       back.vertices.end());
  for (Eigen::Vector3i triangle : back.triangles) {
    mesh.triangles.push_back(triangle.array() + offset);
  }
  MeshTexturingOptions options;
  options.atlas_size = 64;
  options.visibility_resolution = 64;
  const TexturedTriangleMesh textured =
      TextureMesh(mesh, {MakeTextureView()}, options);
  EXPECT_EQ(textured.face_view_ids[0], 0);
  EXPECT_EQ(textured.face_view_ids[1], 0);
  EXPECT_EQ(textured.face_view_ids[2], -1);
  EXPECT_EQ(textured.face_view_ids[3], -1);
}

TEST(MeshTexturingTest, DoesNotRecoverUnsupportedNearCoplanarSurface) {
  DenseTriangleMesh mesh = MakeTexturingPlane();
  DenseTriangleMesh near_plane = MakeTexturingPlane();
  for (FusedPoint& vertex : near_plane.vertices) vertex.xyz.z() = 3.03;
  const int offset = static_cast<int>(mesh.vertices.size());
  mesh.vertices.insert(mesh.vertices.end(), near_plane.vertices.begin(),
                       near_plane.vertices.end());
  for (Eigen::Vector3i triangle : near_plane.triangles) {
    mesh.triangles.push_back(triangle.array() + offset);
  }

  MeshTexturingOptions options;
  options.atlas_size = 64;
  options.visibility_resolution = 64;
  options.occlusion_relative_tolerance = 0.01;
  const TexturedTriangleMesh tolerant =
      TextureMesh(mesh, {MakeTextureView()}, options);
  EXPECT_EQ(tolerant.face_view_ids[2], -1);
  EXPECT_EQ(tolerant.face_view_ids[3], -1);

  options.occlusion_relative_tolerance = 0.0;
  const TexturedTriangleMesh strict =
      TextureMesh(mesh, {MakeTextureView()}, options);
  EXPECT_EQ(strict.face_view_ids[2], -1);
  EXPECT_EQ(strict.face_view_ids[3], -1);
}

TEST(MeshTexturingTest, LeavesRejectedFacesOnNeutralAtlasByDefault) {
  const TexturedTriangleMesh textured =
      TextureMesh(MakeTexturingPlane(), {});
  ASSERT_EQ(textured.face_view_ids.size(), 2U);
  const Eigen::Vector2f fallback_uv = textured.face_uvs.front();
  EXPECT_GE(fallback_uv.minCoeff(), 0.0F);
  EXPECT_LE(fallback_uv.maxCoeff(), 1.0F);
  for (std::size_t face_id = 0; face_id < 2; ++face_id) {
    EXPECT_EQ(textured.face_view_ids[face_id], -1);
    EXPECT_EQ(textured.face_uvs[face_id * 3], fallback_uv);
    EXPECT_EQ(textured.face_uvs[face_id * 3 + 1], fallback_uv);
    EXPECT_EQ(textured.face_uvs[face_id * 3 + 2], fallback_uv);
  }
}

TEST(MeshTexturingTest, BakesVertexColorsForFacesWithoutVisibleView) {
  DenseTriangleMesh mesh = MakeTexturingPlane();
  mesh.vertices[0].color =
      Eigen::Matrix<std::uint8_t, 3, 1>(10, 20, 200);
  mesh.vertices[1].color =
      Eigen::Matrix<std::uint8_t, 3, 1>(20, 200, 30);
  mesh.vertices[2].color =
      Eigen::Matrix<std::uint8_t, 3, 1>(200, 30, 40);
  MeshTexturingOptions options;
  options.atlas_size = 64;
  options.use_vertex_color_fallback = true;
  const TexturedTriangleMesh textured = TextureMesh(mesh, {}, options);
  ASSERT_EQ(textured.face_view_ids.size(), 2U);
  EXPECT_EQ(textured.face_view_ids[0], -1);
  EXPECT_EQ(textured.face_view_ids[1], -1);
  for (std::size_t face_id = 0; face_id < mesh.triangles.size(); ++face_id) {
    const Eigen::Vector2f first =
        textured.face_uvs[face_id * 3 + 1] -
        textured.face_uvs[face_id * 3];
    const Eigen::Vector2f second =
        textured.face_uvs[face_id * 3 + 2] -
        textured.face_uvs[face_id * 3];
    EXPECT_GT(std::abs(first.x() * second.y() -
                       first.y() * second.x()),
              1e-8F);
  }
  cv::Mat difference;
  cv::absdiff(textured.texture_atlas, cv::Scalar(127, 127, 127), difference);
  EXPECT_GT(cv::countNonZero(difference.reshape(1)), 0);
}

TEST(MeshTexturingTest, LocalSeamLevelingReducesSpatialColorDiscontinuity) {
  MeshTexturingOptions options;
  options.atlas_size = 128;
  options.max_atlas_size = 128;
  options.visibility_resolution = 64;
  options.view_selection_smoothing_iterations = 0;
  const DenseTriangleMesh mesh = MakeTwoViewSeamPlane();
  const std::vector<MeshTextureView> views = {
      MakeShiftedGradientView(-1.0, false),
      MakeShiftedGradientView(1.0, true)};

  options.apply_color_correction = false;
  const TexturedTriangleMesh uncorrected =
      TextureMesh(mesh, views, options);
  ASSERT_EQ(uncorrected.face_view_ids[0], 0);
  ASSERT_EQ(uncorrected.face_view_ids[1], 1);

  options.apply_color_correction = true;
  const TexturedTriangleMesh corrected = TextureMesh(mesh, views, options);
  const std::array<std::pair<Eigen::Vector3f, Eigen::Vector3f>, 2>
      seam_samples = {
          std::make_pair(Eigen::Vector3f(0.96F, 0.02F, 0.02F),
                         Eigen::Vector3f(0.96F, 0.02F, 0.02F)),
          std::make_pair(Eigen::Vector3f(0.02F, 0.02F, 0.96F),
                         Eigen::Vector3f(0.02F, 0.96F, 0.02F))};
  double uncorrected_error = 0.0;
  double corrected_error = 0.0;
  for (const auto& [left_weights, right_weights] : seam_samples) {
    uncorrected_error +=
        cv::norm(SampleAtlas(uncorrected, 0, left_weights) -
                 SampleAtlas(uncorrected, 1, right_weights));
    corrected_error +=
        cv::norm(SampleAtlas(corrected, 0, left_weights) -
                 SampleAtlas(corrected, 1, right_weights));
  }
  EXPECT_LT(corrected_error, uncorrected_error * 0.35);
}

TEST(MeshTexturingTest, WritesObjMtlAndPng) {
  MeshTexturingOptions options;
  options.atlas_size = 64;
  options.visibility_resolution = 64;
  const TexturedTriangleMesh textured =
      TextureMesh(MakeTexturingPlane(), {MakeTextureView()}, options);
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "colite_mesh_texturer_test";
  std::filesystem::remove_all(output);
  WriteTexturedTriangleMesh(output, textured);
  EXPECT_TRUE(std::filesystem::exists(output / "mesh.obj"));
  EXPECT_TRUE(std::filesystem::exists(output / "mesh.mtl"));
  EXPECT_TRUE(std::filesystem::exists(output / "texture.png"));
  std::ifstream object(output / "mesh.obj");
  const std::string contents((std::istreambuf_iterator<char>(object)),
                             std::istreambuf_iterator<char>());
  EXPECT_NE(contents.find("mtllib mesh.mtl"), std::string::npos);
  EXPECT_NE(contents.find("vt "), std::string::npos);
  EXPECT_NE(contents.find("usemtl colite_texture"), std::string::npos);
  std::filesystem::remove_all(output);
}

}  // namespace
}  // namespace colite::mvs
