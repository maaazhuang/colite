#pragma once

#include <filesystem>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/patch_match.h"

namespace colite::mvs {

struct MeshTexturingOptions {
  // Initial atlas side. It grows by powers of two up to max_atlas_size before
  // source projections are downscaled.
  int atlas_size = 4096;
  int max_atlas_size = 32768;
  int atlas_patch_padding = 2;
  int inpaint_radius = 5;
  double texture_scale = 1.0;
  double min_cos_normal_angle = 0.1;
  int min_visible_vertices = 3;
  int view_selection_smoothing_iterations = 3;
  // Relabel smaller near-coplanar view islands when one neighboring view is
  // valid for the complete island. Set to 0 to disable.
  int view_selection_min_component_size = 64;
  int visibility_resolution = 1024;
  double occlusion_relative_tolerance = 0.01;
  // Per-face vertex-color atlas patches expose rejected triangles as visible
  // texture fragments. Keep them opt-in; rejected faces otherwise sample the
  // neutral atlas background, matching COLMAP's conservative behavior.
  bool use_vertex_color_fallback = false;
  bool apply_color_correction = true;
  double color_correction_regularization = 0.1;
  int num_threads = -1;

  bool Check() const;
};

struct MeshTextureView {
  PatchMatchView geometry;
  // CV_8UC3 BGR or CV_8UC1 image matching the geometry camera dimensions.
  cv::Mat image;
};

struct TexturedTriangleMesh {
  DenseTriangleMesh mesh;
  // Three wedge UV coordinates per triangle, in OBJ bottom-left convention.
  std::vector<Eigen::Vector2f> face_uvs;
  std::vector<int> face_view_ids;
  cv::Mat texture_atlas;
};

TexturedTriangleMesh TextureMesh(
    const DenseTriangleMesh& mesh,
    const std::vector<MeshTextureView>& views,
    const MeshTexturingOptions& options = {});

// Writes mesh.obj, mesh.mtl, and texture.png into output_directory.
void WriteTexturedTriangleMesh(
    const std::filesystem::path& output_directory,
    const TexturedTriangleMesh& textured_mesh);

}  // namespace colite::mvs
