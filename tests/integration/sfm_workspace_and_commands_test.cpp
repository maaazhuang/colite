#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "colite/controllers/mapper.h"
#include "colite/controllers/recon_sfm_controller.h"
#include "colite/controllers/sequential_matcher.h"
#include "colite/core/camera.h"
#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"
#include "colite/exe/command.h"
#include "colite/exe/compare_colmap.h"
#include "colite/exe/feature_extractor.h"
#include "colite/exe/mapper.h"
#include "colite/exe/sequential_matcher.h"
#include "colite/io/camera_prior.h"
#include "colite/io/sfm_workspace.h"

namespace colite {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

FeatureSet MakeTinyFeatures() {
  FeatureSet features;
  features.keypoints = {{10.0F, 20.0F, 1.5F, 0.1F, 2.0F},
                        {30.0F, 40.0F, 2.5F, 0.2F, 3.0F}};
  features.descriptors.resize(2, 128);
  features.descriptors.setZero();
  features.descriptors(0, 0) = 1.0F;
  features.descriptors(1, 127) = 2.0F;
  return features;
}

Pose3 MappingCameraAt(const double x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-x, 0.0, 0.0);
  return pose;
}

std::vector<Eigen::Vector3d> MappingScenePoints(const double x_offset) {
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 6; ++column) {
      points.emplace_back(x_offset - 1.0 + 0.4 * column,
                          -0.5 + 0.3 * row,
                          4.0 + 0.07 * column + 0.09 * row);
    }
  }
  return points;
}

FeatureSet ProjectMappingFeatures(
    const Camera& camera,
    const Pose3& pose,
    const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const std::optional<Eigen::Vector2d> pixel =
        ProjectWorldPoint(camera, pose, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
         1.0F, 0.0F});
  }
  return features;
}

std::vector<FeatureMatch> IdentityFeatureMatches(const std::size_t count) {
  std::vector<FeatureMatch> matches;
  for (std::size_t idx = 0; idx < count; ++idx) {
    matches.push_back({static_cast<Point2DIdx>(idx),
                       static_cast<Point2DIdx>(idx)});
  }
  return matches;
}

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  file << contents;
}

void WriteTinyColmapModel(const std::filesystem::path& path,
                          const bool candidate) {
  WriteTextFile(path / "cameras.txt",
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                "1 PINHOLE 640 480 500 500 320 240\n");
  if (!candidate) {
    WriteTextFile(path / "images.txt",
                  "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                  "# POINTS2D[] as X Y POINT3D_ID\n"
                  "1 1 0 0 0 0 0 0 1 image1.jpg\n"
                  "10 20 1 30 40 2\n"
                  "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                  "15 25 1\n");
    WriteTextFile(path / "points3D.txt",
                  "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                  "1 0 0 5 255 255 255 0.5 1 0 2 0\n"
                  "2 1 0 5 255 255 255 0.7 1 1\n");
    return;
  }

  WriteTextFile(path / "images.txt",
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "# POINTS2D[] as X Y POINT3D_ID\n"
                "1 1 0 0 0 0.1 0 0 1 image1.jpg\n"
                "10 20 1 30 40 3\n"
                "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                "15 25 1 35 45 3\n"
                "3 1 0 0 0 2 0 0 1 image3.jpg\n"
                "\n");
  WriteTextFile(path / "points3D.txt",
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                "1 0.1 0 5 255 255 255 0.6 1 0 2 0\n"
                "3 2 0 5 255 255 255 0.9 1 1 2 1\n");
}

TEST(SfmWorkspaceTest, RoundTripsAllCameraModelsThroughColmapTextAndBinary) {
  const std::vector<CameraModelId> models = {
      CameraModelId::kSimplePinhole,
      CameraModelId::kPinhole,
      CameraModelId::kSimpleRadial,
      CameraModelId::kRadial,
      CameraModelId::kOpenCV,
      CameraModelId::kOpenCVFisheye,
      CameraModelId::kFullOpenCV,
      CameraModelId::kFOV,
      CameraModelId::kSimpleRadialFisheye,
      CameraModelId::kRadialFisheye,
      CameraModelId::kThinPrismFisheye,
      CameraModelId::kRadTanThinPrismFisheye};
  std::vector<Camera> cameras;
  Reconstruction reconstruction;
  CameraId camera_id = 1;
  for (const CameraModelId model : models) {
    Camera camera;
    camera.id = camera_id++;
    camera.model_id = model;
    camera.width = 1000;
    camera.height = 800;
    camera.params = InitializeCameraParams(model, 700.0, 1000, 800);
    for (const std::size_t idx :
         GetCameraModelSpec(model).distortion_idxs) {
      camera.params[idx] = model == CameraModelId::kFOV ? 0.2 : 0.001;
    }
    cameras.push_back(camera);
    reconstruction.AddCamera(camera);
  }

  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_all_camera_model_io_test";
  const std::filesystem::path workspace = root / "workspace";
  const std::filesystem::path colmap = root / "colmap";
  std::filesystem::remove_all(root);
  WriteWorkspaceCameras(workspace, cameras);
  const std::vector<Camera> workspace_cameras =
      ReadWorkspaceCameras(workspace);
  ASSERT_EQ(workspace_cameras.size(), cameras.size());
  for (std::size_t idx = 0; idx < cameras.size(); ++idx) {
    EXPECT_EQ(workspace_cameras[idx].model_id, cameras[idx].model_id);
    EXPECT_EQ(workspace_cameras[idx].params, cameras[idx].params);
  }

  WriteColmapReconstruction(colmap, reconstruction,
                            ColmapModelFormat::kTextAndBinary);
  const Reconstruction loaded = ReadColmapTextReconstruction(colmap);
  const Reconstruction loaded_binary =
      ReadColmapBinaryReconstruction(colmap);
  EXPECT_EQ(DetectColmapModelFormat(colmap), ColmapModelFormat::kBinary);
  ASSERT_EQ(loaded.Cameras().size(), cameras.size());
  ASSERT_EQ(loaded_binary.Cameras().size(), cameras.size());
  for (const Camera& camera : cameras) {
    ASSERT_TRUE(loaded.ExistsCamera(camera.id));
    EXPECT_EQ(loaded.GetCamera(camera.id).model_id, camera.model_id);
    EXPECT_EQ(loaded.GetCamera(camera.id).params, camera.params);
    ASSERT_TRUE(loaded_binary.ExistsCamera(camera.id));
    EXPECT_EQ(loaded_binary.GetCamera(camera.id).model_id, camera.model_id);
    EXPECT_EQ(loaded_binary.GetCamera(camera.id).params, camera.params);
  }
}

TEST(SfmWorkspaceTest, RoundTripsBinaryPosesColorsAndTracks) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 3;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 320.0, 240.0, 0.01};
  reconstruction.AddCamera(camera);

  for (ImageId image_id : {7U, 9U}) {
    Image image;
    image.id = image_id;
    image.camera_id = camera.id;
    image.name = "folder/image " + std::to_string(image_id) + ".jpg";
    image.registered = true;
    image.cam_from_world = MappingCameraAt(static_cast<double>(image_id));
    image.points2d.push_back(
        {Eigen::Vector2d(10.5 + image_id, 20.25), 0, 42});
    reconstruction.AddImage(std::move(image));
  }
  Point3D point;
  point.id = 42;
  point.xyz = Eigen::Vector3d(1.25, -2.5, 7.75);
  point.color << 17, 101, 249;
  point.reprojection_error = 0.125;
  point.track.AddElement(7, 0);
  point.track.AddElement(9, 0);
  reconstruction.AddPoint3D(std::move(point));

  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_binary_model_test";
  std::filesystem::remove_all(root);
  WriteColmapReconstruction(root, reconstruction,
                            ColmapModelFormat::kTextAndBinary);
  const Reconstruction loaded = ReadColmapReconstruction(root);
  const Reconstruction loaded_text =
      ReadColmapReconstruction(root, ColmapModelFormat::kText);

  ASSERT_EQ(loaded.RegisteredImageIds(), (std::set<ImageId>{7U, 9U}));
  EXPECT_EQ(loaded.GetImage(7).name, "folder/image 7.jpg");
  EXPECT_EQ(loaded_text.GetImage(7).name, "folder/image 7.jpg");
  ASSERT_TRUE(loaded.GetImage(9).cam_from_world.has_value());
  EXPECT_DOUBLE_EQ(loaded.GetImage(9).cam_from_world->t_cw.x(), -9.0);
  ASSERT_TRUE(loaded.ExistsPoint3D(42));
  const Point3D& loaded_point = loaded.GetPoint3D(42);
  EXPECT_EQ(loaded_point.color.x(), 17);
  EXPECT_EQ(loaded_point.color.y(), 101);
  EXPECT_EQ(loaded_point.color.z(), 249);
  EXPECT_EQ(loaded_point.track.elements.size(), 2U);
  EXPECT_EQ(loaded.GetImage(7).points2d[0].point3d_id, 42U);
}

TEST(SfmWorkspaceTest, RoundTripsManifestFeaturesAndMatches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_workspace_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  camera.has_prior_focal_length = true;

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.file_path = "/tmp/image1.jpg";

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";
  image2.prior_position = Eigen::Vector3d(1.0, 2.0, 3.0);

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  MatchGraph graph;
  SetMatches(&graph, 1, 2, {{0, 1}, {1, 0}});
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1.t_cw = Eigen::Vector3d(1.0, 0.0, 0.0);
  geometry.median_triangulation_angle = 3.5;
  geometry.inlier_matches = {{0, 1}};
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);

  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  const std::vector<Image> images = ReadWorkspaceImages(workspace);
  const FeatureSet loaded_features = ReadWorkspaceFeatureSet(workspace, 1);
  MatchGraph loaded_graph;
  ReadWorkspaceMatches(workspace, &loaded_graph);
  ReadWorkspaceVerifiedGeometries(workspace, &loaded_graph);

  ASSERT_EQ(cameras.size(), 1U);
  EXPECT_EQ(cameras[0].width, 640U);
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(images[1].name, "image2.jpg");
  ASSERT_TRUE(images[1].prior_position.has_value());
  EXPECT_NEAR(images[1].prior_position->x(), 1.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->y(), 2.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->z(), 3.0, 1e-12);
  ASSERT_EQ(loaded_features.keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(loaded_features.keypoints[1].response, 3.0F);
  ASSERT_EQ(loaded_features.descriptors.rows(), 2);
  EXPECT_FLOAT_EQ(loaded_features.descriptors(1, 127), 2.0F);

  const std::vector<FeatureMatch>* matches =
      FindMatches(loaded_graph, 1, 2);
  ASSERT_NE(matches, nullptr);
  EXPECT_EQ(matches->size(), 2U);
  const TwoViewGeometry* loaded_geometry =
      FindVerifiedGeometry(loaded_graph, 1, 2);
  ASSERT_NE(loaded_geometry, nullptr);
  EXPECT_EQ(loaded_geometry->type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_NEAR(loaded_geometry->median_triangulation_angle, 3.5, 1e-12);
  EXPECT_EQ(loaded_geometry->inlier_matches.size(), 1U);

  EXPECT_TRUE(std::filesystem::exists(workspace / "features" / "1.bin"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "raw_matches.tsv"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "verified_matches.tsv"));
}

TEST(SfmWorkspaceTest, RoundTripsStageCacheMetadata) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_cache_test";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(workspace / "features");
  {
    std::ofstream file(workspace / "features" / "1.bin");
    file << "feature bytes";
  }

  const std::string hash = StableConfigHash(
      {{"stage", "features"}, {"backend", "vlsift"}, {"max_num_features", "64"}});
  WriteWorkspaceCacheMetadata(workspace, "features", hash,
                              {{"images", "1"},
                               {"feature_extraction_seconds", "0.25"}},
                              {"features/1.bin"});

  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace, "features");

  EXPECT_EQ(metadata.stage, "features");
  EXPECT_EQ(metadata.config_hash, hash);
  EXPECT_EQ(metadata.values.at("images"), "1");
  EXPECT_TRUE(WorkspaceCacheIsValid(workspace, "features", hash));
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash + "x"));

  std::filesystem::remove(workspace / "features" / "1.bin");
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash));
}

TEST(SfmWorkspaceTest, ReadsRegisteredImageWithEmptyPointsLine) {
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() /
      "colite_empty_image_points_model_test";
  std::filesystem::remove_all(model_path);
  WriteTinyColmapModel(model_path, true);

  const Reconstruction model = ReadColmapTextReconstruction(model_path);

  ASSERT_EQ(model.RegisteredImageIds().size(), 3U);
  ASSERT_TRUE(model.ExistsImage(3));
  EXPECT_TRUE(model.GetImage(3).points2d.empty());
  EXPECT_EQ(model.Points3D().size(), 2U);
}

TEST(SfmWorkspaceTest, RemapsCheckpointImagesByName) {
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() /
      "colite_checkpoint_name_remap_test";
  std::filesystem::remove_all(model_path);
  WriteTinyColmapModel(model_path, true);

  Reconstruction schema;
  Camera camera;
  camera.id = 7;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {450.0, 450.0, 320.0, 240.0};
  schema.AddCamera(camera);
  for (ImageId image_id : {11U, 12U, 13U, 14U}) {
    Image image;
    image.id = image_id;
    image.camera_id = 7;
    image.name = image_id == 11U   ? "new_image.jpg"
                 : image_id == 12U ? "image3.jpg"
                 : image_id == 13U ? "image1.jpg"
                                   : "image2.jpg";
    image.file_path = "/current/" + image.name;
    schema.AddImage(std::move(image));
  }

  const Reconstruction resumed = ApplyColmapTextModel(schema, model_path);

  EXPECT_EQ(resumed.RegisteredImageIds(),
            (std::set<ImageId>{12U, 13U, 14U}));
  EXPECT_FALSE(resumed.IsImageRegistered(11));
  EXPECT_EQ(resumed.GetImage(13).file_path, "/current/image1.jpg");
  EXPECT_DOUBLE_EQ(resumed.GetCamera(7).params[0], 500.0);
  ASSERT_TRUE(resumed.ExistsPoint3D(1));
  const Track& track = resumed.GetPoint3D(1).track;
  ASSERT_EQ(track.elements.size(), 2U);
  EXPECT_EQ(track.elements[0].image_id, 13U);
  EXPECT_EQ(track.elements[1].image_id, 14U);
}

TEST(SfmWorkspaceTest, MapperWritesAndResumesMultipleSparseModels) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_mapper_multimodel_resume_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  WriteWorkspaceCameras(workspace, {camera});

  std::vector<Image> images;
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    Image image;
    image.id = image_id;
    image.camera_id = 1;
    image.name = "image_" + std::to_string(image_id) + ".jpg";
    images.push_back(std::move(image));
  }
  WriteWorkspaceImages(workspace, images);

  const std::vector<Eigen::Vector3d> points1 = MappingScenePoints(0.0);
  const std::vector<Eigen::Vector3d> points2 = MappingScenePoints(0.4);
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const bool second_view = image_id == 2 || image_id == 4;
    const std::vector<Eigen::Vector3d>& points =
        image_id <= 2 ? points1 : points2;
    const FeatureSet features = ProjectMappingFeatures(
        camera, MappingCameraAt(second_view ? 1.0 : 0.0), points);
    SetFeatures(&graph, image_id, features);
    WriteWorkspaceFeatureSet(workspace, image_id, features);
  }
  for (const ImagePair pair :
       {ImagePair::Ordered(1, 2), ImagePair::Ordered(3, 4)}) {
    const std::vector<FeatureMatch> matches =
        IdentityFeatureMatches(points1.size());
    SetMatches(&graph, pair.image_id1, pair.image_id2, matches);
    TwoViewGeometry geometry;
    geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    geometry.cam2_from_cam1 = MappingCameraAt(1.0);
    geometry.median_triangulation_angle = 2.0;
    geometry.inlier_matches = matches;
    SetVerifiedGeometry(&graph, pair.image_id1, pair.image_id2, geometry);
  }
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);

  controllers::MapperControllerOptions options;
  options.workspace_path = workspace;
  options.merge_models = false;
  const controllers::MapperControllerSummary first =
      controllers::MapperController(options).Run();

  ASSERT_TRUE(first.success);
  EXPECT_EQ(first.num_models, 2U);
  EXPECT_EQ(first.num_unregistered_images, 0U);
  EXPECT_TRUE(std::filesystem::exists(workspace / "sparse" / "models.txt"));
  EXPECT_TRUE(std::filesystem::exists(
      workspace / "sparse" / "models" / "0" / "points3D.ply"));
  EXPECT_TRUE(std::filesystem::exists(
      workspace / "sparse" / "models" / "1" / "points3D.ply"));

  options.input_model_path = workspace / "sparse" / "models" / "0";
  const controllers::MapperControllerSummary resumed =
      controllers::MapperController(options).Run();
  EXPECT_TRUE(resumed.success);
  EXPECT_FALSE(resumed.cache_hit);
  EXPECT_EQ(resumed.num_models, 2U);
  EXPECT_EQ(resumed.num_unregistered_images, 0U);
}

TEST(SfmWorkspaceTest, SequentialMatcherReusesRawAndGeometryCaches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_match_cache_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.file_path = "/tmp/image1.jpg";

  Image image2 = image1;
  image2.id = 2;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  controllers::SequentialMatcherControllerOptions options;
  options.workspace_path = workspace;
  options.matching_strategy = MatchingStrategy::kExhaustive;
  options.matcher_backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;

  const controllers::SequentialMatcherControllerSummary first =
      controllers::SequentialMatcherController(options).Run();
  const controllers::SequentialMatcherControllerSummary second =
      controllers::SequentialMatcherController(options).Run();

  EXPECT_TRUE(first.success);
  EXPECT_FALSE(first.raw_match_cache_hit);
  EXPECT_FALSE(first.geometry_cache_hit);
  EXPECT_TRUE(second.success);
  EXPECT_TRUE(second.raw_match_cache_hit);
  EXPECT_TRUE(second.geometry_cache_hit);
  EXPECT_EQ(second.num_pairs, first.num_pairs);
  EXPECT_EQ(second.num_verified_inliers, first.num_verified_inliers);
}

TEST(SfmCommandExeTest, PrintsHelpForStageCommands) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunFeatureExtractor({"feature_extractor", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunSequentialMatcher({"sequential_matcher", "--help"}, out,
                                      err),
            0);
  EXPECT_NE(out.str().find("--overlap"), std::string::npos);
  EXPECT_NE(out.str().find("--matching_type"), std::string::npos);
  EXPECT_NE(out.str().find("--guided_matching"), std::string::npos);
  EXPECT_NE(out.str().find("--guided_match_ratio"), std::string::npos);

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunMapper({"mapper", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
  EXPECT_NE(out.str().find("--input_model_path"), std::string::npos);
  EXPECT_NE(out.str().find("--multiple_models"), std::string::npos);
  EXPECT_NE(out.str().find("--merge_models"), std::string::npos);
}

TEST(ColiteCommandExeTest, PrintsTopLevelHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("feature_extractor"), std::string::npos);
  EXPECT_NE(out.str().find("exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("vocab_tree_builder"), std::string::npos);
  EXPECT_NE(out.str().find("compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("recon_auto"), std::string::npos);
  EXPECT_NE(out.str().find("image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("delaunay_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_texturer"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesExhaustiveSpatialAndVocabTreeMatcherHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "exhaustive_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "spatial_matcher", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite spatial_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--spatial_max_distance"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--vocab_tree_num_images"), std::string::npos);
  EXPECT_NE(out.str().find("--vocab_tree_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_builder", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("--output_path"), std::string::npos);
  EXPECT_NE(out.str().find("--branching"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesSubcommandHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "recon_sfm", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("colite recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("--image_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesStandaloneMvsToolHelp) {
  for (const std::string& command : {"poisson_mesher", "delaunay_mesher",
                                     "mesh_simplifier", "mesh_texturer"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("--input_path"), std::string::npos) << command;
    EXPECT_NE(out.str().find("--output_path"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, DispatchesStagedPipelineHelp) {
  for (const std::string& command : {"recon_auto", "image_undistorter",
                                     "patch_match_stereo", "stereo_fusion"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("Usage:"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, ReconAutoDryRunPrintsAllStageBoundaries) {
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite({"colite", "recon_auto", "--image_path", "images",
                            "--output_path", "result", "--dry_run"},
                           out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_texturer"), std::string::npos);
}

TEST(ColiteCommandExeTest, ReconAutoAcceptsSfmCameraMode) {
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(
      exe::RunColite({"colite", "recon_auto", "--image_path", "images",
                      "--output_path", "result", "--stop_after", "sfm",
                      "--sfm.camera_mode", "shared", "--dry_run"},
                     out, err),
      0)
      << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
}

TEST(ColiteCommandExeTest, SurfaceCommandsSupportDryRunStageContract) {
  struct Case {
    std::string command;
    std::vector<std::string> arguments;
    std::string stage;
  };
  const std::vector<Case> cases = {
      {"poisson_mesher",
       {"--input_path", "fused.ply", "--output_path", "mesh.ply"},
       "poisson_mesher"},
      {"delaunay_mesher",
       {"--input_path", "dense", "--output_path", "mesh.ply"},
       "delaunay_mesher"},
      {"mesh_simplifier",
       {"--input_path", "mesh.ply", "--output_path", "simple.ply"},
       "mesh_simplifier"},
      {"mesh_texturer",
       {"--input_path", "mesh.ply", "--output_path", "textured",
        "--model_path", "sparse", "--image_path", "images",
        "--MeshTextureMapping.visibility_resolution", "512",
        "--MeshTextureMapping.occlusion_relative_tolerance", "0.02"},
       "mesh_texturer"}};
  for (const Case& test_case : cases) {
    std::vector<std::string> command{"colite", test_case.command};
    command.insert(command.end(), test_case.arguments.begin(),
                   test_case.arguments.end());
    command.push_back("--dry_run");
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite(command, out, err), 0)
        << test_case.command << ": " << err.str();
    EXPECT_NE(out.str().find("PLAN " + test_case.stage), std::string::npos)
        << test_case.command;
  }
}

TEST(ColiteCommandExeTest, MeshSimplifierWritesAndResumesStageManifest) {
  const auto root = std::filesystem::temp_directory_path() /
                    "colite_mesh_simplifier_stage_test";
  std::filesystem::remove_all(root);
  const auto input = root / "input.ply";
  const auto output = root / "output.ply";
  WriteTextFile(input,
                "ply\nformat ascii 1.0\n"
                "element vertex 4\nproperty float x\nproperty float y\n"
                "property float z\nelement face 2\n"
                "property list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
                "3 0 1 2\n3 0 2 3\n");
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_ratio", "0.75"},
                out, err),
            0)
      << err.str();
  EXPECT_TRUE(std::filesystem::exists(output));
  EXPECT_TRUE(std::filesystem::exists(
      root / ".colite" / "stages" / "mesh_simplifier.json"));

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_ratio", "0.75", "--resume"},
                out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("SKIP mesh_simplifier"), std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ColiteCommandExeTest, ReconAutoCanSelectDelaunayOrStopAtFusion) {
  std::ostringstream delaunay_out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--mesher", "delaunay",
                 "--dry_run"},
                delaunay_out, err),
            0)
      << err.str();
  EXPECT_NE(delaunay_out.str().find("PLAN delaunay_mesher"),
            std::string::npos);

  std::ostringstream fusion_out;
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--stop_after", "fusion",
                 "--dry_run"},
                fusion_out, err),
            0)
      << err.str();
  EXPECT_EQ(fusion_out.str().find("PLAN poisson_mesher"), std::string::npos);
}

TEST(ColiteCompareExeTest, WritesFixedColmapDifferenceReports) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_compare_colmap_test";
  const std::filesystem::path reference = root / "reference";
  const std::filesystem::path candidate = root / "candidate";
  const std::filesystem::path output = root / "report";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(reference, false);
  WriteTinyColmapModel(candidate, true);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunCompareColmap({"compare_colmap",
                                   "--reference_path",
                                   reference.string(),
                                   "--candidate_path",
                                   candidate.string(),
                                   "--output_path",
                                   output.string()},
                                  out,
                                  err),
            0)
      << err.str();

  EXPECT_NE(out.str().find("Common images: 2"), std::string::npos);
  EXPECT_NE(out.str().find("Common points by id: 1"), std::string::npos);
  const std::string image_diff = ReadTextFile(output / "image_diff.tsv");
  const std::string pair_diff = ReadTextFile(output / "pair_diff.tsv");
  const std::string point_diff = ReadTextFile(output / "point_diff.tsv");
  const std::string pose_diff = ReadTextFile(output / "pose_diff.tsv");
  const std::string summary = ReadTextFile(output / "summary.txt");

  EXPECT_NE(image_diff.find("image_name\tstatus\tref_image_id"),
            std::string::npos);
  EXPECT_NE(image_diff.find("image3.jpg\tcandidate_only"), std::string::npos);
  EXPECT_NE(pair_diff.find("image1.jpg\timage2.jpg\t1\t2\t1"),
            std::string::npos);
  EXPECT_NE(point_diff.find("1\tcommon\t0.1"), std::string::npos);
  EXPECT_NE(point_diff.find("2\treference_only"), std::string::npos);
  EXPECT_NE(point_diff.find("3\tcandidate_only"), std::string::npos);
  EXPECT_NE(pose_diff.find("image1.jpg\t1\t1\t0\t0.1\t0.1"),
            std::string::npos);
  EXPECT_NE(summary.find("common_pairs 1"), std::string::npos);
  EXPECT_NE(summary.find("candidate_only_images 1"), std::string::npos);
}

TEST(ColiteCommandExeTest, DispatchesCompareColmapHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "compare_colmap", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("--reference_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, RejectsUnknownSubcommand) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "does_not_exist"}, out, err), 1);
  EXPECT_TRUE(out.str().empty());
  EXPECT_NE(err.str().find("Unknown command: does_not_exist"),
            std::string::npos);
  EXPECT_NE(err.str().find("Available commands:"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorUsesExifFocalPriorByDefault) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_exif_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  std::filesystem::copy_file(
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
          "P1180166.JPG",
      image_dir / "P1180166.JPG",
      std::filesystem::copy_options::overwrite_existing);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--workspace_path",
       workspace.string(),
       "--max_num_features",
       "32",
       "--feature_num_threads",
       "1"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  ASSERT_EQ(cameras.size(), 1U);
  ASSERT_EQ(cameras[0].params.size(), 4U);
  EXPECT_EQ(cameras[0].model_id, CameraModelId::kSimpleRadial);
  EXPECT_NEAR(cameras[0].params[0], 2484.86249133349, 1e-6);
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
}

TEST(SfmCommandExeTest,
     FeatureExtractorCreatesIndividualCamerasForMixedImageDimensions) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_individual_camera_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image1.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image2.png").string(),
      cv::Mat(60, 120, CV_8UC3, cv::Scalar(60, 40, 20))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--workspace_path",
       workspace.string(),
       "--max_num_features",
       "32",
       "--feature_num_threads",
       "1",
       "--feature_extractor",
       "vlsift"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  const std::vector<Image> images = ReadWorkspaceImages(workspace);
  ASSERT_EQ(cameras.size(), 2U);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(cameras[0].width, 100U);
  EXPECT_EQ(cameras[0].height, 80U);
  EXPECT_EQ(cameras[1].width, 120U);
  EXPECT_EQ(cameras[1].height, 60U);
  EXPECT_EQ(images[0].camera_id, cameras[0].id);
  EXPECT_EQ(images[1].camera_id, cameras[1].id);
  EXPECT_NE(out.str().find("Cameras: 2"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorSharedModeRejectsMixedDimensions) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_shared_camera_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image1.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image2.png").string(),
      cv::Mat(60, 120, CV_8UC3, cv::Scalar(60, 40, 20))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--workspace_path",
       workspace.string(),
       "--camera_mode",
       "shared"},
      out,
      err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("identical image dimensions"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorAcceptsCameraModelAndParameters) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_camera_model_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--workspace_path",
       workspace.string(),
       "--camera_model",
       "OPENCV",
       "--camera_params",
       "90,91,50,40,0.01,-0.002,0.0001,-0.0002",
       "--camera_mode",
       "per_image",
       "--max_num_features",
       "16",
       "--feature_num_threads",
       "1"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  ASSERT_EQ(cameras.size(), 1U);
  EXPECT_EQ(cameras[0].model_id, CameraModelId::kOpenCV);
  EXPECT_EQ(cameras[0].params,
            (std::vector<double>{90, 91, 50, 40, 0.01, -0.002,
                                 0.0001, -0.0002}));
}

TEST(CameraPriorTest, EstimatesFocalLengthFromJpegExif) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";

  EXPECT_NEAR(EstimateInitialFocalLengthFromImage(image_path, 0.0),
              2484.86249133349,
              1e-6);
  const ExifCameraMetadata metadata = ReadExifCameraMetadata(image_path);
  EXPECT_TRUE(metadata.make.has_value());
  EXPECT_TRUE(metadata.model.has_value());
  EXPECT_TRUE(metadata.HasAutoCameraIdentity());
}

}  // namespace
}  // namespace colite
