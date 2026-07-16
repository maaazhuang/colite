#include "colite/io/sfm_workspace.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "colite/core/geometry.h"

namespace colite {
namespace {

std::vector<std::string> DataLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open workspace file: " +
                             path.string());
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

std::string Point3DIdForText(Point3DId point3d_id) {
  return point3d_id == kInvalidPoint3DId ? "-1" : std::to_string(point3d_id);
}

int GeometryTypeToInt(TwoViewGeometry::Type type) {
  return static_cast<int>(type);
}

TwoViewGeometry::Type GeometryTypeFromInt(int value) {
  switch (static_cast<TwoViewGeometry::Type>(value)) {
    case TwoViewGeometry::Type::kUndefined:
    case TwoViewGeometry::Type::kDegenerate:
    case TwoViewGeometry::Type::kCalibratedEssential:
    case TwoViewGeometry::Type::kUncalibratedFundamental:
    case TwoViewGeometry::Type::kPlanarHomography:
    case TwoViewGeometry::Type::kPanoramic:
    case TwoViewGeometry::Type::kMultiple:
      return static_cast<TwoViewGeometry::Type>(value);
  }
  throw std::runtime_error("Invalid two-view geometry type");
}

template <typename T>
void WriteBinary(std::ofstream* file, const T& value) {
  file->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void ReadBinary(std::ifstream* file, T* value) {
  file->read(reinterpret_cast<char*>(value), sizeof(T));
  if (!*file) {
    throw std::runtime_error("Unexpected end of binary workspace file");
  }
}

std::filesystem::path NormalizeRequiredFile(
    const std::filesystem::path& path) {
  return path.lexically_normal();
}

std::string ToHex(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setw(16) << std::setfill('0') << value;
  return stream.str();
}

}  // namespace

std::filesystem::path WorkspaceFeaturesPath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "features";
}

std::filesystem::path WorkspaceMatchesPath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "matches";
}

std::filesystem::path WorkspaceSparsePath(
    const std::filesystem::path& workspace_path) {
  return workspace_path / "sparse";
}

std::filesystem::path WorkspaceCacheMetadataPath(
    const std::filesystem::path& workspace_path,
    const std::string& stage) {
  return workspace_path / "cache" / (stage + ".txt");
}

std::string StableConfigHash(
    const std::vector<std::pair<std::string, std::string>>& entries) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&](const char ch) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  };

  for (const auto& [key, value] : entries) {
    for (const char ch : key) {
      mix(ch);
    }
    mix('=');
    for (const char ch : value) {
      mix(ch);
    }
    mix('\n');
  }
  return ToHex(hash);
}

void WriteWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage,
    const std::string& config_hash,
    const std::map<std::string, std::string>& values,
    const std::vector<std::filesystem::path>& required_files) {
  std::filesystem::create_directories(
      WorkspaceCacheMetadataPath(workspace_path, stage).parent_path());
  std::ofstream file(WorkspaceCacheMetadataPath(workspace_path, stage));
  file << "stage\t" << stage << "\n"
       << "config_hash\t" << config_hash << "\n";
  for (const std::filesystem::path& required_file : required_files) {
    file << "required_file\t"
         << NormalizeRequiredFile(required_file).generic_string() << "\n";
  }
  for (const auto& [key, value] : values) {
    file << "value\t" << key << "\t" << value << "\n";
  }
}

WorkspaceCacheMetadata ReadWorkspaceCacheMetadata(
    const std::filesystem::path& workspace_path,
    const std::string& stage) {
  const std::filesystem::path path =
      WorkspaceCacheMetadataPath(workspace_path, stage);
  WorkspaceCacheMetadata metadata;
  std::ifstream file(path);
  if (!file) {
    return metadata;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream stream(line);
    std::string kind;
    std::getline(stream, kind, '\t');
    if (kind == "stage") {
      std::getline(stream, metadata.stage);
    } else if (kind == "config_hash") {
      std::getline(stream, metadata.config_hash);
    } else if (kind == "required_file") {
      std::string required_file;
      std::getline(stream, required_file);
      metadata.required_files.emplace_back(required_file);
    } else if (kind == "value") {
      std::string key;
      std::string value;
      std::getline(stream, key, '\t');
      std::getline(stream, value);
      metadata.values.emplace(std::move(key), std::move(value));
    } else {
      throw std::runtime_error("Invalid workspace cache metadata entry: " +
                               line);
    }
  }
  return metadata;
}

bool WorkspaceCacheIsValid(const std::filesystem::path& workspace_path,
                           const std::string& stage,
                           const std::string& config_hash) {
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, stage);
  if (metadata.stage != stage || metadata.config_hash != config_hash) {
    return false;
  }
  for (const std::filesystem::path& required_file : metadata.required_files) {
    if (!std::filesystem::exists(workspace_path / required_file)) {
      return false;
    }
  }
  return true;
}

void WriteWorkspaceCameras(const std::filesystem::path& workspace_path,
                           const std::vector<Camera>& cameras) {
  std::filesystem::create_directories(workspace_path);
  std::ofstream file(workspace_path / "cameras.txt");
  file << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n";
  for (const Camera& camera : cameras) {
    file << camera.id << " " << CameraModelIdToName(camera.model_id) << " "
         << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      file << " " << std::setprecision(17) << param;
    }
    file << "\n";
  }
}

std::vector<Camera> ReadWorkspaceCameras(
    const std::filesystem::path& workspace_path) {
  std::vector<Camera> cameras;
  for (const std::string& line : DataLines(workspace_path / "cameras.txt")) {
    std::istringstream stream(line);
    Camera camera;
    std::string model_name;
    stream >> camera.id >> model_name >> camera.width >> camera.height;
    camera.model_id = CameraModelNameToId(model_name);
    double param = 0.0;
    while (stream >> param) {
      camera.params.push_back(param);
    }
    if (!stream.eof() || !camera.VerifyParams()) {
      throw std::runtime_error("Invalid workspace camera entry: " + line);
    }
    cameras.push_back(camera);
  }
  return cameras;
}

void WriteWorkspaceImages(const std::filesystem::path& workspace_path,
                          const std::vector<Image>& images) {
  std::filesystem::create_directories(workspace_path);
  std::ofstream file(workspace_path / "images.tsv");
  file << "# image_id\tcamera_id\tname\tfile_path\tprior_x\tprior_y\tprior_z\n";
  for (const Image& image : images) {
    file << image.id << "\t" << image.camera_id << "\t" << image.name << "\t"
         << image.file_path;
    if (image.prior_position.has_value()) {
      file << "\t" << std::setprecision(17) << image.prior_position->x()
           << "\t" << image.prior_position->y() << "\t"
           << image.prior_position->z();
    }
    file << "\n";
  }
}

std::vector<Image> ReadWorkspaceImages(
    const std::filesystem::path& workspace_path) {
  std::vector<Image> images;
  for (const std::string& line : DataLines(workspace_path / "images.tsv")) {
    std::istringstream stream(line);
    Image image;
    std::string image_id_text;
    std::string camera_id_text;
    std::getline(stream, image_id_text, '\t');
    std::getline(stream, camera_id_text, '\t');
    image.id = static_cast<ImageId>(std::stoul(image_id_text));
    image.camera_id = static_cast<CameraId>(std::stoul(camera_id_text));
    std::getline(stream, image.name, '\t');
    std::getline(stream, image.file_path, '\t');
    if (!stream.eof()) {
      std::string x_text;
      std::string y_text;
      std::string z_text;
      std::getline(stream, x_text, '\t');
      std::getline(stream, y_text, '\t');
      std::getline(stream, z_text);
      if (x_text.empty() || y_text.empty() || z_text.empty()) {
        throw std::runtime_error("Invalid workspace image prior entry: " +
                                 line);
      }
      image.prior_position = Eigen::Vector3d(std::stod(x_text),
                                             std::stod(y_text),
                                             std::stod(z_text));
    }
    if (image.id == kInvalidImageId || image.camera_id == kInvalidCameraId ||
        image.name.empty()) {
      throw std::runtime_error("Invalid workspace image entry: " + line);
    }
    images.push_back(image);
  }
  return images;
}

void WriteWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                              const ImageId image_id,
                              const FeatureSet& features) {
  std::filesystem::create_directories(WorkspaceFeaturesPath(workspace_path));
  std::ofstream file(WorkspaceFeaturesPath(workspace_path) /
                         (std::to_string(image_id) + ".bin"),
                     std::ios::binary);
  const std::uint32_t magic = 0x434C4654U;
  const std::uint32_t version = 1;
  const std::uint64_t num_keypoints = features.keypoints.size();
  const std::uint64_t rows = static_cast<std::uint64_t>(
      features.descriptors.rows());
  const std::uint64_t cols = static_cast<std::uint64_t>(
      features.descriptors.cols());
  WriteBinary(&file, magic);
  WriteBinary(&file, version);
  WriteBinary(&file, num_keypoints);
  WriteBinary(&file, rows);
  WriteBinary(&file, cols);
  for (const FeatureKeypoint& keypoint : features.keypoints) {
    WriteBinary(&file, keypoint.x);
    WriteBinary(&file, keypoint.y);
    WriteBinary(&file, keypoint.scale);
    WriteBinary(&file, keypoint.orientation);
    WriteBinary(&file, keypoint.response);
  }
  if (features.descriptors.size() > 0) {
    file.write(reinterpret_cast<const char*>(features.descriptors.data()),
               static_cast<std::streamsize>(features.descriptors.size() *
                                            sizeof(float)));
  }
}

FeatureSet ReadWorkspaceFeatureSet(const std::filesystem::path& workspace_path,
                                   const ImageId image_id) {
  std::ifstream file(WorkspaceFeaturesPath(workspace_path) /
                         (std::to_string(image_id) + ".bin"),
                     std::ios::binary);
  if (!file) {
    throw std::runtime_error("Missing feature file for image " +
                             std::to_string(image_id));
  }
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t num_keypoints = 0;
  std::uint64_t rows = 0;
  std::uint64_t cols = 0;
  ReadBinary(&file, &magic);
  ReadBinary(&file, &version);
  ReadBinary(&file, &num_keypoints);
  ReadBinary(&file, &rows);
  ReadBinary(&file, &cols);
  if (magic != 0x434C4654U || version != 1) {
    throw std::runtime_error("Unsupported feature file format");
  }

  FeatureSet features;
  features.keypoints.resize(static_cast<std::size_t>(num_keypoints));
  for (FeatureKeypoint& keypoint : features.keypoints) {
    ReadBinary(&file, &keypoint.x);
    ReadBinary(&file, &keypoint.y);
    ReadBinary(&file, &keypoint.scale);
    ReadBinary(&file, &keypoint.orientation);
    ReadBinary(&file, &keypoint.response);
  }
  features.descriptors.resize(static_cast<int>(rows), static_cast<int>(cols));
  if (features.descriptors.size() > 0) {
    file.read(reinterpret_cast<char*>(features.descriptors.data()),
              static_cast<std::streamsize>(features.descriptors.size() *
                                           sizeof(float)));
    if (!file) {
      throw std::runtime_error("Truncated descriptor matrix");
    }
  }
  return features;
}

void ReadWorkspaceFeatureSets(const std::filesystem::path& workspace_path,
                              MatchGraph* graph) {
  for (const Image& image : ReadWorkspaceImages(workspace_path)) {
    SetFeatures(graph, image.id, ReadWorkspaceFeatureSet(workspace_path,
                                                         image.id));
  }
}

void WriteWorkspaceMatches(const std::filesystem::path& workspace_path,
                           const MatchGraph& graph) {
  std::filesystem::create_directories(WorkspaceMatchesPath(workspace_path));
  std::ofstream file(WorkspaceMatchesPath(workspace_path) / "raw_matches.tsv");
  file << "# image_id1\timage_id2\tidx1\tidx2\n";
  for (const auto& [pair, matches] : graph.matches) {
    for (const FeatureMatch& match : matches) {
      file << pair.image_id1 << "\t" << pair.image_id2 << "\t" << match.idx1
           << "\t" << match.idx2 << "\n";
    }
  }
}

void ReadWorkspaceMatches(const std::filesystem::path& workspace_path,
                          MatchGraph* graph) {
  std::map<ImagePair, std::vector<FeatureMatch>> grouped;
  const std::filesystem::path path =
      WorkspaceMatchesPath(workspace_path) / "raw_matches.tsv";
  if (!std::filesystem::exists(path)) {
    return;
  }
  for (const std::string& line : DataLines(path)) {
    std::istringstream stream(line);
    ImageId image_id1 = kInvalidImageId;
    ImageId image_id2 = kInvalidImageId;
    FeatureMatch match;
    stream >> image_id1 >> image_id2 >> match.idx1 >> match.idx2;
    if (!stream) {
      throw std::runtime_error("Invalid raw match entry: " + line);
    }
    grouped[ImagePair::Ordered(image_id1, image_id2)].push_back(match);
  }
  for (const auto& [pair, matches] : grouped) {
    SetMatches(graph, pair.image_id1, pair.image_id2, matches);
  }
}

void WriteWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path,
    const MatchGraph& graph) {
  std::filesystem::create_directories(WorkspaceMatchesPath(workspace_path));
  std::ofstream file(WorkspaceMatchesPath(workspace_path) /
                     "verified_matches.tsv");
  file << "# image_id1\timage_id2\ttype\tmedian_angle\tqw\tqx\tqy\tqz\ttx\tty"
          "\ttz\tidx1\tidx2\n";
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (geometry.inlier_matches.empty()) {
      file << pair.image_id1 << "\t" << pair.image_id2 << "\t"
           << GeometryTypeToInt(geometry.type) << "\t"
           << std::setprecision(17) << geometry.median_triangulation_angle
           << "\t" << geometry.cam2_from_cam1.q_cw.w() << "\t"
           << geometry.cam2_from_cam1.q_cw.x() << "\t"
           << geometry.cam2_from_cam1.q_cw.y() << "\t"
           << geometry.cam2_from_cam1.q_cw.z() << "\t"
           << geometry.cam2_from_cam1.t_cw.x() << "\t"
           << geometry.cam2_from_cam1.t_cw.y() << "\t"
           << geometry.cam2_from_cam1.t_cw.z() << "\t-1\t-1\n";
      continue;
    }
    for (const FeatureMatch& match : geometry.inlier_matches) {
      file << pair.image_id1 << "\t" << pair.image_id2 << "\t"
           << GeometryTypeToInt(geometry.type) << "\t"
           << std::setprecision(17) << geometry.median_triangulation_angle
           << "\t" << geometry.cam2_from_cam1.q_cw.w() << "\t"
           << geometry.cam2_from_cam1.q_cw.x() << "\t"
           << geometry.cam2_from_cam1.q_cw.y() << "\t"
           << geometry.cam2_from_cam1.q_cw.z() << "\t"
           << geometry.cam2_from_cam1.t_cw.x() << "\t"
           << geometry.cam2_from_cam1.t_cw.y() << "\t"
           << geometry.cam2_from_cam1.t_cw.z() << "\t" << match.idx1
           << "\t" << match.idx2 << "\n";
    }
  }
}

void ReadWorkspaceVerifiedGeometries(
    const std::filesystem::path& workspace_path,
    MatchGraph* graph) {
  std::map<ImagePair, TwoViewGeometry> grouped;
  const std::filesystem::path path =
      WorkspaceMatchesPath(workspace_path) / "verified_matches.tsv";
  if (!std::filesystem::exists(path)) {
    return;
  }
  for (const std::string& line : DataLines(path)) {
    std::istringstream stream(line);
    ImageId image_id1 = kInvalidImageId;
    ImageId image_id2 = kInvalidImageId;
    int type = 0;
    FeatureMatch match;
    match.idx1 = kInvalidPoint2DIdx;
    match.idx2 = kInvalidPoint2DIdx;
    TwoViewGeometry geometry;
    stream >> image_id1 >> image_id2 >> type >>
        geometry.median_triangulation_angle >>
        geometry.cam2_from_cam1.q_cw.w() >>
        geometry.cam2_from_cam1.q_cw.x() >>
        geometry.cam2_from_cam1.q_cw.y() >>
        geometry.cam2_from_cam1.q_cw.z() >>
        geometry.cam2_from_cam1.t_cw.x() >>
        geometry.cam2_from_cam1.t_cw.y() >>
        geometry.cam2_from_cam1.t_cw.z() >> match.idx1 >> match.idx2;
    if (!stream) {
      throw std::runtime_error("Invalid verified match entry: " + line);
    }
    geometry.type = GeometryTypeFromInt(type);
    const ImagePair pair = ImagePair::Ordered(image_id1, image_id2);
    auto [it, inserted] = grouped.emplace(pair, geometry);
    (void)inserted;
    if (match.idx1 != kInvalidPoint2DIdx && match.idx2 != kInvalidPoint2DIdx) {
      it->second.inlier_matches.push_back(match);
    }
  }
  for (const auto& [pair, geometry] : grouped) {
    SetVerifiedGeometry(graph, pair.image_id1, pair.image_id2, geometry);
  }
}

void WriteWorkspaceSparseReconstruction(
    const std::filesystem::path& workspace_path,
    const Reconstruction& reconstruction) {
  const std::filesystem::path sparse_path = WorkspaceSparsePath(workspace_path);
  std::filesystem::create_directories(sparse_path);

  std::ofstream cameras_file(sparse_path / "cameras.txt");
  cameras_file << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n";
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    (void)camera_id;
    cameras_file << camera.id << " " << CameraModelIdToName(camera.model_id)
                 << " " << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      cameras_file << " " << std::setprecision(17) << param;
    }
    cameras_file << "\n";
  }

  std::ofstream images_file(sparse_path / "images.txt");
  images_file << "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
              << "# POINTS2D[] as X Y POINT3D_ID\n";
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.registered || !image.cam_from_world.has_value()) {
      continue;
    }
    const Pose3& pose = *image.cam_from_world;
    images_file << std::setprecision(17) << image_id << " "
                << pose.q_cw.w() << " " << pose.q_cw.x() << " "
                << pose.q_cw.y() << " " << pose.q_cw.z() << " "
                << pose.t_cw.x() << " " << pose.t_cw.y() << " "
                << pose.t_cw.z() << " " << image.camera_id << " "
                << image.name << "\n";
    for (std::size_t i = 0; i < image.points2d.size(); ++i) {
      if (i > 0) {
        images_file << " ";
      }
      images_file << std::setprecision(17) << image.points2d[i].xy.x()
                  << " " << image.points2d[i].xy.y() << " "
                  << Point3DIdForText(image.points2d[i].point3d_id);
    }
    images_file << "\n";
  }

  std::ofstream points_file(sparse_path / "points3D.txt");
  points_file << "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n";
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    const double track_error =
        Point3DTrackReprojectionError(reconstruction, point3d).value_or(-1.0);
    points_file << point3d_id << " " << std::setprecision(17)
                << point3d.xyz.x() << " " << point3d.xyz.y() << " "
                << point3d.xyz.z() << " 255 255 255 "
                << track_error;
    for (const TrackElement& element : point3d.track.elements) {
      points_file << " " << element.image_id << " " << element.point2d_idx;
    }
    points_file << "\n";
  }

  std::ofstream ply_file(sparse_path / "points3D.ply");
  ply_file << "ply\nformat ascii 1.0\nelement vertex "
           << reconstruction.Points3D().size()
           << "\nproperty double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n";
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    ply_file << std::setprecision(17) << point3d.xyz.x() << " "
             << point3d.xyz.y() << " " << point3d.xyz.z()
             << " 255 255 255\n";
  }
}

}  // namespace colite
