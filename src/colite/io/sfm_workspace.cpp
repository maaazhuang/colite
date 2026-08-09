#include "colite/io/sfm_workspace.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
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

std::vector<std::pair<std::string, std::string>> ColmapImageEntries(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open COLMAP model file: " +
                             path.string());
  }
  std::vector<std::pair<std::string, std::string>> entries;
  std::string header;
  while (std::getline(file, header)) {
    if (header.empty() || header.front() == '#') {
      continue;
    }
    std::string points;
    if (!std::getline(file, points)) {
      throw std::runtime_error(
          "COLMAP images.txt is missing the POINTS2D line after: " + header);
    }
    if (!points.empty() && points.front() == '#') {
      throw std::runtime_error(
          "COLMAP images.txt has a comment where a POINTS2D line is expected");
    }
    entries.emplace_back(std::move(header), std::move(points));
  }
  return entries;
}

bool CameraParametersEqual(const Camera& lhs, const Camera& rhs) {
  if (lhs.model_id != rhs.model_id || lhs.width != rhs.width ||
      lhs.height != rhs.height || lhs.params.size() != rhs.params.size()) {
    return false;
  }
  for (std::size_t idx = 0; idx < lhs.params.size(); ++idx) {
    const double scale =
        std::max({1.0, std::abs(lhs.params[idx]), std::abs(rhs.params[idx])});
    if (std::abs(lhs.params[idx] - rhs.params[idx]) > 1e-10 * scale) {
      return false;
    }
  }
  return true;
}

std::string Point3DIdForText(Point3DId point3d_id) {
  return point3d_id == kInvalidPoint3DId ? "-1" : std::to_string(point3d_id);
}

Point3DId ParsePoint3DId(std::int64_t point3d_id) {
  return point3d_id < 0 ? kInvalidPoint3DId
                        : static_cast<Point3DId>(point3d_id);
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
  if (!*file) {
    throw std::runtime_error("Failed to write binary file");
  }
}

template <typename T>
void ReadBinary(std::ifstream* file, T* value) {
  file->read(reinterpret_cast<char*>(value), sizeof(T));
  if (!*file) {
    throw std::runtime_error("Unexpected end of binary workspace file");
  }
}

std::string ReadNullTerminated(std::ifstream* file) {
  std::string value;
  char character = '\0';
  while (file->get(character)) {
    if (character == '\0') {
      return value;
    }
    value.push_back(character);
  }
  throw std::runtime_error("Unexpected end of COLMAP binary image name");
}

template <typename Id, typename Map>
std::vector<Id> SortedIds(const Map& values) {
  std::vector<Id> ids;
  ids.reserve(values.size());
  for (const auto& [id, value] : values) {
    (void)value;
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

bool HasCompleteModel(const std::filesystem::path& path,
                      const std::string& extension) {
  return std::filesystem::is_regular_file(path / ("cameras." + extension)) &&
         std::filesystem::is_regular_file(path / ("images." + extension)) &&
         std::filesystem::is_regular_file(path / ("points3D." + extension));
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

ColmapModelFormat ColmapModelFormatFromString(const std::string& value) {
  if (value == "auto") return ColmapModelFormat::kAuto;
  if (value == "txt" || value == "text") return ColmapModelFormat::kText;
  if (value == "bin" || value == "binary") return ColmapModelFormat::kBinary;
  if (value == "both" || value == "txt+bin") {
    return ColmapModelFormat::kTextAndBinary;
  }
  throw std::invalid_argument(
      "COLMAP model format must be auto, txt, bin, or both: " + value);
}

std::string ColmapModelFormatToString(const ColmapModelFormat format) {
  switch (format) {
    case ColmapModelFormat::kAuto: return "auto";
    case ColmapModelFormat::kText: return "txt";
    case ColmapModelFormat::kBinary: return "bin";
    case ColmapModelFormat::kTextAndBinary: return "both";
  }
  throw std::invalid_argument("Unknown COLMAP model format");
}

ColmapModelFormat DetectColmapModelFormat(
    const std::filesystem::path& model_path) {
  if (HasCompleteModel(model_path, "bin")) {
    return ColmapModelFormat::kBinary;
  }
  if (HasCompleteModel(model_path, "txt")) {
    return ColmapModelFormat::kText;
  }
  throw std::runtime_error(
      "COLMAP model requires a complete cameras/images/points3D .bin or .txt "
      "file set: " + model_path.string());
}

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
  file << "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] PRIOR_FOCAL=<0|1>\n";
  for (const Camera& camera : cameras) {
    file << camera.id << " " << CameraModelIdToName(camera.model_id) << " "
         << camera.width << " " << camera.height;
    for (const double param : camera.params) {
      file << " " << std::setprecision(17) << param;
    }
    file << " PRIOR_FOCAL=" << (camera.has_prior_focal_length ? 1 : 0)
         << "\n";
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
    const std::size_t num_params =
        GetCameraModelSpec(camera.model_id).param_names.size();
    for (std::size_t idx = 0; idx < num_params; ++idx) {
      double param = 0.0;
      if (!(stream >> param)) {
        throw std::runtime_error("Invalid workspace camera entry: " + line);
      }
      camera.params.push_back(param);
    }
    std::string metadata;
    if (stream >> metadata) {
      if (metadata == "PRIOR_FOCAL=1") {
        camera.has_prior_focal_length = true;
      } else if (metadata != "PRIOR_FOCAL=0") {
        throw std::runtime_error("Invalid workspace camera metadata: " + line);
      }
    }
    std::string trailing;
    if (stream >> trailing || !camera.VerifyParams()) {
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
    const Reconstruction& reconstruction,
    const ColmapModelFormat format) {
  WriteColmapReconstruction(WorkspaceSparsePath(workspace_path),
                            reconstruction, format);
}

void WriteSparseReconstruction(
    const std::filesystem::path& sparse_path,
    const Reconstruction& reconstruction) {
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
                << point3d.xyz.z() << " "
                << static_cast<int>(point3d.color.x()) << " "
                << static_cast<int>(point3d.color.y()) << " "
                << static_cast<int>(point3d.color.z()) << " "
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
             << point3d.xyz.y() << " " << point3d.xyz.z() << " "
             << static_cast<int>(point3d.color.x()) << " "
             << static_cast<int>(point3d.color.y()) << " "
             << static_cast<int>(point3d.color.z()) << "\n";
  }
}

void WriteColmapBinaryReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction) {
  std::filesystem::create_directories(model_path);
  std::ofstream cameras_file(model_path / "cameras.bin", std::ios::binary);
  std::ofstream images_file(model_path / "images.bin", std::ios::binary);
  std::ofstream points_file(model_path / "points3D.bin", std::ios::binary);
  if (!cameras_file || !images_file || !points_file) {
    throw std::runtime_error("Failed to create COLMAP binary model: " +
                             model_path.string());
  }

  const std::vector<CameraId> camera_ids =
      SortedIds<CameraId>(reconstruction.Cameras());
  WriteBinary(&cameras_file, static_cast<std::uint64_t>(camera_ids.size()));
  for (const CameraId camera_id : camera_ids) {
    const Camera& camera = reconstruction.GetCamera(camera_id);
    WriteBinary(&cameras_file, static_cast<std::uint32_t>(camera.id));
    WriteBinary(&cameras_file, static_cast<std::int32_t>(camera.model_id));
    WriteBinary(&cameras_file, static_cast<std::uint64_t>(camera.width));
    WriteBinary(&cameras_file, static_cast<std::uint64_t>(camera.height));
    for (const double parameter : camera.params) {
      WriteBinary(&cameras_file, parameter);
    }
  }

  std::vector<ImageId> registered_image_ids(
      reconstruction.RegisteredImageIds().begin(),
      reconstruction.RegisteredImageIds().end());
  WriteBinary(&images_file,
              static_cast<std::uint64_t>(registered_image_ids.size()));
  for (const ImageId image_id : registered_image_ids) {
    const Image& image = reconstruction.GetImage(image_id);
    if (!image.cam_from_world.has_value()) {
      throw std::runtime_error(
          "Registered image has no pose while writing COLMAP binary model");
    }
    const Pose3& pose = *image.cam_from_world;
    WriteBinary(&images_file, static_cast<std::uint32_t>(image.id));
    WriteBinary(&images_file, pose.q_cw.w());
    WriteBinary(&images_file, pose.q_cw.x());
    WriteBinary(&images_file, pose.q_cw.y());
    WriteBinary(&images_file, pose.q_cw.z());
    WriteBinary(&images_file, pose.t_cw.x());
    WriteBinary(&images_file, pose.t_cw.y());
    WriteBinary(&images_file, pose.t_cw.z());
    WriteBinary(&images_file, static_cast<std::uint32_t>(image.camera_id));
    images_file.write(image.name.data(),
                      static_cast<std::streamsize>(image.name.size()));
    images_file.put('\0');
    if (!images_file) {
      throw std::runtime_error("Failed to write COLMAP binary image name");
    }
    WriteBinary(&images_file,
                static_cast<std::uint64_t>(image.points2d.size()));
    for (const Point2D& point2d : image.points2d) {
      WriteBinary(&images_file, point2d.xy.x());
      WriteBinary(&images_file, point2d.xy.y());
      WriteBinary(&images_file,
                  static_cast<std::uint64_t>(point2d.point3d_id));
    }
  }

  const std::vector<Point3DId> point_ids =
      SortedIds<Point3DId>(reconstruction.Points3D());
  WriteBinary(&points_file, static_cast<std::uint64_t>(point_ids.size()));
  for (const Point3DId point_id : point_ids) {
    const Point3D& point = reconstruction.GetPoint3D(point_id);
    WriteBinary(&points_file, static_cast<std::uint64_t>(point.id));
    WriteBinary(&points_file, point.xyz.x());
    WriteBinary(&points_file, point.xyz.y());
    WriteBinary(&points_file, point.xyz.z());
    WriteBinary(&points_file, point.color.x());
    WriteBinary(&points_file, point.color.y());
    WriteBinary(&points_file, point.color.z());
    const double error =
        Point3DTrackReprojectionError(reconstruction, point)
            .value_or(point.reprojection_error);
    WriteBinary(&points_file, error);
    WriteBinary(&points_file,
                static_cast<std::uint64_t>(point.track.elements.size()));
    for (const TrackElement& element : point.track.elements) {
      WriteBinary(&points_file, static_cast<std::uint32_t>(element.image_id));
      WriteBinary(&points_file,
                  static_cast<std::uint32_t>(element.point2d_idx));
    }
  }
}

void WriteColmapReconstruction(
    const std::filesystem::path& model_path,
    const Reconstruction& reconstruction,
    const ColmapModelFormat format) {
  switch (format) {
    case ColmapModelFormat::kText:
      WriteSparseReconstruction(model_path, reconstruction);
      return;
    case ColmapModelFormat::kBinary:
      WriteColmapBinaryReconstruction(model_path, reconstruction);
      return;
    case ColmapModelFormat::kTextAndBinary:
      WriteSparseReconstruction(model_path, reconstruction);
      WriteColmapBinaryReconstruction(model_path, reconstruction);
      return;
    case ColmapModelFormat::kAuto:
      throw std::invalid_argument(
          "COLMAP auto format is valid for reading but not writing");
  }
  throw std::invalid_argument("Unknown COLMAP model format");
}

Reconstruction ReadColmapTextReconstruction(
    const std::filesystem::path& model_path) {
  Reconstruction reconstruction;
  for (const std::string& line : DataLines(model_path / "cameras.txt")) {
    std::istringstream stream(line);
    Camera camera;
    std::string model_name;
    stream >> camera.id >> model_name >> camera.width >> camera.height;
    camera.model_id = CameraModelNameToId(model_name);
    double parameter = 0.0;
    while (stream >> parameter) {
      camera.params.push_back(parameter);
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid COLMAP camera entry: " + line);
    }
    reconstruction.AddCamera(std::move(camera));
  }

  for (const auto& [header_line, points_line] :
       ColmapImageEntries(model_path / "images.txt")) {
    std::istringstream header(header_line);
    Image image;
    Pose3 pose;
    header >> image.id >> pose.q_cw.w() >> pose.q_cw.x() >> pose.q_cw.y() >>
        pose.q_cw.z() >> pose.t_cw.x() >> pose.t_cw.y() >> pose.t_cw.z() >>
        image.camera_id;
    std::getline(header, image.name);
    const std::size_t name_start = image.name.find_first_not_of(" \t");
    if (name_start != std::string::npos) {
      image.name.erase(0, name_start);
    } else {
      image.name.clear();
    }
    if (!header || !reconstruction.ExistsCamera(image.camera_id)) {
      throw std::runtime_error("Invalid COLMAP image entry: " + header_line);
    }
    image.cam_from_world = pose;
    image.registered = true;
    std::istringstream points(points_line);
    double x = 0.0;
    double y = 0.0;
    std::int64_t point3d_id = -1;
    Point2DIdx point2d_idx = 0;
    while (points >> x >> y >> point3d_id) {
      image.points2d.push_back(
          {Eigen::Vector2d(x, y), point2d_idx++,
           ParsePoint3DId(point3d_id)});
    }
    reconstruction.AddImage(std::move(image));
  }

  for (const std::string& line : DataLines(model_path / "points3D.txt")) {
    std::istringstream stream(line);
    Point3D point;
    int red = 0;
    int green = 0;
    int blue = 0;
    stream >> point.id >> point.xyz.x() >> point.xyz.y() >> point.xyz.z() >>
        red >> green >> blue >> point.reprojection_error;
    if (!stream) {
      throw std::runtime_error("Invalid COLMAP point entry: " + line);
    }
    point.color = Eigen::Matrix<std::uint8_t, 3, 1>(
        static_cast<std::uint8_t>(red), static_cast<std::uint8_t>(green),
        static_cast<std::uint8_t>(blue));
    ImageId image_id = kInvalidImageId;
    Point2DIdx point2d_idx = kInvalidPoint2DIdx;
    while (stream >> image_id >> point2d_idx) {
      point.track.AddElement(image_id, point2d_idx);
    }
    reconstruction.AddPoint3D(std::move(point));
  }
  return reconstruction;
}

Reconstruction ReadColmapBinaryReconstruction(
    const std::filesystem::path& model_path) {
  std::ifstream cameras_file(model_path / "cameras.bin", std::ios::binary);
  std::ifstream images_file(model_path / "images.bin", std::ios::binary);
  std::ifstream points_file(model_path / "points3D.bin", std::ios::binary);
  if (!cameras_file || !images_file || !points_file) {
    throw std::runtime_error("Failed to open complete COLMAP binary model: " +
                             model_path.string());
  }

  Reconstruction reconstruction;
  std::uint64_t num_cameras = 0;
  ReadBinary(&cameras_file, &num_cameras);
  for (std::uint64_t index = 0; index < num_cameras; ++index) {
    Camera camera;
    std::int32_t model_id = -1;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    ReadBinary(&cameras_file, &camera.id);
    ReadBinary(&cameras_file, &model_id);
    ReadBinary(&cameras_file, &width);
    ReadBinary(&cameras_file, &height);
    if (width > std::numeric_limits<std::uint32_t>::max() ||
        height > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("COLMAP binary camera dimensions overflow");
    }
    camera.model_id = static_cast<CameraModelId>(model_id);
    camera.width = static_cast<std::uint32_t>(width);
    camera.height = static_cast<std::uint32_t>(height);
    if (!ExistsCameraModel(camera.model_id)) {
      throw std::runtime_error("Unknown COLMAP binary camera model id: " +
                               std::to_string(model_id));
    }
    camera.params.resize(GetCameraModelSpec(camera.model_id).param_names.size());
    for (double& parameter : camera.params) {
      ReadBinary(&cameras_file, &parameter);
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid COLMAP binary camera");
    }
    reconstruction.AddCamera(std::move(camera));
  }

  std::uint64_t num_images = 0;
  ReadBinary(&images_file, &num_images);
  for (std::uint64_t index = 0; index < num_images; ++index) {
    Image image;
    Pose3 pose;
    ReadBinary(&images_file, &image.id);
    ReadBinary(&images_file, &pose.q_cw.w());
    ReadBinary(&images_file, &pose.q_cw.x());
    ReadBinary(&images_file, &pose.q_cw.y());
    ReadBinary(&images_file, &pose.q_cw.z());
    ReadBinary(&images_file, &pose.t_cw.x());
    ReadBinary(&images_file, &pose.t_cw.y());
    ReadBinary(&images_file, &pose.t_cw.z());
    ReadBinary(&images_file, &image.camera_id);
    image.name = ReadNullTerminated(&images_file);
    std::uint64_t num_points2d = 0;
    ReadBinary(&images_file, &num_points2d);
    if (num_points2d > std::numeric_limits<Point2DIdx>::max()) {
      throw std::runtime_error("COLMAP binary image has too many 2D points");
    }
    image.points2d.reserve(static_cast<std::size_t>(num_points2d));
    for (std::uint64_t point_index = 0; point_index < num_points2d;
         ++point_index) {
      Eigen::Vector2d xy;
      Point3DId point3d_id = kInvalidPoint3DId;
      ReadBinary(&images_file, &xy.x());
      ReadBinary(&images_file, &xy.y());
      ReadBinary(&images_file, &point3d_id);
      image.points2d.push_back(
          {xy, static_cast<Point2DIdx>(point_index), point3d_id});
    }
    if (!reconstruction.ExistsCamera(image.camera_id)) {
      throw std::runtime_error(
          "COLMAP binary image references a missing camera");
    }
    image.cam_from_world = pose;
    image.registered = true;
    reconstruction.AddImage(std::move(image));
  }

  std::uint64_t num_points3d = 0;
  ReadBinary(&points_file, &num_points3d);
  for (std::uint64_t index = 0; index < num_points3d; ++index) {
    Point3D point;
    ReadBinary(&points_file, &point.id);
    ReadBinary(&points_file, &point.xyz.x());
    ReadBinary(&points_file, &point.xyz.y());
    ReadBinary(&points_file, &point.xyz.z());
    ReadBinary(&points_file, &point.color.x());
    ReadBinary(&points_file, &point.color.y());
    ReadBinary(&points_file, &point.color.z());
    ReadBinary(&points_file, &point.reprojection_error);
    std::uint64_t track_length = 0;
    ReadBinary(&points_file, &track_length);
    point.track.elements.reserve(static_cast<std::size_t>(track_length));
    for (std::uint64_t track_index = 0; track_index < track_length;
         ++track_index) {
      TrackElement element;
      ReadBinary(&points_file, &element.image_id);
      ReadBinary(&points_file, &element.point2d_idx);
      point.track.elements.push_back(element);
    }
    reconstruction.AddPoint3D(std::move(point));
  }
  return reconstruction;
}

Reconstruction ReadColmapReconstruction(
    const std::filesystem::path& model_path, ColmapModelFormat format) {
  if (format == ColmapModelFormat::kAuto) {
    format = DetectColmapModelFormat(model_path);
  }
  if (format == ColmapModelFormat::kText) {
    return ReadColmapTextReconstruction(model_path);
  }
  if (format == ColmapModelFormat::kBinary) {
    return ReadColmapBinaryReconstruction(model_path);
  }
  throw std::invalid_argument(
      "Reading COLMAP model format 'both' is ambiguous; use auto, txt, or bin");
}

Reconstruction ReadWorkspaceReconstruction(
    const std::filesystem::path& workspace_path,
    const std::filesystem::path& input_model_path) {
  const std::vector<Camera> workspace_cameras =
      ReadWorkspaceCameras(workspace_path);
  const std::vector<Image> workspace_images =
      ReadWorkspaceImages(workspace_path);
  if (input_model_path.empty()) {
    Reconstruction reconstruction;
    for (const Camera& camera : workspace_cameras) {
      reconstruction.AddCamera(camera);
    }
    for (const Image& image : workspace_images) {
      reconstruction.AddImage(image);
    }
    return reconstruction;
  }

  Reconstruction schema;
  for (const Camera& camera : workspace_cameras) {
    schema.AddCamera(camera);
  }
  for (const Image& image : workspace_images) {
    schema.AddImage(image);
  }
  return ApplyColmapModel(schema, input_model_path);
}

Reconstruction ApplyColmapTextModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path) {
  return ApplyColmapModel(schema, input_model_path,
                          ColmapModelFormat::kText);
}

Reconstruction ApplyColmapModel(
    const Reconstruction& schema,
    const std::filesystem::path& input_model_path,
    const ColmapModelFormat format) {
  const Reconstruction model =
      ReadColmapReconstruction(input_model_path, format);
  std::map<std::string, ImageId> schema_image_ids;
  for (const auto& [image_id, image] : schema.Images()) {
    if (!schema_image_ids.emplace(image.name, image_id).second) {
      throw std::runtime_error("Workspace contains duplicate image name: " +
                               image.name);
    }
  }
  std::map<ImageId, ImageId> model_to_schema_image;
  std::map<CameraId, Camera> cameras;
  std::map<CameraId, Camera> mapped_model_cameras;
  for (const auto& [camera_id, camera] : schema.Cameras()) {
    cameras.emplace(camera_id, camera);
  }
  for (const auto& [model_image_id, model_image] : model.Images()) {
    const auto schema_id_it = schema_image_ids.find(model_image.name);
    if (schema_id_it == schema_image_ids.end()) {
      throw std::runtime_error(
          "Input model image is absent from current image set: " +
          model_image.name);
    }
    const Image& schema_image = schema.GetImage(schema_id_it->second);
    const Camera& model_camera = model.GetCamera(model_image.camera_id);
    Camera optimized_camera = model_camera;
    optimized_camera.id = schema_image.camera_id;
    const Camera& schema_camera = schema.GetCamera(schema_image.camera_id);
    if (optimized_camera.model_id != schema_camera.model_id ||
        optimized_camera.width != schema_camera.width ||
        optimized_camera.height != schema_camera.height) {
      throw std::runtime_error(
          "Input model camera is incompatible with current image: " +
          model_image.name);
    }
    const auto [mapped_it, inserted] = mapped_model_cameras.emplace(
        schema_image.camera_id, optimized_camera);
    if (!inserted &&
        !CameraParametersEqual(mapped_it->second, optimized_camera)) {
      throw std::runtime_error(
          "Input model has incompatible intrinsics for images sharing "
          "workspace camera " +
          std::to_string(schema_image.camera_id));
    }
    cameras[schema_image.camera_id] = std::move(optimized_camera);
    model_to_schema_image.emplace(model_image_id, schema_id_it->second);
  }

  Reconstruction reconstruction;
  for (const auto& [camera_id, camera] : cameras) {
    (void)camera_id;
    reconstruction.AddCamera(camera);
  }
  std::map<ImageId, const Image*> model_images_by_schema_id;
  for (const auto& [model_image_id, schema_image_id] :
       model_to_schema_image) {
    model_images_by_schema_id.emplace(
        schema_image_id, &model.GetImage(model_image_id));
  }
  for (const auto& [image_id, schema_image] : schema.Images()) {
    (void)image_id;
    const auto model_image_it = model_images_by_schema_id.find(image_id);
    if (model_image_it == model_images_by_schema_id.end()) {
      reconstruction.AddImage(schema_image);
      continue;
    }
    Image image = *model_image_it->second;
    image.id = image_id;
    image.camera_id = schema_image.camera_id;
    image.file_path = schema_image.file_path;
    image.prior_position = schema_image.prior_position;
    reconstruction.AddImage(std::move(image));
  }
  if (reconstruction.RegisteredImageIds().size() !=
      model.RegisteredImageIds().size()) {
    throw std::runtime_error(
        "Input model contains images outside the workspace");
  }
  for (const auto& [point_id, point] : model.Points3D()) {
    (void)point_id;
    Point3D remapped = point;
    for (TrackElement& element : remapped.track.elements) {
      element.image_id = model_to_schema_image.at(element.image_id);
    }
    reconstruction.AddPoint3D(std::move(remapped));
  }
  return reconstruction;
}

}  // namespace colite
