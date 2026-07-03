#include "colite/controllers/colmap_compare_controller.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colite::controllers {
namespace {

struct ColmapImage {
  std::size_t image_id = 0;
  Eigen::Quaterniond q_cw = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t_cw = Eigen::Vector3d::Zero();
  std::size_t camera_id = 0;
  std::string name;
  std::size_t num_observations = 0;
};

struct ColmapPoint3D {
  std::size_t point3d_id = 0;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  double error = 0.0;
  std::vector<std::pair<std::size_t, std::size_t>> track;
};

struct ColmapModel {
  std::map<std::size_t, ColmapImage> images_by_id;
  std::map<std::string, std::size_t> image_ids_by_name;
  std::map<std::size_t, ColmapPoint3D> points3d;
};

struct PairKey {
  std::string image_name1;
  std::string image_name2;

  friend bool operator<(const PairKey& lhs, const PairKey& rhs) {
    return std::tie(lhs.image_name1, lhs.image_name2) <
           std::tie(rhs.image_name1, rhs.image_name2);
  }
};

std::string TrimLeadingSpaces(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return "";
  }
  return text.substr(first);
}

bool IsCommentLine(const std::string& line) {
  return !line.empty() && line.front() == '#';
}

std::vector<std::string> ReadDataLines(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Missing COLMAP text file: " + path.string());
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || IsCommentLine(line)) {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

std::map<std::size_t, ColmapImage> ReadColmapImages(
    const std::filesystem::path& images_path) {
  std::ifstream file(images_path);
  if (!file) {
    throw std::runtime_error("Missing COLMAP images file: " +
                             images_path.string());
  }

  std::map<std::size_t, ColmapImage> images;
  std::string header_line;
  while (std::getline(file, header_line)) {
    if (header_line.empty() || IsCommentLine(header_line)) {
      continue;
    }

    ColmapImage image;
    std::istringstream header(header_line);
    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    header >> image.image_id >> qw >> qx >> qy >> qz >> image.t_cw.x() >>
        image.t_cw.y() >> image.t_cw.z() >> image.camera_id;
    if (!header) {
      throw std::runtime_error("Invalid COLMAP image entry: " + header_line);
    }
    std::string name;
    std::getline(header, name);
    image.name = TrimLeadingSpaces(name);
    image.q_cw = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    if (image.name.empty()) {
      throw std::runtime_error("Missing COLMAP image name: " + header_line);
    }

    std::string points_line;
    if (!std::getline(file, points_line)) {
      points_line.clear();
    }
    std::istringstream points(points_line);
    double x = 0.0;
    double y = 0.0;
    long long point3d_id = -1;
    while (points >> x >> y >> point3d_id) {
      (void)x;
      (void)y;
      if (point3d_id >= 0) {
        ++image.num_observations;
      }
    }

    images.emplace(image.image_id, std::move(image));
  }
  return images;
}

std::map<std::size_t, ColmapPoint3D> ReadColmapPoints3D(
    const std::filesystem::path& points_path) {
  std::map<std::size_t, ColmapPoint3D> points3d;
  for (const std::string& line : ReadDataLines(points_path)) {
    ColmapPoint3D point;
    int red = 0;
    int green = 0;
    int blue = 0;
    std::istringstream stream(line);
    stream >> point.point3d_id >> point.xyz.x() >> point.xyz.y() >>
        point.xyz.z() >> red >> green >> blue >> point.error;
    if (!stream) {
      throw std::runtime_error("Invalid COLMAP point3D entry: " + line);
    }
    std::size_t image_id = 0;
    std::size_t point2d_idx = 0;
    while (stream >> image_id >> point2d_idx) {
      point.track.emplace_back(image_id, point2d_idx);
    }
    points3d.emplace(point.point3d_id, std::move(point));
  }
  return points3d;
}

ColmapModel ReadColmapModel(const std::filesystem::path& path) {
  ColmapModel model;
  model.images_by_id = ReadColmapImages(path / "images.txt");
  model.points3d = ReadColmapPoints3D(path / "points3D.txt");
  for (const auto& [image_id, image] : model.images_by_id) {
    model.image_ids_by_name.emplace(image.name, image_id);
  }
  return model;
}

std::map<PairKey, std::size_t> ComputeSharedPointPairs(
    const ColmapModel& model) {
  std::map<PairKey, std::size_t> pair_counts;
  for (const auto& [point3d_id, point] : model.points3d) {
    (void)point3d_id;
    std::set<std::string> track_image_names;
    for (const auto& [image_id, point2d_idx] : point.track) {
      (void)point2d_idx;
      const auto image_it = model.images_by_id.find(image_id);
      if (image_it != model.images_by_id.end()) {
        track_image_names.insert(image_it->second.name);
      }
    }
    std::vector<std::string> names(track_image_names.begin(),
                                   track_image_names.end());
    for (std::size_t i = 0; i < names.size(); ++i) {
      for (std::size_t j = i + 1; j < names.size(); ++j) {
        ++pair_counts[{names[i], names[j]}];
      }
    }
  }
  return pair_counts;
}

Eigen::Vector3d CameraCenterFromPose(const ColmapImage& image) {
  return -(image.q_cw.conjugate() * image.t_cw);
}

double RotationAngleDeg(const Eigen::Quaterniond& ref_q_cw,
                        const Eigen::Quaterniond& candidate_q_cw) {
  const Eigen::Quaterniond delta =
      (candidate_q_cw * ref_q_cw.conjugate()).normalized();
  const double w = std::min(1.0, std::abs(delta.w()));
  return 2.0 * std::acos(w) * 180.0 / M_PI;
}

std::string StatusForPresence(const bool in_reference,
                              const bool in_candidate) {
  if (in_reference && in_candidate) {
    return "common";
  }
  return in_reference ? "reference_only" : "candidate_only";
}

void WriteSummary(const std::filesystem::path& output_path,
                  const ColmapCompareControllerSummary& summary) {
  std::ofstream file(output_path / "summary.txt");
  file << "reference_images " << summary.reference_images << "\n"
       << "candidate_images " << summary.candidate_images << "\n"
       << "common_images " << summary.common_images << "\n"
       << "reference_only_images " << summary.reference_only_images << "\n"
       << "candidate_only_images " << summary.candidate_only_images << "\n"
       << "reference_points " << summary.reference_points << "\n"
       << "candidate_points " << summary.candidate_points << "\n"
       << "common_points_by_id " << summary.common_points_by_id << "\n"
       << "reference_only_points " << summary.reference_only_points << "\n"
       << "candidate_only_points " << summary.candidate_only_points << "\n"
       << "common_pairs " << summary.common_pairs << "\n";
}

void WriteImageDiff(const std::filesystem::path& output_path,
                    const ColmapModel& reference,
                    const ColmapModel& candidate) {
  std::ofstream file(output_path / "image_diff.tsv");
  file << "image_name\tstatus\tref_image_id\tcandidate_image_id"
          "\tref_camera_id\tcandidate_camera_id\tref_observations"
          "\tcandidate_observations\tobservation_delta\n";
  std::set<std::string> image_names;
  for (const auto& [name, image_id] : reference.image_ids_by_name) {
    (void)image_id;
    image_names.insert(name);
  }
  for (const auto& [name, image_id] : candidate.image_ids_by_name) {
    (void)image_id;
    image_names.insert(name);
  }

  for (const std::string& name : image_names) {
    const auto ref_id_it = reference.image_ids_by_name.find(name);
    const auto cand_id_it = candidate.image_ids_by_name.find(name);
    const bool has_ref = ref_id_it != reference.image_ids_by_name.end();
    const bool has_candidate = cand_id_it != candidate.image_ids_by_name.end();
    const ColmapImage* ref =
        has_ref ? &reference.images_by_id.at(ref_id_it->second) : nullptr;
    const ColmapImage* cand =
        has_candidate ? &candidate.images_by_id.at(cand_id_it->second)
                      : nullptr;
    const long long ref_observations =
        ref == nullptr ? 0LL : static_cast<long long>(ref->num_observations);
    const long long candidate_observations =
        cand == nullptr ? 0LL : static_cast<long long>(cand->num_observations);
    file << name << "\t" << StatusForPresence(has_ref, has_candidate) << "\t"
         << (ref == nullptr ? 0 : ref->image_id) << "\t"
         << (cand == nullptr ? 0 : cand->image_id) << "\t"
         << (ref == nullptr ? 0 : ref->camera_id) << "\t"
         << (cand == nullptr ? 0 : cand->camera_id) << "\t"
         << ref_observations << "\t" << candidate_observations << "\t"
         << (candidate_observations - ref_observations) << "\n";
  }
}

void WritePairDiff(const std::filesystem::path& output_path,
                   const std::map<PairKey, std::size_t>& reference_pairs,
                   const std::map<PairKey, std::size_t>& candidate_pairs) {
  std::ofstream file(output_path / "pair_diff.tsv");
  file << "image_name1\timage_name2\tref_shared_points"
          "\tcandidate_shared_points\tshared_point_delta\n";
  std::set<PairKey> pairs;
  for (const auto& [pair, count] : reference_pairs) {
    (void)count;
    pairs.insert(pair);
  }
  for (const auto& [pair, count] : candidate_pairs) {
    (void)count;
    pairs.insert(pair);
  }
  for (const PairKey& pair : pairs) {
    const std::size_t ref_count =
        reference_pairs.count(pair) == 0 ? 0 : reference_pairs.at(pair);
    const std::size_t candidate_count =
        candidate_pairs.count(pair) == 0 ? 0 : candidate_pairs.at(pair);
    file << pair.image_name1 << "\t" << pair.image_name2 << "\t" << ref_count
         << "\t" << candidate_count << "\t"
         << (static_cast<long long>(candidate_count) -
             static_cast<long long>(ref_count))
         << "\n";
  }
}

void WritePointDiff(const std::filesystem::path& output_path,
                    const ColmapModel& reference,
                    const ColmapModel& candidate) {
  std::ofstream file(output_path / "point_diff.tsv");
  file << "point3d_id\tstatus\txyz_delta\terror_delta\tref_track_length"
          "\tcandidate_track_length\ttrack_length_delta\tref_error"
          "\tcandidate_error\tref_x\tref_y\tref_z\tcandidate_x"
          "\tcandidate_y\tcandidate_z\n";
  std::set<std::size_t> point_ids;
  for (const auto& [point_id, point] : reference.points3d) {
    (void)point;
    point_ids.insert(point_id);
  }
  for (const auto& [point_id, point] : candidate.points3d) {
    (void)point;
    point_ids.insert(point_id);
  }
  for (const std::size_t point_id : point_ids) {
    const auto ref_it = reference.points3d.find(point_id);
    const auto cand_it = candidate.points3d.find(point_id);
    const bool has_ref = ref_it != reference.points3d.end();
    const bool has_candidate = cand_it != candidate.points3d.end();
    const ColmapPoint3D* ref = has_ref ? &ref_it->second : nullptr;
    const ColmapPoint3D* cand = has_candidate ? &cand_it->second : nullptr;
    const double xyz_delta =
        ref != nullptr && cand != nullptr ? (cand->xyz - ref->xyz).norm() : -1.0;
    const double error_delta =
        ref != nullptr && cand != nullptr ? cand->error - ref->error : 0.0;
    const long long ref_track_length =
        ref == nullptr ? 0LL : static_cast<long long>(ref->track.size());
    const long long cand_track_length =
        cand == nullptr ? 0LL : static_cast<long long>(cand->track.size());
    file << std::setprecision(12) << point_id << "\t"
         << StatusForPresence(has_ref, has_candidate) << "\t" << xyz_delta
         << "\t" << error_delta << "\t" << ref_track_length << "\t"
         << cand_track_length << "\t"
         << (cand_track_length - ref_track_length) << "\t"
         << (ref == nullptr ? -1.0 : ref->error) << "\t"
         << (cand == nullptr ? -1.0 : cand->error) << "\t"
         << (ref == nullptr ? 0.0 : ref->xyz.x()) << "\t"
         << (ref == nullptr ? 0.0 : ref->xyz.y()) << "\t"
         << (ref == nullptr ? 0.0 : ref->xyz.z()) << "\t"
         << (cand == nullptr ? 0.0 : cand->xyz.x()) << "\t"
         << (cand == nullptr ? 0.0 : cand->xyz.y()) << "\t"
         << (cand == nullptr ? 0.0 : cand->xyz.z()) << "\n";
  }
}

void WritePoseDiff(const std::filesystem::path& output_path,
                   const ColmapModel& reference,
                   const ColmapModel& candidate) {
  std::ofstream file(output_path / "pose_diff.tsv");
  file << "image_name\tref_registered\tcandidate_registered"
          "\trotation_angle_deg\ttranslation_delta_norm\tcenter_delta_norm"
          "\tref_tx\tref_ty\tref_tz\tcandidate_tx\tcandidate_ty\tcandidate_tz"
          "\tref_center_x\tref_center_y\tref_center_z\tcandidate_center_x"
          "\tcandidate_center_y\tcandidate_center_z\n";
  std::set<std::string> image_names;
  for (const auto& [name, image_id] : reference.image_ids_by_name) {
    (void)image_id;
    image_names.insert(name);
  }
  for (const auto& [name, image_id] : candidate.image_ids_by_name) {
    (void)image_id;
    image_names.insert(name);
  }
  for (const std::string& name : image_names) {
    const auto ref_id_it = reference.image_ids_by_name.find(name);
    const auto cand_id_it = candidate.image_ids_by_name.find(name);
    const ColmapImage* ref =
        ref_id_it == reference.image_ids_by_name.end()
            ? nullptr
            : &reference.images_by_id.at(ref_id_it->second);
    const ColmapImage* cand =
        cand_id_it == candidate.image_ids_by_name.end()
            ? nullptr
            : &candidate.images_by_id.at(cand_id_it->second);
    const bool common = ref != nullptr && cand != nullptr;
    const Eigen::Vector3d ref_center =
        ref == nullptr ? Eigen::Vector3d::Zero() : CameraCenterFromPose(*ref);
    const Eigen::Vector3d cand_center =
        cand == nullptr ? Eigen::Vector3d::Zero() : CameraCenterFromPose(*cand);
    file << std::setprecision(12) << name << "\t" << (ref != nullptr ? 1 : 0)
         << "\t" << (cand != nullptr ? 1 : 0) << "\t"
         << (common ? RotationAngleDeg(ref->q_cw, cand->q_cw) : -1.0)
         << "\t" << (common ? (cand->t_cw - ref->t_cw).norm() : -1.0)
         << "\t" << (common ? (cand_center - ref_center).norm() : -1.0)
         << "\t" << (ref == nullptr ? 0.0 : ref->t_cw.x()) << "\t"
         << (ref == nullptr ? 0.0 : ref->t_cw.y()) << "\t"
         << (ref == nullptr ? 0.0 : ref->t_cw.z()) << "\t"
         << (cand == nullptr ? 0.0 : cand->t_cw.x()) << "\t"
         << (cand == nullptr ? 0.0 : cand->t_cw.y()) << "\t"
         << (cand == nullptr ? 0.0 : cand->t_cw.z()) << "\t"
         << ref_center.x() << "\t" << ref_center.y() << "\t"
         << ref_center.z() << "\t" << cand_center.x() << "\t"
         << cand_center.y() << "\t" << cand_center.z() << "\n";
  }
}

ColmapCompareControllerSummary MakeSummary(
    const ColmapModel& reference,
    const ColmapModel& candidate,
    const std::map<PairKey, std::size_t>& reference_pairs,
    const std::map<PairKey, std::size_t>& candidate_pairs) {
  ColmapCompareControllerSummary summary;
  summary.reference_images = reference.images_by_id.size();
  summary.candidate_images = candidate.images_by_id.size();
  for (const auto& [name, image_id] : reference.image_ids_by_name) {
    (void)image_id;
    if (candidate.image_ids_by_name.count(name) > 0) {
      ++summary.common_images;
    } else {
      ++summary.reference_only_images;
    }
  }
  for (const auto& [name, image_id] : candidate.image_ids_by_name) {
    (void)image_id;
    if (reference.image_ids_by_name.count(name) == 0) {
      ++summary.candidate_only_images;
    }
  }

  summary.reference_points = reference.points3d.size();
  summary.candidate_points = candidate.points3d.size();
  for (const auto& [point_id, point] : reference.points3d) {
    (void)point;
    if (candidate.points3d.count(point_id) > 0) {
      ++summary.common_points_by_id;
    } else {
      ++summary.reference_only_points;
    }
  }
  for (const auto& [point_id, point] : candidate.points3d) {
    (void)point;
    if (reference.points3d.count(point_id) == 0) {
      ++summary.candidate_only_points;
    }
  }

  for (const auto& [pair, count] : reference_pairs) {
    (void)count;
    if (candidate_pairs.count(pair) > 0) {
      ++summary.common_pairs;
    }
  }
  return summary;
}

}  // namespace

ColmapCompareController::ColmapCompareController(
    ColmapCompareControllerOptions options)
    : options_(std::move(options)) {}

ColmapCompareControllerSummary ColmapCompareController::Run() const {
  if (options_.reference_path.empty() || options_.candidate_path.empty() ||
      options_.output_path.empty()) {
    throw std::invalid_argument(
        "reference_path, candidate_path, and output_path are required");
  }

  const ColmapModel reference = ReadColmapModel(options_.reference_path);
  const ColmapModel candidate = ReadColmapModel(options_.candidate_path);
  const std::map<PairKey, std::size_t> reference_pairs =
      ComputeSharedPointPairs(reference);
  const std::map<PairKey, std::size_t> candidate_pairs =
      ComputeSharedPointPairs(candidate);
  const ColmapCompareControllerSummary summary =
      MakeSummary(reference, candidate, reference_pairs, candidate_pairs);

  std::filesystem::create_directories(options_.output_path);
  WriteSummary(options_.output_path, summary);
  WriteImageDiff(options_.output_path, reference, candidate);
  WritePairDiff(options_.output_path, reference_pairs, candidate_pairs);
  WritePointDiff(options_.output_path, reference, candidate);
  WritePoseDiff(options_.output_path, reference, candidate);
  return summary;
}

}  // namespace colite::controllers
