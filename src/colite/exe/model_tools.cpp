#include "colite/exe/model_tools.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string_view>

#include <Eigen/Geometry>
#include <glog/logging.h>

#include "colite/core/geometry.h"
#include "colite/io/sfm_workspace.h"
#include "colite/exe/sfm_options.h"
#include "colite/sfm/global_bundle_adjuster.h"
#include "colite/sfm/image_registrar.h"
#include "colite/sfm/incremental_triangulator.h"
#include "colite/sfm/similarity_transform.h"

namespace colite::exe {
namespace {

struct Options {
  bool help = false;
  std::filesystem::path input;
  std::filesystem::path output;
  ColmapModelFormat input_format = ColmapModelFormat::kAuto;
  ColmapModelFormat output_format = ColmapModelFormat::kTextAndBinary;
  std::map<std::string, std::string> values;
};

std::string Value(const std::vector<std::string>& args, std::size_t* idx) {
  if (*idx + 1 >= args.size()) throw std::invalid_argument("Missing option value");
  return args[++(*idx)];
}

Options Parse(const std::vector<std::string>& raw_args,
              const std::vector<std::string>& extras,
              const std::string& mapper_prefix = "", bool converter = false) {
  const auto args = NormalizeSfmArguments(raw_args, SfmOptionScope::kModel);
  Options options;
  for (std::size_t idx = 1; idx < args.size(); ++idx) {
    std::string arg = args[idx];
    if (arg == "--help" || arg == "-h") options.help = true;
    else if (arg == "--input_path") options.input = Value(args, &idx);
    else if (arg == "--output_path") options.output = Value(args, &idx);
    else if (arg == "--Colite.input_format") {
      options.input_format = ColmapModelFormatFromString(Value(args, &idx));
    } else if ((converter && arg == "--output_type") || (!converter && arg == "--Colite.output_format")) {
      options.output_format = ColmapModelFormatFromString(Value(args, &idx));
    } else if (!mapper_prefix.empty() && arg.rfind(mapper_prefix, 0) == 0) {
      const std::string value = Value(args, &idx);
      IncrementalMappingOptions mapping;
      if (!controllers::ApplySfmMapperOption(arg, value, &mapping)) {
        throw std::invalid_argument("Unsupported mapper option: " + arg);
      }
      if (!options.values.emplace(arg, value).second) {
        throw std::invalid_argument("Duplicate option: " + arg);
      }
    } else if (std::find(extras.begin(), extras.end(), arg) != extras.end()) {
      options.values[arg] = Value(args, &idx);
    } else throw std::invalid_argument("Unknown argument: " + arg);
  }
  return options;
}

IncrementalMappingOptions ModelMappingOptions(const Options& options) {
  IncrementalMappingOptions mapping;
  for (const auto& [name, value] : options.values) {
    if (name.rfind("--Mapper.", 0) == 0) {
      controllers::ApplySfmMapperOption(name, value, &mapping);
    }
  }
  return mapping;
}

void Validate(const Options& options, bool output) {
  if (options.input.empty()) throw std::invalid_argument("Missing --input_path");
  if (output && options.output.empty()) throw std::invalid_argument("Missing --output_path");
  if (options.input_format == ColmapModelFormat::kTextAndBinary) {
    throw std::invalid_argument("--Colite.input_format does not accept both");
  }
  if (options.output_format == ColmapModelFormat::kAuto) {
    throw std::invalid_argument("--output_type does not accept auto");
  }
}

bool Bool(const std::string& text) {
  if (text == "1" || text == "true") return true;
  if (text == "0" || text == "false") return false;
  throw std::invalid_argument("Expected 0/1");
}

Eigen::Vector3d Vector3(std::string text) {
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream stream(text);
  Eigen::Vector3d vector;
  std::string trailing;
  if (!(stream >> vector.x() >> vector.y() >> vector.z()) ||
      (stream >> trailing) || !vector.allFinite()) {
    throw std::invalid_argument("Expected x,y,z");
  }
  return vector;
}

std::map<std::string, Eigen::Vector3d> References(
    const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Failed to read reference positions");
  std::map<std::string, Eigen::Vector3d> result;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::istringstream stream(line);
    std::string name;
    Eigen::Vector3d xyz;
    if (!(stream >> name >> xyz.x() >> xyz.y() >> xyz.z()) || !xyz.allFinite()) {
      throw std::runtime_error("Invalid reference line: " + line);
    }
    result[name] = xyz;
  }
  return result;
}

std::size_t SetPriors(const std::map<std::string, Eigen::Vector3d>& priors,
                      Reconstruction* model) {
  std::size_t count = 0;
  for (const auto& [id, image] : model->Images()) {
    const auto it = priors.find(image.name);
    if (it != priors.end()) {
      model->GetImage(id).prior_position = it->second;
      ++count;
    }
  }
  return count;
}

SimilarityTransform3 Transform(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::vector<double> values;
  double value;
  while (file >> value) values.push_back(value);
  if (values.size() != 12 && values.size() != 16) {
    throw std::runtime_error("Transform must be a 3x4 or 4x4 matrix");
  }
  Eigen::Matrix3d scaled_rotation;
  Eigen::Vector3d translation;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) scaled_rotation(row, col) = values[row * 4 + col];
    translation[row] = values[row * 4 + 3];
  }
  const double scale = (scaled_rotation.col(0).norm() +
                        scaled_rotation.col(1).norm() +
                        scaled_rotation.col(2).norm()) / 3.0;
  if (!(scale > 1e-12)) throw std::runtime_error("Invalid transform scale");
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      scaled_rotation / scale, Eigen::ComputeFullU | Eigen::ComputeFullV);
  SimilarityTransform3 transform;
  transform.scale = scale;
  transform.rotation = svd.matrixU() * svd.matrixV().transpose();
  transform.translation = translation;
  transform.valid = transform.rotation.determinant() > 0.0;
  if (!transform.valid) throw std::runtime_error("Transform is not a proper Sim(3)");
  return transform;
}

int Fail(const std::exception& exception, std::ostream& err) {
  err << "Error: " << exception.what() << "\n";
  LOG(ERROR) << exception.what();
  return 1;
}

}  // namespace

int RunModelConverter(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {}, "", true);
    if (options.help) {
      out << "Usage: model_converter --input_path <model> --output_path <dir> "
             "[--Colite.input_format auto|txt|bin] [--output_type txt|bin|both]\n";
      return 0;
    }
    Validate(options, true);
    const Reconstruction model =
        ReadColmapReconstruction(options.input, options.input_format);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "cameras " << model.Cameras().size() << "\nimages "
        << model.RegisteredImageIds().size() << "\npoints3d "
        << model.Points3D().size() << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunModelAnalyzer(const std::vector<std::string>& args,
                     std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {});
    if (options.help) { out << "Usage: model_analyzer --input_path <model> [--Colite.input_format auto|txt|bin]\n"; return 0; }
    Validate(options, false);
    LOG(INFO) << "Model analyzer: input=" << options.input.string();
    const Reconstruction model = ReadColmapReconstruction(options.input, options.input_format);
    std::size_t observations = 0, errors = 0;
    double error_sum = 0.0;
    Eigen::Vector3d minimum = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d maximum = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
    for (const auto& [id, point] : model.Points3D()) {
      (void)id;
      observations += point.track.Length();
      minimum = minimum.cwiseMin(point.xyz);
      maximum = maximum.cwiseMax(point.xyz);
      for (const TrackElement& element : point.track.elements) {
        const Image& image = model.GetImage(element.image_id);
        if (!image.HasPose() || element.point2d_idx >= image.points2d.size()) continue;
        const auto error = ReprojectionError(model.GetCamera(image.camera_id),
            *image.cam_from_world, point.xyz, image.points2d[element.point2d_idx].xy);
        if (error) { error_sum += *error; ++errors; }
      }
    }
    out << std::setprecision(17) << "cameras " << model.Cameras().size() << "\n"
        << "images " << model.Images().size() << "\nregistered_images "
        << model.RegisteredImageIds().size() << "\npoints3d " << model.Points3D().size()
        << "\nobservations " << observations << "\nmean_track_length "
        << (model.Points3D().empty() ? 0.0 : static_cast<double>(observations) / model.Points3D().size())
        << "\nmean_reprojection_error " << (errors ? error_sum / errors : -1.0) << "\n";
    if (!model.Points3D().empty()) out << "bounds_min " << minimum.transpose() << "\nbounds_max " << maximum.transpose() << "\n";
    LOG(INFO) << "Model analyzer complete: registered_images="
              << model.RegisteredImageIds().size()
              << " points3d=" << model.Points3D().size();
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunModelTransformer(const std::vector<std::string>& args,
                        std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--transform_path"});
    if (options.help) { out << "Usage: model_transformer --input_path <model> --output_path <dir> --transform_path <matrix.txt>\n"; return 0; }
    Validate(options, true);
    if (!options.values.count("--transform_path")) throw std::invalid_argument("Missing --transform_path");
    LOG(INFO) << "Model transformer: input=" << options.input.string()
              << " output=" << options.output.string();
    Reconstruction model = ReadColmapReconstruction(options.input, options.input_format);
    const SimilarityTransform3 transform = Transform(options.values.at("--transform_path"));
    TransformReconstruction(transform, &model);
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "scale " << transform.scale << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunModelAligner(const std::vector<std::string>& args,
                    std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--ref_images_path", "--max_error"});
    if (options.help) { out << "Usage: model_aligner --input_path <model> --output_path <dir> --ref_images_path <name-x-y-z.txt> [--max_error 20]\n"; return 0; }
    Validate(options, true);
    if (!options.values.count("--ref_images_path")) throw std::invalid_argument("Missing --ref_images_path");
    LOG(INFO) << "Model aligner: input=" << options.input.string()
              << " output=" << options.output.string();
    Reconstruction model = ReadColmapReconstruction(options.input, options.input_format);
    const std::size_t assigned = SetPriors(References(options.values.at("--ref_images_path")), &model);
    const double threshold = options.values.count("--max_error") ? controllers::SfmDouble(options.values.at("--max_error")) : 20.0;
    const auto summary = AlignReconstructionToPositionPriors(threshold, &model);
    if (!summary.success) throw std::runtime_error("Similarity alignment failed");
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "assigned_positions " << assigned << "\nalignment_inliers " << summary.num_inliers
        << "\nscale " << summary.scale << "\nmean_error " << summary.mean_error << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunModelCropper(const std::vector<std::string>& args,
                    std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--min", "--max"});
    if (options.help) { out << "Usage: model_cropper --input_path <model> --output_path <dir> --min x,y,z --max x,y,z\n"; return 0; }
    Validate(options, true);
    if (!options.values.count("--min") || !options.values.count("--max")) throw std::invalid_argument("Missing crop bounds");
    LOG(INFO) << "Model cropper: input=" << options.input.string()
              << " output=" << options.output.string();
    const Eigen::Vector3d minimum = Vector3(options.values.at("--min"));
    const Eigen::Vector3d maximum = Vector3(options.values.at("--max"));
    if ((minimum.array() > maximum.array()).any()) throw std::invalid_argument("Invalid crop bounds");
    Reconstruction model = ReadColmapReconstruction(options.input, options.input_format);
    std::vector<Point3DId> removed;
    for (const auto& [id, point] : model.Points3D()) {
      if ((point.xyz.array() < minimum.array()).any() ||
          (point.xyz.array() > maximum.array()).any()) removed.push_back(id);
    }
    for (const Point3DId id : removed) model.DeletePoint3D(id);
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "removed_points " << removed.size() << "\nremaining_points " << model.Points3D().size() << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunBundleAdjuster(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--BundleAdjustmentCeres.max_num_iterations", "--BundleAdjustment.refine_focal_length",
        "--BundleAdjustment.refine_principal_point", "--BundleAdjustment.refine_extra_params", "--ref_images_path",
        "--Colite.position_prior_stddev", "--Colite.position_prior_loss_scale", "--alignment_max_error"});
    if (options.help) {
      out << "Usage: bundle_adjuster --input_path <model> --output_path <dir> [--BundleAdjustmentCeres.max_num_iterations 50] [--ref_images_path positions.txt]\n";
      PrintSfmOptionsHelp(out, SfmOptionScope::kModel);
      return 0;
    }
    Validate(options, true);
    LOG(INFO) << "Bundle adjuster: input=" << options.input.string()
              << " output=" << options.output.string();
    Reconstruction model = ReadColmapReconstruction(options.input, options.input_format);
    GlobalBundleAdjustmentOptions ba;
    if (options.values.count("--BundleAdjustmentCeres.max_num_iterations")) ba.max_num_iterations = controllers::SfmInt(options.values.at("--BundleAdjustmentCeres.max_num_iterations"));
    if (!options.values.count("--BundleAdjustment.refine_focal_length") || Bool(options.values.at("--BundleAdjustment.refine_focal_length"))) ba.refine_camera_params = ba.refine_camera_params | CameraParameterGroup::kFocalLength;
    if (options.values.count("--BundleAdjustment.refine_principal_point") && Bool(options.values.at("--BundleAdjustment.refine_principal_point"))) ba.refine_camera_params = ba.refine_camera_params | CameraParameterGroup::kPrincipalPoint;
    if (!options.values.count("--BundleAdjustment.refine_extra_params") || Bool(options.values.at("--BundleAdjustment.refine_extra_params"))) ba.refine_camera_params = ba.refine_camera_params | CameraParameterGroup::kDistortion;
    if (options.values.count("--ref_images_path")) {
      SetPriors(References(options.values.at("--ref_images_path")), &model);
      const double threshold = options.values.count("--alignment_max_error") ? controllers::SfmDouble(options.values.at("--alignment_max_error")) : 20.0;
      if (!AlignReconstructionToPositionPriors(threshold, &model).success) throw std::runtime_error("Position-prior alignment failed");
      ba.use_position_priors = true;
      ba.gauge_fixing = BundleGaugeFixing::kNone;
      if (options.values.count("--Colite.position_prior_stddev")) ba.position_prior_stddev = controllers::SfmDouble(options.values.at("--Colite.position_prior_stddev"));
      if (options.values.count("--Colite.position_prior_loss_scale")) ba.position_prior_loss_scale = controllers::SfmDouble(options.values.at("--Colite.position_prior_loss_scale"));
    }
    const GlobalBundleAdjustmentSummary summary = AdjustGlobalBundle(ba, &model);
    if (!summary.success) throw std::runtime_error(summary.message);
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "initial_mean_reprojection_error " << summary.initial_mean_reprojection_error
        << "\nfinal_mean_reprojection_error " << summary.final_mean_reprojection_error
        << "\nposition_priors " << summary.num_position_priors
        << "\ninitial_mean_position_error " << summary.initial_mean_position_error
        << "\nfinal_mean_position_error " << summary.final_mean_position_error << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunPointTriangulator(const std::vector<std::string>& args,
                         std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--Colite.workspace_path"}, "--Mapper.tri_");
    if (options.help) {
      out << "Usage: point_triangulator --Colite.workspace_path <workspace> --input_path <model> --output_path <dir>\n"
          << "Triangulation: --Mapper.tri_max_transitivity, --Mapper.tri_create_max_angle_error,\n"
          << "  --Mapper.tri_continue_max_angle_error, --Mapper.tri_merge_max_reproj_error,\n"
          << "  --Mapper.tri_complete_max_reproj_error, --Mapper.tri_complete_max_transitivity,\n"
          << "  --Mapper.tri_min_angle, --Mapper.tri_ignore_two_view_tracks\n";
      return 0;
    }
    Validate(options, true);
    if (!options.values.count("--Colite.workspace_path")) throw std::invalid_argument("Missing --Colite.workspace_path");
    LOG(INFO) << "Point triangulator: workspace="
              << options.values.at("--Colite.workspace_path")
              << " input=" << options.input.string();
    const std::filesystem::path workspace = options.values.at("--Colite.workspace_path");
    Reconstruction model = ReadWorkspaceReconstruction(workspace);
    model = ApplyColmapModel(model, options.input, options.input_format);
    MatchGraph graph;
    ReadWorkspaceFeatureSets(workspace, &graph);
    ReadWorkspaceVerifiedGeometries(workspace, &graph);
    const IncrementalTriangulatorCache cache(graph);
    IncrementalTriangulatorOptions triangulation =
        ModelMappingOptions(options).mapper.triangulation;
    triangulation.complete_neighbor_observations = true;
    IncrementalTriangulatorSummary total;
    for (const ImageId image_id : model.RegisteredImageIds()) {
      const auto summary = TriangulateRegisteredImage(
          graph, image_id, triangulation, &model, &cache);
      total.num_created_points += summary.num_created_points;
      total.num_created_observations += summary.num_created_observations;
      total.num_completed_observations += summary.num_completed_observations;
      total.num_merged_points += summary.num_merged_points;
    }
    const auto completed = CompleteAndMergeAllTracks(
        graph, triangulation, &model, &cache);
    total.num_completed_observations += completed.num_completed_observations;
    total.num_merged_points += completed.num_merged_points;
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "created_points " << total.num_created_points
        << "\ncreated_observations " << total.num_created_observations
        << "\ncompleted_observations " << total.num_completed_observations
        << "\nmerged_points " << total.num_merged_points
        << "\npoints3d " << model.Points3D().size() << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunImageRegistrator(const std::vector<std::string>& args,
                        std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(
        args, {"--Colite.workspace_path", "--image_id"}, "--Mapper.abs_pose_");
    if (options.help) {
      out << "Usage: image_registrator --Colite.workspace_path <workspace> --input_path <model> --output_path <dir> [--image_id id]\n"
          << "PnP: --Mapper.abs_pose_max_error, --Mapper.abs_pose_min_num_inliers,\n"
          << "  --Mapper.abs_pose_min_inlier_ratio\n"
          ;
      return 0;
    }
    Validate(options, true);
    if (!options.values.count("--Colite.workspace_path")) throw std::invalid_argument("Missing --Colite.workspace_path");
    std::vector<ImageId> candidates;
    if (options.values.count("--image_id")) {
      const auto image_id = controllers::SfmCount(options.values.at("--image_id"));
      if (image_id >= std::numeric_limits<ImageId>::max()) {
        throw std::invalid_argument("--image_id exceeds the valid image ID range");
      }
      candidates.push_back(static_cast<ImageId>(image_id));
    }
    LOG(INFO) << "Image registrator: workspace="
              << options.values.at("--Colite.workspace_path")
              << " input=" << options.input.string();
    const std::filesystem::path workspace = options.values.at("--Colite.workspace_path");
    Reconstruction model = ReadWorkspaceReconstruction(workspace);
    model = ApplyColmapModel(model, options.input, options.input_format);
    MatchGraph graph;
    ReadWorkspaceFeatureSets(workspace, &graph);
    ReadWorkspaceVerifiedGeometries(workspace, &graph);
    const ImageRegistrarOptions registration =
        ModelMappingOptions(options).mapper.registration;
    if (candidates.empty()) {
      for (const auto& [image_id, image] : model.Images()) {
        if (!image.registered) candidates.push_back(image_id);
      }
      std::sort(candidates.begin(), candidates.end());
    }
    std::size_t registered = 0;
    std::size_t inliers = 0;
    for (const ImageId image_id : candidates) {
      if (!model.ExistsImage(image_id) || model.IsImageRegistered(image_id)) continue;
      const ImageRegistrationSummary summary = RegisterImageWithExistingPoints(
          graph, image_id, registration, &model);
      if (summary.success) { ++registered; inliers += summary.num_inliers; }
    }
    UpdatePoint3DReprojectionErrors(&model);
    WriteColmapReconstruction(options.output, model, options.output_format);
    out << "attempted_images " << candidates.size() << "\nregistered_images "
        << registered << "\ninliers " << inliers << "\ntotal_registered_images "
        << model.RegisteredImageIds().size() << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

int RunModelSplitter(const std::vector<std::string>& args,
                     std::ostream& out, std::ostream& err) {
  try {
    const Options options = Parse(args, {"--Mapper.min_model_size"});
    if (options.help) { out << "Usage: model_splitter --input_path <model> --output_path <dir> [--Mapper.min_model_size 2]\n"; return 0; }
    Validate(options, true);
    LOG(INFO) << "Model splitter: input=" << options.input.string()
              << " output=" << options.output.string();
    const std::size_t min_size = options.values.count("--Mapper.min_model_size")
                                     ? controllers::SfmCount(options.values.at("--Mapper.min_model_size")) : 2;
    const Reconstruction source = ReadColmapReconstruction(options.input, options.input_format);
    std::map<ImageId, std::set<ImageId>> adjacency;
    for (const ImageId image_id : source.RegisteredImageIds()) adjacency[image_id];
    for (const auto& [point_id, point] : source.Points3D()) {
      (void)point_id;
      for (const TrackElement& lhs : point.track.elements) {
        for (const TrackElement& rhs : point.track.elements) {
          if (lhs.image_id != rhs.image_id) adjacency[lhs.image_id].insert(rhs.image_id);
        }
      }
    }
    std::set<ImageId> visited;
    std::vector<std::set<ImageId>> components;
    for (const auto& [start, neighbors] : adjacency) {
      (void)neighbors;
      if (!visited.insert(start).second) continue;
      std::set<ImageId> component;
      std::vector<ImageId> stack{start};
      while (!stack.empty()) {
        const ImageId image_id = stack.back(); stack.pop_back();
        component.insert(image_id);
        for (const ImageId neighbor : adjacency[image_id]) {
          if (visited.insert(neighbor).second) stack.push_back(neighbor);
        }
      }
      if (component.size() >= min_size) components.push_back(std::move(component));
    }
    std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.size() > rhs.size();
    });
    std::filesystem::create_directories(options.output);
    std::ofstream manifest(options.output / "models.txt");
    manifest << "# MODEL_INDEX REGISTERED_IMAGES POINTS3D PATH\n";
    for (std::size_t component_idx = 0; component_idx < components.size(); ++component_idx) {
      const auto& component = components[component_idx];
      Reconstruction model;
      std::set<CameraId> camera_ids;
      for (const ImageId image_id : component) camera_ids.insert(source.GetImage(image_id).camera_id);
      for (const CameraId camera_id : camera_ids) model.AddCamera(source.GetCamera(camera_id));
      for (const ImageId image_id : component) {
        Image image = source.GetImage(image_id);
        for (Point2D& point : image.points2d) point.point3d_id = kInvalidPoint3DId;
        model.AddImage(std::move(image));
      }
      for (const auto& [point_id, source_point] : source.Points3D()) {
        Point3D point = source_point;
        point.track.elements.erase(
            std::remove_if(point.track.elements.begin(), point.track.elements.end(),
                           [&](const TrackElement& element) {
                             return component.count(element.image_id) == 0;
                           }), point.track.elements.end());
        if (point.track.Length() >= 2) model.AddPoint3D(std::move(point));
      }
      const std::filesystem::path path = options.output / std::to_string(component_idx);
      UpdatePoint3DReprojectionErrors(&model);
      WriteColmapReconstruction(path, model, options.output_format);
      manifest << component_idx << " " << model.RegisteredImageIds().size()
               << " " << model.Points3D().size() << " " << component_idx << "\n";
    }
    out << "models " << components.size() << "\n";
    return 0;
  } catch (const std::exception& e) { return Fail(e, err); }
}

}  // namespace colite::exe
