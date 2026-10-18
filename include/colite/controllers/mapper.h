#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <set>
#include <string>

#include "colite/core/camera.h"
#include "colite/io/sfm_workspace.h"

namespace colite::controllers {

struct MapperControllerOptions {
  std::filesystem::path workspace_path;
  std::filesystem::path input_model_path;
  ColmapModelFormat input_model_format = ColmapModelFormat::kAuto;
  ColmapModelFormat output_model_format =
      ColmapModelFormat::kTextAndBinary;
  bool multiple_models = true;
  bool merge_models = true;
  std::size_t max_num_models = 50;
  std::size_t min_num_images_per_model = 10;
  CameraParameterGroup ba_refine_camera_params =
      CameraParameterGroup::kFocalLength | CameraParameterGroup::kDistortion;
  std::set<std::string> ba_refine_camera_param_names;
  std::map<std::string, std::string> mapper_options;
  bool use_position_priors = false;
  double position_prior_stddev = 5.0;
  double position_prior_loss_scale = 2.0;
  double position_prior_alignment_max_error = 20.0;
};

struct MapperControllerSummary {
  bool success = false;
  bool cache_hit = false;
  std::size_t num_registered_images = 0;
  std::size_t num_points3d = 0;
  std::size_t num_models = 0;
  std::size_t num_unregistered_images = 0;
  std::size_t num_position_prior_aligned_models = 0;
  double final_mean_position_error = -1.0;
};

class MapperController {
 public:
  explicit MapperController(MapperControllerOptions options);

  MapperControllerSummary Run() const;

 private:
  MapperControllerOptions options_;
};

}  // namespace colite::controllers
