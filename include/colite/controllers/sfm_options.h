#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

inline IncrementalMappingOptions ColmapSfmMappingDefaults() {
  IncrementalMappingOptions options;
  options.initialization.min_num_inliers = 100;
  options.initialization.min_triangulation_angle_deg = 16.0;
  options.initialization.max_num_initialization_trials = 200;
  options.relax_initialization_constraints = true;
  options.mapper.registration.max_reprojection_error_px = 12.0;
  options.mapper.registration.min_num_inliers = 30;
  options.mapper.registration.min_num_correspondences = 30;
  options.mapper.registration.min_inlier_ratio = 0.25;
  options.mapper.use_structureless_registration = true;
  options.mapper.observation_filter.min_triangulation_angle_deg = 1.5;
  options.mapper.local_retriangulation_filter.min_triangulation_angle_deg = 1.5;
  options.global_observation_filter.min_triangulation_angle_deg = 1.5;
  options.registration_retry_reprojection_error_multiplier = 1.0;
  options.registration_retry_max_reprojection_error_px = 12.0;
  return options;
}

inline double SfmDouble(const std::string& value) {
  std::size_t consumed = 0;
  const double number = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(number)) {
    throw std::invalid_argument("Expected a finite number: " + value);
  }
  return number;
}

inline std::size_t SfmCount(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a non-negative integer: " + value);
  }
  const auto number = std::stoull(value);
  if (number > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("Integer is too large: " + value);
  }
  return static_cast<std::size_t>(number);
}

inline int SfmInt(const std::string& value) {
  std::size_t consumed = 0;
  const int number = std::stoi(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("Expected an integer: " + value);
  }
  return number;
}

inline bool SfmBool(const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE") return true;
  if (value == "0" || value == "false" || value == "FALSE") return false;
  throw std::invalid_argument("Expected 0/1 or true/false: " + value);
}

// This registry contains only controls that are actually implemented. Similar
// names with different semantics (e.g. init_max_reg_trials) are not aliases.
inline bool ApplySfmMapperOption(const std::string& name,
                                 const std::string& value,
                                 IncrementalMappingOptions* options) {
  auto positive = [&]() {
    const double number = SfmDouble(value);
    if (number <= 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto angle = [&]() {
    const double number = SfmDouble(value);
    if (number < 0 || number > 180) throw std::invalid_argument(name + " must be in [0, 180] degrees");
    return number;
  };
  auto count = [&]() {
    const auto number = SfmCount(value);
    if (number == 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto iterations = [&]() {
    const int number = SfmInt(value);
    if (number <= 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto& mapper = options->mapper;
  auto& tri = mapper.triangulation;
  if (name == "--Mapper.init_min_num_inliers") options->initialization.min_num_inliers = count();
  else if (name == "--Mapper.init_max_error") options->initialization.max_reprojection_error_px = positive();
  else if (name == "--Mapper.init_min_tri_angle") options->initialization.min_triangulation_angle_deg = angle();
  else if (name == "--Mapper.abs_pose_max_error") mapper.registration.max_reprojection_error_px = positive();
  else if (name == "--Mapper.abs_pose_min_num_inliers") {
    mapper.registration.min_num_inliers = count();
    mapper.registration.min_num_correspondences =
        std::max<std::size_t>(6, mapper.registration.min_num_inliers);
  }
  else if (name == "--Mapper.abs_pose_min_inlier_ratio") {
    const double number = SfmDouble(value);
    if (number < 0 || number > 1) throw std::invalid_argument(name + " must be in [0, 1]");
    mapper.registration.min_inlier_ratio = number;
  }
  else if (name == "--Mapper.max_reg_trials") options->max_num_registration_trials_per_image = count();
  else if (name == "--Mapper.init_num_trials") options->initialization.max_num_initialization_trials = count();
  else if (name == "--Mapper.structure_less_registration_fallback") mapper.use_structureless_registration = SfmBool(value);
  else if (name == "--Colite.registration_retry_error_multiplier") {
    const double number = positive();
    if (number < 1) throw std::invalid_argument(name + " must be >= 1");
    options->registration_retry_reprojection_error_multiplier = number;
  } else if (name == "--Colite.registration_retry_max_error") options->registration_retry_max_reprojection_error_px = positive();
  else if (name == "--Mapper.filter_max_reproj_error") {
    mapper.observation_filter.max_reprojection_error_px = positive();
    mapper.local_retriangulation_filter.max_reprojection_error_px = positive();
    options->global_observation_filter.max_reprojection_error_px = positive();
  } else if (name == "--Mapper.filter_min_tri_angle") {
    mapper.observation_filter.min_triangulation_angle_deg = angle();
    mapper.local_retriangulation_filter.min_triangulation_angle_deg = angle();
    options->global_observation_filter.min_triangulation_angle_deg = angle();
  } else if (name == "--Mapper.ba_local_num_images") {
    const auto number = count();
    if (number < 2) throw std::invalid_argument(name + " must be >= 2");
    mapper.local_bundle_adjustment.max_num_images = number;
  } else if (name == "--Mapper.ba_local_max_num_iterations") mapper.local_bundle_adjustment.max_num_iterations = iterations();
  else if (name == "--Mapper.ba_global_max_num_iterations") options->global_bundle_adjustment.max_num_iterations = iterations();
  else if (name == "--Mapper.tri_max_transitivity") tri.max_transitivity = count();
  else if (name == "--Mapper.tri_create_max_angle_error") tri.create_max_angle_error_deg = angle();
  else if (name == "--Mapper.tri_continue_max_angle_error") tri.continue_max_angle_error_deg = angle();
  else if (name == "--Mapper.tri_merge_max_reproj_error") tri.merge_max_reprojection_error_px = positive();
  else if (name == "--Mapper.tri_complete_max_reproj_error") tri.complete_max_reprojection_error_px = positive();
  else if (name == "--Mapper.tri_complete_max_transitivity") tri.complete_max_transitivity = count();
  else if (name == "--Mapper.tri_min_angle") tri.min_triangulation_angle_deg = angle();
  else if (name == "--Mapper.tri_ignore_two_view_tracks") tri.ignore_two_view_tracks = SfmBool(value);
  else return false;
  return true;
}

inline void ApplySfmMapperOptions(const std::map<std::string, std::string>& values,
                                  IncrementalMappingOptions* options) {
  for (const auto& [name, value] : values) {
    if (!ApplySfmMapperOption(name, value, options)) {
      throw std::invalid_argument("Unsupported mapper option: " + name);
    }
  }
}

inline bool ApplySfmGeometryOption(const std::string& name,
                                   const std::string& value,
                                   TwoViewGeometryOptions* options) {
  if (name == "--TwoViewGeometry.max_error") {
    options->max_error_px = SfmDouble(value);
    if (options->max_error_px <= 0) throw std::invalid_argument(name + " must be positive");
  } else if (name == "--TwoViewGeometry.confidence") {
    options->confidence = SfmDouble(value);
    if (options->confidence <= 0 || options->confidence >= 1) {
      throw std::invalid_argument(name + " must be in (0, 1)");
    }
  } else if (name == "--TwoViewGeometry.min_num_inliers") {
    options->min_num_inliers = SfmInt(value);
    if (options->min_num_inliers < 8) throw std::invalid_argument(name + " must be >= 8");
  } else if (name == "--TwoViewGeometry.max_num_trials") {
    options->max_num_iterations = SfmInt(value);
    if (options->max_num_iterations < options->min_num_iterations) {
      throw std::invalid_argument(name + " must be >= 100");
    }
  } else return false;
  return true;
}

inline std::string SfmGeometryKey(const TwoViewGeometryOptions& options) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << options.max_error_px << ',' << options.confidence << ','
         << options.min_num_inliers << ',' << options.min_num_iterations << ','
         << options.max_num_iterations << ',' << options.min_E_F_inlier_ratio << ','
         << options.max_H_inlier_ratio << ','
         << options.max_panoramic_triangulation_angle_deg << ',' << options.estimate_homography;
  return stream.str();
}

inline std::string SfmNumberKey(double number) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << number;
  return stream.str();
}

}  // namespace colite::controllers
