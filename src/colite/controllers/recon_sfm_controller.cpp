#include "colite/controllers/recon_sfm_controller.h"

#include <algorithm>
#include <thread>

#include "colite/core/geometry.h"

namespace colite::controllers {

ReconSfmController::ReconSfmController(ReconSfmControllerOptions options)
    : options_(options) {}

const ReconSfmControllerOptions& ReconSfmController::Options() const {
  return options_;
}

int ReconSfmController::EffectiveFeatureExtractionThreads() const {
  return controllers::EffectiveFeatureExtractionThreads(
      options_.feature_num_threads);
}

void ReconSfmController::MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    MatchGraph* graph) const {
  NearbyPairMatchingOptions nearby_options;
  nearby_options.image_index_window = options_.nearby_match_window;
  nearby_options.max_ratio = options_.nearby_match_ratio;
  controllers::MatchAllImagePairs(image_ids, base_options, nearby_options,
                                  graph);
}

int EffectiveFeatureExtractionThreads(const int requested_threads) {
  if (requested_threads > 0) {
    return requested_threads;
  }
  const unsigned int hardware_threads = std::thread::hardware_concurrency();
  return hardware_threads == 0 ? 1 : static_cast<int>(hardware_threads);
}

ReconstructionQualityStats ComputeReconstructionQualityStats(
    const Reconstruction& reconstruction) {
  ReconstructionQualityStats stats;
  double total_error = 0.0;
  double max_error = 0.0;

  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    for (const TrackElement& element : point3d.track.elements) {
      if (!reconstruction.ExistsImage(element.image_id)) {
        continue;
      }
      const Image& image = reconstruction.GetImage(element.image_id);
      if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
          element.point2d_idx >= image.points2d.size()) {
        continue;
      }
      const Camera& camera = reconstruction.GetCamera(image.camera_id);
      const auto error =
          ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                            image.points2d[element.point2d_idx].xy);
      if (!error.has_value()) {
        continue;
      }
      total_error += *error;
      max_error = std::max(max_error, *error);
      ++stats.num_observations;
    }
  }

  if (stats.num_observations > 0) {
    stats.mean_reprojection_error =
        total_error / static_cast<double>(stats.num_observations);
    stats.max_reprojection_error = max_error;
  }
  return stats;
}

void MatchAllImagePairs(
    const std::vector<ImageId>& image_ids,
    const DescriptorMatchingOptions& base_options,
    const NearbyPairMatchingOptions& nearby_options,
    MatchGraph* graph) {
  DescriptorMatchingOptions relaxed_nearby_options = base_options;
  relaxed_nearby_options.max_ratio = nearby_options.max_ratio;
  relaxed_nearby_options.mutual_check = true;

  for (std::size_t i = 0; i < image_ids.size(); ++i) {
    for (std::size_t j = i + 1; j < image_ids.size(); ++j) {
      const ImageId image_id1 = image_ids[i];
      const ImageId image_id2 = image_ids[j];
      const FeatureSet* features1 = FindFeatures(*graph, image_id1);
      const FeatureSet* features2 = FindFeatures(*graph, image_id2);
      if (features1 == nullptr || features2 == nullptr) {
        continue;
      }

      // 近邻图像通常承担增量建图主链路。这里对相邻图像对放宽 ratio，
      // 同时保留互检，尽量增加几何验证候选而不引入明显的单向歧义匹配。
      const bool use_nearby_relaxation =
          nearby_options.image_index_window > 0 &&
          j - i <= nearby_options.image_index_window;
      const DescriptorMatchingOptions& pair_options =
          use_nearby_relaxation ? relaxed_nearby_options : base_options;
      SetMatches(graph, image_id1, image_id2,
                 MatchFeatureSets(*features1, *features2, pair_options));
    }
  }
}

}  // namespace colite::controllers
