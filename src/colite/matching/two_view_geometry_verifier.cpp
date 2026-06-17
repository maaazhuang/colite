#include "colite/matching/two_view_geometry_verifier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "colite/core/geometry.h"

namespace colite {
namespace {

constexpr std::size_t kMinPointsForRobustTwoViewGeometry = 8;

struct MatchedPoints {
  std::vector<cv::Point2f> pixels1;
  std::vector<cv::Point2f> pixels2;
  std::vector<cv::Point2f> normalized1;
  std::vector<cv::Point2f> normalized2;
  std::vector<FeatureMatch> matches;
};

bool IsValidFeatureIndex(Point2DIdx idx, const FeatureSet& features) {
  return idx != kInvalidPoint2DIdx &&
         idx < static_cast<Point2DIdx>(features.keypoints.size());
}

cv::Point2f NormalizedPointFromPixel(const Camera& camera,
                                     const FeatureKeypoint& keypoint) {
  const Eigen::Vector3d ray =
      UnprojectPixel(camera, Eigen::Vector2d(keypoint.x, keypoint.y), 1.0);
  return {static_cast<float>(ray.x() / ray.z()),
          static_cast<float>(ray.y() / ray.z())};
}

MatchedPoints ExtractMatchedPoints(const Camera& camera1,
                                   const FeatureSet& features1,
                                   const Camera& camera2,
                                   const FeatureSet& features2,
                                   const std::vector<FeatureMatch>& matches) {
  MatchedPoints points;
  points.pixels1.reserve(matches.size());
  points.pixels2.reserve(matches.size());
  points.normalized1.reserve(matches.size());
  points.normalized2.reserve(matches.size());
  points.matches.reserve(matches.size());

  for (const FeatureMatch& match : matches) {
    if (!IsValidFeatureIndex(match.idx1, features1) ||
        !IsValidFeatureIndex(match.idx2, features2)) {
      throw std::out_of_range("Feature match index is outside keypoint range");
    }

    const FeatureKeypoint& keypoint1 = features1.keypoints[match.idx1];
    const FeatureKeypoint& keypoint2 = features2.keypoints[match.idx2];
    points.pixels1.emplace_back(keypoint1.x, keypoint1.y);
    points.pixels2.emplace_back(keypoint2.x, keypoint2.y);
    points.normalized1.push_back(NormalizedPointFromPixel(camera1, keypoint1));
    points.normalized2.push_back(NormalizedPointFromPixel(camera2, keypoint2));
    points.matches.push_back(match);
  }

  return points;
}

Eigen::Matrix3d CvMatToEigen3d(const cv::Mat& input) {
  cv::Mat matrix;
  input.convertTo(matrix, CV_64F);

  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = matrix.at<double>(row, col);
    }
  }
  return result;
}

cv::Mat FirstThreeByThreeBlock(const cv::Mat& input) {
  if (input.rows < 3 || input.cols != 3) {
    return {};
  }
  return input.rowRange(0, 3).clone();
}

std::vector<FeatureMatch> InlierMatchesFromMask(
    const std::vector<FeatureMatch>& matches,
    const cv::Mat& mask) {
  std::vector<FeatureMatch> inliers;
  if (mask.empty()) {
    return inliers;
  }

  inliers.reserve(matches.size());
  for (int i = 0; i < mask.rows; ++i) {
    if (mask.at<uchar>(i, 0) != 0) {
      inliers.push_back(matches[static_cast<std::size_t>(i)]);
    }
  }
  return inliers;
}

double NormalizedRansacThreshold(const Camera& camera1,
                                 const Camera& camera2,
                                 double max_error_px) {
  const double focal = 0.5 * (camera1.MeanFocalLength() +
                             camera2.MeanFocalLength());
  if (focal <= 0.0 || !std::isfinite(focal)) {
    return max_error_px;
  }
  return max_error_px / focal;
}

Pose3 RelativePoseFromCv(const cv::Mat& rotation, const cv::Mat& translation) {
  Pose3 cam2_from_cam1;
  const Eigen::Matrix3d r = CvMatToEigen3d(rotation);
  cam2_from_cam1.q_cw = Eigen::Quaterniond(r).normalized();
  cam2_from_cam1.t_cw =
      Eigen::Vector3d(translation.at<double>(0), translation.at<double>(1),
                      translation.at<double>(2));
  return cam2_from_cam1;
}

double ComputeMedianTriangulationAngle(
    const Camera& camera1,
    const FeatureSet& features1,
    const Camera& camera2,
    const FeatureSet& features2,
    const Pose3& cam2_from_cam1,
    const std::vector<FeatureMatch>& inlier_matches) {
  if (inlier_matches.empty()) {
    return -1.0;
  }

  Pose3 cam1_from_world;
  cam1_from_world.q_cw = Eigen::Quaterniond::Identity();
  cam1_from_world.t_cw = Eigen::Vector3d::Zero();

  std::vector<double> angles;
  angles.reserve(inlier_matches.size());
  for (const FeatureMatch& match : inlier_matches) {
    const FeatureKeypoint& keypoint1 = features1.keypoints[match.idx1];
    const FeatureKeypoint& keypoint2 = features2.keypoints[match.idx2];
    const Eigen::Vector3d bearing1 =
        PixelToBearing(camera1, Eigen::Vector2d(keypoint1.x, keypoint1.y));
    const Eigen::Vector3d bearing2 =
        PixelToBearing(camera2, Eigen::Vector2d(keypoint2.x, keypoint2.y));
    angles.push_back(TriangulationAngleDegrees(
        cam1_from_world, bearing1, cam2_from_cam1, bearing2));
  }

  std::sort(angles.begin(), angles.end());
  return angles[angles.size() / 2];
}

}  // namespace

TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1,
    const FeatureSet& features1,
    const Camera& camera2,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& matches,
    const TwoViewGeometryOptions& options) {
  TwoViewGeometry geometry;
  const MatchedPoints points =
      ExtractMatchedPoints(camera1, features1, camera2, features2, matches);

  const std::size_t min_required = std::max(
      kMinPointsForRobustTwoViewGeometry,
      static_cast<std::size_t>(std::max(options.min_num_inliers, 0)));
  if (points.matches.size() < min_required) {
    geometry.type = TwoViewGeometry::Type::kDegenerate;
    return geometry;
  }

  cv::Mat fundamental_mask;
  const cv::Mat fundamental = cv::findFundamentalMat(
      points.pixels1, points.pixels2, cv::FM_RANSAC, options.max_error_px,
      options.confidence, fundamental_mask);
  std::vector<FeatureMatch> fundamental_inliers;
  if (!fundamental.empty() && fundamental.rows == 3 && fundamental.cols == 3) {
    geometry.F = CvMatToEigen3d(fundamental);
    fundamental_inliers =
        InlierMatchesFromMask(points.matches, fundamental_mask);
  }

  if (options.estimate_homography) {
    cv::Mat homography_mask;
    const cv::Mat homography = cv::findHomography(
        points.pixels1, points.pixels2, cv::RANSAC, options.max_error_px,
        homography_mask);
    if (!homography.empty() && homography.rows == 3 && homography.cols == 3) {
      geometry.H = CvMatToEigen3d(homography);
    }
  }

  cv::Mat essential_mask;
  cv::Mat essential = cv::findEssentialMat(
      points.normalized1, points.normalized2, 1.0, cv::Point2d(0.0, 0.0),
      cv::RANSAC, options.confidence,
      NormalizedRansacThreshold(camera1, camera2, options.max_error_px),
      essential_mask);
  essential = FirstThreeByThreeBlock(essential);

  if (!essential.empty()) {
    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat recover_mask = essential_mask.clone();
    cv::recoverPose(essential, points.normalized1, points.normalized2, rotation,
                    translation, 1.0, cv::Point2d(0.0, 0.0), recover_mask);

    geometry.E = CvMatToEigen3d(essential);
    geometry.cam2_from_cam1 = RelativePoseFromCv(rotation, translation);
    geometry.inlier_matches = InlierMatchesFromMask(points.matches, recover_mask);
    geometry.median_triangulation_angle = ComputeMedianTriangulationAngle(
        camera1, features1, camera2, features2, geometry.cam2_from_cam1,
        geometry.inlier_matches);

    if (geometry.inlier_matches.size() >=
        static_cast<std::size_t>(options.min_num_inliers)) {
      geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
      return geometry;
    }
  }

  if (fundamental_inliers.size() >=
      static_cast<std::size_t>(options.min_num_inliers)) {
    geometry.type = TwoViewGeometry::Type::kUncalibratedFundamental;
    geometry.inlier_matches = fundamental_inliers;
    return geometry;
  }

  geometry.type = TwoViewGeometry::Type::kDegenerate;
  geometry.inlier_matches.clear();
  geometry.median_triangulation_angle = -1.0;
  return geometry;
}

}  // namespace colite
