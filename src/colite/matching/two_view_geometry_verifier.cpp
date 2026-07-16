#include "colite/matching/two_view_geometry_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "colite/core/geometry.h"

#ifdef COLITE_HAS_POSELIB
#include <PoseLib/solvers/relpose_5pt.h>
#endif

namespace colite {
namespace {

constexpr std::size_t kMinPointsForRobustTwoViewGeometry = 8;
constexpr double kMinHomogeneousScale = 1e-12;

struct MatchedPoints {
  std::vector<Eigen::Vector2d> pixels1;
  std::vector<Eigen::Vector2d> pixels2;
  std::vector<Eigen::Vector2d> normalized1;
  std::vector<Eigen::Vector2d> normalized2;
  std::vector<FeatureMatch> matches;
};

struct NormalizedPoints2D {
  std::vector<Eigen::Vector2d> points;
  Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
};

struct MatrixRansacResult {
  Eigen::Matrix3d model = Eigen::Matrix3d::Zero();
  std::vector<std::size_t> inlier_indices;
};

bool IsValidFeatureIndex(Point2DIdx idx, const FeatureSet& features) {
  return idx != kInvalidPoint2DIdx &&
         idx < static_cast<Point2DIdx>(features.keypoints.size());
}

Eigen::Vector2d NormalizedPointFromPixel(const Camera& camera,
                                         const FeatureKeypoint& keypoint) {
  const Eigen::Vector3d ray =
      UnprojectPixel(camera, Eigen::Vector2d(keypoint.x, keypoint.y), 1.0);
  return {ray.x() / ray.z(), ray.y() / ray.z()};
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

NormalizedPoints2D NormalizePoints(const std::vector<Eigen::Vector2d>& points,
                                   const std::vector<std::size_t>& indices) {
  NormalizedPoints2D result;
  result.points.reserve(indices.size());
  if (indices.empty()) {
    return result;
  }

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const std::size_t idx : indices) {
    centroid += points[idx];
  }
  centroid /= static_cast<double>(indices.size());

  double mean_distance = 0.0;
  for (const std::size_t idx : indices) {
    mean_distance += (points[idx] - centroid).norm();
  }
  mean_distance /= static_cast<double>(indices.size());
  const double scale =
      mean_distance > 0.0 ? std::sqrt(2.0) / mean_distance : 1.0;

  result.transform << scale, 0.0, -scale * centroid.x(), 0.0, scale,
      -scale * centroid.y(), 0.0, 0.0, 1.0;
  for (const std::size_t idx : indices) {
    result.points.push_back((result.transform *
                             points[idx].homogeneous())
                                .hnormalized());
  }
  return result;
}

std::vector<std::size_t> AllIndices(const std::size_t count) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

std::vector<std::size_t> SampleUniqueIndices(std::mt19937* rng,
                                             const std::size_t count,
                                             const std::size_t sample_size) {
  std::vector<std::size_t> indices = AllIndices(count);
  std::shuffle(indices.begin(), indices.end(), *rng);
  indices.resize(sample_size);
  return indices;
}

std::optional<Eigen::Matrix3d> EstimateFundamentalMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  if (indices.size() < 8) {
    return std::nullopt;
  }

  const NormalizedPoints2D norm1 = NormalizePoints(points1, indices);
  const NormalizedPoints2D norm2 = NormalizePoints(points2, indices);

  Eigen::MatrixXd a(indices.size(), 9);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const double x1 = norm1.points[i].x();
    const double y1 = norm1.points[i].y();
    const double x2 = norm2.points[i].x();
    const double y2 = norm2.points[i].y();
    a.row(static_cast<Eigen::Index>(i)) << x2 * x1, x2 * y1, x2, y2 * x1,
        y2 * y1, y2, x1, y1, 1.0;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  Eigen::Matrix3d f =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          svd.matrixV().col(8).data());

  const Eigen::JacobiSVD<Eigen::Matrix3d> f_svd(
      f, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = f_svd.singularValues();
  singular_values.z() = 0.0;
  f = f_svd.matrixU() * singular_values.asDiagonal() *
      f_svd.matrixV().transpose();

  Eigen::Matrix3d denormalized =
      norm2.transform.transpose() * f * norm1.transform;
  const double norm = denormalized.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return std::nullopt;
  }
  return denormalized / norm;
}

std::optional<Eigen::Matrix3d> EstimateEssentialMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  auto essential = EstimateFundamentalMatrix(points1, points2, indices);
  if (!essential) {
    return std::nullopt;
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      *essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const double sigma =
      0.5 * (svd.singularValues().x() + svd.singularValues().y());
  Eigen::Matrix3d corrected = svd.matrixU() *
                              Eigen::Vector3d(sigma, sigma, 0.0).asDiagonal() *
                              svd.matrixV().transpose();
  const double norm = corrected.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return std::nullopt;
  }
  return corrected / norm;
}

std::optional<Eigen::Matrix3d> EstimateHomographyMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  if (indices.size() < 4) {
    return std::nullopt;
  }

  const NormalizedPoints2D norm1 = NormalizePoints(points1, indices);
  const NormalizedPoints2D norm2 = NormalizePoints(points2, indices);

  Eigen::MatrixXd a(static_cast<Eigen::Index>(2 * indices.size()), 9);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const double x = norm1.points[i].x();
    const double y = norm1.points[i].y();
    const double xp = norm2.points[i].x();
    const double yp = norm2.points[i].y();
    a.row(static_cast<Eigen::Index>(2 * i))
        << -x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp;
    a.row(static_cast<Eigen::Index>(2 * i + 1))
        << 0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  Eigen::Matrix3d h =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          svd.matrixV().col(8).data());
  h = norm2.transform.inverse() * h * norm1.transform;

  const double scale =
      std::abs(h(2, 2)) > kMinHomogeneousScale ? h(2, 2) : h.norm();
  if (std::abs(scale) <= kMinHomogeneousScale || !std::isfinite(scale)) {
    return std::nullopt;
  }
  return h / scale;
}

double SampsonError(const Eigen::Matrix3d& matrix,
                    const Eigen::Vector2d& point1,
                    const Eigen::Vector2d& point2) {
  const Eigen::Vector3d x1 = point1.homogeneous();
  const Eigen::Vector3d x2 = point2.homogeneous();
  const Eigen::Vector3d mx1 = matrix * x1;
  const Eigen::Vector3d mtx2 = matrix.transpose() * x2;
  const double x2tmx1 = x2.dot(mx1);
  const double denom =
      mx1.x() * mx1.x() + mx1.y() * mx1.y() + mtx2.x() * mtx2.x() +
      mtx2.y() * mtx2.y();
  if (denom <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(x2tmx1) / std::sqrt(denom);
}

double HomographyTransferError(const Eigen::Matrix3d& homography,
                               const Eigen::Vector2d& point1,
                               const Eigen::Vector2d& point2) {
  const Eigen::Vector3d transferred = homography * point1.homogeneous();
  if (std::abs(transferred.z()) <= kMinHomogeneousScale) {
    return std::numeric_limits<double>::infinity();
  }
  return (transferred.hnormalized() - point2).norm();
}

template <typename Estimator, typename Residual>
MatrixRansacResult RunMatrixRansac(const std::size_t num_points,
                                   const std::size_t sample_size,
                                   const int min_num_iterations,
                                   const int max_num_iterations,
                                   const double confidence,
                                   const double max_error,
                                   Estimator estimator,
                                   Residual residual) {
  MatrixRansacResult best;
  if (num_points < sample_size) {
    return best;
  }

  std::mt19937 rng(0);
  const int min_iterations = std::max(1, min_num_iterations);
  const int max_iterations = std::max(min_iterations, max_num_iterations);
  int required_iterations = max_iterations;
  auto score_model = [&](const Eigen::Matrix3d& model) {
    std::vector<std::size_t> inliers;
    inliers.reserve(num_points);
    for (std::size_t idx = 0; idx < num_points; ++idx) {
      const double error = residual(model, idx);
      if (std::isfinite(error) && error <= max_error) {
        inliers.push_back(idx);
      }
    }
    return inliers;
  };

  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    const std::vector<std::size_t> sample =
        num_points == sample_size ? AllIndices(num_points)
                                  : SampleUniqueIndices(&rng, num_points,
                                                        sample_size);
    std::optional<Eigen::Matrix3d> model = estimator(sample);
    if (!model) {
      continue;
    }

    std::vector<std::size_t> inliers = score_model(*model);

    // One local-optimization step substantially stabilizes the eight-point
    // and homography minimal hypotheses without coupling this module to
    // OpenCV calib3d.
    if (inliers.size() >= sample_size) {
      const std::optional<Eigen::Matrix3d> local_model = estimator(inliers);
      if (local_model) {
        std::vector<std::size_t> local_inliers = score_model(*local_model);
        if (local_inliers.size() >= inliers.size()) {
          model = local_model;
          inliers = std::move(local_inliers);
        }
      }
    }

    if (inliers.size() > best.inlier_indices.size()) {
      best.model = *model;
      best.inlier_indices = std::move(inliers);
      if (best.inlier_indices.size() == num_points) {
        break;
      }
      const double inlier_ratio =
          static_cast<double>(best.inlier_indices.size()) / num_points;
      const double all_inlier_probability =
          std::pow(inlier_ratio, static_cast<double>(sample_size));
      if (all_inlier_probability > 0.0 && all_inlier_probability < 1.0) {
        const double safe_confidence = std::clamp(confidence, 0.0, 1.0 - 1e-12);
        const double estimate =
            std::log(1.0 - safe_confidence) /
            std::log(1.0 - all_inlier_probability);
        if (std::isfinite(estimate)) {
          required_iterations = std::clamp(
              static_cast<int>(std::ceil(estimate)), min_iterations,
              max_iterations);
          required_iterations = std::max(required_iterations, iteration + 1);
        }
      }
    }
  }

  if (best.inlier_indices.size() >= sample_size) {
    const std::optional<Eigen::Matrix3d> refined =
        estimator(best.inlier_indices);
    if (refined) {
      std::vector<std::size_t> refined_inliers = score_model(*refined);
      if (refined_inliers.size() >= best.inlier_indices.size()) {
        best.model = *refined;
        best.inlier_indices = std::move(refined_inliers);
      }
    }
  }

  return best;
}

MatrixRansacResult RunEssentialRansac(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometryOptions& options,
    const double max_error) {
#ifndef COLITE_HAS_POSELIB
  return RunMatrixRansac(
      points1.size(), 8, options.min_num_iterations,
      options.max_num_iterations, options.confidence, max_error,
      [&](const std::vector<std::size_t>& sample) {
        return EstimateEssentialMatrix(points1, points2, sample);
      },
      [&](const Eigen::Matrix3d& model, const std::size_t idx) {
        return SampsonError(model, points1[idx], points2[idx]);
      });
#else
  constexpr std::size_t kSampleSize = 5;
  MatrixRansacResult best;
  if (points1.size() < kSampleSize || points1.size() != points2.size()) {
    return best;
  }
  auto score_model = [&](const Eigen::Matrix3d& model) {
    std::vector<std::size_t> inliers;
    inliers.reserve(points1.size());
    for (std::size_t idx = 0; idx < points1.size(); ++idx) {
      const double error = SampsonError(model, points1[idx], points2[idx]);
      if (std::isfinite(error) && error <= max_error) {
        inliers.push_back(idx);
      }
    }
    return inliers;
  };

  std::mt19937 rng(0);
  const int min_iterations = std::max(1, options.min_num_iterations);
  const int max_iterations =
      std::max(min_iterations, options.max_num_iterations);
  int required_iterations = max_iterations;
  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    const std::vector<std::size_t> sample =
        SampleUniqueIndices(&rng, points1.size(), kSampleSize);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    rays1.reserve(kSampleSize);
    rays2.reserve(kSampleSize);
    for (const std::size_t idx : sample) {
      rays1.push_back(points1[idx].homogeneous().normalized());
      rays2.push_back(points2[idx].homogeneous().normalized());
    }
    std::vector<Eigen::Matrix3d> models;
    poselib::relpose_5pt(rays1, rays2, &models);
    for (Eigen::Matrix3d model : models) {
      const double norm = model.norm();
      if (!(norm > 0.0) || !std::isfinite(norm)) {
        continue;
      }
      model /= norm;
      std::vector<std::size_t> inliers = score_model(model);
      if (inliers.size() >= 8) {
        const std::optional<Eigen::Matrix3d> local =
            EstimateEssentialMatrix(points1, points2, inliers);
        if (local) {
          std::vector<std::size_t> local_inliers = score_model(*local);
          if (local_inliers.size() >= inliers.size()) {
            model = *local;
            inliers = std::move(local_inliers);
          }
        }
      }
      if (inliers.size() <= best.inlier_indices.size()) {
        continue;
      }
      best.model = model;
      best.inlier_indices = std::move(inliers);
      if (best.inlier_indices.size() == points1.size()) {
        return best;
      }
      const double ratio =
          static_cast<double>(best.inlier_indices.size()) / points1.size();
      const double sample_success = std::pow(ratio, 5.0);
      if (sample_success > 0.0 && sample_success < 1.0) {
        const double confidence =
            std::clamp(options.confidence, 0.0, 1.0 - 1e-12);
        const double estimate = std::log(1.0 - confidence) /
                                std::log(1.0 - sample_success);
        if (std::isfinite(estimate)) {
          required_iterations = std::clamp(
              static_cast<int>(std::ceil(estimate)), min_iterations,
              max_iterations);
          required_iterations =
              std::max(required_iterations, iteration + 1);
        }
      }
    }
  }
  return best;
#endif
}

std::vector<FeatureMatch> MatchesFromInlierIndices(
    const std::vector<FeatureMatch>& matches,
    const std::vector<std::size_t>& inlier_indices) {
  std::vector<FeatureMatch> inliers;
  inliers.reserve(inlier_indices.size());
  for (const std::size_t idx : inlier_indices) {
    inliers.push_back(matches[idx]);
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

double InlierRatio(const std::size_t numerator,
                   const std::size_t denominator) {
  if (denominator == 0) {
    return numerator == 0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

bool HasEnoughInliers(const MatrixRansacResult& result,
                      const TwoViewGeometryOptions& options) {
  return result.inlier_indices.size() >=
         static_cast<std::size_t>(std::max(options.min_num_inliers, 0));
}

const std::vector<std::size_t>& PreferMoreInliers(
    const std::vector<std::size_t>& lhs,
    const std::vector<std::size_t>& rhs) {
  return rhs.size() > lhs.size() ? rhs : lhs;
}

std::optional<Eigen::Vector3d> TriangulateNormalizedPoint(
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation) {
  Eigen::Matrix<double, 3, 4> p1;
  p1 << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 4> p2;
  p2 << rotation, translation;

  Eigen::Matrix4d a;
  a.row(0) = point1.x() * p1.row(2) - p1.row(0);
  a.row(1) = point1.y() * p1.row(2) - p1.row(1);
  a.row(2) = point2.x() * p2.row(2) - p2.row(0);
  a.row(3) = point2.y() * p2.row(2) - p2.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(a, Eigen::ComputeFullV);
  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) <= kMinHomogeneousScale) {
    return std::nullopt;
  }
  return homogeneous.hnormalized();
}

bool HasPositiveDepthInBothViews(const Eigen::Vector3d& point_cam1,
                                 const Eigen::Matrix3d& rotation,
                                 const Eigen::Vector3d& translation) {
  const Eigen::Vector3d point_cam2 = rotation * point_cam1 + translation;
  return point_cam1.z() > 0.0 && point_cam2.z() > 0.0;
}

Pose3 ChoosePoseFromEssentialMatrix(
    const Eigen::Matrix3d& essential,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& inlier_indices) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  Eigen::Matrix3d vt = svd.matrixV().transpose();
  if (u.determinant() < 0.0) {
    u.col(2) *= -1.0;
  }
  if (vt.determinant() < 0.0) {
    vt.row(2) *= -1.0;
  }

  Eigen::Matrix3d w;
  w << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates;
  candidates.emplace_back(u * w * vt, u.col(2));
  candidates.emplace_back(u * w * vt, -u.col(2));
  candidates.emplace_back(u * w.transpose() * vt, u.col(2));
  candidates.emplace_back(u * w.transpose() * vt, -u.col(2));

  std::size_t best_positive_depths = 0;
  Pose3 best_pose;
  for (auto& [rotation, translation] : candidates) {
    if (rotation.determinant() < 0.0) {
      rotation = -rotation;
      translation = -translation;
    }

    std::size_t positive_depths = 0;
    for (const std::size_t idx : inlier_indices) {
      const auto point_cam1 =
          TriangulateNormalizedPoint(points1[idx], points2[idx], rotation,
                                     translation);
      if (point_cam1 &&
          HasPositiveDepthInBothViews(*point_cam1, rotation, translation)) {
        ++positive_depths;
      }
    }

    if (positive_depths > best_positive_depths) {
      best_positive_depths = positive_depths;
      best_pose.q_cw = Eigen::Quaterniond(rotation).normalized();
      best_pose.t_cw = translation.normalized();
    }
  }
  return best_pose;
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

bool HomographyLooksLikePureRotation(const Camera& camera1,
                                     const Camera& camera2,
                                     const Eigen::Matrix3d& homography) {
  if (homography.norm() <= 0.0) {
    return false;
  }

  const Eigen::Matrix3d normalized =
      camera2.CalibrationMatrix().inverse() * homography *
      camera1.CalibrationMatrix();
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      normalized, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d singular_values = svd.singularValues();
  const double min_sv = singular_values.minCoeff();
  const double max_sv = singular_values.maxCoeff();
  if (min_sv <= 0.0 || !std::isfinite(min_sv) || !std::isfinite(max_sv)) {
    return false;
  }

  // 纯旋转单应在归一化相机平面中等价于一个旋转矩阵乘尺度，
  // 因此三个奇异值应非常接近；平面+平移通常会拉开奇异值。
  return max_sv / min_sv < 1.02;
}

Pose3 PoseFromPureRotationHomography(const Camera& camera1,
                                     const Camera& camera2,
                                     const Eigen::Matrix3d& homography) {
  const Eigen::Matrix3d normalized =
      camera2.CalibrationMatrix().inverse() * homography *
      camera1.CalibrationMatrix();
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      normalized, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
  if (rotation.determinant() < 0.0) {
    rotation = -rotation;
  }

  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond(rotation).normalized();
  pose.t_cw = Eigen::Vector3d::Zero();
  return pose;
}

TwoViewGeometry::Type ClassifyHomographyDegeneracy(
    const Camera& camera1,
    const Camera& camera2,
    const Eigen::Matrix3d& homography,
    const double median_triangulation_angle,
    const TwoViewGeometryOptions& options) {
  if (median_triangulation_angle >= 0.0 &&
      median_triangulation_angle <=
          options.max_panoramic_triangulation_angle_deg) {
    return TwoViewGeometry::Type::kPanoramic;
  }
  if (HomographyLooksLikePureRotation(camera1, camera2, homography)) {
    return TwoViewGeometry::Type::kPanoramic;
  }
  return TwoViewGeometry::Type::kPlanarHomography;
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

  const MatrixRansacResult fundamental = RunMatrixRansac(
      points.matches.size(), 8, options.min_num_iterations,
      options.max_num_iterations, options.confidence,
      options.max_error_px,
      [&](const std::vector<std::size_t>& sample) {
        return EstimateFundamentalMatrix(points.pixels1, points.pixels2,
                                         sample);
      },
      [&](const Eigen::Matrix3d& model, const std::size_t idx) {
        return SampsonError(model, points.pixels1[idx], points.pixels2[idx]);
      });
  if (!fundamental.inlier_indices.empty()) {
    geometry.F = fundamental.model;
  }

  MatrixRansacResult homography;
  if (options.estimate_homography) {
    homography = RunMatrixRansac(
        points.matches.size(), 4, options.min_num_iterations,
        options.max_num_iterations, options.confidence,
        options.max_error_px,
        [&](const std::vector<std::size_t>& sample) {
          return EstimateHomographyMatrix(points.pixels1, points.pixels2,
                                          sample);
        },
        [&](const Eigen::Matrix3d& model, const std::size_t idx) {
          return HomographyTransferError(model, points.pixels1[idx],
                                         points.pixels2[idx]);
        });
    if (!homography.inlier_indices.empty()) {
      geometry.H = homography.model;
    }
  }

  const double normalized_threshold =
      NormalizedRansacThreshold(camera1, camera2, options.max_error_px);
  const MatrixRansacResult essential = RunEssentialRansac(
      points.normalized1, points.normalized2, options, normalized_threshold);

  const bool essential_valid = HasEnoughInliers(essential, options);
  const bool fundamental_valid = HasEnoughInliers(fundamental, options);
  const bool homography_valid =
      options.estimate_homography && HasEnoughInliers(homography, options);

  if (!essential_valid && !fundamental_valid && !homography_valid) {
    geometry.type = TwoViewGeometry::Type::kDegenerate;
    geometry.inlier_matches.clear();
    geometry.median_triangulation_angle = -1.0;
    return geometry;
  }

  if (essential_valid &&
      InlierRatio(essential.inlier_indices.size(),
                  fundamental.inlier_indices.size()) >
          options.min_E_F_inlier_ratio) {
    geometry.E = essential.model;
    geometry.cam2_from_cam1 = ChoosePoseFromEssentialMatrix(
        geometry.E, points.normalized1, points.normalized2,
        essential.inlier_indices);
    const std::vector<std::size_t>& epipolar_inliers =
        PreferMoreInliers(essential.inlier_indices,
                          fundamental.inlier_indices);
    geometry.inlier_matches = MatchesFromInlierIndices(points.matches,
                                                       epipolar_inliers);
    geometry.median_triangulation_angle = ComputeMedianTriangulationAngle(
        camera1, features1, camera2, features2, geometry.cam2_from_cam1,
        geometry.inlier_matches);
    if (homography_valid &&
        InlierRatio(homography.inlier_indices.size(),
                    essential.inlier_indices.size()) >
            options.max_H_inlier_ratio) {
      if (homography.inlier_indices.size() > geometry.inlier_matches.size()) {
        geometry.inlier_matches =
            MatchesFromInlierIndices(points.matches, homography.inlier_indices);
      }
      geometry.type = ClassifyHomographyDegeneracy(
          camera1, camera2, geometry.H, geometry.median_triangulation_angle,
          options);
      if (geometry.type == TwoViewGeometry::Type::kPanoramic) {
        geometry.cam2_from_cam1 =
            PoseFromPureRotationHomography(camera1, camera2, geometry.H);
        geometry.median_triangulation_angle = 0.0;
      }
      return geometry;
    }
    geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    return geometry;
  }

  if (fundamental_valid) {
    if (homography_valid &&
        InlierRatio(homography.inlier_indices.size(),
                    fundamental.inlier_indices.size()) >
            options.max_H_inlier_ratio) {
      geometry.inlier_matches =
          MatchesFromInlierIndices(points.matches, homography.inlier_indices);
      geometry.type = ClassifyHomographyDegeneracy(
          camera1, camera2, geometry.H, geometry.median_triangulation_angle,
          options);
      if (geometry.type == TwoViewGeometry::Type::kPanoramic) {
        geometry.cam2_from_cam1 =
            PoseFromPureRotationHomography(camera1, camera2, geometry.H);
        geometry.median_triangulation_angle = 0.0;
      }
      return geometry;
    }
    geometry.type = TwoViewGeometry::Type::kUncalibratedFundamental;
    geometry.inlier_matches =
        MatchesFromInlierIndices(points.matches, fundamental.inlier_indices);
    return geometry;
  }

  if (homography_valid) {
    geometry.type = ClassifyHomographyDegeneracy(
        camera1, camera2, geometry.H, geometry.median_triangulation_angle,
        options);
    geometry.inlier_matches =
        MatchesFromInlierIndices(points.matches, homography.inlier_indices);
    if (geometry.type == TwoViewGeometry::Type::kPanoramic) {
      geometry.cam2_from_cam1 =
          PoseFromPureRotationHomography(camera1, camera2, geometry.H);
      geometry.median_triangulation_angle = 0.0;
    }
    return geometry;
  }

  geometry.type = TwoViewGeometry::Type::kDegenerate;
  geometry.inlier_matches.clear();
  geometry.median_triangulation_angle = -1.0;
  return geometry;
}

}  // namespace colite
