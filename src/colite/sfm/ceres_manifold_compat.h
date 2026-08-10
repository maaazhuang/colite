#pragma once

#include <vector>

#include <ceres/problem.h>
#include <ceres/version.h>

#if CERES_VERSION_MAJOR > 2 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
#include <ceres/manifold.h>
#else
#include <ceres/local_parameterization.h>
#endif

namespace colite::ceres_compat {

inline void AddEigenQuaternionParameterBlock(ceres::Problem* problem,
                                             double* values) {
#if CERES_VERSION_MAJOR > 2 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->AddParameterBlock(values, 4, new ceres::EigenQuaternionManifold());
#else
  problem->AddParameterBlock(
      values, 4, new ceres::EigenQuaternionParameterization());
#endif
}

inline void SetSubsetParameterization(
    ceres::Problem* problem,
    double* values,
    int size,
    const std::vector<int>& constant_indices) {
#if CERES_VERSION_MAJOR > 2 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(
      values, new ceres::SubsetManifold(size, constant_indices));
#else
  problem->SetParameterization(
      values, new ceres::SubsetParameterization(size, constant_indices));
#endif
}

}  // namespace colite::ceres_compat
