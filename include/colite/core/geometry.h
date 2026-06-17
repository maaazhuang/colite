#pragma once

#include <optional>

#include <Eigen/Core>

#include "colite/core/camera.h"
#include "colite/core/pose.h"

namespace colite {

// 将像素坐标转换为相机坐标系中的单位视线方向。
// 该函数会经过相机模型去畸变；未实现反投影的模型会抛出异常。
Eigen::Vector3d PixelToBearing(const Camera& camera,
                               const Eigen::Vector2d& pixel);

// 将世界坐标点投影到图像平面。
// 返回 std::nullopt 表示点位于相机后方或深度无效。
std::optional<Eigen::Vector2d> ProjectWorldPoint(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world);

// 计算单个观测的重投影误差，单位为像素。
// 如果点无法投影，返回 std::nullopt。
std::optional<double> ReprojectionError(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world,
    const Eigen::Vector2d& observed_pixel);

// 根据两张图像的位姿和像素观测进行线性三角化。
// 输入位姿均采用 world -> camera 约定，输出点位于世界坐标系。
// 退化或数值不稳定时返回 std::nullopt。
std::optional<Eigen::Vector3d> TriangulatePoint(
    const Camera& camera1,
    const Pose3& cam1_from_world,
    const Eigen::Vector2d& pixel1,
    const Camera& camera2,
    const Pose3& cam2_from_world,
    const Eigen::Vector2d& pixel2);

// 计算两条观测射线之间的夹角，单位为角度。
// 该值常用于过滤基线过小、三角化不稳定的 track。
double TriangulationAngleDegrees(const Pose3& cam1_from_world,
                                 const Eigen::Vector3d& bearing1,
                                 const Pose3& cam2_from_world,
                                 const Eigen::Vector3d& bearing2);

}  // namespace colite

