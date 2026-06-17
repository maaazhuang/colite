#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace colite {

// 三维刚体位姿，统一采用 world -> camera 约定。
// 即：X_cam = q_cw * X_world + t_cw。
struct Pose3 {
  Eigen::Quaterniond q_cw = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t_cw = Eigen::Vector3d::Zero();

  // 将世界坐标点变换到相机坐标系。
  Eigen::Vector3d TransformPoint(const Eigen::Vector3d& point_world) const {
    return q_cw * point_world + t_cw;
  }

  // 将一组同坐标系点统一左乘该位姿，常用于批量投影前的坐标变换。
  Eigen::Matrix3Xd TransformPoints(const Eigen::Matrix3Xd& points_world) const {
    return (q_cw.toRotationMatrix() * points_world).colwise() + t_cw;
  }

  // 相机中心在世界坐标系中的位置：C = -R^T t。
  Eigen::Vector3d CameraCenter() const {
    return -(q_cw.conjugate() * t_cw);
  }

  // 返回该刚体变换的逆变换。字段名仍沿用 q_cw/t_cw，但数学含义为反向变换。
  Pose3 Inverse() const {
    Pose3 inverse;
    inverse.q_cw = q_cw.conjugate();
    inverse.t_cw = -(inverse.q_cw * t_cw);
    return inverse;
  }

  // 变换复合：返回 this(other(X))。
  Pose3 operator*(const Pose3& other) const {
    Pose3 composed;
    composed.q_cw = q_cw * other.q_cw;
    composed.t_cw = q_cw * other.t_cw + t_cw;
    return composed;
  }
};

}  // namespace colite
