#ifndef POSELIB_RELPOSE_UPRIGHT_3PT_DING_H_
#define POSELIB_RELPOSE_UPRIGHT_3PT_DING_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Upright relative pose (rotation about the y axis) from three point correspondences.
// The implementation is based on finding intersections of two conics:
//   Y. Ding, J. Yang, V. Larsson, C. Olsson, K. Astrom, Revisiting the P3P Problem, CVPR 2023
// Ported from upstream PoseLib (authors Yaqing Ding, Mark Shachkov). Translations are unit length.
int relpose_upright_3pt_ding(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                             CameraPoseVector *output);

// Known common direction g_cam1 in view 1 and g_cam2 in view 2 (g_cam2 = R * g_cam1).
int relpose_upright_3pt_ding(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                             const Eigen::Vector3d &g_cam1, const Eigen::Vector3d &g_cam2, CameraPoseVector *output);

} // namespace poselib

#endif
