#ifndef POSELIB_RELPOSE_6PT_ONEFOCAL_H
#define POSELIB_RELPOSE_6PT_ONEFOCAL_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solves for relative pose with one unknown focal length from 6 correspondences
// The solver is created using Larsson et al. CVPR 2017
int relpose_6pt_onefocal(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                         CameraOneFocaPoseVector *out_focal_poses);
}; // namespace poselib

#endif