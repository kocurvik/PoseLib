#pragma once

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Two view relative pose from 2 line directions + 2 points (2L2P). The rotation is fully determined by the two
// line directions (frame from their bisectors, all consistent sign choices), the translation direction follows
// linearly from the epipolar constraints of the two points (sign by cheirality). Unit length translation.
//   x1, x2 - two bearing vectors in each view
//   d1, d2 - directions of the two lines in each view
int relpose_2l2p(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                 const std::vector<Eigen::Vector3d> &d1, const std::vector<Eigen::Vector3d> &d2,
                 CameraPoseVector *output);

} // namespace poselib
