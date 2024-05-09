//
// Created by kocur on 09-May-24.
//

#ifndef POSELIB_RELPOSE_KFK_H
#define POSELIB_RELPOSE_KFK_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Computes the fundamental matrix and k for division model from 8 point correspondences.
int relpose_kFk(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                std::vector<FCam> *F_cam);

}; // namespace poselib

#endif
