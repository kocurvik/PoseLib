//
// Created by kocur on 14-May-24.
//

#ifndef POSELIB_RELPOSE_K2FK1_9PT_H
#define POSELIB_RELPOSE_K2FK1_9PT_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
int relpose_k2Fk1_9pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                      std::vector<ProjectiveImagePair> *F_cam_pair);
}

#endif //POSELIB_RELPOSE_K2FK1_9PT_H
