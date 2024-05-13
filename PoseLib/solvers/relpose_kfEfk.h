#ifndef POSELIB_RELPOSE_KFEKF_H
#define POSELIB_RELPOSE_KFEKF_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Computes the E, f and k for division model from 7 point correspondences.
int relpose_kfEfk(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                  std::vector<ImagePair> *image_pairs);

}; // namespace poselib

#endif
