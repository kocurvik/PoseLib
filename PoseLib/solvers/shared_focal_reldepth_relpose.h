//
// Created by kocur on 19-Nov-24.
//

#ifndef POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
#define POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace poselib {

void shared_focal_reldepth_relpose(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                                   const std::vector<double> &d1, const std::vector<double> &d2,
                                   std::vector<MonoDepthImagePair> *models);

} // namespace poselib

#endif // POSELIB_SHARED_FOCAL_RELDEPTH_RELPOSE_H
