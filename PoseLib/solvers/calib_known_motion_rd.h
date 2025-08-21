//
// Created by kocur on 09-Jan-25.
//

#ifndef POSELIB_CALIB_KNOWN_MOTION_RD_H
#define POSELIB_CALIB_KNOWN_MOTION_RD_H

#include "PoseLib/camera_pose.h"

namespace poselib {
void calib_known_motion_shared_frd_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                      const Eigen::Matrix3d &E, const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_shared_frd_3p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                      const Eigen::Matrix3d &E, const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_shared_frd_norot_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                            const Eigen::Matrix3d &E, const CameraPose &pose, ImagePairVector *models);
} //namespace poselib

#endif // POSELIB_CALIB_KNOWN_MOTION_RD_H
