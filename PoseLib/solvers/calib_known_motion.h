//
// Created by kocur on 02-Oct-24.
//

#ifndef POSELIB_CALIB_KNOWN_MOTION_H
#define POSELIB_CALIB_KNOWN_MOTION_H

#include "PoseLib/camera_pose.h"
namespace poselib {

void calib_known_motion_f_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                              const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_f_3p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                             const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_fpp_7p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                               const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_shared_f_1p(const poselib::Point2D &x1, const poselib::Point2D &x2, const Eigen::Matrix3d &E,
                                    const CameraPose &pose, ImagePairVector *models);

void calib_known_motion_shared_f_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                    const Eigen::Matrix3d &E, const CameraPose &pose,
                                    poselib::ImagePairVector *models);

void calib_known_motion_shared_fpp_4p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                               const Eigen::Matrix3d &E, const CameraPose &pose,
                                               ImagePairVector *models);

} //namespace poselib

#endif // POSELIB_CALIB_KNOWN_MOTION_H
