#pragma once

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Three view relative pose from 2 oriented features, e.g. SIFT features with orientation and scale (port of
// theory/solver_rotation_2sift_orientation.cpp and theory/test_twosift3view.m on the mlde branch initial). The scales
// give the depth ratios of a point between the views (depth ratio = inverse scale ratio):
//   r21 = depth of point 0 in view 2 / depth in view 1, r22 the same for point 1,
//   r31 = depth of point 0 in view 3 / depth in view 1.
// The depth of point 1 in views 1 and 3 follows from the length of the vector between the points being the same in all
// views. The vector fixes the rotations up to an angle about it, the two angles follow from the rotation part of the
// trifocal line constraint det([l1, R2^T l2, R3^T l3]) = 0 of both lines. Up to 2 x 2 x 8 solutions, the translations are
// in the units of the depth of point 0 in view 1.
//   x1, x2, x3 - two points (z > 0) in each view
//   l1, l2, l3 - two homogeneous 2D lines (normalized image coordinates) in each view, e.g. through the points
int threeview_2sift(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                    const std::vector<Eigen::Vector3d> &x3, const std::vector<Eigen::Vector3d> &l1,
                    const std::vector<Eigen::Vector3d> &l2, const std::vector<Eigen::Vector3d> &l3, double r21,
                    double r22, double r31, std::vector<ThreeViewGeometry> *output);

} // namespace poselib
