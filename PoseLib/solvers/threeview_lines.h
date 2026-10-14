#ifndef POSELIB_THREEVIEW_LINES_H
#define POSELIB_THREEVIEW_LINES_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Minimal solvers for three-view relative pose using points and lines with scale-invariant depth.
// All 3D inputs are expressed in the respective camera frames and each view may have an unknown depth scale.
// The returned poses map view 1 to view 2 and view 3 (X_k = R_k * X_1 + t_k) with the translation of both
// expressed in the depth scale of view 1, i.e. pose12 and pose13 share a consistent scale.
//
// Since line endpoints are not in correspondence across the views the sign of line directions is ambiguous.
// The solvers therefore return multiple candidate poses.

// 2 points with depth + 1 line direction (theory/2p1l.m)
//   X1, X2, X3 - two 3D points (depth * ray) in each view
//   d1, d2, d3 - direction of the line in each view
int threeview_2p1l(const std::vector<Eigen::Vector3d> &X1, const std::vector<Eigen::Vector3d> &X2,
                   const std::vector<Eigen::Vector3d> &X3, const Eigen::Vector3d &d1, const Eigen::Vector3d &d2,
                   const Eigen::Vector3d &d3, std::vector<ThreeViewCameraPose> *output);

// 2 lines with depth. Rotation from line directions, scale and translation from the common perpendicular.
//   L1, L2, L3 - two 3D line segments in each view
int threeview_2l(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2, const std::vector<Line3D> &L3,
                 std::vector<ThreeViewCameraPose> *output);

// 2 lines with depth. Rotation from line directions, translation from the constraint that the 3D lines
// from view 1 lie on the back-projected planes of the 2D lines in views 2 and 3. These planes fix only two
// components of the translation, the remaining one (along l_a x l_b) is recovered from the depth in views 2 and 3.
//   L1, L2, L3 - two 3D line segments in each view
//   l2, l3     - homogeneous 2D lines (normalized image coordinates) in views 2 and 3
int threeview_2l_planes(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2, const std::vector<Line3D> &L3,
                        const std::vector<Eigen::Vector3d> &l2, const std::vector<Eigen::Vector3d> &l3,
                        std::vector<ThreeViewCameraPose> *output);

} // namespace poselib

#endif // POSELIB_THREEVIEW_LINES_H
