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

// 1 line direction + 3 points without depth. The line direction acts as a known vertical direction: views 1-2 are
// solved with the upright 3 point relative pose solver, view 3 with the upright 2 point absolute pose solver
// (points 0, 1 triangulated from views 1-2). For each pose of views 1-2 only the view 3 solution (over the
// direction sign and the up2p solutions) with the smallest angular error of point 2 is kept.
//   x1, x2, x3 - three bearing vectors in each view
//   d1, d2, d3 - direction of the line in each view
int threeview_1l3p(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                   const std::vector<Eigen::Vector3d> &x3, const Eigen::Vector3d &d1, const Eigen::Vector3d &d2,
                   const Eigen::Vector3d &d3, std::vector<ThreeViewCameraPose> *output);

// Two view relative pose from 2 line directions + 2 points (2L2P). The rotation is fully determined by the two
// line directions (frame from their bisectors, all consistent sign choices), the translation direction follows
// linearly from the epipolar constraints of the two points (sign by cheirality). Unit length translation.
//   x1, x2 - two bearing vectors in each view
//   d1, d2 - directions of the two lines in each view
int relpose_2l2p(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                 const std::vector<Eigen::Vector3d> &d1, const std::vector<Eigen::Vector3d> &d2,
                 CameraPoseVector *output);

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
