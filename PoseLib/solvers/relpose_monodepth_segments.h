#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Relative pose with monocular depths from matched 3D segments (depth * ray in the camera frames, the X1 endpoints match
// and the X2 endpoints match), the model is scale * X2 = R * X1 + t as in relpose_monodepth_3pt without the shifts. A
// segment gives 6 of the 7 constraints, only the rotation about the segment is free.

// 1 segment + 1 point without its depth (x1, x2 in homogeneous normalized coordinates). The scale, the direction and the
// midpoint of the segment fix everything except the angle about the segment, which follows from the epipolar constraint
// of the point in the form A cos(angle) + B sin(angle) + C = 0. Up to 2 solutions, those with the point in front of both
// cameras are returned.
int relpose_monodepth_1s1p(const Line3D &S1, const Line3D &S2, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2,
                           std::vector<MonoDepthTwoViewGeometry> *output);

// 2 segments, overdetermined: least squares alignment of the 4 endpoints (Horn's scale, symmetric in the two views). No
// solution if the endpoints are close to collinear.
int relpose_monodepth_2s(const Line3D &Sa1, const Line3D &Sb1, const Line3D &Sa2, const Line3D &Sb2,
                         std::vector<MonoDepthTwoViewGeometry> *output);

} // namespace poselib
