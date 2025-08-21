// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_RELATIVE_H_
#define POSELIB_RELATIVE_H_

#include "../../misc/essential.h"
#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"
#include "PoseLib/robust/utils.h"

namespace poselib {

inline void deriv_essential_wrt_pose(const Eigen::Matrix3d &E, const Eigen::Matrix3d &R,
                                     const Eigen::Matrix<double, 3, 2> &tangent_basis, Eigen::Matrix<double, 9, 3> &dR,
                                     Eigen::Matrix<double, 9, 2> &dt) {
    // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
    dR.block<3, 1>(0, 0).setZero();
    dR.block<3, 1>(0, 1) = -E.col(2);
    dR.block<3, 1>(0, 2) = E.col(1);
    dR.block<3, 1>(3, 0) = E.col(2);
    dR.block<3, 1>(3, 1).setZero();
    dR.block<3, 1>(3, 2) = -E.col(0);
    dR.block<3, 1>(6, 0) = -E.col(1);
    dR.block<3, 1>(6, 1) = E.col(0);
    dR.block<3, 1>(6, 2).setZero();

    // Each column is vec(skew(tangent_basis[k])*R)
    dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
    dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
    dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
    dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
    dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
    dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));
}

inline void setup_tangent_basis(const Eigen::Vector3d &t, Eigen::Matrix<double, 3, 2> &tangent_basis) {
    // We start by setting up a basis for the updates in the translation (orthogonal to t)
    // We find the minimum element of t and cross product with the corresponding basis vector.
    // (this ensures that the first cross product is not close to the zero vector)
    if (std::abs(t.x()) < std::abs(t.y())) {
        // x < y
        if (std::abs(t.x()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitX()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    } else {
        // x > y
        if (std::abs(t.y()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitY()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    }
    tangent_basis.col(1) = tangent_basis.col(0).cross(t).normalized();
}

// Minimize Sampson error with pinhole camera model. Assumes image points are in the normalized image plane.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class PinholeRelativePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    PinholeRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 5;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);
        setup_tangent_basis(pose.t, tangent_basis);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }

    typedef CameraPose param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Minimize Tangent Sampson error with any camera model. Assumes fixed camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class FixCameraRelativePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    FixCameraRelativePoseRefiner(const std::vector<Point3D> &unproj_points2D_1,
                                 const std::vector<Point3D> &unproj_points2D_2,
                                 const std::vector<Eigen::Matrix<double, 3, 2>> &J1inv,
                                 const std::vector<Eigen::Matrix<double, 3, 2>> &J2inv,
                                 const ResidualWeightVector &w = ResidualWeightVector())
        : d1(unproj_points2D_1), d2(unproj_points2D_2), M1(J1inv), M2(J2inv), weights(w) {
        this->num_params = 5;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);
            double nJc_sq = (M2[k].transpose() * E * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        setup_tangent_basis(pose.t, tangent_basis);

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << M1[k].transpose() * E.transpose() * d2[k], M2[k].transpose() * E * d1[k];
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(0) * M1[k](0, 0) * d2[k](0) + J_C(1) * M1[k](0, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](0) + J_C(3) * M2[k](0, 1) * d1[k](0));
            dF(1) -= s * (J_C(0) * M1[k](0, 0) * d2[k](1) + J_C(1) * M1[k](0, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](0) + J_C(3) * M2[k](1, 1) * d1[k](0));
            dF(2) -= s * (J_C(0) * M1[k](0, 0) * d2[k](2) + J_C(1) * M1[k](0, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](0) + J_C(3) * M2[k](2, 1) * d1[k](0));
            dF(3) -= s * (J_C(0) * M1[k](1, 0) * d2[k](0) + J_C(1) * M1[k](1, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](1) + J_C(3) * M2[k](0, 1) * d1[k](1));
            dF(4) -= s * (J_C(0) * M1[k](1, 0) * d2[k](1) + J_C(1) * M1[k](1, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](1) + J_C(3) * M2[k](1, 1) * d1[k](1));
            dF(5) -= s * (J_C(0) * M1[k](1, 0) * d2[k](2) + J_C(1) * M1[k](1, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](1) + J_C(3) * M2[k](2, 1) * d1[k](1));
            dF(6) -= s * (J_C(0) * M1[k](2, 0) * d2[k](0) + J_C(1) * M1[k](2, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](2) + J_C(3) * M2[k](0, 1) * d1[k](2));
            dF(7) -= s * (J_C(0) * M1[k](2, 0) * d2[k](1) + J_C(1) * M1[k](2, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](2) + J_C(3) * M2[k](1, 1) * d1[k](2));
            dF(8) -= s * (J_C(0) * M1[k](2, 0) * d2[k](2) + J_C(1) * M1[k](2, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](2) + J_C(3) * M2[k](2, 1) * d1[k](2));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }

    typedef CameraPose param_t;
    const std::vector<Point3D> &d1;
    const std::vector<Point3D> &d2;
    const std::vector<Eigen::Matrix<double, 3, 2>> &M1;
    const std::vector<Eigen::Matrix<double, 3, 2>> &M2;

    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Minimize Tangent Sampson error with any camera model. Allows for optimization of camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class CameraRelativePoseRefiner : public RefinerBase<ImagePair, Accumulator> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    CameraRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                              const std::vector<size_t> &cam1_ref_idx, const std::vector<size_t> &cam2_ref_idx,
                              const bool shared_camera = false, // Shared intrinsics only use camera1
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), camera1_refine_idx(cam1_ref_idx), camera2_refine_idx(cam2_ref_idx),
          shared_intrinsics(shared_camera), weights(w) {
        this->num_params = 5 + (shared_intrinsics ? cam1_ref_idx.size() : cam1_ref_idx.size() + cam2_ref_idx.size());
        d1.reserve(x1.size());
        d2.reserve(x2.size());
        M1.reserve(x1.size());
        M2.reserve(x2.size());
        d1_p.reserve(x1.size());
        d2_p.reserve(x2.size());
    }

    double compute_residual(Accumulator &acc, const ImagePair &pair) {
        Eigen::Matrix3d E;
        essential_from_motion(pair.pose, &E);

        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        camera1.unproject_with_jac(x1, &d1, &M1);
        camera2.unproject_with_jac(x2, &d2, &M2);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);
            double nJc_sq = (M2[k].transpose() * E * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImagePair &pair) {
        const CameraPose &pose = pair.pose;
        setup_tangent_basis(pose.t, tangent_basis);

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
            camera1.unproject_with_jac(x1, &d1, &M1, &d1_p);
            camera2.unproject_with_jac(x2, &d2, &M2, &d2_p);
        } else {
            camera1.unproject_with_jac(x1, &d1, &M1);
            camera2.unproject_with_jac(x2, &d2, &M2);
        }

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << M1[k].transpose() * E.transpose() * d2[k], M2[k].transpose() * E * d1[k];
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(0) * M1[k](0, 0) * d2[k](0) + J_C(1) * M1[k](0, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](0) + J_C(3) * M2[k](0, 1) * d1[k](0));
            dF(1) -= s * (J_C(0) * M1[k](0, 0) * d2[k](1) + J_C(1) * M1[k](0, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](0) + J_C(3) * M2[k](1, 1) * d1[k](0));
            dF(2) -= s * (J_C(0) * M1[k](0, 0) * d2[k](2) + J_C(1) * M1[k](0, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](0) + J_C(3) * M2[k](2, 1) * d1[k](0));
            dF(3) -= s * (J_C(0) * M1[k](1, 0) * d2[k](0) + J_C(1) * M1[k](1, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](1) + J_C(3) * M2[k](0, 1) * d1[k](1));
            dF(4) -= s * (J_C(0) * M1[k](1, 0) * d2[k](1) + J_C(1) * M1[k](1, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](1) + J_C(3) * M2[k](1, 1) * d1[k](1));
            dF(5) -= s * (J_C(0) * M1[k](1, 0) * d2[k](2) + J_C(1) * M1[k](1, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](1) + J_C(3) * M2[k](2, 1) * d1[k](1));
            dF(6) -= s * (J_C(0) * M1[k](2, 0) * d2[k](0) + J_C(1) * M1[k](2, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](2) + J_C(3) * M2[k](0, 1) * d1[k](2));
            dF(7) -= s * (J_C(0) * M1[k](2, 0) * d2[k](1) + J_C(1) * M1[k](2, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](2) + J_C(3) * M2[k](1, 1) * d1[k](2));
            dF(8) -= s * (J_C(0) * M1[k](2, 0) * d2[k](2) + J_C(1) * M1[k](2, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](2) + J_C(3) * M2[k](2, 1) * d1[k](2));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J(1, this->num_params);
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
                // Jacobian w.r.t. unprojected points
                Eigen::Matrix<double, 1, 3> J_d1, J_d2;
                J_d1 = (d2[k].transpose() * E -
                        C * inv_nJ_C * inv_nJ_C * (d1[k].transpose() * E.transpose() * M2[k] * M2[k].transpose() * E)) *
                       inv_nJ_C;
                J_d2 = (d1[k].transpose() * E.transpose() -
                        C * inv_nJ_C * inv_nJ_C * (d2[k].transpose() * E * M1[k] * M1[k].transpose() * E.transpose())) *
                       inv_nJ_C;

                // Jacobian w.r.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J_M11, J_M12, J_M21, J_M22;
                J_M11 = -s * inv_nJ_C * M1[k].col(0).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M12 = -s * inv_nJ_C * M1[k].col(1).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M21 = -s * inv_nJ_C * M2[k].col(0).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();
                J_M22 = -s * inv_nJ_C * M2[k].col(1).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x1[k](0) + eps, x1[k](1);
                x_e2 << x1[k](0), x1[k](1) + eps;
                camera1.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera1.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d1_p[k];
                dp_e1 /= eps;
                dp_e2 -= d1_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                    J(0, 5 + i) = J_d1.dot(d1_p[k].col(camera1_refine_idx[i])) +
                                  J_M11.dot(dp_e1.col(camera1_refine_idx[i])) +
                                  J_M12.dot(dp_e2.col(camera1_refine_idx[i]));
                }

                // For second camera
                x_e1 << x2[k](0) + eps, x2[k](1);
                x_e2 << x2[k](0), x2[k](1) + eps;
                camera2.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera2.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d2_p[k];
                dp_e1 /= eps;
                dp_e2 -= d2_p[k];
                dp_e2 /= eps;

                if (shared_intrinsics) {
                    for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                        J(0, 5 + i) += J_d2.dot(d2_p[k].col(camera1_refine_idx[i])) +
                                       J_M21.dot(dp_e1.col(camera1_refine_idx[i])) +
                                       J_M22.dot(dp_e2.col(camera1_refine_idx[i]));
                    }
                } else {
                    for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                        J(0, 5 + camera1_refine_idx.size() + i) = J_d2.dot(d2_p[k].col(camera2_refine_idx[i])) +
                                                                  J_M21.dot(dp_e1.col(camera2_refine_idx[i])) +
                                                                  J_M22.dot(dp_e2.col(camera2_refine_idx[i]));
                    }
                }
            }

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    ImagePair step(const Eigen::VectorXd &dp, const ImagePair &pair) const {
        ImagePair image_pair_new;
        image_pair_new.camera1 = pair.camera1;
        image_pair_new.camera2 = pair.camera2;

        image_pair_new.pose.q = quat_step_post(pair.pose.q, dp.block<3, 1>(0, 0));
        image_pair_new.pose.t = pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        if (shared_intrinsics) {
            // We have shared intrinsics for both cameras
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(5 + i);
            }
            image_pair_new.camera2 = image_pair_new.camera1;
        } else {
            // Update intrinsics for first camera
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(5 + i);
            }
            // and second camera
            for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                image_pair_new.camera2.params[camera2_refine_idx[i]] += dp(5 + camera1_refine_idx.size() + i);
            }
        }
        return image_pair_new;
    }

    typedef ImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<size_t> camera1_refine_idx;
    const std::vector<size_t> camera2_refine_idx;
    const bool shared_intrinsics;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;

    // Pre-allocated vectors for undistortion
    std::vector<Point3D> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2;
    std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> d1_p, d2_p;
};

// Minimize Tangent Sampson error with any camera model. Allows for optimization of camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class ThreeViewSharedCameraRefiner : public RefinerBase<ImageTriplet, Accumulator> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    ThreeViewSharedCameraRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                 const std::vector<Point2D> &points2D_3,
                                 const std::vector<size_t> &cam_ref_idx,
                                 double alpha, double weight_alpha,
                                 const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), x3(points2D_3), camera_refine_idx(cam_ref_idx), alpha(alpha),
        weight_alpha(weight_alpha), weights(w) {
        this->num_params = 11 + cam_ref_idx.size();
        d1.reserve(x1.size());
        d2.reserve(x2.size());
        d3.reserve(x2.size());
        M1.reserve(x1.size());
        M2.reserve(x2.size());
        M3.reserve(x2.size());
        d1_p.reserve(x1.size());
        d2_p.reserve(x2.size());
        d3_p.reserve(x2.size());
    }

    double compute_residual(Accumulator &acc, const ImageTriplet &image_triplet) {
        Eigen::Matrix3d E12, E13, E23;
        essential_from_motion(image_triplet.poses.pose12, &E12);
        essential_from_motion(image_triplet.poses.pose13, &E13);
        essential_from_motion(image_triplet.poses.pose23(), &E23);

        const Camera &camera = image_triplet.camera;

        camera.unproject_with_jac(x1, &d1, &M1);
        camera.unproject_with_jac(x2, &d2, &M2);
        camera.unproject_with_jac(x3, &d3, &M3);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E12 * d1[k]);
            double nJc_sq = (M2[k].transpose() * E12 * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E12.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);

            C = d3[k].dot(E13 * d1[k]);
            nJc_sq = (M3[k].transpose() * E13 * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E13.transpose() * d3[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);

            C = d3[k].dot(E23 * d2[k]);
            nJc_sq = (M3[k].transpose() * E23 * d2[k]).squaredNorm() +
                            (M2[k].transpose() * E23.transpose() * d3[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }

        Eigen::Vector3d t13 = image_triplet.poses.pose13.t;
//        acc.add_residual(t13.norm() - alpha, weight_alpha * acc.residual_count);
        acc.add_residual(t13.norm() - alpha, weight_alpha * d1.size());

        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImageTriplet &image_triplet) {
        const CameraPose &pose12 = image_triplet.poses.pose12;
        const CameraPose &pose13 = image_triplet.poses.pose13;
        const CameraPose &pose23 = image_triplet.poses.pose23();
        setup_tangent_basis(pose12.t, tangent_basis);

        Eigen::Matrix3d E12, E13, E23, R12, R13, R23;
        R12 = pose12.R();
        R13 = pose13.R();
        R23 = pose23.R();
        essential_from_motion(pose12, &E12);
        essential_from_motion(pose13, &E13);
        essential_from_motion(pose23, &E23);

        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE12dr12;
        Eigen::Matrix<double, 9, 2> dE12dt12;

        // Each column is vec(E12*skew(e_k)) where e_k is k:th basis vector
        dE12dr12.block<3, 1>(0, 0).setZero();
        dE12dr12.block<3, 1>(0, 1) = -E12.col(2);
        dE12dr12.block<3, 1>(0, 2) = E12.col(1);
        dE12dr12.block<3, 1>(3, 0) = E12.col(2);
        dE12dr12.block<3, 1>(3, 1).setZero();
        dE12dr12.block<3, 1>(3, 2) = -E12.col(0);
        dE12dr12.block<3, 1>(6, 0) = -E12.col(1);
        dE12dr12.block<3, 1>(6, 1) = E12.col(0);
        dE12dr12.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R12)
        dE12dt12.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R12.col(0));
        dE12dt12.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R12.col(0));
        dE12dt12.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R12.col(1));
        dE12dt12.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R12.col(1));
        dE12dt12.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R12.col(2));
        dE12dt12.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R12.col(2));

        // Matrices contain the jacobians of E12 w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dE13dr13;
        Eigen::Matrix<double, 9, 3> dE13dt13;

        // Each column is vec(E13*skew(e_k)) where e_k is k:th basis vector
        dE13dr13.block<3, 1>(0, 0).setZero();
        dE13dr13.block<3, 1>(0, 1) = -E13.col(2);
        dE13dr13.block<3, 1>(0, 2) = E13.col(1);
        dE13dr13.block<3, 1>(3, 0) = E13.col(2);
        dE13dr13.block<3, 1>(3, 1).setZero();
        dE13dr13.block<3, 1>(3, 2) = -E13.col(0);
        dE13dr13.block<3, 1>(6, 0) = -E13.col(1);
        dE13dr13.block<3, 1>(6, 1) = E13.col(0);
        dE13dr13.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(e_k)*R13) where e_k is k:th basis vector
        Eigen::Matrix3d dE13dt13_0, dE13dt13_1, dE13dt13_2;
        dE13dt13_0.row(0).setZero();
        dE13dt13_0.row(1) = -R13.row(2);
        dE13dt13_0.row(2) = R13.row(1);

        dE13dt13_1.row(0) = R13.row(2);
        dE13dt13_1.row(1).setZero();
        dE13dt13_1.row(2) = - R13.row(0);

        dE13dt13_2.row(0) = - R13.row(1);
        dE13dt13_2.row(1) = R13.row(0);
        dE13dt13_2.row(2).setZero();

        dE13dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE13dt13_0.data(), dE13dt13_0.size());
        dE13dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE13dt13_1.data(), dE13dt13_1.size());
        dE13dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE13dt13_2.data(), dE13dt13_2.size());

        //TODO: this part calculates dE23dX and is not optimized yet

        // define skew(e_k)
        Eigen::Matrix3d b_0 = skew(Eigen::Vector3d::UnitX());
        Eigen::Matrix3d b_1 = skew(Eigen::Vector3d::UnitY());
        Eigen::Matrix3d b_2 = skew(Eigen::Vector3d::UnitZ());

        Eigen::Matrix3d dE23dr12_0, dE23dr12_1, dE23dr12_2;
        dE23dr12_0 = skew(R13 * b_0 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_0 * R12.transpose();
        dE23dr12_1 = skew(R13 * b_1 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_1 * R12.transpose();
        dE23dr12_2 = skew(R13 * b_2 * R12.transpose() * pose12.t) * R23 - skew(pose23.t) * R13 * b_2 * R12.transpose();

        Eigen::Matrix3d dE23dr13_0, dE23dr13_1, dE23dr13_2;
        dE23dr13_0 = - dE23dr12_0;
        dE23dr13_1 = - dE23dr12_1;
        dE23dr13_2 = - dE23dr12_2;

        // dE23dt12 = skew(tangent_basis_k) * R23
        Eigen::Matrix3d dE23dt12_0, dE23dt12_1;
        dE23dt12_0 = - skew(R23 * tangent_basis.col(0)) * R23;
        dE23dt12_1 = - skew(R23 * tangent_basis.col(1)) * R23;

        // dE23dt13 = skew(e_k) * R23
        Eigen::Matrix3d dE23dt13_0, dE23dt13_1, dE23dt13_2;
        dE23dt13_0.row(0).setZero();
        dE23dt13_0.row(1) = -R23.row(2);
        dE23dt13_0.row(2) = R23.row(1);

        dE23dt13_1.row(0) = R23.row(2);
        dE23dt13_1.row(1).setZero();
        dE23dt13_1.row(2) = - R23.row(0);

        dE23dt13_2.row(0) = - R23.row(1);
        dE23dt13_2.row(1) = R23.row(0);
        dE23dt13_2.row(2).setZero();

        Eigen::Matrix<double, 9, 3> dE23dr12;
        Eigen::Matrix<double, 9, 3> dE23dr13;
        Eigen::Matrix<double, 9, 2> dE23dt12;
        Eigen::Matrix<double, 9, 3> dE23dt13;

        dE23dr12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr12_0.data(), dE23dr12_0.size());
        dE23dr12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr12_1.data(), dE23dr12_1.size());
        dE23dr12.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr12_2.data(), dE23dr12_2.size());

        dE23dr13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dr13_0.data(), dE23dr13_0.size());
        dE23dr13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dr13_1.data(), dE23dr13_1.size());
        dE23dr13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dr13_2.data(), dE23dr13_2.size());

        dE23dt12.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt12_0.data(), dE23dt12_0.size());
        dE23dt12.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt12_1.data(), dE23dt12_1.size());

        dE23dt13.col(0) = Eigen::Map<Eigen::VectorXd>(dE23dt13_0.data(), dE23dt13_0.size());
        dE23dt13.col(1) = Eigen::Map<Eigen::VectorXd>(dE23dt13_1.data(), dE23dt13_1.size());
        dE23dt13.col(2) = Eigen::Map<Eigen::VectorXd>(dE23dt13_2.data(), dE23dt13_2.size());

        const Camera &camera = image_triplet.camera;

        if (camera_refine_idx.size() > 0) {
            camera.unproject_with_jac(x1, &d1, &M1, &d1_p);
            camera.unproject_with_jac(x2, &d2, &M2, &d2_p);
            camera.unproject_with_jac(x3, &d3, &M3, &d3_p);
        } else {
            camera.unproject_with_jac(x1, &d1, &M1);
            camera.unproject_with_jac(x2, &d2, &M2);
            camera.unproject_with_jac(x3, &d3, &M3);
        }

        for (size_t k = 0; k < x1.size(); ++k) {
            double C12 = d2[k].dot(E12 * d1[k]);

            // J12_C is the Jacobian of the epipolar constraint w.r12.t. the image points
            Eigen::Vector4d J12_C;
            J12_C << M1[k].transpose() * E12.transpose() * d2[k], M2[k].transpose() * E12 * d1[k];
            const double nJ12_C = J12_C.norm();
            const double inv_nJ12_C = 1.0 / nJ12_C;
            const double r12 = C12 * inv_nJ12_C;

            // Compute Jacobian of Sampson error w.r12.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF12;
            dF12 << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s12 = C12 * inv_nJ12_C * inv_nJ12_C;
            dF12(0) -= s12 * (J12_C(0) * M1[k](0, 0) * d2[k](0) + J12_C(1) * M1[k](0, 1) * d2[k](0) +
                          J12_C(2) * M2[k](0, 0) * d1[k](0) + J12_C(3) * M2[k](0, 1) * d1[k](0));
            dF12(1) -= s12 * (J12_C(0) * M1[k](0, 0) * d2[k](1) + J12_C(1) * M1[k](0, 1) * d2[k](1) +
                          J12_C(2) * M2[k](1, 0) * d1[k](0) + J12_C(3) * M2[k](1, 1) * d1[k](0));
            dF12(2) -= s12 * (J12_C(0) * M1[k](0, 0) * d2[k](2) + J12_C(1) * M1[k](0, 1) * d2[k](2) +
                          J12_C(2) * M2[k](2, 0) * d1[k](0) + J12_C(3) * M2[k](2, 1) * d1[k](0));
            dF12(3) -= s12 * (J12_C(0) * M1[k](1, 0) * d2[k](0) + J12_C(1) * M1[k](1, 1) * d2[k](0) +
                          J12_C(2) * M2[k](0, 0) * d1[k](1) + J12_C(3) * M2[k](0, 1) * d1[k](1));
            dF12(4) -= s12 * (J12_C(0) * M1[k](1, 0) * d2[k](1) + J12_C(1) * M1[k](1, 1) * d2[k](1) +
                          J12_C(2) * M2[k](1, 0) * d1[k](1) + J12_C(3) * M2[k](1, 1) * d1[k](1));
            dF12(5) -= s12 * (J12_C(0) * M1[k](1, 0) * d2[k](2) + J12_C(1) * M1[k](1, 1) * d2[k](2) +
                          J12_C(2) * M2[k](2, 0) * d1[k](1) + J12_C(3) * M2[k](2, 1) * d1[k](1));
            dF12(6) -= s12 * (J12_C(0) * M1[k](2, 0) * d2[k](0) + J12_C(1) * M1[k](2, 1) * d2[k](0) +
                          J12_C(2) * M2[k](0, 0) * d1[k](2) + J12_C(3) * M2[k](0, 1) * d1[k](2));
            dF12(7) -= s12 * (J12_C(0) * M1[k](2, 0) * d2[k](1) + J12_C(1) * M1[k](2, 1) * d2[k](1) +
                          J12_C(2) * M2[k](1, 0) * d1[k](2) + J12_C(3) * M2[k](1, 1) * d1[k](2));
            dF12(8) -= s12 * (J12_C(0) * M1[k](2, 0) * d2[k](2) + J12_C(1) * M1[k](2, 1) * d2[k](2) +
                          J12_C(2) * M2[k](2, 0) * d1[k](2) + J12_C(3) * M2[k](2, 1) * d1[k](2));
            dF12 *= inv_nJ12_C;

            // and then w.r.t. the pose12 parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J12(1, this->num_params);
            J12.setZero();
            J12.block<1, 3>(0, 0) = dF12 * dE12dr12;
            J12.block<1, 2>(0, 3) = dF12 * dE12dt12;

            if (camera_refine_idx.size() > 0) {
                // Jacobian w.r12.t. unprojected points
                Eigen::Matrix<double, 1, 3> J12_d1, J12_d2;
                J12_d1 = (d2[k].transpose() * E12 -
                           C12 * inv_nJ12_C * inv_nJ12_C * (d1[k].transpose() * E12.transpose() * M2[k] * M2[k].transpose() * E12)) *
                         inv_nJ12_C;
                J12_d2 = (d1[k].transpose() * E12.transpose() -
                           C12 * inv_nJ12_C * inv_nJ12_C * (d2[k].transpose() * E12 * M1[k] * M1[k].transpose() * E12.transpose())) *
                         inv_nJ12_C;

                // Jacobian w.r12.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J12_M11, J12_M12, J12_M21, J12_M22;
                J12_M11 = -s12 * inv_nJ12_C * M1[k].col(0).transpose() * E12.transpose() * d2[k] * d2[k].transpose() * E12;
                J12_M12 = -s12 * inv_nJ12_C * M1[k].col(1).transpose() * E12.transpose() * d2[k] * d2[k].transpose() * E12;
                J12_M21 = -s12 * inv_nJ12_C * M2[k].col(0).transpose() * E12 * d1[k] * d1[k].transpose() * E12.transpose();
                J12_M22 = -s12 * inv_nJ12_C * M2[k].col(1).transpose() * E12 * d1[k] * d1[k].transpose() * E12.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r12.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x1[k](0) + eps, x1[k](1);
                x_e2 << x1[k](0), x1[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d1_p[k];
                dp_e1 /= eps;
                dp_e2 -= d1_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J12(0, 11 + i) = J12_d1.dot(d1_p[k].col(camera_refine_idx[i])) +
                                    J12_M11.dot(dp_e1.col(camera_refine_idx[i])) +
                                    J12_M12.dot(dp_e2.col(camera_refine_idx[i]));
                }

                // For second camera
                x_e1 << x2[k](0) + eps, x2[k](1);
                x_e2 << x2[k](0), x2[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d2_p[k];
                dp_e1 /= eps;
                dp_e2 -= d2_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J12(0, 11 + i) += J12_d2.dot(d2_p[k].col(camera_refine_idx[i])) +
                                      J12_M21.dot(dp_e1.col(camera_refine_idx[i])) +
                                      J12_M22.dot(dp_e2.col(camera_refine_idx[i]));
                }
            }
            
            acc.add_jacobian(r12, J12, weights[k]);
                      
            
            double C13 = d3[k].dot(E13 * d1[k]);

            // J13_C is the Jacobian of the epipolar constraint w.r13.t. the image points
            Eigen::Vector4d J13_C;
            J13_C << M1[k].transpose() * E13.transpose() * d3[k], M3[k].transpose() * E13 * d1[k];
            const double nJ13_C = J13_C.norm();
            const double inv_nJ13_C = 1.0 / nJ13_C;
            const double r13 = C13 * inv_nJ13_C;

            // Compute Jacobian of Sampson error w.r13.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF13;
            dF13 << d1[k](0) * d3[k](0), d1[k](0) * d3[k](1), d1[k](0) * d3[k](2), d1[k](1) * d3[k](0),
                d1[k](1) * d3[k](1), d1[k](1) * d3[k](2), d1[k](2) * d3[k](0), d1[k](2) * d3[k](1), d1[k](2) * d3[k](2);
            const double s13 = C13 * inv_nJ13_C * inv_nJ13_C;
            dF13(0) -= s13 * (J13_C(0) * M1[k](0, 0) * d3[k](0) + J13_C(1) * M1[k](0, 1) * d3[k](0) +
                          J13_C(2) * M3[k](0, 0) * d1[k](0) + J13_C(3) * M3[k](0, 1) * d1[k](0));
            dF13(1) -= s13 * (J13_C(0) * M1[k](0, 0) * d3[k](1) + J13_C(1) * M1[k](0, 1) * d3[k](1) +
                          J13_C(2) * M3[k](1, 0) * d1[k](0) + J13_C(3) * M3[k](1, 1) * d1[k](0));
            dF13(2) -= s13 * (J13_C(0) * M1[k](0, 0) * d3[k](2) + J13_C(1) * M1[k](0, 1) * d3[k](2) +
                          J13_C(2) * M3[k](2, 0) * d1[k](0) + J13_C(3) * M3[k](2, 1) * d1[k](0));
            dF13(3) -= s13 * (J13_C(0) * M1[k](1, 0) * d3[k](0) + J13_C(1) * M1[k](1, 1) * d3[k](0) +
                          J13_C(2) * M3[k](0, 0) * d1[k](1) + J13_C(3) * M3[k](0, 1) * d1[k](1));
            dF13(4) -= s13 * (J13_C(0) * M1[k](1, 0) * d3[k](1) + J13_C(1) * M1[k](1, 1) * d3[k](1) +
                          J13_C(2) * M3[k](1, 0) * d1[k](1) + J13_C(3) * M3[k](1, 1) * d1[k](1));
            dF13(5) -= s13 * (J13_C(0) * M1[k](1, 0) * d3[k](2) + J13_C(1) * M1[k](1, 1) * d3[k](2) +
                          J13_C(2) * M3[k](2, 0) * d1[k](1) + J13_C(3) * M3[k](2, 1) * d1[k](1));
            dF13(6) -= s13 * (J13_C(0) * M1[k](2, 0) * d3[k](0) + J13_C(1) * M1[k](2, 1) * d3[k](0) +
                          J13_C(2) * M3[k](0, 0) * d1[k](2) + J13_C(3) * M3[k](0, 1) * d1[k](2));
            dF13(7) -= s13 * (J13_C(0) * M1[k](2, 0) * d3[k](1) + J13_C(1) * M1[k](2, 1) * d3[k](1) +
                          J13_C(2) * M3[k](1, 0) * d1[k](2) + J13_C(3) * M3[k](1, 1) * d1[k](2));
            dF13(8) -= s13 * (J13_C(0) * M1[k](2, 0) * d3[k](2) + J13_C(1) * M1[k](2, 1) * d3[k](2) +
                          J13_C(2) * M3[k](2, 0) * d1[k](2) + J13_C(3) * M3[k](2, 1) * d1[k](2));
            dF13 *= inv_nJ13_C;

            // and then w.r.t. the pose13 parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J13(1, this->num_params);
            J13.setZero();
            J13.block<1, 3>(0, 5) = dF13 * dE13dr13;
            J13.block<1, 3>(0, 8) = dF13 * dE13dt13;

            if (camera_refine_idx.size() > 0) {
                // Jacobian w.r13.t. unprojected points
                Eigen::Matrix<double, 1, 3> J13_d1, J13_d3;
                J13_d1 = (d3[k].transpose() * E13 -
                           C13 * inv_nJ13_C * inv_nJ13_C * (d1[k].transpose() * E13.transpose() * M3[k] * M3[k].transpose() * E13)) *
                         inv_nJ13_C;
                J13_d3 = (d1[k].transpose() * E13.transpose() -
                           C13 * inv_nJ13_C * inv_nJ13_C * (d3[k].transpose() * E13 * M1[k] * M1[k].transpose() * E13.transpose())) *
                         inv_nJ13_C;

                // Jacobian w.r13.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J13_M11, J13_M13, J13_M31, J13_M32;
                J13_M11 = -s13 * inv_nJ13_C * M1[k].col(0).transpose() * E13.transpose() * d3[k] * d3[k].transpose() * E13;
                J13_M13 = -s13 * inv_nJ13_C * M1[k].col(1).transpose() * E13.transpose() * d3[k] * d3[k].transpose() * E13;
                J13_M31 = -s13 * inv_nJ13_C * M3[k].col(0).transpose() * E13 * d1[k] * d1[k].transpose() * E13.transpose();
                J13_M32 = -s13 * inv_nJ13_C * M3[k].col(1).transpose() * E13 * d1[k] * d1[k].transpose() * E13.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r13.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x1[k](0) + eps, x1[k](1);
                x_e2 << x1[k](0), x1[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d1_p[k];
                dp_e1 /= eps;
                dp_e2 -= d1_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J13(0, 11 + i) = J13_d1.dot(d1_p[k].col(camera_refine_idx[i])) +
                                    J13_M11.dot(dp_e1.col(camera_refine_idx[i])) +
                                    J13_M13.dot(dp_e2.col(camera_refine_idx[i]));
                }

                // For second camera
                x_e1 << x3[k](0) + eps, x3[k](1);
                x_e2 << x3[k](0), x3[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d3_p[k];
                dp_e1 /= eps;
                dp_e2 -= d3_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J13(0, 11 + i) += J13_d3.dot(d3_p[k].col(camera_refine_idx[i])) +
                                      J13_M31.dot(dp_e1.col(camera_refine_idx[i])) +
                                      J13_M32.dot(dp_e2.col(camera_refine_idx[i]));
                }
            }
            
            acc.add_jacobian(r13, J13, weights[k]);



            double C23 = d3[k].dot(E23 * d2[k]);

            // J23_C is the Jacobian of the epipolar constraint w.r23.t. the image points
            Eigen::Vector4d J23_C;
            J23_C << M2[k].transpose() * E23.transpose() * d3[k], M3[k].transpose() * E23 * d2[k];
            const double nJ23_C = J23_C.norm();
            const double inv_nJ23_C = 1.0 / nJ23_C;
            const double r23 = C23 * inv_nJ23_C;

            // Compute Jacobian of Sampson error w.r23.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF23;
            dF23 << d2[k](0) * d3[k](0), d2[k](0) * d3[k](1), d2[k](0) * d3[k](2), d2[k](1) * d3[k](0),
                    d2[k](1) * d3[k](1), d2[k](1) * d3[k](2), d2[k](2) * d3[k](0), d2[k](2) * d3[k](1), d2[k](2) * d3[k](2);
            const double s23 = C23 * inv_nJ23_C * inv_nJ23_C;
            dF23(0) -= s23 * (J23_C(0) * M2[k](0, 0) * d3[k](0) + J23_C(1) * M2[k](0, 1) * d3[k](0) +
                              J23_C(2) * M3[k](0, 0) * d2[k](0) + J23_C(3) * M3[k](0, 1) * d2[k](0));
            dF23(1) -= s23 * (J23_C(0) * M2[k](0, 0) * d3[k](1) + J23_C(1) * M2[k](0, 1) * d3[k](1) +
                              J23_C(2) * M3[k](1, 0) * d2[k](0) + J23_C(3) * M3[k](1, 1) * d2[k](0));
            dF23(2) -= s23 * (J23_C(0) * M2[k](0, 0) * d3[k](2) + J23_C(1) * M2[k](0, 1) * d3[k](2) +
                              J23_C(2) * M3[k](2, 0) * d2[k](0) + J23_C(3) * M3[k](2, 1) * d2[k](0));
            dF23(3) -= s23 * (J23_C(0) * M2[k](1, 0) * d3[k](0) + J23_C(1) * M2[k](1, 1) * d3[k](0) +
                              J23_C(2) * M3[k](0, 0) * d2[k](1) + J23_C(3) * M3[k](0, 1) * d2[k](1));
            dF23(4) -= s23 * (J23_C(0) * M2[k](1, 0) * d3[k](1) + J23_C(1) * M2[k](1, 1) * d3[k](1) +
                              J23_C(2) * M3[k](1, 0) * d2[k](1) + J23_C(3) * M3[k](1, 1) * d2[k](1));
            dF23(5) -= s23 * (J23_C(0) * M2[k](1, 0) * d3[k](2) + J23_C(1) * M2[k](1, 1) * d3[k](2) +
                              J23_C(2) * M3[k](2, 0) * d2[k](1) + J23_C(3) * M3[k](2, 1) * d2[k](1));
            dF23(6) -= s23 * (J23_C(0) * M2[k](2, 0) * d3[k](0) + J23_C(1) * M2[k](2, 1) * d3[k](0) +
                              J23_C(2) * M3[k](0, 0) * d2[k](2) + J23_C(3) * M3[k](0, 1) * d2[k](2));
            dF23(7) -= s23 * (J23_C(0) * M2[k](2, 0) * d3[k](1) + J23_C(1) * M2[k](2, 1) * d3[k](1) +
                              J23_C(2) * M3[k](1, 0) * d2[k](2) + J23_C(3) * M3[k](1, 1) * d2[k](2));
            dF23(8) -= s23 * (J23_C(0) * M2[k](2, 0) * d3[k](2) + J23_C(1) * M2[k](2, 1) * d3[k](2) +
                              J23_C(2) * M3[k](2, 0) * d2[k](2) + J23_C(3) * M3[k](2, 1) * d2[k](2));
            dF23 *= inv_nJ23_C;

            // and then w.r.t. the pose23 parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J23(1, this->num_params);
            J23.block<1, 3>(0, 0) = dF23 * dE23dr12;
            J23.block<1, 2>(0, 3) = dF23 * dE23dt12;
            J23.block<1, 3>(0, 5) = dF23 * dE23dr13;
            J23.block<1, 3>(0, 8) = dF23 * dE23dt13;

            if (camera_refine_idx.size() > 0) {
                // Jacobian w.r23.t. unprojected points
                Eigen::Matrix<double, 1, 3> J23_d2, J23_d3;
                J23_d2 = (d3[k].transpose() * E23 -
                          C23 * inv_nJ23_C * inv_nJ23_C * (d2[k].transpose() * E23.transpose() * M3[k] * M3[k].transpose() * E23)) *
                         inv_nJ23_C;
                J23_d3 = (d2[k].transpose() * E23.transpose() -
                          C23 * inv_nJ23_C * inv_nJ23_C * (d3[k].transpose() * E23 * M2[k] * M2[k].transpose() * E23.transpose())) *
                         inv_nJ23_C;

                // Jacobian w.r23.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J23_M21, J23_M33, J23_M31, J23_M32;
                J23_M21 = -s23 * inv_nJ23_C * M2[k].col(0).transpose() * E23.transpose() * d3[k] * d3[k].transpose() * E23;
                J23_M33 = -s23 * inv_nJ23_C * M2[k].col(1).transpose() * E23.transpose() * d3[k] * d3[k].transpose() * E23;
                J23_M31 = -s23 * inv_nJ23_C * M3[k].col(0).transpose() * E23 * d2[k] * d2[k].transpose() * E23.transpose();
                J23_M32 = -s23 * inv_nJ23_C * M3[k].col(1).transpose() * E23 * d2[k] * d2[k].transpose() * E23.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r23.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x2[k](0) + eps, x2[k](1);
                x_e2 << x2[k](0), x2[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d2_p[k];
                dp_e1 /= eps;
                dp_e2 -= d2_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J23(0, 11 + i) = J23_d2.dot(d2_p[k].col(camera_refine_idx[i])) +
                                     J23_M21.dot(dp_e1.col(camera_refine_idx[i])) +
                                     J23_M33.dot(dp_e2.col(camera_refine_idx[i]));
                }

                // For second camera
                x_e1 << x3[k](0) + eps, x3[k](1);
                x_e2 << x3[k](0), x3[k](1) + eps;
                camera.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d3_p[k];
                dp_e1 /= eps;
                dp_e2 -= d3_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
                    J23(0, 11 + i) += J23_d3.dot(d3_p[k].col(camera_refine_idx[i])) +
                                      J23_M31.dot(dp_e1.col(camera_refine_idx[i])) +
                                      J23_M32.dot(dp_e2.col(camera_refine_idx[i]));
                }
            }
            acc.add_jacobian(r23, J23, weights[k]);
        }

        // add jacobian for relative distance
        Eigen::Matrix<double, 1, Eigen::Dynamic> J_alpha(1, this->num_params);
        Eigen::Vector3d t13 = pose13.t;
         double t13_norm = t13.norm();
        J_alpha.setZero();
        J_alpha.block<1, 3>(0, 8) = t13 / t13.norm();
        acc.add_jacobian(t13_norm - alpha, J_alpha, weight_alpha * d1.size());
    }

    ImageTriplet step(const Eigen::VectorXd &dp, const ImageTriplet &image_triplet) const {
        ImageTriplet image_triplet_new;

        image_triplet_new.camera = image_triplet.camera;

        image_triplet_new.poses.pose12.q = quat_step_post(image_triplet.poses.pose12.q, dp.block<3, 1>(0, 0));
        image_triplet_new.poses.pose12.t = image_triplet.poses.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);

        image_triplet_new.poses.pose13.q = quat_step_post(image_triplet.poses.pose13.q, dp.block<3, 1>(5, 0));
        image_triplet_new.poses.pose13.t = image_triplet.poses.pose13.t + dp.block<3, 1>(8, 0);

        // We have shared intrinsics for both cameras
        for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
            image_triplet_new.camera.params[camera_refine_idx[i]] += dp(11 + i);
        }

        return image_triplet_new;
    }

    typedef ImageTriplet param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<Point2D> &x3;
    const std::vector<size_t> camera_refine_idx;
    const double alpha, weight_alpha;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;

    // Pre-allocated vectors for undistortion
    std::vector<Point3D> d1, d2, d3;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2, M3;
    std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> d1_p, d2_p, d3_p;
};

// Minimize Tangent Sampson error with any camera model. Allows for optimization of camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class CameraRelativeFixPoseRefiner : public RefinerBase<ImagePair, Accumulator> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    CameraRelativeFixPoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                              const std::vector<size_t> &cam1_ref_idx, const std::vector<size_t> &cam2_ref_idx,
                              const bool shared_camera = false, // Shared intrinsics only use camera1
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), camera1_refine_idx(cam1_ref_idx), camera2_refine_idx(cam2_ref_idx),
          shared_intrinsics(shared_camera), weights(w) {
        this->num_params = shared_intrinsics ? cam1_ref_idx.size() : cam1_ref_idx.size() + cam2_ref_idx.size();
        d1.reserve(x1.size());
        d2.reserve(x2.size());
        M1.reserve(x1.size());
        M2.reserve(x2.size());
        d1_p.reserve(x1.size());
        d2_p.reserve(x2.size());
    }

    double compute_residual(Accumulator &acc, const ImagePair &pair) {
        Eigen::Matrix3d E;
        essential_from_motion(pair.pose, &E);

        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        camera1.unproject_with_jac(x1, &d1, &M1);
        camera2.unproject_with_jac(x2, &d2, &M2);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);
            double nJc_sq = (M2[k].transpose() * E * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImagePair &pair) {
        const CameraPose &pose = pair.pose;
        setup_tangent_basis(pose.t, tangent_basis);

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
            camera1.unproject_with_jac(x1, &d1, &M1, &d1_p);
            camera2.unproject_with_jac(x2, &d2, &M2, &d2_p);
        } else {
            camera1.unproject_with_jac(x1, &d1, &M1);
            camera2.unproject_with_jac(x2, &d2, &M2);
        }

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << M1[k].transpose() * E.transpose() * d2[k], M2[k].transpose() * E * d1[k];
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(0) * M1[k](0, 0) * d2[k](0) + J_C(1) * M1[k](0, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](0) + J_C(3) * M2[k](0, 1) * d1[k](0));
            dF(1) -= s * (J_C(0) * M1[k](0, 0) * d2[k](1) + J_C(1) * M1[k](0, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](0) + J_C(3) * M2[k](1, 1) * d1[k](0));
            dF(2) -= s * (J_C(0) * M1[k](0, 0) * d2[k](2) + J_C(1) * M1[k](0, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](0) + J_C(3) * M2[k](2, 1) * d1[k](0));
            dF(3) -= s * (J_C(0) * M1[k](1, 0) * d2[k](0) + J_C(1) * M1[k](1, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](1) + J_C(3) * M2[k](0, 1) * d1[k](1));
            dF(4) -= s * (J_C(0) * M1[k](1, 0) * d2[k](1) + J_C(1) * M1[k](1, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](1) + J_C(3) * M2[k](1, 1) * d1[k](1));
            dF(5) -= s * (J_C(0) * M1[k](1, 0) * d2[k](2) + J_C(1) * M1[k](1, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](1) + J_C(3) * M2[k](2, 1) * d1[k](1));
            dF(6) -= s * (J_C(0) * M1[k](2, 0) * d2[k](0) + J_C(1) * M1[k](2, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](2) + J_C(3) * M2[k](0, 1) * d1[k](2));
            dF(7) -= s * (J_C(0) * M1[k](2, 0) * d2[k](1) + J_C(1) * M1[k](2, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](2) + J_C(3) * M2[k](1, 1) * d1[k](2));
            dF(8) -= s * (J_C(0) * M1[k](2, 0) * d2[k](2) + J_C(1) * M1[k](2, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](2) + J_C(3) * M2[k](2, 1) * d1[k](2));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J(1, this->num_params);

            if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
                // Jacobian w.r.t. unprojected points
                Eigen::Matrix<double, 1, 3> J_d1, J_d2;
                J_d1 = (d2[k].transpose() * E -
                        C * inv_nJ_C * inv_nJ_C * (d1[k].transpose() * E.transpose() * M2[k] * M2[k].transpose() * E)) *
                       inv_nJ_C;
                J_d2 = (d1[k].transpose() * E.transpose() -
                        C * inv_nJ_C * inv_nJ_C * (d2[k].transpose() * E * M1[k] * M1[k].transpose() * E.transpose())) *
                       inv_nJ_C;

                // Jacobian w.r.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J_M11, J_M12, J_M21, J_M22;
                J_M11 = -s * inv_nJ_C * M1[k].col(0).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M12 = -s * inv_nJ_C * M1[k].col(1).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M21 = -s * inv_nJ_C * M2[k].col(0).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();
                J_M22 = -s * inv_nJ_C * M2[k].col(1).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x1[k](0) + eps, x1[k](1);
                x_e2 << x1[k](0), x1[k](1) + eps;
                camera1.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera1.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d1_p[k];
                dp_e1 /= eps;
                dp_e2 -= d1_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                    J(0, i) = J_d1.dot(d1_p[k].col(camera1_refine_idx[i])) +
                                  J_M11.dot(dp_e1.col(camera1_refine_idx[i])) +
                                  J_M12.dot(dp_e2.col(camera1_refine_idx[i]));
                }

                // For second camera
                x_e1 << x2[k](0) + eps, x2[k](1);
                x_e2 << x2[k](0), x2[k](1) + eps;
                camera2.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera2.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d2_p[k];
                dp_e1 /= eps;
                dp_e2 -= d2_p[k];
                dp_e2 /= eps;

                if (shared_intrinsics) {
                    for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                        J(0, i) += J_d2.dot(d2_p[k].col(camera1_refine_idx[i])) +
                                   J_M21.dot(dp_e1.col(camera1_refine_idx[i])) +
                                   J_M22.dot(dp_e2.col(camera1_refine_idx[i]));
                    }
                } else {
                    for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                        J(0, camera1_refine_idx.size() + i) = J_d2.dot(d2_p[k].col(camera2_refine_idx[i])) +
                                                              J_M21.dot(dp_e1.col(camera2_refine_idx[i])) +
                                                              J_M22.dot(dp_e2.col(camera2_refine_idx[i]));
                    }
                }
            }

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    ImagePair step(const Eigen::VectorXd &dp, const ImagePair &pair) const {
        ImagePair image_pair_new;
        image_pair_new.camera1 = pair.camera1;
        image_pair_new.camera2 = pair.camera2;

        image_pair_new.pose.q = pair.pose.q;
        image_pair_new.pose.t = pair.pose.t;

        if (shared_intrinsics) {
            // We have shared intrinsics for both cameras
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(i);
            }
            image_pair_new.camera2 = image_pair_new.camera1;
        } else {
            // Update intrinsics for first camera
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(i);
            }
            // and second camera
            for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                image_pair_new.camera2.params[camera2_refine_idx[i]] += dp(camera1_refine_idx.size() + i);
            }
        }
        return image_pair_new;
    }

    typedef ImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<size_t> camera1_refine_idx;
    const std::vector<size_t> camera2_refine_idx;
    const bool shared_intrinsics;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;

    // Pre-allocated vectors for undistortion
    std::vector<Point3D> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2;
    std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> d1_p, d2_p;
};

// Minimize Sampson error with pinhole camera model for relative pose and one unknown focal length shared by both
// cameras.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class SharedFocalRelativePoseRefiner : public RefinerBase<ImagePair, Accumulator> {
  public:
    SharedFocalRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 6;
    }

    double compute_residual(Accumulator &acc, const ImagePair &image_pair) {
        Eigen::Matrix3d E, F;
        essential_from_motion(image_pair.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, image_pair.camera1.focal());
        F = K_inv * E * K_inv;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImagePair &image_pair) {
        setup_tangent_basis(image_pair.pose.t, tangent_basis);

        Eigen::Matrix3d E, F, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        double focal = image_pair.camera1.focal();
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        F = K_inv * E * K_inv;

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;
        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(5) = dF * df;

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    ImagePair step(const Eigen::VectorXd &dp, const ImagePair &image_pair) const {
        CameraPose new_pose;
        new_pose.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        new_pose.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera new_camera = Camera("SIMPLE_PINHOLE", {image_pair.camera1.focal() + dp(5), 0, 0}, -1, -1);

        return ImagePair(new_pose, new_camera, new_camera);
    }

    typedef ImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};
} // namespace poselib

#endif