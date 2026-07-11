#include "relpose_reldepth_3pt.h"
#include "PoseLib/misc/univariate.h"

#include <cmath>

namespace poselib {

int essential_3pt_rel_depth_impl(const std::vector<Eigen::Vector3d> &x1h, const std::vector<Eigen::Vector3d> &x2h,
                                 const std::vector<double> &sigma, std::vector<MonoDepthTwoViewGeometry> *models) {

    static const double TOL_IMAG = 1e-1;

    const double c0 = sigma[0] * sigma[0], c1 = sigma[1] * sigma[1], c2 = x1h[0].squaredNorm(),
                 c3 = x1h[1].squaredNorm(), c4 = x2h[0].squaredNorm() * c0,
                 c5 = x2h[1].squaredNorm() * c1,
                 c6 = 2 * sigma[0] * sigma[1] * x2h[0].transpose() * x2h[1],
                 c7 = 2 * x1h[0].transpose() * x1h[1],
                 b0 = 2 * sigma[0] * x2h[0].transpose() * x2h[2],
                 b1 = 2 * x1h[0].transpose() * x1h[2],
                 b2 = 2 * sigma[1] * x2h[1].transpose() * x2h[2],
                 b3 = 2 * x1h[1].transpose() * x1h[2], b4 = x1h[2].squaredNorm(),
                 b5 = x2h[2].squaredNorm();

    
    std::complex<double> lambda1[2];
    univariate::solve_quadratic(c5 - c3, c7 - c6, c4 - c2, lambda1);

    
    size_t num_sols = 0;
    for (int k = 0; k < 2; ++k) {
        if(std::abs(lambda1[k].imag()) >= TOL_IMAG)
            continue; // Imaginary solution
        const double lambda1_real = lambda1[k].real();
        if(lambda1_real < 0)
            continue; // Negative depth

        const double a0 = b2 * lambda1_real - b0, a1 = (b3 * lambda1_real - b1) / a0,
                     a2 = (c2 - c4 + (c5 - c3) * lambda1_real * lambda1_real) / a0;

        std::complex<double> lambda2[2];
        univariate::solve_quadratic(b5 * a1 * a1 - b4, b1 - a1 * b0 + 2 * a1 * a2 * b5,
                                    b5 * a2 * a2 - b0 * a2 - c2 + c4, lambda2);

        for (int m = 0; m < 2; ++m) {
            if(std::abs(lambda2[m].imag()) >= TOL_IMAG)
                continue;
            const double lambda2_real = lambda2[m].real();
            if(lambda2_real < 0)
                continue;

            double lambda2s = a1 * lambda2_real + a2;
            if(lambda2s < 0)
                continue;

            Eigen::Vector3d v1 = sigma[0] * x2h[0] - sigma[1] * lambda1_real * x2h[1];
            Eigen::Vector3d v2 = sigma[0] * x2h[0] - lambda2s * x2h[2];
            Eigen::Matrix3d Y;
            Y << v1, v2, v1.cross(v2);

            Eigen::Vector3d u1 = x1h[0] - lambda1_real * x1h[1];
            Eigen::Vector3d u2 = x1h[0] - lambda2_real * x1h[2];
            Eigen::Matrix3d X;
            X << u1, u2, u1.cross(u2);
            X = X.inverse().eval();

            Eigen::Matrix3d rot = Y * X;
            double det_rot = rot.determinant();
            rot /= std::cbrt(det_rot);

            CameraPose pose;
            Eigen::Quaterniond q_flip(rot);

            pose.q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
            pose.t = sigma[0] * x2h[0] - pose.rotate(x1h[0]);
            pose.t.normalize();
            models->emplace_back(pose);

            num_sols++;
        }
    }

    return num_sols;
}


int essential_3pt_relative_depth(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                                 const std::vector<double> &d1, const std::vector<double> &d2,
                                 std::vector<MonoDepthTwoViewGeometry> *models, bool all_permutations) {
    models->clear();
    std::vector<double> sigma(3);
    for (size_t i = 0; i < 3; ++i) {
        sigma[i] = d2[i] / d1[i];
    }
    essential_3pt_rel_depth_impl(x1, x2, sigma, models);

    if(all_permutations) {
        std::vector<Eigen::Vector3d> x1_copy = x1;
        std::vector<Eigen::Vector3d> x2_copy = x2;
        std::vector<double> sigma_copy = sigma;

        // [0 1 2] -> [2 1 0]
        std::swap(x1_copy[0], x1_copy[2]);
        std::swap(x2_copy[0], x2_copy[2]);
        std::swap(sigma_copy[0], sigma_copy[2]);
        essential_3pt_rel_depth_impl(x1_copy, x2_copy, sigma_copy, models);

        // [2 1 0] -> [2 0 1]
        std::swap(x1_copy[1], x1_copy[2]);
        std::swap(x2_copy[1], x2_copy[2]);
        std::swap(sigma_copy[1], sigma_copy[2]);
        essential_3pt_rel_depth_impl(x1_copy, x2_copy, sigma_copy, models);
    }
    return models->size();
}

} // namespace poselib
