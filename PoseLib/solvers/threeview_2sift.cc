#include "threeview_2sift.h"

#include <Eigen/Dense>
#include <complex>

namespace poselib {

namespace {

// Generated action matrix solver (theory/solver_rotation_2sift_orientation.cpp on the mlde branch initial). Solves
//     det([a_i, K(x) b_i, K(y) c_i]) = 0, i = 0, 1,
// with K(k) = [1-k^2, 0, 2k; 0, 1+k^2, 0; -2k, 0, 1-k^2] / (1+k^2) the Cayley parametrized rotation about the y axis.
// Input d = [a_0 a_1 b_0 b_1 c_0 c_1] (18 values). Returns up to 8 real solutions (x, y).
void solve_rotation(const double *d, std::vector<Eigen::Vector2d> *sols) {
    sols->clear();
    double coeffs[18];
    coeffs[0] = d[2] * d[7] * d[12] + d[1] * d[8] * d[12] - d[2] * d[6] * d[13] + d[0] * d[8] * d[13] -
                d[1] * d[6] * d[14] - d[0] * d[7] * d[14];
    coeffs[1] = -2 * d[1] * d[6] * d[12] - 2 * d[0] * d[7] * d[12] - 2 * d[2] * d[7] * d[14] - 2 * d[1] * d[8] * d[14];
    coeffs[2] = 2 * d[1] * d[6] * d[12] + 2 * d[0] * d[6] * d[13] + 2 * d[2] * d[8] * d[13] + 2 * d[1] * d[8] * d[14];
    coeffs[3] = -d[2] * d[7] * d[12] - d[1] * d[8] * d[12] - d[2] * d[6] * d[13] + d[0] * d[8] * d[13] +
                d[1] * d[6] * d[14] + d[0] * d[7] * d[14];
    coeffs[4] = 4 * d[1] * d[8] * d[12] - 4 * d[1] * d[6] * d[14];
    coeffs[5] = d[2] * d[7] * d[12] - d[1] * d[8] * d[12] + d[2] * d[6] * d[13] - d[0] * d[8] * d[13] +
                d[1] * d[6] * d[14] - d[0] * d[7] * d[14];
    coeffs[6] = -2 * d[1] * d[6] * d[12] + 2 * d[0] * d[6] * d[13] + 2 * d[2] * d[8] * d[13] - 2 * d[1] * d[8] * d[14];
    coeffs[7] = 2 * d[1] * d[6] * d[12] - 2 * d[0] * d[7] * d[12] - 2 * d[2] * d[7] * d[14] + 2 * d[1] * d[8] * d[14];
    coeffs[8] = -d[2] * d[7] * d[12] + d[1] * d[8] * d[12] + d[2] * d[6] * d[13] - d[0] * d[8] * d[13] -
                d[1] * d[6] * d[14] + d[0] * d[7] * d[14];
    coeffs[9] = d[5] * d[10] * d[15] + d[4] * d[11] * d[15] - d[5] * d[9] * d[16] + d[3] * d[11] * d[16] -
                d[4] * d[9] * d[17] - d[3] * d[10] * d[17];
    coeffs[10] = -2 * d[4] * d[9] * d[15] - 2 * d[3] * d[10] * d[15] - 2 * d[5] * d[10] * d[17] - 2 * d[4] * d[11] * d[17];
    coeffs[11] = 2 * d[4] * d[9] * d[15] + 2 * d[3] * d[9] * d[16] + 2 * d[5] * d[11] * d[16] + 2 * d[4] * d[11] * d[17];
    coeffs[12] = -d[5] * d[10] * d[15] - d[4] * d[11] * d[15] - d[5] * d[9] * d[16] + d[3] * d[11] * d[16] +
                 d[4] * d[9] * d[17] + d[3] * d[10] * d[17];
    coeffs[13] = 4 * d[4] * d[11] * d[15] - 4 * d[4] * d[9] * d[17];
    coeffs[14] = d[5] * d[10] * d[15] - d[4] * d[11] * d[15] + d[5] * d[9] * d[16] - d[3] * d[11] * d[16] +
                 d[4] * d[9] * d[17] - d[3] * d[10] * d[17];
    coeffs[15] = -2 * d[4] * d[9] * d[15] + 2 * d[3] * d[9] * d[16] + 2 * d[5] * d[11] * d[16] - 2 * d[4] * d[11] * d[17];
    coeffs[16] = 2 * d[4] * d[9] * d[15] - 2 * d[3] * d[10] * d[15] - 2 * d[5] * d[10] * d[17] + 2 * d[4] * d[11] * d[17];
    coeffs[17] = -d[5] * d[10] * d[15] + d[4] * d[11] * d[15] + d[5] * d[9] * d[16] - d[3] * d[11] * d[16] -
                 d[4] * d[9] * d[17] + d[3] * d[10] * d[17];

    static const int coeffs_ind[] = {0,  9,  1, 0,  9, 10, 3, 1, 10, 12, 3,  12, 4,  2, 11, 13, 6,  4,
                                     13, 15, 6, 15, 2, 11, 5, 14, 7,  5, 14, 16, 8, 7, 16, 17, 8, 17};
    static const int C_ind[] = {0,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 16, 17, 18, 19, 20, 21,
                                22, 23, 25, 26, 28, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46};

    Eigen::Matrix<double, 4, 12> C = Eigen::Matrix<double, 4, 12>::Zero();
    for (int i = 0; i < 36; i++)
        C(C_ind[i]) = coeffs[coeffs_ind[i]];

    const Eigen::Matrix4d C0 = C.leftCols<4>();
    const Eigen::FullPivLU<Eigen::Matrix4d> lu(C0);
    if (!lu.isInvertible())
        return;
    const Eigen::Matrix<double, 4, 8> C12 = lu.solve(C.rightCols<8>());

    // RR = [-C12; I_8], AM = rows {1, 2, 3, 0, 7, 4, 5, 6} of RR
    Eigen::Matrix<double, 8, 8> AM = Eigen::Matrix<double, 8, 8>::Zero();
    AM.row(0) = -C12.row(1);
    AM.row(1) = -C12.row(2);
    AM.row(2) = -C12.row(3);
    AM.row(3) = -C12.row(0);
    AM(4, 3) = 1.0;
    AM(5, 0) = 1.0;
    AM(6, 1) = 1.0;
    AM(7, 2) = 1.0;

    Eigen::EigenSolver<Eigen::Matrix<double, 8, 8>> es(AM);
    const Eigen::Matrix<std::complex<double>, 8, 1> D = es.eigenvalues();
    const Eigen::Matrix<std::complex<double>, 8, 8> V = es.eigenvectors();
    for (int i = 0; i < 8; ++i) {
        if (std::abs(D(i).imag()) > 1e-8 * (1.0 + std::abs(D(i))))
            continue;
        // quotient ring basis: x*y^2, x*y, x, x*y^3, y^3, y^2, y, 1
        if (std::abs(V(7, i)) < 1e-12)
            continue;
        const std::complex<double> y = V(6, i) / V(7, i);
        if (std::abs(y.imag()) > 1e-6 * (1.0 + std::abs(y)))
            continue;
        sols->emplace_back(D(i).real(), y.real());
    }
}

inline Eigen::Matrix3d cayley_y(double k) {
    Eigen::Matrix3d K;
    K << 1 - k * k, 0, 2 * k, 0, 1 + k * k, 0, -2 * k, 0, 1 - k * k;
    return K / (1 + k * k);
}

// Real roots of a x^2 + b x + c, also when a vanishes
void quadratic_roots(double a, double b, double c, std::vector<double> *roots) {
    roots->clear();
    if (std::abs(a) < 1e-12 * (std::abs(b) + std::abs(c))) {
        if (std::abs(b) > 0)
            roots->push_back(-c / b);
        return;
    }
    const double disc = b * b - 4 * a * c;
    if (disc < 0)
        return;
    const double sq = std::sqrt(disc);
    roots->push_back((-b + sq) / (2 * a));
    roots->push_back((-b - sq) / (2 * a));
}

} // namespace

int threeview_2sift(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                    const std::vector<Eigen::Vector3d> &x3, const std::vector<Eigen::Vector3d> &l1,
                    const std::vector<Eigen::Vector3d> &l2, const std::vector<Eigen::Vector3d> &l3, double r21,
                    double r22, double r31, std::vector<ThreeViewGeometry> *output) {
    output->clear();
    // points with unit z (depth = z coordinate), point 0 has depth 1 in view 1
    const Eigen::Vector3d a0 = x1[0] / x1[0](2), a1 = x1[1] / x1[1](2);
    const Eigen::Vector3d b0 = r21 * x2[0] / x2[0](2), b1 = r22 * x2[1] / x2[1](2);
    const Eigen::Vector3d c0 = r31 * x3[0] / x3[0](2), c1 = x3[1] / x3[1](2);

    std::vector<double> ms, ns;
    std::vector<Eigen::Vector2d> sols;
    // |a0 - m a1| = |b0 - m b1|: quadratic in m (depth of point 1 in view 1)
    quadratic_roots(a1.squaredNorm() - b1.squaredNorm(), 2 * (b0.dot(b1) - a0.dot(a1)), a0.squaredNorm() - b0.squaredNorm(),
                    &ms);
    for (double m : ms) {
        if (m <= 0)
            continue;
        const Eigen::Vector3d u1 = a0 - m * a1, u2 = b0 - m * b1;
        // |a0 - m a1| = |c0 - n c1|: quadratic in n (depth of point 1 in view 3)
        quadratic_roots(c1.squaredNorm(), -2 * c0.dot(c1), c0.squaredNorm() - u1.squaredNorm(), &ns);
        for (double n : ns) {
            if (n <= 0)
                continue;
            const Eigen::Vector3d u3 = c0 - n * c1;
            if (u1.squaredNorm() < 1e-24 || u2.squaredNorm() < 1e-24 || u3.squaredNorm() < 1e-24)
                continue;

            // The vector between the points is rotated by R_k into view k. With Q_k u_k ~ e_y the rotations are
            // R_k = Q_k^T K_k^T Q_1 with K_k a rotation about y, and det([l1, R2^T l2, R3^T l3]) =
            // det([Q1 l1, K2 Q2 l2, K3 Q3 l3]) = 0 for both lines (the planes contain the direction of the 3D line)
            const Eigen::Matrix3d Q[3] = {Eigen::Quaterniond::FromTwoVectors(u1, Eigen::Vector3d::UnitY()).toRotationMatrix(),
                                          Eigen::Quaterniond::FromTwoVectors(u2, Eigen::Vector3d::UnitY()).toRotationMatrix(),
                                          Eigen::Quaterniond::FromTwoVectors(u3, Eigen::Vector3d::UnitY()).toRotationMatrix()};
            const std::vector<Eigen::Vector3d> *ls[3] = {&l1, &l2, &l3};
            double data[18];
            for (int k = 0; k < 3; ++k) {
                for (int i = 0; i < 2; ++i) {
                    const Eigen::Vector3d ql = Q[k] * (*ls[k])[i].normalized();
                    for (int j = 0; j < 3; ++j)
                        data[6 * k + 3 * i + j] = ql(j);
                }
            }
            solve_rotation(data, &sols);
            for (const Eigen::Vector2d &k : sols) {
                const Eigen::Matrix3d R2 = Q[1].transpose() * cayley_y(k(0)).transpose() * Q[0];
                const Eigen::Matrix3d R3 = Q[2].transpose() * cayley_y(k(1)).transpose() * Q[0];
                output->emplace_back(CameraPose(R2, Eigen::Vector3d(b0 - R2 * a0)),
                                     CameraPose(R3, Eigen::Vector3d(c0 - R3 * a0)));
            }
        }
    }
    return output->size();
}

} // namespace poselib
