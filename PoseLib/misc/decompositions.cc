#include <Eigen/Dense>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/colmap_models.h"

namespace poselib {
std::pair<Camera, Camera> focals_from_fundamental(const Eigen::Matrix3d &F, const Point2D &pp1, const Point2D &pp2) {
    Eigen::Vector3d p1 = pp1.homogeneous();
    Eigen::Vector3d p2 = pp2.homogeneous();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector3d e1 = svd.matrixV().col(2);
    Eigen::Vector3d e2 = svd.matrixU().col(2);

    Eigen::DiagonalMatrix<double, 3> II(1.0, 1.0, 0.0);

    Eigen::Matrix3d s_e1, s_e2;
    s_e1 << 0, -e1(2), e1(1), e1(2), 0, -e1(0), -e1(1), e1(0), 0;
    s_e2 << 0, -e2(2), e2(1), e2(2), 0, -e2(0), -e2(1), e2(0), 0;

    Eigen::MatrixXd f1 = (-p2.transpose() * s_e2 * II * F * (p1 * p1.transpose()) * F.transpose() * p2) /
                         (p2.transpose() * s_e2 * II * F * II * F.transpose() * p2);

    Eigen::MatrixXd f2 = (-p1.transpose() * s_e1 * II * F.transpose() * (p2 * p2.transpose()) * F * p1) /
                         (p1.transpose() * s_e1 * II * F.transpose() * II * F * p1);

    Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{std::sqrt(f1(0, 0)), pp1(0), pp1(1)}, -1, -1);
    Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{std::sqrt(f2(0, 0)), pp2(0), pp2(1)}, -1, -1);

    return std::pair<Camera, Camera>(camera1, camera2);
}

void fast_eigenvector_solver_autocal(double *eigv, int neig, Eigen::Matrix<double, 16, 16> &AM,
                                     Eigen::Matrix<double, 2, 16> &sols) {
    static const int ind[] = {3, 5, 9, 15};
    // Truncated action matrix containing non-trivial rows
    Eigen::Matrix<double, 4, 16> AMs;
    double zi[7];

    for (int i = 0; i < 4; i++) {
        AMs.row(i) = AM.row(ind[i]);
    }

    for (int i = 0; i < neig; i++) {
        zi[0] = eigv[i];
        for (int j = 1; j < 7; j++) {
            zi[j] = zi[j - 1] * eigv[i];
        }
        Eigen::Matrix<double, 4, 4> AA;
        AA.col(0) = AMs.col(3);
        AA.col(1) = AMs.col(2) + zi[0] * AMs.col(4) + zi[1] * AMs.col(5);
        AA.col(2) = AMs.col(1) + zi[0] * AMs.col(6) + zi[1] * AMs.col(7) + zi[2] * AMs.col(8) + zi[3] * AMs.col(9);
        AA.col(3) = AMs.col(0) + zi[0] * AMs.col(10) + zi[1] * AMs.col(11) + zi[2] * AMs.col(12) + zi[3] * AMs.col(13) +
                    zi[4] * AMs.col(14) + zi[5] * AMs.col(15);
        AA(0, 0) = AA(0, 0) - zi[0];
        AA(1, 1) = AA(1, 1) - zi[2];
        AA(2, 2) = AA(2, 2) - zi[4];
        AA(3, 3) = AA(3, 3) - zi[6];

        Eigen::Matrix<double, 3, 1> s = AA.leftCols(3).householderQr().solve(-AA.col(3));
        sols(0, i) = s(2);
        sols(1, i) = zi[0];
    }
}

Eigen::MatrixXd solver_robust_autocal(const Eigen::VectorXd &data, int *num_sols) {
    // Compute coefficients
    const double *d = data.data();
    Eigen::VectorXd coeffs(30);
    coeffs[0] = d[15];
    coeffs[1] = d[16];
    coeffs[2] = d[18];
    coeffs[3] = d[21];
    coeffs[4] = d[25];
    coeffs[5] = d[17];
    coeffs[6] = d[19];
    coeffs[7] = d[22];
    coeffs[8] = d[26];
    coeffs[9] = d[20];
    coeffs[10] = d[23];
    coeffs[11] = d[27];
    coeffs[12] = d[24];
    coeffs[13] = d[28];
    coeffs[14] = d[29];
    coeffs[15] = d[0];
    coeffs[16] = d[1];
    coeffs[17] = d[3];
    coeffs[18] = d[6];
    coeffs[19] = d[10];
    coeffs[20] = d[2];
    coeffs[21] = d[4];
    coeffs[22] = d[7];
    coeffs[23] = d[11];
    coeffs[24] = d[5];
    coeffs[25] = d[8];
    coeffs[26] = d[12];
    coeffs[27] = d[9];
    coeffs[28] = d[13];
    coeffs[29] = d[14];

    // Setup elimination template
    static const int coeffs0_ind[] = {
        0,  15, 1,  0,  15, 16, 2,  1,  0,  15, 16, 17, 3,  2,  1,  0,  15, 16, 17, 18, 4,  3,  2,  1,  16,
        17, 18, 19, 4,  3,  2,  17, 18, 19, 4,  3,  18, 19, 5,  0,  15, 20, 6,  5,  20, 1,  15, 0,  16, 21,
        7,  6,  5,  20, 21, 2,  16, 15, 1,  17, 0,  22, 8,  7,  6,  5,  20, 21, 22, 3,  17, 16, 2,  18, 1,
        23, 8,  7,  6,  21, 22, 23, 4,  18, 17, 3,  19, 2,  9,  5,  20, 0,  15, 24, 10, 9,  24, 6,  20, 5,
        21, 1,  16, 0,  15, 25, 11, 10, 9,  24, 25, 7,  21, 20, 6,  22, 2,  17, 1,  16, 5,  26, 12, 9,  24,
        5,  20, 0,  15, 27, 13, 12, 27, 10, 24, 9,  25, 6,  21, 5,  1,  16, 20, 28, 11, 10, 9,  24, 25, 26,
        8,  22, 21, 7,  23, 3,  18, 2,  17, 6,  8,  7,  22, 23, 19, 18, 4,  3,  4,  19};
    static const int coeffs1_ind[] = {
        14, 29, 14, 29, 12, 27, 14, 29, 12, 27, 9,  24, 14, 12, 27, 9,  24, 5,  20, 29, 14, 29, 13, 27, 12, 28,
        10, 25, 9,  6,  21, 24, 13, 12, 27, 28, 11, 25, 24, 10, 26, 7,  22, 6,  2,  17, 21, 9,  29, 14, 13, 28,
        12, 10, 25, 27, 14, 29, 28, 27, 13, 11, 26, 10, 7,  22, 25, 12, 13, 12, 27, 28, 26, 25, 11, 8,  23, 7,
        3,  18, 22, 10, 11, 10, 25, 26, 23, 22, 8,  4,  19, 3,  18, 7,  14, 13, 28, 29, 29, 13, 11, 26, 28, 14,
        14, 29, 28, 11, 8,  23, 26, 13, 13, 28, 26, 8,  4,  19, 23, 11, 11, 26, 23, 4,  19, 8,  8,  23, 19, 4};
    static const int C0_ind[] = {
        0,   19,  20,  21,  26,  39,  40,  41,  42,  45,  46,  59,  60,  61,  62,  63,  64,  65,  66,  79,  80,  81,
        82,  83,  84,  85,  86,  99,  101, 102, 103, 104, 105, 106, 122, 123, 124, 125, 140, 147, 151, 159, 160, 161,
        166, 167, 168, 170, 171, 179, 180, 181, 182, 185, 186, 187, 188, 189, 190, 191, 198, 199, 200, 201, 202, 203,
        204, 205, 206, 207, 208, 209, 210, 211, 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 238,
        240, 247, 251, 252, 253, 259, 260, 261, 266, 267, 268, 270, 271, 272, 273, 274, 277, 279, 280, 281, 282, 285,
        286, 287, 288, 289, 290, 291, 292, 293, 294, 297, 298, 299, 300, 307, 311, 312, 313, 315, 316, 319, 320, 321,
        326, 327, 328, 330, 331, 332, 333, 334, 335, 336, 337, 339, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
        351, 352, 353, 354, 357, 358, 362, 363, 364, 365, 368, 369, 370, 378, 383, 384};
    static const int C1_ind[] = {
        15,  16,  32,  33,  35,  36,  47,  51,  52,  53,  55,  56,  60,  67,  71,  72,  73,  75,  76,  79,  81,  86,
        87,  88,  90,  91,  92,  93,  94,  95,  96,  97,  101, 102, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
        115, 116, 117, 118, 128, 130, 132, 133, 134, 135, 136, 137, 142, 145, 148, 149, 150, 152, 153, 154, 155, 156,
        157, 158, 162, 163, 164, 165, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 182, 183, 184, 185, 188, 189,
        190, 192, 193, 194, 197, 198, 214, 215, 216, 217, 229, 234, 235, 236, 237, 238, 243, 244, 249, 254, 255, 256,
        257, 258, 263, 264, 269, 274, 275, 276, 277, 278, 283, 284, 289, 294, 297, 298, 303, 304, 309, 318};

    Eigen::Matrix<double, 20, 20> C0;
    C0.setZero();
    Eigen::Matrix<double, 20, 16> C1;
    C1.setZero();
    for (int i = 0; i < 170; i++) {
        C0(C0_ind[i]) = coeffs(coeffs0_ind[i]);
    }
    for (int i = 0; i < 130; i++) {
        C1(C1_ind[i]) = coeffs(coeffs1_ind[i]);
    }

    //    Matrix<double, 20, 16> C12 = C0.partialPivLu().solve(C1);
    Eigen::Matrix<double, 20, 16> C12 = C0.fullPivLu().solve(C1);
    //    Matrix<double, 20, 16> C12 = C0.bdcSvd().solve(C1);

    //    std::cout << "C0" << std::endl;
    //    std::cout << C0 << std::endl;
    //
    //
    //    std::cout << "C1" << std::endl;
    //    std::cout << C1 << std::endl;

    // Setup action matrix
    Eigen::Matrix<double, 20, 16> RR;
    RR << -C12.bottomRows(4), Eigen::Matrix<double, 16, 16>::Identity(16, 16);

    static const int AM_ind[] = {14, 10, 8, 0, 9, 1, 11, 12, 13, 2, 15, 16, 17, 18, 19, 3};
    Eigen::Matrix<double, 16, 16> AM;
    for (int i = 0; i < 16; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::Matrix<double, 2, 16> sols;
    sols.setZero();

    // Solve eigenvalue problem

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM, false);
    Eigen::ArrayXcd D = es.eigenvalues();

    int nroots = 0;
    double eigv[16];
    for (int i = 0; i < 16; i++) {
        if (std::abs(D(i).imag()) < 1e-6)
            eigv[nroots++] = D(i).real();
    }

    fast_eigenvector_solver_autocal(eigv, nroots, AM, sols);
    *num_sols = nroots;
    return sols;
}

std::tuple<Camera, Camera, int> focals_from_fundamental_iterative(const Eigen::Matrix3d &F, const Camera &camera1_prior,
                                                                  const Camera &camera2_prior, const int &max_iters,
                                                                  const Eigen::Vector4d &weights) {
    Eigen::Vector2d pp1_prior = camera1_prior.principal_point();
    Eigen::Vector2d pp2_prior = camera2_prior.principal_point();

    double f1_prior = camera1_prior.focal(), f2_prior = camera2_prior.focal();

    double w1 = weights[0], w2 = weights[1], w3 = weights[2], w4 = weights[3];

    Eigen::Matrix3d T1, T2T;
    T1.setIdentity();
    T1(0, 2) = pp1_prior(0);
    T1(1, 2) = pp1_prior(1);

    T2T.setIdentity();
    T2T(2, 0) = pp2_prior(0);
    T2T(2, 1) = pp2_prior(1);

    Eigen::Matrix3d G = T2T * F * T1;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(G, Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.computeV();
    svd.computeU();
    Eigen::Vector3d singularValues = svd.singularValues();

    // Extract the first two singular values
    double s1 = singularValues(0);
    double s2 = singularValues(1);

    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d U = svd.matrixU();

    double U11 = U(0, 0), U12 = U(0, 1);
    double U21 = U(1, 0), U22 = U(1, 1);
    double U31 = U(2, 0), U32 = U(2, 1);

    double V11 = V(0, 0), V12 = V(0, 1);
    double V21 = V(1, 0), V22 = V(1, 1);
    double V31 = V(2, 0), V32 = V(2, 1);

    double U11_sq = U11 * U11, U12_sq = U12 * U12;
    double U21_sq = U21 * U21, U22_sq = U22 * U22;
    double U31_sq = U31 * U31, U32_sq = U32 * U32;

    double V11_sq = V11 * V11, V12_sq = V12 * V12;
    double V21_sq = V21 * V21, V22_sq = V22 * V22;
    double V31_sq = V31 * V31, V32_sq = V32 * V32;

    double f1prior_sq = std::pow(f1_prior, 2);
    double f2prior_sq = std::pow(f2_prior, 2);

    double cf1p1k, cf1p2k, cu1p1k, cu1p2k, cv1p1k, cv1p2k, cf2p1k, cf2p2k, cu2p1k, cu2p2k, cv2p1k, cv2p2k;
    double cf1p1k_sq, cf1p2k_sq, cu1p1k_sq, cu1p2k_sq, cv1p1k_sq, cv1p2k_sq, cf2p1k_sq, cf2p2k_sq, cu2p1k_sq, cu2p2k_sq,
        cv2p1k_sq, cv2p2k_sq;

    double c11, c12, c13, c14, c15, c16, c17, c18, c19, c110, c111, c112, c113, c114, c115;
    double c21, c22, c23, c24, c25, c26, c27, c28, c29, c210, c211, c212, c213, c214, c215;

    double f1n = f1_prior, u1n = 0.0, v1n = 0.0;
    double f2n = f2_prior, u2n = 0.0, v2n = 0.0;

    double f1n_sq, u1n_sq, v1n_sq, f2n_sq, u2n_sq, v2n_sq;

    double df1, du1, dv1, df2, du2, dv2;

    Eigen::VectorXd err(max_iters + 1);
    err.setZero();
    int k;

    for (k = 0; k < max_iters + 1; k++) {

        f1n_sq = std::pow(f1n, 2);
        u1n_sq = std::pow(u1n, 2);
        v1n_sq = std::pow(v1n, 2);

        f2n_sq = std::pow(f2n, 2);
        u2n_sq = std::pow(u2n, 2);
        v2n_sq = std::pow(v2n, 2);

        cf1p1k =
            1 / w1 *
            (s2 * (2 * V11 * V12 * f1n + 2 * V21 * V22 * f1n) *
                 (U12 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U22 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U32 * (U32 + U12 * u2n + U22 * v2n)) +
             s1 * (2 * f1n * V11 * V11 + 2 * f1n * V21 * V21) *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)));
        cf1p2k =
            1 / w1 *
            (s1 * (2 * V11 * V12 * f1n + 2 * V21 * V22 * f1n) *
                 (U11 * (U31 * u2n + U11 * (f2n_sq + u2n_sq) + U21 * u2n * v2n) +
                  U21 * (U31 * v2n + U21 * (f2n_sq + v2n_sq) + U11 * u2n * v2n) + U31 * (U31 + U11 * u2n + U21 * v2n)) +
             s2 * (2 * f1n * V12 * V12 + 2 * f1n * V22 * V22) *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)));

        cu1p1k =
            1 / w2 *
            (s1 *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V11 * V31 + V11 * (V31 + 2 * V11 * u1n + V21 * v1n) + V11 * V21 * v1n) +
             s2 *
                 (U12 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U22 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U32 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V12 * V31 + V11 * (V32 + 2 * V12 * u1n + V22 * v1n) + V12 * V21 * v1n));
        cu1p2k =
            1 / w2 *
            (s1 *
                 (U11 * (U31 * u2n + U11 * (f2n_sq + u2n_sq) + U21 * u2n * v2n) +
                  U21 * (U31 * v2n + U21 * (f2n_sq + v2n_sq) + U11 * u2n * v2n) + U31 * (U31 + U11 * u2n + U21 * v2n)) *
                 (V12 * V31 + V11 * (V32 + 2 * V12 * u1n + V22 * v1n) + V12 * V21 * v1n) +
             s2 *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V12 * V32 + V12 * (V32 + 2 * V12 * u1n + V22 * v1n) + V12 * V22 * v1n));

        cv1p1k =
            1 / w2 *
            (s1 *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V21 * V31 + V21 * (V31 + V11 * u1n + 2 * V21 * v1n) + V11 * V21 * u1n) +
             s2 *
                 (U12 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U22 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U32 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V22 * V31 + V21 * (V32 + V12 * u1n + 2 * V22 * v1n) + V11 * V22 * u1n));
        cv1p2k =
            1 / w2 *
            (s1 *
                 (U11 * (U31 * u2n + U11 * (f2n_sq + u2n_sq) + U21 * u2n * v2n) +
                  U21 * (U31 * v2n + U21 * (f2n_sq + v2n_sq) + U11 * u2n * v2n) + U31 * (U31 + U11 * u2n + U21 * v2n)) *
                 (V22 * V31 + V21 * (V32 + V12 * u1n + 2 * V22 * v1n) + V11 * V22 * u1n) +
             s2 *
                 (U11 * (U32 * u2n + U12 * (f2n_sq + u2n_sq) + U22 * u2n * v2n) +
                  U21 * (U32 * v2n + U22 * (f2n_sq + v2n_sq) + U12 * u2n * v2n) + U31 * (U32 + U12 * u2n + U22 * v2n)) *
                 (V22 * V32 + V22 * (V32 + V12 * u1n + 2 * V22 * v1n) + V12 * V22 * u1n));

        cf2p1k =
            1 / w3 *
            (s1 * (2 * U11 * U12 * f2n + 2 * U21 * U22 * f2n) *
                 (V11 * (V31 * u1n + V11 * (f1n_sq + u1n_sq) + V21 * u1n * v1n) +
                  V21 * (V31 * v1n + V21 * (f1n_sq + v1n_sq) + V11 * u1n * v1n) + V31 * (V31 + V11 * u1n + V21 * v1n)) +
             s2 * (2 * f2n * U12 * U12 + 2 * f2n * U22 * U22) *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)));
        cf2p2k =
            1 / w3 *
            (s2 * (2 * U11 * U12 * f2n + 2 * U21 * U22 * f2n) *
                 (V12 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V22 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V32 * (V32 + V12 * u1n + V22 * v1n)) +
             s1 * (2 * f2n * U11 * U11 + 2 * f2n * U21 * U21) *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)));

        cu2p1k =
            1 / w4 *
            (s1 *
                 (V11 * (V31 * u1n + V11 * (f1n_sq + u1n_sq) + V21 * u1n * v1n) +
                  V21 * (V31 * v1n + V21 * (f1n_sq + v1n_sq) + V11 * u1n * v1n) + V31 * (V31 + V11 * u1n + V21 * v1n)) *
                 (U12 * U31 + U11 * (U32 + 2 * U12 * u2n + U22 * v2n) + U12 * U21 * v2n) +
             s2 *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U12 * U32 + U12 * (U32 + 2 * U12 * u2n + U22 * v2n) + U12 * U22 * v2n));
        cu2p2k =
            1 / w4 *
            (s1 *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U11 * U31 + U11 * (U31 + 2 * U11 * u2n + U21 * v2n) + U11 * U21 * v2n) +
             s2 *
                 (V12 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V22 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V32 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U12 * U31 + U11 * (U32 + 2 * U12 * u2n + U22 * v2n) + U12 * U21 * v2n));

        cv2p1k =
            1 / w4 *
            (s1 *
                 (V11 * (V31 * u1n + V11 * (f1n_sq + u1n_sq) + V21 * u1n * v1n) +
                  V21 * (V31 * v1n + V21 * (f1n_sq + v1n_sq) + V11 * u1n * v1n) + V31 * (V31 + V11 * u1n + V21 * v1n)) *
                 (U22 * U31 + U21 * (U32 + U12 * u2n + 2 * U22 * v2n) + U11 * U22 * u2n) +
             s2 *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U22 * U32 + U22 * (U32 + U12 * u2n + 2 * U22 * v2n) + U12 * U22 * u2n));
        cv2p2k =
            1 / w4 *
            (s1 *
                 (V11 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V21 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V31 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U21 * U31 + U21 * (U31 + U11 * u2n + 2 * U21 * v2n) + U11 * U21 * u2n) +
             s2 *
                 (V12 * (V32 * u1n + V12 * (f1n_sq + u1n_sq) + V22 * u1n * v1n) +
                  V22 * (V32 * v1n + V22 * (f1n_sq + v1n_sq) + V12 * u1n * v1n) + V32 * (V32 + V12 * u1n + V22 * v1n)) *
                 (U22 * U31 + U21 * (U32 + U12 * u2n + 2 * U22 * v2n) + U11 * U22 * u2n));

        cf1p1k_sq = std::pow(cf1p1k, 2);
        cf1p2k_sq = std::pow(cf1p2k, 2);
        cu1p1k_sq = std::pow(cu1p1k, 2);
        cu1p2k_sq = std::pow(cu1p2k, 2);
        cv1p1k_sq = std::pow(cv1p1k, 2);
        cv1p2k_sq = std::pow(cv1p2k, 2);
        cf2p1k_sq = std::pow(cf2p1k, 2);
        cf2p2k_sq = std::pow(cf2p2k, 2);
        cu2p1k_sq = std::pow(cu2p1k, 2);
        cu2p2k_sq = std::pow(cu2p2k, 2);
        cv2p1k_sq = std::pow(cv2p1k, 2);
        cv2p2k_sq = std::pow(cv2p2k, 2);

        c11 = s1 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k));
        c12 = s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s1 *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) +
              s2 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k));
        c13 = s1 *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                   U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) +
              s1 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior));
        c14 = s1 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s1 *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k)) *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s2 *
                  (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                   U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) +
              s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k));
        c15 = s1 *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                   U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) +
              s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s1 *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s2 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                   U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) +
              s1 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s2 *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior));
        c16 = s1 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) +
              s2 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq) +
              s1 *
                  (V11 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V21 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
              s2 *
                  (U12 * (U12 * (cf2p1k_sq + cu2p1k_sq) + U22 * cu2p1k * cv2p1k) +
                   U22 * (U22 * (cf2p1k_sq + cv2p1k_sq) + U12 * cu2p1k * cv2p1k)) *
                  (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
              s1 *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                   U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior));
        c17 = s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s1 *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) +
              s2 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                   U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k));
        c18 = s1 *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s2 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                   U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) +
              s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s1 *
                  (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                  (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                   U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) +
              s1 *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                   U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior));
        c19 = s2 *
                  (U12 * (U12 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U22 * cu2p1k * cv2p2k +
                          U22 * cu2p2k * cv2p1k) +
                   U22 * (U22 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U12 * cu2p1k * cv2p2k +
                          U12 * cu2p2k * cv2p1k)) *
                  (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
              s1 *
                  (V11 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V21 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
              s1 *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s1 *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                  (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                   U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s2 *
                  (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                   U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s1 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) +
              s2 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq);
        c110 = s1 * (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) *
                   (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                    U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
               s2 * (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq) *
                   (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                    V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
               s2 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U32 * (U12 * cu2p1k + U22 * cv2p1k) + U12 * (U32 * cu2p1k + 2 * U12 * cf2p1k * f2_prior) +
                    U22 * (U32 * cv2p1k + 2 * U22 * cf2p1k * f2_prior)) +
               s1 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V31 * (V11 * cu1p1k + V21 * cv1p1k) + V11 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                    V21 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior));
        c111 = s1 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
               s2 *
                   (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                    U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k)) *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k));
        c112 = s1 *
                   (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s2 *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                    U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) +
               s1 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s2 *
                   (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                    U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k)) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior));
        c113 = s1 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) +
               s2 *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq) +
               s1 *
                   (V11 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V21 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
               s2 *
                   (U12 * (U12 * (cf2p2k_sq + cu2p2k_sq) + U22 * cu2p2k * cv2p2k) +
                    U22 * (U22 * (cf2p2k_sq + cv2p2k_sq) + U12 * cu2p2k * cv2p2k)) *
                   (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
               s1 *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                   (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s2 *
                   (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                    U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior));
        c114 = s1 * (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s2 * (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s2 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U32 * (U12 * cu2p2k + U22 * cv2p2k) + U12 * (U32 * cu2p2k + 2 * U12 * cf2p2k * f2_prior) +
                    U22 * (U32 * cv2p2k + 2 * U22 * cf2p2k * f2_prior)) +
               s1 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V31 * (V11 * cu1p2k + V21 * cv1p2k) + V11 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V21 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior));
        c115 = s1 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V31_sq + V11_sq * f1prior_sq + V21_sq * f1prior_sq) +
               s2 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U32_sq + U12_sq * f2prior_sq + U22_sq * f2prior_sq);

        c21 = s1 *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k));
        c22 = s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k)) +
              s1 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) +
              s2 *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k)) *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k));
        c23 = s1 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s1 *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                   V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior));
        c24 = s1 *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s1 *
                  (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) +
              s2 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                   V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k)) +
              s2 *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k)) +
              s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k));
        c25 = s1 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                   V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior)) +
              s1 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s2 *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k)) *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s1 *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                   V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior));
        c26 = s2 *
                  (U12 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U22 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) +
              s1 *
                  (V12 * (V11 * (cf1p1k_sq + cu1p1k_sq) + V21 * cu1p1k * cv1p1k) +
                   V22 * (V21 * (cf1p1k_sq + cv1p1k_sq) + V11 * cu1p1k * cv1p1k)) *
                  (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq) +
              s2 *
                  (V12 * (V12 * (cf1p1k_sq + cu1p1k_sq) + V22 * cu1p1k * cv1p1k) +
                   V22 * (V22 * (cf1p1k_sq + cv1p1k_sq) + V12 * cu1p1k * cv1p1k)) *
                  (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
              s1 *
                  (U11 * (U11 * (cf2p1k_sq + cu2p1k_sq) + U21 * cu2p1k * cv2p1k) +
                   U21 * (U21 * (cf2p1k_sq + cv2p1k_sq) + U11 * cu2p1k * cv2p1k)) *
                  (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
              s1 *
                  (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                   V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior));
        c27 = s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                   V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k)) +
              s1 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) +
              s2 *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k)) *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k));
        c28 = s1 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s2 *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k)) *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
              s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                   V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior)) +
              s1 *
                  (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                   V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                  (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s2 *
                  (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                   V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k)) *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
              s1 *
                  (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                   U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                  (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                   V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior));
        c29 = s1 *
                  (U11 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U21 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
              s2 *
                  (V12 * (V12 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V22 * cu1p1k * cv1p2k +
                          V22 * cu1p2k * cv1p1k) +
                   V22 * (V22 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V12 * cu1p1k * cv1p2k +
                          V12 * cu1p2k * cv1p1k)) *
                  (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
              s1 *
                  (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                   V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
              s1 *
                  (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                  (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                   V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
              s2 *
                  (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                   U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) *
                  (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                   V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior)) +
              s2 *
                  (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                   U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                  (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                   V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior)) +
              s2 *
                  (U12 * (U11 * (2 * cf2p1k * cf2p2k + 2 * cu2p1k * cu2p2k) + U21 * cu2p1k * cv2p2k +
                          U21 * cu2p2k * cv2p1k) +
                   U22 * (U21 * (2 * cf2p1k * cf2p2k + 2 * cv2p1k * cv2p2k) + U11 * cu2p1k * cv2p2k +
                          U11 * cu2p2k * cv2p1k)) *
                  (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) +
              s1 *
                  (V12 * (V11 * (2 * cf1p1k * cf1p2k + 2 * cu1p1k * cu1p2k) + V21 * cu1p1k * cv1p2k +
                          V21 * cu1p2k * cv1p1k) +
                   V22 * (V21 * (2 * cf1p1k * cf1p2k + 2 * cv1p1k * cv1p2k) + V11 * cu1p1k * cv1p2k +
                          V11 * cu1p2k * cv1p1k)) *
                  (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq);
        c210 = s2 * (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) *
                   (U32 * (U11 * cu2p1k + U21 * cv2p1k) + U12 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                    U22 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
               s1 * (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq) *
                   (V32 * (V11 * cu1p1k + V21 * cv1p1k) + V12 * (V31 * cu1p1k + 2 * V11 * cf1p1k * f1_prior) +
                    V22 * (V31 * cv1p1k + 2 * V21 * cf1p1k * f1_prior)) +
               s1 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U31 * (U11 * cu2p1k + U21 * cv2p1k) + U11 * (U31 * cu2p1k + 2 * U11 * cf2p1k * f2_prior) +
                    U21 * (U31 * cv2p1k + 2 * U21 * cf2p1k * f2_prior)) +
               s2 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V32 * (V12 * cu1p1k + V22 * cv1p1k) + V12 * (V32 * cu1p1k + 2 * V12 * cf1p1k * f1_prior) +
                    V22 * (V32 * cv1p1k + 2 * V22 * cf1p1k * f1_prior));
        c211 = s1 *
                   (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) +
               s2 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                    V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k));
        c212 = s1 *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s2 *
                   (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                    V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k)) *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s1 *
                   (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s2 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                    V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior));
        c213 = s2 *
                   (U12 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U22 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) +
               s1 *
                   (V12 * (V11 * (cf1p2k_sq + cu1p2k_sq) + V21 * cu1p2k * cv1p2k) +
                    V22 * (V21 * (cf1p2k_sq + cv1p2k_sq) + V11 * cu1p2k * cv1p2k)) *
                   (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq) +
               s2 *
                   (V12 * (V12 * (cf1p2k_sq + cu1p2k_sq) + V22 * cu1p2k * cv1p2k) +
                    V22 * (V22 * (cf1p2k_sq + cv1p2k_sq) + V12 * cu1p2k * cv1p2k)) *
                   (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) +
               s1 *
                   (U11 * (U11 * (cf2p2k_sq + cu2p2k_sq) + U21 * cu2p2k * cv2p2k) +
                    U21 * (U21 * (cf2p2k_sq + cv2p2k_sq) + U11 * cu2p2k * cv2p2k)) *
                   (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) +
               s1 *
                   (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s2 *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) *
                   (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                    V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior));
        c214 = s2 * (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) *
                   (U32 * (U11 * cu2p2k + U21 * cv2p2k) + U12 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U22 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s1 * (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq) *
                   (V32 * (V11 * cu1p2k + V21 * cv1p2k) + V12 * (V31 * cu1p2k + 2 * V11 * cf1p2k * f1_prior) +
                    V22 * (V31 * cv1p2k + 2 * V21 * cf1p2k * f1_prior)) +
               s1 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U31 * (U11 * cu2p2k + U21 * cv2p2k) + U11 * (U31 * cu2p2k + 2 * U11 * cf2p2k * f2_prior) +
                    U21 * (U31 * cv2p2k + 2 * U21 * cf2p2k * f2_prior)) +
               s2 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V32 * (V12 * cu1p2k + V22 * cv1p2k) + V12 * (V32 * cu1p2k + 2 * V12 * cf1p2k * f1_prior) +
                    V22 * (V32 * cv1p2k + 2 * V22 * cf1p2k * f1_prior));
        c215 = s2 * (U31 * U32 + U11 * U12 * f2prior_sq + U21 * U22 * f2prior_sq) *
                   (V32_sq + V12_sq * f1prior_sq + V22_sq * f1prior_sq) +
               s1 * (V31 * V32 + V11 * V12 * f1prior_sq + V21 * V22 * f1prior_sq) *
                   (U31_sq + U11_sq * f2prior_sq + U21_sq * f2prior_sq);
        Eigen::VectorXd ec(30);

        ec << c11, c12, c13, c14, c15, c16, c17, c18, c19, c110, c111, c112, c113, c114, c115, c21, c22, c23, c24, c25,
            c26, c27, c28, c29, c210, c211, c212, c213, c214, c215;
        ec /= ec.norm();
        //        ec.head(15) /= ec(14);
        //        ec.tail(15) /= ec(29);

        int nroots = 0;
        Eigen::Matrix<double, 2, 16> sols = solver_robust_autocal(ec, &nroots);

        double best_res = 20000000;
        double l1 = 10000, l2 = 10000, ll1, ll2;

        for (int i = 0; i < nroots; i++) {
            ll2 = sols(1, i);
            ll1 = sols(0, i);
            double res = std::abs(ll1) + std::abs(ll2);

            //            Eigen::VectorXd monomials(15);
            //            monomials << std::pow(ll1, 4),
            //                std::pow(ll1, 3)* ll2,
            //                std::pow(ll1, 3),
            //                std::pow(ll1, 2)* std::pow(ll2, 2),
            //                std::pow(ll1, 2)* ll2,
            //                std::pow(ll1, 2),
            //                ll1* std::pow(ll2, 3),
            //                ll1* std::pow(ll2, 2),
            //                ll1* ll2,
            //                ll1,
            //                std::pow(ll2, 4), std::pow(ll2, 3), std::pow(ll2, 2), ll2, 1;
            //
            //             double res = std::abs(ec.head(15).dot(monomials)) + std::abs(ec.tail(15).dot(monomials));

            if (res < best_res) {
                l1 = ll1;
                l2 = ll2;
                best_res = res;
            }
        }

        df1 = l1 * cf1p1k + l2 * cf1p2k;
        du1 = l1 * cu1p1k + l2 * cu1p2k;
        dv1 = l1 * cv1p1k + l2 * cv1p2k;

        df2 = l1 * cf2p1k + l2 * cf2p2k;
        du2 = l1 * cu2p1k + l2 * cu2p2k;
        dv2 = l1 * cv2p1k + l2 * cv2p2k;
        err(k) = df1 * df1 + du1 * du1 + dv1 * dv1 + df2 * df2 + du2 * du2 + dv2 * dv2;

        f1n = df1 + f1_prior;
        u1n = du1;
        v1n = dv1;

        f2n = df2 + f2_prior;
        u2n = du2;
        v2n = dv2;

        // next iter
        if (k > 0) {
            if ((std::abs(err(k - 1) - err(k)) / std::abs(err(k - 1)) < 1e-4) || (std::abs(err(k)) < 1e-8))
                break;
        }
    }

    Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{f1n, u1n + pp1_prior(0), v1n + pp1_prior(1)}, -1, -1);
    Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{f2n, u2n + pp2_prior(0), v2n + pp2_prior(1)}, -1, -1);

    return std::tuple<Camera, Camera, int>(camera1, camera2, k);
}


// Author: Yaqing Ding (yaqing.ding@cvut.cz)
inline void refine_lambda_s(double &s, double &p, double &q, const double h1,
                            const double h2, const double h3, const double h4,
                            const double h5, const double h6, const double h7,
                            const double h8, const double h9, const double h11,
                            const double h22, const double h33,
                            const double h44, const double h55,
                            const double h66, const double h77,
                            const double h88, const double h99, const double g1,
                            const double g2, const double g3) {
    for (int iter = 0; iter < 5; ++iter) {
        double psqr = p * p;
        double qsqr = q * q;
        double g4 = h11 + h44 + h77 - s;
        double g5 = h33 + h66 + h99 - s;
        double g6 = h22 + h55 + h88 - s;
        double r1 = g4 * psqr - 2.0 * p * g1 + g5;
        double r2 = g6 * qsqr - 2.0 * q * g3 + g5;
        double r3 = -p * g1 - q * g3 + g5 + g2 * p * q;

        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-12)
            return;
        double J11 = -psqr - 1.0;
        double J12 = 2.0 * (p * g4 - g1);
        double J21 = -qsqr - 1.0;
        double J23 = 2.0 * (q * g6 - g3);

        double J32 = g2 * q - g1;
        double J33 = g2 * p - g3;
        double detJ = 1.0 / (-J12 * J23 - J11 * J23 * J32 - J12 * J21 * J33);
        s += (J23 * J32 * r1 + J12 * J33 * r2 - J12 * J23 * r3) * detJ;
        p += ((J23 + J21 * J33) * r1 - J11 * J33 * r2 + J11 * J23 * r3) * detJ;
        q += (-J21 * J32 * r1 + (J12 + J11 * J32) * r2 + J12 * J21 * r3) * detJ;
    }
}

inline void refine_lambda3(double &p, double &q, const double h1,
                           const double h2, const double h3, const double h4,
                           const double h5, const double h6, const double h7,
                           const double h8, const double h9) {

    for (int iter = 0; iter < 5; ++iter) {
        double r1 = h1 * h1 * p * p - 2.0 * h1 * h3 * p + h3 * h3 +
                    h4 * h4 * p * p - 2.0 * h4 * h6 * p + h6 * h6 +
                    h7 * h7 * p * p - 2.0 * h7 * h9 * p + h9 * h9 - p * p - 1.0;
        double r2 = h2 * h2 * q * q - 2.0 * h2 * h3 * q + h3 * h3 +
                    h5 * h5 * q * q - 2.0 * h5 * h6 * q + h6 * h6 +
                    h8 * h8 * q * q - 2.0 * h8 * h9 * q + h9 * h9 - q * q - 1.0;
        double r3 = h1 * h1 * p * p - 2 * h1 * h2 * p * q + h2 * h2 * q * q +
                    h4 * h4 * p * p - 2 * h4 * h5 * p * q + h5 * h5 * q * q +
                    h7 * h7 * p * p - 2 * h7 * h8 * p * q + h8 * h8 * q * q -
                    p * p - q * q;
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-12)
            return;
        double J11 =
            -p - h1 * (h3 - h1 * p) - h4 * (h6 - h4 * p) - h7 * (h9 - h7 * p);
        double J22 =
            -q - h2 * (h3 - h2 * q) - h5 * (h6 - h5 * q) - h8 * (h9 - h8 * q);
        double J31 = h1 * (h1 * p - h2 * q) - p + h4 * (h4 * p - h5 * q) +
                     h7 * (h7 * p - h8 * q);
        double J32 = -q - h2 * (h1 * p - h2 * q) - h5 * (h4 * p - h5 * q) -
                     h8 * (h7 * p - h8 * q);
        double detJ =
            0.5 / (J11 * J11 * J22 * J22 + J11 * J11 * J32 * J32 +
                   J22 * J22 * J31 * J31);
        p -= (J11 * r1 * (J22 * J22 + J32 * J32) -
              r3 * (J31 * J32 * J32 - J31 * (J22 * J22 + J32 * J32)) -
              J22 * J31 * J32 * r2) *
             detJ;
        q -= (J22 * r2 * (J11 * J11 + J31 * J31) -
              r3 * (J31 * J31 * J32 - J32 * (J11 * J11 + J31 * J31)) -
              J11 * J31 * J32 * r1) *
             detJ;
    }
}

inline void refine_lambda(double &p, double &q, const double h1,
                          const double h2, const double h3, const double h4,
                          const double h5, const double h6, const double h7,
                          const double h8, const double h9) {

    for (int iter = 0; iter < 5; ++iter) {
        double r1 = h1 * h1 * p * p - 2.0 * h1 * h3 * p + h3 * h3 +
                    h4 * h4 * p * p - 2.0 * h4 * h6 * p + h6 * h6 +
                    h7 * h7 * p * p - 2.0 * h7 * h9 * p + h9 * h9 - p * p - 1.0;
        double r2 = h2 * h2 * q * q - 2.0 * h2 * h3 * q + h3 * h3 +
                    h5 * h5 * q * q - 2.0 * h5 * h6 * q + h6 * h6 +
                    h8 * h8 * q * q - 2.0 * h8 * h9 * q + h9 * h9 - q * q - 1.0;
        if (std::abs(r1) + std::abs(r2) < 1e-12)
            return;
        double J11 =
            -p - h1 * (h3 - h1 * p) - h4 * (h6 - h4 * p) - h7 * (h9 - h7 * p);
        double J22 =
            -q - h2 * (h3 - h2 * q) - h5 * (h6 - h5 * q) - h8 * (h9 - h8 * q);
        double detJ = 0.5 / (J11 * J22);
        p += (-J22 * r1) * detJ;
        q += (-J11 * r2) * detJ;
    }
}

void solve_quadratic(double a, double b, double c, double roots[2]) {

    double sq = std::sqrt(b * b - 4.0 * a * c);

    roots[0] = (b > 0) ? (2.0 * c) / (-b - sq) : (2.0 * c) / (-b + sq);
    roots[1] = c / (a * roots[0]);
}

void hdecom_svd(Eigen::Matrix3d &HH, std::vector<Eigen::Matrix3d> &Rest, std::vector<Eigen::Vector3d> &Test, std::vector<Eigen::Vector3d> &Nest) {

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(HH, Eigen::ComputeFullV);
    Eigen::Matrix3d H2 = HH / svd.singularValues()[1];

    Eigen::Vector3d S2 = svd.singularValues();
    Eigen::Matrix3d V2 = svd.matrixV();

    if (abs(S2(0) - S2(2)) < 1.0e-6)
    {
        Rest.push_back(H2);
        Test.push_back(Eigen::Vector3d(0.0,0.0,0.0));
        return;
    }

    if (V2.determinant() < 0) {
        V2 *= -1;
    }

    double s1 = S2(0) * S2(0) / (S2(1) * S2(1));
    double s3 = S2(2) * S2(2) / (S2(1) * S2(1));

    Eigen::Vector3d v1 = V2.col(0);
    Eigen::Vector3d v2 = V2.col(1);
    Eigen::Vector3d v3 = V2.col(2);

    Eigen::Vector3d u1 = (std::sqrt(1.0 - s3) * v1 + std::sqrt(s1 - 1.0) * v3) /
                         std::sqrt(s1 - s3);
    Eigen::Vector3d u2 = (std::sqrt(1.0 - s3) * v1 - std::sqrt(s1 - 1.0) * v3) /
                         std::sqrt(s1 - s3);

    Eigen::Matrix3d U1;
    Eigen::Matrix3d W1;
    Eigen::Matrix3d U2;
    Eigen::Matrix3d W2;
    U1.col(0) = v2;
    U1.col(1) = u1;
    U1.col(2) = v2.cross(u1);

    W1.col(0) = H2 * v2;
    W1.col(1) = H2 * u1;
    W1.col(2) = (H2 * v2).cross(H2 * u1);

    U2.col(0) = v2;
    U2.col(1) = u2;
    U2.col(2) = v2.cross(u2);

    W2.col(0) = H2 * v2;
    W2.col(1) = H2 * u2;
    W2.col(2) = (H2 * v2).cross(H2 * u2);

    // # compute the rotation matrices
    Eigen::Matrix3d R1 = W1 * U1.transpose();
    Eigen::Matrix3d R2 = W2 * U2.transpose();

    Eigen::Vector3d n1 = v2.cross(u1);

    if (n1(2) < 0)
    {
        n1 = -n1;
    }
    Eigen::Vector3d t1 = (H2 - R1) * n1;

    Eigen::Vector3d n2 = v2.cross(u2);

    if (n2(2) < 0)
    {
        n2 = -n2;
    }
    Eigen::Vector3d t2 = (H2 - R2) * n2;

    Rest.push_back(R1);
    Rest.push_back(R2);
    Test.push_back(t1);
    Test.push_back(t2);
    Nest.push_back(n1);
    Nest.push_back(n2);

}

void hdecom_ding(Eigen::Matrix3d &HH, std::vector<Eigen::Matrix3d> &Rest,
                 std::vector<Eigen::Vector3d> &Test,
                 std::vector<Eigen::Vector3d> &Nest) {

    Eigen::Matrix3d H = HH;
    Eigen::Matrix3d M = H.transpose() * H;

    if (abs(M(0, 1)) + abs(M(0, 2)) + abs(M(1, 0)) + abs(M(1, 2)) + abs(M(2, 0)) +
            abs(M(2, 1)) <
        1e-5) {
        Rest.push_back(H / sqrt(M(0,0)));
        Test.push_back(Eigen::Vector3d(0.0,0.0,0.0));
        return;
    }

    double m1122 = M(0, 0) * M(1, 1);
    double m12sqr = M(0, 1) * M(0, 1);
    double m13sqr = M(0, 2) * M(0, 2);
    double m23sqr = M(1, 2) * M(1, 2);

    double c2 = -(M(0, 0) + M(1, 1) + M(2, 2));
    double c1 = m1122 + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2) -
                (m12sqr + m13sqr + m23sqr);
    double c0 = m12sqr * M(2, 2) + m13sqr * M(1, 1) + m23sqr * M(0, 0) -
                m1122 * M(2, 2) - 2 * M(0, 1) * M(0, 2) * M(1, 2);

    double c22 = c2 * c2;
    double a = c1 - c22 / 3.0;
    double b = (2.0 * c22 * c2 - 9.0 * c2 * c1) / 27.0 + c0;

    double c = 3.0 * b / (2.0 * a) * std::sqrt((-3.0 / a));

    if (c < 0.99999) {
        double d = 2.0 * std::sqrt((-a / 3.0));

        double sigma = d * sin(asin(-c) / 3.0) - c2 / 3.0;
        // double sigma =
        //     d * cos(acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0;

        H = H / sqrt(sigma);
        Eigen::Matrix3d Htmp = H;
        bool flag = 0;

        double h1 = H(0, 0);
        double h2 = H(0, 1);
        double h3 = H(0, 2);
        double h4 = H(1, 0);
        double h5 = H(1, 1);
        double h6 = H(1, 2);
        double h7 = H(2, 0);
        double h8 = H(2, 1);
        double h9 = H(2, 2);
        double h11 = h1 * h1;
        double h22 = h2 * h2;
        double h33 = h3 * h3;
        double h44 = h4 * h4;
        double h55 = h5 * h5;
        double h66 = h6 * h6;
        double h77 = h7 * h7;
        double h88 = h8 * h8;
        double h99 = h9 * h9;

        double g1 = h1 * h3 + h4 * h6 + h7 * h9;
        double g2 = h1 * h2 + h4 * h5 + h7 * h8;
        double g3 = h2 * h3 + h5 * h6 + h8 * h9;
        double p0 = h33 + h66 + h99 - 1.0;
        double p1 = -2.0 * g1;
        double p2 = h11 + h44 + h77 - 1.0;

        Eigen::Matrix3d Rrand;
        if (abs(p0) < 1e-8 || abs(p1) < 1e-8 || abs(p2) < 1e-8) { // special planes
            Eigen::Vector4d qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
            Rrand =
                Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();
            H = Htmp * (Rrand.transpose());
            h1 = H(0, 0);
            h2 = H(0, 1);
            h3 = H(0, 2);
            h4 = H(1, 0);
            h5 = H(1, 1);
            h6 = H(1, 2);
            h7 = H(2, 0);
            h8 = H(2, 1);
            h9 = H(2, 2);
            h11 = h1 * h1;
            h22 = h2 * h2;
            h33 = h3 * h3;
            h44 = h4 * h4;
            h55 = h5 * h5;
            h66 = h6 * h6;
            h77 = h7 * h7;
            h88 = h8 * h8;
            h99 = h9 * h9;

            g1 = h1 * h3 + h4 * h6 + h7 * h9;
            g2 = h1 * h2 + h4 * h5 + h7 * h8;
            g3 = h2 * h3 + h5 * h6 + h8 * h9;
            p0 = h33 + h66 + h99 - 1.0;
            p1 = -2.0 * g1;
            p2 = h11 + h44 + h77 - 1.0;
            flag = 1;
        }

        double pa[2];
        double qa[2];
        double q0[2];
        double q1[2];
        solve_quadratic(p2, p1, p0, pa);
        for (int i = 0; i < 2; ++i) {
            q0[i] = p0 + pa[i] * p1 / 2.0;
            q1[i] = g2 * pa[i] - g3;
        }
        if (abs(q1[0]) < 1e-8 || (abs(q1[1]) < 1e-8 && flag == 0)) { // special planes
            Eigen::Vector4d qgt = Eigen::Quaternion<double>::UnitRandom().coeffs();
            Rrand =
                Eigen::Quaterniond(qgt(0), qgt(1), qgt(2), qgt(3)).toRotationMatrix();
            H = Htmp * (Rrand.transpose());
            h1 = H(0, 0);
            h2 = H(0, 1);
            h3 = H(0, 2);
            h4 = H(1, 0);
            h5 = H(1, 1);
            h6 = H(1, 2);
            h7 = H(2, 0);
            h8 = H(2, 1);
            h9 = H(2, 2);
            h11 = h1 * h1;
            h22 = h2 * h2;
            h33 = h3 * h3;
            h44 = h4 * h4;
            h55 = h5 * h5;
            h66 = h6 * h6;
            h77 = h7 * h7;
            h88 = h8 * h8;
            h99 = h9 * h9;

            g1 = h1 * h3 + h4 * h6 + h7 * h9;
            g2 = h1 * h2 + h4 * h5 + h7 * h8;
            g3 = h2 * h3 + h5 * h6 + h8 * h9;
            p0 = h33 + h66 + h99 - 1.0;
            p1 = -2.0 * g1;
            p2 = h11 + h44 + h77 - 1.0;
            flag = 1;
            solve_quadratic(p2, p1, p0, pa);
            for (int i = 0; i < 2; ++i) {
                q0[i] = p0 + pa[i] * p1 / 2.0;
                q1[i] = g2 * pa[i] - g3;
            }
        }

        qa[0] = -q0[0] / q1[0];
        qa[1] = -q0[1] / q1[1];

        double s = 1.0;
        for (int k = 0; k < 2; ++k) {
            double p = pa[k];
            double q = qa[k];

            refine_lambda_s(s, p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9, h11, h22,
                            h33, h44, h55, h66, h77, h88, h99, g1, g2, g3);
            // refine_lambda3(p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9);
            // refine_lambda(p, q, h1, h2, h3, h4, h5, h6, h7, h8, h9);
            Eigen::Vector3d z1(h3 - p * h1, h6 - p * h4, h9 - p * h7);
            Eigen::Vector3d z2(h3 - q * h2, h6 - q * h5, h9 - q * h8);
            Eigen::Matrix3d B;
            B << z1, z2, z1.cross(z2);
            Eigen::Matrix3d A;

            A << -p, 0, q, 0, -q, p, 1.0, 1.0, p * q;
            Eigen::Matrix3d R = B * A.inverse();
            Eigen::Vector3d t(h3 - R(0, 2), h6 - R(1, 2), h9 - R(2, 2));

            Eigen::Vector3d n = A.col(2) / A.col(2).norm();
            t = t / n(2);
            if (flag == 1) {
                R = R * Rrand;
                n = Rrand.transpose() * n;
            }

            if (n(2) < 0)
                n = -n;

            Rest.push_back(R);
            Test.push_back(t);
            Nest.push_back(n);
        }

    } else {
        hdecom_svd(HH, Rest, Test, Nest);
    }
}



} // namespace poselib
