// Port of relpose_upright_3pt from upstream PoseLib.
// Copyright (c) 2020, Viktor Larsson. All rights reserved. BSD-3-Clause, see LICENSE.
// Author: Yaqing Ding
//         Mark Shachkov

#include "relpose_upright_3pt_ding.h"

#include <array>
#include <cmath>

namespace poselib {

namespace {

// Helpers from upstream p3p_common.h / univariate.cc (not present in this branch)
bool root2real(double b, double c, double &r1, double &r2) {
    double THRESHOLD = -1.0e-12;
    double v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    double y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

std::array<Eigen::Vector3d, 2> compute_pq(Eigen::Matrix3d C) {
    std::array<Eigen::Vector3d, 2> pq;
    Eigen::Matrix3d C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);

    pq[0] = C.col(0);
    pq[1] = C.row(0);

    return pq;
}

bool solve_cubic_single_real(double c2, double c1, double c0, double &root) {
    double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    if (c != 0) {
        if (c > 0) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
            return true;
        } else {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
    } else {
        root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
    }
    return false;
}

void add_solution(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2, const double *m,
                  double cq, double sq, CameraPoseVector *output) {
    double lambda = -(m[0] + m[1] * cq + m[2] * sq) / (m[3] + m[4] * cq + m[5] * sq);
    if (lambda < 0)
        return;

    Eigen::Matrix3d R;
    R.setIdentity();
    R(0, 0) = cq;
    R(0, 2) = sq;
    R(2, 0) = -sq;
    R(2, 2) = cq;

    Eigen::Vector3d trans = lambda * x2[0] - R * x1[0];
    trans.normalize();
    output->emplace_back(R, trans);
}

} // namespace

int relpose_upright_3pt_ding(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                             CameraPoseVector *output) {
    output->clear();

    Eigen::Vector3d u1 = x1[0].cross(x1[1]);
    Eigen::Vector3d v1 = x2[1].cross(x2[0]);
    Eigen::Vector3d u2 = x1[0].cross(x1[2]);
    Eigen::Vector3d v2 = x2[2].cross(x2[0]);

    double a[12];
    a[0] = u1(0);
    a[1] = u1(1);
    a[2] = u1(2);
    a[3] = x2[1](0);
    a[4] = x2[1](1);
    a[5] = x2[1](2);
    a[6] = v1(0);
    a[7] = v1(1);
    a[8] = v1(2);
    a[9] = x1[1](0);
    a[10] = x1[1](1);
    a[11] = x1[1](2);

    double b[12];
    b[0] = u2(0);
    b[1] = u2(1);
    b[2] = u2(2);
    b[3] = x2[2](0);
    b[4] = x2[2](1);
    b[5] = x2[2](2);
    b[6] = v2(0);
    b[7] = v2(1);
    b[8] = v2(2);
    b[9] = x1[2](0);
    b[10] = x1[2](1);
    b[11] = x1[2](2);

    double m[12];
    m[0] = a[1] * a[4];
    m[1] = a[0] * a[3] + a[2] * a[5];
    m[2] = a[2] * a[3] - a[0] * a[5];
    m[5] = a[8] * a[9] - a[6] * a[11];
    m[4] = -a[6] * a[9] - a[8] * a[11];
    m[3] = -a[7] * a[10];
    m[8] = b[2] * b[3] - b[0] * b[5];
    m[7] = b[0] * b[3] + b[2] * b[5];
    m[6] = b[1] * b[4];
    m[11] = b[8] * b[9] - b[6] * b[11];
    m[10] = -b[6] * b[9] - b[8] * b[11];
    m[9] = -b[7] * b[10];

    Eigen::Matrix3d D1, D2;
    // first conic
    D1 << m[0] * m[9] - m[3] * m[6], (m[0] * m[10] + m[1] * m[9] - m[3] * m[7] - m[4] * m[6]) * 0.5,
        (m[0] * m[11] + m[2] * m[9] - m[3] * m[8] - m[5] * m[6]) * 0.5,
        (m[0] * m[10] + m[1] * m[9] - m[3] * m[7] - m[4] * m[6]) * 0.5, m[1] * m[10] - m[4] * m[7],
        (m[1] * m[11] + m[2] * m[10] - m[4] * m[8] - m[5] * m[7]) * 0.5,
        (m[0] * m[11] + m[2] * m[9] - m[3] * m[8] - m[5] * m[6]) * 0.5,
        (m[1] * m[11] + m[2] * m[10] - m[4] * m[8] - m[5] * m[7]) * 0.5, m[2] * m[11] - m[5] * m[8];
    // circle
    D2 << -1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;

    Eigen::Matrix3d DX1, DX2;
    DX1 << D1.col(1).cross(D1.col(2)), D1.col(2).cross(D1.col(0)), D1.col(0).cross(D1.col(1));
    DX2 << D2.col(1).cross(D2.col(2)), D2.col(2).cross(D2.col(0)), D2.col(0).cross(D2.col(1));

    double k3 = D2.col(0).dot(DX2.col(0));
    double k2 = (D1.array() * DX2.array()).sum();
    double k1 = (D2.array() * DX1.array()).sum();
    double k0 = D1.col(0).dot(DX1.col(0));

    double k3_inv = 1.0 / k3;
    k2 *= k3_inv;
    k1 *= k3_inv;
    k0 *= k3_inv;

    double s;
    bool G = solve_cubic_single_real(k2, k1, k0, s);

    Eigen::Matrix3d C = D1 + s * D2;
    std::array<Eigen::Vector3d, 2> pq = compute_pq(C);

    for (int i = 0; i < 2; ++i) {
        // [p1 p2 p3] * [1; cos; sin] = 0
        double p1 = pq[i](0);
        double p2 = pq[i](1);
        double p3 = pq[i](2);
        const size_t n_before = output->size();

        bool switch_23 = std::abs(p3) <= std::abs(p2);

        if (switch_23) {
            double w0 = -p1 / p2;
            double w1 = -p3 / p2;
            // find intersections between line [p1 p2 p3] * [1; cos; sin] = 0 and circle sin^2+cos^2=1
            double ca = 1.0 / (w1 * w1 + 1.0);
            double cb = 2.0 * w0 * w1 * ca;
            double cc = (w0 * w0 - 1.0) * ca;
            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double sq : taus)
                add_solution(x1, x2, m, w0 + w1 * sq, sq, output);
        } else {
            double w0 = -p1 / p3;
            double w1 = -p2 / p3;
            double ca = 1.0 / (w1 * w1 + 1);
            double cb = 2.0 * w0 * w1 * ca;
            double cc = (w0 * w0 - 1.0) * ca;

            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double cq : taus)
                add_solution(x1, x2, m, cq, w0 + w1 * cq, output);
        }

        if (output->size() > n_before && G)
            break;
    }

    return output->size();
}

int relpose_upright_3pt_ding(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                             const Eigen::Vector3d &g_cam1, const Eigen::Vector3d &g_cam2, CameraPoseVector *output) {
    // Rotate camera coordinate systems so that the common direction is the y axis
    Eigen::Matrix3d Rc1 = Eigen::Quaterniond::FromTwoVectors(g_cam1, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d Rc2 = Eigen::Quaterniond::FromTwoVectors(g_cam2, Eigen::Vector3d::UnitY()).toRotationMatrix();

    std::vector<Eigen::Vector3d> x1_upright(3), x2_upright(3);
    for (int i = 0; i < 3; ++i) {
        x1_upright[i] = Rc1 * x1[i];
        x2_upright[i] = Rc2 * x2[i];
    }

    int n_sols = relpose_upright_3pt_ding(x1_upright, x2_upright, output);

    // De-rotate coordinate systems
    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix3d R = (*output)[i].R();
        Eigen::Vector3d t = (*output)[i].t;
        t = Rc2.transpose() * t;
        R = Rc2.transpose() * R * Rc1;
        (*output)[i] = CameraPose(R, t);
    }
    return n_sols;
}

} // namespace poselib
