#include "relpose_2l2p.h"

#include <cmath>

namespace poselib {

namespace {

// Orthonormal frame symmetric w.r.t. two unit directions (bisectors)
inline bool frame_from_two_directions(const Eigen::Vector3d &da, const Eigen::Vector3d &db, Eigen::Matrix3d *F) {
    Eigen::Vector3d e1 = da + db;
    Eigen::Vector3d e2 = da - db;
    double n1 = e1.norm(), n2 = e2.norm();
    if (n1 < 1e-6 || n2 < 1e-6)
        return false;
    e1 /= n1;
    e2 /= n2;
    F->col(0) = e1;
    F->col(1) = e2;
    F->col(2) = e1.cross(e2);
    return true;
}

// Candidate rotations R such that R * frame(view 1) = frame(view k) for all consistent sign choices
// of the line directions in view k. Directions must be unit vectors.
void direction_pair_rotations(const Eigen::Vector3d &da1, const Eigen::Vector3d &db1, const Eigen::Vector3d &dak,
                              const Eigen::Vector3d &dbk, std::vector<Eigen::Matrix3d> *rotations) {
    rotations->clear();
    Eigen::Matrix3d F1;
    if (!frame_from_two_directions(da1, db1, &F1))
        return;

    const double c1 = da1.dot(db1);
    const double ck = dak.dot(dbk);
    // if the angle between the lines is close to 90 deg the sign of the cosine is not informative
    const bool check_sign = std::abs(c1) > 0.1 && std::abs(ck) > 0.1;

    for (int sa : {1, -1}) {
        for (int sb : {1, -1}) {
            if (check_sign && (c1 * sa * sb * ck) < 0)
                continue;
            Eigen::Matrix3d Fk;
            if (!frame_from_two_directions(sa * dak, sb * dbk, &Fk))
                continue;
            rotations->push_back(Fk * F1.transpose());
        }
    }
}

// Depths of a point in both views given the relative pose (least squares)
inline bool triangulate_depths(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Vector3d &x1,
                               const Eigen::Vector3d &x2, double *lambda1, double *lambda2) {
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = R * x1;
    A.col(1) = -x2;
    const Eigen::Matrix2d AtA = A.transpose() * A;
    if (std::abs(AtA.determinant()) < 1e-12 * AtA.trace() * AtA.trace())
        return false;
    const Eigen::Vector2d lambda = AtA.inverse() * (A.transpose() * -t);
    *lambda1 = lambda(0);
    *lambda2 = lambda(1);
    return true;
}

} // namespace

int relpose_2l2p(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                 const std::vector<Eigen::Vector3d> &d1, const std::vector<Eigen::Vector3d> &d2,
                 CameraPoseVector *output) {
    output->clear();
    std::vector<Eigen::Matrix3d> rotations;
    direction_pair_rotations(d1[0].normalized(), d1[1].normalized(), d2[0].normalized(), d2[1].normalized(),
                             &rotations);

    for (const Eigen::Matrix3d &R : rotations) {
        // epipolar constraint x2^T [t]x R x1 = t^T (R x1 x x2) = 0 for both points
        const Eigen::Vector3d n0 = (R * x1[0]).cross(x2[0]);
        const Eigen::Vector3d n1 = (R * x1[1]).cross(x2[1]);
        Eigen::Vector3d t = n0.cross(n1);
        const double nt = t.norm();
        if (nt < 1e-12 * n0.norm() * n1.norm())
            continue;
        t /= nt;
        for (int sign : {1, -1}) {
            const Eigen::Vector3d ts = sign * t;
            bool valid = true;
            for (int i = 0; i < 2 && valid; ++i) {
                double l1, l2;
                valid = triangulate_depths(R, ts, x1[i], x2[i], &l1, &l2) && l1 > 0 && l2 > 0;
            }
            if (valid) {
                output->emplace_back(R, ts);
                break;
            }
        }
    }
    return output->size();
}

} // namespace poselib
