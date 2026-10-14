#include "threeview_lines.h"

namespace poselib {

namespace {

// Orthonormal frame with first axis along u and second axis in the plane of u and d
inline bool frame_from_vector_and_direction(const Eigen::Vector3d &u, const Eigen::Vector3d &d, Eigen::Matrix3d *F) {
    Eigen::Vector3d e1 = u.normalized();
    Eigen::Vector3d e2 = d - d.dot(e1) * e1;
    double n2 = e2.norm();
    if (n2 < 1e-8 * d.norm())
        return false;
    e2 /= n2;
    F->col(0) = e1;
    F->col(1) = e2;
    F->col(2) = e1.cross(e2);
    return true;
}

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

// Feet of the common perpendicular of two 3D lines
inline bool common_perpendicular(const Line3D &La, const Line3D &Lb, Eigen::Vector3d *Fa, Eigen::Vector3d *Fb) {
    const Eigen::Vector3d u = La.X2 - La.X1;
    const Eigen::Vector3d v = Lb.X2 - Lb.X1;
    const Eigen::Vector3d w0 = La.X1 - Lb.X1;
    const double a = u.dot(u), b = u.dot(v), c = v.dot(v), d = u.dot(w0), e = v.dot(w0);
    const double den = a * c - b * b;
    if (den < 1e-12 * a * c)
        return false;
    const double s = (b * e - c * d) / den;
    const double t = (a * e - b * d) / den;
    *Fa = La.X1 + s * u;
    *Fb = Lb.X1 + t * v;
    return true;
}

// Candidate rotations R such that R * frame(view 1) = frame(view k) for all consistent sign choices
// of the line directions in view k.
void line_pair_rotations(const std::vector<Line3D> &L1, const std::vector<Line3D> &Lk,
                         std::vector<Eigen::Matrix3d> *rotations) {
    rotations->clear();
    const Eigen::Vector3d da1 = (L1[0].X2 - L1[0].X1).normalized();
    const Eigen::Vector3d db1 = (L1[1].X2 - L1[1].X1).normalized();
    const Eigen::Vector3d dak = (Lk[0].X2 - Lk[0].X1).normalized();
    const Eigen::Vector3d dbk = (Lk[1].X2 - Lk[1].X1).normalized();

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

// Check that the midpoints of the view 1 segments are in front of camera k
inline bool cheirality(const std::vector<Line3D> &L1, const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    for (const Line3D &L : L1) {
        Eigen::Vector3d m = R * (0.5 * (L.X1 + L.X2)) + t;
        if (m(2) <= 0)
            return false;
    }
    return true;
}

void two_line_poses_perpendicular(const std::vector<Line3D> &L1, const std::vector<Line3D> &Lk,
                                  std::vector<CameraPose> *poses) {
    poses->clear();
    Eigen::Vector3d Fa1, Fb1, Fak, Fbk;
    if (!common_perpendicular(L1[0], L1[1], &Fa1, &Fb1))
        return;
    if (!common_perpendicular(Lk[0], Lk[1], &Fak, &Fbk))
        return;

    const double dist1 = (Fb1 - Fa1).norm();
    const double distk = (Fbk - Fak).norm();
    if (dist1 < 1e-10 || distk < 1e-10)
        return;
    const double s = distk / dist1;
    const Eigen::Vector3d m1 = 0.5 * (Fa1 + Fb1);
    const Eigen::Vector3d mk = 0.5 * (Fak + Fbk);

    std::vector<Eigen::Matrix3d> rotations;
    line_pair_rotations(L1, Lk, &rotations);
    for (const Eigen::Matrix3d &R : rotations) {
        Eigen::Vector3d t = mk / s - R * m1;
        if (!cheirality(L1, R, t))
            continue;
        poses->emplace_back(R, t);
    }
}

void two_line_poses_planes(const std::vector<Line3D> &L1, const std::vector<Line3D> &Lk,
                           const std::vector<Eigen::Vector3d> &lk, std::vector<CameraPose> *poses) {
    poses->clear();
    std::vector<Eigen::Matrix3d> rotations;
    line_pair_rotations(L1, Lk, &rotations);

    // With known R each line gives a single constraint l^T (R X + t) = 0 (the second endpoint only constrains R).
    // The two lines thus fix t up to the component along n = l_a x l_b, which is recovered from the depth in view k:
    // the lines R * L1 + t and sigma * Lk (sigma - unknown depth scale) must coincide, i.e. their components
    // orthogonal to the line direction must agree.
    Eigen::Matrix<double, 2, 3> A;
    A.row(0) = lk[0].normalized().transpose();
    A.row(1) = lk[1].normalized().transpose();
    Eigen::Vector3d n = A.row(0).cross(A.row(1)).transpose();
    if (n.norm() < 1e-8)
        return;
    n.normalize();

    for (const Eigen::Matrix3d &R : rotations) {
        Eigen::Vector2d b;
        for (int i = 0; i < 2; ++i)
            b(i) = -A.row(i).dot(R * (0.5 * (L1[i].X1 + L1[i].X2)));
        // minimal norm solution of A t = b
        const Eigen::Vector3d t0 = A.transpose() * (A * A.transpose()).inverse() * b;

        // (I - d d^T) (R X1 + t0 + mu n - sigma Xk) = 0 for both lines
        Eigen::Matrix<double, 6, 2> M;
        Eigen::Matrix<double, 6, 1> rhs;
        for (int i = 0; i < 2; ++i) {
            const Eigen::Vector3d d = (Lk[i].X2 - Lk[i].X1).normalized();
            const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - d * d.transpose();
            M.block<3, 1>(3 * i, 0) = P * n;
            M.block<3, 1>(3 * i, 1) = -P * Lk[i].X1;
            rhs.segment<3>(3 * i) = -P * (R * L1[i].X1 + t0);
        }
        const Eigen::Vector2d sol = M.colPivHouseholderQr().solve(rhs);
        if (!sol.allFinite() || sol(1) <= 0)
            continue;
        const Eigen::Vector3d t = t0 + sol(0) * n;
        if (!cheirality(L1, R, t))
            continue;
        poses->emplace_back(R, t);
    }
}

void combine(const std::vector<CameraPose> &poses12, const std::vector<CameraPose> &poses13,
             std::vector<ThreeViewCameraPose> *output) {
    for (const CameraPose &p12 : poses12)
        for (const CameraPose &p13 : poses13)
            output->emplace_back(p12, p13);
}

} // namespace

int threeview_2p1l(const std::vector<Eigen::Vector3d> &X1, const std::vector<Eigen::Vector3d> &X2,
                   const std::vector<Eigen::Vector3d> &X3, const Eigen::Vector3d &d1, const Eigen::Vector3d &d2,
                   const Eigen::Vector3d &d3, std::vector<ThreeViewCameraPose> *output) {
    output->clear();

    // The vector between the two points is aligned exactly, the line direction fixes the rotation about it.
    // This is equivalent to the Householder + rotation about y formulation in theory/2p1l.m
    const Eigen::Vector3d u1 = X1[0] - X1[1];
    const Eigen::Vector3d u2 = X2[0] - X2[1];
    const Eigen::Vector3d u3 = X3[0] - X3[1];
    const double n1 = u1.norm();
    if (n1 < 1e-12)
        return 0;

    Eigen::Matrix3d F1;
    if (!frame_from_vector_and_direction(u1, d1, &F1))
        return 0;

    std::vector<CameraPose> poses[2];
    const Eigen::Vector3d *uk[2] = {&u2, &u3};
    const Eigen::Vector3d *dk[2] = {&d2, &d3};
    const std::vector<Eigen::Vector3d> *Xk[2] = {&X2, &X3};

    for (int k = 0; k < 2; ++k) {
        const double s = uk[k]->norm() / n1;
        if (s < 1e-12)
            return 0;
        for (int sign : {1, -1}) {
            Eigen::Matrix3d Fk;
            if (!frame_from_vector_and_direction(*uk[k], sign * (*dk[k]), &Fk))
                continue;
            Eigen::Matrix3d R = Fk * F1.transpose();
            Eigen::Vector3d t = (*Xk[k])[0] / s - R * X1[0];
            poses[k].emplace_back(R, t);
        }
    }

    combine(poses[0], poses[1], output);
    return output->size();
}

int threeview_2l(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2, const std::vector<Line3D> &L3,
                 std::vector<ThreeViewCameraPose> *output) {
    output->clear();
    std::vector<CameraPose> poses12, poses13;
    two_line_poses_perpendicular(L1, L2, &poses12);
    if (poses12.empty())
        return 0;
    two_line_poses_perpendicular(L1, L3, &poses13);
    combine(poses12, poses13, output);
    return output->size();
}

int threeview_2l_planes(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2, const std::vector<Line3D> &L3,
                        const std::vector<Eigen::Vector3d> &l2, const std::vector<Eigen::Vector3d> &l3,
                        std::vector<ThreeViewCameraPose> *output) {
    output->clear();
    std::vector<CameraPose> poses12, poses13;
    two_line_poses_planes(L1, L2, l2, &poses12);
    if (poses12.empty())
        return 0;
    two_line_poses_planes(L1, L3, l3, &poses13);
    combine(poses12, poses13, output);
    return output->size();
}

} // namespace poselib
