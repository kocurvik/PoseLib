#include "threeview_lines.h"

#include "PoseLib/solvers/relpose_upright_3pt.h"
#include "PoseLib/solvers/up2p.h"

#include <limits>

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

void line_pair_rotations(const std::vector<Line3D> &L1, const std::vector<Line3D> &Lk,
                         std::vector<Eigen::Matrix3d> *rotations) {
    direction_pair_rotations((L1[0].X2 - L1[0].X1).normalized(), (L1[1].X2 - L1[1].X1).normalized(),
                             (Lk[0].X2 - Lk[0].X1).normalized(), (Lk[1].X2 - Lk[1].X1).normalized(), rotations);
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

// Depths of the point with bearings x1, x2 (view 1 and view 2 frames) given the pose: lambda2 x2 = R lambda1 x1 + t
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

int threeview_1l3p(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                   const std::vector<Eigen::Vector3d> &x3, const Eigen::Vector3d &d1, const Eigen::Vector3d &d2,
                   const Eigen::Vector3d &d3, std::vector<ThreeViewCameraPose> *output) {
    output->clear();

    // Rotate views 1, 2 so that the line direction is the y axis, the remaining rotation is about y
    const Eigen::Matrix3d Rc1 = Eigen::Quaterniond::FromTwoVectors(d1, Eigen::Vector3d::UnitY()).toRotationMatrix();
    std::vector<Eigen::Vector3d> x1u(3), x2u(3);
    for (int i = 0; i < 3; ++i)
        x1u[i] = Rc1 * x1[i].normalized();

    std::vector<Eigen::Vector3d> x3_up2p = {x3[0], x3[1]};
    std::vector<Eigen::Vector3d> X_up2p(2);
    const Eigen::Vector3d x3_check = x3[2].normalized();

    for (int s2 : {1, -1}) {
        const Eigen::Matrix3d Rc2 =
            Eigen::Quaterniond::FromTwoVectors(s2 * d2, Eigen::Vector3d::UnitY()).toRotationMatrix();
        for (int i = 0; i < 3; ++i)
            x2u[i] = Rc2 * x2[i].normalized();

        CameraPoseVector poses12_upright;
        relpose_upright_3pt(x1u, x2u, &poses12_upright);

        for (const CameraPose &pu : poses12_upright) {
            const Eigen::Matrix3d R12 = Rc2.transpose() * pu.R() * Rc1;
            const Eigen::Vector3d t12 = Rc2.transpose() * pu.t;

            // all three points in front of cameras 1, 2
            Eigen::Vector3d X[3];
            bool valid = true;
            for (int i = 0; i < 3 && valid; ++i) {
                double l1, l2;
                const Eigen::Vector3d b1 = x1[i].normalized();
                valid = triangulate_depths(R12, t12, b1, x2[i].normalized(), &l1, &l2) && l1 > 0 && l2 > 0;
                X[i] = l1 * b1;
            }
            if (!valid)
                continue;
            X_up2p[0] = X[0];
            X_up2p[1] = X[1];

            // view 3: absolute pose with known direction, keep the solution best explaining point 2
            double best_err = std::numeric_limits<double>::max();
            CameraPose best13;
            for (int s3 : {1, -1}) {
                CameraPoseVector poses13;
                up2p(x3_up2p, X_up2p, s3 * d3, d1, &poses13);
                for (const CameraPose &p13 : poses13) {
                    const Eigen::Vector3d Y = p13.R() * X[2] + p13.t;
                    if (Y(2) <= 0)
                        continue;
                    const double err = 1.0 - Y.normalized().dot(x3_check);
                    if (err < best_err) {
                        best_err = err;
                        best13 = p13;
                    }
                }
            }
            if (best_err == std::numeric_limits<double>::max())
                continue;
            output->emplace_back(CameraPose(R12, t12), best13);
        }
    }
    return output->size();
}

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

namespace {

// Generated action matrix solver (theory/solver_rotation_2sift_orientation.cpp). Solves
//     det([a_i, K(x) b_i, K(y) c_i]) = 0, i = 0, 1,
// with K(k) = [1-k^2, 0, 2k; 0, 1+k^2, 0; -2k, 0, 1-k^2] / (1+k^2) the Cayley parametrized rotation about the y axis.
// Input d = [a_0 a_1 b_0 b_1 c_0 c_1] (18 values). Returns up to 8 real solutions (x, y).
int solve_rotation_2p2l(const double *d, std::vector<Eigen::Vector2d> *sols) {
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
        return 0;
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
    return sols->size();
}

inline Eigen::Matrix3d cayley_y(double k) {
    Eigen::Matrix3d K;
    K << 1 - k * k, 0, 2 * k, 0, 1 + k * k, 0, -2 * k, 0, 1 - k * k;
    return K / (1 + k * k);
}

} // namespace

int threeview_2p2l(const std::vector<Eigen::Vector3d> &X1, const std::vector<Eigen::Vector3d> &X2,
                   const std::vector<Eigen::Vector3d> &X3, const std::vector<Eigen::Vector3d> &l1,
                   const std::vector<Eigen::Vector3d> &l2, const std::vector<Eigen::Vector3d> &l3,
                   std::vector<ThreeViewCameraPose> *output) {
    output->clear();

    // The vector between the two points is rotated by R_k into view k. With Q_k u_k ~ e_y the rotations are
    // R_k = Q_k^T K_k^T Q_1 with K_k a rotation about y (theory/test_twosift3view.m uses Householder reflections).
    const Eigen::Vector3d u[3] = {X1[0] - X1[1], X2[0] - X2[1], X3[0] - X3[1]};
    const double n1 = u[0].norm();
    if (n1 < 1e-12)
        return 0;
    const double s[3] = {1.0, u[1].norm() / n1, u[2].norm() / n1};
    if (s[1] < 1e-12 || s[2] < 1e-12)
        return 0;

    Eigen::Matrix3d Q[3];
    for (int k = 0; k < 3; ++k)
        Q[k] = Eigen::Quaterniond::FromTwoVectors(u[k], Eigen::Vector3d::UnitY()).toRotationMatrix();

    // det([l1, R2^T l2, R3^T l3]) = det([Q1 l1, K2 Q2 l2, K3 Q3 l3])
    const std::vector<Eigen::Vector3d> *ls[3] = {&l1, &l2, &l3};
    double data[18];
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < 2; ++i) {
            const Eigen::Vector3d ql = Q[k] * (*ls[k])[i].normalized();
            for (int j = 0; j < 3; ++j)
                data[6 * k + 3 * i + j] = ql(j);
        }
    }

    std::vector<Eigen::Vector2d> sols;
    solve_rotation_2p2l(data, &sols);

    for (const Eigen::Vector2d &k : sols) {
        const Eigen::Matrix3d R2 = Q[1].transpose() * cayley_y(k(0)).transpose() * Q[0];
        const Eigen::Matrix3d R3 = Q[2].transpose() * cayley_y(k(1)).transpose() * Q[0];
        const Eigen::Vector3d t2 = X2[0] / s[1] - R2 * X1[0];
        const Eigen::Vector3d t3 = X3[0] / s[2] - R3 * X1[0];
        output->emplace_back(CameraPose(R2, t2), CameraPose(R3, t3));
    }
    return output->size();
}

int threeview_2p2l_ratio(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                         const std::vector<Eigen::Vector3d> &x3, double r21, double r22, double r31,
                         const std::vector<Eigen::Vector3d> &l1, const std::vector<Eigen::Vector3d> &l2,
                         const std::vector<Eigen::Vector3d> &l3, std::vector<ThreeViewCameraPose> *output) {
    output->clear();
    // points with unit z (depth = z coordinate), point 0 has depth 1 in view 1
    const Eigen::Vector3d a0 = x1[0] / x1[0](2), a1 = x1[1] / x1[1](2);
    const Eigen::Vector3d b0 = r21 * x2[0] / x2[0](2), b1 = r22 * x2[1] / x2[1](2);
    const Eigen::Vector3d c0 = r31 * x3[0] / x3[0](2), c1 = x3[1] / x3[1](2);

    // |a0 - m a1| = |b0 - m b1|: quadratic in m (depth of point 1 in view 1)
    const double qa = a1.squaredNorm() - b1.squaredNorm();
    const double qb = -2 * a0.dot(a1) + 2 * b0.dot(b1);
    const double qc = a0.squaredNorm() - b0.squaredNorm();
    std::vector<double> ms;
    if (std::abs(qa) < 1e-12 * (std::abs(qb) + std::abs(qc))) {
        if (std::abs(qb) > 0)
            ms.push_back(-qc / qb);
    } else {
        const double disc = qb * qb - 4 * qa * qc;
        if (disc < 0)
            return 0;
        const double sq = std::sqrt(disc);
        ms.push_back((-qb + sq) / (2 * qa));
        ms.push_back((-qb - sq) / (2 * qa));
    }

    std::vector<Eigen::Vector3d> X1(2), X2(2), X3(2);
    std::vector<ThreeViewCameraPose> models;
    for (double m : ms) {
        if (m <= 0)
            continue;
        // |a0 - m a1| = |c0 - n c1|: quadratic in n (depth of point 1 in view 3)
        const double len2 = (a0 - m * a1).squaredNorm();
        const double pa = c1.squaredNorm(), pb = -2 * c0.dot(c1), pc = c0.squaredNorm() - len2;
        const double disc = pb * pb - 4 * pa * pc;
        if (disc < 0)
            continue;
        const double sq = std::sqrt(disc);
        for (double n : {(-pb + sq) / (2 * pa), (-pb - sq) / (2 * pa)}) {
            if (n <= 0)
                continue;
            X1[0] = a0;
            X1[1] = m * a1;
            X2[0] = b0;
            X2[1] = m * b1;
            X3[0] = c0;
            X3[1] = n * c1;
            threeview_2p2l(X1, X2, X3, l1, l2, l3, &models);
            output->insert(output->end(), models.begin(), models.end());
        }
    }
    return output->size();
}

} // namespace poselib
