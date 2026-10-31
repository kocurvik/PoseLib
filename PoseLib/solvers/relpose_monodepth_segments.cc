#include "relpose_monodepth_segments.h"

#include <cmath>

namespace poselib {

int relpose_monodepth_1s1p(const Line3D &S1, const Line3D &S2, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2,
                           std::vector<MonoDepthTwoViewGeometry> *output) {
    output->clear();
    Eigen::Vector3d a = S1.X2 - S1.X1, b = S2.X2 - S2.X1;
    const double la = a.norm(), lb = b.norm();
    if (la < 1e-12 || lb < 1e-12)
        return 0;
    a /= la;
    b /= lb;
    const double scale = la / lb;

    const Eigen::Vector3d m1 = 0.5 * (S1.X1 + S1.X2);
    const Eigen::Vector3d p = 0.5 * scale * (S2.X1 + S2.X2);
    const Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(a, b).toRotationMatrix();

    // With R = Rb(angle) * R0 and t = p - R * m1: x2^T [t]_x R x1 = (x2 x p)^T R x1 - x2^T R (m1 x x1)
    const Eigen::Vector3d g = x2.cross(p);
    const Eigen::Vector3d u = R0 * x1, v = R0 * m1.cross(x1);
    const Eigen::Vector3d u_par = b.dot(u) * b, v_par = b.dot(v) * b;
    const double A = g.dot(u - u_par) - x2.dot(v - v_par);
    const double B = g.dot(b.cross(u)) - x2.dot(b.cross(v));
    const double C = g.dot(u_par) - x2.dot(v_par);

    const double rho = std::sqrt(A * A + B * B);
    if (rho < 1e-12 || std::abs(C) > rho)
        return 0;
    const double phi = std::atan2(B, A);
    const double delta = std::acos(-C / rho);

    for (double angle : {phi + delta, phi - delta}) {
        const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, b).toRotationMatrix() * R0;
        const Eigen::Vector3d t = p - R * m1;

        // depths of the point: lambda1 * R * x1 + t = lambda2 * x2
        const Eigen::Vector3d Rx1 = R * x1;
        const double rr = Rx1.squaredNorm(), xx = x2.squaredNorm(), rx = Rx1.dot(x2);
        const double det = rx * rx - rr * xx;
        if (std::abs(det) < 1e-12 * rr * xx)
            continue;
        const double tr = t.dot(Rx1), tx = t.dot(x2);
        const double lambda1 = (xx * tr - rx * tx) / det;
        const double lambda2 = (rx * tr - rr * tx) / det;
        if (lambda1 > 0 && lambda2 > 0)
            output->emplace_back(CameraPose(R, t), scale);
    }
    return output->size();
}

int relpose_monodepth_2s(const Line3D &Sa1, const Line3D &Sb1, const Line3D &Sa2, const Line3D &Sb2,
                         std::vector<MonoDepthTwoViewGeometry> *output) {
    output->clear();
    const Eigen::Vector3d X1[4] = {Sa1.X1, Sa1.X2, Sb1.X1, Sb1.X2};
    const Eigen::Vector3d X2[4] = {Sa2.X1, Sa2.X2, Sb2.X1, Sb2.X2};
    const Eigen::Vector3d c1 = 0.25 * (X1[0] + X1[1] + X1[2] + X1[3]);
    const Eigen::Vector3d c2 = 0.25 * (X2[0] + X2[1] + X2[2] + X2[3]);

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    double var1 = 0, var2 = 0;
    for (int k = 0; k < 4; ++k) {
        const Eigen::Vector3d y1 = X1[k] - c1, y2 = X2[k] - c2;
        H += y2 * y1.transpose();
        var1 += y1.squaredNorm();
        var2 += y2.squaredNorm();
    }
    if (var1 < 1e-24 || var2 < 1e-24)
        return 0;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()(1) < 1e-6 * svd.singularValues()(0))
        return 0;
    const Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Vector3d sign(1, 1, (U * V.transpose()).determinant() < 0 ? -1 : 1);
    const Eigen::Matrix3d R = U * sign.asDiagonal() * V.transpose();

    const double scale = std::sqrt(var1 / var2);
    output->emplace_back(CameraPose(R, Eigen::Vector3d(scale * c2 - R * c1)), scale);
    return 1;
}

} // namespace poselib
