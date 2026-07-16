#include "relpose_monodepth_3pt_varying_focal.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/types.h"

#include <cmath>
#include <limits>

namespace poselib {
int relpose_monodepth_3pt_varying_focal(const std::vector<Eigen::Vector3d> &x1h,
                                        const std::vector<Eigen::Vector3d> &x2h, const std::vector<double> &depth1,
                                        const std::vector<double> &depth2, std::vector<MonoDepthImagePair> *models) {
    models->clear();
    models->reserve(1);

    double a[18];
    a[0] = x1h[0][0] * depth1[0];
    a[1] = x1h[1][0] * depth1[1];
    a[2] = x1h[2][0] * depth1[2];
    a[3] = x1h[0][1] * depth1[0];
    a[4] = x1h[1][1] * depth1[1];
    a[5] = x1h[2][1] * depth1[2];
    a[6] = depth1[0];
    a[7] = depth1[1];
    a[8] = depth1[2];

    a[9] = x2h[0][0] * depth2[0];
    a[10] = x2h[1][0] * depth2[1];
    a[11] = x2h[2][0] * depth2[2];
    a[12] = x2h[0][1] * depth2[0];
    a[13] = x2h[1][1] * depth2[1];
    a[14] = x2h[2][1] * depth2[2];
    a[15] = depth2[0];
    a[16] = depth2[1];
    a[17] = depth2[2];

    double b[18];
    b[0] = a[0] - a[1];
    b[1] = a[3] - a[4];
    b[2] = a[6] - a[7];
    b[3] = a[0] - a[2];
    b[4] = a[3] - a[5];
    b[5] = a[6] - a[8];
    b[6] = a[1] - a[2];
    b[7] = a[4] - a[5];
    b[8] = a[7] - a[8];
    b[9] = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];
    b[12] = a[9] - a[11];
    b[13] = a[12] - a[14];
    b[14] = a[15] - a[17];
    b[15] = a[10] - a[11];
    b[16] = a[13] - a[14];
    b[17] = a[16] - a[17];

    Eigen::Matrix3d A;
    A << std::pow(b[0], 2) + std::pow(b[1], 2), -std::pow(b[9], 2) - std::pow(b[10], 2), -std::pow(b[11], 2),
        std::pow(b[3], 2) + std::pow(b[4], 2), -std::pow(b[12], 2) - std::pow(b[13], 2), -std::pow(b[14], 2),
        std::pow(b[6], 2) + std::pow(b[7], 2), -std::pow(b[15], 2) - std::pow(b[16], 2), -std::pow(b[17], 2);
    Eigen::Vector3d B;
    B << b[2] * b[2], b[5] * b[5], b[8] * b[8];
    Eigen::Vector3d sol = -A.partialPivLu().solve(B);

    if (sol(0) > 0 && sol(1) > 0 && sol(2) > 0) {
        double f = std::sqrt(sol(0));
        double s = std::sqrt(sol(2));
        double w = std::sqrt(sol(1) / sol(2));

        Eigen::Matrix3d K1inv;
        K1inv << f, 0, 0, 0, f, 0, 0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << w, 0, 0, 0, w, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * ((depth2[0]) * K2inv * x2h[0] - (depth2[1]) * K2inv * x2h[1]);
        Eigen::Vector3d v2 = s * ((depth2[0]) * K2inv * x2h[0] - (depth2[2]) * K2inv * x2h[2]);
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * K1inv * x1h[0] - (depth1[1]) * K1inv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * K1inv * x1h[0] - (depth1[2]) * K1inv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * K1inv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * K2inv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        double focal1 = 1.0 / f;
        double focal2 = 1.0 / w;

        MonoDepthTwoViewGeometry pose = MonoDepthTwoViewGeometry(rot, trans, s);
        Camera camera1 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{focal1, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{focal2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }

    return models->size();
}

void relpose_monodepth_varying_focal_4p4d(const std::vector<Eigen::Vector3d> &x1,
                                              const std::vector<Eigen::Vector3d> &x2,
                                              const std::vector<double> &d1, const std::vector<double> &d2,
                                              std::vector<MonoDepthImagePair> *models) {
    models->clear();
    Eigen::MatrixXd coefficients(12, 12);
    int i;

    // Form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    size_t row = 0;
    for (i = 0; i < 4; i++)
    {
        double u11 = x1[i](0), v11 = x1[i](1), u12 = x2[i](0), v12 = x2[i](1);
        double q1 = d1[i], q2 = d2[i];
        double q = q2 / q1;

        coefficients(row, 0) = -u11;
        coefficients(row, 1) = -v11;
        coefficients(row, 2) = -1;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = 0;
        coefficients(row, 10) = q;
        coefficients(row, 11) = -q * v12;
        ++row;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = -u11;
        coefficients(row, 4) = -v11;
        coefficients(row, 5) = -1;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = -q;
        coefficients(row, 10) = 0;
        coefficients(row, 11) = q * u12;
        ++row;

        if (i == 3)
            break;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = -u11;
        coefficients(row, 7) = -v11;
        coefficients(row, 8) = -1;
        coefficients(row, 9) = q * v12;
        coefficients(row, 10) = -q * u12;
        coefficients(row, 11) = 0;
        ++row;
    }

    Eigen::Matrix<double, 12, 1> f1 = coefficients.block<11, 11>(0, 0).partialPivLu().solve(-coefficients.block<11, 1>(0, 11)).homogeneous();

    Eigen::Matrix3d F;
    F << f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8];

    //    std::cout << "F: " << std::endl << F << std::endl;
    //    std::cout << "Ep: " << x2h[0].transpose() * F * x1h[0] << std::endl;
    //    std::cout << "Det: " << F.determinant() << std::endl;

    std::pair<Camera, Camera> cameras = focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

    Camera camera1 = cameras.first;
    Camera camera2 = cameras.second;

    const double focal1 = camera1.focal();
    const double focal2 = camera2.focal();

    if (!std::isfinite(focal1) || !std::isfinite(focal2) || focal1 <= 0.0 || focal2 <= 0.0)
        return;

    //    if (focal1 < opt.max_focal_1 or focal1 > opt.max_focal_1 or
    //        focal2 < opt.min_focal_2 or focal2 > opt.max_focal_2)
    //        return;

    Eigen::DiagonalMatrix<double, 3> K1(focal1, focal1, 1.0);
    Eigen::DiagonalMatrix<double, 3> K2(focal2, focal2, 1.0);

    Eigen::Matrix3d E = K2 * F * K1;

    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = Eigen::Vector3d(x1[i](0) / focal1, x1[i](1) / focal1, 1.0).normalized();
        x2h[i] = Eigen::Vector3d(x2[i](0) / focal2, x2[i](1) / focal2, 1.0).normalized();
    }

    std::vector<CameraPose> poses;
    motion_from_essential(E, x1h, x2h, &poses);

    models->reserve(poses.size());

    for (const CameraPose &pose : poses) {
        const Eigen::Vector3d bearing1_0(x1[0](0) / focal1, x1[0](1) / focal1, 1.0);
        const Eigen::Vector3d bearing2_0(x2[0](0) / focal2, x2[0](1) / focal2, 1.0);
        const Eigen::Vector3d point2_0 = d2[0] * bearing2_0;

        const Eigen::Vector3d bearing1_1(x1[1](0) / focal1, x1[1](1) / focal1, 1.0);
        const Eigen::Vector3d bearing2_1(x2[1](0) / focal2, x2[1](1) / focal2, 1.0);
        const Eigen::Vector3d point2_1 = d2[1] * bearing2_1;

        const Eigen::Vector3d lhs = point2_1 - point2_0;
        const Eigen::Vector3d rhs = pose.rotate(d1[1] * bearing1_1 - d1[0] * bearing1_0);
        const double scale_numerator = lhs.dot(rhs);
        const double scale_denominator = lhs.squaredNorm();
        const double scale_reference = point2_1.squaredNorm() + point2_0.squaredNorm();

        constexpr double kScaleDegeneracyTolerance = 64.0 * std::numeric_limits<double>::epsilon();
        if (!std::isfinite(scale_denominator) || !std::isfinite(scale_reference) ||
            scale_denominator <= kScaleDegeneracyTolerance * scale_reference)
            continue;
        const double scale = scale_numerator / scale_denominator;
        if (!std::isfinite(scale) || scale <= 0.0)
            continue;

        CameraPose scaled_pose = pose;
        scaled_pose.t = scale * d2[0] * bearing2_0 - pose.rotate(d1[0] * bearing1_0);
        if (!scaled_pose.t.allFinite())
            continue;
        models->emplace_back(MonoDepthTwoViewGeometry(scaled_pose, scale), camera1, camera2);
    }
}
} // namespace poselib
