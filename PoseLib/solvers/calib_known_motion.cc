//
// Created by kocur on 02-Oct-24.
//

#include "calib_known_motion.h"
#include "PoseLib/misc/univariate.h"

#include <iostream>

namespace poselib {

//void solve_quadratic(double a, double b, double c, std::vector<double> *sols){
//    sols->reserve(2);
//
//    // linear equation
//    if (std::abs(a) < 1e-12) {
//        sols->push_back(-c / b);
//        return;
//    }
//
//    double s = std::sqrt(b * b - 4*a*c);
//
//    // imaginary roots
//    if (std::isnan(s))
//        return;
//
//    double sign = (0.0 < s) - (s < 0.0);
//    double temp = -0.5 * (b + sign * s);
//    double f1 = temp / a;
//    double f2 = c / temp;
//
//    sols->push_back(f1);
//    sols->push_back(f2);
//}

void calib_known_motion_f_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                              const CameraPose &pose, ImagePairVector *models) {
    models->reserve(2);

    double a = E(2, 2)*(-E(2, 0)*x1[0](0) + E(2, 0)*x1[1](0) - E(2, 1)*x1[0](1) + E(2, 1)*x1[1](1))/(E(0, 2)*x2[0](0) - E(0, 2)*x2[1](0) + E(1, 2)*x2[0](1) - E(1, 2)*x2[1](1));
    double b = (-E(0, 0)*E(2, 2)*x1[0](0)*x2[0](0) + E(0, 0)*E(2, 2)*x1[1](0)*x2[1](0) - E(0, 1)*E(2, 2)*x1[0](1)*x2[0](0) + E(0, 1)*E(2, 2)*x1[1](1)*x2[1](0) - E(0, 2)*E(2, 0)*x1[0](0)*x2[1](0) + E(0, 2)*E(2, 0)*x1[1](0)*x2[0](0) - E(0, 2)*E(2, 1)*x1[0](1)*x2[1](0) + E(0, 2)*E(2, 1)*x1[1](1)*x2[0](0) - E(1, 0)*E(2, 2)*x1[0](0)*x2[0](1) + E(1, 0)*E(2, 2)*x1[1](0)*x2[1](1) - E(1, 1)*E(2, 2)*x1[0](1)*x2[0](1) + E(1, 1)*E(2, 2)*x1[1](1)*x2[1](1) - E(1, 2)*E(2, 0)*x1[0](0)*x2[1](1) + E(1, 2)*E(2, 0)*x1[1](0)*x2[0](1) - E(1, 2)*E(2, 1)*x1[0](1)*x2[1](1) + E(1, 2)*E(2, 1)*x1[1](1)*x2[0](1))/(E(0, 2)*x2[0](0) - E(0, 2)*x2[1](0) + E(1, 2)*x2[0](1) - E(1, 2)*x2[1](1));
    double c = (-E(0, 0)*E(0, 2)*x1[0](0)*x2[0](0)*x2[1](0) + E(0, 0)*E(0, 2)*x1[1](0)*x2[0](0)*x2[1](0) - E(0, 0)*E(1, 2)*x1[0](0)*x2[0](0)*x2[1](1) + E(0, 0)*E(1, 2)*x1[1](0)*x2[0](1)*x2[1](0) - E(0, 1)*E(0, 2)*x1[0](1)*x2[0](0)*x2[1](0) + E(0, 1)*E(0, 2)*x1[1](1)*x2[0](0)*x2[1](0) - E(0, 1)*E(1, 2)*x1[0](1)*x2[0](0)*x2[1](1) + E(0, 1)*E(1, 2)*x1[1](1)*x2[0](1)*x2[1](0) - E(0, 2)*E(1, 0)*x1[0](0)*x2[0](1)*x2[1](0) + E(0, 2)*E(1, 0)*x1[1](0)*x2[0](0)*x2[1](1) - E(0, 2)*E(1, 1)*x1[0](1)*x2[0](1)*x2[1](0) + E(0, 2)*E(1, 1)*x1[1](1)*x2[0](0)*x2[1](1) - E(1, 0)*E(1, 2)*x1[0](0)*x2[0](1)*x2[1](1) + E(1, 0)*E(1, 2)*x1[1](0)*x2[0](1)*x2[1](1) - E(1, 1)*E(1, 2)*x1[0](1)*x2[0](1)*x2[1](1) + E(1, 1)*E(1, 2)*x1[1](1)*x2[0](1)*x2[1](1))/(E(0, 2)*x2[0](0) - E(0, 2)*x2[1](0) + E(1, 2)*x2[0](1) - E(1, 2)*x2[1](1));

    double sols[2];
    int num_sols = univariate::solve_quadratic_real(a, b, c, sols);

    for (int i = 0; i < num_sols; ++i) {
        double sol = sols[i];
        if (sol < 0.0 or std::isnan(sol))
            continue;
        
        double f2 = sol;
        double f1 = (-E(0, 0)*x1[0](0)*x2[0](0) + E(0, 0)*x1[1](0)*x2[1](0) - E(0, 1)*x1[0](1)*x2[0](0) + E(0, 1)*x1[1](1)*x2[1](0) - E(1, 0)*x1[0](0)*x2[0](1) + E(1, 0)*x1[1](0)*x2[1](1) - E(1, 1)*x1[0](1)*x2[0](1) + E(1, 1)*x1[1](1)*x2[1](1) - E(2, 0)*f2*x1[0](0) + E(2, 0)*f2*x1[1](0) - E(2, 1)*f2*x1[0](1) + E(2, 1)*f2*x1[1](1))/(E(0, 2)*x2[0](0) - E(0, 2)*x2[1](0) + E(1, 2)*x2[0](1) - E(1, 2)*x2[1](1));
        
        if (f1 < 0.0 or std::isnan(f1))
            continue;

        Camera cam1 = Camera("SIMPLE_PINHOLE", {f1, 0.0, 0.0}, -1, -1);
        Camera cam2 = Camera("SIMPLE_PINHOLE", {f2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, cam1, cam2);        
    }
}

void calib_known_motion_f_3p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                             const CameraPose &pose, ImagePairVector *models) {
    Eigen::Matrix2d A;
    Eigen::Vector2d b;

    for (int row = 0; row < 2; ++row){
        int j = row + 1;
        // f1
        A(row, 0) = E(0, 2)*x2[row](0) - E(0, 2)*x2[j](0) + E(1, 2)*x2[row](1) - E(1, 2)*x2[j](1);
        A(row, 1) = E(2, 0)*x1[row](0) - E(2, 0)*x1[j](0) + E(2, 1)*x1[row](1) - E(2, 1)*x1[j](1);
        b(row) = E(0, 0)*x1[row](0)*x2[row](0) - E(0, 0)*x1[j](0)*x2[j](0) + E(0, 1)*x1[row](1)*x2[row](0) - E(0, 1)*x1[j](1)*x2[j](0) + E(1, 0)*x1[row](0)*x2[row](1) - E(1, 0)*x1[j](0)*x2[j](1) + E(1, 1)*x1[row](1)*x2[row](1) - E(1, 1)*x1[j](1)*x2[j](1);
    }

    Eigen::Vector2d sol = A.inverse() * -b;

    Camera cam1 = Camera("SIMPLE_PINHOLE", {sol(0), 0.0, 0.0}, -1, -1);
    Camera cam2 = Camera("SIMPLE_PINHOLE", {sol(1), 0.0, 0.0}, -1, -1);
    models->emplace_back(pose, cam1, cam2);
}

void calib_known_motion_fpp_7p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Eigen::Matrix3d &E,
                               const CameraPose &pose, ImagePairVector *models) {
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> b;

    for (int row = 0; row < 6; ++row){
        int j = row + 1;
        // f1
        A(row, 0) = E(0, 2)*x2[row](0) - E(0, 2)*x2[j](0) + E(1, 2)*x2[row](1) - E(1, 2)*x2[j](1);
        // px1
        A(row, 1) = -E(0, 0)*x2[row](0) + E(0, 0)*x2[j](0) - E(1, 0)*x2[row](1) + E(1, 0)*x2[j](1);
        // px2
        A(row, 2) = -E(0, 0)*x1[row](0) + E(0, 0)*x1[j](0) - E(0, 1)*x1[row](1) + E(0, 1)*x1[j](1);
        // f2
        A(row, 3) = E(2, 0)*x1[row](0) - E(2, 0)*x1[j](0) + E(2, 1)*x1[row](1) - E(2, 1)*x1[j](1);
        // py1
        A(row, 4) = -E(0, 1)*x2[row](0) + E(0, 1)*x2[j](0) - E(1, 1)*x2[row](1) + E(1, 1)*x2[j](1);
        // py2
        A(row, 5) = -E(1, 0)*x1[row](0) + E(1, 0)*x1[j](0) - E(1, 1)*x1[row](1) + E(1, 1)*x1[j](1);
        b(row) = E(0, 0)*x1[row](0)*x2[row](0) - E(0, 0)*x1[j](0)*x2[j](0) + E(0, 1)*x1[row](1)*x2[row](0) - E(0, 1)*x1[j](1)*x2[j](0) + E(1, 0)*x1[row](0)*x2[row](1) - E(1, 0)*x1[j](0)*x2[j](1) + E(1, 1)*x1[row](1)*x2[row](1) - E(1, 1)*x1[j](1)*x2[j](1);
    }

    Eigen::Matrix<double, 6, 1> sol = A.colPivHouseholderQr().solve(-b);

    Camera cam1 = Camera("SIMPLE_PINHOLE", {sol(0), sol(1), sol(2)}, -1, -1);
    Camera cam2 = Camera("SIMPLE_PINHOLE", {sol(3), sol(4), sol(5)}, -1, -1);
    models->emplace_back(pose, cam1, cam2);
}

void calib_known_motion_shared_f_1p(const poselib::Point2D &x1, const poselib::Point2D &x2, const Eigen::Matrix3d &E,
                                     const CameraPose &pose, poselib::ImagePairVector *models) {
    double a = E(2, 2);
    double b = E(2,0) * x1(0) + x1(1) * E(2, 1) + x2(0) * E(0, 2) + x2(1) * E(1, 2);
    double c = x1(0) * x2(0) * E(0, 0) + x1(0) * x2(1) * E(1, 0) + x1(1) * x2(0) * E(0, 1) + x1(1) * x2(1) * E(1, 1);

    models->reserve(2);

    double sols[2];
    int num_sols = univariate::solve_quadratic_real(a, b, c, sols);

    for (int i = 0; i < num_sols; ++i) {
        double sol = sols[i];
        if (sol > 0.0 and not std::isnan(sol)) {
            Camera cam = Camera("SIMPLE_PINHOLE", {sol, 0.0, 0.0}, -1, -1);
            models->emplace_back(pose, cam, cam);
        }
    }
}

void calib_known_motion_shared_f_2p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                    const Eigen::Matrix3d &E, const CameraPose &pose,
                                    poselib::ImagePairVector *models) {
    double f = (-E(0, 0)*x1[0](0)*x2[0](0) + E(0, 0)*x1[1](0)*x2[1](0) - E(0, 1)*x1[0](1)*x2[0](0) + E(0, 1)*x1[1](1)*x2[1](0) - E(1, 0)*x1[0](0)*x2[0](1) + E(1, 0)*x1[1](0)*x2[1](1) - E(1, 1)*x1[0](1)*x2[0](1) + E(1, 1)*x1[1](1)*x2[1](1))/(E(0, 2)*x2[0](0) - E(0, 2)*x2[1](0) + E(1, 2)*x2[0](1) - E(1, 2)*x2[1](1) + E(2, 0)*x1[0](0) - E(2, 0)*x1[1](0) + E(2, 1)*x1[0](1) - E(2, 1)*x1[1](1));

    Camera cam = Camera("SIMPLE_PINHOLE", {f, 0.0, 0.0}, -1, -1);
    models->emplace_back(pose, cam, cam);
}

void calib_known_motion_shared_fpp_4p(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                      const Eigen::Matrix3d &E, const CameraPose &pose,
                                      ImagePairVector *models){
    Eigen::Matrix3d A;
    Eigen::Vector3d b;

    // constexpr int is[3] = {0, 0, 0};
    // constexpr int js[3] = {1, 2, 3};
    
    for (int row = 0; row < 3; ++row){
//        int i = is[row];
        int i = row;
        int j = row + 1;
        // focal
        A(row, 0) = E(0, 2)*x2[i](0) - E(0, 2)*x2[j](0) + E(1, 2)*x2[i](1) - E(1, 2)*x2[j](1) + E(2, 0)*x1[i](0) - E(2, 0)*x1[j](0) + E(2, 1)*x1[i](1) - E(2, 1)*x1[j](1);
        // px
        A(row, 1) = -E(0, 0)*x1[i](0) + E(0, 0)*x1[j](0) - E(0, 0)*x2[i](0) + E(0, 0)*x2[j](0) - E(0, 1)*x1[i](1) + E(0, 1)*x1[j](1) - E(1, 0)*x2[i](1) + E(1, 0)*x2[j](1);
        // py
        A(row, 2) = -E(0, 1)*x2[i](0) + E(0, 1)*x2[j](0) - E(1, 0)*x1[i](0) + E(1, 0)*x1[j](0) - E(1, 1)*x1[i](1) + E(1, 1)*x1[j](1) - E(1, 1)*x2[i](1) + E(1, 1)*x2[j](1);
        // 1
        b(row) = E(0, 0)*x1[i](0)*x2[i](0) - E(0, 0)*x1[j](0)*x2[j](0) + E(0, 1)*x1[i](1)*x2[i](0) - E(0, 1)*x1[j](1)*x2[j](0) + E(1, 0)*x1[i](0)*x2[i](1) - E(1, 0)*x1[j](0)*x2[j](1) + E(1, 1)*x1[i](1)*x2[i](1) - E(1, 1)*x1[j](1)*x2[j](1);
    }

    Eigen::Vector3d sol = A.inverse() * -b;

    Camera camera = Camera("SIMPLE_PINHOLE", {sol(0), sol(1), sol(2)}, -1, -1);
    ImagePair image_pair = ImagePair(pose, camera, camera);
    models->push_back(image_pair);
}

} // namespace poselib
