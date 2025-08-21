//
// Created by kocur on 09-May-25.
//

#include <iostream>
#include "relpose_3v_TRF.h"

namespace poselib {

void compute_solutions_3v_TRF(double k, double t12y, double t13x, double t13y, const Point2D &x1, const Point2D &x2,
                              const Point2D &x3, double alpha, const RelativePoseOptions &opt,
                              std::vector<ImageTriplet> *solutions){
    double c0 = x1(0)*x2(1) - x1(1)*x2(0);
    double c1 = x1(0)*x3(1) - x1(1)*x3(0);
    double c2 = 1.0/c0*1.0/c1;
    double c3 = t12y*x2(0);
    double c4 = t12y*x1(0);
    double c5 = k*x1(1);
    double c6 = x2(0)*x2(0);
    double c7 = x2(1)*x2(1);
    double c8 = k*x2(1);
    double c9 = x1(0)*x1(0);
    double c10 = x1(1)*x1(1);
    double c11 = c3*k;
    double c12 = c4*k;
    double c13 = c10*c11 - c10*c8 + c11*c9 - c12*c6 - c12*c7 + c3 - c4 + c5*c6 + c5*c7 - c8*c9 + x1(1) - x2(1);
    double c14 = (c1*c1)*(c13*c13);
    double c15 = t13y*x3(0);
    double c16 = t13x*x3(1);
    double c17 = t13y*x1(0);
    double c18 = c5*t13x;
    double c19 = x3(0)*x3(0);
    double c20 = x3(1)*x3(1);
    double c21 = c15*k;
    double c22 = c16*k;
    double c23 = c17*k;
    double c24 = c10*c21 - c10*c22 + c15 - c16 - c17 + c18*c19 + c18*c20 - c19*c23 - c20*c23 + c21*c9 - c22*c9 + t13x*x1(1);
    double c25 = alpha*(c0*c0)*(c24*c24);
    double c26 = c14 - c25;
    double c27 = -alpha*t13x*t13x - alpha*t13y*t13y + 1 + t12y*t12y;
    double c28 = std::sqrt(-c27*1.0/c26);
    if (std::isnan(c28))
        return;
    double c29 = 1.0/c27;
    double c30 = c1*c13*c28;
    double c31 = c0*c24*c28;
        
    double w = c2*c28*c29*(-c14 + c25);
    double t12z = -c30;
    double t13z = -c31;

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (w > 0.0){
        Eigen::Vector3d t12, t13;
        t12 << 1.0, t12y, t12z;
        t13 << t13x, t13y, t13z;

        double theta = 180.0 / 3.14159265359 * (std::acos(t12.dot(t13) / (t12.norm() * t13.norm())));

        if (opt.theta_tol == 0.0 or std::fabs(theta - opt.theta) < opt.theta_tol) {
            double t12_norm = t12.norm();
            ThreeViewCameraPose poses = ThreeViewCameraPose(CameraPose(R, t12 / t12_norm),
                                                            CameraPose(R, t13 / t12_norm));
            double focal = 1.0 / w;
            double scaled_k = k * focal * focal;
            Camera camera("SIMPLE_DIVISION", std::vector<double>{focal, 0.0, 0.0, scaled_k}, -1, -1);
            solutions->emplace_back(ImageTriplet(poses, camera));
        }

//        ThreeViewCameraPose neg_poses = ThreeViewCameraPose(CameraPose(R, -t12), CameraPose(R, -t13));
//        solutions->emplace_back(ImageTriplet(neg_poses, camera));

    }

    w = c2*c26*c28*c29;
    t12z = c30;
    t13z = c31;

    if (w > 0.0){

//        std::cout << "k: " << k << std::endl;
//        std::cout << "f: " << 1 / w << std::endl;
        Eigen::Vector3d t12, t13;
        t12 << 1.0, t12y, t12z;
        t13 << t13x, t13y, t13z;

        double theta = 180.0 / 3.14159265359 * (std::acos(t12.dot(t13) / (t12.norm() * t13.norm())));

        if (opt.theta_tol == 0.0 or std::fabs(theta - opt.theta) < opt.theta_tol) {
            double t12_norm = t12.norm();
            ThreeViewCameraPose poses = ThreeViewCameraPose(CameraPose(R, t12 / t12_norm),
                                                            CameraPose(R, t13 / t12_norm));
            double focal = 1.0 / w;
            double scaled_k = k * focal * focal;
            Camera camera("SIMPLE_DIVISION", std::vector<double>{focal, 0.0, 0.0, scaled_k}, -1, -1);
            solutions->emplace_back(ImageTriplet(poses, camera));
        }
//        ThreeViewCameraPose neg_poses = ThreeViewCameraPose(CameraPose(R, -t12), CameraPose(R, -t13));
//        solutions->emplace_back(ImageTriplet(neg_poses, camera));
    }
}

void relpose_3v_trf(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                    const std::vector<Point2D> &x3, double alpha, const RelativePoseOptions &opt,
                    std::vector<ImageTriplet> *solutions) {

    solutions->reserve(8);
    std::vector<double> d(13);
    d[0] = x1[0](0);
    d[1] = x1[0](1);
    d[2] = x2[0](0);
    d[3] = x2[0](1);
    d[4] = x3[0](0);
    d[5] = x3[0](1);

    d[6] = x1[1](0);
    d[7] = x1[1](1);
    d[8] = x2[1](0);
    d[9] = x2[1](1);
    d[10] = x3[1](0);
    d[11] = x3[1](1);

    d[12] = alpha;

    Eigen::VectorXd coeffs(42);
    coeffs[0] = std::pow(d[0],2)*d[2] + std::pow(d[1],2)*d[2] - d[0]*std::pow(d[2],2) - d[0]*std::pow(d[3],2);
    coeffs[1] = -d[1]*d[2] + d[0]*d[3];
    coeffs[2] = d[1]*std::pow(d[2],2) - std::pow(d[0],2)*d[3] - std::pow(d[1],2)*d[3] + d[1]*std::pow(d[3],2);
    coeffs[3] = -d[0] + d[2];
    coeffs[4] = d[1] - d[3];
    coeffs[5] = std::pow(d[6],2)*d[8] + std::pow(d[7],2)*d[8] - d[6]*std::pow(d[8],2) - d[6]*std::pow(d[9],2);
    coeffs[6] = -d[7]*d[8] + d[6]*d[9];
    coeffs[7] = d[7]*std::pow(d[8],2) - std::pow(d[6],2)*d[9] - std::pow(d[7],2)*d[9] + d[7]*std::pow(d[9],2);
    coeffs[8] = -d[6] + d[8];
    coeffs[9] = d[7] - d[9];
    coeffs[10] = d[1]*std::pow(d[4],2) - std::pow(d[0],2)*d[5] - std::pow(d[1],2)*d[5] + d[1]*std::pow(d[5],2);
    coeffs[11] = std::pow(d[0],2)*d[4] + std::pow(d[1],2)*d[4] - d[0]*std::pow(d[4],2) - d[0]*std::pow(d[5],2);
    coeffs[12] = -d[1]*d[4] + d[0]*d[5];
    coeffs[13] = d[1] - d[5];
    coeffs[14] = -d[0] + d[4];
    coeffs[15] = d[7]*std::pow(d[10],2) - std::pow(d[6],2)*d[11] - std::pow(d[7],2)*d[11] + d[7]*std::pow(d[11],2);
    coeffs[16] = std::pow(d[6],2)*d[10] + std::pow(d[7],2)*d[10] - d[6]*std::pow(d[10],2) - d[6]*std::pow(d[11],2);
    coeffs[17] = -d[7]*d[10] + d[6]*d[11];
    coeffs[18] = d[7] - d[11];
    coeffs[19] = -d[6] + d[10];
    coeffs[20] = -std::pow(d[2],2)*d[4] - std::pow(d[3],2)*d[4] + d[2]*std::pow(d[4],2) + d[2]*std::pow(d[5],2);
    coeffs[21] = d[3]*d[4] - d[2]*d[5];
    coeffs[22] = d[3]*std::pow(d[4],2) - std::pow(d[2],2)*d[5] - std::pow(d[3],2)*d[5] + d[3]*std::pow(d[5],2);
    coeffs[23] = std::pow(d[2],2)*d[4] + std::pow(d[3],2)*d[4] - d[2]*std::pow(d[4],2) - d[2]*std::pow(d[5],2);
    coeffs[24] = -d[3]*d[4] + d[2]*d[5];
    coeffs[25] = -d[3]*std::pow(d[4],2) + std::pow(d[2],2)*d[5] + std::pow(d[3],2)*d[5] - d[3]*std::pow(d[5],2);
    coeffs[26] = d[2] - d[4];
    coeffs[27] = d[3] - d[5];
    coeffs[28] = -d[2] + d[4];
    coeffs[29] = -d[3] + d[5];
    coeffs[30] = -std::pow(d[8],2)*d[10] - std::pow(d[9],2)*d[10] + d[8]*std::pow(d[10],2) + d[8]*std::pow(d[11],2);
    coeffs[31] = d[9]*d[10] - d[8]*d[11];
    coeffs[32] = d[9]*std::pow(d[10],2) - std::pow(d[8],2)*d[11] - std::pow(d[9],2)*d[11] + d[9]*std::pow(d[11],2);
    coeffs[33] = std::pow(d[8],2)*d[10] + std::pow(d[9],2)*d[10] - d[8]*std::pow(d[10],2) - d[8]*std::pow(d[11],2);
    coeffs[34] = -d[9]*d[10] + d[8]*d[11];
    coeffs[35] = -d[9]*std::pow(d[10],2) + std::pow(d[8],2)*d[11] + std::pow(d[9],2)*d[11] - d[9]*std::pow(d[11],2);
    coeffs[36] = d[8] - d[10];
    coeffs[37] = d[9] - d[11];
    coeffs[38] = -d[8] + d[10];
    coeffs[39] = -d[9] + d[11];
    coeffs[40] = 1;
    coeffs[41] = -d[12];

    // Setup elimination template
    static const int coeffs0_ind[] = { 1,6,21,31,12,17,24,34,0,5,20,30,10,15,22,32,11,16,23,33,2,7,25,35 };
    static const int coeffs1_ind[] = { 4,9,29,39,3,8,26,36,13,18,27,37,14,19,28,38 };
    static const int C0_ind[] = { 0,2,4,5,7,9,10,11,12,14,16,17,19,21,22,23,25,27,28,29,30,32,34,35 } ;
    static const int C1_ind[] = { 0,2,4,5,6,8,10,11,13,15,16,17,19,21,22,23 };

    Eigen::Matrix<double,6,6> C0; C0.setZero();
    Eigen::Matrix<double,6,4> C1; C1.setZero();
    for (int i = 0; i < 24; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
    for (int i = 0; i < 16; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

//    Eigen::Matrix<double,6,4> C12 = C0.partialPivLu().solve(C1);
    Eigen::Matrix<double,6,4> C12 = C0.householderQr().solve(C1);

    // Setup action matrix
    Eigen::Matrix<double,8, 4> RR;
    RR << -C12.bottomRows(4), Eigen::Matrix<double,4,4>::Identity(4, 4);

    static const int AM_ind[] = { 3,0,1,2 };
    Eigen::Matrix<double, 4, 4> AM;
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

//    Eigen::Matrix<std::complex<double>, 7, 4> sols;
//    sols.setZero();

    // Solve eigenvalue problem
    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4> > es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).array().replicate(4, 1)).eval();

//    sols.row(1) = D.transpose().array();
//    sols.row(2) = V.row(1).array();
//    sols.row(4) = V.row(2).array();
//    sols.row(5) = V.row(3).array();

    for (int i; i < 4; ++i){
        if (std::abs(D(i).imag()) > 1e-8 or
        std::abs(V(1, i).imag()) > 1e-8 or
        std::abs(V(1, i).imag()) > 1e-8 or
        std::abs(V(1, i).imag()) > 1e-8)
            continue;

        double k = D(i).real();
        double t12y = V(1, i).real();
        double t13x = V(2, i).real();
        double t13y = V(3, i).real();

        compute_solutions_3v_TRF(k, t12y, t13x, t13y, x1[0], x2[0], x3[0], alpha, opt, solutions);
    }
}
// Action =  x2
// Quotient ring basis (V) = 1,x3,x5,x6,
// Available monomials (RR*V) = x2*x3,x2*x5,x2*x6,x2,1,x3,x5,x6,


}