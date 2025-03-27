//
// Created by kocur on 14-May-24.
//

#ifndef POSELIB_RELPOSE_K2FK1_9PT_UTILS_H
#define POSELIB_RELPOSE_K2FK1_9PT_UTILS_H

#include <Eigen/Eigen>

namespace poselib {
void relpose_k2Fk1_9pt_prepare_data(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& C,
                                    const Eigen::VectorXd& D, double* data);

Eigen::MatrixXd relpose_k2Fk1_9pt_fundamental_from_sol(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                                                       const Eigen::MatrixXd& C, const Eigen::VectorXd& D,
                                                       const Eigen::VectorXd& ks2, const Eigen::VectorXd& ks1);
}

#endif //POSELIB_RELPOSE_K2FK1_9PT_UTILS_H
