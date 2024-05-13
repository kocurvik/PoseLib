//
// Created by kocur on 11-May-24.
//

#ifndef POSELIB_RELPOSE_KFEFK_UTILS_H
#define POSELIB_RELPOSE_KFEFK_UTILS_H

#include <Eigen/Eigen>

namespace poselib {
    void relpose_kfEfk_prepare_data(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::VectorXd &D,
                                    double *data);

    Eigen::MatrixXd relpose_kfEfk_fundamental_from_sol(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
                                                       const Eigen::VectorXd &D, const Eigen::VectorXd &ks,
                                                       const Eigen::VectorXd &fs1);
}


#endif //POSELIB_RELPOSE_KFEFK_UTILS_H
