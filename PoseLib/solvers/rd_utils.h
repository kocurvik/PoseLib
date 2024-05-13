#ifndef POSELIB_RD_UTILS_H
#define POSELIB_RD_UTILS_H

#include <Eigen/Eigen>
#include <iostream>

namespace poselib {
void normalize_data_eqs(double *data, const std::vector<int> &sz);

void reorderRows(const Eigen::MatrixXd &inMatrix, const std::vector<int> &indices, Eigen::MatrixXd &outMatrix);

void reorderColumns(const Eigen::MatrixXd &inMatrix, const std::vector<int> &indices, Eigen::MatrixXd &outMatrix);

void elimcoeffs_k(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C,
                  const Eigen::MatrixXd &D, int nelim, const std::vector<int> &ordo, Eigen::MatrixXd &Ar,
                  Eigen::MatrixXd &Br, Eigen::MatrixXd &Cr, Eigen::MatrixXd &Dr);

bool lincoeffs_k(const Eigen::MatrixXd &u, const Eigen::MatrixXd &v, int ktype, Eigen::MatrixXd &A, Eigen::MatrixXd &B,
                 Eigen::MatrixXd &C, Eigen::MatrixXd &D);
} // namespace poselib

#endif