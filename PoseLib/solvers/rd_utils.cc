#include <iostream>
#include <Eigen/Eigen>

namespace poselib {
    void normalize_data_eqs(
            double *data,
            const std::vector<int> &sz) {
        int count = 0;
        for (size_t i = 0; i < sz.size(); ++i) {
            double max_val = 0.0;
            // Find max absolute value in the current segment
            for (int j = 0; j < sz[i]; ++j) {
                double abs_val = std::abs(data[count + j]);
                if (abs_val > max_val) {
                    max_val = abs_val;
                }
            }
            // Normalize the current segment
            for (int j = 0; j < sz[i]; ++j) {
                data[count + j] /= max_val;
            }
            count += sz[i];
        }
    }

    void reorderRows(
            const Eigen::MatrixXd &inMatrix,
            const std::vector<int> &indices,
            Eigen::MatrixXd &outMatrix) {
        outMatrix.resize(indices.size(), inMatrix.cols());
        for (size_t i = 0; i < indices.size(); ++i) {
            outMatrix.row(i) = inMatrix.row(indices[i]);
        }
    }

    void reorderColumns(
            const Eigen::MatrixXd &inMatrix,
            const std::vector<int> &indices,
            Eigen::MatrixXd &outMatrix) {
        outMatrix.resize(inMatrix.rows(), indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            outMatrix.col(i) = inMatrix.col(indices[i]);
        }
    }


    void elimcoeffs_k(
            const Eigen::MatrixXd &A,
            const Eigen::MatrixXd &B,
            const Eigen::MatrixXd &C,
            const Eigen::MatrixXd &D,
            int nelim,
            const std::vector<int> &ordo,
            Eigen::MatrixXd &Ar,
            Eigen::MatrixXd &Br,
            Eigen::MatrixXd &Cr,
            Eigen::MatrixXd &Dr) {
        int np = A.rows();

        Eigen::MatrixXd A_reordered, B_reordered, C_reordered, D_reordered;
        reorderColumns(A, ordo, A_reordered);
        reorderColumns(B, ordo, B_reordered);
        reorderColumns(C, ordo, C_reordered);
        reorderColumns(D, ordo, D_reordered);

        Eigen::MatrixXd AA = A_reordered.leftCols(nelim);
        Eigen::MatrixXd AAs = AA.topRows(nelim);
        Eigen::MatrixXd A_copy = A_reordered.rightCols(A_reordered.cols() - nelim);
        Eigen::MatrixXd B_copy = B_reordered.rightCols(B_reordered.cols() - nelim);
        Eigen::MatrixXd C_copy = C_reordered.rightCols(C_reordered.cols() - nelim);
        Eigen::MatrixXd D_copy = D_reordered.rightCols(D_reordered.cols() - nelim);

        Eigen::MatrixXd inv_AAs = AAs.inverse();

        Ar = inv_AAs * A_copy.topRows(nelim);
        Br = inv_AAs * B_copy.topRows(nelim);
        Cr = inv_AAs * C_copy.topRows(nelim);
        Dr = inv_AAs * D_copy.topRows(nelim);

        for (int i = nelim; i < np; ++i) {
            Ar.conservativeResize(Ar.rows() + 1, Ar.cols());
            Br.conservativeResize(Br.rows() + 1, Br.cols());
            Cr.conservativeResize(Cr.rows() + 1, Cr.cols());
            Dr.conservativeResize(Dr.rows() + 1, Dr.cols());

            Ar.row(i) = A_copy.row(i) - AA.row(i).head(nelim) * Ar.topRows(nelim);
            Br.row(i) = B_copy.row(i) - AA.row(i).head(nelim) * Br.topRows(nelim);
            Cr.row(i) = C_copy.row(i) - AA.row(i).head(nelim) * Cr.topRows(nelim);
            Dr.row(i) = D_copy.row(i) - AA.row(i).head(nelim) * Dr.topRows(nelim);
            /*for (int j = 0; j < nelim; ++j) {
                Ar.row(i) -= AA.row(i) * Ar.topRows(nelim).col(j) * A_copy.col(j);
                Br.row(i) -= AA.row(i) * Br.topRows(nelim).col(j) * B_copy.col(j);
                Cr.row(i) -= AA.row(i) * Cr.topRows(nelim).col(j) * C_copy.col(j);
                Dr.row(i) -= AA.row(i) * Dr.topRows(nelim).col(j) * D_copy.col(j);
            }*/
        }
    }

    bool lincoeffs_k(
            const Eigen::MatrixXd &u,
            const Eigen::MatrixXd &v,
            int ktype,
            Eigen::MatrixXd &A,
            Eigen::MatrixXd &B,
            Eigen::MatrixXd &C,
            Eigen::MatrixXd &D) {
        int np = u.cols();
        A.resize(np, 9);
        B.resize(np, 9);
        C.resize(np, 9);
        D.resize(np, 9);

        for (int i = 0; i < np; ++i) {
            double u1 = u(0, i);
            double u2 = u(1, i);
            double v1 = v(0, i);
            double v2 = v(1, i);
            switch (ktype) {
                case 1: // kF
                    A.row(i) << u1 * v1, u1 * v2, u1, u2 * v1, u2 * v2, u2, v1, v2, 1;
                    B.row(i) << 0, 0, u1 * (v1 * v1 + v2 * v2), 0, 0, u2 * (v1 * v1 + v2 * v2), 0, 0, v1 * v1 + v2 * v2;
                    break;
                case 2: // kFk
                    A.row(i) << u1 * v1, u1 * v2, u1, u2 * v1, u2 * v2, u2, v1, v2, 1;
                    B.row(i) << 0, 0, u1 * (v1 * v1 + v2 * v2), 0, 0, u2 * (v1 * v1 + v2 * v2), v1 *
                                                                                                (u1 * u1 + u2 * u2),
                            v2 * (u1 * u1 + u2 * u2), u1 * u1 + u2 * u2 + v1 * v1 + v2 * v2;
                    D.row(i) << 0, 0, 0, 0, 0, 0, 0, 0, (u1 * u1 + u2 * u2) * (v1 * v1 + v2 * v2);
                    break;
                case 3: // k1Fk2
                    A.row(i) << u1 * v1, u1 * v2, u1, u2 * v1, u2 * v2, u2, v1, v2, 1;
                    B.row(i) << 0, 0, u1 * (v1 * v1 + v2 * v2), 0, 0, u2 * (v1 * v1 + v2 * v2), 0, 0, v1 * v1 + v2 * v2;
                    C.row(i) << 0, 0, 0, 0, 0, 0, v1 * (u1 * u1 + u2 * u2), v2 * (u1 * u1 + u2 * u2), u1 * u1 + u2 * u2;
                    D.row(i) << 0, 0, 0, 0, 0, 0, 0, 0, (u1 * u1 + u2 * u2) * (v1 * v1 + v2 * v2);
                    break;
                default:
                    std::cerr << "Invalid ktype!" << std::endl;
                    return false;
            }
        }
        return true;
    }
}