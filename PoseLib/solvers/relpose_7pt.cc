#include "relpose_7pt.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/misc/univariate.h"

#include <Eigen/Dense>

namespace poselib {

int relpose_7pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                std::vector<Eigen::Matrix3d> *fundamental_matrices) {

    // Compute nullspace to epipolar constraints
    Eigen::Matrix<double, 9, 7> epipolar_constraints;
    for (size_t i = 0; i < 7; ++i) {
        epipolar_constraints.col(i) << x1[i](0) * x2[i], x1[i](1) * x2[i], x1[i](2) * x2[i];
    }
    Eigen::Matrix<double, 9, 9> Q = epipolar_constraints.fullPivHouseholderQr().matrixQ();
    Eigen::Matrix<double, 9, 2> N = Q.rightCols(2);

    // coefficients for det(F(x)) = 0
    const double c3 = N(0, 0) * N(4, 0) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 0) - N(1, 0) * N(3, 0) * N(8, 0) +
                      N(1, 0) * N(5, 0) * N(6, 0) + N(2, 0) * N(3, 0) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 0);
    const double c2 = N(0, 0) * N(4, 0) * N(8, 1) + N(0, 0) * N(4, 1) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 1) -
                      N(0, 0) * N(5, 1) * N(7, 0) + N(0, 1) * N(4, 0) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 0) -
                      N(1, 0) * N(3, 0) * N(8, 1) - N(1, 0) * N(3, 1) * N(8, 0) + N(1, 0) * N(5, 0) * N(6, 1) +
                      N(1, 0) * N(5, 1) * N(6, 0) - N(1, 1) * N(3, 0) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 0) +
                      N(2, 0) * N(3, 0) * N(7, 1) + N(2, 0) * N(3, 1) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 1) -
                      N(2, 0) * N(4, 1) * N(6, 0) + N(2, 1) * N(3, 0) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 0);
    const double c1 = N(0, 0) * N(4, 1) * N(8, 1) - N(0, 0) * N(5, 1) * N(7, 1) + N(0, 1) * N(4, 0) * N(8, 1) +
                      N(0, 1) * N(4, 1) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 1) - N(0, 1) * N(5, 1) * N(7, 0) -
                      N(1, 0) * N(3, 1) * N(8, 1) + N(1, 0) * N(5, 1) * N(6, 1) - N(1, 1) * N(3, 0) * N(8, 1) -
                      N(1, 1) * N(3, 1) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 1) + N(1, 1) * N(5, 1) * N(6, 0) +
                      N(2, 0) * N(3, 1) * N(7, 1) - N(2, 0) * N(4, 1) * N(6, 1) + N(2, 1) * N(3, 0) * N(7, 1) +
                      N(2, 1) * N(3, 1) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 1) - N(2, 1) * N(4, 1) * N(6, 0);
    const double c0 = N(0, 1) * N(4, 1) * N(8, 1) - N(0, 1) * N(5, 1) * N(7, 1) - N(1, 1) * N(3, 1) * N(8, 1) +
                      N(1, 1) * N(5, 1) * N(6, 1) + N(2, 1) * N(3, 1) * N(7, 1) - N(2, 1) * N(4, 1) * N(6, 1);

    // Solve the cubic
    double inv_c3 = 1.0 / c3;
    double roots[3];
    int n_roots = univariate::solve_cubic_real(c2 * inv_c3, c1 * inv_c3, c0 * inv_c3, roots);

    // Reshape back into 3x3 matrices
    fundamental_matrices->clear();
    fundamental_matrices->reserve(n_roots);
    for (int i = 0; i < n_roots; ++i) {
        Eigen::Matrix<double, 9, 1> f = N.col(0) * roots[i] + N.col(1);
        f.normalize();
        fundamental_matrices->push_back(Eigen::Map<Eigen::Matrix3d>(f.data()));
    }

    return n_roots;
}

void varying_focal_relpose_from_F(const std::vector<Eigen::Matrix3d> &Fs,
                                  std::vector<Point3D> &x1h, std::vector<Point3D> &x2h,
                                  ImagePairVector *models) {
    for (const Eigen::Matrix3d &F : Fs) {
        std::pair<Camera, Camera> cameras =
            focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
        Camera camera1 = cameras.first;
        Camera camera2 = cameras.second;

        const double focal1 = camera1.focal();
        const double focal2 = camera2.focal();

        if (std::isnan(focal1))
            continue;
        if (std::isnan(focal2))
            continue;

        Eigen::DiagonalMatrix<double, 3> K1(focal1, focal1, 1.0);
        Eigen::DiagonalMatrix<double, 3> K2(focal2, focal2, 1.0);

        Eigen::Matrix3d E = K2 * F * K1;

        std::vector<CameraPose> poses;
        motion_from_essential(E, x1h, x2h, &poses);

        for (const CameraPose &pose : poses) {
            models->emplace_back(pose, camera1, camera2);
        }
    }
}

void varying_focal_relpose_from_projective_pair(const std::vector<ProjectiveImagePair> &proj_pairs,
                                                std::vector<Point3D> &x1s, std::vector<Point3D> &x2s,
                                                ImagePairVector *models) {
    for (const ProjectiveImagePair &proj_pair : proj_pairs) {

        std::pair<Camera, Camera> cameras =
            focals_from_fundamental(proj_pair.F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());

        const double focal1 = cameras.first.focal();
        const double focal2 = cameras.second.focal();

        if (std::isnan(focal1))
            continue;
        if (std::isnan(focal2))
            continue;

        Camera camera1 = Camera("SIMPLE_DIVISION", {focal1, 0, 0, proj_pair.camera1.params[3]}, -1, -1);
        Camera camera2 = Camera("SIMPLE_DIVISION", {focal2, 0, 0, proj_pair.camera2.params[3]}, -1, -1);

        Eigen::DiagonalMatrix<double, 3> K1(focal1, focal1, 1.0);
        Eigen::DiagonalMatrix<double, 3> K2(focal2, focal2, 1.0);

        Eigen::Matrix3d E = K2 * proj_pair.F * K1;

        std::vector<CameraPose> poses;

        std::vector<Point3D> x1h(x1s.size());
        std::vector<Point3D> x2h(x1s.size());

        for (size_t i = 0; i < x1s.size(); ++i){
            camera1.unproject(x1s[i].hnormalized(), &x1h[i]);
            camera2.unproject(x2s[i].hnormalized(), &x2h[i]);
        }

        motion_from_essential(E, x1h, x2h, &poses);

        for (const CameraPose &pose : poses) {
            models->emplace_back(pose, camera1, camera2);
        }
    }
}

} // namespace poselib