#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/robust/optim/hybrid.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/monodepth_relpose.h>
#include <PoseLib/robust/optim/relative.h>
#include <PoseLib/robust/optim/twoview_lines.h>

using namespace poselib;

namespace test::twoview_lines {

// Two cameras looking at the origin, points and line segments with depth in both, and a perturbed relative pose (so
// that the residuals are not zero)
void setup_scene(int N, CameraPose &pose, std::vector<Point2D> &x1, std::vector<Point2D> &x2, std::vector<double> &d1,
                 std::vector<double> &d2, std::vector<TwoViewLine> &lines, size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng("twoview_lines_scene", case_index);
    auto camera = [&]() {
        Eigen::Vector3d cc = 2.0 * test_rng::symmetric_vec3(rng).normalized();
        Eigen::Vector3d r3 = (test_rng::symmetric_vec3(rng, 0.1) - cc).normalized();
        Eigen::Vector3d r2 = r3.cross(Eigen::Vector3d::UnitX()).normalized();
        Eigen::Matrix3d R;
        R.row(0) = r2.cross(r3);
        R.row(1) = r2;
        R.row(2) = r3;
        return CameraPose(R, -R * cc);
    };
    CameraPose p1 = camera(), p2 = camera();
    std::vector<Line3D> L1, L2;
    for (int i = 0; i < N;) {
        Eigen::Vector3d X = test_rng::symmetric_vec3(rng), Y = test_rng::symmetric_vec3(rng);
        Eigen::Vector3d Z1 = p1.apply(X), Z2 = p2.apply(X), W1 = p1.apply(Y), W2 = p2.apply(Y);
        if (Z1(2) <= 0.1 || Z2(2) <= 0.1 || W1(2) <= 0.1 || W2(2) <= 0.1)
            continue;
        x1.push_back(Z1.hnormalized());
        x2.push_back(Z2.hnormalized());
        d1.push_back(Z1(2));
        d2.push_back(Z2(2));
        L1.emplace_back(Z1, W1);
        L2.emplace_back(Z2, W2);
        ++i;
    }
    lines = make_twoview_lines(L1, L2);
    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - R * p1.t;
    pose = CameraPose(R, t);
    pose.q = quat_step_post(pose.q, Eigen::Vector3d(0.01, -0.02, 0.015));
    pose.t += Eigen::Vector3d(0.01, 0.02, -0.01);
}

template <typename Refiner, typename Model> bool check_refiner(Refiner &refiner, const Model &model) {
    double jac_err = verify_jacobian<Refiner, Model>(refiner, model, 1e-6);
    REQUIRE_SMALL(jac_err, 1e-6)
    // compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, model);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, model);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i)
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);
    return true;
}

bool test_twoview_line_relpose_jacobian() {
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    std::vector<TwoViewLine> lines;
    setup_scene(10, pose, x1, x2, d1, d2, lines);
    for (int line_residual = 0; line_residual < 4; ++line_residual) {
        PinholeRelativePoseRefiner<UniformWeightVector, TestAccumulator> pts_refiner(x1, x2);
        TwoViewLineRefiner<CameraPose, 5, TestAccumulator> line_refiner(lines, line_residual, 0.7, 3.0, &pts_refiner);
        // the step of the point refiner uses the tangent basis set in its compute_jacobian
        TestAccumulator dummy;
        pts_refiner.compute_jacobian(dummy, pose);
        if (!check_refiner(line_refiner, pose))
            return false;
        HybridRefiner<CameraPose, TestAccumulator> refiner;
        refiner.register_refiner(&pts_refiner);
        refiner.register_refiner(&line_refiner);
        if (!check_refiner(refiner, pose))
            return false;
    }
    return true;
}

template <int NumParams> bool check_monodepth(bool refine_shift) {
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    std::vector<TwoViewLine> lines;
    setup_scene(10, pose, x1, x2, d1, d2, lines, 1);
    MonoDepthTwoViewGeometry geometry(pose, 1.1);
    if (refine_shift) {
        geometry.shift1 = 0.05;
        geometry.shift2 = -0.03;
    }
    for (int line_residual = 0; line_residual < 4; ++line_residual) {
        MonoDepthRelPoseRefiner<UniformWeightVector, TestAccumulator> pts_refiner(x1, x2, d1, d2, 0.5, 1.0,
                                                                                  refine_shift);
        TwoViewLineRefiner<MonoDepthTwoViewGeometry, NumParams, TestAccumulator> line_refiner(lines, line_residual, 0.7,
                                                                                              3.0, &pts_refiner);
        if (!check_refiner(line_refiner, geometry))
            return false;
        HybridRefiner<MonoDepthTwoViewGeometry, TestAccumulator> refiner;
        refiner.register_refiner(&pts_refiner);
        refiner.register_refiner(&line_refiner);
        if (!check_refiner(refiner, geometry))
            return false;
    }
    return true;
}

bool test_twoview_line_monodepth_jacobian() { return check_monodepth<7>(false); }
bool test_twoview_line_monodepth_shift_jacobian() { return check_monodepth<9>(true); }

} // namespace test::twoview_lines

using namespace test::twoview_lines;
std::vector<Test> register_optim_twoview_lines_test() {
    return {TEST(test_twoview_line_relpose_jacobian), TEST(test_twoview_line_monodepth_jacobian),
            TEST(test_twoview_line_monodepth_shift_jacobian)};
}
