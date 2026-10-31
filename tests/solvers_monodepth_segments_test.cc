#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/robust.h>
#include <PoseLib/solvers/relpose_monodepth_segments.h>

using namespace poselib;

namespace test::monodepth_segments {

struct Scene {
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    std::vector<Line3D> S1, S2;
    CameraPose pose;
    double scale;
};

// Two cameras looking at the origin, the depths of view 2 are divided by the scale, so scale * X2 = R * X1 + t
Scene setup_scene(int N, size_t case_index) {
    test_rng::Rng rng = test_rng::make_rng("monodepth_segments_scene", case_index);
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
    const CameraPose p1 = camera(), p2 = camera();
    Scene s;
    s.scale = 0.5 + 2.0 * (test_rng::symmetric_vec3(rng)(0) + 1.0);
    const Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    s.pose = CameraPose(R, p2.t - R * p1.t);
    for (int i = 0; i < N;) {
        const Eigen::Vector3d X = test_rng::symmetric_vec3(rng), Y = test_rng::symmetric_vec3(rng);
        const Eigen::Vector3d Z1 = p1.apply(X), Z2 = p2.apply(X), W1 = p1.apply(Y), W2 = p2.apply(Y);
        if (std::min({Z1(2), Z2(2), W1(2), W2(2)}) <= 0.1)
            continue;
        s.x1.push_back(Z1.hnormalized());
        s.x2.push_back(Z2.hnormalized());
        s.d1.push_back(Z1(2));
        s.d2.push_back(Z2(2) / s.scale);
        s.S1.emplace_back(Z1, W1);
        s.S2.emplace_back(Z2 / s.scale, W2 / s.scale);
        ++i;
    }
    return s;
}

bool close_to(const MonoDepthTwoViewGeometry &g, const Scene &s, double tol) {
    return (g.pose.R() - s.pose.R()).norm() < tol && (g.pose.t - s.pose.t).norm() < tol &&
           std::abs(g.scale - s.scale) < tol;
}

bool test_1s1p_solver() {
    for (size_t case_index = 0; case_index < 200; ++case_index) {
        const Scene s = setup_scene(2, case_index);
        std::vector<MonoDepthTwoViewGeometry> models;
        relpose_monodepth_1s1p(s.S1[0], s.S2[0], s.x1[1].homogeneous(), s.x2[1].homogeneous(), &models);
        REQUIRE(models.size() >= 1);
        bool found = false;
        for (const MonoDepthTwoViewGeometry &g : models)
            found |= close_to(g, s, 1e-6);
        REQUIRE(found);
        // every solution satisfies the epipolar constraint of the point and the two 3D-3D matches
        for (const MonoDepthTwoViewGeometry &g : models) {
            const Eigen::Matrix3d R = g.pose.R();
            const Eigen::Vector3d x1 = s.x1[1].homogeneous(), x2 = s.x2[1].homogeneous();
            REQUIRE(std::abs(x2.dot(g.pose.t.cross(R * x1))) < 1e-8);
            REQUIRE((g.scale * s.S2[0].X1 - R * s.S1[0].X1 - g.pose.t).norm() < 1e-8);
            REQUIRE((g.scale * s.S2[0].X2 - R * s.S1[0].X2 - g.pose.t).norm() < 1e-8);
        }
    }
    return true;
}

bool test_2s_solver() {
    for (size_t case_index = 0; case_index < 200; ++case_index) {
        const Scene s = setup_scene(2, case_index);
        std::vector<MonoDepthTwoViewGeometry> models;
        relpose_monodepth_2s(s.S1[0], s.S1[1], s.S2[0], s.S2[1], &models);
        REQUIRE(models.size() == 1);
        REQUIRE(close_to(models[0], s, 1e-8));
    }
    return true;
}

bool test_2s_collinear() {
    const Line3D a(Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 1)), b(Eigen::Vector3d(2, 0, 1), Eigen::Vector3d(3, 0, 1));
    std::vector<MonoDepthTwoViewGeometry> models;
    REQUIRE(relpose_monodepth_2s(a, b, a, b, &models) == 0);
    return true;
}

bool test_segment_estimators() {
    for (int solver : {0, 1}) {
        for (size_t case_index = 0; case_index < 20; ++case_index) {
            const Scene s = setup_scene(100, case_index);
            MonoDepthRelativePoseOptions opt;
            opt.segment_solver = solver;
            opt.ransac.min_iterations = 50;
            opt.ransac.max_iterations = 50;
            opt.max_errors = {0.01, 0.001};
            const Camera camera("SIMPLE_PINHOLE", {1.0, 0.0, 0.0}, -1, -1);
            std::vector<Line2D> l1, l2;
            std::vector<Eigen::Vector2d> depths1, depths2;
            for (size_t k = 0; k < s.S1.size(); ++k) {
                l1.emplace_back(s.S1[k].X1.hnormalized(), s.S1[k].X2.hnormalized());
                l2.emplace_back(s.S2[k].X1.hnormalized(), s.S2[k].X2.hnormalized());
                depths1.emplace_back(s.S1[k].X1(2), s.S1[k].X2(2));
                depths2.emplace_back(s.S2[k].X1(2), s.S2[k].X2(2));
            }
            MonoDepthTwoViewGeometry geometry;
            std::vector<char> inliers;
            const RansacStats stats = estimate_monodepth_relative_pose_segments(
                s.x1, s.x2, s.d1, s.d2, l1, l2, depths1, depths2, camera, camera, opt, &geometry, &inliers);
            REQUIRE(stats.num_inliers == 100);
            REQUIRE(close_to(geometry, s, 1e-5));
        }
    }
    return true;
}

} // namespace test::monodepth_segments

using namespace test::monodepth_segments;

std::vector<Test> register_solvers_monodepth_segments_test() {
    return {TEST(test_1s1p_solver), TEST(test_2s_solver), TEST(test_2s_collinear), TEST(test_segment_estimators)};
}
