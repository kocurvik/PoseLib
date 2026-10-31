#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/robust.h>
#include <PoseLib/robust/estimators/threeview_relpose.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/threeview.h>
#include <PoseLib/solvers/threeview_2sift.h>

using namespace poselib;

namespace test::threeview {

struct Scene {
    std::vector<Point2D> x1, x2, x3;
    std::vector<double> d1, d2, d3;
    std::vector<Line3D> L1, L2, L3;
    ThreeViewGeometry gt; // the depths are metric, so scale2 = scale3 = 1 before scale_depths
};

// Three cameras looking at the origin, points and line segments with depth in all of them
Scene setup_scene(int N, size_t case_index) {
    test_rng::Rng rng = test_rng::make_rng("threeview_scene", case_index);
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
    const CameraPose p1 = camera(), p2 = camera(), p3 = camera();
    Scene s;
    for (int i = 0; i < N;) {
        const Eigen::Vector3d X = test_rng::symmetric_vec3(rng), Y = test_rng::symmetric_vec3(rng);
        const Eigen::Vector3d Z1 = p1.apply(X), Z2 = p2.apply(X), Z3 = p3.apply(X);
        const Eigen::Vector3d W1 = p1.apply(Y), W2 = p2.apply(Y), W3 = p3.apply(Y);
        if (std::min({Z1(2), Z2(2), Z3(2), W1(2), W2(2), W3(2)}) <= 0.1)
            continue;
        s.x1.push_back(Z1.hnormalized());
        s.x2.push_back(Z2.hnormalized());
        s.x3.push_back(Z3.hnormalized());
        s.d1.push_back(Z1(2));
        s.d2.push_back(Z2(2));
        s.d3.push_back(Z3(2));
        s.L1.emplace_back(Z1, W1);
        s.L2.emplace_back(Z2, W2);
        s.L3.emplace_back(Z3, W3);
        ++i;
    }
    const auto relative = [](const CameraPose &a, const CameraPose &b) {
        const Eigen::Matrix3d R = b.R() * a.R().transpose();
        return CameraPose(R, Eigen::Vector3d(b.t - R * a.t));
    };
    s.gt = ThreeViewGeometry(relative(p1, p2), relative(p1, p3));
    return s;
}

// A model away from the optimum, so that the residuals are not zero
ThreeViewGeometry perturbed(const ThreeViewGeometry &gt) {
    ThreeViewGeometry m = gt;
    m.pose12.q = quat_step_post(m.pose12.q, Eigen::Vector3d(0.01, -0.02, 0.015));
    m.pose12.t += Eigen::Vector3d(0.01, 0.02, -0.01);
    m.pose13.q = quat_step_post(m.pose13.q, Eigen::Vector3d(-0.02, 0.01, 0.01));
    m.pose13.t += Eigen::Vector3d(-0.02, 0.01, 0.015);
    m.scale2 = 1.1;
    m.scale3 = 0.9;
    return m;
}

// All combinations of the terms: Sampson, reprojection and each kind of line residual
bool test_threeview_jacobian() {
    const Scene s = setup_scene(10, 0);
    const ThreeViewLines lines = make_threeview_lines(s.L1, s.L2, s.L3), no_lines;
    const ThreeViewGeometry model = perturbed(s.gt);
    for (double weight_sampson : {0.0, 0.8}) {
        for (double scale_reproj : {0.0, 0.5}) {
            for (int line_residual = -1; line_residual < 4; ++line_residual) {
                if (weight_sampson == 0.0 && scale_reproj == 0.0 && line_residual < 0)
                    continue;
                ThreeViewRefiner<TestAccumulator> refiner(s.x1, s.x2, s.x3, s.d1, s.d2, s.d3,
                                                          line_residual < 0 ? no_lines : lines, std::max(line_residual, 0),
                                                          scale_reproj, weight_sampson, 0.7, 3.0);
                // the step uses the tangent basis set in compute_jacobian
                TestAccumulator dummy;
                dummy.reset_jacobian();
                refiner.compute_jacobian(dummy, model);
                const double jac_err = verify_jacobian<decltype(refiner), ThreeViewGeometry>(refiner, model, 1e-6);
                // the tangent basis of t12 changes with the finite difference steps of t12
                REQUIRE_SMALL(jac_err, 1e-5)
                // compute_residual and compute_jacobian are compatible
                TestAccumulator acc;
                acc.reset_residual();
                const double r1 = refiner.compute_residual(acc, model);
                acc.reset_jacobian();
                refiner.compute_jacobian(acc, model);
                double r2 = 0.0;
                for (size_t i = 0; i < acc.rs.size(); ++i)
                    r2 += acc.weights[i] * acc.rs[i].squaredNorm();
                REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);
            }
        }
    }
    return true;
}

double rotation_error(const CameraPose &a, const CameraPose &b) { return (a.R().transpose() * b.R() - Eigen::Matrix3d::Identity()).norm(); }

double direction_error(const CameraPose &a, const CameraPose &b) { return (a.t.normalized() - b.t.normalized()).norm(); }

// The estimated poses up to the common scale of the translations
bool close_to_gt(const ThreeViewGeometry &m, const ThreeViewGeometry &gt, double tol) {
    return rotation_error(m.pose12, gt.pose12) < tol && rotation_error(m.pose13, gt.pose13) < tol &&
           direction_error(m.pose12, gt.pose12) < tol && direction_error(m.pose13, gt.pose13) < tol &&
           direction_error(m.pose23(), gt.pose23()) < tol;
}

// Noise free data with outliers: both solvers and all refinement terms find the poses
bool test_threeview_estimator() {
    Scene s = setup_scene(60, 1);
    // unknown scales of the depths of views 2 and 3
    const double scale2 = 2.0, scale3 = 0.5;
    for (double &d : s.d2)
        d /= scale2;
    for (double &d : s.d3)
        d /= scale3;
    test_rng::Rng rng = test_rng::make_rng("threeview_outliers", 0);
    for (int k = 0; k < 15; ++k)
        s.x3[k] += test_rng::symmetric_vec3(rng).head<2>();

    for (int solver : {0, 1}) {
        for (double weight_reproj : {0.0, 1.0}) {
            for (bool line_refine : {false, true}) {
                ThreeViewOptions opt;
                opt.threeview_solver = solver;
                opt.weight_reproj = weight_reproj;
                opt.line_refine = line_refine;
                opt.line_residual = 1;
                opt.max_errors = {0.08, 0.01};
                opt.ransac.min_iterations = opt.ransac.max_iterations = 200;
                ThreeViewGeometry model;
                std::vector<char> inliers;
                const auto lines2D = [&](const std::vector<Line3D> &L, std::vector<Line2D> *l,
                                         std::vector<Eigen::Vector2d> *ld) {
                    for (const Line3D &line : L) {
                        if (!line_refine)
                            break;
                        l->emplace_back(line.X1.hnormalized(), line.X2.hnormalized());
                        ld->emplace_back(line.X1(2), line.X2(2));
                    }
                };
                std::vector<Line2D> l1, l2, l3;
                std::vector<Eigen::Vector2d> ld1, ld2, ld3;
                lines2D(s.L1, &l1, &ld1);
                lines2D(s.L2, &l2, &ld2);
                lines2D(s.L3, &l3, &ld3);
                const Camera camera("PINHOLE", {1.0, 1.0, 0.0, 0.0}, -1, -1);
                RansacStats stats = estimate_threeview_relative_pose(s.x1, s.x2, s.x3, s.d1, s.d2, s.d3, l1, l2, l3,
                                                                     ld1, ld2, ld3, ThreeViewFeatures(), camera, camera,
                                                                     camera, opt, &model, &inliers);
                REQUIRE_EQ(stats.num_inliers, static_cast<size_t>(45));
                REQUIRE(close_to_gt(model, s.gt, 1e-5));
                // the outliers are not inliers
                for (int k = 0; k < 15; ++k)
                    REQUIRE(!inliers[k]);
                // the depths fix the scales when they are used
                if (solver == 1 || weight_reproj > 0.0) {
                    REQUIRE_SMALL(std::abs(model.scale2 - scale2), 1e-5);
                    REQUIRE_SMALL(std::abs(model.scale3 - scale3), 1e-5);
                    REQUIRE_SMALL((model.pose12.t - s.gt.pose12.t).norm(), 1e-5);
                }
            }
        }
    }
    return true;
}

// LM converges back to the optimum from a perturbed model for every combination of the terms
bool test_threeview_refine() {
    const Scene s = setup_scene(30, 2);
    const ThreeViewLines lines = make_threeview_lines(s.L1, s.L2, s.L3), no_lines;
    for (double weight_sampson : {0.0, 1.0}) {
        for (double scale_reproj : {0.0, 1.0}) {
            for (bool use_lines : {false, true}) {
                if (weight_sampson == 0.0 && scale_reproj == 0.0)
                    continue;
                ThreeViewGeometry model = perturbed(s.gt);
                model.scale2 = model.scale3 = 1.0;
                if (scale_reproj == 0.0) {
                    // the scale is free without the depths, keep the perturbation but use the correct t12 norm
                    model.pose12.t = model.pose12.t.normalized() * s.gt.pose12.t.norm();
                } else {
                    model.scale2 = 1.05;
                    model.scale3 = 0.95;
                }
                BundleOptions opt;
                opt.loss_type = BundleOptions::LossType::TRIVIAL;
                opt.max_iterations = 200;
                refine_threeview(s.x1, s.x2, s.x3, s.d1, s.d2, s.d3, use_lines ? lines : no_lines, 2, &model,
                                 scale_reproj, weight_sampson, opt, 0.5, 1.0);
                REQUIRE(close_to_gt(model, s.gt, 1e-5));
            }
        }
    }
    return true;
}

// Empty models instead of a crash when the depths are missing
bool test_threeview_missing_depth() {
    const Scene s = setup_scene(10, 3);
    ThreeViewOptions opt;
    opt.threeview_solver = 1;
    ThreeViewGeometry model;
    std::vector<char> inliers;
    const std::vector<double> none;
    const std::vector<Line3D> empty;
    RansacStats stats = ransac_threeview(s.x1, s.x2, s.x3, none, none, none, empty, empty, empty, ThreeViewFeatures(),
                                         opt, &model, &inliers);
    REQUIRE_EQ(stats.num_inliers, static_cast<size_t>(0));
    REQUIRE_EQ(inliers.size(), s.x1.size());
    // the same without features for 2-SIFT
    opt.threeview_solver = 2;
    stats = ransac_threeview(s.x1, s.x2, s.x3, none, none, none, empty, empty, empty, ThreeViewFeatures(), opt, &model,
                             &inliers);
    REQUIRE_EQ(stats.num_inliers, static_cast<size_t>(0));
    return true;
}

// Is one of the models the ground truth, with the translations in the units of the given depth of point 0 in view 1
bool has_solution(const std::vector<ThreeViewGeometry> &models, const ThreeViewGeometry &gt, double d0, double tol) {
    for (const ThreeViewGeometry &m : models) {
        if (rotation_error(m.pose12, gt.pose12) < tol && rotation_error(m.pose13, gt.pose13) < tol &&
            (m.pose12.t - gt.pose12.t / d0).norm() < tol && (m.pose13.t - gt.pose13.t / d0).norm() < tol)
            return true;
    }
    return false;
}

// General motion with the true depth ratios (the scales of real SIFT features), lines through the points
bool test_threeview_2sift_solver() {
    int failures = 0;
    const int trials = 200;
    for (int trial = 0; trial < trials; ++trial) {
        const Scene s = setup_scene(2, 100 + trial);
        std::vector<Eigen::Vector3d> x[3], l[3];
        const std::vector<Line3D> *L[3] = {&s.L1, &s.L2, &s.L3};
        const std::vector<Point2D> *pts[3] = {&s.x1, &s.x2, &s.x3};
        for (int v = 0; v < 3; ++v) {
            for (int i = 0; i < 2; ++i) {
                x[v].push_back((*pts[v])[i].homogeneous());
                // the line through the point and the other end of the 3D segment (X1 is the point)
                l[v].push_back((*L[v])[i].X1.cross((*L[v])[i].X2));
            }
        }
        std::vector<ThreeViewGeometry> models;
        threeview_2sift(x[0], x[1], x[2], l[0], l[1], l[2], s.d2[0] / s.d1[0], s.d2[1] / s.d1[1], s.d3[0] / s.d1[0],
                        &models);
        failures += !has_solution(models, s.gt, s.d1[0], 1e-6);
    }
    // rare near degenerate samples
    REQUIRE(failures <= trials / 50);
    return true;
}

// Motion that keeps the depths (rotation about z and translation in the image plane), so the assumption of the same
// scale in all views holds exactly: the estimator finds the poses from the features, with outliers among the features
bool test_threeview_2sift_estimator() {
    test_rng::Rng rng = test_rng::make_rng("threeview_2sift", 0);
    const auto planar = [](double angle, const Eigen::Vector2d &t) {
        const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        return CameraPose(R, Eigen::Vector3d(t(0), t(1), 0.0));
    };
    const ThreeViewGeometry gt(planar(0.3, {0.5, -0.2}), planar(-0.4, {-0.3, 0.6}));
    std::vector<Point2D> x[3];
    ThreeViewFeatures features;
    const Camera camera("PINHOLE", {1.0, 1.0, 0.0, 0.0}, -1, -1);
    for (int k = 0; k < 60; ++k) {
        const Eigen::Vector3d X = test_rng::symmetric_vec3(rng) + Eigen::Vector3d(0.0, 0.0, 4.0);
        const Eigen::Vector3d Y = X + 0.3 * test_rng::symmetric_vec3(rng);
        const Eigen::Vector3d P[3] = {X, gt.pose12.apply(X), gt.pose13.apply(X)};
        const Eigen::Vector3d Q[3] = {Y, gt.pose12.apply(Y), gt.pose13.apply(Y)};
        for (int v = 0; v < 3; ++v) {
            x[v].push_back(P[v].hnormalized());
            if (k < 20) {
                features.x[v].push_back(P[v].hnormalized());
                features.l[v].emplace_back(P[v].hnormalized(), Q[v].hnormalized());
            }
        }
    }
    for (int k = 0; k < 5; ++k)
        features.x[2][k] += test_rng::symmetric_vec3(rng).head<2>();

    ThreeViewOptions opt;
    opt.threeview_solver = 2;
    opt.max_errors = {0.08, 0.01};
    opt.ransac.min_iterations = opt.ransac.max_iterations = 200;
    ThreeViewGeometry model;
    std::vector<char> inliers;
    const std::vector<double> none;
    const std::vector<Line2D> no_lines;
    const std::vector<Eigen::Vector2d> no_depths;
    RansacStats stats = estimate_threeview_relative_pose(x[0], x[1], x[2], none, none, none, no_lines, no_lines, no_lines,
                                                         no_depths, no_depths, no_depths, features, camera, camera,
                                                         camera, opt, &model, &inliers);
    REQUIRE_EQ(stats.num_inliers, static_cast<size_t>(60));
    REQUIRE(close_to_gt(model, gt, 1e-5));
    return true;
}

} // namespace test::threeview

using namespace test::threeview;
std::vector<Test> register_threeview_test() {
    return {TEST(test_threeview_jacobian), TEST(test_threeview_estimator), TEST(test_threeview_refine),
            TEST(test_threeview_missing_depth), TEST(test_threeview_2sift_solver), TEST(test_threeview_2sift_estimator)};
}
