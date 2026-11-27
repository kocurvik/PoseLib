#include "test.h"
#include "test_rng.h"

#include <PoseLib/robust/bundle.h>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

using namespace poselib;

// Checks that every public refinement entry point in bundle.h passes its weights on to the refiner.
//
// Each case evaluates the initial cost (max_iterations = 0, trivial loss) once without weights and once with all
// weights set to 2. The cost is linear in the weights, so it must double exactly. If the weights are dropped
// anywhere between the entry point and the refiner, the two costs are equal.

namespace {

struct Scene {
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<Point3D> X;
    std::vector<double> d1, d2;
    std::vector<Line2D> lines2D;
    std::vector<Line3D> lines3D;

    Scene() {
        test_rng::Rng rng = test_rng::make_rng("bundle_weights_scene");
        pose = CameraPose(Eigen::Vector4d(1.0, 0.1, -0.05, 0.02).normalized(), Eigen::Vector3d(1.0, 0.2, 0.1));
        for (size_t i = 0; i < 10; ++i) {
            // The data does not need to be consistent with the pose, only give non-zero residuals
            x1.push_back(test_rng::symmetric_vec2(rng, 0.5));
            x2.push_back(test_rng::symmetric_vec2(rng, 0.5));
            X.push_back(test_rng::symmetric_vec3(rng) + Eigen::Vector3d(0.0, 0.0, 4.0));
            d1.push_back(2.0 + 0.1 * i);
            d2.push_back(3.0 - 0.1 * i);
            lines2D.emplace_back(test_rng::symmetric_vec2(rng, 0.5), test_rng::symmetric_vec2(rng, 0.5));
            lines3D.emplace_back(test_rng::symmetric_vec3(rng) + Eigen::Vector3d(0.0, 0.0, 4.0),
                                 test_rng::symmetric_vec3(rng) + Eigen::Vector3d(0.0, 0.0, 4.0));
        }
    }
};

BundleOptions initial_cost_options() {
    BundleOptions opt;
    opt.max_iterations = 0;
    opt.loss_type = BundleOptions::LossType::TRIVIAL;
    return opt;
}

// Weights for n residuals: empty when w < 0, otherwise all equal to w
std::vector<double> weights(size_t n, double w) { return w < 0 ? std::vector<double>() : std::vector<double>(n, w); }

std::vector<std::vector<double>> weights(const std::vector<size_t> &sizes, double w) {
    std::vector<std::vector<double>> out;
    if (w >= 0) {
        for (size_t n : sizes) {
            out.push_back(weights(n, w));
        }
    }
    return out;
}

Camera pinhole(double f = 1.0) { return Camera("SIMPLE_PINHOLE", {f, 0.0, 0.0}, -1, -1); }

Eigen::Matrix3d skew(const Eigen::Vector3d &t) {
    Eigen::Matrix3d S;
    S << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
    return S;
}

// One case per weight vector: returns the initial cost with all weights set to w (no weights for w < 0)
typedef std::pair<std::string, std::function<double(double)>> WeightCase;

std::vector<WeightCase> weight_cases(const Scene &s) {
    const BundleOptions opt = initial_cost_options();
    const size_t n = s.x1.size();
    const Eigen::Matrix3d E = skew(s.pose.t) * s.pose.R();
    const Eigen::Matrix3d H = (Eigen::Matrix3d() << 1.1, 0.1, 0.3, -0.05, 0.9, -0.2, 0.01, -0.02, 1.0).finished();
    const Camera division1("DIVISION", {1.0, 1.0, 0.0, 0.0, -0.2}, -1, -1);
    const Camera division2("DIVISION", {1.0, 1.0, 0.0, 0.0, -0.1}, -1, -1);

    // Two-camera rigs for the generalized and hybrid problems
    // The second camera must differ from s.pose, otherwise one camera pair has zero baseline (E = 0)
    const std::vector<CameraPose> rig = {
        CameraPose(), CameraPose(Eigen::Vector4d(1.0, -0.05, 0.1, 0.0).normalized(), Eigen::Vector3d(0.0, 1.0, 0.2))};
    const std::vector<Camera> rig_cameras = {pinhole(), pinhole()};
    std::vector<PairwiseMatches> matches(2);
    for (size_t k = 0; k < 2; ++k) {
        matches[k].cam_id1 = k;
        matches[k].cam_id2 = 1 - k;
        matches[k].x1 = s.x1;
        matches[k].x2 = s.x2;
    }
    std::vector<PairwiseMatches> hybrid_matches = matches;
    for (PairwiseMatches &m : hybrid_matches) {
        m.cam_id2 = 0;
    }
    const std::vector<size_t> match_sizes = {n, n};
    const std::vector<std::vector<Point2D>> gen_x = {s.x1, s.x2};
    const std::vector<std::vector<Point3D>> gen_X = {s.X, s.X};

    std::vector<Point3D> b1, b2;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2;
    pinhole().unproject_with_jac(s.x1, &b1, &M1);
    pinhole().unproject_with_jac(s.x2, &b2, &M2);

    return {
        // Absolute pose
        {"bundle_adjust (CameraPose)",
         [=](double w) {
             CameraPose pose = s.pose;
             return bundle_adjust(s.x1, s.X, &pose, opt, weights(n, w)).initial_cost;
         }},
        {"bundle_adjust (Image)",
         [=](double w) {
             Image image(s.pose, pinhole());
             return bundle_adjust(s.x1, s.X, &image, opt, weights(n, w)).initial_cost;
         }},
        {"bundle_adjust (points+lines, CameraPose): weights_pts",
         [=](double w) {
             CameraPose pose = s.pose;
             return bundle_adjust(s.x1, s.X, {}, {}, &pose, opt, opt, weights(n, w), {}).initial_cost;
         }},
        {"bundle_adjust (points+lines, CameraPose): weights_lines",
         [=](double w) {
             CameraPose pose = s.pose;
             return bundle_adjust({}, {}, s.lines2D, s.lines3D, &pose, opt, opt, {}, weights(n, w)).initial_cost;
         }},
        {"bundle_adjust (points+lines, Image): weights_pts",
         [=](double w) {
             Image image(s.pose, pinhole());
             return bundle_adjust(s.x1, s.X, {}, {}, &image, opt, opt, weights(n, w), {}).initial_cost;
         }},
        {"bundle_adjust (points+lines, Image): weights_lines",
         [=](double w) {
             Image image(s.pose, pinhole());
             return bundle_adjust({}, {}, s.lines2D, s.lines3D, &image, opt, opt, {}, weights(n, w)).initial_cost;
         }},
        {"bundle_adjust_1D_radial",
         [=](double w) {
             CameraPose pose = s.pose;
             const Camera camera("1D_RADIAL", {0.0, 0.0}, -1, -1);
             return bundle_adjust_1D_radial(s.x1, s.X, &pose, camera, opt, weights(n, w)).initial_cost;
         }},
        {"generalized_bundle_adjust (calibrated)",
         [=](double w) {
             CameraPose pose = s.pose;
             return generalized_bundle_adjust(gen_x, gen_X, rig, &pose, opt, weights({n, n}, w)).initial_cost;
         }},
        {"generalized_bundle_adjust (cameras)",
         [=](double w) {
             CameraPose pose = s.pose;
             return generalized_bundle_adjust(gen_x, gen_X, rig, rig_cameras, &pose, opt, weights({n, n}, w))
                 .initial_cost;
         }},

        // Relative pose
        {"refine_relpose (CameraPose)",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_relpose(s.x1, s.x2, &pose, opt, weights(n, w)).initial_cost;
         }},
        {"refine_relpose (ImagePair)",
         [=](double w) {
             ImagePair pair(s.pose, pinhole(), pinhole());
             return refine_relpose(s.x1, s.x2, &pair, opt, weights(n, w)).initial_cost;
         }},
        {"refine_relpose (bearings and jacobians)",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_relpose(b1, b2, M1, M2, &pose, opt, weights(n, w)).initial_cost;
         }},
        {"refine_shared_focal_relpose",
         [=](double w) {
             ImagePair pair(s.pose, pinhole(1.2), pinhole(1.2));
             return refine_shared_focal_relpose(s.x1, s.x2, &pair, opt, weights(n, w)).initial_cost;
         }},
        {"refine_varying_focal_relpose",
         [=](double w) {
             ImagePair pair(s.pose, pinhole(1.2), pinhole(0.8));
             return refine_varying_focal_relpose(s.x1, s.x2, &pair, opt, weights(n, w)).initial_cost;
         }},
        {"refine_monodepth_relpose",
         [=](double w) {
             MonoDepthTwoViewGeometry geometry(s.pose, 1.1);
             return refine_monodepth_relpose(s.x1, s.x2, s.d1, s.d2, &geometry, 1.0, 1.0, opt, false, weights(n, w))
                 .initial_cost;
         }},
        {"refine_monodepth_shared_focal_relpose",
         [=](double w) {
             MonoDepthImagePair pair(MonoDepthTwoViewGeometry(s.pose, 1.1), pinhole(1.2), pinhole(1.2));
             return refine_monodepth_shared_focal_relpose(s.x1, s.x2, s.d1, s.d2, &pair, 1.0, 1.0, opt, weights(n, w))
                 .initial_cost;
         }},
        {"refine_monodepth_varying_focal_relpose",
         [=](double w) {
             MonoDepthImagePair pair(MonoDepthTwoViewGeometry(s.pose, 1.1), pinhole(1.2), pinhole(0.8));
             return refine_monodepth_varying_focal_relpose(s.x1, s.x2, s.d1, s.d2, &pair, 1.0, 1.0, opt, weights(n, w))
                 .initial_cost;
         }},

        // Fundamental matrix and homography
        {"refine_fundamental",
         [=](double w) {
             Eigen::Matrix3d F = E;
             return refine_fundamental(s.x1, s.x2, &F, opt, weights(n, w)).initial_cost;
         }},
        {"refine_rd_fundamental",
         [=](double w) {
             ProjectiveImagePair pair(E, division1, division2);
             return refine_rd_fundamental(s.x1, s.x2, &pair, opt, weights(n, w)).initial_cost;
         }},
        {"refine_shared_rd_fundamental",
         [=](double w) {
             ProjectiveImagePair pair(E, division1, division1);
             return refine_shared_rd_fundamental(s.x1, s.x2, &pair, opt, weights(n, w)).initial_cost;
         }},
        {"refine_homography",
         [=](double w) {
             Eigen::Matrix3d HH = H;
             return refine_homography(s.x1, s.x2, &HH, opt, weights(n, w)).initial_cost;
         }},

        // Generalized relative and hybrid pose
        {"refine_generalized_relpose (calibrated)",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_generalized_relpose(matches, rig, rig, &pose, opt, weights(match_sizes, w)).initial_cost;
         }},
        {"refine_generalized_relpose (cameras)",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_generalized_relpose(matches, rig, rig_cameras, rig, rig_cameras, &pose, opt,
                                               weights(match_sizes, w))
                 .initial_cost;
         }},
        {"refine_hybrid_pose: weights_abs",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_hybrid_pose(s.x1, s.X, {}, rig, &pose, opt, 1.0, weights(n, w), {}).initial_cost;
         }},
        {"refine_hybrid_pose: weights_rel",
         [=](double w) {
             CameraPose pose = s.pose;
             return refine_hybrid_pose({}, {}, hybrid_matches, rig, &pose, opt, 1.0, {}, weights(match_sizes, w))
                 .initial_cost;
         }},
    };
}

bool test_refinement_entry_points_forward_weights() {
    const Scene scene;
    for (const WeightCase &c : weight_cases(scene)) {
        const double cost = c.second(-1.0);
        const double weighted_cost = c.second(2.0);
        REQUIRE_M(cost > 0.0, c.first + ": initial cost is zero, the case does not test anything");
        REQUIRE_M(std::abs(weighted_cost - 2.0 * cost) <= 1e-9 * cost,
                  c.first + ": weights were not passed to the refiner (cost " + std::to_string(cost) +
                      ", with all weights 2: " + std::to_string(weighted_cost) + ")");
    }
    return true;
}

} // namespace

std::vector<Test> register_bundle_weights_test() { return {TEST(test_refinement_entry_points_forward_weights)}; }
