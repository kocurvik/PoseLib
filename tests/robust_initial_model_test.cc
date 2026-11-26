#include "test.h"
#include "test_rng.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/robust.h>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace poselib;

// These tests check that an initial model supplied with ransac.score_initial_model = true is actually used.
// RANSAC is run with zero iterations on noise-free data, so the only hypothesis is the initial model
// (ground truth). The estimate should therefore stay at the ground truth.

namespace test::robust_initial_model {

constexpr double kTol = 1e-6;

CameraPose random_relative_pose(test_rng::Rng &rng) {
    Eigen::Vector3d aa = test_rng::symmetric_vec3(rng, 0.3);
    Eigen::Matrix3d R = Eigen::AngleAxisd(aa.norm(), aa.normalized()).toRotationMatrix();
    Eigen::Vector3d t = test_rng::symmetric_vec3(rng);
    t(2) = 0.2 * t(2);
    t.normalize();
    return CameraPose(R, t);
}

// 3D points in front of camera 1 (identity) and camera 2 (pose)
std::vector<Point3D> random_points(test_rng::Rng &rng, const CameraPose &pose, size_t N, double spread = 1.0) {
    std::vector<Point3D> X;
    while (X.size() < N) {
        Eigen::Vector3d Xi = test_rng::symmetric_vec3(rng);
        Xi.head<2>() *= spread;
        Xi(2) += 4.0;
        if (pose.apply(Xi)(2) > 0.5) {
            X.push_back(Xi);
        }
    }
    return X;
}

bool pose_close(const CameraPose &est, const CameraPose &gt) {
    REQUIRE_SMALL((est.R() - gt.R()).norm(), kTol);
    REQUIRE_SMALL((est.t - gt.t).norm(), kTol);
    return true;
}

bool fundamental_close(Eigen::Matrix3d F, Eigen::Matrix3d F_gt) {
    F /= F.norm();
    F_gt /= F_gt.norm();
    if (F.cwiseProduct(F_gt).sum() < 0) {
        F = -F;
    }
    REQUIRE_SMALL((F - F_gt).norm(), kTol);
    return true;
}

RansacOptions initial_model_only_ransac() {
    RansacOptions opt;
    opt.score_initial_model = true;
    opt.min_iterations = 0;
    opt.max_iterations = 0;
    return opt;
}

// Random world-to-camera pose and N world points in front of the camera
CameraPose random_absolute_pose(test_rng::Rng &rng) {
    Eigen::Vector3d aa = test_rng::symmetric_vec3(rng, 1.0);
    Eigen::Matrix3d R = Eigen::AngleAxisd(aa.norm(), aa.normalized()).toRotationMatrix();
    return CameraPose(R, 2.0 * test_rng::symmetric_vec3(rng));
}

std::vector<Point3D> random_world_points(test_rng::Rng &rng, const CameraPose &pose, size_t N) {
    std::vector<Point3D> X;
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xc = test_rng::symmetric_vec3(rng);
        Xc(2) += 4.0;
        X.push_back(pose.R().transpose() * (Xc - pose.t));
    }
    return X;
}

bool check_absolute_pose_focal(bool estimate_distortion) {
    test_rng::Rng rng =
        test_rng::make_rng(estimate_distortion ? "absolute_pose_focal_distortion" : "absolute_pose_focal");
    const double focal = 1200.0;
    Camera camera = estimate_distortion ? Camera("SIMPLE_DIVISION", {focal, 320.0, 240.0, 0.0}, 640, 480)
                                        : Camera("SIMPLE_PINHOLE", {focal, 320.0, 240.0}, 640, 480);
    CameraPose pose_gt = random_absolute_pose(rng);
    std::vector<Point3D> X = random_world_points(rng, pose_gt, 50);
    std::vector<Point2D> x(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        camera.project(pose_gt.apply(X[i]), &x[i]);
    }

    AbsolutePoseOptions opt;
    opt.ransac = initial_model_only_ransac();
    opt.estimate_focal_length = true;
    opt.estimate_extra_params = estimate_distortion;
    Image image(pose_gt, camera);
    std::vector<char> inliers;
    estimate_absolute_pose(x, X, opt, &image, &inliers);

    REQUIRE(pose_close(image.pose, pose_gt));
    REQUIRE_SMALL(image.camera.focal() / focal - 1.0, kTol);
    REQUIRE_SMALL((image.camera.principal_point() - camera.principal_point()).norm(), kTol);
    return true;
}

bool test_absolute_pose_focal() { return check_absolute_pose_focal(false); }

bool test_absolute_pose_focal_distortion() { return check_absolute_pose_focal(true); }

bool test_absolute_pose_pnplf() {
    test_rng::Rng rng = test_rng::make_rng("absolute_pose_pnplf");
    const double focal = 1200.0;
    Camera camera("SIMPLE_PINHOLE", {focal, 320.0, 240.0}, 640, 480);
    CameraPose pose_gt = random_absolute_pose(rng);
    std::vector<Point3D> X = random_world_points(rng, pose_gt, 30);
    std::vector<Point3D> L1 = random_world_points(rng, pose_gt, 20);
    std::vector<Point3D> L2 = random_world_points(rng, pose_gt, 20);
    std::vector<Point2D> x(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        camera.project(pose_gt.apply(X[i]), &x[i]);
    }
    std::vector<Line2D> lines2D(L1.size());
    std::vector<Line3D> lines3D(L1.size());
    for (size_t i = 0; i < L1.size(); ++i) {
        lines3D[i] = Line3D(L1[i], L2[i]);
        camera.project(pose_gt.apply(L1[i]), &lines2D[i].x1);
        camera.project(pose_gt.apply(L2[i]), &lines2D[i].x2);
    }

    AbsolutePoseOptions opt;
    opt.ransac = initial_model_only_ransac();
    Image image(pose_gt, camera);
    std::vector<char> inliers_points, inliers_lines;
    estimate_absolute_pose_pnplf(x, X, lines2D, lines3D, camera, opt, &image, &inliers_points, &inliers_lines);

    REQUIRE(pose_close(image.pose, pose_gt));
    REQUIRE_SMALL(image.camera.focal() / focal - 1.0, kTol);
    return true;
}

bool test_relative_pose_tangent_sampson() {
    test_rng::Rng rng = test_rng::make_rng("relative_pose_tangent_sampson");
    CameraPose pose_gt = random_relative_pose(rng);
    std::vector<Point3D> X = random_points(rng, pose_gt, 50);
    Camera camera1(SimplePinholeCameraModel::model_id, {1000.0, 0.0, 0.0}, 0, 0);
    Camera camera2(SimplePinholeCameraModel::model_id, {1500.0, 0.0, 0.0}, 0, 0);
    std::vector<Point2D> x1(X.size()), x2(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        camera1.project(X[i], &x1[i]);
        camera2.project(pose_gt.apply(X[i]), &x2[i]);
    }

    RelativePoseOptions opt;
    opt.ransac = initial_model_only_ransac();
    opt.tangent_sampson = true;
    CameraPose pose = pose_gt;
    std::vector<char> inliers;
    estimate_relative_pose(x1, x2, camera1, camera2, opt, &pose, &inliers);

    REQUIRE(pose_close(pose, pose_gt));
    return true;
}

// Generates monodepth correspondences for cameras with focal lengths f1, f2 (pp at origin)
void setup_monodepth(test_rng::Rng &rng, double f1, double f2, CameraPose *pose_gt, std::vector<Point2D> *x1,
                     std::vector<Point2D> *x2, std::vector<double> *d1, std::vector<double> *d2) {
    *pose_gt = random_relative_pose(rng);
    std::vector<Point3D> X = random_points(rng, *pose_gt, 50);
    for (const Point3D &Xi : X) {
        Eigen::Vector3d Z2 = pose_gt->apply(Xi);
        x1->push_back(f1 * Xi.hnormalized());
        x2->push_back(f2 * Z2.hnormalized());
        d1->push_back(Xi(2));
        d2->push_back(Z2(2));
    }
}

MonoDepthRelativePoseOptions monodepth_options() {
    MonoDepthRelativePoseOptions opt;
    opt.ransac = initial_model_only_ransac();
    opt.max_errors = {4.0, 1.0};
    return opt;
}

bool test_monodepth_relative_pose() {
    test_rng::Rng rng = test_rng::make_rng("monodepth_relative_pose");
    const double f1 = 1000.0, f2 = 1500.0;
    CameraPose pose_gt;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth(rng, f1, f2, &pose_gt, &x1, &x2, &d1, &d2);

    Camera camera1(SimplePinholeCameraModel::model_id, {f1, 0.0, 0.0}, 0, 0);
    Camera camera2(SimplePinholeCameraModel::model_id, {f2, 0.0, 0.0}, 0, 0);
    // Non-zero shifts (e.g. from an earlier run with estimate_shift) must be discarded when shifts are not estimated
    MonoDepthTwoViewGeometry geometry(pose_gt, 1.0, 0.5, -0.5);
    std::vector<char> inliers;
    estimate_monodepth_relative_pose(x1, x2, d1, d2, camera1, camera2, monodepth_options(), &geometry, &inliers);

    REQUIRE(pose_close(geometry.pose, pose_gt));
    REQUIRE_SMALL(geometry.scale - 1.0, kTol);
    REQUIRE_EQ(geometry.shift1, 0.0);
    REQUIRE_EQ(geometry.shift2, 0.0);
    return true;
}

bool test_shared_focal_monodepth_relative_pose() {
    test_rng::Rng rng = test_rng::make_rng("shared_focal_monodepth_relative_pose");
    const double focal = 1200.0;
    CameraPose pose_gt;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth(rng, focal, focal, &pose_gt, &x1, &x2, &d1, &d2);

    Camera camera(SimplePinholeCameraModel::model_id, {focal, 0.0, 0.0}, -1, -1);
    MonoDepthImagePair image_pair(MonoDepthTwoViewGeometry(pose_gt, 1.0, 0.5, -0.5), camera, camera);
    std::vector<char> inliers;
    estimate_shared_focal_monodepth_relative_pose(x1, x2, d1, d2, Point2D::Zero(), monodepth_options(), &image_pair,
                                                  &inliers);

    REQUIRE(pose_close(image_pair.geometry.pose, pose_gt));
    REQUIRE_SMALL(image_pair.geometry.scale - 1.0, kTol);
    REQUIRE_EQ(image_pair.geometry.shift1, 0.0);
    REQUIRE_EQ(image_pair.geometry.shift2, 0.0);
    REQUIRE_SMALL(image_pair.camera1.focal() / focal - 1.0, kTol);
    REQUIRE_SMALL(image_pair.camera2.focal() / focal - 1.0, kTol);
    return true;
}

bool test_varying_focal_monodepth_relative_pose() {
    test_rng::Rng rng = test_rng::make_rng("varying_focal_monodepth_relative_pose");
    const double f1 = 1000.0, f2 = 1500.0;
    CameraPose pose_gt;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth(rng, f1, f2, &pose_gt, &x1, &x2, &d1, &d2);

    Camera camera1(SimplePinholeCameraModel::model_id, {f1, 0.0, 0.0}, -1, -1);
    Camera camera2(SimplePinholeCameraModel::model_id, {f2, 0.0, 0.0}, -1, -1);
    MonoDepthImagePair image_pair(MonoDepthTwoViewGeometry(pose_gt, 1.0, 0.5, -0.5), camera1, camera2);
    std::vector<char> inliers;
    estimate_varying_focal_monodepth_relative_pose(x1, x2, d1, d2, Point2D::Zero(), Point2D::Zero(),
                                                   monodepth_options(), &image_pair, &inliers);

    REQUIRE(pose_close(image_pair.geometry.pose, pose_gt));
    REQUIRE_SMALL(image_pair.geometry.scale - 1.0, kTol);
    REQUIRE_EQ(image_pair.geometry.shift1, 0.0);
    REQUIRE_EQ(image_pair.geometry.shift2, 0.0);
    REQUIRE_SMALL(image_pair.camera1.focal() / f1 - 1.0, kTol);
    REQUIRE_SMALL(image_pair.camera2.focal() / f2 - 1.0, kTol);
    return true;
}

// Each helper returns true if the estimator throws std::invalid_argument
bool focal_relative_pose_throws(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Point2D &pp,
                                const RelativePoseOptions &opt, bool shared, ImagePair *image_pair,
                                std::vector<char> *inliers) {
    try {
        if (shared) {
            estimate_shared_focal_relative_pose(x1, x2, pp, opt, image_pair, inliers);
        } else {
            estimate_varying_focal_relative_pose(x1, x2, pp, pp, opt, image_pair, inliers);
        }
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

bool focal_monodepth_relative_pose_throws(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                          const std::vector<double> &d1, const std::vector<double> &d2,
                                          const Point2D &pp, const MonoDepthRelativePoseOptions &opt, bool shared,
                                          MonoDepthImagePair *md_pair, std::vector<char> *inliers) {
    try {
        if (shared) {
            estimate_shared_focal_monodepth_relative_pose(x1, x2, d1, d2, pp, opt, md_pair, inliers);
        } else {
            estimate_varying_focal_monodepth_relative_pose(x1, x2, d1, d2, pp, pp, opt, md_pair, inliers);
        }
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

bool rd_fundamental_throws(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                           const RelativePoseOptions &opt, bool shared, ProjectiveImagePair *model,
                           std::vector<char> *inliers) {
    try {
        if (shared) {
            estimate_shared_rd_fundamental(x1, x2, {}, opt, model, inliers);
        } else {
            estimate_rd_fundamental(x1, x2, {}, opt, model, inliers);
        }
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

// Runs all four focal estimators (shared ones only if shared_only is set) with the given initial cameras and checks
// that each throws and leaves the valid camera1 (PINHOLE, would be converted to SIMPLE_PINHOLE) unchanged
bool all_focal_estimators_throw(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                const std::vector<double> &d1, const std::vector<double> &d2, const Point2D &pp,
                                const RelativePoseOptions &opt, const Camera &camera1, const Camera &camera2,
                                bool shared_only) {
    std::vector<char> inliers;
    for (int s = 0; s < 2; ++s) {
        const bool shared = (s == 0);
        if (!shared && shared_only)
            continue;
        ImagePair image_pair(CameraPose(), camera1, camera2);
        MonoDepthImagePair md_pair(MonoDepthTwoViewGeometry(), camera1, camera2);
        REQUIRE(focal_relative_pose_throws(x1, x2, pp, opt, shared, &image_pair, &inliers));
        REQUIRE(
            focal_monodepth_relative_pose_throws(x1, x2, d1, d2, pp, monodepth_options(), shared, &md_pair, &inliers));
        REQUIRE_EQ(image_pair.camera1.model_id, camera1.model_id);
        REQUIRE_EQ(md_pair.camera1.model_id, camera1.model_id);
    }
    return true;
}

// Initial PINHOLE cameras with non-zero principal points: the estimate must keep the initial focal lengths and return
// SIMPLE_PINHOLE cameras with the principal points passed as arguments
bool test_focal_relative_pose_principal_point() {
    test_rng::Rng rng = test_rng::make_rng("focal_relative_pose_principal_point");
    const double f1 = 1000.0, f2 = 1500.0;
    const Point2D pp1(320.0, 240.0), pp2(400.0, 300.0);

    for (bool shared : {true, false}) {
        const double f2_used = shared ? f1 : f2;
        const Point2D pp2_used = shared ? pp1 : pp2;
        CameraPose pose_gt;
        std::vector<Point2D> x1, x2;
        std::vector<double> d1, d2;
        setup_monodepth(rng, f1, f2_used, &pose_gt, &x1, &x2, &d1, &d2);
        for (size_t i = 0; i < x1.size(); ++i) {
            x1[i] += pp1;
            x2[i] += pp2_used;
        }
        const Camera camera1("PINHOLE", {f1, f1, pp1(0), pp1(1)}, -1, -1);
        const Camera camera2("PINHOLE", {f2_used, f2_used, pp2_used(0), pp2_used(1)}, -1, -1);
        std::vector<char> inliers;

        RelativePoseOptions opt;
        opt.ransac = initial_model_only_ransac();
        ImagePair image_pair(pose_gt, camera1, camera2);
        MonoDepthImagePair md_pair(MonoDepthTwoViewGeometry(pose_gt, 1.0), camera1, camera2);
        if (shared) {
            estimate_shared_focal_relative_pose(x1, x2, pp1, opt, &image_pair, &inliers);
            estimate_shared_focal_monodepth_relative_pose(x1, x2, d1, d2, pp1, monodepth_options(), &md_pair, &inliers);
        } else {
            estimate_varying_focal_relative_pose(x1, x2, pp1, pp2, opt, &image_pair, &inliers);
            estimate_varying_focal_monodepth_relative_pose(x1, x2, d1, d2, pp1, pp2, monodepth_options(), &md_pair,
                                                           &inliers);
        }

        REQUIRE(pose_close(image_pair.pose, pose_gt));
        REQUIRE(pose_close(md_pair.geometry.pose, pose_gt));
        const ImagePair md_result(md_pair.geometry.pose, md_pair.camera1, md_pair.camera2);
        for (const ImagePair &result : {image_pair, md_result}) {
            REQUIRE_EQ(result.camera1.model_id, SimplePinholeCameraModel::model_id);
            REQUIRE_EQ(result.camera2.model_id, SimplePinholeCameraModel::model_id);
            REQUIRE_SMALL(result.camera1.focal() / f1 - 1.0, kTol);
            REQUIRE_SMALL(result.camera2.focal() / f2_used - 1.0, kTol);
            REQUIRE_SMALL((result.camera1.principal_point() - pp1).norm(), kTol);
            REQUIRE_SMALL((result.camera2.principal_point() - pp2_used).norm(), kTol);
        }
    }
    return true;
}

bool test_focal_relative_pose_invalid_initial_model() {
    test_rng::Rng rng = test_rng::make_rng("focal_relative_pose_invalid_initial_model");
    std::vector<Point2D> x1, x2;
    std::vector<double> d1(20, 4.0), d2(20, 4.0);
    for (size_t i = 0; i < 20; ++i) {
        x1.push_back(test_rng::symmetric_vec2(rng));
        x2.push_back(test_rng::symmetric_vec2(rng));
    }
    const Point2D pp(320.0, 240.0);
    RelativePoseOptions opt;
    opt.ransac = initial_model_only_ransac();

    const Camera valid("PINHOLE", {1000.0, 1000.0, pp(0), pp(1)}, -1, -1);
    const std::vector<Camera> invalid_cameras = {
        Camera("SIMPLE_RADIAL", {1000.0, pp(0), pp(1), 0.0}, -1, -1), // unsupported model
        Camera("PINHOLE", {1000.0, 1100.0, pp(0), pp(1)}, -1, -1),    // fx != fy
        Camera("SIMPLE_PINHOLE", {1000.0, 0.0, 0.0}, -1, -1),         // principal point differs from pp
        Camera("SIMPLE_PINHOLE", {-1000.0, pp(0), pp(1)}, -1, -1)};   // negative focal
    for (const Camera &invalid : invalid_cameras) {
        REQUIRE(all_focal_estimators_throw(x1, x2, d1, d2, pp, opt, valid, invalid, false));
    }

    // The shared variants require both cameras to have the same focal length
    REQUIRE(all_focal_estimators_throw(x1, x2, d1, d2, pp, opt, valid,
                                       Camera("SIMPLE_PINHOLE", {1100.0, pp(0), pp(1)}, -1, -1), true));
    return true;
}

bool check_rd_fundamental(bool shared) {
    test_rng::Rng rng = test_rng::make_rng(shared ? "shared_rd_fundamental" : "rd_fundamental");
    const double f1 = 1000.0, f2 = shared ? 1000.0 : 1500.0;
    // Strong distortion and wide field of view, so a wrongly scaled initial distortion is not recovered by refinement
    const double k1 = -0.6, k2 = shared ? -0.6 : -0.4;
    CameraPose pose_gt = random_relative_pose(rng);
    std::vector<Point3D> X = random_points(rng, pose_gt, 50, 3.0);

    // Distorted image points
    Camera camera1("DIVISION", {f1, f1, 0.0, 0.0, k1}, -1, -1);
    Camera camera2("DIVISION", {f2, f2, 0.0, 0.0, k2}, -1, -1);
    std::vector<Point2D> x1(X.size()), x2(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        camera1.project(X[i], &x1[i]);
        camera2.project(pose_gt.apply(X[i]), &x2[i]);
    }

    // Ground truth model: F acts on undistorted pixel coordinates, distortion is expressed in pixels
    Eigen::Matrix3d E;
    essential_from_motion(pose_gt, &E);
    Eigen::DiagonalMatrix<double, 3> K1_inv(1.0, 1.0, f1), K2_inv(1.0, 1.0, f2);
    Eigen::Matrix3d F_gt = K2_inv * E * K1_inv;
    const double k1_px = k1 / (f1 * f1), k2_px = k2 / (f2 * f2);

    // The same initial model expressed with DIVISION and SIMPLE_DIVISION cameras, and with a focal length that is
    // unit only up to floating point error
    const double f_almost_one = f1 * (1.0 / f1) - 1e-15;
    std::vector<ProjectiveImagePair> initial_models = {
        ProjectiveImagePair(F_gt, Camera("DIVISION", {1.0, 1.0, 0.0, 0.0, k1_px}, -1, -1),
                            Camera("DIVISION", {1.0, 1.0, 0.0, 0.0, k2_px}, -1, -1)),
        ProjectiveImagePair(F_gt, Camera("SIMPLE_DIVISION", {1.0, 0.0, 0.0, k1_px}, -1, -1),
                            Camera("SIMPLE_DIVISION", {1.0, 0.0, 0.0, k2_px}, -1, -1)),
        ProjectiveImagePair(F_gt, Camera("SIMPLE_DIVISION", {f_almost_one, 1e-12, 0.0, k1_px}, -1, -1),
                            Camera("DIVISION", {1.0, f_almost_one, 0.0, -1e-12, k2_px}, -1, -1))};

    for (ProjectiveImagePair model : initial_models) {
        RelativePoseOptions opt;
        opt.ransac = initial_model_only_ransac();
        std::vector<char> inliers;
        if (shared) {
            estimate_shared_rd_fundamental(x1, x2, {}, opt, &model, &inliers);
        } else {
            estimate_rd_fundamental(x1, x2, {}, opt, &model, &inliers);
        }

        // The estimators assume canonical DIVISION cameras {1, 1, 0, 0, k}
        for (const Camera &cam : {model.camera1, model.camera2}) {
            REQUIRE_EQ(cam.model_id, DivisionCameraModel::model_id);
            REQUIRE_EQ(cam.params[0], 1.0);
            REQUIRE_EQ(cam.params[1], 1.0);
            REQUIRE_EQ(cam.params[2], 0.0);
            REQUIRE_EQ(cam.params[3], 0.0);
        }
        REQUIRE(fundamental_close(model.F, F_gt));
        REQUIRE_SMALL(model.camera1.params[4] / k1_px - 1.0, kTol);
        REQUIRE_SMALL(model.camera2.params[4] / k2_px - 1.0, kTol);
    }
    return true;
}

bool test_rd_fundamental() { return check_rd_fundamental(false); }

bool test_shared_rd_fundamental() { return check_rd_fundamental(true); }

bool test_rd_fundamental_invalid_initial_model() {
    test_rng::Rng rng = test_rng::make_rng("rd_fundamental_invalid_initial_model");
    std::vector<Point2D> x1, x2;
    for (size_t i = 0; i < 20; ++i) {
        x1.push_back(test_rng::symmetric_vec2(rng));
        x2.push_back(test_rng::symmetric_vec2(rng));
    }
    // 9 points: below the minimal sample, so this also checks that validation happens before the early return
    const std::vector<Point2D> x1_few(x1.begin(), x1.begin() + 9), x2_few(x2.begin(), x2.begin() + 9);
    RelativePoseOptions opt;
    opt.ransac = initial_model_only_ransac();
    std::vector<char> inliers;

    const std::vector<Camera> invalid_cameras = {Camera("SIMPLE_PINHOLE", {1.0, 0.0, 0.0}, -1, -1),
                                                 Camera("DIVISION", {1000.0, 1000.0, 0.0, 0.0, 0.0}, -1, -1),
                                                 Camera("SIMPLE_DIVISION", {1.0, 0.5, 0.0, 0.0}, -1, -1)};
    // camera1 is valid but would be converted to DIVISION, so this also checks that a failed check modifies nothing
    const Camera valid("SIMPLE_DIVISION", {1.0, 0.0, 0.0, 0.0}, -1, -1);
    for (const Camera &invalid : invalid_cameras) {
        ProjectiveImagePair model(Eigen::Matrix3d::Identity(), valid, invalid);
        REQUIRE(rd_fundamental_throws(x1_few, x2_few, opt, false, &model, &inliers));
        REQUIRE_EQ(model.camera1.model_id, SimpleDivisionCameraModel::model_id);
    }

    // The shared variant requires both cameras to have the same distortion
    ProjectiveImagePair model(Eigen::Matrix3d::Identity(), Camera("DIVISION", {1.0, 1.0, 0.0, 0.0, -0.3}, -1, -1),
                              Camera("DIVISION", {1.0, 1.0, 0.0, 0.0, -0.1}, -1, -1));
    REQUIRE(rd_fundamental_throws(x1_few, x2_few, opt, true, &model, &inliers));
    return true;
}

} // namespace test::robust_initial_model

using namespace test::robust_initial_model;
std::vector<Test> register_robust_initial_model_test() {
    return {TEST(test_absolute_pose_focal),
            TEST(test_absolute_pose_focal_distortion),
            TEST(test_absolute_pose_pnplf),
            TEST(test_relative_pose_tangent_sampson),
            TEST(test_monodepth_relative_pose),
            TEST(test_shared_focal_monodepth_relative_pose),
            TEST(test_varying_focal_monodepth_relative_pose),
            TEST(test_focal_relative_pose_principal_point),
            TEST(test_focal_relative_pose_invalid_initial_model),
            TEST(test_rd_fundamental),
            TEST(test_shared_rd_fundamental),
            TEST(test_rd_fundamental_invalid_initial_model)};
}
