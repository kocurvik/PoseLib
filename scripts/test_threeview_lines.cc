// Standalone checks for the three view line solvers and the line Jacobian accumulator.
// Build (from repo root):
//   g++ -O2 -std=c++17 -I. -I<eigen include> scripts/test_threeview_lines.cc PoseLib/solvers/threeview_lines.cc
#include "PoseLib/robust/lm_impl.h"
#include "PoseLib/robust/robust_loss.h"
#include "PoseLib/robust/threeview_lines_jacobian.h"
#include "PoseLib/solvers/threeview_lines.h"

#include <iostream>
#include <random>

using namespace poselib;

static std::mt19937 rng(42);
static double randn() { return std::normal_distribution<double>(0.0, 1.0)(rng); }
static double randu(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
static Eigen::Vector3d randv() { return Eigen::Vector3d(randn(), randn(), randn()); }

static Eigen::Matrix3d rand_rot(double angle = 0.3) {
    return Eigen::AngleAxisd(angle * randn(), randv().normalized()).toRotationMatrix();
}

static double rot_err(const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2) {
    return Eigen::AngleAxisd(R1.transpose() * R2).angle();
}

static double trans_err(const Eigen::Vector3d &t1, const Eigen::Vector3d &t2) { return (t1 - t2).norm() / t1.norm(); }

struct Scene {
    Eigen::Matrix3d R2, R3;
    Eigen::Vector3d t2, t3;
    double s[3]; // depth scale per view
};

static Scene random_scene() {
    Scene sc;
    sc.R2 = rand_rot();
    sc.R3 = rand_rot();
    sc.t2 = randv();
    sc.t3 = randv();
    sc.s[0] = randu(0.5, 2);
    sc.s[1] = randu(0.5, 2);
    sc.s[2] = randu(0.5, 2);
    return sc;
}

static Eigen::Vector3d random_point() { return Eigen::Vector3d(randu(-2, 2), randu(-2, 2), randu(4, 8)); }

// Returns 3D line segment in each camera frame, segments in different views are different parts of the line
static void random_line(const Scene &sc, Line3D L[3], Line2D l[3], bool flip_random = true) {
    Eigen::Vector3d P = random_point();
    Eigen::Vector3d d = randv().normalized();
    const Eigen::Matrix3d R[3] = {Eigen::Matrix3d::Identity(), sc.R2, sc.R3};
    const Eigen::Vector3d t[3] = {Eigen::Vector3d::Zero(), sc.t2, sc.t3};
    for (int k = 0; k < 3; ++k) {
        double a = randu(-1.0, -0.3), b = randu(0.3, 1.0);
        if (flip_random && randu(0, 1) < 0.5)
            std::swap(a, b);
        Eigen::Vector3d A = R[k] * (P + a * d) + t[k];
        Eigen::Vector3d B = R[k] * (P + b * d) + t[k];
        L[k] = Line3D(sc.s[k] * A, sc.s[k] * B);
        l[k] = Line2D(A.hnormalized(), B.hnormalized());
    }
}

static double best_error(const std::vector<ThreeViewCameraPose> &models, const Scene &sc) {
    double best = 1e10;
    for (const ThreeViewCameraPose &m : models) {
        double e = std::max({rot_err(m.pose12.R(), sc.R2), rot_err(m.pose13.R(), sc.R3), trans_err(sc.s[0] * sc.t2, m.pose12.t),
                             trans_err(sc.s[0] * sc.t3, m.pose13.t)});
        best = std::min(best, e);
    }
    return best;
}

static int test_solvers() {
    int fails = 0;
    const int trials = 1000;
    int fail_2p1l = 0, fail_2l = 0, fail_2lp = 0;
    double max_models[3] = {0, 0, 0};
    for (int trial = 0; trial < trials; ++trial) {
        Scene sc = random_scene();
        const Eigen::Matrix3d R[3] = {Eigen::Matrix3d::Identity(), sc.R2, sc.R3};
        const Eigen::Vector3d t[3] = {Eigen::Vector3d::Zero(), sc.t2, sc.t3};

        // 2p1l
        std::vector<Eigen::Vector3d> X[3];
        Eigen::Vector3d P1 = random_point(), P2 = random_point();
        for (int k = 0; k < 3; ++k) {
            X[k].push_back(sc.s[k] * (R[k] * P1 + t[k]));
            X[k].push_back(sc.s[k] * (R[k] * P2 + t[k]));
        }
        Line3D L[3], Lb[3];
        Line2D l[3], lb[3];
        random_line(sc, L, l);
        random_line(sc, Lb, lb);

        std::vector<ThreeViewCameraPose> models;
        threeview_2p1l(X[0], X[1], X[2], L[0].X2 - L[0].X1, L[1].X2 - L[1].X1, L[2].X2 - L[2].X1, &models);
        max_models[0] = std::max<double>(max_models[0], models.size());
        if (best_error(models, sc) > 1e-6)
            fail_2p1l++;

        std::vector<Line3D> L1 = {L[0], Lb[0]}, L2 = {L[1], Lb[1]}, L3 = {L[2], Lb[2]};
        threeview_2l(L1, L2, L3, &models);
        max_models[1] = std::max<double>(max_models[1], models.size());
        if (best_error(models, sc) > 1e-6)
            fail_2l++;

        std::vector<Eigen::Vector3d> l2 = {l[1].x1.homogeneous().cross(l[1].x2.homogeneous()),
                                           lb[1].x1.homogeneous().cross(lb[1].x2.homogeneous())};
        std::vector<Eigen::Vector3d> l3 = {l[2].x1.homogeneous().cross(l[2].x2.homogeneous()),
                                           lb[2].x1.homogeneous().cross(lb[2].x2.homogeneous())};
        threeview_2l_planes(L1, L2, L3, l2, l3, &models);
        max_models[2] = std::max<double>(max_models[2], models.size());
        if (best_error(models, sc) > 1e-6)
            fail_2lp++;
    }
    std::cout << "2p1l failures: " << fail_2p1l << "/" << trials << " (max models " << max_models[0] << ")\n";
    std::cout << "2l failures: " << fail_2l << "/" << trials << " (max models " << max_models[1] << ")\n";
    std::cout << "2l_planes failures: " << fail_2lp << "/" << trials << " (max models " << max_models[2] << ")\n";
    // allow a few near-degenerate configurations
    fails += (fail_2p1l > trials / 100) + (fail_2l > trials / 100) + (fail_2lp > trials / 100);
    return fails;
}

static int test_jacobian(bool sampson) {
    Scene sc = random_scene();
    std::vector<Line2D> l1, l2, l3;
    for (int i = 0; i < 20; ++i) {
        Line3D L[3];
        Line2D l[3];
        random_line(sc, L, l);
        // add noise so that residuals are non-zero
        for (int k = 0; k < 3; ++k) {
            l[k].x1 += 0.01 * Eigen::Vector2d(randn(), randn());
            l[k].x2 += 0.01 * Eigen::Vector2d(randn(), randn());
        }
        l1.push_back(l[0]);
        l2.push_back(l[1]);
        l3.push_back(l[2]);
    }
    std::vector<ThreeViewLineTrack> tracks = make_line_tracks(l1, l2, l3);
    ThreeViewCameraPose pose(CameraPose(sc.R2, Eigen::Vector3d(sc.t2.normalized())), CameraPose(sc.R3, Eigen::Vector3d(sc.t3 / sc.t2.norm())));

    ThreeViewLineResidual res(pose, true, sampson);
    const double h = 1e-6;
    double max_rel_err = 0.0;
    for (const ThreeViewLineTrack &track : tracks) {
        Eigen::Matrix<double, 3, 2> r;
        std::array<Eigen::Matrix<double, 2, 11>, 3> J;
        res.residuals_and_jacobians(track, &r, &J);

        for (int p = 0; p < 11; ++p) {
            Eigen::Matrix<double, 11, 1> dp = Eigen::Matrix<double, 11, 1>::Zero();
            dp(p) = h;
            ThreeViewLineResidual res_p(threeview_step(dp, pose, res.tangent_basis), true, sampson);
            ThreeViewLineResidual res_m(threeview_step(-dp, pose, res.tangent_basis), true, sampson);
            Eigen::Matrix<double, 3, 2> rp, rm;
            std::array<Eigen::Matrix<double, 2, 11>, 3> Jtmp;
            res_p.residuals_and_jacobians(track, &rp, &Jtmp);
            res_m.residuals_and_jacobians(track, &rm, &Jtmp);
            for (int j = 0; j < 3; ++j) {
                for (int e = 0; e < 2; ++e) {
                    const double num = (rp(j, e) - rm(j, e)) / (2 * h);
                    const double ana = J[j](e, p);
                    const double err = std::abs(num - ana) / std::max(1.0, std::abs(num));
                    max_rel_err = std::max(max_rel_err, err);
                }
            }
        }

        // residuals() should be consistent with residuals_and_jacobians()
        std::array<double, 3> r_sq;
        res.residuals(track, &r_sq);
        for (int j = 0; j < 3; ++j)
            max_rel_err = std::max(max_rel_err, std::abs(r_sq[j] - r.row(j).squaredNorm()));
    }
    std::cout << "line jacobian max error: " << max_rel_err << "\n";
    return max_rel_err > 1e-6;
}

// Convergence of the refinement for different perturbations, optionally removing tracks where two back-projected
// planes are close to each other (ill-conditioned line transfer)
static int test_refinement_basin(double rot_noise, double t_noise, double min_plane_angle_deg, bool sampson) {
    int converged = 0;
    const int trials = 100;
    for (int trial = 0; trial < trials; ++trial) {
        Scene sc = random_scene();
        const double scale = 1.0 / sc.t2.norm();
        ThreeViewCameraPose gt(CameraPose(sc.R2, Eigen::Vector3d(sc.t2 * scale)),
                               CameraPose(sc.R3, Eigen::Vector3d(sc.t3 * scale)));
        const Eigen::Matrix3d R[3] = {Eigen::Matrix3d::Identity(), sc.R2, sc.R3};
        std::vector<Line2D> l1, l2, l3;
        while (l1.size() < 50) {
            Line3D L[3];
            Line2D l[3];
            random_line(sc, L, l);
            double min_angle = 180;
            Eigen::Vector3d n[3];
            for (int k = 0; k < 3; ++k)
                n[k] = (R[k].transpose() * l[k].x1.homogeneous().cross(l[k].x2.homogeneous())).normalized();
            for (int a = 0; a < 3; ++a)
                for (int b = a + 1; b < 3; ++b)
                    min_angle = std::min(min_angle, std::acos(std::min(1.0, std::abs(n[a].dot(n[b])))) * 180 / M_PI);
            if (min_angle < min_plane_angle_deg)
                continue;
            l1.push_back(l[0]);
            l2.push_back(l[1]);
            l3.push_back(l[2]);
        }
        std::vector<ThreeViewLineTrack> tracks = make_line_tracks(l1, l2, l3);
        ThreeViewCameraPose pose(
            CameraPose(Eigen::Matrix3d(sc.R2 * rand_rot(rot_noise)),
                       Eigen::Vector3d(sc.t2 * scale + t_noise * randv()).normalized()),
            CameraPose(Eigen::Matrix3d(sc.R3 * rand_rot(rot_noise)), Eigen::Vector3d(sc.t3 * scale + t_noise * randv())));
        TrivialLoss loss(1.0);
        UniformWeightVector w;
        ThreeViewLineJacobianAccumulator<TrivialLoss> accum(tracks, loss, w, 1.0, 1.0, sampson);
        BundleOptions opt;
        opt.max_iterations = 200;
        lm_impl<decltype(accum)>(accum, &pose, opt);
        double err = std::max({rot_err(pose.pose12.R(), gt.pose12.R()), rot_err(pose.pose13.R(), gt.pose13.R()),
                               (pose.pose12.t / pose.pose12.t.norm() - gt.pose12.t).norm(), (pose.pose13.t / pose.pose12.t.norm() - gt.pose13.t).norm()});
        converged += err < 1e-6;
    }
    std::cout << "basin (sampson " << sampson << ") rot " << rot_noise << " t " << t_noise << " min plane angle " << min_plane_angle_deg
              << ": converged " << converged << "/" << trials << "\n";
    return converged;
}

int main() {
    int fails = 0;
    fails += test_solvers();
    for (bool sampson : {false, true})
        fails += test_jacobian(sampson);
    // Sampson: converges also with ill-conditioned tracks
    fails += test_refinement_basin(0.02, 0.05, 0.0, true) < 95;
    // transfer distance: ill-conditioned tracks (near-coinciding planes) shrink the basin
    fails += test_refinement_basin(0.002, 0.005, 2.0, false) < 95;
    for (bool sampson : {false, true})
        for (double ang : {0.0, 2.0, 5.0})
            test_refinement_basin(0.02, 0.05, ang, sampson);
    std::cout << (fails == 0 ? "ALL PASSED" : "FAILED") << std::endl;
    return fails;
}
