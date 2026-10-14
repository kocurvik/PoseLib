#include "threeview_lines.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/lm_impl.h"
#include "PoseLib/robust/robust_loss.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/solvers/relpose_upright_3pt.h"
#include "PoseLib/solvers/relpose_upright_3pt_ding.h"
#include "PoseLib/solvers/threeview_lines.h"

namespace poselib {

namespace {
// Minimal number of tracks for local optimization (each line track gives 2 constraints on the 11 dof pose)
constexpr size_t min_lo_line_tracks = 6;

std::vector<Point2D> select_points(const std::vector<Point2D> &x, const std::vector<char> &inliers) {
    std::vector<Point2D> out;
    for (size_t k = 0; k < x.size(); ++k)
        if (inliers[k])
            out.push_back(x[k]);
    return out;
}
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Lines only

void ThreeViewLineEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        L1s[k] = L1[sample[k]];
        L2s[k] = L2[sample[k]];
        L3s[k] = L3[sample[k]];
        l2s[k] = tracks[sample[k]].l[1];
        l3s[k] = tracks[sample[k]].l[2];
    }

    if (opt.line_solver == 1) {
        threeview_2l_planes(L1s, L2s, L3s, l2s, l3s, models);
    } else {
        threeview_2l(L1s, L2s, L3s, models);
    }
}

double ThreeViewLineEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    return compute_threeview_line_msac_score(three_view_pose, tracks, opt.max_line_error * opt.max_line_error,
                                             inlier_count, nullptr, opt.line_sampson);
}

void ThreeViewLineEstimator::refine_model(ThreeViewCameraPose *three_view_pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_line_error;
    bundle_opt.max_iterations = opt.lo_iterations;

    // Find approximate inliers and bundle over these with a truncated loss
    size_t num_inl;
    std::vector<char> inliers;
    compute_threeview_line_msac_score(*three_view_pose, tracks, 5 * opt.max_line_error * opt.max_line_error,
                                      &num_inl, &inliers, opt.line_sampson);
    if (num_inl < min_lo_line_tracks)
        return;

    std::vector<ThreeViewLineTrack> inlier_tracks;
    inlier_tracks.reserve(num_inl);
    for (size_t i = 0; i < tracks.size(); ++i)
        if (inliers[i])
            inlier_tracks.push_back(tracks[i]);

    TruncatedLoss loss_fn(bundle_opt.loss_scale);
    UniformWeightVector weights;
    ThreeViewLineJacobianAccumulator<TruncatedLoss, UniformWeightVector> accum(inlier_tracks, loss_fn, weights, 1.0, 1.0,
                                                                               opt.line_sampson);
    lm_impl<decltype(accum)>(accum, three_view_pose, bundle_opt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Points + lines

void ThreeViewHybridEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    const size_t num_sample_pts = sample_sz - 1;
    draw_sample(num_sample_pts, x1.size(), &sample, rng);
    const size_t line_idx = random_int(rng) % tracks.size();
    const Eigen::Vector3d d1 = L1[line_idx].X2 - L1[line_idx].X1;
    const Eigen::Vector3d d2 = L2[line_idx].X2 - L2[line_idx].X1;
    const Eigen::Vector3d d3 = L3[line_idx].X2 - L3[line_idx].X1;

    if (opt.hybrid_solver == 1) {
        for (size_t k = 0; k < num_sample_pts; ++k) {
            X1s[k] = x1[sample[k]].homogeneous();
            X2s[k] = x2[sample[k]].homogeneous();
            X3s[k] = x3[sample[k]].homogeneous();
        }
        threeview_1l3p(X1s, X2s, X3s, d1, d2, d3, models);
        return;
    }

    for (size_t k = 0; k < num_sample_pts; ++k) {
        X1s[k] = X1[sample[k]];
        X2s[k] = X2[sample[k]];
        X3s[k] = X3[sample[k]];
    }
    threeview_2p1l(X1s, X2s, X3s, d1, d2, d3, models);
}

double ThreeViewHybridEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    const double sq_th = opt.max_epipolar_error * opt.max_epipolar_error;
    size_t inl12, inl13, inl23;
    double score = compute_sampson_msac_score(three_view_pose.pose12, x1, x2, sq_th, &inl12);
    score += compute_sampson_msac_score(three_view_pose.pose13, x1, x3, sq_th, &inl13);
    score += compute_sampson_msac_score(three_view_pose.pose23(), x2, x3, sq_th, &inl23);

    std::vector<char> inliers;
    size_t point_inliers = get_inliers(three_view_pose, x1, x2, x3, sq_th, &inliers);

    size_t line_inliers = 0;
    if (!tracks.empty()) {
        score += opt.line_weight * compute_threeview_line_msac_score(three_view_pose, tracks,
                                                                     opt.max_line_error * opt.max_line_error,
                                                                     &line_inliers, nullptr, opt.line_sampson);
    }
    *inlier_count = point_inliers + line_inliers;
    return score;
}

void ThreeViewHybridEstimator::refine_model(ThreeViewCameraPose *three_view_pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;

    std::vector<char> point_inliers;
    const int num_pts = get_inliers(*three_view_pose, x1, x2, x3,
                                    5 * (opt.max_epipolar_error * opt.max_epipolar_error), &point_inliers);
    size_t num_lines;
    std::vector<char> line_inliers;
    compute_threeview_line_msac_score(*three_view_pose, tracks, 5 * opt.max_line_error * opt.max_line_error,
                                      &num_lines, &line_inliers, opt.line_sampson);

    if (num_pts <= 4 && num_lines < min_lo_line_tracks)
        return;

    std::vector<Point2D> x1_inl, x2_inl, x3_inl;
    x1_inl.reserve(num_pts);
    x2_inl.reserve(num_pts);
    x3_inl.reserve(num_pts);
    for (size_t k = 0; k < x1.size(); ++k) {
        if (point_inliers[k]) {
            x1_inl.push_back(x1[k]);
            x2_inl.push_back(x2[k]);
            x3_inl.push_back(x3[k]);
        }
    }
    std::vector<ThreeViewLineTrack> inlier_tracks;
    inlier_tracks.reserve(num_lines);
    for (size_t i = 0; i < tracks.size(); ++i)
        if (line_inliers[i])
            inlier_tracks.push_back(tracks[i]);

    TruncatedLoss loss_fn(bundle_opt.loss_scale);
    ThreeViewHybridJacobianAccumulator<TruncatedLoss> accum(x1_inl, x2_inl, x3_inl, inlier_tracks, loss_fn,
                                                            opt.line_weight, opt.max_epipolar_error / opt.max_line_error,
                                                            opt.line_sampson);
    lm_impl<decltype(accum)>(accum, three_view_pose, bundle_opt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Two views, points + line direction

void RelativePoseLineDirectionEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    draw_sample(sample_sz, x1.size(), &sample, rng);
    const size_t line_idx = random_int(rng) % L1.size();
    const Eigen::Vector3d d1 = (L1[line_idx].X2 - L1[line_idx].X1).normalized();
    const Eigen::Vector3d d2 = (L2[line_idx].X2 - L2[line_idx].X1).normalized();

    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }

    CameraPoseVector poses;
    for (int sign : {1, -1}) {
        if (opt.upright_solver == 1) {
            const Eigen::Matrix3d Rc1 =
                Eigen::Quaterniond::FromTwoVectors(d1, Eigen::Vector3d::UnitY()).toRotationMatrix();
            const Eigen::Matrix3d Rc2 =
                Eigen::Quaterniond::FromTwoVectors(sign * d2, Eigen::Vector3d::UnitY()).toRotationMatrix();
            std::vector<Eigen::Vector3d> x1u(3), x2u(3);
            for (int i = 0; i < 3; ++i) {
                x1u[i] = Rc1 * x1s[i];
                x2u[i] = Rc2 * x2s[i];
            }
            relpose_upright_3pt(x1u, x2u, &poses);
            for (CameraPose &p : poses)
                p = CameraPose(Eigen::Matrix3d(Rc2.transpose() * p.R() * Rc1), Eigen::Vector3d(Rc2.transpose() * p.t));
        } else {
            relpose_upright_3pt_ding(x1s, x2s, d1, sign * d2, &poses);
        }
        for (const CameraPose &p : poses)
            if (check_cheirality(p, x1s, x2s))
                models->push_back(p);
    }
}

double RelativePoseLineDirectionEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseLineDirectionEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    if (num_inl <= 5)
        return;
    refine_relpose(select_points(x1, inliers), select_points(x2, inliers), pose, bundle_opt);
}

} // namespace poselib
