#include "threeview_lines.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/lm_impl.h"
#include "PoseLib/robust/robust_loss.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_upright_3pt.h"
#include "PoseLib/solvers/relpose_upright_3pt_ding.h"
#include "PoseLib/solvers/threeview_lines.h"

namespace poselib {

namespace {
// Minimal number of tracks for local optimization (each line track gives 2 constraints on the 11 dof pose)
constexpr size_t min_lo_line_tracks = 6;

template <typename T> std::vector<T> select_points(const std::vector<T> &x, const std::vector<char> &inliers) {
    std::vector<T> out;
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
    const size_t num_sample_pts = sample_sz - num_sample_lines;
    draw_sample(num_sample_pts, x1.size(), &sample, rng);

    if (opt.hybrid_solver == 2) {
        draw_sample(2, tracks.size(), &line_sample, rng);
        for (size_t k = 0; k < 2; ++k) {
            X1s[k] = X1[sample[k]];
            X2s[k] = X2[sample[k]];
            X3s[k] = X3[sample[k]];
            for (int j = 0; j < 3; ++j)
                ls[j][k] = tracks[line_sample[k]].l[j];
        }
        threeview_2p2l(X1s, X2s, X3s, ls[0], ls[1], ls[2], models);
        return;
    }

    if (opt.hybrid_solver == 3) {
        // 5p3v as in ThreeViewRelativePoseEstimator: 5pt for views 1-2, p3p for view 3 from 3 triangulated points
        models->clear();
        for (size_t k = 0; k < 5; ++k) {
            X1s[k] = x1[sample[k]].homogeneous().normalized();
            X2s[k] = x2[sample[k]].homogeneous().normalized();
        }
        std::vector<CameraPose> models12, models13;
        relpose_5pt(X1s, X2s, &models12);
        std::vector<Point3D> X(3), x3s(3);
        for (size_t k = 0; k < 3; ++k)
            x3s[k] = x3[sample[k]].homogeneous().normalized();
        for (const CameraPose &pose12 : models12) {
            for (size_t k = 0; k < 3; ++k)
                X[k] = triangulate(pose12, x1[sample[k]].homogeneous(), x2[sample[k]].homogeneous());
            p3p(x3s, X, &models13);
            for (const CameraPose &pose13 : models13)
                models->emplace_back(pose12, pose13);
        }
        return;
    }

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
        if (opt.threeview_line_error == 1)
            score += opt.line_weight * compute_threeview_angular_line_msac_score(three_view_pose, angular_lines,
                                                                                 opt.line_residual, line_scale,
                                                                                 sq_th, &line_inliers);
        else
            score += opt.line_weight * compute_threeview_line_msac_score(three_view_pose, tracks,
                                                                         opt.max_line_error * opt.max_line_error,
                                                                         &line_inliers, nullptr, opt.line_sampson);
    }
    // with 5p3v the lines are not sampled and not counted in the inlier ratio (num_data = number of points)
    *inlier_count = opt.hybrid_solver == 3 ? point_inliers : point_inliers + line_inliers;
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
    if (opt.threeview_line_error == 1)
        compute_threeview_angular_line_msac_score(*three_view_pose, angular_lines, opt.line_residual, line_scale,
                                                  5 * (opt.max_epipolar_error * opt.max_epipolar_error), &num_lines,
                                                  &line_inliers);
    else
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
    TruncatedLoss loss_fn(bundle_opt.loss_scale);
    if (opt.threeview_line_error == 1) {
        const ThreeViewAngularLines inlier_lines = select_angular_lines(angular_lines, line_inliers);
        ThreeViewHybridAngularJacobianAccumulator<TruncatedLoss> accum(x1_inl, x2_inl, x3_inl, inlier_lines,
                                                                       opt.line_residual, loss_fn, opt.line_weight,
                                                                       line_scale);
        lm_impl<decltype(accum)>(accum, three_view_pose, bundle_opt);
        return;
    }

    std::vector<ThreeViewLineTrack> inlier_tracks;
    inlier_tracks.reserve(num_lines);
    for (size_t i = 0; i < tracks.size(); ++i)
        if (line_inliers[i])
            inlier_tracks.push_back(tracks[i]);

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
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    if (opt.twoview_solver == 2)
        relpose_5pt(x1s, x2s, models);
    else if (opt.twoview_solver == 1)
        generate_models_2l2p(models);
    else
        generate_models_1l3p(models);

    if (opt.line_check_angle > 0)
        direction_check(models);
}

void RelativePoseLineDirectionEstimator::direction_check(std::vector<CameraPose> *models) {
    const double min_cos = std::cos(opt.line_check_angle * M_PI / 180.0);
    std::vector<size_t> idx(opt.line_check_num);
    for (size_t &i : idx)
        i = random_int(rng) % L1.size();

    size_t num_valid = 0;
    for (size_t m = 0; m < models->size(); ++m) {
        const Eigen::Matrix3d R = (*models)[m].R();
        bool valid = true;
        for (size_t i : idx) {
            const Eigen::Vector3d d1 = (L1[i].X2 - L1[i].X1).normalized();
            const Eigen::Vector3d d2 = (L2[i].X2 - L2[i].X1).normalized();
            if (std::abs((R * d1).dot(d2)) < min_cos) {
                valid = false;
                break;
            }
        }
        if (valid)
            (*models)[num_valid++] = (*models)[m];
    }
    models->resize(num_valid);
}

void RelativePoseLineDirectionEstimator::generate_models_2l2p(std::vector<CameraPose> *models) {
    // cos(10 deg), nearly parallel lines do not determine the rotation
    constexpr double max_cos = 0.98480775301;
    draw_sample(2, L1.size(), &line_sample, rng);
    std::vector<Eigen::Vector3d> d1(2), d2(2);
    for (int k = 0; k < 2; ++k) {
        d1[k] = (L1[line_sample[k]].X2 - L1[line_sample[k]].X1).normalized();
        d2[k] = (L2[line_sample[k]].X2 - L2[line_sample[k]].X1).normalized();
    }
    if (std::abs(d1[0].dot(d1[1])) > max_cos || std::abs(d2[0].dot(d2[1])) > max_cos)
        return;
    relpose_2l2p(x1s, x2s, d1, d2, models);
}

void RelativePoseLineDirectionEstimator::generate_models_1l3p(std::vector<CameraPose> *models) {
    const size_t line_idx = random_int(rng) % L1.size();
    const Eigen::Vector3d d1 = (L1[line_idx].X2 - L1[line_idx].X1).normalized();
    const Eigen::Vector3d d2 = (L2[line_idx].X2 - L2[line_idx].X1).normalized();

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
    const double sq_threshold = opt.max_epipolar_error * opt.max_epipolar_error;
    double score = compute_sampson_msac_score(pose, x1, x2, sq_threshold, inlier_count);
    if (opt.twoview_line_refine)
        score += opt.line_weight *
                 compute_twoview_line_msac_score(pose.R(), lines, opt.line_residual, line_scale, sq_threshold);
    return score;
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
    if (!opt.twoview_line_refine) {
        refine_relpose(select_points(x1, inliers), select_points(x2, inliers), pose, bundle_opt);
        return;
    }
    std::vector<char> line_inliers;
    compute_twoview_line_msac_score(pose->R(), lines, opt.line_residual, line_scale,
                                    5 * (opt.max_epipolar_error * opt.max_epipolar_error), &line_inliers);
    refine_relpose_lines(select_points(x1, inliers), select_points(x2, inliers), select_points(lines, line_inliers),
                         opt.line_residual, pose, bundle_opt, opt.line_weight, line_scale);
}

} // namespace poselib
