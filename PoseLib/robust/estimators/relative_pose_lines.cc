#include "relative_pose_lines.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/solvers/relpose_2l2p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_upright_3pt.h"
#include "PoseLib/solvers/relpose_upright_3pt_qep.h"

namespace poselib {

namespace {
template <typename T> std::vector<T> select_points(const std::vector<T> &x, const std::vector<char> &inliers) {
    std::vector<T> out;
    for (size_t k = 0; k < x.size(); ++k)
        if (inliers[k])
            out.push_back(x[k]);
    return out;
}
} // namespace

void RelativePoseLineEstimator::generate_models(std::vector<CameraPose> *models) {
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

void RelativePoseLineEstimator::direction_check(std::vector<CameraPose> *models) {
    const double min_cos = std::cos(opt.line_check_angle * M_PI / 180.0);
    std::vector<size_t> idx(opt.line_check_num);
    for (size_t &i : idx)
        i = random_int(rng) % L1.size();

    size_t num_valid = 0;
    for (size_t m = 0; m < models->size(); ++m) {
        const Eigen::Matrix3d R = (*models)[m].R();
        bool valid = true;
        for (size_t i : idx) {
            if (std::abs((R * lines[i].d1).dot(lines[i].d2)) < min_cos) {
                valid = false;
                break;
            }
        }
        if (valid)
            (*models)[num_valid++] = (*models)[m];
    }
    models->resize(num_valid);
}

void RelativePoseLineEstimator::generate_models_2l2p(std::vector<CameraPose> *models) {
    // cos(10 deg), nearly parallel lines do not determine the rotation
    constexpr double max_cos = 0.98480775301;
    draw_sample(2, L1.size(), &line_sample, rng);
    std::vector<Eigen::Vector3d> d1(2), d2(2);
    for (int k = 0; k < 2; ++k) {
        d1[k] = lines[line_sample[k]].d1;
        d2[k] = lines[line_sample[k]].d2;
    }
    if (std::abs(d1[0].dot(d1[1])) > max_cos || std::abs(d2[0].dot(d2[1])) > max_cos)
        return;
    relpose_2l2p(x1s, x2s, d1, d2, models);
}

void RelativePoseLineEstimator::generate_models_1l3p(std::vector<CameraPose> *models) {
    const size_t line_idx = random_int(rng) % L1.size();
    const Eigen::Vector3d &d1 = lines[line_idx].d1;
    const Eigen::Vector3d &d2 = lines[line_idx].d2;

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
            relpose_upright_3pt_qep(x1u, x2u, &poses);
            for (CameraPose &p : poses)
                p = CameraPose(Eigen::Matrix3d(Rc2.transpose() * p.R() * Rc1), Eigen::Vector3d(Rc2.transpose() * p.t));
        } else {
            relpose_upright_3pt(x1s, x2s, d1, Eigen::Vector3d(sign * d2), &poses);
        }
        for (const CameraPose &p : poses)
            if (check_cheirality(p, x1s, x2s))
                models->push_back(p);
    }
}

double RelativePoseLineEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    const double sq_threshold = opt.max_error * opt.max_error;
    double score = compute_sampson_msac_score(pose, x1, x2, sq_threshold, inlier_count);
    if (opt.line_refine)
        score += opt.line_weight *
                 compute_twoview_line_msac_score(pose.R(), lines, opt.line_residual, line_scale, sq_threshold);
    return score;
}

void RelativePoseLineEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_error * opt.max_error), &inliers);
    if (num_inl <= 5)
        return;
    if (!opt.line_refine) {
        refine_relpose(select_points(x1, inliers), select_points(x2, inliers), pose, bundle_opt);
        return;
    }
    std::vector<char> line_inliers;
    compute_twoview_line_msac_score(pose->R(), lines, opt.line_residual, line_scale,
                                    5 * (opt.max_error * opt.max_error), &line_inliers);
    refine_relpose_lines(select_points(x1, inliers), select_points(x2, inliers), select_points(lines, line_inliers),
                         opt.line_residual, pose, bundle_opt, opt.line_weight, line_scale);
}

double MonoDepthRelativePoseLineEstimator::score_model(const MonoDepthTwoViewGeometry &model,
                                                       size_t *inlier_count) const {
    double score = point_estimator.score_model(model, inlier_count);
    if (opt.line_refine)
        score += opt.line_weight * compute_twoview_line_msac_score(model.pose.R(), lines, opt.line_residual,
                                                                   line_scale, opt.max_errors[1] * opt.max_errors[1]);
    return score;
}

void MonoDepthRelativePoseLineEstimator::refine_model(MonoDepthTwoViewGeometry *model) const {
    if (!opt.line_refine) {
        point_estimator.refine_model(model);
        return;
    }
    // As RelativePoseMonoDepthEstimator::refine_model: all points (and lines) with a truncated loss
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.max_iterations = 25;
    bundle_opt.loss_scale = opt.max_errors[1];
    refine_monodepth_relpose_lines(x1, x2, d1, d2, lines, opt.line_residual, model, scale_reproj, opt.weight_sampson,
                                   bundle_opt, opt.estimate_shift, opt.line_weight, line_scale);
}

} // namespace poselib
