#include "threeview_relpose.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/threeview_2sift.h"

#include <algorithm>
#include <cmath>

namespace poselib {

namespace {
template <typename T> std::vector<T> select(const std::vector<T> &x, const std::vector<char> &inliers) {
    std::vector<T> out;
    if (x.empty())
        return out;
    for (size_t k = 0; k < inliers.size(); ++k)
        if (inliers[k])
            out.push_back(x[k]);
    return out;
}

// Depth in view 1 of the point seen at x1 and x2 from the epipolar geometry, x2 x (z R x1 + t) = 0
double triangulate_depth(const CameraPose &pose, const Eigen::Vector3d &x1, const Eigen::Vector3d &x2) {
    const Eigen::Vector3d a = x2.cross(pose.rotate(x1));
    return -a.dot(x2.cross(pose.t)) / a.squaredNorm();
}

double median(std::vector<double> &v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

double sampson_sq(const Eigen::Matrix3d &E, const Point2D &a, const Point2D &b) {
    const Eigen::Vector3d Ea = E * a.homogeneous();
    const double C = Ea(0) * b(0) + Ea(1) * b(1) + Ea(2);
    const double denom = Ea.head<2>().squaredNorm() + (E.block<3, 2>(0, 0).transpose() * b.homogeneous()).squaredNorm();
    return C * C / denom;
}

// MSAC score of the pairs 12, 13, 23 and the number of points with all three below the threshold
double threeview_sampson_score(const ThreeViewGeometry &model, const std::vector<Point2D> &x1,
                               const std::vector<Point2D> &x2, const std::vector<Point2D> &x3, double sq_threshold,
                               size_t *inlier_count, std::vector<char> *inliers) {
    Eigen::Matrix3d E12, E13, E23;
    essential_from_motion(model.pose12, &E12);
    essential_from_motion(model.pose13, &E13);
    essential_from_motion(model.pose23(), &E23);
    double score = 0.0;
    *inlier_count = 0;
    if (inliers)
        inliers->resize(x1.size());
    for (size_t k = 0; k < x1.size(); ++k) {
        const double r12 = sampson_sq(E12, x1[k], x2[k]), r13 = sampson_sq(E13, x1[k], x3[k]),
                     r23 = sampson_sq(E23, x2[k], x3[k]);
        // written so that nan (degenerate E) counts as an outlier
        const bool inlier = r12 < sq_threshold && r13 < sq_threshold && r23 < sq_threshold;
        score += (r12 < sq_threshold ? r12 : sq_threshold) + (r13 < sq_threshold ? r13 : sq_threshold) +
                 (r23 < sq_threshold ? r23 : sq_threshold);
        *inlier_count += inlier;
        if (inliers)
            (*inliers)[k] = inlier;
    }
    return score;
}
} // namespace

ThreeViewEstimator::ThreeViewEstimator(const ThreeViewOptions &opt, const std::vector<Point2D> &x1,
                                       const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                                       const std::vector<double> &d1, const std::vector<double> &d2,
                                       const std::vector<double> &d3, const std::vector<Line3D> &L1,
                                       const std::vector<Line3D> &L2, const std::vector<Line3D> &L3,
                                       const ThreeViewFeatures &features)
    : sample_sz(opt.threeview_solver == 0 ? 5 : (opt.threeview_solver == 1 ? 3 : 2)), num_data(x1.size()), opt(opt),
      x1(x1), x2(x2), x3(x3), d1(d1), d2(d2), d3(d3), lines(make_threeview_lines(L1, L2, L3)), features(features),
      sq_threshold(opt.max_errors[1] * opt.max_errors[1]),
      line_scale(opt.max_errors[1] / std::sin(opt.max_line_angle * M_PI / 180.0)),
      scale_reproj(opt.weight_reproj > 0.0 && opt.max_errors[0] > 0.0
                       ? opt.weight_reproj * sq_threshold / (opt.max_errors[0] * opt.max_errors[0])
                       : 0.0) {
    rng = opt.ransac.seed;
    x1s.resize(sample_sz);
    x2s.resize(sample_sz);
    x3s.resize(3);
    sample.resize(sample_sz);
}

void ThreeViewEstimator::generate_models(std::vector<ThreeViewGeometry> *models) {
    models->clear();
    if (opt.threeview_solver == 2) {
        generate_models_2sift(models);
        return;
    }
    draw_sample(sample_sz, num_data, &sample, rng);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
    }
    for (size_t k = 0; k < 3; ++k)
        x3s[k] = x3[sample[k]].homogeneous().normalized();
    if (opt.threeview_solver == 0)
        generate_models_5pt(models);
    else
        generate_models_repose(models);
}

void ThreeViewEstimator::generate_models_5pt(std::vector<ThreeViewGeometry> *models) {
    std::vector<Eigen::Vector3d> x1n(5), x2n(5), X(3);
    for (size_t k = 0; k < 5; ++k) {
        x1n[k] = x1s[k].normalized();
        x2n[k] = x2s[k].normalized();
    }
    CameraPoseVector poses12, poses13;
    relpose_5pt(x1n, x2n, &poses12);
    for (const CameraPose &pose12 : poses12) {
        bool valid = true;
        for (size_t k = 0; k < 3 && valid; ++k) {
            const double z = triangulate_depth(pose12, x1s[k], x2s[k]);
            valid = z > 0.0;
            X[k] = z * x1s[k];
        }
        if (!valid)
            continue;
        poses13.clear();
        p3p(x3s, X, &poses13);
        for (const CameraPose &pose13 : poses13)
            models->emplace_back(pose12, pose13);
    }
}

void ThreeViewEstimator::generate_models_repose(std::vector<ThreeViewGeometry> *models) {
    std::vector<Eigen::Vector3d> X(3), x2n(3);
    for (size_t k = 0; k < 3; ++k) {
        X[k] = d1[sample[k]] * x1s[k];
        x2n[k] = x2s[k].normalized();
    }
    CameraPoseVector poses12, poses13;
    p3p(x2n, X, &poses12);
    p3p(x3s, X, &poses13);
    // The scale of the depths of a view from the depth of the first point in that view
    const double z2 = d2[sample[0]], z3 = d3[sample[0]];
    for (const CameraPose &pose12 : poses12) {
        const double scale2 = pose12.apply(X[0])(2) / z2;
        for (const CameraPose &pose13 : poses13)
            models->emplace_back(pose12, pose13, scale2, pose13.apply(X[0])(2) / z3);
    }
}

void ThreeViewEstimator::generate_models_2sift(std::vector<ThreeViewGeometry> *models) {
    draw_sample(2, features.x[0].size(), &sample, rng);
    std::vector<Eigen::Vector3d> x[3], l[3];
    for (int v = 0; v < 3; ++v) {
        for (size_t k : sample) {
            x[v].push_back(features.x[v][k].homogeneous());
            l[v].push_back(features.l[v][k].x1.homogeneous().cross(features.l[v][k].x2.homogeneous()));
        }
    }
    // the same scale of a feature in all views
    threeview_2sift(x[0], x[1], x[2], l[0], l[1], l[2], 1.0, 1.0, 1.0, models);
}

double ThreeViewEstimator::score_model(const ThreeViewGeometry &model, size_t *inlier_count) const {
    double score = threeview_sampson_score(model, x1, x2, x3, sq_threshold, inlier_count, nullptr);
    if (opt.line_refine) {
        const Eigen::Matrix3d R12 = model.pose12.R(), R13 = model.pose13.R();
        const double th = sq_threshold;
        score += opt.line_weight *
                 (compute_twoview_line_msac_score(R12, lines.l12, opt.line_residual, line_scale, th) +
                  compute_twoview_line_msac_score(R13, lines.l13, opt.line_residual, line_scale, th) +
                  compute_twoview_line_msac_score(R13 * R12.transpose(), lines.l23, opt.line_residual, line_scale, th));
    }
    return score;
}

size_t ThreeViewEstimator::get_inliers(const ThreeViewGeometry &model, double sq_th, std::vector<char> *inliers) const {
    size_t count;
    threeview_sampson_score(model, x1, x2, x3, sq_th, &count, inliers);
    return count;
}

void ThreeViewEstimator::refine_model(ThreeViewGeometry *model) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_errors[1];
    bundle_opt.max_iterations = 25;
    // Find approximate inliers and bundle over these with a truncated loss
    refine(model, 5 * sq_threshold, bundle_opt);
}

void ThreeViewEstimator::refine(ThreeViewGeometry *model, double sq_th, const BundleOptions &bundle_opt) const {
    std::vector<char> inliers;
    if (get_inliers(*model, sq_th, &inliers) <= 3)
        return;
    const std::vector<Point2D> x1i = select(x1, inliers), x2i = select(x2, inliers), x3i = select(x3, inliers);
    const std::vector<double> d1i = select(d1, inliers), d2i = select(d2, inliers), d3i = select(d3, inliers);

    ThreeViewLines lines_inl;
    if (opt.line_refine) {
        const Eigen::Matrix3d R12 = model->pose12.R(), R13 = model->pose13.R();
        const auto select_lines = [&](const Eigen::Matrix3d &R, const std::vector<TwoViewLine> &l) {
            std::vector<char> line_inliers;
            compute_twoview_line_msac_score(R, l, opt.line_residual, line_scale, sq_th, &line_inliers);
            return select(l, line_inliers);
        };
        lines_inl.l12 = select_lines(R12, lines.l12);
        lines_inl.l13 = select_lines(R13, lines.l13);
        lines_inl.l23 = select_lines(R13 * R12.transpose(), lines.l23);
    }
    if (scale_reproj > 0.0 && opt.threeview_solver != 1)
        fit_depth_scales(x1i, x2i, x3i, d1i, d2i, d3i, model);
    refine_threeview(x1i, x2i, x3i, d1i, d2i, d3i, lines_inl, opt.line_residual, model, scale_reproj,
                     opt.weight_sampson, bundle_opt, opt.line_weight, line_scale);
}

void fit_depth_scales(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                      const std::vector<double> &d1, const std::vector<double> &d2, const std::vector<double> &d3,
                      ThreeViewGeometry *model) {
    // Depths in view 1 from the pair 12, in the current scale of the translations
    std::vector<double> z1(x1.size()), ratios;
    for (size_t k = 0; k < x1.size(); ++k) {
        z1[k] = triangulate_depth(model->pose12, x1[k].homogeneous(), x2[k].homogeneous());
        if (z1[k] > 0.0 && d1[k] > 0.0)
            ratios.push_back(d1[k] / z1[k]);
    }
    if (ratios.empty())
        return;
    const double alpha = median(ratios);
    model->pose12.t *= alpha;
    model->pose13.t *= alpha;

    const auto view_scale = [&](const CameraPose &pose, const std::vector<double> &d, double fallback) {
        std::vector<double> r;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double z = pose.apply(alpha * z1[k] * x1[k].homogeneous())(2);
            if (z1[k] > 0.0 && z > 0.0 && d[k] > 0.0)
                r.push_back(z / d[k]);
        }
        return r.empty() ? fallback : median(r);
    };
    model->scale2 = view_scale(model->pose12, d2, model->scale2);
    model->scale3 = view_scale(model->pose13, d3, model->scale3);
}

} // namespace poselib
