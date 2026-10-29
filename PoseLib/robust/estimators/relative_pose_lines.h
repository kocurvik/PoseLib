#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/estimators/relative_pose.h"
#include "PoseLib/robust/optim/twoview_lines.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/types.h"

#include <vector>

namespace poselib {

// Two view relative pose from points and lines with depth (3D line directions in both views).
// Minimal solver (opt.twoview_solver):
//   0 - 1L3P, the upright relative pose problem with 1 line + 3 points (opt.upright_solver: 0 - Ding et al.
//       CVPR 2023, 1 - quadratic eigenvalue problem), both signs of d2 are tried
//   1 - 2L2P, rotation from 2 line directions + translation from 2 points (relpose_2l2p), pairs of lines with
//       an angle below 10 deg in view 1 are skipped
//   2 - 5pt (points only, lines are used only by the direction check and opt.line_refine)
// Direction check (opt.line_check_angle > 0, degrees): each hypothesis is tested against opt.line_check_num random
// line tracks before scoring and dropped unless R d1 is within opt.line_check_angle of +-d2 for all of them.
// Scoring and refinement use the points only (Sampson) unless opt.line_refine is set: then the MSAC score and the
// local optimization also use the lines (opt.line_residual, scaled by line_scale = max_error / sin(max_line_angle),
// weighted by opt.line_weight, see TwoViewLineRefiner). The lines are not counted in the sample size since their
// inlier ratio is not measured, i.e. sample_sz = number of sampled points and num_data = number of points.
//   x1, x2 - points in normalized image coordinates
//   L1, L2 - 3D line segments in the camera frames (depth * ray), arbitrary scale per view
class RelativePoseLineEstimator {
  public:
    RelativePoseLineEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &x1,
                              const std::vector<Point2D> &x2, const std::vector<Line3D> &L1,
                              const std::vector<Line3D> &L2)
        : sample_sz(opt.twoview_solver == 1 ? 2 : (opt.twoview_solver == 2 ? 5 : 3)), num_data(x1.size()), opt(opt),
          x1(x1), x2(x2), L1(L1), L2(L2), lines(make_twoview_lines(L1, L2)),
          line_scale(opt.max_error / std::sin(opt.max_line_angle * M_PI / 180.0)) {
        rng = opt.ransac.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
        line_sample.resize(2);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz; // number of sampled points
    const size_t num_data;

  private:
    void generate_models_1l3p(std::vector<CameraPose> *models);
    void generate_models_2l2p(std::vector<CameraPose> *models);
    void direction_check(std::vector<CameraPose> *models);

    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1, &x2;
    const std::vector<Line3D> &L1, &L2;
    const std::vector<TwoViewLine> lines;
    const double line_scale;

    RNG_t rng;
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample, line_sample;
};

// RePoseD (calibrated relative pose with monocular depths, RelativePoseMonoDepthEstimator) with lines with depth in
// the MSAC score and the local optimization (opt.line_refine, as in RelativePoseLineEstimator; the line residuals are
// in the units of the Sampson error). The minimal samples are those of RelativePoseMonoDepthEstimator.
//   x1, x2 - points in normalized image coordinates, d1, d2 - their depths
//   L1, L2 - 3D line segments in the camera frames (depth * ray), arbitrary scale per view
class MonoDepthRelativePoseLineEstimator {
  public:
    MonoDepthRelativePoseLineEstimator(const MonoDepthRelativePoseOptions &opt, const std::vector<Point2D> &x1,
                                       const std::vector<Point2D> &x2, const std::vector<double> &d1,
                                       const std::vector<double> &d2, const std::vector<Line3D> &L1,
                                       const std::vector<Line3D> &L2)
        : point_estimator(opt, x1, x2, d1, d2), sample_sz(point_estimator.sample_sz), num_data(x1.size()), opt(opt),
          x1(x1), x2(x2), d1(d1), d2(d2), lines(make_twoview_lines(L1, L2)),
          line_scale(opt.max_errors[1] / std::sin(opt.max_line_angle * M_PI / 180.0)),
          scale_reproj(opt.max_errors[0] > 0.0
                           ? (opt.max_errors[1] * opt.max_errors[1]) / (opt.max_errors[0] * opt.max_errors[0])
                           : 0.0) {}

    void generate_models(std::vector<MonoDepthTwoViewGeometry> *models) { point_estimator.generate_models(models); }
    double score_model(const MonoDepthTwoViewGeometry &model, size_t *inlier_count) const;
    void refine_model(MonoDepthTwoViewGeometry *model) const;

  private:
    RelativePoseMonoDepthEstimator point_estimator;

  public:
    const size_t sample_sz;
    const size_t num_data;

  private:
    const MonoDepthRelativePoseOptions &opt;
    const std::vector<Point2D> &x1, &x2;
    const std::vector<double> &d1, &d2;
    const std::vector<TwoViewLine> lines;
    const double line_scale;
    const double scale_reproj;
};

} // namespace poselib
