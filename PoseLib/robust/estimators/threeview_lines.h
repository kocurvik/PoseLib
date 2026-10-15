#ifndef POSELIB_ROBUST_ESTIMATORS_THREEVIEW_LINES_H
#define POSELIB_ROBUST_ESTIMATORS_THREEVIEW_LINES_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/threeview_lines_jacobian.h"
#include "PoseLib/types.h"

namespace poselib {

// Three view relative pose from line tracks with (scale-invariant) depth of their endpoints.
//   l1, l2, l3 - 2D line segments in normalized image coordinates
//   L1, L2, L3 - corresponding 3D segments in the camera frames (depth * ray), arbitrary scale per view
// Minimal solver is selected by opt.line_solver (0 - common perpendicular, 1 - planes).
class ThreeViewLineEstimator {
  public:
    ThreeViewLineEstimator(const RansacOptions &ransac_opt, const std::vector<Line2D> &l1,
                           const std::vector<Line2D> &l2, const std::vector<Line2D> &l3,
                           const std::vector<Line3D> &L1, const std::vector<Line3D> &L2,
                           const std::vector<Line3D> &L3)
        : num_data(l1.size()), opt(ransac_opt), L1(L1), L2(L2), L3(L3), tracks(make_line_tracks(l1, l2, l3)),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        L1s.resize(sample_sz);
        L2s.resize(sample_sz);
        L3s.resize(sample_sz);
        l2s.resize(sample_sz);
        l3s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz = 2;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Line3D> &L1, &L2, &L3;
    const std::vector<ThreeViewLineTrack> tracks;

    RandomSampler sampler;
    std::vector<Line3D> L1s, L2s, L3s;
    std::vector<Eigen::Vector3d> l2s, l3s;
    std::vector<size_t> sample;
};

// Hybrid estimator using points with depth and line tracks with depth. Minimal solver is 2p1l.
//   x1, x2, x3 - points in normalized image coordinates
//   X1, X2, X3 - corresponding 3D points in the camera frames (depth * ray), arbitrary scale per view
//   l*, L*     - see ThreeViewLineEstimator
// The score is the sum of the point (Sampson) and line MSAC scores, the line one weighted by opt.line_weight.
// Minimal solver is selected by opt.hybrid_solver: 0 - 2p1l, 1 - 1l3p (1 line direction + 3 points, the point
// depths X* are not used).
class ThreeViewHybridEstimator {
  public:
    ThreeViewHybridEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &x1,
                             const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                             const std::vector<Eigen::Vector3d> &X1, const std::vector<Eigen::Vector3d> &X2,
                             const std::vector<Eigen::Vector3d> &X3, const std::vector<Line2D> &l1,
                             const std::vector<Line2D> &l2, const std::vector<Line2D> &l3,
                             const std::vector<Line3D> &L1, const std::vector<Line3D> &L2,
                             const std::vector<Line3D> &L3)
        : sample_sz(ransac_opt.hybrid_solver == 1 ? 4 : 3), num_data(x1.size() + l1.size()), opt(ransac_opt), x1(x1),
          x2(x2), x3(x3), X1(X1), X2(X2), X3(X3), L1(L1), L2(L2), L3(L3), tracks(make_line_tracks(l1, l2, l3)) {
        rng = opt.seed;
        X1s.resize(sample_sz - 1);
        X2s.resize(sample_sz - 1);
        X3s.resize(sample_sz - 1);
        sample.resize(sample_sz - 1);
    }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz; // number of points + 1 line
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1, &x2, &x3;
    const std::vector<Eigen::Vector3d> &X1, &X2, &X3;
    const std::vector<Line3D> &L1, &L2, &L3;
    const std::vector<ThreeViewLineTrack> tracks;

    RNG_t rng;
    std::vector<Eigen::Vector3d> X1s, X2s, X3s;
    std::vector<size_t> sample;
};

// Two view relative pose from points and 3D line directions. A line track (with endpoint depths) gives a common
// direction d1, d2 = R d1 in both views. Minimal solver is selected by opt.twoview_solver:
//   0 - 1L3P, the upright relative pose problem with 1 line + 3 points (opt.upright_solver: 0 - Ding et al.
//       CVPR 2023, 1 - quadratic eigenvalue problem), both signs of d2 are tried
//   1 - 2L2P, rotation from 2 line directions + translation from 2 points (relpose_2l2p), pairs of lines with
//       an angle below 10 deg in view 1 are skipped
//   2 - 5pt (points only, lines are used only by the direction check below)
// Direction check (opt.line_check_angle > 0, degrees): each hypothesis is tested against opt.line_check_num random
// line tracks before scoring and dropped unless R d1 is within opt.line_check_angle of +-d2 for all of them.
// Scoring and refinement use the points only (Sampson) unless opt.twoview_line_refine is set: then the MSAC score
// and the local optimization also use the lines (opt.line_residual: direction (R d1) x d2, plane n2^T R d1 and
// n1^T R^T d2, or both; scaled by line_scale = max_epipolar_error / sin(opt.max_line_angle), weighted by
// opt.line_weight, see RelativePoseHybridLineJacobianAccumulator). The lines are not counted in the sample size since their
// inlier ratio is not measured, i.e. sample_sz = number of sampled points and num_data = number of points.
//   x1, x2 - points in normalized image coordinates
//   L1, L2 - 3D line segments in the camera frames (depth * ray), arbitrary scale per view
class RelativePoseLineDirectionEstimator {
  public:
    RelativePoseLineDirectionEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &x1,
                                       const std::vector<Point2D> &x2, const std::vector<Line3D> &L1,
                                       const std::vector<Line3D> &L2)
        : sample_sz(ransac_opt.twoview_solver == 1 ? 2 : (ransac_opt.twoview_solver == 2 ? 5 : 3)), num_data(x1.size()), opt(ransac_opt), x1(x1), x2(x2),
          L1(L1), L2(L2), lines(make_twoview_lines(L1, L2)),
          line_scale(ransac_opt.max_epipolar_error / std::sin(ransac_opt.max_line_angle * M_PI / 180.0)) {
        rng = opt.seed;
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

    const RansacOptions &opt;
    const std::vector<Point2D> &x1, &x2;
    const std::vector<Line3D> &L1, &L2;
    const std::vector<TwoViewLine> lines;
    const double line_scale;

    RNG_t rng;
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample, line_sample;
};

} // namespace poselib

#endif
