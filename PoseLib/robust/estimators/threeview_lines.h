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
class ThreeViewHybridEstimator {
  public:
    ThreeViewHybridEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &x1,
                             const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                             const std::vector<Eigen::Vector3d> &X1, const std::vector<Eigen::Vector3d> &X2,
                             const std::vector<Eigen::Vector3d> &X3, const std::vector<Line2D> &l1,
                             const std::vector<Line2D> &l2, const std::vector<Line2D> &l3,
                             const std::vector<Line3D> &L1, const std::vector<Line3D> &L2,
                             const std::vector<Line3D> &L3)
        : num_data(x1.size() + l1.size()), opt(ransac_opt), x1(x1), x2(x2), x3(x3), X1(X1), X2(X2), X3(X3), L1(L1),
          L2(L2), L3(L3), tracks(make_line_tracks(l1, l2, l3)) {
        rng = opt.seed;
        X1s.resize(2);
        X2s.resize(2);
        X3s.resize(2);
        sample.resize(2);
    }

    void generate_models(std::vector<ThreeViewCameraPose> *models);
    double score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const;
    void refine_model(ThreeViewCameraPose *three_view_pose) const;

    const size_t sample_sz = 3;
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

} // namespace poselib

#endif
