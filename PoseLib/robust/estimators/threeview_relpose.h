#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/optim/threeview.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/types.h"

#include <vector>

namespace poselib {

// Three view relative pose from point tracks, optionally with depths and line tracks with depth. Minimal solver
// (opt.threeview_solver):
//   0 - 5pt + P3P: 5 points give the poses of views 1-2, 3 of them are triangulated and give view 3 with P3P
//   1 - RePoseD: 3 points with depths, P3P from the points X = d1 x1 of view 1 to view 2 and to view 3, all
//       combinations of the solutions, so the translations share the scale of the depths of view 1
//   2 - 2-SIFT: 2 oriented features (threeview_2sift), each with the same scale in all views, i.e. depth ratios 1
// The MSAC score is the sum of the Sampson scores of the pairs 12, 13, 23, with opt.line_refine also of the line
// scores of the three pairs. A point is an inlier if it is an inlier for all three pairs. The LO and the final
// refinement use ThreeViewRefiner with the Sampson errors (opt.weight_sampson), the symmetric reprojection errors
// (opt.weight_reproj) and the lines (opt.line_refine) on the inliers. With 5pt + P3P or 2-SIFT and reprojection errors
// the translations and scales are first fitted to the depths (fit_depth_scales).
//   x1, x2, x3 - points in normalized image coordinates
//   d1, d2, d3 - their depths, may be empty for 5pt + P3P and 2-SIFT without reprojection errors
//   L1, L2, L3 - 3D line segments in the camera frames (depth * ray), arbitrary scale per view
//   features   - oriented features in normalized image coordinates, only sampled by 2-SIFT. They are not scored unless
//                they are also in x1, x2, x3.
class ThreeViewEstimator {
  public:
    ThreeViewEstimator(const ThreeViewOptions &opt, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                       const std::vector<Point2D> &x3, const std::vector<double> &d1, const std::vector<double> &d2,
                       const std::vector<double> &d3, const std::vector<Line3D> &L1, const std::vector<Line3D> &L2,
                       const std::vector<Line3D> &L3, const ThreeViewFeatures &features);

    void generate_models(std::vector<ThreeViewGeometry> *models);
    double score_model(const ThreeViewGeometry &model, size_t *inlier_count) const;
    void refine_model(ThreeViewGeometry *model) const;

    // Refinement on the points and lines with an error below the threshold, used for LO and the final refinement
    void refine(ThreeViewGeometry *model, double sq_threshold, const BundleOptions &bundle_opt) const;
    // Point inliers (all three pairs below the threshold), returns their number
    size_t get_inliers(const ThreeViewGeometry &model, double sq_threshold, std::vector<char> *inliers) const;

    const size_t sample_sz; // number of sampled points
    const size_t num_data;

  private:
    void generate_models_5pt(std::vector<ThreeViewGeometry> *models);
    void generate_models_repose(std::vector<ThreeViewGeometry> *models);
    void generate_models_2sift(std::vector<ThreeViewGeometry> *models);

    const ThreeViewOptions &opt;
    const std::vector<Point2D> &x1, &x2, &x3;
    const std::vector<double> &d1, &d2, &d3;
    const ThreeViewLines lines;
    const ThreeViewFeatures &features;
    const double sq_threshold;
    const double line_scale;
    const double scale_reproj;

    RNG_t rng;
    std::vector<Eigen::Vector3d> x1s, x2s, x3s;
    std::vector<size_t> sample;
};

// Scales the translations so that the depths of view 1 match the triangulated points and sets scale2, scale3, using
// the medians of the depth ratios over the points of x1, x2, x3 with positive depths.
void fit_depth_scales(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                      const std::vector<double> &d1, const std::vector<double> &d2, const std::vector<double> &d3,
                      ThreeViewGeometry *model);

} // namespace poselib
