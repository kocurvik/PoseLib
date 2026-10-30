// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_ROBUST_TYPES_H_
#define POSELIB_ROBUST_TYPES_H_

#include "alignment.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    size_t lo_iterations = 25;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_reproj_error = 12.0;  // used for 2D-3D matches
    double max_epipolar_error = 1.0; // used for 2D-2D matches
    unsigned long seed = 0;
    // If we should use PROSAC sampling. Assumes data is sorted
    bool progressive_sampling = false;
    size_t max_prosac_iterations = 100000;
    // Whether we should use real focal length checking: https://arxiv.org/abs/2311.16304
    // Assumes that principal points of both cameras are at origin.
    bool real_focal_check = false;
    size_t sample_sz = 0;
    size_t inner_refine = 0;
    size_t inner_refine_extra = 1;
    bool threeview_check = false;
    bool threeview_check_extra = false;
    double delta = 0.0;
    bool use_hc = false;
    bool use_net = false;
    bool init_net = false;
    bool oracle = false;
    Eigen::Matrix3d gt_E = Eigen::Matrix3d::Zero();
    bool use_affine = false;
    bool early_nonminimal = false;
    bool early_lm = false;
    bool use_homography = false;
    bool use_para = false;
    int use_nister = 0;
    double nister_scale = 1.0;
    // Three view line / hybrid estimators (threeview_lines.h)
    int line_solver = 0;          // 0 - 2 lines, common perpendicular, 1 - 2 lines, planes
    double max_line_error = 1.0;  // threshold for line transfer error (pixels)
    double line_weight = 1.0;     // weight of the line score and residuals in the hybrid estimator
    int hybrid_solver = 0;        // hybrid estimator minimal solver: 0 - 2p1l, 1 - 1l3p, 2 - 2p2l, 3 - 5p3v
    int upright_solver = 0;       // two view relpose with line direction: 0 - Ding et al. CVPR23, 1 - QEP
    int twoview_solver = 0;       // two view relpose with line directions: 0 - 1L3P, 1 - 2L2P, 2 - 5pt
    double line_check_angle = 0;  // two view relpose with line directions: direction check of hypotheses (deg, 0 off)
    size_t line_check_num = 1;    // number of line tracks tested by the direction check
    bool line_sampson = false;    // Sampson-normalized line transfer error instead of transfer distance
    bool twoview_line_refine = false; // two view relpose with line directions: lines also in scoring, LO, refinement
    double max_line_angle = 2.0;  // two view relpose with line directions: line direction threshold (deg)
    int line_residual = 0;        // two view line residual (twoview_line_refine): 0 - direction, 1 - plane, 2 - both,
                                  // 3 - plane with separately robustified components
    int threeview_line_error = 0; // three view hybrid line error in scoring, LO, refinement: 0 - transfer distance
                                  // (max_line_error), 1 - angular residuals of the pairs 12, 13, 23 (line_residual
                                  // 0-2, max_line_angle, uses the line depths)
};

struct RansacStats {
    size_t refinements = 0;
    size_t iterations = 0;
    size_t num_inliers = 0;
    double inlier_ratio = 0;
    double model_score = std::numeric_limits<double>::max();
};

struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY,
        // This is the TR-IRLS scheme from Le and Zach, 3DV 2021
        TRUNCATED_LE_ZACH,
        // Cauchy truncated at loss_scale (upstream PoseLib, recommended for the monodepth relative pose estimators)
        TRUNCATED_CAUCHY
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-10;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
    double min_lambda = 1e-10;
    double max_lambda = 1e10;
    bool verbose = false;
};

struct BundleStats {
    size_t iterations = 0;
    double initial_cost;
    double cost;
    double lambda;
    size_t invalid_steps;
    double step_norm;
    double grad_norm;
};

typedef Eigen::Vector2d Point2D;
typedef Eigen::Vector3d Point3D;

// Used to store pairwise matches for generalized pose estimation
struct PairwiseMatches {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t cam_id1, cam_id2;
    std::vector<Point2D> x1, x2;
};

struct Line2D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line2D() {}
    Line2D(const Eigen::Vector2d &e1, const Eigen::Vector2d &e2) : x1(e1), x2(e2) {}
    Eigen::Vector2d x1, x2;
};
struct Line3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line3D() {}
    Line3D(const Eigen::Vector3d &e1, const Eigen::Vector3d &e2) : X1(e1), X2(e2) {}
    Eigen::Vector3d X1, X2;
};

} // namespace poselib

#endif