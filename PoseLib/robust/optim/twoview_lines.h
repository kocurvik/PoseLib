#pragma once

#include "../../camera_pose.h"
#include "../../types.h"
#include "jacobian_accumulator.h"
#include "refiner_base.h"

#include <vector>

namespace poselib {

// Two view lines with depth
// -------------------------
// A line track with endpoint depths gives its 3D direction d1, d2 (unit) in both camera frames, d2 = +-R d1, and
// the normals n1, n2 (unit) of the planes through the camera centers and the 2D lines (these do not depend on the
// depths). Two residual terms per line, both depending on the rotation only (sines of angles):
//   direction  r = (R d1) x d2                  (3-vector, invariant to the sign of d2, uses both depths)
//   plane      r = (n2^T R d1, n1^T R^T d2)     (R d1 lies in the plane of l2 and R^T d2 in the plane of l1,
//                                                each component uses the depth in one view only)
// line_residual: 0 - direction, 1 - plane, 2 - both (two separately robustified terms per line), 3 - plane with
// the two components robustified separately (a bad depth in one view does not discard the other component).
// With the update R <- R * exp([w]_x), d(R d1) = -R [d1]_x w, so
//   d(direction)/dw = [d2]_x R [d1]_x,  d(n2^T R d1)/dw = -n2^T R [d1]_x,  d(d2^T R n1)/dw = -d2^T R [n1]_x.
// The residuals are multiplied by residual_scale (point threshold / sin(line threshold)) so that the same robust
// loss as for the points can be used, and the cost is multiplied by the line weight.

struct TwoViewLine {
    Eigen::Vector3d d1, d2, n1, n2;
};

// L1, L2 - 3D line segments in the camera frames (depth * ray)
inline std::vector<TwoViewLine> make_twoview_lines(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2) {
    std::vector<TwoViewLine> lines(L1.size());
    for (size_t k = 0; k < L1.size(); ++k) {
        lines[k].d1 = (L1[k].X2 - L1[k].X1).normalized();
        lines[k].d2 = (L2[k].X2 - L2[k].X1).normalized();
        lines[k].n1 = L1[k].X1.cross(L1[k].X2).normalized();
        lines[k].n2 = L2[k].X1.cross(L2[k].X2).normalized();
    }
    return lines;
}

inline Eigen::Matrix3d twoview_skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return S;
}

inline Eigen::Vector3d line_direction_residual(const Eigen::Matrix3d &R, const TwoViewLine &l) {
    return (R * l.d1).cross(l.d2);
}

inline Eigen::Vector2d line_plane_residual(const Eigen::Matrix3d &R, const TwoViewLine &l) {
    return Eigen::Vector2d(l.n2.dot(R * l.d1), l.d2.dot(R * l.n1));
}

inline bool uses_direction_residual(int line_residual) { return line_residual == 0 || line_residual == 2; }

// MSAC score of the lines in the units of the point score: sum over lines and terms of min(s^2 |r|^2, sq_threshold).
// A line is an inlier if all its terms are below the threshold (line_residual 3: if any plane component is).
inline double compute_twoview_line_msac_score(const Eigen::Matrix3d &R, const std::vector<TwoViewLine> &lines,
                                              int line_residual, double residual_scale, double sq_threshold,
                                              std::vector<char> *inliers = nullptr) {
    const double s_sq = residual_scale * residual_scale;
    double score = 0.0;
    if (inliers)
        inliers->resize(lines.size());
    for (size_t k = 0; k < lines.size(); ++k) {
        bool inlier = true;
        if (uses_direction_residual(line_residual)) {
            const double r_sq = s_sq * line_direction_residual(R, lines[k]).squaredNorm();
            score += std::min(r_sq, sq_threshold);
            inlier &= r_sq < sq_threshold;
        }
        if (line_residual == 3) {
            const Eigen::Vector2d r = line_plane_residual(R, lines[k]);
            const double r0_sq = s_sq * r(0) * r(0), r1_sq = s_sq * r(1) * r(1);
            score += std::min(r0_sq, sq_threshold) + std::min(r1_sq, sq_threshold);
            inlier = r0_sq < sq_threshold || r1_sq < sq_threshold;
        } else if (line_residual != 0) {
            const double r_sq = s_sq * line_plane_residual(R, lines[k]).squaredNorm();
            score += std::min(r_sq, sq_threshold);
            inlier &= r_sq < sq_threshold;
        }
        if (inliers)
            (*inliers)[k] = inlier;
    }
    return score;
}

inline Eigen::Matrix3d model_rotation(const CameraPose &pose) { return pose.R(); }
inline Eigen::Matrix3d model_rotation(const MonoDepthTwoViewGeometry &geometry) { return geometry.pose.R(); }

// Line residuals for any two view model whose first three update parameters are the rotation R <- R * exp([w]_x)
// (PinholeRelativePoseRefiner, MonoDepthRelPoseRefiner). Meant to be combined with the point refiner in a
// HybridRefiner (registered after it, so that the point refiner's step is used); step() delegates to step_refiner.
//   NumParams - number of parameters of the model update (5 for CameraPose, 7 / 9 for MonoDepthTwoViewGeometry)
template <typename Model, int NumParams, typename Accumulator = NormalAccumulator>
class TwoViewLineRefiner : public RefinerBase<Model, Accumulator> {
  public:
    TwoViewLineRefiner(const std::vector<TwoViewLine> &lines, int line_residual, double line_weight,
                       double residual_scale, const RefinerBase<Model, Accumulator> *step_refiner)
        : lines(lines), line_residual(line_residual), line_weight(line_weight), residual_scale(residual_scale),
          step_refiner(step_refiner) {
        this->num_params = NumParams;
    }

    double compute_residual(Accumulator &acc, const Model &model) {
        const Eigen::Matrix3d R = model_rotation(model);
        for (const TwoViewLine &l : lines) {
            if (uses_direction_residual(line_residual)) {
                const Eigen::Vector3d r = residual_scale * line_direction_residual(R, l);
                acc.add_residual(r, line_weight);
            }
            if (line_residual == 3) {
                const Eigen::Vector2d r = residual_scale * line_plane_residual(R, l);
                acc.add_residual(r(0), line_weight);
                acc.add_residual(r(1), line_weight);
            } else if (line_residual != 0) {
                const Eigen::Vector2d r = residual_scale * line_plane_residual(R, l);
                acc.add_residual(r, line_weight);
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Model &model) {
        const Eigen::Matrix3d R = model_rotation(model);
        for (const TwoViewLine &l : lines) {
            if (uses_direction_residual(line_residual)) {
                const Eigen::Vector3d r = residual_scale * line_direction_residual(R, l);
                Eigen::Matrix<double, 3, NumParams> J = Eigen::Matrix<double, 3, NumParams>::Zero();
                J.template leftCols<3>() = residual_scale * twoview_skew(l.d2) * R * twoview_skew(l.d1);
                acc.add_jacobian(r, J, line_weight);
            }
            if (line_residual == 0)
                continue;
            const Eigen::Vector2d r = residual_scale * line_plane_residual(R, l);
            Eigen::Matrix<double, 2, NumParams> J = Eigen::Matrix<double, 2, NumParams>::Zero();
            J.template block<1, 3>(0, 0) = -residual_scale * l.n2.transpose() * R * twoview_skew(l.d1);
            J.template block<1, 3>(1, 0) = -residual_scale * l.d2.transpose() * R * twoview_skew(l.n1);
            if (line_residual == 3) {
                const Eigen::Matrix<double, 1, NumParams> J0 = J.row(0), J1 = J.row(1);
                acc.add_jacobian(r(0), J0, line_weight);
                acc.add_jacobian(r(1), J1, line_weight);
            } else {
                acc.add_jacobian(r, J, line_weight);
            }
        }
    }

    Model step(const Eigen::VectorXd &dp, const Model &model) const { return step_refiner->step(dp, model); }

    typedef Model param_t;
    const std::vector<TwoViewLine> &lines;
    const int line_residual;
    const double line_weight;
    const double residual_scale;
    const RefinerBase<Model, Accumulator> *step_refiner;
};

} // namespace poselib
