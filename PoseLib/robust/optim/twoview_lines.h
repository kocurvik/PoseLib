#pragma once

#include "../../camera_pose.h"
#include "../../types.h"
#include "jacobian_accumulator.h"
#include "refiner_base.h"

#include <array>
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
// the two components robustified separately (a bad depth in one view does not discard the other component),
// 4 - reprojection, 5 - plane + reprojection.
// Reprojection (4, 5, MonoDepthTwoViewGeometry only, the line depths must be in the frame of the point depths, e.g.
// from the same depth map): as the RePoseD point reprojection, the endpoints of the line in view 1 are mapped to
// view 2, Z = R ((z + shift1) a) + t, and those of view 2 to view 1, Z = R^T (scale (z + shift2) a - t), and the
// residual is the signed distance of the projection of Z to the line in the other image, n^T Z / (Z_3 |n_12|),
// four separately robustified scalars per line (an endpoint behind the camera is skipped, as for the points). The
// endpoints do not have to correspond along the line. The residuals are multiplied by reproj_scale (sqrt of the
// scale_reproj of the points) to be in the units of the point residuals, and depend on R, t, scale and the shifts:
//   d(n^T Z / (Z_3 c))/dZ = (n - (n^T Z / Z_3) e3)^T / (Z_3 c),  c = |n_12|
//   forward  dZ/dw = -R [X]_x, dZ/dt = I, dZ/dshift1 = R a          (X = (z + shift1) a)
//   backward dZ/dw = [Z]_x, dZ/dt = -R^T, dZ/dscale = R^T (z + shift2) a, dZ/dshift2 = scale R^T a
// With the update R <- R * exp([w]_x), d(R d1) = -R [d1]_x w, so
//   d(direction)/dw = [d2]_x R [d1]_x,  d(n2^T R d1)/dw = -n2^T R [d1]_x,  d(d2^T R n1)/dw = -d2^T R [n1]_x.
// The residuals are multiplied by residual_scale (point threshold / sin(line threshold)) so that the same robust
// loss as for the points can be used, and the cost is multiplied by the line weight.

struct TwoViewLine {
    Eigen::Vector3d d1, d2, n1, n2;
    // endpoint rays (z = 1) and z-depths, for the reprojection residual
    Eigen::Vector3d a1, b1, a2, b2;
    double za1, zb1, za2, zb2;
};

// L1, L2 - 3D line segments in the camera frames (depth * ray)
inline std::vector<TwoViewLine> make_twoview_lines(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2) {
    std::vector<TwoViewLine> lines(L1.size());
    for (size_t k = 0; k < L1.size(); ++k) {
        lines[k].d1 = (L1[k].X2 - L1[k].X1).normalized();
        lines[k].d2 = (L2[k].X2 - L2[k].X1).normalized();
        lines[k].n1 = L1[k].X1.cross(L1[k].X2).normalized();
        lines[k].n2 = L2[k].X1.cross(L2[k].X2).normalized();
        lines[k].za1 = L1[k].X1(2);
        lines[k].zb1 = L1[k].X2(2);
        lines[k].za2 = L2[k].X1(2);
        lines[k].zb2 = L2[k].X2(2);
        lines[k].a1 = L1[k].X1 / lines[k].za1;
        lines[k].b1 = L1[k].X2 / lines[k].zb1;
        lines[k].a2 = L2[k].X1 / lines[k].za2;
        lines[k].b2 = L2[k].X2 / lines[k].zb2;
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
inline bool uses_plane_residual(int line_residual) {
    return line_residual == 1 || line_residual == 2 || line_residual == 3 || line_residual == 5;
}
inline bool uses_reprojection_residual(int line_residual) { return line_residual == 4 || line_residual == 5; }

// Signed distance (normalized image coordinates) of the projection of Z to the image line with normal n
inline double line_point_distance(const Eigen::Vector3d &n, const Eigen::Vector3d &Z) {
    return n.dot(Z) / (Z(2) * n.head<2>().norm());
}

// One of the four endpoint terms of the reprojection residual, see the top of this file
struct LineEndpointTerm {
    Eigen::Vector3d Z; // the endpoint mapped to the other view
    Eigen::Vector3d n; // normal of the line in the other view
    Eigen::Vector3d a; // ray of the endpoint in its own view
    double z;          // depth of the endpoint in its own view (without the shift)
    bool forward;      // view 1 -> view 2
};

inline std::array<LineEndpointTerm, 4> line_endpoint_terms(const MonoDepthTwoViewGeometry &g, const TwoViewLine &l) {
    const Eigen::Matrix3d R = g.pose.R();
    const Eigen::Vector3d &t = g.pose.t;
    std::array<LineEndpointTerm, 4> terms;
    terms[0] = {R * ((l.za1 + g.shift1) * l.a1) + t, l.n2, l.a1, l.za1, true};
    terms[1] = {R * ((l.zb1 + g.shift1) * l.b1) + t, l.n2, l.b1, l.zb1, true};
    terms[2] = {R.transpose() * (g.scale * (l.za2 + g.shift2) * l.a2 - t), l.n1, l.a2, l.za2, false};
    terms[3] = {R.transpose() * (g.scale * (l.zb2 + g.shift2) * l.b2 - t), l.n1, l.b2, l.zb2, false};
    return terms;
}

// MSAC score of the reprojection terms, an endpoint behind the camera costs the threshold. A line is an inlier if all
// four terms are below the threshold.
inline double compute_twoview_line_reprojection_msac_score(const MonoDepthTwoViewGeometry &g,
                                                           const std::vector<TwoViewLine> &lines, double reproj_scale,
                                                           double sq_threshold, std::vector<char> *inliers = nullptr) {
    const double s_sq = reproj_scale * reproj_scale;
    double score = 0.0;
    if (inliers)
        inliers->resize(lines.size());
    for (size_t k = 0; k < lines.size(); ++k) {
        bool inlier = true;
        for (const LineEndpointTerm &term : line_endpoint_terms(g, lines[k])) {
            if (term.Z(2) <= 0) {
                score += sq_threshold;
                inlier = false;
                continue;
            }
            const double r = line_point_distance(term.n, term.Z);
            const double r_sq = s_sq * r * r;
            score += std::min(r_sq, sq_threshold);
            inlier &= r_sq < sq_threshold;
        }
        if (inliers)
            (*inliers)[k] = inlier;
    }
    return score;
}

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
        } else if (uses_plane_residual(line_residual)) {
            const double r_sq = s_sq * line_plane_residual(R, lines[k]).squaredNorm();
            score += std::min(r_sq, sq_threshold);
            inlier &= r_sq < sq_threshold;
        }
        if (inliers)
            (*inliers)[k] = inlier;
    }
    return score;
}

// MSAC score of all line terms of line_residual for the RePoseD model (the rotation-only terms with residual_scale, the
// reprojection terms with reproj_scale), a line is an inlier if it is one for both.
inline double compute_twoview_line_msac_score(const MonoDepthTwoViewGeometry &g, const std::vector<TwoViewLine> &lines,
                                              int line_residual, double residual_scale, double reproj_scale,
                                              double sq_threshold, std::vector<char> *inliers = nullptr) {
    double score = 0.0;
    std::vector<char> rot_inliers, reproj_inliers;
    if (line_residual != 4)
        score += compute_twoview_line_msac_score(g.pose.R(), lines, line_residual, residual_scale, sq_threshold,
                                                 inliers ? &rot_inliers : nullptr);
    if (uses_reprojection_residual(line_residual))
        score += compute_twoview_line_reprojection_msac_score(g, lines, reproj_scale, sq_threshold,
                                                              inliers ? &reproj_inliers : nullptr);
    if (inliers) {
        inliers->assign(lines.size(), 1);
        for (size_t k = 0; k < lines.size(); ++k)
            (*inliers)[k] = (rot_inliers.empty() || rot_inliers[k]) && (reproj_inliers.empty() || reproj_inliers[k]);
    }
    return score;
}

// Reprojection terms of the line refiner, only for MonoDepthTwoViewGeometry (no-ops for CameraPose)
template <typename Accumulator>
void add_line_reprojection_residuals(Accumulator &, const CameraPose &, const TwoViewLine &, double, double) {}

template <typename Accumulator>
void add_line_reprojection_residuals(Accumulator &acc, const MonoDepthTwoViewGeometry &g, const TwoViewLine &l,
                                     double reproj_scale, double weight) {
    for (const LineEndpointTerm &term : line_endpoint_terms(g, l))
        if (term.Z(2) > 0)
            acc.add_residual(reproj_scale * line_point_distance(term.n, term.Z), weight);
}

template <int NumParams, typename Accumulator>
void add_line_reprojection_jacobians(Accumulator &, const CameraPose &, const TwoViewLine &, double, double) {}

template <int NumParams, typename Accumulator>
void add_line_reprojection_jacobians(Accumulator &acc, const MonoDepthTwoViewGeometry &g, const TwoViewLine &l,
                                     double reproj_scale, double weight) {
    const Eigen::Matrix3d R = g.pose.R();
    for (const LineEndpointTerm &term : line_endpoint_terms(g, l)) {
        const Eigen::Vector3d &Z = term.Z;
        if (Z(2) <= 0)
            continue;
        const double inv_z = 1.0 / Z(2), c = term.n.head<2>().norm();
        const double nZ = term.n.dot(Z);
        Eigen::RowVector3d dr = term.n.transpose();
        dr(2) -= nZ * inv_z;
        dr *= inv_z / c;
        Eigen::Matrix<double, 1, NumParams> J = Eigen::Matrix<double, 1, NumParams>::Zero();
        if (term.forward) {
            const Eigen::Vector3d X = (term.z + g.shift1) * term.a;
            J.template block<1, 3>(0, 0) = -dr * R * twoview_skew(X);
            J.template block<1, 3>(0, 3) = dr;
            if constexpr (NumParams == 9)
                J(0, 7) = (dr * R * term.a).value();
        } else {
            J.template block<1, 3>(0, 0) = dr * twoview_skew(Z);
            J.template block<1, 3>(0, 3) = -dr * R.transpose();
            J(0, 6) = (term.z + g.shift2) * (dr * R.transpose() * term.a).value();
            if constexpr (NumParams == 9)
                J(0, 8) = g.scale * (dr * R.transpose() * term.a).value();
        }
        const Eigen::Matrix<double, 1, NumParams> Js = reproj_scale * J;
        acc.add_jacobian(reproj_scale * nZ * inv_z / c, Js, weight);
    }
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
                       double residual_scale, const RefinerBase<Model, Accumulator> *step_refiner,
                       double reproj_scale = 0.0)
        : lines(lines), line_residual(line_residual), line_weight(line_weight), residual_scale(residual_scale),
          reproj_scale(reproj_scale), step_refiner(step_refiner) {
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
            } else if (uses_plane_residual(line_residual)) {
                const Eigen::Vector2d r = residual_scale * line_plane_residual(R, l);
                acc.add_residual(r, line_weight);
            }
            if (uses_reprojection_residual(line_residual))
                add_line_reprojection_residuals(acc, model, l, reproj_scale, line_weight);
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
            if (uses_reprojection_residual(line_residual))
                add_line_reprojection_jacobians<NumParams>(acc, model, l, reproj_scale, line_weight);
            if (!uses_plane_residual(line_residual))
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
    const double reproj_scale;
    const RefinerBase<Model, Accumulator> *step_refiner;
};

} // namespace poselib
