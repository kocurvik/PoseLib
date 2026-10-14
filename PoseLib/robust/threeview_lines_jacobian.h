#ifndef POSELIB_ROBUST_THREEVIEW_LINES_JACOBIAN_H_
#define POSELIB_ROBUST_THREEVIEW_LINES_JACOBIAN_H_

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/quaternion.h"
#include "PoseLib/robust/jacobian_impl.h"
#include "PoseLib/types.h"

#include <array>

namespace poselib {

// Line residuals for three views
// ------------------------------
// The world frame is the frame of view 1, i.e. P1 = [I | 0], P2 = [R12 | t12], P3 = [R13 | t13].
// Each 2D line l_k (homogeneous, normalized image coordinates) back-projects to the world plane
//     pi_k = (n_k, c_k),  n_k = R_k^T l_k,  c_k = t_k^T l_k.
// To transfer the line into view j we take the planes of the two remaining views a, b and form the plane
// of their pencil passing through the camera center C_j = -R_j^T t_j:
//     m = e_b n_a - e_a n_b,  e_k = n_k^T C_j + c_k.
// The transferred line in view j is l'_j = R_j m. The residuals measure whether the detected segment endpoints
// p_j, q_j lie on l'_j, normalized by the gradient w.r.t. all observed endpoints (Sampson approximation, see
// ThreeViewLineResidual). For each line track this gives 3 views x 2 endpoints = 6 residuals.
//
// The parametrization of the updates matches ThreeViewRelativePoseJacobianAccumulator (11 parameters):
//     dp(0:3)  - rotation update of pose12 (R12 <- R12 * exp([w]_x))
//     dp(3:5)  - update of t12 in its tangent space (fixed norm)
//     dp(5:8)  - rotation update of pose13
//     dp(8:11) - update of t13

inline Eigen::Matrix<double, 3, 2> threeview_tangent_basis(const Eigen::Vector3d &t) {
    // Same construction as in ThreeViewRelativePoseJacobianAccumulator
    Eigen::Matrix<double, 3, 2> tangent_basis;
    if (std::abs(t.x()) < std::abs(t.y())) {
        if (std::abs(t.x()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitX()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    } else {
        if (std::abs(t.y()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitY()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    }
    tangent_basis.col(1) = tangent_basis.col(0).cross(t).normalized();
    return tangent_basis;
}

inline ThreeViewCameraPose threeview_step(const Eigen::Matrix<double, 11, 1> &dp, const ThreeViewCameraPose &pose,
                                          const Eigen::Matrix<double, 3, 2> &tangent_basis) {
    CameraPose pose12_new, pose13_new;
    pose12_new.q = quat_step_post(pose.pose12.q, dp.block<3, 1>(0, 0));
    pose12_new.t = pose.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);
    pose13_new.q = quat_step_post(pose.pose13.q, dp.block<3, 1>(5, 0));
    pose13_new.t = pose.pose13.t + dp.block<3, 1>(8, 0);
    return ThreeViewCameraPose(pose12_new, pose13_new);
}

inline Eigen::Matrix3d skew_mat(const Eigen::Vector3d &v) {
    Eigen::Matrix3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}

// Pre-processed line track (calibrated coordinates)
struct ThreeViewLineTrack {
    std::array<Eigen::Vector3d, 3> l; // homogeneous lines l = p x q (not normalized)
    std::array<Eigen::Vector3d, 3> p; // homogeneous first endpoints
    std::array<Eigen::Vector3d, 3> q; // homogeneous second endpoints
};

inline ThreeViewLineTrack make_line_track(const Line2D &l1, const Line2D &l2, const Line2D &l3) {
    ThreeViewLineTrack track;
    const Line2D *ls[3] = {&l1, &l2, &l3};
    for (int k = 0; k < 3; ++k) {
        track.p[k] = ls[k]->x1.homogeneous();
        track.q[k] = ls[k]->x2.homogeneous();
        track.l[k] = track.p[k].cross(track.q[k]);
    }
    return track;
}

inline std::vector<ThreeViewLineTrack> make_line_tracks(const std::vector<Line2D> &l1, const std::vector<Line2D> &l2,
                                                        const std::vector<Line2D> &l3) {
    std::vector<ThreeViewLineTrack> tracks(l1.size());
    for (size_t i = 0; i < l1.size(); ++i)
        tracks[i] = make_line_track(l1[i], l2[i], l3[i]);
    return tracks;
}

// Evaluates the Sampson-normalized line transfer residuals (and optionally their Jacobians) for a fixed pose.
//
// For target view j, endpoint x in {p_j, q_j} and the remaining views a, b the transfer constraint is
//     g = x^T l'_j = w^T m,   w = R_j^T x,   m = e_b n_a - e_a n_b.
// g is linear in l_a and l_b:
//     dg/dl_a = R_a v_a,  v_a = e_b w - (w^T n_b) (C_j - C_a)
//     dg/dl_b = R_b v_b,  v_b = (w^T n_a) (C_j - C_b) - e_a w
// and l_k = p_k x q_k, so dg/dp_k = ([q_k]_x R_k v_k)_xy and dg/dq_k = (-[p_k]_x R_k v_k)_xy. Together with
// dg/dx = (l'_j)_xy this gives the gradient G of g w.r.t. all observed endpoints and the residual is r = g / |G|.
// Without the gradient w.r.t. the endpoints in views a, b (sampson = false) this is the point-line distance of x to
// l'_j (transfer distance).
//
// Note: the Sampson version is not well suited for scoring hypotheses. For degenerate poses with two coinciding
// camera centers the transfer into the third view becomes ill-conditioned and its Sampson error vanishes, while
// the constraints in the other two views reduce to rotation consistency. The transfer distance penalizes these.
class ThreeViewLineResidual {
  public:
    explicit ThreeViewLineResidual(const ThreeViewCameraPose &pose, bool compute_jacobians = false,
                                   bool sampson = false)
        : sampson(sampson) {
        R[0].setIdentity();
        t[0].setZero();
        R[1] = pose.pose12.R();
        t[1] = pose.pose12.t;
        R[2] = pose.pose13.R();
        t[2] = pose.pose13.t;
        for (int k = 0; k < 3; ++k)
            C[k] = -R[k].transpose() * t[k];

        if (compute_jacobians) {
            tangent_basis = threeview_tangent_basis(pose.pose12.t);
            for (int k = 0; k < 3; ++k)
                dC[k].setZero();
            dC[1].block<3, 3>(0, 0) = skew_mat(C[1]);
            dC[1].block<3, 2>(0, 3) = -R[1].transpose() * tangent_basis;
            dC[2].block<3, 3>(0, 5) = skew_mat(C[2]);
            dC[2].block<3, 3>(0, 8) = -R[2].transpose();
        }
    }

    // Squared residual norm (sum over both endpoints) for each of the three views. Returns false if degenerate.
    bool residuals(const ThreeViewLineTrack &track, std::array<double, 3> *r_sq) const {
        Eigen::Matrix<double, 3, 2> r;
        if (!evaluate<false>(track, &r, nullptr))
            return false;
        for (int j = 0; j < 3; ++j)
            (*r_sq)[j] = r.row(j).squaredNorm();
        return true;
    }

    // Residuals r(j, 0/1) and Jacobians J[j].row(0/1) for endpoints p_j, q_j. Returns false if degenerate.
    bool residuals_and_jacobians(const ThreeViewLineTrack &track, Eigen::Matrix<double, 3, 2> *r,
                                 std::array<Eigen::Matrix<double, 2, 11>, 3> *J) const {
        return evaluate<true>(track, r, J);
    }

    Eigen::Matrix<double, 3, 2> tangent_basis;

  private:
    typedef Eigen::Matrix<double, 3, 11> Mat3x11;
    typedef Eigen::Matrix<double, 1, 11> Row11;

    // First column of the rotation parameters of view k (-1 for view 1 which is fixed)
    static int rot_col(int k) { return k == 0 ? -1 : (k == 1 ? 0 : 5); }

    // d(R_k y) = R_k dy - R_k [y]_x dw_k
    void rotate_jacobian(int k, const Eigen::Vector3d &y, const Mat3x11 &dy, Mat3x11 *out) const {
        *out = R[k] * dy;
        if (k > 0)
            out->block<3, 3>(0, rot_col(k)) -= R[k] * skew_mat(y);
    }

    template <bool JAC>
    bool evaluate(const ThreeViewLineTrack &track, Eigen::Matrix<double, 3, 2> *r,
                  std::array<Eigen::Matrix<double, 2, 11>, 3> *J) const {
        Eigen::Vector3d n[3];
        double c[3];
        Mat3x11 dn[3];
        Row11 dc[3];
        for (int k = 0; k < 3; ++k) {
            n[k] = R[k].transpose() * track.l[k];
            c[k] = t[k].dot(track.l[k]);
        }
        if (JAC) {
            for (int k = 0; k < 3; ++k) {
                dn[k].setZero();
                dc[k].setZero();
            }
            dn[1].block<3, 3>(0, 0) = skew_mat(n[1]);
            dn[2].block<3, 3>(0, 5) = skew_mat(n[2]);
            dc[1].block<1, 2>(0, 3) = track.l[1].transpose() * tangent_basis;
            dc[2].block<1, 3>(0, 8) = track.l[2].transpose();
        }

        for (int j = 0; j < 3; ++j) {
            const int a = (j + 1) % 3, b = (j + 2) % 3;
            const double ea = n[a].dot(C[j]) + c[a];
            const double eb = n[b].dot(C[j]) + c[b];
            const Eigen::Vector3d m = eb * n[a] - ea * n[b];
            const Eigen::Vector3d lj = R[j] * m;
            const Eigen::Vector3d Da = C[j] - C[a];
            const Eigen::Vector3d Db = C[j] - C[b];

            Row11 dea, deb;
            Mat3x11 dlj, dDa, dDb;
            if (JAC) {
                dea = C[j].transpose() * dn[a] + n[a].transpose() * dC[j] + dc[a];
                deb = C[j].transpose() * dn[b] + n[b].transpose() * dC[j] + dc[b];
                const Mat3x11 dm = n[a] * deb + eb * dn[a] - n[b] * dea - ea * dn[b];
                rotate_jacobian(j, m, dm, &dlj);
                dDa = dC[j] - dC[a];
                dDb = dC[j] - dC[b];
            }

            const Eigen::Vector3d *ends[2] = {&track.p[j], &track.q[j]};
            for (int e = 0; e < 2; ++e) {
                const Eigen::Vector3d &x = *ends[e];
                const Eigen::Vector3d w = R[j].transpose() * x;
                const double g = x.dot(lj);
                const double wna = w.dot(n[a]), wnb = w.dot(n[b]);
                const Eigen::Vector3d Va = eb * w - wnb * Da;
                const Eigen::Vector3d Vb = wna * Db - ea * w;
                const Eigen::Vector3d va = R[a] * Va;
                const Eigen::Vector3d vb = R[b] * Vb;

                // gradient of g w.r.t. the observed endpoints (only w.r.t. x for the transfer distance)
                Eigen::Matrix<double, 10, 1> G = Eigen::Matrix<double, 10, 1>::Zero();
                G.segment<2>(0) = lj.topRows<2>();
                if (sampson) {
                    G.segment<2>(2) = track.q[a].cross(va).topRows<2>();
                    G.segment<2>(4) = va.cross(track.p[a]).topRows<2>();
                    G.segment<2>(6) = track.q[b].cross(vb).topRows<2>();
                    G.segment<2>(8) = vb.cross(track.p[b]).topRows<2>();
                }
                const double sigma_sq = G.squaredNorm();
                if (sigma_sq < 1e-24)
                    return false;
                const double sigma = std::sqrt(sigma_sq);
                (*r)(j, e) = g / sigma;

                if (JAC) {
                    Eigen::Matrix<double, 10, 11> dG = Eigen::Matrix<double, 10, 11>::Zero();
                    dG.block<2, 11>(0, 0) = dlj.topRows<2>();
                    if (sampson) {
                        Mat3x11 dw = Mat3x11::Zero();
                        if (j > 0)
                            dw.block<3, 3>(0, rot_col(j)) = skew_mat(w);
                        const Row11 dwna = n[a].transpose() * dw + w.transpose() * dn[a];
                        const Row11 dwnb = n[b].transpose() * dw + w.transpose() * dn[b];
                        const Mat3x11 dVa = w * deb + eb * dw - Da * dwnb - wnb * dDa;
                        const Mat3x11 dVb = Db * dwna + wna * dDb - w * dea - ea * dw;
                        Mat3x11 dva, dvb;
                        rotate_jacobian(a, Va, dVa, &dva);
                        rotate_jacobian(b, Vb, dVb, &dvb);
                        dG.block<2, 11>(2, 0) = (skew_mat(track.q[a]) * dva).topRows<2>();
                        dG.block<2, 11>(4, 0) = (-skew_mat(track.p[a]) * dva).topRows<2>();
                        dG.block<2, 11>(6, 0) = (skew_mat(track.q[b]) * dvb).topRows<2>();
                        dG.block<2, 11>(8, 0) = (-skew_mat(track.p[b]) * dvb).topRows<2>();
                    }

                    const Row11 dg = x.transpose() * dlj;
                    const Row11 dsigma_sq = 2.0 * G.transpose() * dG;
                    (*J)[j].row(e) = dg / sigma - (g / (2.0 * sigma_sq * sigma)) * dsigma_sq;
                }
            }
        }
        return true;
    }

    const bool sampson;
    Eigen::Matrix3d R[3];
    Eigen::Vector3d t[3], C[3];
    Mat3x11 dC[3];
};

// MSAC score of the line tracks. A track is an inlier if its error in all three views is below the threshold.
// The per-view error is the mean squared error (transfer distance or Sampson) of the two endpoints.
inline double compute_threeview_line_msac_score(const ThreeViewCameraPose &pose,
                                                const std::vector<ThreeViewLineTrack> &tracks, double sq_threshold,
                                                size_t *inlier_count, std::vector<char> *inliers = nullptr,
                                                bool sampson = false) {
    ThreeViewLineResidual res(pose, false, sampson);
    *inlier_count = 0;
    if (inliers != nullptr)
        inliers->assign(tracks.size(), 0);
    double score = 0.0;
    std::array<double, 3> r_sq;
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (!res.residuals(tracks[i], &r_sq)) {
            score += 3 * sq_threshold;
            continue;
        }
        bool inlier = true;
        for (int j = 0; j < 3; ++j) {
            const double err = 0.5 * r_sq[j];
            if (err < sq_threshold) {
                score += err;
            } else {
                score += sq_threshold;
                inlier = false;
            }
        }
        if (inlier) {
            (*inlier_count)++;
            if (inliers != nullptr)
                (*inliers)[i] = 1;
        }
    }
    return score;
}

// The robust loss is applied per track and view on the mean squared residual of the two endpoints
// (consistent with compute_threeview_line_msac_score). The residuals are multiplied by residual_scale before
// applying the loss (used to express line residuals in the units of the point threshold in hybrid refinement)
// and the cost is multiplied by weight_scale.
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class ThreeViewLineJacobianAccumulator {
  public:
    ThreeViewLineJacobianAccumulator(const std::vector<ThreeViewLineTrack> &line_tracks, const LossFunction &l,
                                     const ResidualWeightVector &w = ResidualWeightVector(), double weight_scale = 1.0,
                                     double residual_scale = 1.0, bool sampson = false)
        : tracks(line_tracks), loss_fn(l), weights(w), weight_scale(weight_scale),
          residual_scale_sq(residual_scale * residual_scale), sampson(sampson) {}

    double residual(const ThreeViewCameraPose &three_view_pose) const {
        ThreeViewLineResidual res(three_view_pose, false, sampson);
        double cost = 0.0;
        std::array<double, 3> r_sq;
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (!res.residuals(tracks[i], &r_sq))
                continue;
            for (int j = 0; j < 3; ++j)
                cost += weight_scale * weights[i] * loss_fn.loss(0.5 * residual_scale_sq * r_sq[j]);
        }
        return cost;
    }

    size_t accumulate(const ThreeViewCameraPose &three_view_pose, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) {
        ThreeViewLineResidual res(three_view_pose, true, sampson);
        tangent_basis = res.tangent_basis;

        Eigen::Matrix<double, 3, 2> r;
        std::array<Eigen::Matrix<double, 2, 11>, 3> J;
        size_t num_residuals = 0;
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (!res.residuals_and_jacobians(tracks[i], &r, &J))
                continue;
            for (int j = 0; j < 3; ++j) {
                const double r_sq = 0.5 * residual_scale_sq * r.row(j).squaredNorm();
                // d/dp loss(0.5 * s^2 * |r|^2) = loss'(.) * s^2 * J^T r (dropping the factor 2 as elsewhere)
                const double weight = 0.5 * residual_scale_sq * weight_scale * weights[i] * loss_fn.weight(r_sq);
                if (weight == 0.0)
                    continue;
                num_residuals += 2;
                Jtr += weight * J[j].transpose() * r.row(j).transpose();
                JtJ.template selfadjointView<Eigen::Lower>().rankUpdate(J[j].transpose(), weight);
            }
        }
        return num_residuals;
    }

    ThreeViewCameraPose step(Eigen::Matrix<double, 11, 1> dp, const ThreeViewCameraPose &three_view_pose) const {
        return threeview_step(dp, three_view_pose, tangent_basis);
    }

    typedef ThreeViewCameraPose param_t;
    static constexpr size_t num_params = 11;

  private:
    const std::vector<ThreeViewLineTrack> &tracks;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    const double weight_scale;
    const double residual_scale_sq;
    const bool sampson;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Joint refinement of points (Sampson error in all three pairs) and lines (line transfer error).
// Line residuals are multiplied by line_residual_scale (e.g. point threshold / line threshold) so that a single
// robust loss can be used, and the line cost is weighted by line_weight.
template <typename LossFunction>
class ThreeViewHybridJacobianAccumulator {
  public:
    ThreeViewHybridJacobianAccumulator(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                       const std::vector<Point2D> &x3, const std::vector<ThreeViewLineTrack> &tracks,
                                       const LossFunction &l, double line_weight, double line_residual_scale = 1.0,
                                       bool line_sampson = false)
        : point_accum(x1, x2, x3, l, point_weights),
          line_accum(tracks, l, line_weights, line_weight, line_residual_scale, line_sampson) {}

    double residual(const ThreeViewCameraPose &three_view_pose) const {
        return point_accum.residual(three_view_pose) + line_accum.residual(three_view_pose);
    }

    size_t accumulate(const ThreeViewCameraPose &three_view_pose, Eigen::Matrix<double, 11, 11> &JtJ,
                      Eigen::Matrix<double, 11, 1> &Jtr) {
        // Both accumulators use the same (deterministic) tangent basis for t12
        size_t n = point_accum.accumulate(three_view_pose, JtJ, Jtr);
        n += line_accum.accumulate(three_view_pose, JtJ, Jtr);
        return n;
    }

    ThreeViewCameraPose step(Eigen::Matrix<double, 11, 1> dp, const ThreeViewCameraPose &three_view_pose) const {
        return line_accum.step(dp, three_view_pose);
    }

    typedef ThreeViewCameraPose param_t;
    static constexpr size_t num_params = 11;

  private:
    UniformWeightVector point_weights, line_weights;
    ThreeViewRelativePoseJacobianAccumulator<LossFunction, UniformWeightVector> point_accum;
    ThreeViewLineJacobianAccumulator<LossFunction, UniformWeightVector> line_accum;
};

} // namespace poselib

#endif
