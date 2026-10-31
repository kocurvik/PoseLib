#pragma once

#include "../../camera_pose.h"
#include "../../misc/essential.h"
#include "../../types.h"
#include "jacobian_accumulator.h"
#include "monodepth_relpose.h"
#include "refiner_base.h"
#include "relative.h"
#include "twoview_lines.h"

#include <vector>

namespace poselib {

// Three view relative pose refinement
// -----------------------------------
// All poses are relative to view 1: X_2 = R12 X_1 + t12, X_3 = R13 X_1 + t13 (ThreeViewGeometry). Point tracks with
// three kinds of residuals, each can be switched off:
//   Sampson       the Sampson errors of the pairs 12, 13, 23, weighted by weight_sampson
//   reprojection  the six directed reprojection errors i -> j, multiplied by sqrt(scale_reproj). A point in view i is
//                 X_i = s_i d_i x_i with s_1 = 1, s_2 = scale2, s_3 = scale3, so the depths of view 1 fix the scale of
//                 the translations
//   lines         the residuals of TwoViewLine for the pairs 12, 13, 23 (line_residual), multiplied by line_scale and
//                 weighted by line_weight, every pair is robustified separately
// With the update R <- R exp([w]_x) and P = R1i^T (X_i - t1i) the point X_i in the frame of view 1:
//   dP/dw1i = [P]_x, dP/dt1i = -R1i^T, dP/ds_i = R1i^T d_i x_i,
//   dZ/dP = R1j, dZ/dw1j = -R1j [P]_x, dZ/dt1j = I          for Z = R1j P + t1j.
// For the pair 23 R23 = R13 R12^T and t23 = t13 - R23 t12. With B_k = R13 [e_k]_x R12^T:
//   dR23/dw13_k = B_k, dR23/dw12_k = -B_k, dt23/dw13_k = -B_k t12, dt23/dw12_k = B_k t12,
//   dE23/dw13_k = -dE23/dw12_k = [-B_k t12]_x R23 + [t23]_x B_k, dE23/dt13_k = [e_k]_x R23, dE23/dt12_k = -R23 [e_k]_x.
// Parameters: (w12, t12, w13, t13, scale2, scale3) = 14 with the reprojection errors. Without them the depth scales
// are not observable and neither is the common scale of the translations, so the parameters are
// (w12, tangent of t12, w13, t13) = 11 and scale2, scale3 are kept.

// Lines of the pairs 12, 13, 23 of a three view track
struct ThreeViewLines {
    std::vector<TwoViewLine> l12, l13, l23;
};

inline ThreeViewLines make_threeview_lines(const std::vector<Line3D> &L1, const std::vector<Line3D> &L2,
                                           const std::vector<Line3D> &L3) {
    return {make_twoview_lines(L1, L2), make_twoview_lines(L1, L3), make_twoview_lines(L2, L3)};
}

template <typename Accumulator = NormalAccumulator>
class ThreeViewRefiner : public RefinerBase<ThreeViewGeometry, Accumulator> {
  public:
    ThreeViewRefiner(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const std::vector<Point2D> &x3,
                     const std::vector<double> &d1, const std::vector<double> &d2, const std::vector<double> &d3,
                     const ThreeViewLines &lines, int line_residual, double scale_reproj, double weight_sampson,
                     double line_weight, double line_scale)
        : x{&x1, &x2, &x3}, d{&d1, &d2, &d3}, lines(lines), line_residual(line_residual), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), line_weight(line_weight), line_scale(line_scale) {
        this->num_params = scale_reproj > 0.0 ? 14 : 11;
    }

    double compute_residual(Accumulator &acc, const ThreeViewGeometry &model) {
        const Views v(model);
        Eigen::Matrix3d E12, E13, E23;
        essential_from_motion(model.pose12, &E12);
        essential_from_motion(model.pose13, &E13);
        essential_from_motion(model.pose23(), &E23);
        const double sr = std::sqrt(std::max(scale_reproj, 0.0));

        for (size_t k = 0; k < x[0]->size(); ++k) {
            if (weight_sampson > 0.0) {
                acc.add_residual(compute_sampson_residual((*x[0])[k], (*x[1])[k], E12), weight_sampson);
                acc.add_residual(compute_sampson_residual((*x[0])[k], (*x[2])[k], E13), weight_sampson);
                acc.add_residual(compute_sampson_residual((*x[1])[k], (*x[2])[k], E23), weight_sampson);
            }
            if (scale_reproj > 0.0) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        if (i == j)
                            continue;
                        const Eigen::Vector3d P = v.R[i].transpose() * (v.s[i] * (*d[i])[k] * (*x[i])[k].homogeneous() - v.t[i]);
                        const Eigen::Vector3d Z = v.R[j] * P + v.t[j];
                        if (Z(2) > 0) {
                            const Eigen::Vector2d r = sr * (Z.hnormalized() - (*x[j])[k]);
                            acc.add_residual(r, 1.0);
                        }
                    }
                }
            }
        }
        add_line_residuals(acc, v.R[1], lines.l12);
        add_line_residuals(acc, v.R[2], lines.l13);
        add_line_residuals(acc, v.R[2] * v.R[1].transpose(), lines.l23);
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ThreeViewGeometry &model) {
        if (scale_reproj <= 0.0)
            setup_tangent_basis(model.pose12.t, tangent_basis);

        const Views v(model);
        if (weight_sampson > 0.0)
            add_sampson_jacobians(acc, model, v);
        if (scale_reproj > 0.0) {
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (i != j)
                        add_reprojection_jacobians(acc, v, i, j);
        }
        add_line_jacobians(acc, v, lines.l12, 0);
        add_line_jacobians(acc, v, lines.l13, 1);
        add_line_jacobians(acc, v, lines.l23, 2);
    }

    ThreeViewGeometry step(const Eigen::VectorXd &dp, const ThreeViewGeometry &model) const {
        ThreeViewGeometry out = model;
        out.pose12.q = quat_step_post(model.pose12.q, dp.block<3, 1>(0, 0));
        if (scale_reproj > 0.0) {
            out.pose12.t = model.pose12.t + dp.block<3, 1>(3, 0);
            out.pose13.q = quat_step_post(model.pose13.q, dp.block<3, 1>(6, 0));
            out.pose13.t = model.pose13.t + dp.block<3, 1>(9, 0);
            out.scale2 = model.scale2 + dp(12);
            out.scale3 = model.scale3 + dp(13);
        } else {
            out.pose12.t = model.pose12.t + tangent_basis * dp.block<2, 1>(3, 0);
            out.pose13.q = quat_step_post(model.pose13.q, dp.block<3, 1>(5, 0));
            out.pose13.t = model.pose13.t + dp.block<3, 1>(8, 0);
        }
        return out;
    }

    typedef ThreeViewGeometry param_t;

  private:
    typedef Eigen::Matrix<double, 3, 3> Mat3;
    typedef Eigen::Matrix<double, 9, 3> Mat93;

    // First column of the parameters of views 2 and 3 in the full 14 parameter vector
    static constexpr int rot_col[3] = {-1, 0, 6};
    static constexpr int trans_col[3] = {-1, 3, 9};
    static constexpr int scale_col[3] = {-1, 12, 13};

    struct Views {
        explicit Views(const ThreeViewGeometry &m) {
            R[0].setIdentity();
            R[1] = m.pose12.R();
            R[2] = m.pose13.R();
            t[0].setZero();
            t[1] = m.pose12.t;
            t[2] = m.pose13.t;
            s[0] = 1.0;
            s[1] = m.scale2;
            s[2] = m.scale3;
        }
        Mat3 R[3];
        Eigen::Vector3d t[3];
        double s[3];
    };

    // The Jacobian with respect to the full 14 parameters -> the parameters of the refiner
    template <int Rows> Eigen::Matrix<double, Rows, 11> reduce(const Eigen::Matrix<double, Rows, 14> &J) const {
        Eigen::Matrix<double, Rows, 11> out;
        out.template leftCols<3>() = J.template leftCols<3>();
        out.template middleCols<2>(3) = J.template middleCols<3>(3) * tangent_basis;
        out.template middleCols<6>(5) = J.template middleCols<6>(6);
        return out;
    }

    void add(Accumulator &acc, double r, const Eigen::Matrix<double, 1, 14> &J, double w) const {
        if (scale_reproj > 0.0) {
            acc.add_jacobian(r, J, w);
        } else {
            const Eigen::Matrix<double, 1, 11> Jr = reduce<1>(J);
            acc.add_jacobian(r, Jr, w);
        }
    }

    template <int Rows>
    void add(Accumulator &acc, const Eigen::Matrix<double, Rows, 1> &r, const Eigen::Matrix<double, Rows, 14> &J,
             double w) const {
        if (scale_reproj > 0.0) {
            acc.add_jacobian(r, J, w);
        } else {
            const Eigen::Matrix<double, Rows, 11> Jr = reduce<Rows>(J);
            acc.add_jacobian(r, Jr, w);
        }
    }

    void add_sampson_jacobians(Accumulator &acc, const ThreeViewGeometry &model, const Views &v) const {
        const Mat3 &R12 = v.R[1], &R13 = v.R[2];
        const Eigen::Vector3d &t12 = v.t[1];
        const CameraPose pose23 = model.pose23();
        const Mat3 R23 = R13 * R12.transpose();
        Mat3 E12, E13, E23;
        essential_from_motion(model.pose12, &E12);
        essential_from_motion(model.pose13, &E13);
        essential_from_motion(pose23, &E23);

        Mat93 dE12_w, dE12_t, dE13_w, dE13_t, dE23_w13, dE23_t13, dE23_t12;
        deriv_essential_wrt_rotation(E12, dE12_w);
        deriv_essential_wrt_translation(R12, dE12_t);
        deriv_essential_wrt_rotation(E13, dE13_w);
        deriv_essential_wrt_translation(R13, dE13_t);
        deriv_essential_wrt_translation(R23, dE23_t13);
        deriv_essential_wrt_rotation(R23, dE23_t12);
        dE23_t12 = -dE23_t12;
        const Mat3 t23x = twoview_skew(pose23.t);
        for (int k = 0; k < 3; ++k) {
            const Mat3 B = R13 * twoview_skew(Eigen::Vector3d::Unit(k)) * R12.transpose();
            const Mat3 C = twoview_skew(-B * t12) * R23 + t23x * B;
            dE23_w13.col(k) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(C.data());
        }

        Eigen::Matrix<double, 1, 9> dF;
        Eigen::Matrix<double, 1, 14> J;
        for (size_t k = 0; k < x[0]->size(); ++k) {
            const Point2D &a = (*x[0])[k], &b = (*x[1])[k], &c = (*x[2])[k];
            double r = compute_sampson_jacobian(a, b, E12, dF);
            J.setZero();
            J.block<1, 3>(0, 0) = dF * dE12_w;
            J.block<1, 3>(0, 3) = dF * dE12_t;
            add(acc, r, J, weight_sampson);

            r = compute_sampson_jacobian(a, c, E13, dF);
            J.setZero();
            J.block<1, 3>(0, 6) = dF * dE13_w;
            J.block<1, 3>(0, 9) = dF * dE13_t;
            add(acc, r, J, weight_sampson);

            r = compute_sampson_jacobian(b, c, E23, dF);
            J.block<1, 3>(0, 6) = dF * dE23_w13;
            J.block<1, 3>(0, 0) = -J.block<1, 3>(0, 6);
            J.block<1, 3>(0, 9) = dF * dE23_t13;
            J.block<1, 3>(0, 3) = dF * dE23_t12;
            add(acc, r, J, weight_sampson);
        }
    }

    void add_reprojection_jacobians(Accumulator &acc, const Views &v, int i, int j) const {
        const double sr = std::sqrt(scale_reproj);
        const Mat3 Rji = v.R[j] * v.R[i].transpose();
        for (size_t k = 0; k < x[0]->size(); ++k) {
            const Eigen::Vector3d ray = (*d[i])[k] * (*x[i])[k].homogeneous();
            const Eigen::Vector3d P = v.R[i].transpose() * (v.s[i] * ray - v.t[i]);
            const Eigen::Vector3d Z = v.R[j] * P + v.t[j];
            if (Z(2) <= 0)
                continue;
            const double inv_z = 1.0 / Z(2);
            Eigen::Matrix<double, 2, 3> Jproj;
            Jproj << inv_z, 0.0, -Z(0) * inv_z * inv_z, 0.0, inv_z, -Z(1) * inv_z * inv_z;

            Eigen::Matrix<double, 3, 14> dZ = Eigen::Matrix<double, 3, 14>::Zero();
            const Mat3 Px = twoview_skew(P);
            if (i > 0) {
                dZ.block<3, 3>(0, rot_col[i]) = v.R[j] * Px;
                dZ.block<3, 3>(0, trans_col[i]) = -Rji;
                dZ.col(scale_col[i]) = Rji * ray;
            }
            if (j > 0) {
                dZ.block<3, 3>(0, rot_col[j]) = -v.R[j] * Px;
                dZ.block<3, 3>(0, trans_col[j]).setIdentity();
            }
            const Eigen::Vector2d r = sr * (Z.hnormalized() - (*x[j])[k]);
            const Eigen::Matrix<double, 2, 14> J = sr * Jproj * dZ;
            add<2>(acc, r, J, 1.0);
        }
    }

    void add_line_residuals(Accumulator &acc, const Mat3 &R, const std::vector<TwoViewLine> &pair_lines) const {
        for (const TwoViewLine &l : pair_lines) {
            if (uses_direction_residual(line_residual)) {
                const Eigen::Vector3d r = line_scale * line_direction_residual(R, l);
                acc.add_residual(r, line_weight);
            }
            if (line_residual == 0)
                continue;
            const Eigen::Vector2d r = line_scale * line_plane_residual(R, l);
            if (line_residual == 3) {
                acc.add_residual(r(0), line_weight);
                acc.add_residual(r(1), line_weight);
            } else {
                acc.add_residual(r, line_weight);
            }
        }
    }

    // pair: 0 - 12, 1 - 13, 2 - 23
    void add_line_jacobians(Accumulator &acc, const Views &v, const std::vector<TwoViewLine> &pair_lines,
                            int pair) const {
        const Mat3 &R12 = v.R[1], &R13 = v.R[2];
        const Mat3 R = pair == 0 ? R12 : (pair == 1 ? R13 : Mat3(R13 * R12.transpose()));
        const int col = pair == 0 ? rot_col[1] : rot_col[2];
        // d(R u)/dw of the second rotation of the pair = -dRu(u), for the pair 23 d(R23 u)/dw12 = dRu(u)
        const auto dRu = [&](const Eigen::Vector3d &u) -> Mat3 {
            return pair == 2 ? Mat3(R13 * twoview_skew(R12.transpose() * u)) : Mat3(R * twoview_skew(u));
        };
        // The row(s) of a line Jacobian from the coefficients C of R u: J = line_scale * C * d(R u)/dw
        const auto fill = [&](auto &J, const auto &C, const Eigen::Vector3d &u) {
            const auto du = dRu(u);
            J.template middleCols<3>(col) = -line_scale * C * du;
            if (pair == 2)
                J.template middleCols<3>(rot_col[1]) = line_scale * C * du;
        };
        for (const TwoViewLine &l : pair_lines) {
            if (uses_direction_residual(line_residual)) {
                const Eigen::Vector3d r = line_scale * line_direction_residual(R, l);
                Eigen::Matrix<double, 3, 14> J = Eigen::Matrix<double, 3, 14>::Zero();
                const Mat3 C = -twoview_skew(l.d2);
                fill(J, C, l.d1);
                add<3>(acc, r, J, line_weight);
            }
            if (line_residual == 0)
                continue;
            const Eigen::Vector2d r = line_scale * line_plane_residual(R, l);
            Eigen::Matrix<double, 1, 14> J0 = Eigen::Matrix<double, 1, 14>::Zero(), J1 = J0;
            const Eigen::Matrix<double, 1, 3> C0 = l.n2.transpose(), C1 = l.d2.transpose();
            fill(J0, C0, l.d1);
            fill(J1, C1, l.n1);
            if (line_residual == 3) {
                add(acc, r(0), J0, line_weight);
                add(acc, r(1), J1, line_weight);
            } else {
                Eigen::Matrix<double, 2, 14> J;
                J.row(0) = J0;
                J.row(1) = J1;
                add<2>(acc, r, J, line_weight);
            }
        }
    }

    const std::vector<Point2D> *x[3];
    const std::vector<double> *d[3];
    const ThreeViewLines &lines;
    const int line_residual;
    const double scale_reproj, weight_sampson, line_weight, line_scale;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

} // namespace poselib
