# Three view relative pose from lines with scale-invariant depth (branch `mlde`)

Additions on top of the `threeview` branch for three view relative pose using 3D line segments (2D segments with
depth of both endpoints, e.g. predicted by a monocular network) and points with depth. Depth may have an unknown
scale in each view. All poses map view 1 to views 2 and 3 (`X_k = R_k X_1 + t_k`); `pose12` and `pose13` share the
scale of view 1. Results of the experiments are in the `mlde` project (`D:\Research\code\mlde\RESULTS.md`).

## Minimal solvers (`PoseLib/solvers/threeview_lines.h`)

* `threeview_2p1l` — 2 points with depth + 1 line direction (port of `mlde/theory/2p1l.m`, formulated with
  orthonormal frames: the vector between the points is aligned exactly, the line direction fixes the rotation about
  it). Up to 4 solutions (sign ambiguity of the line direction in views 2, 3).
* `threeview_2l` — 2 lines with depth. Rotation from the two line directions (frame from their bisectors, all
  consistent direction sign combinations), scale and translation from the feet of the common perpendicular.
  Up to 4 x 4 solutions, pruned by cheirality.
* `threeview_2l_planes` — 2 lines with depth. Rotation as above; translation from the back-projected planes of the
  2D lines in views 2, 3 (`l^T (R X + t) = 0`). With known R each line gives only one constraint on t, so the
  component along `l_a x l_b` is recovered from the depth (`(I - d d^T)(R X_1 + t - sigma X_k) = 0`).

Endpoints are not assumed to correspond between views.

## Line residual and refinement (`PoseLib/robust/threeview_lines_jacobian.h`)

For each track and target view j the planes of the other two views are intersected in the pencil through the
camera center of j, giving the transferred line `l'_j = R_j (e_b n_a - e_a n_b)`. Residuals are the distances of the
detected endpoints of view j to `l'_j` (transfer distance, default), optionally normalized by the gradient w.r.t.
all observed endpoints (Sampson, `line_sampson`). 6 residuals per track, robust loss applied per view on the mean of
the two endpoints. Jacobians are analytic (same 11 parameter update as `ThreeViewRelativePoseJacobianAccumulator`)
and checked against finite differences in `scripts/test_threeview_lines.cc`.

The Sampson version has a larger convergence basin in refinement but is not suited for scoring: for poses with two
coinciding camera centers it vanishes for the third view, so RANSAC prefers such degenerate hypotheses.

* `ThreeViewLineJacobianAccumulator` — lines only.
* `ThreeViewHybridJacobianAccumulator` — points (Sampson) + lines; line residuals are scaled by
  `line_residual_scale` (point threshold / line threshold) and weighted by `line_weight`.
* `refine_3v_lines`, `refine_3v_hybrid` in `robust/bundle.h`.

## Estimators (`PoseLib/robust/estimators/threeview_lines.h`)

* `ThreeViewLineEstimator` — line-only LO-RANSAC, solver selected by `RansacOptions::line_solver`
  (0 common perpendicular, 1 planes).
* `ThreeViewHybridEstimator` — points + lines, 2p1l solver, score = point Sampson MSAC + `line_weight` x line MSAC.

New `RansacOptions`: `line_solver`, `max_line_error` (px), `line_weight`, `line_sampson`.

Entry points (`robust.h`, pixel inputs, cameras unproject points/endpoints and lift them with the depths):
`estimate_3v_relative_pose_lines`, `estimate_3v_relative_pose_hybrid`.

## Python bindings

```python
pose, info = poselib.estimate_three_view_relative_pose_lines(l1, l2, l3, d1, d2, d3, cam1, cam2, cam3, ransac_opt, bundle_opt)
# l*: (N, 4) segments x1 y1 x2 y2 in px, d*: (N, 2) endpoint depths
pose, info = poselib.estimate_three_view_relative_pose_hybrid(x1, x2, x3, xd1, xd2, xd3, l1, l2, l3, d1, d2, d3,
                                                              cam1, cam2, cam3, ransac_opt, bundle_opt)
# x*: (M, 2) points in px, xd*: (M,) depths; info['inliers'] (points), info['line_inliers']
poselib.refine_three_view_lines(...), poselib.refine_three_view_hybrid(...)   # calibrated inputs
poselib.threeview_2p1l(...), poselib.threeview_2l(...), poselib.threeview_2l_planes(...)
```

## Tests

```shell
g++ -O2 -std=c++17 -I. -I<eigen include> scripts/test_threeview_lines.cc PoseLib/solvers/threeview_lines.cc
```

Note: with `sample_sz = 4` the existing threeview estimator requires `inner_refine_extra = 0` (default 1 reads out
of bounds of the sample).
