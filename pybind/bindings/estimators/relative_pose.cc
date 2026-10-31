#include "../../helpers.h"
#include "../../pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poselib {
namespace {

std::pair<CameraPose, py::dict> estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                               const std::vector<Eigen::Vector2d> &points2D_2,
                                                               const Camera &camera1, const Camera &camera2,
                                                               const py::dict &opt_dict,
                                                               const std::optional<CameraPose> &initial_pose) {

    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats =
        estimate_relative_pose(points2D_1, points2D_2, camera1, camera2, opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict>
estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                               const std::vector<Eigen::Vector2d> &points2D_2, const py::dict &camera1_dict,
                               const py::dict &camera2_dict, const py::dict &opt_dict,
                               const std::optional<CameraPose> &initial_pose) {
    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    return estimate_relative_pose_wrapper(points2D_1, points2D_2, camera1, camera2, opt_dict,
                                          initial_pose);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const Camera &camera1,
    const Camera &camera2, const py::dict &opt_dict,
    const std::optional<MonoDepthTwoViewGeometry> &initial_pose) {
    MonoDepthRelativePoseOptions opt;
    update_monodepth_relative_pose_options(opt_dict, opt);

    MonoDepthTwoViewGeometry monodepth_geometry;
    if (initial_pose.has_value()) {
        monodepth_geometry = initial_pose.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_monodepth_relative_pose(points2D_1, points2D_2, depth_1, depth_2, camera1, camera2,
                                                         opt, &monodepth_geometry, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(monodepth_geometry, output_dict);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const py::dict &camera1_dict,
    const py::dict &camera2_dict, const py::dict &opt_dict,
    const std::optional<MonoDepthTwoViewGeometry> &initial_pose) {
    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    return estimate_monodepth_relative_pose_wrapper(points2D_1, points2D_2, depth_1, depth_2, camera1, camera2,
                                                    opt_dict, initial_pose);
}

std::vector<Line2D> lines_from_vectors(const std::vector<Eigen::Vector4d> &lines) {
    std::vector<Line2D> out;
    out.reserve(lines.size());
    for (const Eigen::Vector4d &l : lines)
        out.emplace_back(l.head<2>(), l.tail<2>());
    return out;
}

std::pair<CameraPose, py::dict> estimate_relative_pose_lines_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<Eigen::Vector4d> &lines2D_1, const std::vector<Eigen::Vector4d> &lines2D_2,
    const std::vector<Eigen::Vector2d> &line_depths_1, const std::vector<Eigen::Vector2d> &line_depths_2,
    const Camera &camera1, const Camera &camera2, const py::dict &opt_dict) {
    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);
    const std::vector<Line2D> l1 = lines_from_vectors(lines2D_1), l2 = lines_from_vectors(lines2D_2);
    CameraPose pose;
    std::vector<char> inlier_mask;
    py::gil_scoped_release release;
    RansacStats stats = estimate_relative_pose_lines(points2D_1, points2D_2, l1, l2, line_depths_1, line_depths_2,
                                                     camera1, camera2, opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;
    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict> estimate_relative_pose_lines_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<Eigen::Vector4d> &lines2D_1, const std::vector<Eigen::Vector4d> &lines2D_2,
    const std::vector<Eigen::Vector2d> &line_depths_1, const std::vector<Eigen::Vector2d> &line_depths_2,
    const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &opt_dict) {
    return estimate_relative_pose_lines_wrapper(points2D_1, points2D_2, lines2D_1, lines2D_2, line_depths_1,
                                                line_depths_2, camera_from_dict(camera1_dict),
                                                camera_from_dict(camera2_dict), opt_dict);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_lines_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2,
    const std::vector<Eigen::Vector4d> &lines2D_1, const std::vector<Eigen::Vector4d> &lines2D_2,
    const std::vector<Eigen::Vector2d> &line_depths_1, const std::vector<Eigen::Vector2d> &line_depths_2,
    const Camera &camera1, const Camera &camera2, const py::dict &opt_dict) {
    MonoDepthRelativePoseOptions opt;
    update_monodepth_relative_pose_options(opt_dict, opt);
    const std::vector<Line2D> l1 = lines_from_vectors(lines2D_1), l2 = lines_from_vectors(lines2D_2);
    MonoDepthTwoViewGeometry geometry;
    std::vector<char> inlier_mask;
    py::gil_scoped_release release;
    RansacStats stats =
        estimate_monodepth_relative_pose_lines(points2D_1, points2D_2, depth_1, depth_2, l1, l2, line_depths_1,
                                               line_depths_2, camera1, camera2, opt, &geometry, &inlier_mask);
    py::gil_scoped_acquire acquire;
    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(geometry, output_dict);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_lines_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2,
    const std::vector<Eigen::Vector4d> &lines2D_1, const std::vector<Eigen::Vector4d> &lines2D_2,
    const std::vector<Eigen::Vector2d> &line_depths_1, const std::vector<Eigen::Vector2d> &line_depths_2,
    const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &opt_dict) {
    return estimate_monodepth_relative_pose_lines_wrapper(points2D_1, points2D_2, depth_1, depth_2, lines2D_1,
                                                          lines2D_2, line_depths_1, line_depths_2,
                                                          camera_from_dict(camera1_dict),
                                                          camera_from_dict(camera2_dict), opt_dict);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_segments_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2,
    const std::vector<Eigen::Vector4d> &segments2D_1, const std::vector<Eigen::Vector4d> &segments2D_2,
    const std::vector<Eigen::Vector2d> &segment_depths_1, const std::vector<Eigen::Vector2d> &segment_depths_2,
    const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &opt_dict) {
    MonoDepthRelativePoseOptions opt;
    update_monodepth_relative_pose_options(opt_dict, opt);
    const Camera camera1 = camera_from_dict(camera1_dict), camera2 = camera_from_dict(camera2_dict);
    const std::vector<Line2D> s1 = lines_from_vectors(segments2D_1), s2 = lines_from_vectors(segments2D_2);
    MonoDepthTwoViewGeometry geometry;
    std::vector<char> inlier_mask;
    py::gil_scoped_release release;
    RansacStats stats =
        estimate_monodepth_relative_pose_segments(points2D_1, points2D_2, depth_1, depth_2, s1, s2, segment_depths_1,
                                                  segment_depths_2, camera1, camera2, opt, &geometry, &inlier_mask);
    py::gil_scoped_acquire acquire;
    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(geometry, output_dict);
}

std::pair<ThreeViewGeometry, py::dict> estimate_threeview_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<Eigen::Vector2d> &points2D_3, const std::vector<double> &depth_1,
    const std::vector<double> &depth_2, const std::vector<double> &depth_3,
    const std::vector<Eigen::Vector4d> &lines2D_1, const std::vector<Eigen::Vector4d> &lines2D_2,
    const std::vector<Eigen::Vector4d> &lines2D_3, const std::vector<Eigen::Vector2d> &line_depths_1,
    const std::vector<Eigen::Vector2d> &line_depths_2, const std::vector<Eigen::Vector2d> &line_depths_3,
    const std::vector<Eigen::Vector2d> &feature_points_1, const std::vector<Eigen::Vector2d> &feature_points_2,
    const std::vector<Eigen::Vector2d> &feature_points_3, const std::vector<Eigen::Vector4d> &feature_lines_1,
    const std::vector<Eigen::Vector4d> &feature_lines_2, const std::vector<Eigen::Vector4d> &feature_lines_3,
    const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &camera3_dict,
    const py::dict &opt_dict) {
    ThreeViewOptions opt;
    update_threeview_options(opt_dict, opt);
    ThreeViewFeatures features;
    features.x[0] = feature_points_1;
    features.x[1] = feature_points_2;
    features.x[2] = feature_points_3;
    features.l[0] = lines_from_vectors(feature_lines_1);
    features.l[1] = lines_from_vectors(feature_lines_2);
    features.l[2] = lines_from_vectors(feature_lines_3);
    const Camera camera1 = camera_from_dict(camera1_dict), camera2 = camera_from_dict(camera2_dict),
                 camera3 = camera_from_dict(camera3_dict);
    const std::vector<Line2D> l1 = lines_from_vectors(lines2D_1), l2 = lines_from_vectors(lines2D_2),
                              l3 = lines_from_vectors(lines2D_3);
    ThreeViewGeometry geometry;
    std::vector<char> inlier_mask;
    py::gil_scoped_release release;
    RansacStats stats = estimate_threeview_relative_pose(points2D_1, points2D_2, points2D_3, depth_1, depth_2, depth_3,
                                                         l1, l2, l3, line_depths_1, line_depths_2, line_depths_3,
                                                         features, camera1, camera2, camera3, opt, &geometry,
                                                         &inlier_mask);
    py::gil_scoped_acquire acquire;
    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(geometry, output_dict);
}

std::pair<ImagePair, py::dict>
estimate_shared_focal_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                            const std::vector<Eigen::Vector2d> &points2D_2, const Eigen::Vector2d &pp,
                                            const py::dict &opt_dict,
                                            const std::optional<ImagePair> &initial_image_pair) {

    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);

    ImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        opt.ransac.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    std::vector<Image> output;

    py::gil_scoped_release release;
    RansacStats stats = estimate_shared_focal_relative_pose(points2D_1, points2D_2, pp, opt,
                                                            &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<MonoDepthImagePair, py::dict> estimate_monodepth_shared_focal_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const Eigen::Vector2d &pp,
    const py::dict &opt_dict, const std::optional<MonoDepthImagePair> &initial_image_pair) {

    MonoDepthRelativePoseOptions opt;
    update_monodepth_relative_pose_options(opt_dict, opt);

    MonoDepthImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        opt.ransac.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_shared_focal_monodepth_relative_pose(points2D_1, points2D_2, depth_1, depth_2, pp, opt,
                                                                      &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<ImagePair, py::dict>
estimate_varying_focal_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                             const std::vector<Eigen::Vector2d> &points2D_2,
                                             const Eigen::Vector2d &pp1, const Eigen::Vector2d &pp2,
                                             const py::dict &opt_dict,
                                             const std::optional<ImagePair> &initial_image_pair) {

    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);

    ImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        opt.ransac.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats =
        estimate_varying_focal_relative_pose(points2D_1, points2D_2, pp1, pp2, opt, &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<MonoDepthImagePair, py::dict> estimate_monodepth_varying_focal_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const Eigen::Vector2d &pp1,
    const Eigen::Vector2d &pp2, const py::dict &opt_dict,
    const std::optional<MonoDepthImagePair> &initial_image_pair) {

    MonoDepthRelativePoseOptions opt;
    update_monodepth_relative_pose_options(opt_dict, opt);

    MonoDepthImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        opt.ransac.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_varying_focal_monodepth_relative_pose(points2D_1, points2D_2, depth_1, depth_2, pp1,
                                                                       pp2, opt, &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<CameraPose, py::dict> refine_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                             const std::vector<Eigen::Vector2d> &points2D_2,
                                                             const CameraPose &initial_pose, const Camera &camera1,
                                                             const Camera &camera2, const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Normalize image points
    std::vector<Eigen::Vector2d> x1_calib = points2D_1;
    std::vector<Eigen::Vector2d> x2_calib = points2D_2;

    for (size_t i = 0; i < x1_calib.size(); ++i) {
        camera1.unproject(points2D_1[i], &x1_calib[i]);
        camera2.unproject(points2D_2[i], &x2_calib[i]);
    }
    bundle_opt.loss_scale *= (1.0 / camera1.focal() + 1.0 / camera2.focal()) * 0.5;

    CameraPose refined_pose = initial_pose;
    BundleStats stats = refine_relpose(x1_calib, x2_calib, &refined_pose, bundle_opt);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

std::pair<CameraPose, py::dict> refine_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                             const std::vector<Eigen::Vector2d> &points2D_2,
                                                             const CameraPose &initial_pose,
                                                             const py::dict &camera1_dict, const py::dict &camera2_dict,
                                                             const py::dict &bundle_opt_dict) {

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);
    return refine_relative_pose_wrapper(points2D_1, points2D_2, initial_pose, camera1, camera2, bundle_opt_dict);
}

std::pair<Eigen::Matrix3d, py::dict> estimate_fundamental_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                  const std::vector<Eigen::Vector2d> &points2D_2,
                                                                  const py::dict &opt_dict,
                                                                  const std::optional<Eigen::Matrix3d> &initial_F) {
    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);

    Eigen::Matrix3d F;
    if (initial_F.has_value()) {
        F = initial_F.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_fundamental(points2D_1, points2D_2, opt, &F, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(F, output_dict);
}

std::pair<Eigen::Matrix3d, py::dict> refine_fundamental_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                const std::vector<Eigen::Vector2d> &points2D_2,
                                                                const Eigen::Matrix3d &initial_F,
                                                                const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Normalize image points
    std::vector<Eigen::Vector2d> x1_norm = points2D_1;
    std::vector<Eigen::Vector2d> x2_norm = points2D_2;

    Eigen::Matrix3d T1, T2;
    double scale = normalize_points(x1_norm, x2_norm, T1, T2, true, true, true);
    BundleOptions bundle_opt_scaled = bundle_opt;
    bundle_opt_scaled.loss_scale /= scale;

    Eigen::Matrix3d refined_F = T2.transpose().inverse() * initial_F * T1.inverse();
    BundleStats stats = refine_fundamental(x1_norm, x2_norm, &refined_F, bundle_opt_scaled);

    refined_F = T2.transpose() * refined_F * T1;
    refined_F /= refined_F.norm();

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_F, output_dict);
}

std::pair<CameraPose, py::dict> estimate_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
    const std::vector<Camera> &cameras1, const std::vector<CameraPose> &camera2_ext,
    const std::vector<Camera> &cameras2, const py::dict &opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    RelativePoseOptions opt;
    update_relative_pose_options(opt_dict, opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<std::vector<char>> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_generalized_relative_pose(matches, camera1_ext, cameras1, camera2_ext, cameras2,
                                                           opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vectors(inlier_mask);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict> estimate_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
    const std::vector<py::dict> &cameras1_dict, const std::vector<CameraPose> &camera2_ext,
    const std::vector<py::dict> &cameras2_dict, const py::dict &opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    std::vector<Camera> cameras1, cameras2;
    for (const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for (const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    return estimate_generalized_relative_pose_wrapper(matches, camera1_ext, cameras1, camera2_ext, cameras2,
                                                      opt_dict, initial_pose);
}

std::pair<CameraPose, py::dict> refine_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const CameraPose &initial_pose,
    const std::vector<CameraPose> &camera1_ext, const std::vector<Camera> &cameras1,
    const std::vector<CameraPose> &camera2_ext, const std::vector<Camera> &cameras2, const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Compute normalized matches
    std::vector<PairwiseMatches> calib_matches = matches;
    for (PairwiseMatches &m : calib_matches) {
        for (size_t k = 0; k < m.x1.size(); ++k) {
            cameras1[m.cam_id1].unproject(m.x1[k], &m.x1[k]);
            cameras2[m.cam_id2].unproject(m.x2[k], &m.x2[k]);
        }
    }

    double scaling_factor = 0;
    for (size_t k = 0; k < cameras1.size(); ++k) {
        scaling_factor += 1.0 / cameras1[k].focal();
    }
    for (size_t k = 0; k < cameras2.size(); ++k) {
        scaling_factor += 1.0 / cameras2[k].focal();
    }
    scaling_factor /= cameras1.size() + cameras2.size();

    bundle_opt.loss_scale *= scaling_factor;

    CameraPose refined_pose = initial_pose;
    BundleStats stats = refine_generalized_relpose(calib_matches, camera1_ext, camera2_ext, &refined_pose, bundle_opt);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

std::pair<CameraPose, py::dict> refine_generalized_relative_pose_wrapper(const std::vector<PairwiseMatches> &matches,
                                                                         const CameraPose &initial_pose,
                                                                         const std::vector<CameraPose> &camera1_ext,
                                                                         const std::vector<py::dict> &cameras1_dict,
                                                                         const std::vector<CameraPose> &camera2_ext,
                                                                         const std::vector<py::dict> &cameras2_dict,
                                                                         const py::dict &bundle_opt_dict) {

    std::vector<Camera> cameras1, cameras2;
    for (const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for (const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    return refine_generalized_relative_pose_wrapper(matches, initial_pose, camera1_ext, cameras1, camera2_ext, cameras2,
                                                    bundle_opt_dict);
}

std::pair<std::vector<CameraPose>, std::vector<Point3D>> motion_from_homography_wrapper(Eigen::Matrix3d &H) {
    std::vector<CameraPose> poses;
    std::vector<Point3D> normals;
    motion_from_homography(H, &poses, &normals);
    return std::make_pair(poses, normals);
}

std::tuple<Camera, Camera, int> focals_from_fundamental_iterative_wrapper(const Eigen::Matrix3d F,
                                                                          const py::dict &camera1_dict,
                                                                          const py::dict &camera2_dict,
                                                                          const int max_iters,
                                                                          const Eigen::Vector4d &weights) {

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    py::gil_scoped_release release;
    return focals_from_fundamental_iterative(F, camera1, camera2, max_iters, weights);
}

} // namespace

void register_relative_pose(py::module &m) {
    m.def("estimate_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &, const Camera &,
                            const Camera &, const py::dict &, const std::optional<CameraPose> &>(
              &estimate_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1"), py::arg("camera2"),
          py::arg("opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Relative pose estimation with non-linear refinement.");

    m.def(
        "estimate_relative_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &, const py::dict &,
                          const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
            &estimate_relative_pose_wrapper),
        py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1_dict"), py::arg("camera2_dict"),
        py::arg("opt") = py::dict(), py::arg("initial_pose") = py::none(),
        "Relative pose estimation with non-linear refinement.");

    m.def("estimate_monodepth_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<double> &, const std::vector<double> &, const Camera &, const Camera &,
                            const py::dict &, const std::optional<MonoDepthTwoViewGeometry> &>(
              &estimate_monodepth_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("camera1"),
          py::arg("camera2"), py::arg("opt") = py::dict(),
          py::arg("initial_pose") = py::none(), "Pose estimation using depth estimates with non-linear refinement.");

    m.def(
        "estimate_monodepth_relative_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                          const std::vector<double> &, const std::vector<double> &, const py::dict &, const py::dict &,
                          const py::dict &, const std::optional<MonoDepthTwoViewGeometry> &>(
            &estimate_monodepth_relative_pose_wrapper),
        py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("camera1_dict"),
        py::arg("camera2_dict"), py::arg("opt") = py::dict(),
        py::arg("initial_pose") = py::none(),
        "Relative pose estimation using depth estimates with non-linear refinement.");

    using V2 = std::vector<Eigen::Vector2d>;
    using V4 = std::vector<Eigen::Vector4d>;
    m.def("estimate_relative_pose_lines",
          py::overload_cast<const V2 &, const V2 &, const V4 &, const V4 &, const V2 &, const V2 &, const Camera &,
                            const Camera &, const py::dict &>(&estimate_relative_pose_lines_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("lines2D_1"), py::arg("lines2D_2"),
          py::arg("line_depths_1"), py::arg("line_depths_2"), py::arg("camera1"), py::arg("camera2"),
          py::arg("opt") = py::dict(),
          "Relative pose from points and lines with depth (1L3P / 2L2P / 5pt, optional line residuals in scoring, "
          "LO and refinement).");
    m.def("estimate_relative_pose_lines",
          py::overload_cast<const V2 &, const V2 &, const V4 &, const V4 &, const V2 &, const V2 &, const py::dict &,
                            const py::dict &, const py::dict &>(&estimate_relative_pose_lines_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("lines2D_1"), py::arg("lines2D_2"),
          py::arg("line_depths_1"), py::arg("line_depths_2"), py::arg("camera1_dict"), py::arg("camera2_dict"),
          py::arg("opt") = py::dict(),
          "Relative pose from points and lines with depth (1L3P / 2L2P / 5pt, optional line residuals in scoring, "
          "LO and refinement).");
    m.def("estimate_monodepth_relative_pose_lines",
          py::overload_cast<const V2 &, const V2 &, const std::vector<double> &, const std::vector<double> &,
                            const V4 &, const V4 &, const V2 &, const V2 &, const Camera &, const Camera &,
                            const py::dict &>(&estimate_monodepth_relative_pose_lines_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("lines2D_1"),
          py::arg("lines2D_2"), py::arg("line_depths_1"), py::arg("line_depths_2"), py::arg("camera1"),
          py::arg("camera2"), py::arg("opt") = py::dict(),
          "RePoseD with lines with depth (optional line residuals in scoring, LO and refinement).");
    m.def("estimate_monodepth_relative_pose_lines",
          py::overload_cast<const V2 &, const V2 &, const std::vector<double> &, const std::vector<double> &,
                            const V4 &, const V4 &, const V2 &, const V2 &, const py::dict &, const py::dict &,
                            const py::dict &>(&estimate_monodepth_relative_pose_lines_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("lines2D_1"),
          py::arg("lines2D_2"), py::arg("line_depths_1"), py::arg("line_depths_2"), py::arg("camera1_dict"),
          py::arg("camera2_dict"), py::arg("opt") = py::dict(),
          "RePoseD with lines with depth (optional line residuals in scoring, LO and refinement).");

    m.def("estimate_monodepth_relative_pose_segments", &estimate_monodepth_relative_pose_segments_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("segments2D_1"),
          py::arg("segments2D_2"), py::arg("segment_depths_1"), py::arg("segment_depths_2"), py::arg("camera1_dict"),
          py::arg("camera2_dict"), py::arg("opt") = py::dict(),
          "RePoseD with minimal samples from matched segments with depth (segment_solver: 0 - 1S1P, 1 - 2S), score and "
          "refinement from the points only.");

    m.def("estimate_threeview_relative_pose", &estimate_threeview_relative_pose_wrapper, py::arg("points2D_1"),
          py::arg("points2D_2"), py::arg("points2D_3"), py::arg("depth_1"), py::arg("depth_2"), py::arg("depth_3"),
          py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines2D_3"), py::arg("line_depths_1"),
          py::arg("line_depths_2"), py::arg("line_depths_3"), py::arg("feature_points_1"), py::arg("feature_points_2"),
          py::arg("feature_points_3"), py::arg("feature_lines_1"), py::arg("feature_lines_2"),
          py::arg("feature_lines_3"), py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("camera3_dict"),
          py::arg("opt") = py::dict(),
          "Three view relative pose (5pt + P3P, RePoseD or 2-SIFT) with Sampson, symmetric reprojection and line "
          "residuals in the refinement.");

    m.def("estimate_shared_focal_relative_pose", &estimate_shared_focal_relative_pose_wrapper, py::arg("points2D_1"),
          py::arg("points2D_2"), py::arg("pp") = Eigen::Vector2d::Zero(), py::arg("opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with unknown equal focal lengths with non-linear refinement.");

    m.def("estimate_varying_focal_relative_pose", &estimate_varying_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("pp1") = Eigen::Vector2d::Zero(),
          py::arg("pp2") = Eigen::Vector2d::Zero(), py::arg("opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with two different unknown focal lengths with non-linear refinement.");

    m.def("estimate_monodepth_shared_focal_relative_pose", &estimate_monodepth_shared_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"),
          py::arg("pp") = Eigen::Vector2d::Zero(), py::arg("opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with depth estimates and unknown equal focal lengths with non-linear refinement.");

    m.def("estimate_monodepth_varying_focal_relative_pose", &estimate_monodepth_varying_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"),
          py::arg("pp1") = Eigen::Vector2d::Zero(), py::arg("pp2") = Eigen::Vector2d::Zero(),
          py::arg("opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with depth estimates and unknown different focal lengths with non-linear "
          "refinement.");

    m.def("estimate_fundamental", &estimate_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("opt") = py::dict(), py::arg("initial_F") = py::none(),
          "Fundamental matrix estimation with non-linear refinement. Note: if you have known intrinsics you should use "
          "estimate_relative_pose instead!");

    m.def("estimate_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const std::vector<CameraPose> &,
                            const std::vector<Camera> &, const std::vector<CameraPose> &, const std::vector<Camera> &,
                            const py::dict &, const std::optional<CameraPose> &>(
              &estimate_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("camera1_ext"), py::arg("cameras1"), py::arg("camera2_ext"), py::arg("cameras2"),
          py::arg("opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Generalized relative pose estimation with non-linear refinement.");
    m.def(
        "estimate_generalized_relative_pose",
        py::overload_cast<const std::vector<PairwiseMatches> &, const std::vector<CameraPose> &,
                          const std::vector<py::dict> &, const std::vector<CameraPose> &, const std::vector<py::dict> &,
                          const py::dict &, const std::optional<CameraPose> &>(
            &estimate_generalized_relative_pose_wrapper),
        py::arg("matches"), py::arg("camera1_ext"), py::arg("camera1_dict"), py::arg("camera2_ext"),
        py::arg("camera2_dict"), py::arg("opt") = py::dict(),
        py::arg("initial_pose") = py::none(), "Generalized relative pose estimation with non-linear refinement.");

    m.def("motion_from_homography", &motion_from_homography_wrapper, py::arg("H"));
    m.def("focals_from_fundamental", &focals_from_fundamental, py::arg("F"), py::arg("pp1"), py::arg("pp2"));
    m.def("focals_from_fundamental_iterative", &focals_from_fundamental_iterative, py::arg("F"), py::arg("camera1"),
          py::arg("camera2"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));
    m.def("focals_from_fundamental_iterative", &focals_from_fundamental_iterative_wrapper, py::arg("F"),
          py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));

    // Stand-alone non-linear refinement
    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const CameraPose &, const Camera &, const Camera &, const py::dict &>(
              &refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1"), py::arg("camera2"),
          py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");
    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const CameraPose &, const py::dict &, const py::dict &, const py::dict &>(
              &refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1_dict"),
          py::arg("camera2_dict"), py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");

    m.def("refine_fundamental", &refine_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("initial_F"), py::arg("bundle_options") = py::dict(), "Fundamental matrix non-linear refinement.");

    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const CameraPose &, const std::vector<CameraPose> &,
                            const std::vector<Camera> &, const std::vector<CameraPose> &, const std::vector<Camera> &,
                            const py::dict &>(&refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("cameras1"),
          py::arg("camera2_ext"), py::arg("cameras2"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");
    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const CameraPose &, const std::vector<CameraPose> &,
                            const std::vector<py::dict> &, const std::vector<CameraPose> &,
                            const std::vector<py::dict> &, const py::dict &>(&refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("camera1_dict"),
          py::arg("camera2_ext"), py::arg("camera2_dict"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");
}

} // namespace poselib
