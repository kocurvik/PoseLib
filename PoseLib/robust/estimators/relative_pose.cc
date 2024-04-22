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

#include "relative_pose.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_6pt_focal.h"
#include "PoseLib/solvers/relpose_7pt.h"

#include <iostream>

namespace poselib {

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_5pt(x1s, x2s, models);
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 5) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }
    refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt);
}

void ThreeViewRelativePoseEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = x1[sample[k]].homogeneous().normalized();
        x2n[k] = x2[sample[k]].homogeneous().normalized();
    }

    // if we use 4 pts we approx the last as the mean of the prev 4 pts
    if (sample_sz == 4){
//        x1n[4] = (0.25 * (x1[sample[0]] + x1[sample[1]] + x1[sample[2]] + x1[sample[3]])).homogeneous().normalized();
//        x2n[4] = (0.25 * (x2[sample[0]] + x2[sample[1]] + x2[sample[2]] + x2[sample[3]])).homogeneous().normalized();
        x1n[4] = ((x1[sample[0]] + x1[sample[1]] + x1[sample[2]]) / 3.0).homogeneous().normalized();
        x2n[4] = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0).homogeneous().normalized();
    }

    for (size_t k = 0; k < sample_sz_13; ++k) {
        x1s[k] = x1[sample[k]].homogeneous();
        x2s[k] = x2[sample[k]].homogeneous();
        x3s[k] = x3[sample[k]].homogeneous().normalized();
    }
    estimate_models(models);

    if (sample_sz == 4 and opt.delta > 0.0){
        delta(models);

    }
}

void ThreeViewRelativePoseEstimator::delta(std::vector<ThreeViewCameraPose> *models) {
    Point2D x2n4 = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0);
    double mx2 = x2n4(0), my2 = x2n4(1);

    double tr_xA = x2[sample[0]](0), tr_yA = x2[sample[0]](1);
    double tr_xB = x2[sample[1]](0), tr_yB = x2[sample[1]](1);
    double tr_xC = x2[sample[2]](0), tr_yC = x2[sample[2]](1);

    double slopeAB = (tr_yB - tr_yA) / (tr_xB - tr_xA);
    double slopeAC = (tr_yC - tr_yA) / (tr_xC - tr_xA);
    double slopeBC = (tr_yC - tr_yB) / (tr_xC - tr_xB);

    double interceptAB = tr_yA - slopeAB * tr_xA;
    double interceptAC = tr_yA - slopeAC * tr_xA;
    double interceptBC = tr_yB - slopeBC * tr_xB;

    // find y coord in lines AB, AC, and BC, in the 2nd view for x=mean
    double yAB = slopeAB * mx2 + interceptAB;
    double yAC = slopeAC * mx2 + interceptAC;
    double yBC = slopeBC * mx2 + interceptBC;

    // find x coord in lines AB, AC, and BC, in the 2nd view for y=mean
    double xAB = (my2 - interceptAB) / slopeAB;
    double xAC = (my2 - interceptAC) / slopeAC;
    double xBC = (my2 - interceptBC) / slopeBC;

    // find min and max y and min and max x of thriangle vertices
    double min_y = std::min(std::min(tr_yA, tr_yB), tr_yC);
    double max_y = std::max(std::max(tr_yA, tr_yB), tr_yC);
    double min_x = std::min(std::min(tr_xA, tr_xB), tr_xC);
    double max_x = std::max(std::max(tr_xA, tr_xB), tr_xC);

    double max_distY;
    if (yAB>max_y || yAB<min_y) {
        max_distY = std::abs(yAC-yBC);
    } else if (yAC>max_y || yAC<min_y) {
        max_distY = std::abs(yAB-yBC);
    } else {
        max_distY = std::abs(yAB-yAC);
    }

    double max_distX;
    if (xAB>max_x || xAB<min_x) {
        max_distX = std::abs(xAC-xBC);
    } else if (xAC>max_x || xAC<min_x) {
        max_distX = std::abs(xAB-xBC);
    } else {
        max_distX = std::abs(xAB-xAC);
    }

    int idx;
    double scale;
    if (max_distX > max_distY){
        scale = max_distX;
        idx = 0;
    } else {
        scale = max_distY;
        idx = 1;
    }

//    Point2D x2n4 = ((x2[sample[0]] + x2[sample[1]] + x2[sample[2]]) / 3.0);
//
//    double min_x = std::min({x2[sample[0]][0], x2[sample[1]][0], x2[sample[2]][0]});
//    double min_y = std::min({x2[sample[0]][1], x2[sample[1]][1], x2[sample[2]][1]});
//    double max_x = std::max({x2[sample[0]][0], x2[sample[1]][0], x2[sample[2]][0]});
//    double max_y = std::max({x2[sample[0]][1], x2[sample[1]][1], x2[sample[2]][1]});
//
//    int idx;
//    double scale;
//    if ((max_x - min_x) > (max_y - min_y)) {
//        scale = max_x - min_x;
//        idx = 0;
//    } else {
//        scale = max_y - min_y;
//        idx = 1;
//    }

    x2n4(idx) += opt.delta * scale;
    x2n[4] = x2n4.homogeneous().normalized();
    estimate_models(models);

    x2n4(idx) -= 2 * opt.delta * scale;
    x2n[4] = x2n4.homogeneous().normalized();
    estimate_models(models);
}

void ThreeViewRelativePoseEstimator::estimate_models(std::vector<ThreeViewCameraPose> *models) {
    std::vector<CameraPose> models12;
    relpose_5pt(x1n, x2n, &models12);

    std::vector<Point3D> triangulated_12;
    triangulated_12.reserve(3);

    for (CameraPose pose12 : models12){
        for (size_t i = 0; i < sample_sz_13; i++){
            triangulated_12[i] = triangulate(pose12, x1s[i], x2s[i]);
        }

        std::vector<CameraPose> models13;
        p3p(x3s, triangulated_12, &models13);

        for (CameraPose pose13 : models13){
            ThreeViewCameraPose three_view_pose = ThreeViewCameraPose(pose12, pose13);
            if (opt.threeview_check){
                size_t inlier_4p_13 = 0;
                size_t inlier_4p_23 = 0;
                std::vector<Point2D> x1c = {x1[sample[3]]};
                std::vector<Point2D> x2c = {x2[sample[3]]};
                std::vector<Point2D> x3c = {x3[sample[3]]};
                compute_sampson_msac_score(three_view_pose.pose13, x1c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_13);
                compute_sampson_msac_score(three_view_pose.pose23(), x2c, x3c, 4 * opt.max_epipolar_error * opt.max_epipolar_error, &inlier_4p_23);
                if (inlier_4p_13 + inlier_4p_23 < 2) {
                    continue;
                }
            }

            if (opt.inner_refine > 0) {
                inner_refine(&three_view_pose);
            }
            models->emplace_back(three_view_pose);
        }
    }
}

void ThreeViewRelativePoseEstimator::inner_refine(ThreeViewCameraPose *three_view_pose) const {
    std::vector<Point2D> x1r, x2r, x3r;
    x1r.resize(4);
    x2r.resize(4);
    x3r.resize(4);
    for (size_t k = 0; k < 4; ++k) {
        x1r[k] = x1[sample[k]];
        x2r[k] = x2[sample[k]];
        x3r[k] = x3[sample[k]];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::CAUCHY;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.inner_refine;
    refine_3v_relpose(x1r, x2r, x3r, three_view_pose, bundle_opt);
}

double ThreeViewRelativePoseEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13, inlier_count23;
    // TODO: calc inliers better w/o redundant computation

    double score12 = compute_sampson_msac_score(three_view_pose.pose12, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(three_view_pose.pose13, x1, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);
    double score23 = compute_sampson_msac_score(three_view_pose.pose23(), x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count23);

    std::vector<char> inliers;
    *inlier_count = get_inliers(three_view_pose, x1, x2, x3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
    return score12 + score13 + score23;
}

void ThreeViewRelativePoseEstimator::refine_model(ThreeViewCameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, x3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
            x3_inlier.push_back(x3[pt_k]);
        }
    }

//    size_t inliers_before;
//    double score_before = score_model(*pose, &inliers_before);
//    std::cout << "Score before: " << score_before <<std::endl;
//    std::cout << "Inliers before: " << inliers_before <<std::endl;

    refine_3v_relpose(x1_inlier, x2_inlier, x3_inlier, pose, bundle_opt);

//    size_t inliers_after;
//    double score_after = score_model(*pose, &inliers_after);
//    std::cout << "Score after: " << score_after <<std::endl;
//    std::cout << "Inliers after: " << inliers_after <<std::endl;
}

void SharedFocalRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_6pt_focal(x1s, x2s, models);
}

double SharedFocalRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
    // K_inv << 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    Eigen::Matrix3d K_inv;
    // K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, calib_pose->camera.focal();
    K_inv << 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(F, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 6) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }

    refine_shared_focal_relpose(x1_inlier, x2_inlier, image_pair, bundle_opt);
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // TODO replace by general 6pt solver?

    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_sampson_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error,
                                           &local_inlier_count);
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers with a relaxed threshold
        std::vector<char> inliers;
        int num_inl = get_inliers(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }

    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}

void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);

    if (opt.real_focal_check) {
        for (int i = models->size() - 1; i >= 0; i--) {
            if (!calculate_RFC((*models)[i]))
                models->erase(models->begin() + i);
        }
    }
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

} // namespace poselib
