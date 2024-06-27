//
// Created by kocur on 26-Jun-24.
//

#include "threeview_hc.h"
#include "homotopy.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/misc/essential.h"

#include <cassert>
#include <iostream>
#include <fstream>

namespace hruby
{
    Eigen::Matrix3d product(Eigen::Matrix3d ZZ, Eigen::Matrix3d Zi)
    {
        Eigen::Matrix3d R;

        R(0,0) = ZZ(0,0)*Zi(0,0) + ZZ(0,1)*Zi(1,0) + ZZ(0,2)*Zi(2,0);
        R(0,1) = ZZ(0,0)*Zi(0,1) + ZZ(0,1)*Zi(1,1) + ZZ(0,2)*Zi(2,1);
        R(0,2) = ZZ(0,0)*Zi(0,2) + ZZ(0,1)*Zi(1,2) + ZZ(0,2)*Zi(2,2);

        R(1,0) = ZZ(1,0)*Zi(0,0) + ZZ(1,1)*Zi(1,0) + ZZ(1,2)*Zi(2,0);
        R(1,1) = ZZ(1,0)*Zi(0,1) + ZZ(1,1)*Zi(1,1) + ZZ(1,2)*Zi(2,1);
        R(1,2) = ZZ(1,0)*Zi(0,2) + ZZ(1,1)*Zi(1,2) + ZZ(1,2)*Zi(2,2);

        R(2,0) = ZZ(2,0)*Zi(0,0) + ZZ(2,1)*Zi(1,0) + ZZ(2,2)*Zi(2,0);
        R(2,1) = ZZ(2,0)*Zi(0,1) + ZZ(2,1)*Zi(1,1) + ZZ(2,2)*Zi(2,1);
        R(2,2) = ZZ(2,0)*Zi(0,2) + ZZ(2,1)*Zi(1,2) + ZZ(2,2)*Zi(2,2);

        return R;
    }

    Eigen::Vector3d product_(Eigen::Matrix3d R, Eigen::Vector3d x1)
    {
        Eigen::Vector3d Rx1;

        Rx1(0) = R(0,0)*x1(0) + R(0,1)*x1(1) + R(0,2)*x1(2);
        Rx1(1) = R(1,0)*x1(0) + R(1,1)*x1(1) + R(1,2)*x1(2);
        Rx1(2) = R(2,0)*x1(0) + R(2,1)*x1(1) + R(2,2)*x1(2);

        return Rx1;
    }

}

namespace poselib {
void poselib::ThreeViewRelativePoseHCEstimator::generate_models(std::vector<ThreeViewCameraPose> *models) {
    sampler.generate_sample(&sample);

    for (size_t k = 0; k < sample_sz; ++k) {
        x1n[k] = xx1[sample[k]].homogeneous().normalized();
        x2n[k] = xx2[sample[k]].homogeneous().normalized();
        x3n[k] = xx3[sample[k]].homogeneous().normalized();
    }

    models->clear();

    std::vector<Eigen::Vector2d> pts1(4), pts2(4), pts3(4);

    for (size_t k = 0; k < sample_sz; k++) {
        pts1[k] = xx1[sample[k]];
        pts2[k] = xx2[sample[k]];
        pts3[k] = xx3[sample[k]];
    }

    double params[48];
    static double solution[12];
    int num_steps;

    // Code adapted from
    // https://github.com/petrhruby97/learning_minimal/blob/main/4p3v/src/data_sampler.cxx
    //normalize the sampled problem
    std::vector<Eigen::Vector3d> P(4);
    std::vector<Eigen::Vector3d> Q(4);
    std::vector<Eigen::Vector3d> R(4);
    for(int l = 0; l < 4; ++l) {
        P[l] = pts1[l].homogeneous();
        Q[l] = pts2[l].homogeneous();
        R[l] = pts3[l].homogeneous();
    }

    std::vector<Eigen::Vector2d> Pn(4);
    std::vector<Eigen::Vector2d> Qn(4);
    std::vector<Eigen::Vector2d> Rn(4);

    Eigen::Matrix3d CP;
    Eigen::Matrix3d CQ;
    Eigen::Matrix3d CR;

    int perm3[3];
    int perm4[4];
    normalize(P, Q, R, Pn, Qn, Rn, CP, CQ, CR, perm3, perm4);

    // keeps track which view moved where
    int ix[3];
    ix[perm3[0]] = 0;
    ix[perm3[1]] = 1;
    ix[perm3[2]] = 2;

    double problem[24];

    std::vector<Eigen::Vector2d> points1_n(4);
    std::vector<Eigen::Vector2d> points2_n(4);
    std::vector<Eigen::Vector2d> points3_n(4);

    for(int a=0;a<4;++a)
    {
        points1_n[a] = Pn[a];
        points2_n[a] = Qn[a];
        points3_n[a] = Rn[a];

        problem[a] = points1_n[a](0);
        problem[a+4] = points1_n[a](1);
        problem[a+8] = points2_n[a](0);
        problem[a+12] = points2_n[a](1);
        problem[a+16] = points3_n[a](0);
        problem[a+20] = points3_n[a](1);
    }

//     std::cout << " ViewTripletHrubySolver::estimateModel: normalization done " << std::endl;

    // select the starting point using the neural network
    float orig[24];
    for(int a=0;a<24;++a)
        orig[a] = (float)problem[a];

    //evaluate the MLP
    Eigen::Map<Eigen::VectorXf> input_n2(orig,24);
    Eigen::VectorXf input_ = input_n2;
    Eigen::VectorXf output_;

    // The following copy operations will add overhead, but ensure const
    // std::vector<std::vector<float>> ws__ = ws;
    // std::vector<std::vector<float>> bs__ = bs;
    // std::vector<std::vector<float>> ps__ = ps;

    int layers = num_layers;
    for(int i=0;i<layers;++i)
    {
        const float * ws_ = &ws[i][0];
        const float * bs_ = &bs[i][0];
        const float * ps_ = &ps[i][0];
        const Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned > weights = Eigen::Map< const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>, Eigen::Aligned >(ws_,a_[i],b_[i]);
        const Eigen::Map<const Eigen::VectorXf> bias = Eigen::Map<const Eigen::VectorXf>(bs_,a_[i]);

        output_ = weights*input_+bias;

        if(i==layers-1) break;

        const Eigen::Map<const Eigen::VectorXf> prelu = Eigen::Map<const Eigen::VectorXf>(ps_,a_[i]);
        input_ = output_.cwiseMax(output_.cwiseProduct(prelu));
    }

//     std::cout << " ViewTripletHrubySolver::estimateModel: done with network: " << layers << std::endl;

    //find the output with the highest score
    double best = -1000;
    int p = 0;

    for(int j=1;j<a_[layers-1];++j)
    {
        if(output_(j) > best)
        {
            best = output_(j);
            p = j;
        }
    }
    p = p-1;
//    std::cout << " ViewTripletHrubySolver::estimateModel: found best solution: " << p << std::endl;
    if(p==-1) return; // tsattler: I assume this means failure. Not 100% sure.
    // continue;

    //copy the start problem
    for(int a=0;a<24;a++)
    {
        params[a] = problems_anchors[p][a];
        params[a+24] = problem[a];
    }

    //copy the start solution
    double start[12];
    for(int a=0;a<12;++a)
        start[a] = start_anchors[p][a];

    //track the problem
    homotopy::track_settings settings;
    int status = homotopy::track(settings, start, params, solution, &num_steps);

//     std::cout << " ViewTripletHrubySolver::estimateModel: tracked solution " << status << std::endl;

    //evaluate the solution
    if(status == 2)
    {
        // valid solution, so let's decompose the solution and construt two
        // essential matrices.
        // 1st: re-order the original datapoints and the solutions accordingly.
        double params_reordered[24];
        for (int idx_ = 0; idx_ < 3; ++idx_) {
            for (int pt_ = 0; pt_ < 4; ++pt_) {
                params_reordered[ix[idx_] * 8 + pt_] = params[24 + idx_ * 8 + pt_];
                params_reordered[ix[idx_] * 8 + 4 + pt_] = params[24 + idx_ * 8 + 4 + pt_];
            }
        }
        double solution_reordered[12];
        solution_reordered[ix[0] * 4] = 1.0;
        solution_reordered[ix[0] * 4 + 1] = solution[0];
        solution_reordered[ix[0] * 4 + 2] = solution[1];
        solution_reordered[ix[0] * 4 + 3] = solution[2];
        solution_reordered[ix[1] * 4] = solution[3];
        solution_reordered[ix[1] * 4 + 1] = solution[4];
        solution_reordered[ix[1] * 4 + 2] = solution[5];
        solution_reordered[ix[1] * 4 + 3] = solution[6];
        solution_reordered[ix[2] * 4] = solution[7];
        solution_reordered[ix[2] * 4 + 1] = solution[8];
        solution_reordered[ix[2] * 4 + 2] = solution[9];
        solution_reordered[ix[2] * 4 + 3] = solution[10];

        // 2nd: extracts the relative transformations and composes the essential
        // matrices.
        Eigen::Matrix3d R12, R13;
        Eigen::Vector3d t12, t13;
        extract_pose12(params_reordered, solution_reordered, R12, t12);
        extract_pose13(params_reordered, solution_reordered, R13, t13);

        // 3rd: account for normalizing transformations.
        std::vector<Eigen::Matrix3d> norm_trans(3);
        norm_trans[ix[0]] = CP;
        norm_trans[ix[1]] = CQ;
        norm_trans[ix[2]] = CR;

//        std::cout << "N0" << std::endl << norm_trans[0] << std::endl;
//        std::cout << "N1" << std::endl << norm_trans[1] << std::endl;
//        std::cout << "N2" << std::endl << norm_trans[2] << std::endl;

        Eigen::Matrix3d RR12 = norm_trans[1].transpose() * R12 * norm_trans[0];
        Eigen::Matrix3d RR13 = norm_trans[2].transpose() * R13 * norm_trans[0];
        CameraPose pose12(RR12, norm_trans[1].transpose() * t12);
        CameraPose pose13(RR13, norm_trans[2].transpose() * t13);
        models->emplace_back(pose12, pose13);

//        Eigen::Matrix3d t12x, t13x;
//        t12x << 0, -t12(2), t12(1), t12(2), 0, -t12(0), -t12(1), t12(0), 0;
//        t13x << 0, -t13(2), t13(1), t13(2), 0, -t13(0), -t13(1), t13(0), 0;
//
//        Eigen::Matrix3d E12 = t12x * R12;
//        Eigen::Matrix3d E13 = t13x * R13;
//
//        E12 = norm_trans[1].transpose() * E12 * norm_trans[0];
//        E13 = norm_trans[2].transpose() * E13 * norm_trans[0];
//
//        std::vector<CameraPose> poses12, poses13;
//        motion_from_essential(E12, x1n, x2n, &poses12);
//        motion_from_essential(E13, x1n, x3n, &poses13);
//
//        models->reserve(poses12.size() * poses13.size());
//
//        for (const CameraPose& pose12: poses12){
//            for (const CameraPose& pose13: poses13){
//                models->emplace_back(pose12, pose13);
//            }
//        }

//        std::cout << "Models size: " << models->size() << std::endl;


    } else {
        return;  // tsattler: this looks like failure.
    }
    // std::cout << " ViewTripletHrubySolver::estimateModel: extracted Es " << std::endl;
}

double poselib::ThreeViewRelativePoseHCEstimator::score_model(const ThreeViewCameraPose &three_view_pose, size_t *inlier_count) const {
    size_t inlier_count12, inlier_count13, inlier_count23;
    // TODO: calc inliers better w/o redundant computation

    double score12 = compute_sampson_msac_score(three_view_pose.pose12, xx1, xx2, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count12);
    double score13 = compute_sampson_msac_score(three_view_pose.pose13, xx1, xx3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count13);
    double score23 = compute_sampson_msac_score(three_view_pose.pose23(), xx2, xx3, opt.max_epipolar_error * opt.max_epipolar_error, &inlier_count23);

//    std::cout << "Pose 12 R: " << std::endl << three_view_pose.pose12.R() << std::endl;
//    std::cout << "Pose 12 t: " << std::endl << three_view_pose.pose12.t.transpose() << std::endl;
//    std::cout << "Pose 13 R: " << std::endl << three_view_pose.pose13.R() << std::endl;
//    std::cout << "Pose 13 t: " << std::endl << three_view_pose.pose13.t.transpose() << std::endl;
//    std::cout << "Pose 23 R: " << std::endl << three_view_pose.pose23().R() << std::endl;
//    std::cout << "Pose 23 t: " << std::endl << three_view_pose.pose23().t.transpose() << std::endl;

    std::vector<char> inliers;
    *inlier_count = get_inliers(three_view_pose, xx1, xx2, xx3, opt.max_epipolar_error * opt.max_epipolar_error, &inliers);
//    std::cout << "Inlier count all: " << *inlier_count << std::endl;
//    std::cout << "Inlier count 12: " << inlier_count12 << std::endl;
//    std::cout << "Inlier count 13: " << inlier_count13 << std::endl;
//    std::cout << "Inlier count 23: " << inlier_count23 << std::endl;
    return score12 + score13 + score23;
}

void poselib::ThreeViewRelativePoseHCEstimator::refine_model(ThreeViewCameraPose *pose) const {
    if (opt.lo_iterations == 0)
        return;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = opt.lo_iterations;
//    bundle_opt.verbose = true;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, xx1, xx2, xx3, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier, x3_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);
    x3_inlier.reserve(num_inl);

    if (num_inl <= 4) {
        return;
    }

    for (size_t pt_k = 0; pt_k < xx1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(xx1[pt_k]);
            x2_inlier.push_back(xx2[pt_k]);
            x3_inlier.push_back(xx3[pt_k]);
        }
    }

    refine_3v_relpose(x1_inlier, x2_inlier, x3_inlier, pose, bundle_opt);
}

bool ThreeViewRelativePoseHCEstimator::load_anchors(const std::string& data_file)
{
    std::ifstream f;
    f.open(data_file);

    if(!f.good())
    {
        f.close();
        std::cout << "Anchor file not available\n";
        return false;
    }

    int n;
    f >> n;
    // std::cout << n << " anchors\n";

    problems_anchors.resize(n);
    start_anchors.resize(n);
    depths_anchors.resize(n);

    //load the problems
    for(int i=0;i<n;i++)
    {
        std::vector<double> problem(24);
        std::vector<double> cst(12);
        std::vector<double> depth(13);

        //load the points
        for(int j=0;j<24;j++)
        {
            double u;
            f >> u;

            problem[j] = u;
        }
        problems_anchors[i] = problem;

        //load the depths and convert them to the solution
        double first_depth;
        f >> first_depth;
        depth[0] = first_depth;
        for(int j=0;j<11;j++)
        {
            double u;
            f >> u;

            cst[j] = u/first_depth;
            depth[j+1] = u;
        }
        double l;
        f >> l;
        cst[11] = l;
        depth[12] = l;

        start_anchors[i] = cst;
        depths_anchors[i] = depth;
    }
    f.close();

    // std::cout << " anchors loaded" << std::endl;
    return true;
}

bool ThreeViewRelativePoseHCEstimator::load_NN(const std::string& nn_file)
{
    std::ifstream fnn;
    fnn.open(nn_file);
    if (!fnn.is_open()) return false;
    fnn >> num_layers;
    int layers = num_layers;
    ws = std::vector<std::vector<float>>(layers);
    bs = std::vector<std::vector<float>>(layers);
    ps = std::vector<std::vector<float>>(layers-1);
    a_ = std::vector<int>(layers);
    b_ = std::vector<int>(layers);
    for(int i=0;i<layers;++i)
    {
        int a;
        int b;
        fnn >> a;
        fnn >> b;
        a_[i] = a;
        b_[i] = b;

        // std::cout << a << " " << b << "\n";

        std::vector<float> __attribute__((aligned(16))) cw(a*b);
        for(int j=0;j<a*b;++j)
        {
            float u;
            fnn >> u;
            cw[j] = u;
        }
        ws[i] = cw;

        fnn >> a;
        fnn >> b;
        std::vector<float> __attribute__((aligned(16))) cb(a);
        for(int j=0;j<a;++j)
        {
            float u;
            fnn >> u;
            cb[j] = u;
        }
        bs[i] = cb;

        if(i==layers-1)
            break;

        fnn >> a;
        fnn >> b;
        std::vector<float> __attribute__((aligned(16))) cp(a);
        for(int j=0;j<a;++j)
        {
            float u;
            fnn >> u;
            cp[j] = u;
        }
        ps[i] = cp;
    }
    fnn.close();

    // std::cout << " NN loaded " << std::endl;
    return true;
}

void ThreeViewRelativePoseHCEstimator::order_points(std::vector<Eigen::Vector3d> &P, int * perm4, int ix) const {
    double angle1 = std::atan2(P[ix](1), P[ix](0));
    if(angle1 < 0)
        angle1 = angle1 + 2*acos(-1.0);

    //obtain the relative angles
    double angles[4];
    for(int i=0;i<4;++i)
    {
        if(i==ix)
        {
            angles[ix] = 7;
            continue;
        }
        //if negative, add 2*pi to obtain a positive number
        double cur_ang = std::atan2(P[i](1), P[i](0));
        if(cur_ang < 0)
            cur_ang = cur_ang + 2*acos(-1.0);
        //subtract the angle of the longest point from other angles (if negative, add 2*pi)
        double ang = cur_ang - angle1;
        if(ang < 0)
            ang = ang + 2*acos(-1.0);
        angles[i] = ang;
    }

    perm4[3] = ix;

    for(int i=0;i<3;++i)
    {
        double min = 7;
        int next = -1;
        for(int j=0;j<4;++j)
        {
            if(angles[j] < min)
            {
                next = j;
                min = angles[j];
            }
        }
        perm4[i] = next;
        angles[next] = 7;
    }
}

void ThreeViewRelativePoseHCEstimator::extract_pose12(double params[24], double solution[12], Eigen::Matrix3d &R, Eigen::Vector3d &t) const
{
    const Eigen::Vector3d x1(solution[0] * params[0], solution[0] * params[4], solution[0]);
    const Eigen::Vector3d x2(solution[1] * params[1], solution[1] * params[5], solution[1]);
    const Eigen::Vector3d x3(solution[2] * params[2], solution[2] * params[6], solution[2]);

    const Eigen::Vector3d y1(solution[4] * params[8], solution[4] * params[12], solution[4]);
    const Eigen::Vector3d y2(solution[5] * params[9], solution[5] * params[13], solution[5]);
    const Eigen::Vector3d y3(solution[6] * params[10], solution[6] * params[14], solution[6]);

    const Eigen::Vector3d z2 = x2-x1;
    const Eigen::Vector3d z3 = x3-x1;
    Eigen::Vector3d z1;// = z2.cross(z3);
    z1(0) = z2(1)*z3(2) - z2(2)*z3(1);
    z1(1) = z2(2)*z3(0) - z2(0)*z3(2);
    z1(2) = z2(0)*z3(1) - z2(1)*z3(0);
    const Eigen::Vector3d zz2 = y2-y1;
    const Eigen::Vector3d zz3 = y3-y1;
    Eigen::Vector3d zz1;// = zz2.cross(zz3);
    zz1(0) = zz2(1)*zz3(2) - zz2(2)*zz3(1);
    zz1(1) = zz2(2)*zz3(0) - zz2(0)*zz3(2);
    zz1(2) = zz2(0)*zz3(1) - zz2(1)*zz3(0);
    Eigen::Matrix3d Z;
    Z << z1, z2, z3;
    Eigen::Matrix3d ZZ;
    ZZ << zz1, zz2, zz3;

    double a = Z(0,0);
    double b = Z(0,1);
    double c = Z(0,2);

    double d = Z(1,0);
    double e = Z(1,1);
    double f = Z(1,2);

    double g = Z(2,0);
    double h = Z(2,1);
    double i = Z(2,2);

    Eigen::Matrix3d Zi;
    Zi(0,0) = e*i-f*h;
    Zi(0,1) = c*h-b*i;
    Zi(0,2) = b*f-c*e;

    Zi(1,0) = f*g-d*i;
    Zi(1,1) = a*i-c*g;
    Zi(1,2) = c*d-a*f;

    Zi(2,0) = d*h-e*g;
    Zi(2,1) = b*g-a*h;
    Zi(2,2) = a*e-b*d;

    double q = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);

    Zi = (1/q) * Zi;

    R = hruby::product(ZZ, Zi);

    Eigen::Vector3d Rx1 = hruby::product_(R, x1);

    //R = ZZ * Zi;
    t = (y1 - Rx1);
}

void ThreeViewRelativePoseHCEstimator::extract_pose13(double params[24], double solution[12], Eigen::Matrix3d &R, Eigen::Vector3d &t) const
{
    const Eigen::Vector3d x1(solution[0] * params[0], solution[0] * params[4], solution[0]);
    const Eigen::Vector3d x2(solution[1] * params[1], solution[1] * params[5], solution[1]);
    const Eigen::Vector3d x3(solution[2] * params[2], solution[2] * params[6], solution[2]);

    const Eigen::Vector3d y1(solution[8] * params[16], solution[8] * params[20], solution[8]);
    const Eigen::Vector3d y2(solution[9] * params[17], solution[9] * params[21], solution[9]);
    const Eigen::Vector3d y3(solution[10] * params[18], solution[10] * params[22], solution[10]);

    const Eigen::Vector3d z2 = x2-x1;
    const Eigen::Vector3d z3 = x3-x1;
    Eigen::Vector3d z1;// = z2.cross(z3);
    z1(0) = z2(1)*z3(2) - z2(2)*z3(1);
    z1(1) = z2(2)*z3(0) - z2(0)*z3(2);
    z1(2) = z2(0)*z3(1) - z2(1)*z3(0);
    const Eigen::Vector3d zz2 = y2-y1;
    const Eigen::Vector3d zz3 = y3-y1;
    Eigen::Vector3d zz1;// = zz2.cross(zz3);
    zz1(0) = zz2(1)*zz3(2) - zz2(2)*zz3(1);
    zz1(1) = zz2(2)*zz3(0) - zz2(0)*zz3(2);
    zz1(2) = zz2(0)*zz3(1) - zz2(1)*zz3(0);
    Eigen::Matrix3d Z;
    Z << z1, z2, z3;
    Eigen::Matrix3d ZZ;
    ZZ << zz1, zz2, zz3;

    double a = Z(0,0);
    double b = Z(0,1);
    double c = Z(0,2);

    double d = Z(1,0);
    double e = Z(1,1);
    double f = Z(1,2);

    double g = Z(2,0);
    double h = Z(2,1);
    double i = Z(2,2);

    Eigen::Matrix3d Zi;
    Zi(0,0) = e*i-f*h;
    Zi(0,1) = c*h-b*i;
    Zi(0,2) = b*f-c*e;

    Zi(1,0) = f*g-d*i;
    Zi(1,1) = a*i-c*g;
    Zi(1,2) = c*d-a*f;

    Zi(2,0) = d*h-e*g;
    Zi(2,1) = b*g-a*h;
    Zi(2,2) = a*e-b*d;

    double q = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);

    Zi = (1/q) * Zi;

    R = hruby::product(ZZ, Zi);

    Eigen::Vector3d Rx1 = hruby::product_(R, x1);

    //R = ZZ * Zi;
    t = (y1 - Rx1);
}

void ThreeViewRelativePoseHCEstimator::normalize(std::vector<Eigen::Vector3d> &P,std::vector<Eigen::Vector3d> &Q,std::vector<Eigen::Vector3d> &R,
                                                 std::vector<Eigen::Vector2d> &P1, std::vector<Eigen::Vector2d> &Q1, std::vector<Eigen::Vector2d> &R1,
                                                 Eigen::Matrix3d &CP1, Eigen::Matrix3d &CQ1, Eigen::Matrix3d &CR1, int * perm, int * perm4) const
{
    //project the points to a sphere and obtain the centroids
    Eigen::Vector3d centroidP = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroidQ = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroidR = Eigen::Vector3d::Zero();
    for(int i=0;i<4;++i)
    {
        P[i] = P[i]/P[i].norm();
        Q[i] = Q[i]/Q[i].norm();
        R[i] = R[i]/R[i].norm();
        centroidP = centroidP + P[i];
        centroidQ = centroidQ + Q[i];
        centroidR = centroidR + R[i];
    }
    centroidP = 0.25*centroidP;
    centroidQ = 0.25*centroidQ;
    centroidR = 0.25*centroidR;
    centroidP = centroidP/centroidP.norm();
    centroidQ = centroidQ/centroidQ.norm();
    centroidR = centroidR/centroidR.norm();

    //identify the first point and view
    int ix;
    int view;
    int second;
    int first;
    double best = 99999999;
    for(int i=0;i<4;++i)
    {
        double ang = P[i].transpose() * centroidP;
        if(ang < best)
        {
            best = ang;
            ix = i;
            view = 0;
        }

        ang = Q[i].transpose() * centroidQ;
        if(ang < best)
        {
            best = ang;
            ix = i;
            view = 1;
        }

        ang = R[i].transpose() * centroidR;
        if(ang < best)
        {
            best = ang;
            ix = i;
            view = 2;
        }
    }

    //order the other two views
    if(!view)
    {
        double ang1 = Q[ix].transpose() * centroidQ;
        double ang2 = R[ix].transpose() * centroidR;
        if(ang1 < ang2)
        {
            second = 1;
            first = 2;
        }
        else
        {
            second = 2;
            first = 1;
        }
    }
    else if(view == 1)
    {
        double ang0 = P[ix].transpose() * centroidP;
        double ang2 = R[ix].transpose() * centroidR;
        if(ang0 < ang2)
        {
            second = 0;
            first = 2;
        }
        else
        {
            second = 2;
            first = 0;
        }
    }
    else
    {
        double ang0 = P[ix].transpose() * centroidP;
        double ang1 = Q[ix].transpose() * centroidQ;
        if(ang0 < ang1)
        {
            second = 0;
            first = 1;
        }
        else
        {
            second = 1;
            first = 0;
        }
    }

    //rotate the centroid to zero and the given point to y axis
    Eigen::Vector3d p0 = centroidP/centroidP.norm();
    Eigen::Vector3d p1 = p0.cross(P[ix]);
    p1 = p1/p1.norm();
    Eigen::Vector3d p2 = p0.cross(p1);
    p2 = p2/p2.norm();
    Eigen::Matrix3d Zp;
    Zp << p0, p1, p2;

    Eigen::Vector3d q0 = centroidQ/centroidQ.norm();
    Eigen::Vector3d q1 = q0.cross(Q[ix]);
    q1 = q1/q1.norm();
    Eigen::Vector3d q2 = q0.cross(q1);
    q2 = q2/q2.norm();
    Eigen::Matrix3d Zq;
    Zq << q0, q1, q2;

    Eigen::Vector3d r0 = centroidR/centroidR.norm();
    Eigen::Vector3d r1 = r0.cross(R[ix]);
    r1 = r1/r1.norm();
    Eigen::Vector3d r2 = r0.cross(r1);
    r2 = r2/r2.norm();
    Eigen::Matrix3d Zr;
    Zr << r0, r1, r2;

    Eigen::Matrix3d ZZ;
    ZZ << 0,0,-1,0,1,0,1,0,0;
    Eigen::Matrix3d CP = ZZ * Zp.transpose();
    Eigen::Matrix3d CQ = ZZ * Zq.transpose();
    Eigen::Matrix3d CR = ZZ * Zr.transpose();

    //rotate the points and project them back to the plane
    for(int i=0;i<4;++i)
    {
        P[i] = CP * P[i];
        P[i] = P[i]/P[i](2);

        Q[i] = CQ * Q[i];
        Q[i] = Q[i]/Q[i](2);

        R[i] = CR * R[i];
        R[i] = R[i]/R[i](2);
    }

    perm[first] = 0;
    perm[second] = 1;
    perm[view] = 2;

    //order the points
    if(view==0)
        order_points(P, perm4, ix);
    else if(view==1)
        order_points(Q, perm4, ix);
    else
        order_points(R, perm4, ix);

    //permute the view with the longest point
    if(view==0) //perm[2]==0 means that old view 0 is new view 2, perm4[j]=q means that old pnt q is new pnt j
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            R1[j] = p;
            CR1 = CP;
        }
    }
    else if(view==1)
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            R1[j] = p;
            CR1 = CQ;
        }
    }
    else
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            R1[j] = p;
            CR1 = CR;
        }
    }

    //permute the view with the second longest point
    if(second==0)
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CP;
        }
    }
    else if(second==1)
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CQ;
        }
    }
    else
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            Q1[j] = p;
            CQ1 = CR;
        }
    }

    //permute the view with the shortest point
    if(first==0)
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = P[perm4[j]](0);
            p(1) = P[perm4[j]](1);
            P1[j] = p;
            CP1 = CP;
        }
    }
    else if(first==1)
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = Q[perm4[j]](0);
            p(1) = Q[perm4[j]](1);
            P1[j] = p;
            CP1 = CQ;
        }
    }
    else
    {
        for(int j=0;j<4;j++)
        {
            Eigen::Vector2d p;
            p(0) = R[perm4[j]](0);
            p(1) = R[perm4[j]](1);
            P1[j] = p;
            CP1 = CR;
        }
    }
}
} // namespace poselib
