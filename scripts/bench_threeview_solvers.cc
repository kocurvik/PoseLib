// Runtime of one hypothesis generation (generate_models of the RANSAC estimators) and number of returned poses
// for the three view solvers on synthetic data (normalized image coordinates, f = 800 px for noise and thresholds).
// Build (WSL, from repo root, after pip install . has built libPoseLib.a):
//   B=build/temp.linux-x86_64-cpython-312; T=/mnt/d/Research/code/libtorch
//   g++ -O3 -march=native -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++17 -I. -I$B/generated_headers
//       -isystem $T/include -isystem $T/include/torch/csrc/api/include -isystem /usr/local/eigen3.4/include/eigen3
//       scripts/bench_threeview_solvers.cc $B/PoseLib/libPoseLib.a -L$T/lib -Wl,-rpath,$T/lib -ltorch -ltorch_cpu -lc10
//       -o bench_threeview_solvers
#include "PoseLib/robust/estimators/relative_pose.h"
#include "PoseLib/robust/estimators/threeview_lines.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

using namespace poselib;

static std::mt19937 rng(42);
static double randn() { return std::normal_distribution<double>(0.0, 1.0)(rng); }
static double randu(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

constexpr double focal = 800.0;
constexpr double px_noise = 0.5;
constexpr double depth_noise = 0.05;
constexpr double outlier_ratio = 0.3;

struct Data {
    std::vector<Point2D> x[3];
    std::vector<Eigen::Vector3d> X[3];
    std::vector<Line2D> l[3];
    std::vector<Line3D> L[3];
};

static Eigen::Vector3d rand_point() { return Eigen::Vector3d(randu(-2, 2), randu(-1.5, 1.5), randu(4, 8)); }
static Point2D noisy(const Eigen::Vector3d &Y) {
    return Y.hnormalized() + Eigen::Vector2d(randn(), randn()) * px_noise / focal;
}

static Data make_data(size_t num_points, size_t num_lines) {
    Eigen::Matrix3d Rk[3];
    Eigen::Vector3d tk[3];
    Rk[0].setIdentity();
    tk[0].setZero();
    for (int k = 1; k < 3; ++k) {
        Rk[k] = Eigen::AngleAxisd(0.2 * randn(), Eigen::Vector3d::Random().normalized()).toRotationMatrix();
        tk[k] = Eigen::Vector3d(randn(), 0.3 * randn(), 0.3 * randn());
    }
    Data d;
    for (size_t i = 0; i < num_points; ++i) {
        const bool outlier = randu(0, 1) < outlier_ratio;
        const Eigen::Vector3d X = rand_point();
        for (int k = 0; k < 3; ++k) {
            Eigen::Vector3d Y = Rk[k] * X + tk[k];
            if (outlier && k > 0)
                Y = rand_point();
            d.x[k].push_back(noisy(Y));
            d.X[k].push_back(Eigen::Vector3d(d.x[k].back().homogeneous()) * Y(2) * (1 + depth_noise * randn()));
        }
    }
    for (size_t i = 0; i < num_lines; ++i) {
        const bool outlier = randu(0, 1) < outlier_ratio;
        const Eigen::Vector3d A = rand_point(), B = rand_point();
        for (int k = 0; k < 3; ++k) {
            Eigen::Vector3d YA = Rk[k] * A + tk[k];
            Eigen::Vector3d YB = Rk[k] * B + tk[k];
            if (outlier && k > 0) {
                YA = rand_point();
                YB = rand_point();
            }
            const Point2D a = noisy(YA), b = noisy(YB);
            d.l[k].emplace_back(a, b);
            d.L[k].emplace_back(Eigen::Vector3d(a.homogeneous()) * YA(2) * (1 + depth_noise * randn()),
                                Eigen::Vector3d(b.homogeneous()) * YB(2) * (1 + depth_noise * randn()));
        }
    }
    return d;
}

template <typename Model = ThreeViewCameraPose, typename Estimator>
static void bench(const std::string &name, Estimator &est, int runs) {
    std::vector<Model> models;
    size_t total = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        models.clear();
        est.generate_models(&models);
        total += models.size();
    }
    auto end = std::chrono::high_resolution_clock::now();
    const double us = std::chrono::duration<double, std::micro>(end - start).count() / runs;
    std::cout << "| " << std::left << std::setw(20) << name << " | " << std::right << std::fixed << std::setprecision(1)
              << std::setw(8) << us << " | " << std::setprecision(2) << std::setw(8)
              << static_cast<double>(total) / runs << " |\n";
}

int main(int argc, char **argv) {
    const int runs = argc > 1 ? std::atoi(argv[1]) : 10000;
    const size_t num_points = argc > 2 ? std::atoi(argv[2]) : 190;
    const size_t num_lines = argc > 3 ? std::atoi(argv[3]) : 50;
    const Data d = make_data(num_points, num_lines);

    RansacOptions base;
    base.max_epipolar_error = 1.0 / focal;
    base.max_line_error = 3.0 / focal;
    base.inner_refine_extra = 0;
    base.threeview_check_extra = false;

    std::cout << "points " << num_points << ", lines " << num_lines << ", runs " << runs << "\n";
    std::cout << "| solver               | time(us) | #models  |\n|---|---|---|\n";

    RansacOptions o5 = base;
    o5.sample_sz = 5;
    ThreeViewRelativePoseEstimator e5(o5, d.x[0], d.x[1], d.x[2]);
    bench("5p3v", e5, runs);

    RansacOptions o4 = base;
    o4.sample_sz = 4;
    ThreeViewRelativePoseEstimator e4(o4, d.x[0], d.x[1], d.x[2]);
    bench("4p3v(M)", e4, runs);

    RansacOptions o4d = o4;
    o4d.delta = 0.08;
    o4d.inner_refine = 2;
    o4d.threeview_check = true;
    ThreeViewRelativePoseEstimator e4d(o4d, d.x[0], d.x[1], d.x[2]);
    bench("4p3v(M-D) + R + C", e4d, runs);

    RansacOptions ol = base;
    ol.line_solver = 0;
    ThreeViewLineEstimator el(ol, d.l[0], d.l[1], d.l[2], d.L[0], d.L[1], d.L[2]);
    bench("2L", el, runs);

    RansacOptions olp = base;
    olp.line_solver = 1;
    ThreeViewLineEstimator elp(olp, d.l[0], d.l[1], d.l[2], d.L[0], d.L[1], d.L[2]);
    bench("2L(P)", elp, runs);

    ThreeViewHybridEstimator eh(base, d.x[0], d.x[1], d.x[2], d.X[0], d.X[1], d.X[2], d.l[0], d.l[1], d.l[2], d.L[0],
                                d.L[1], d.L[2]);
    bench("2p1l", eh, runs);

    RansacOptions o1l3p = base;
    o1l3p.hybrid_solver = 1;
    ThreeViewHybridEstimator e1l3p(o1l3p, d.x[0], d.x[1], d.x[2], d.X[0], d.X[1], d.X[2], d.l[0], d.l[1], d.l[2],
                                   d.L[0], d.L[1], d.L[2]);
    bench("1L3P", e1l3p, runs);

    std::cout << "two view (views 1-2)\n";
    RansacOptions o5pt = base;
    RelativePoseEstimator e5pt(o5pt, d.x[0], d.x[1]);
    bench<CameraPose>("5pt", e5pt, runs);

    RansacOptions oding = base;
    oding.upright_solver = 0;
    RelativePoseLineDirectionEstimator eding(oding, d.x[0], d.x[1], d.L[0], d.L[1]);
    bench<CameraPose>("1L3P-2v (Ding)", eding, runs);

    RansacOptions oqep = base;
    oqep.upright_solver = 1;
    RelativePoseLineDirectionEstimator eqep(oqep, d.x[0], d.x[1], d.L[0], d.L[1]);
    bench<CameraPose>("1L3P-2v (QEP)", eqep, runs);
    return 0;
}
