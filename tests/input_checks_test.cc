#include "PoseLib/misc/input_checks.h"
#include "PoseLib/robust.h"
#include "test.h"

#include <stdexcept>
#include <string>

using namespace poselib;

namespace {

// Runs f and returns the message of the std::invalid_argument it throws, or "" if it throws nothing.
template <typename F> std::string invalid_argument_message(F f) {
    try {
        f();
    } catch (const std::invalid_argument &e) {
        return e.what();
    }
    return "";
}

bool contains(const std::string &msg, const std::string &expected) { return msg.find(expected) != std::string::npos; }

bool test_input_check_macros() {
    std::vector<double> a(3), b(3), c(4), empty;
    const size_t good = 2, bad = 3;

    REQUIRE(invalid_argument_message([&]() { POSELIB_CHECK_SAME_SIZE(a, b); }).empty());
    REQUIRE(invalid_argument_message([&]() { POSELIB_CHECK_SIZE_OR_EMPTY(empty, a); }).empty());
    REQUIRE(invalid_argument_message([&]() { POSELIB_CHECK_INDEX(good, a); }).empty());

    std::string msg = invalid_argument_message([&]() { POSELIB_CHECK_SAME_SIZE(a, b, c); });
    REQUIRE_M(contains(msg, "a, b, c must have the same size (got 3, 3, 4)"), msg);
    msg = invalid_argument_message([&]() { POSELIB_CHECK_SIZE_OR_EMPTY(c, a); });
    REQUIRE_M(contains(msg, "c must be empty or have the same size as a (got 4, 3)"), msg);
    msg = invalid_argument_message([&]() { POSELIB_CHECK_INDEX(bad, a); });
    REQUIRE_M(contains(msg, "bad (3) is out of range for a (size 3)"), msg);
    return true;
}

bool test_estimate_homography_size_mismatch() {
    std::vector<Point2D> x1(9, Point2D(1.0, 2.0)), x2(4, Point2D(3.0, 4.0));
    HomographyOptions opt;
    Eigen::Matrix3d H;
    std::vector<char> inliers;

    const std::string msg = invalid_argument_message([&]() { estimate_homography(x1, x2, opt, &H, &inliers); });
    REQUIRE_M(contains(msg, "estimate_homography: x1, x2 must have the same size (got 9, 4)"), msg);

    // Matching sizes are accepted (too few points, so it returns without estimating anything)
    x1.resize(3);
    x2.resize(3);
    REQUIRE(invalid_argument_message([&]() { estimate_homography(x1, x2, opt, &H, &inliers); }).empty());
    return true;
}

bool test_bundle_adjust_size_mismatch() {
    std::vector<Point2D> x(6, Point2D(0.1, 0.2));
    std::vector<Point3D> X(6, Point3D(0.0, 0.0, 1.0));
    std::vector<double> weights(4, 1.0);
    CameraPose pose;

    // The CameraPose overload must forward the weights (they used to be dropped silently)
    const std::string msg = invalid_argument_message([&]() { bundle_adjust(x, X, &pose, BundleOptions(), weights); });
    REQUIRE_M(contains(msg, "weights must be empty or have the same size as x (got 4, 6)"), msg);
    return true;
}

bool test_pairwise_matches_index_checks() {
    // The checks run before any geometry is computed, so the values do not matter
    const std::vector<CameraPose> ext(2);
    const std::vector<Camera> cameras(2, Camera("SIMPLE_PINHOLE", {1.0, 0.0, 0.0}, -1, -1));
    PairwiseMatches m;
    m.cam_id1 = 0;
    m.cam_id2 = 0;
    m.x1.assign(5, Point2D(0.1, 0.2));
    m.x2.assign(5, Point2D(0.3, 0.4));
    CameraPose pose;

    // Generalized relative pose: cam_id1 out of range
    std::vector<PairwiseMatches> bad = {m, m};
    bad[1].cam_id1 = 2;
    std::string msg =
        invalid_argument_message([&]() { refine_generalized_relpose(bad, ext, cameras, ext, cameras, &pose); });
    REQUIRE_M(contains(msg, "m.cam_id1 (2) is out of range for cameras1 (size 2)"), msg);

    // Generalized relative pose: x1 / x2 of different length
    bad = {m, m};
    bad[0].x2.pop_back();
    msg = invalid_argument_message([&]() { refine_generalized_relpose(bad, ext, cameras, ext, cameras, &pose); });
    REQUIRE_M(contains(msg, "m.x1, m.x2 must have the same size (got 5, 4)"), msg);

    // Hybrid pose: cam_id2 refers to the single query camera, so it must be 0
    const std::vector<Point2D> x(5, Point2D(0.1, 0.2));
    const std::vector<Point3D> X(5, Point3D(0.0, 0.0, 1.0));
    bad = {m, m};
    bad[1].cam_id2 = 1;
    msg = invalid_argument_message([&]() { refine_hybrid_pose(x, X, bad, ext, &pose); });
    REQUIRE_M(contains(msg, "m.cam_id2 (1) is out of range for query_ext (size 1)"), msg);
    return true;
}

} // namespace

std::vector<Test> register_input_checks_test() {
    return {TEST(test_input_check_macros), TEST(test_estimate_homography_size_mismatch),
            TEST(test_bundle_adjust_size_mismatch), TEST(test_pairwise_matches_index_checks)};
}
