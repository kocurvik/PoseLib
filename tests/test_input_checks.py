import numpy as np
import poselib
import pytest


def _points(n, dim, seed=0):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, dim))


def test_estimate_homography_rejects_mismatched_sizes():
    # Used to crash the interpreter (see issue #122)
    with pytest.raises(ValueError, match=r"must have the same size \(got 9, 4\)"):
        poselib.estimate_homography(_points(9, 2), _points(4, 2))


def test_refine_homography_rejects_mismatched_sizes():
    # The wrapper normalizes the points before calling the C++ refinement
    with pytest.raises(ValueError, match=r"must have the same size \(got 9, 4\)"):
        poselib.refine_homography(_points(9, 2), _points(4, 2), np.eye(3))


def test_refine_generalized_relative_pose_rejects_bad_cam_id():
    camera = {
        "model": "SIMPLE_PINHOLE",
        "width": -1,
        "height": -1,
        "params": [1.0, 0.0, 0.0],
    }
    m = poselib.PairwiseMatches()
    m.cam_id1 = 2
    m.cam_id2 = 0
    m.x1 = _points(10, 2)
    m.x2 = _points(10, 2, seed=1)
    with pytest.raises(
        ValueError, match=r"m.cam_id1 \(2\) is out of range for cameras1 \(size 2\)"
    ):
        poselib.refine_generalized_relative_pose(
            [m],
            poselib.CameraPose(),
            [poselib.CameraPose(), poselib.CameraPose()],
            [camera, camera],
            [poselib.CameraPose()],
            [camera],
        )
