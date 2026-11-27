import poselib
import pytest


def test_camera_model_is_fixed_after_construction():
    camera = poselib.Camera("SIMPLE_PINHOLE", [500.0, 320.0, 240.0], 640, 480)
    with pytest.raises(AttributeError):
        camera.model_id = 4


def test_camera_params_keep_their_length():
    camera = poselib.Camera("SIMPLE_PINHOLE", [500.0, 320.0, 240.0], 640, 480)
    camera.params = [600.0, 320.0, 240.0]
    assert camera.focal() == 600.0
    with pytest.raises(ValueError, match=r"expected 3, got 4"):
        camera.params = [600.0, 320.0, 240.0, 0.1]

    # A default camera has the null model, which takes no params
    with pytest.raises(ValueError, match=r"expected 0, got 3"):
        poselib.Camera().params = [10.0, 3.0, 0.0]


def test_camera_dict_is_validated():
    with pytest.raises(ValueError, match=r"OPENCV needs 8 params"):
        poselib.Camera(
            {
                "model": "OPENCV",
                "width": 640,
                "height": 480,
                "params": [500.0, 320.0, 240.0],
            }
        )
    with pytest.raises(ValueError, match=r'Unknown camera model "PINHOL"'):
        poselib.Camera({"model": "PINHOL", "params": [500.0, 500.0, 320.0, 240.0]})
