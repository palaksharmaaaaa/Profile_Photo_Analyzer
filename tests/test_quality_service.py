import numpy as np
import pytest

from app.core.config import settings
from app.services.quality_service import (
    assess_face_quality,
    check_image_information,
    estimate_yaw_ratio,
)


def textured(size=600, seed=0):
    """Mid-grey image with a noisy square 'face' in the middle."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    lo, hi = size // 3, 2 * size // 3
    image[lo:hi, lo:hi] = rng.integers(60, 190, (hi - lo, hi - lo, 3), dtype=np.uint8)
    return image


FRONTAL = np.array([[250, 260], [350, 260], [300, 300], [260, 350], [340, 350]], dtype=np.float32)


def test_information_rejects_empty_array():
    result = check_image_information(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result["has_information"] is False


def test_information_rejects_solid_color():
    result = check_image_information(np.full((200, 200, 3), 200, dtype=np.uint8))
    assert result["has_information"] is False
    assert "no visual information" in result["reason"]


def test_information_rejects_near_uniform_image():
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    image[0, 0] = 101
    assert check_image_information(image)["has_information"] is False


def test_information_accepts_textured_image():
    result = check_image_information(textured())
    assert result["has_information"] is True
    assert result["entropy"] > settings.information.min_entropy


def test_information_downscales_large_images():
    big = np.repeat(np.repeat(textured(300), 4, axis=0), 4, axis=1)  # 1200x1200
    assert check_image_information(big)["has_information"] is True


def test_yaw_ratio_frontal_is_near_zero():
    assert abs(estimate_yaw_ratio(FRONTAL)) < 0.05


def test_yaw_ratio_grows_when_turned():
    turned = FRONTAL.copy()
    turned[2, 0] = 360  # nose far to the right of the eye midpoint
    assert estimate_yaw_ratio(turned) > 0.5


@pytest.mark.parametrize("points", [None, np.zeros((2, 2)), np.zeros((5,)), np.array([[1, 1], [1, 1], [5, 5]])])
def test_yaw_ratio_unavailable(points):
    assert estimate_yaw_ratio(points) is None


def test_quality_complete_for_good_frontal_face():
    result = assess_face_quality(textured(), [200, 200, 400, 400], FRONTAL)
    assert result["status"] == "LIKELY_COMPLETE"
    assert result["usable_for_demographics"] is True
    assert result["demographics_caveat"] is None
    assert result["min_side_px"] == 200


def test_quality_invalid_box():
    result = assess_face_quality(textured(), [100, 100, 100, 100], FRONTAL)
    assert result["status"] == "UNCERTAIN"
    assert result["usable_for_demographics"] is False


def test_quality_flags_edge_touching_face():
    result = assess_face_quality(textured(), [0, 100, 150, 300], FRONTAL)
    assert result["status"] == "PARTIAL_OR_CROPPED"
    assert result["touches_image_edge"] is True


def test_quality_flags_dark_and_bright_faces():
    dark = np.full((600, 600, 3), 10, dtype=np.uint8)
    dark[200:400, 200:400] = np.random.default_rng(1).integers(0, 30, (200, 200, 3), dtype=np.uint8)
    assert "underexposed" in assess_face_quality(dark, [200, 200, 400, 400], FRONTAL)["reason"]

    bright = np.full((600, 600, 3), 250, dtype=np.uint8)
    bright[200:400, 200:400] = np.random.default_rng(2).integers(235, 256, (200, 200, 3), dtype=np.uint8)
    assert "overexposed" in assess_face_quality(bright, [200, 200, 400, 400], FRONTAL)["reason"]


def test_quality_flags_tiny_and_huge_faces():
    tiny = assess_face_quality(textured(1200), [500, 500, 530, 530], FRONTAL)
    assert "too small" in tiny["reason"]
    assert tiny["usable_for_demographics"] is False
    assert "only 30px" in tiny["demographics_caveat"]

    huge = assess_face_quality(textured(600), [20, 20, 590, 590], FRONTAL)
    assert huge["status"] in ("UNCERTAIN", "PARTIAL_OR_CROPPED")


def test_quality_flags_huge_face_that_is_not_at_edge(monkeypatch):
    monkeypatch.setattr(settings.quality, "max_face_area_ratio", 0.05)
    result = assess_face_quality(textured(), [200, 200, 400, 400], FRONTAL)
    assert "fills almost the whole frame" in result["reason"]


def test_quality_flags_blur_independent_of_resolution():
    image = np.full((600, 600, 3), 128, dtype=np.uint8)
    image[200:400, 200:400] = 140  # smooth patch: no detail
    result = assess_face_quality(image, [200, 200, 400, 400], FRONTAL)
    assert "blurry" in result["reason"]


def test_quality_requires_landmarks():
    assert "landmarks" in assess_face_quality(textured(), [200, 200, 400, 400], None)["reason"].lower()
    outside = FRONTAL.copy()
    outside[0] = [-5, 10]
    assert "landmarks" in assess_face_quality(textured(), [200, 200, 400, 400], outside)["reason"].lower()
    malformed = np.array([1.0, 2.0, 3.0])
    result = assess_face_quality(textured(), [200, 200, 400, 400], malformed)
    assert result["landmarks_available"] is True
    assert result["landmarks_inside_image"] is False


def test_quality_flags_head_turn():
    turned = FRONTAL.copy()
    turned[2, 0] = 365
    result = assess_face_quality(textured(), [200, 200, 400, 400], turned)
    assert "turned" in result["reason"]
    assert result["usable_for_demographics"] is False
    assert "turned too far" in result["demographics_caveat"]


def test_quality_moderate_turn_is_flagged_but_still_usable_for_demographics():
    slight = FRONTAL.copy()
    slight[2, 0] = 345  # yaw ratio 0.45: not frontal, but within what the age model tolerates
    result = assess_face_quality(textured(), [200, 200, 400, 400], slight)
    assert result["status"] == "UNCERTAIN"
    assert result["usable_for_demographics"] is True
