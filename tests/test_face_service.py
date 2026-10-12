from unittest.mock import patch

import numpy as np
import pytest

from app.core.config import settings
from app.services import face_service as fs
from tests.conftest import FakeDetector, FakeSession, yunet_row

IMG = np.full((300, 300, 3), 120, dtype=np.uint8)
BOX = [100, 100, 200, 200]


# ---- age brackets ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (0, "0 to 3"),
        (2.9, "0 to 3"),
        (3, "3 to 6"),
        (17.99, "15 to 18"),
        (18, "18 to 21"),
        (52.4, "51 to 54"),
        (-4, "0 to 3"),
    ],
)
def test_three_year_brackets(age, expected):
    assert fs.age_to_three_year_group(age) == expected


def test_group_bounds():
    assert fs._group_bounds("51 to 54") == (51.0, 54.0)


# ---- detection --------------------------------------------------------------------------------


def test_detect_faces_small_image_uses_second_upscaled_pass_and_merges():
    detector = FakeDetector([yunet_row(100, 100, 80, 80, 0.9)], original_width=300)
    with patch.object(fs, "get_yunet_detector", return_value=detector):
        faces, model = fs.detect_faces(IMG)
    assert model == fs.MODEL_NAME_YUNET
    assert len(faces) == 1  # both passes found the same face; NMS keeps one
    assert faces[0].bbox == [100, 100, 180, 180]
    assert faces[0].kps.shape == (5, 2)
    assert faces[0].det_score == pytest.approx(0.9)


def test_detect_faces_large_image_single_pass(monkeypatch):
    monkeypatch.setattr(settings, "SMALL_IMAGE_UPSCALE_SIDE", 100)
    detector = FakeDetector([yunet_row(100, 100, 80, 80, 0.9)], original_width=300)
    with patch.object(fs, "get_yunet_detector", return_value=detector) as get:
        fs.detect_faces(IMG)
    assert get.call_count == 1


def test_detect_faces_keeps_distinct_faces_sorted_by_score():
    rows = [yunet_row(10, 10, 60, 60, 0.7), yunet_row(200, 200, 60, 60, 0.95)]
    with patch.object(fs, "get_yunet_detector", return_value=FakeDetector(rows, 300)):
        faces, _ = fs.detect_faces(IMG)
    assert [f.det_score for f in faces] == pytest.approx([0.95, 0.7])


def test_detect_faces_skips_degenerate_boxes_and_handles_no_faces():
    with patch.object(fs, "get_yunet_detector", return_value=FakeDetector([yunet_row(500, 500, 40, 40)], 300)):
        assert fs.detect_faces(IMG)[0] == []
    with patch.object(fs, "get_yunet_detector", return_value=FakeDetector(None, 300)):
        faces, model = fs.detect_faces(IMG)
    assert faces == [] and model == fs.MODEL_NAME_YUNET  # ran, found nothing


def test_detect_faces_reports_unavailable_detector_as_not_run():
    with patch.object(fs, "get_yunet_detector", return_value=None):
        assert fs.detect_faces(IMG) == ([], None)
    assert fs.detect_faces(np.zeros((0, 0, 3), np.uint8)) == ([], None)
    assert fs.detect_faces(None) == ([], None)


# ---- demographics ------------------------------------------------------------------------------


def session_returning(ages, genders):
    ages, genders = np.asarray(ages, dtype=np.float32), np.asarray(genders, dtype=np.float32)
    return FakeSession(lambda names, feeds: [np.stack([ages, genders], axis=1)], input_name="pixel_values")


def predict(ages, genders, **kwargs):
    with patch.object(fs, "get_session", return_value=session_returning(ages, genders)):
        return fs.predict_face_demographics(IMG, BOX, **kwargs)


def test_views_cover_every_margin_mirrored():
    views = fs._build_views(IMG, BOX)
    assert views.shape == (2 * len(settings.AGE_CROP_MARGINS), 3, 224, 224)
    assert fs._build_views(IMG, [10, 10, 10, 10]) is None


def test_confident_adult_estimate():
    demo = predict([30] * 6, [0.02] * 6)
    assert demo.model == fs.MODEL_NAME_AGE_GENDER
    assert demo.age_status == "Estimated"
    assert demo.age_group == "30 to 33"
    assert demo.estimated_age == 30.0
    low, high = demo.age_range
    assert low < 30 < high and high - low > 6  # a wide interval: the model error is several years
    assert 0.05 < demo.age_confidence < 0.3  # honest: a 3-year bracket is rarely certain
    assert demo.p_under_18 < 0.05
    assert demo.gender == "Male" and demo.gender_confidence == pytest.approx(0.98)


def test_female_gender():
    assert predict([30] * 6, [0.96] * 6).gender == "Female"


def test_minor_probability_rises_near_the_adult_boundary():
    demo = predict([17] * 6, [0.9] * 6)
    assert demo.age_group == "15 to 18"
    assert demo.p_under_18 > settings.MINOR_RISK_PROBABILITY


def test_age_is_unsure_when_views_disagree_but_minor_probability_is_kept():
    demo = predict([10, 30, 12, 32, 11, 31], [0.9] * 6)
    assert demo.age_status == fs.UNSURE and demo.age_group == fs.UNSURE
    assert demo.estimated_age is None and demo.age_range is None and demo.age_confidence is None
    assert demo.p_under_18 is not None
    assert any("disagree" in note for note in demo.notes)


@pytest.mark.parametrize("age", [-5, 0.2, 99])
def test_age_outside_credible_range_is_unsure(age):
    demo = predict([age] * 6, [0.9] * 6)
    assert demo.age_status == fs.UNSURE
    assert any("outside the range" in note for note in demo.notes)


def test_gender_not_estimated_for_apparent_children():
    demo = predict([8] * 6, [0.99] * 6)
    assert demo.age_group == "6 to 9"
    assert demo.gender == fs.UNSURE
    assert any("children" in note for note in demo.notes)


def test_gender_unsure_when_views_disagree_or_score_is_low():
    flip = predict([30] * 6, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    assert flip.gender == fs.UNSURE and any("Gender views disagree" in n for n in flip.notes)
    low = predict([30] * 6, [0.62] * 6)
    assert low.gender == fs.UNSURE
    assert low.gender_confidence == pytest.approx(0.62)


def test_unusable_face_skips_inference_and_says_unsure():
    with patch.object(fs, "get_session") as get:
        demo = fs.predict_face_demographics(IMG, BOX, usable=False, caveat="too small")
    get.assert_not_called()
    assert demo.age_group == demo.gender == fs.UNSURE
    assert demo.model is None and demo.notes == ["too small"]
    assert fs.predict_face_demographics(IMG, BOX, usable=False).notes


def test_missing_model_or_unusable_crop_is_not_detected():
    with patch.object(fs, "get_session", return_value=None):
        demo = fs.predict_face_demographics(IMG, BOX)
    assert demo.age_group == demo.gender == fs.NOT_DETECTED and demo.model is None

    with patch.object(fs, "get_session", return_value=session_returning([30] * 6, [0.5] * 6)):
        assert fs.predict_face_demographics(IMG, [10, 10, 10, 10]).age_group == fs.NOT_DETECTED


def test_inference_failure_is_not_detected():
    def explode(names, feeds):
        raise RuntimeError("ort")

    with patch.object(fs, "get_session", return_value=FakeSession(explode)):
        demo = fs.predict_face_demographics(IMG, BOX)
    assert demo.age_group == fs.NOT_DETECTED and demo.model is None
