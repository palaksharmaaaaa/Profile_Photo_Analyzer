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


# ---- SCRFD ---------------------------------------------------------------------------------------


def scrfd_outputs(size=640, anchor=None, score=0.9, distances=(2, 2, 3, 3)):
    """SCRFD raw outputs with one detection: stride-8 cell (10, 10), box = centre (80, 80) +- distances * 8."""
    outputs = []
    for stride in fs.SCRFD_STRIDES:
        n = (size // stride) ** 2 * 2
        outputs.append(np.zeros((n, 1), dtype=np.float32))
    for stride in fs.SCRFD_STRIDES:
        outputs.append(np.zeros(((size // stride) ** 2 * 2, 4), dtype=np.float32))
    for stride in fs.SCRFD_STRIDES:
        outputs.append(np.zeros(((size // stride) ** 2 * 2, 10), dtype=np.float32))
    index = (10 * (size // 8) + 10) * 2 if anchor is None else anchor
    outputs[0][index, 0] = score
    outputs[3][index] = distances
    outputs[6][index] = [-1, -1, 1, -1, 0, 0, -1, 1, 1, 1]
    return outputs


def test_scrfd_decode_box_and_landmarks():
    ((box, score, kps),) = fs._decode_scrfd(scrfd_outputs(), 640, 0.4)
    assert box.tolist() == [64.0, 64.0, 104.0, 104.0]
    assert score == pytest.approx(0.9)
    assert kps.tolist() == [[72.0, 72.0], [88.0, 72.0], [80.0, 80.0], [72.0, 88.0], [88.0, 88.0]]


def test_scrfd_decode_threshold_and_bad_shapes():
    assert fs._decode_scrfd(scrfd_outputs(score=0.3), 640, 0.4) == []
    with pytest.raises(ValueError, match="Unexpected SCRFD output size"):
        fs._decode_scrfd(scrfd_outputs(size=320), 640, 0.4)


@pytest.mark.parametrize(
    ("height", "width", "expected"), [(218, 178, 640), (640, 640, 640), (700, 500, 704), (5000, 4000, 1280)]
)
def test_scrfd_input_size(height, width, expected):
    assert fs._scrfd_input_size(height, width) == expected


def scrfd_session(**kwargs):
    return FakeSession(lambda names, feeds: scrfd_outputs(**kwargs), input_name="input.1")


def test_scrfd_pass_maps_boxes_back_to_the_image():
    image = np.full((640, 640, 3), 90, dtype=np.uint8)
    with patch.object(fs, "get_session", return_value=scrfd_session()):
        (face,) = fs._scrfd_pass(image)
    assert face.bbox == [64, 64, 104, 104]
    assert face.source == fs.MODEL_NAME_SCRFD and face.det_score == pytest.approx(0.9)
    assert face.kps.shape == (5, 2)


def test_scrfd_pass_rescales_small_images():
    image = np.full((320, 320, 3), 90, dtype=np.uint8)  # padded and upscaled 2x into the 640 canvas
    with patch.object(fs, "get_session", return_value=scrfd_session()):
        (face,) = fs._scrfd_pass(image)
    assert face.bbox == [32, 32, 52, 52]


def test_scrfd_pass_drops_boxes_outside_the_image_and_reports_missing_model():
    image = np.full((160, 320, 3), 90, dtype=np.uint8)  # content fills only the top half of the 640 canvas
    far = scrfd_outputs(anchor=(70 * 80 + 70) * 2)  # centre (560, 560) lies below the image content
    with patch.object(fs, "get_session", return_value=FakeSession(lambda n, f: far, "input.1")):
        assert fs._scrfd_pass(image) == []
    with patch.object(fs, "get_session", return_value=None), pytest.raises(RuntimeError):
        fs._scrfd_pass(image)


# ---- detection (SCRFD + YuNet) ---------------------------------------------------------------------


def both_detectors(scrfd=None, yunet=None):
    sessions = {"scrfd": scrfd}
    return (
        patch.object(fs, "get_session", side_effect=sessions.get),
        patch.object(fs, "get_yunet_detector", return_value=yunet),
    )


def test_detect_faces_merges_both_detectors_and_keeps_the_yunet_box_for_the_same_face():
    image = np.full((640, 640, 3), 90, dtype=np.uint8)
    yunet = FakeDetector([yunet_row(66, 66, 36, 36, 0.99), yunet_row(300, 300, 60, 60, 0.8)], original_width=640)
    p_session, p_yunet = both_detectors(scrfd_session(), yunet)
    with p_session, p_yunet:
        faces, ran = fs.detect_faces(image)
    assert ran == [fs.MODEL_NAME_SCRFD, fs.MODEL_NAME_YUNET]
    assert len(faces) == 2  # the overlapping pair collapsed into one
    assert all(f.source == fs.MODEL_NAME_YUNET for f in faces)
    assert faces[0].bbox == [66, 66, 102, 102]


def test_detect_faces_union_raises_recall_when_one_detector_misses():
    image = np.full((640, 640, 3), 90, dtype=np.uint8)
    miss = FakeSession(lambda n, f: scrfd_outputs(score=0.0), "input.1")
    yunet = FakeDetector([yunet_row(300, 300, 60, 60, 0.8)], original_width=640)
    p_session, p_yunet = both_detectors(miss, yunet)
    with p_session, p_yunet:
        faces, ran = fs.detect_faces(image)
    assert [f.source for f in faces] == [fs.MODEL_NAME_YUNET] and len(ran) == 2


def test_detect_faces_scrfd_adds_faces_that_yunet_misses():
    image = np.full((640, 640, 3), 90, dtype=np.uint8)
    p_session, p_yunet = both_detectors(scrfd_session(), FakeDetector(None, 640))
    with p_session, p_yunet:
        faces, ran = fs.detect_faces(image)
    assert [f.source for f in faces] == [fs.MODEL_NAME_SCRFD] and faces[0].bbox == [64, 64, 104, 104]


def test_detect_faces_small_image_uses_second_upscaled_yunet_pass(monkeypatch):
    yunet = FakeDetector([yunet_row(100, 100, 80, 80, 0.9)], original_width=300)
    p_session, p_yunet = both_detectors(None, yunet)
    with p_session, p_yunet as get:
        faces, ran = fs.detect_faces(IMG)
    assert ran == [fs.MODEL_NAME_YUNET] and len(faces) == 1 and faces[0].bbox == [100, 100, 180, 180]
    assert get.call_count == 2  # 1x and 2x

    monkeypatch.setattr(settings, "SMALL_IMAGE_UPSCALE_SIDE", 100)
    p_session, p_yunet = both_detectors(None, yunet)
    with p_session, p_yunet as get:
        fs.detect_faces(IMG)
    assert get.call_count == 1


def test_detect_faces_yunet_skips_slivers_and_handles_no_faces():
    p_session, p_yunet = both_detectors(None, FakeDetector([yunet_row(500, 500, 40, 40)], 300))
    with p_session, p_yunet:
        assert fs.detect_faces(IMG)[0] == []
    p_session, p_yunet = both_detectors(None, FakeDetector(None, 300))
    with p_session, p_yunet:
        faces, ran = fs.detect_faces(IMG)
    assert faces == [] and ran == [fs.MODEL_NAME_YUNET]  # ran, found nothing


def test_detect_faces_reports_unavailable_detectors_as_not_run():
    p_session, p_yunet = both_detectors(None, None)
    with p_session, p_yunet:
        assert fs.detect_faces(IMG) == ([], [])
    assert fs.detect_faces(np.zeros((0, 0, 3), np.uint8)) == ([], [])
    assert fs.detect_faces(None) == ([], [])


# ---- MiVOLO preprocessing -----------------------------------------------------------------------------


def test_crop_rect_expands_by_margin_and_clamps():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert fs._crop_rect(image, [50, 20, 90, 60], 0.0).shape[:2] == (40, 40)
    assert fs._crop_rect(image, [50, 20, 90, 60], 0.5).shape[:2] == (80, 80)
    assert fs._crop_rect(image, [0, 0, 40, 40], 0.5).shape[:2] == (60, 60)  # clamped at the top-left corner


@pytest.mark.parametrize("box", [None, [10, 10, 10, 20], [30, 30, 20, 40], [500, 500, 600, 600]])
def test_crop_rect_rejects_unusable_boxes(box):
    assert fs._crop_rect(np.zeros((100, 100, 3), np.uint8), box, 0.1) is None


def test_letterbox_normalise_keeps_aspect_pads_black_and_normalises():
    wide = np.full((50, 100, 3), 255, dtype=np.uint8)
    out = fs._letterbox_normalise(wide)
    assert out.shape == (3, 384, 384)
    white = (1.0 - fs._IMAGENET_MEAN[0]) / fs._IMAGENET_STD[0]
    black = (0.0 - fs._IMAGENET_MEAN[0]) / fs._IMAGENET_STD[0]
    assert out[0, 192, 192] == pytest.approx(white, abs=1e-4)  # image content in the middle
    assert out[0, 5, 5] == pytest.approx(black, abs=1e-4)  # black padding above and below
    assert out[0, 192, 192] != out[0, 5, 5]


def test_letterbox_normalise_missing_crop_is_black_and_upscales_small_crops():
    none = fs._letterbox_normalise(None)
    assert np.allclose(none[1], (0.0 - fs._IMAGENET_MEAN[1]) / fs._IMAGENET_STD[1])
    assert fs._letterbox_normalise(np.full((10, 12, 3), 7, np.uint8)).shape == (3, 384, 384)


def test_views_are_face_channels_then_body_channels_for_every_configured_view():
    body = [20, 20, 280, 290]
    views = fs._build_views(IMG, BOX, body)
    assert len(views) == len(settings.AGE_VIEWS)
    assert all(v.shape == (1, 6, 384, 384) for v in views)
    no_body = fs._build_views(IMG, BOX, None)
    assert np.allclose(no_body[0][0, 3:], fs._letterbox_normalise(None))
    assert not np.allclose(views[0][0, 3:], no_body[0][0, 3:])
    assert fs._build_views(IMG, [10, 10, 10, 10], None) == []


def test_mirrored_view_flips_the_face():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:, 100:130] = 255  # a bright stripe on the left of the face box
    plain, mirrored = fs._build_views(image, BOX, None)[:2]
    assert not np.allclose(plain, mirrored)
    assert np.allclose(plain[0, :3, :, ::-1], mirrored[0, :3], atol=1e-5)


# ---- MiVOLO output handling ---------------------------------------------------------------------------


def head(age, female_logit=-4.0, male_logit=4.0):
    return np.array([[male_logit, female_logit, (age - fs.MIVOLO_AGE_OFFSET) / fs.MIVOLO_AGE_SCALE]], dtype=np.float32)


def mivolo_session(heads):
    """A fake MiVOLO session returning one head per call (repeating the last one)."""
    queue = list(heads)

    def run(names, feeds):
        assert feeds[next(iter(feeds))].shape == (1, 6, 384, 384)
        return [queue.pop(0) if len(queue) > 1 else queue[0]]

    return FakeSession(run, input_name="faces_bodies")


def predict(heads, **kwargs):
    with patch.object(fs, "get_session", return_value=mivolo_session(heads)):
        return fs.predict_face_demographics(IMG, BOX, **kwargs)


def test_parse_head_converts_logits_and_age():
    age, p_female = fs._parse_head(head(30.0, female_logit=2.0, male_logit=-2.0)[0])
    assert age == pytest.approx(30.0, abs=1e-3)
    assert p_female == pytest.approx(0.982, abs=0.001)


def test_confident_adult_male_estimate():
    demo = predict([head(30.0)])
    assert demo.model == fs.MODEL_NAME_AGE_GENDER
    assert demo.age_status == "Estimated" and demo.age_group == "30 to 33" and demo.estimated_age == 30.0
    low, high = demo.age_range
    assert low < 30 < high and high - low > 6  # a wide interval: the model error is several years
    assert 0.05 < demo.age_confidence < 0.3
    assert demo.p_under_18 < 0.05
    assert demo.gender == "Male" and demo.gender_confidence > 0.99


def test_female_gender():
    assert predict([head(30.0, female_logit=5.0, male_logit=-5.0)]).gender == "Female"


def test_minor_probability_rises_near_the_adult_boundary():
    demo = predict([head(17.0, female_logit=5.0, male_logit=-5.0)])
    assert demo.age_group == "15 to 18"
    assert demo.p_under_18 > settings.MINOR_RISK_PROBABILITY


def test_age_is_unsure_when_views_disagree_but_minor_probability_is_kept():
    demo = predict([head(10.0), head(30.0), head(12.0)])
    assert demo.age_status == fs.UNSURE and demo.age_group == fs.UNSURE
    assert demo.estimated_age is None and demo.age_range is None and demo.age_confidence is None
    assert demo.p_under_18 is not None
    assert any("disagree" in note for note in demo.notes)


@pytest.mark.parametrize("age", [-5.0, 0.2, 99.0])
def test_age_outside_credible_range_is_unsure(age):
    demo = predict([head(age)])
    assert demo.age_status == fs.UNSURE
    assert any("outside the range" in note for note in demo.notes)


def test_gender_not_estimated_for_apparent_children():
    demo = predict([head(8.0, female_logit=5.0, male_logit=-5.0)])
    assert demo.age_group == "6 to 9"
    assert demo.gender == fs.UNSURE
    assert any("children" in note for note in demo.notes)


def test_gender_unsure_when_views_disagree_or_score_is_low():
    flip = predict([head(30.0, 5.0, -5.0), head(30.0, -5.0, 5.0), head(30.0, 5.0, -5.0)])
    assert flip.gender == fs.UNSURE and any("Gender views disagree" in n for n in flip.notes)
    low = predict([head(30.0, female_logit=0.5, male_logit=0.0)])
    assert low.gender == fs.UNSURE
    assert low.gender_confidence == pytest.approx(0.622, abs=0.01)


def test_body_box_is_passed_through_to_the_model():
    seen = []

    def run(names, feeds):
        seen.append(next(iter(feeds.values()))[0, 3:].copy())
        return [head(30.0)]

    with patch.object(fs, "get_session", return_value=FakeSession(run, "faces_bodies")):
        fs.predict_face_demographics(IMG, BOX, body_box=[20, 20, 280, 290])
        fs.predict_face_demographics(IMG, BOX)
    assert len(seen) == 2 * len(settings.AGE_VIEWS)
    assert not np.allclose(seen[0], seen[len(settings.AGE_VIEWS)])  # with vs without a body


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

    with patch.object(fs, "get_session", return_value=mivolo_session([head(30.0)])):
        assert fs.predict_face_demographics(IMG, [10, 10, 10, 10]).age_group == fs.NOT_DETECTED


def test_inference_failure_is_not_detected():
    def explode(names, feeds):
        raise RuntimeError("ort")

    with patch.object(fs, "get_session", return_value=FakeSession(explode)):
        demo = fs.predict_face_demographics(IMG, BOX)
    assert demo.age_group == fs.NOT_DETECTED and demo.model is None
