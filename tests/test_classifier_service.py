from unittest.mock import patch

import numpy as np
import pytest

from app.core.config import settings
from app.services import classifier_service as cs
from app.services.semantic_service import SemanticResult
from tests.conftest import FakeSession


def sem(name, probs, kind=None):
    """Build a SemanticResult from a {class: probability} dict."""
    ordered = sorted(probs.items(), key=lambda kv: -kv[1])
    top, runner = ordered[0], ordered[1][1] if len(ordered) > 1 else 0.0
    return SemanticResult(
        name=name,
        display=name.replace("_", " ").title(),
        kind=kind
        or {"real_human": "human", "animal": "animal", "bird": "bird", "not_face": "not_a_face"}.get(name, name),
        probability=top[1],
        margin=round(top[1] - runner, 4),
        probabilities=probs,
    )


IMG = np.full((200, 200, 3), 120, dtype=np.uint8)
BOX = [50, 50, 150, 150]


# ---- image type ------------------------------------------------------------------------------


def image_probs(**overrides):
    base = {
        "real_photo": 0.0, "animal_photo": 0.0, "object_scene": 0.0,
        "cartoon": 0.0, "anime": 0.0, "painting": 0.0, "render_3d": 0.0, "diagram": 0.0,
    }  # fmt: skip
    base.update(overrides)
    return base


def run_image_type(probs):
    top = max(probs, key=probs.get)
    result = sem(top, probs)
    with patch.object(cs.semantic_service, "classify_images", return_value=[result]):
        return cs.classify_image_type(IMG)


def test_image_type_photo():
    out = run_image_type(image_probs(real_photo=0.9, cartoon=0.1))
    assert out["status"] == "PHOTO_LIKE"
    assert out["label"] == "Real Photo"
    assert out["photo_share"] == pytest.approx(0.9)
    assert out["model"] == cs.MODEL_NAME_SIGLIP


def test_image_type_synthetic_uses_label_from_winning_group():
    out = run_image_type(image_probs(real_photo=0.45, cartoon=0.25, anime=0.15, painting=0.15))
    assert out["status"] == "NEEDS_REVIEW"

    out = run_image_type(image_probs(real_photo=0.31, painting=0.4, cartoon=0.2, anime=0.09))
    assert out["status"] == "NON_PHOTO"
    assert out["label"] == "Painting / Sketch"  # not "Real Photo", which is the top single class


def test_image_type_unavailable_is_reported_as_uncertain_not_invented():
    with patch.object(cs.semantic_service, "classify_images", return_value=None):
        out = cs.classify_image_type(IMG)
    assert out["status"] == "UNCERTAIN"
    assert out["model"] is None
    assert out["predictions"] == []
    assert out["photo_share"] is None
    assert out["synthetic_share"] is None


# ---- crop verdicts ------------------------------------------------------------------------------


def face_probs(**overrides):
    base = {
        "real_human": 0.0, "animal": 0.0, "bird": 0.0, "cartoon": 0.0, "anime": 0.0,
        "painting": 0.0, "render_3d": 0.0, "doll_statue": 0.0, "diagram": 0.0, "not_face": 0.0,
    }  # fmt: skip
    base.update(overrides)
    return base


def verdict(probs):
    top = max(probs, key=probs.get)
    return cs._verdict_from_probabilities(sem(top, probs))


def test_verdict_human_requires_probability_and_margin():
    assert verdict(face_probs(real_human=0.9, painting=0.1))["verdict"] == cs.VERDICT_HUMAN
    assert verdict(face_probs(real_human=0.72, painting=0.20, cartoon=0.08))["verdict"] == cs.VERDICT_HUMAN
    assert (
        verdict(face_probs(real_human=0.68, painting=0.25, cartoon=0.07))["verdict"] == cs.VERDICT_UNSURE
    )  # below 0.70
    assert verdict(face_probs(real_human=0.5, painting=0.4, cartoon=0.1))["verdict"] == cs.VERDICT_UNSURE
    assert verdict(face_probs(real_human=0.5, painting=0.2, cartoon=0.3))["verdict"] == cs.VERDICT_UNSURE


def test_verdict_non_human_only_when_human_share_is_low():
    out = verdict(face_probs(animal=0.8, real_human=0.05, bird=0.15))
    assert out["verdict"] == cs.VERDICT_NON_HUMAN
    assert out["face_type"] == "animal"
    assert verdict(face_probs(painting=0.5, real_human=0.4, cartoon=0.1))["verdict"] == cs.VERDICT_UNSURE


def test_verdict_not_face_needs_confidence():
    assert verdict(face_probs(not_face=0.7, real_human=0.1, animal=0.2))["verdict"] == cs.VERDICT_NOT_FACE
    weak = verdict(face_probs(not_face=0.4, real_human=0.1, animal=0.3, bird=0.2))
    assert weak["verdict"] == cs.VERDICT_UNSURE
    assert "Unsure" in weak["label"]


def test_classify_boxes_empty_and_unavailable():
    assert cs.classify_boxes(IMG, []) == []
    with patch.object(cs.semantic_service, "classify_images", return_value=None):
        out = cs.classify_boxes(IMG, [BOX, [10, 10, 10, 10]])
    assert [v["verdict"] for v in out] == [cs.VERDICT_UNAVAILABLE, cs.VERDICT_UNAVAILABLE]
    assert out[0]["confidence"] is None


def test_classify_boxes_skips_degenerate_boxes_but_keeps_order():
    human = sem("real_human", face_probs(real_human=0.95, painting=0.05))
    with patch.object(cs.semantic_service, "classify_images", return_value=[human]) as classify:
        out = cs.classify_boxes(IMG, [[5, 5, 5, 5], BOX])
    assert [v["verdict"] for v in out] == [cs.VERDICT_UNAVAILABLE, cs.VERDICT_HUMAN]
    assert len(classify.call_args.args[0]) == 1  # only the valid crop was embedded


# ---- YOLOX / anime decoding -----------------------------------------------------------------------


def yolox_output(cell, cls_id, obj=0.9, cls_score=0.95, wh=np.log(10.0)):
    out = np.zeros((1, 8400, 85), dtype=np.float32)
    out[0, cell, 0:2] = 30.0  # centre offset: (30 + grid 10) * stride 8 = 320
    out[0, cell, 2:4] = wh
    out[0, cell, 4] = obj
    out[0, cell, 5 + cls_id] = cls_score
    return out


def test_yolox_grid_shape():
    grid, stride = cs._yolox_grid()
    assert grid.shape == (8400, 2)
    assert stride.shape == (8400, 1)
    assert set(np.unique(stride)) == {8.0, 16.0, 32.0}


def test_decode_yolox_finds_animals_in_original_coordinates():
    cell = 10 * 80 + 10
    out = cs._decode_yolox(yolox_output(cell, 16), ratio=0.5, width=1000, height=1000, threshold=0.45)
    assert len(out) == 1
    assert out[0]["label"] == "Dog"
    assert out[0]["face_type"] == "animal"
    assert out[0]["source"] == cs.MODEL_NAME_YOLOX
    assert out[0]["box_xyxy"] == [560, 560, 720, 720]  # (320 +- 40) / 0.5
    assert out[0]["confidence"] == pytest.approx(0.855, abs=0.01)


def test_decode_yolox_ignores_non_animals_low_scores_and_bad_shapes():
    cell = 10 * 80 + 10
    assert cs._decode_yolox(yolox_output(cell, 0), 1.0, 640, 640, 0.45) == []  # person is not an animal
    assert cs._decode_yolox(yolox_output(cell, 16, obj=0.2), 1.0, 640, 640, 0.45) == []
    assert cs._decode_yolox(np.zeros((8400, 85)), 1.0, 640, 640, 0.45) == []
    assert cs._decode_yolox(np.zeros((1, 100, 85)), 1.0, 640, 640, 0.45) == []
    assert cs._decode_yolox(np.zeros((1, 8400, 4)), 1.0, 640, 640, 0.45) == []


def test_decode_yolox_labels_birds():
    out = cs._decode_yolox(yolox_output(10 * 80 + 10, 14), 1.0, 640, 640, 0.45)
    assert out[0]["face_type"] == "bird"


def anime_output(score=0.8):
    out = np.zeros((1, 5, 8400), dtype=np.float32)
    out[0, :4, 100] = [320, 320, 100, 120]
    out[0, 4, 100] = score
    return out


def test_decode_anime_face():
    out = cs._decode_anime(anime_output(), 1.0, 640, 640, 0.35)
    assert len(out) == 1
    assert out[0]["face_type"] == "anime"
    assert out[0]["box_xyxy"] == [270, 260, 370, 380]


def test_decode_anime_threshold_and_shape():
    assert cs._decode_anime(anime_output(0.1), 1.0, 640, 640, 0.35) == []
    assert cs._decode_anime(np.zeros((5, 8400)), 1.0, 640, 640, 0.35) == []
    assert cs._decode_anime(np.zeros((1, 3, 8400)), 1.0, 640, 640, 0.35) == []


def test_to_entities_drops_degenerate_boxes_and_nms_handles_empty():
    boxes = np.array([[10.0, 10.0, 10.0, 50.0]])
    assert cs._to_entities(boxes, np.array([0.9]), [("animal", "Dog")], 0.3, "x", 100, 100) == []
    assert cs._nms([], [], 0.3) == []


def test_letterbox_preserves_aspect_and_pads_grey():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    canvas, ratio = cs._letterbox(image, to_rgb=False)
    assert canvas.shape == (640, 640, 3)
    assert ratio == pytest.approx(3.2)
    assert canvas[0, 0, 0] == 255
    assert canvas[639, 639, 0] == cs.LETTERBOX_PAD
    rgb_canvas, _ = cs._letterbox(
        np.dstack([np.zeros((10, 10)), np.zeros((10, 10)), np.full((10, 10), 200)]).astype(np.uint8), True
    )
    assert rgb_canvas[0, 0, 0] == 200  # BGR blue became RGB's first channel


def test_decode_yolox_can_return_people_for_body_crops():
    cell = 10 * 80 + 10
    out = cs._decode_yolox(yolox_output(cell, 0), 1.0, 640, 640, 0.5, cs.PERSON_CLASS)
    assert len(out) == 1
    assert out[0]["face_type"] == "person"
    assert out[0]["box_xyxy"] == [280, 280, 360, 360]
    assert cs._decode_yolox(yolox_output(cell, 16), 1.0, 640, 640, 0.5, cs.PERSON_CLASS) == []  # a dog is not a person


def test_detect_entities_runs_all_models_and_reports_them():
    both = yolox_output(10 * 80 + 10, 15)
    both[0, 20 * 80 + 20, 0:2] = 20.0
    both[0, 20 * 80 + 20, 2:4] = np.log(40.0)
    both[0, 20 * 80 + 20, 4] = 0.9
    both[0, 20 * 80 + 20, 5] = 0.95  # class 0: a person
    yolox = FakeSession(lambda n, f: [both])
    anime = FakeSession(lambda n, f: [anime_output()])
    sessions = {"yolox": yolox, "anime_face": anime}
    with patch.object(cs, "get_session", side_effect=sessions.get):
        entities, persons, executed = cs.detect_entities_and_persons(np.zeros((640, 640, 3), np.uint8))
    assert {e["face_type"] for e in entities} == {"animal", "anime"}
    assert len(persons) == 1
    assert persons[0][3] - persons[0][1] > 100
    assert executed == [cs.MODEL_NAME_YOLOX, cs.MODEL_NAME_ANIME_FACE]


def test_detect_entities_reports_only_models_that_really_ran():
    def explode(names, feeds):
        raise RuntimeError("inference error")

    sessions = {"yolox": FakeSession(explode), "anime_face": None}
    with patch.object(cs, "get_session", side_effect=sessions.get):
        entities, persons, executed = cs.detect_entities_and_persons(np.zeros((64, 64, 3), np.uint8))
    assert entities == []
    assert persons == []
    assert executed == []

    sessions = {"yolox": None, "anime_face": FakeSession(explode)}
    with patch.object(cs, "get_session", side_effect=sessions.get):
        assert cs.detect_entities_and_persons(np.zeros((64, 64, 3), np.uint8)) == ([], [], [])


def test_detect_entities_empty_image():
    assert cs.detect_entities_and_persons(np.zeros((0, 0, 3), np.uint8)) == ([], [], [])
    assert cs.detect_entities_and_persons(None) == ([], [], [])


def test_thresholds_come_from_settings():
    assert settings.verification.human_min_prob > settings.verification.nonhuman_max_human_prob
