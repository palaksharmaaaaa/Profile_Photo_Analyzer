"""Orchestrator tests. Every model-backed collaborator is replaced by a controllable fake (see ``Harness``)."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.schemas.analysis import AnalysisResponse, Status
from app.services import analyzer as az
from app.services.face_service import Demographics, DetectedFace
from app.services.image_utils import DecodedImage
from tests.conftest import textured

SIZE = 600
BOX_A = [200, 200, 400, 400]  # the textured square in the middle of the test image
BOX_B = [20, 100, 120, 200]
BOX_C = [450, 450, 550, 550]


# ---- builders ---------------------------------------------------------------------------------------------


def kps_for(box):
    """Five frontal landmarks (eyes, nose, mouth corners) laid out inside ``box``."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    points = [(0.3, 0.35), (0.7, 0.35), (0.5, 0.55), (0.35, 0.8), (0.65, 0.8)]
    return np.array([[x1 + fx * w, y1 + fy * h] for fx, fy in points], dtype=np.float32)


def detected(box, score=0.95):
    return DetectedFace(bbox=list(box), det_score=score, kps=kps_for(box))


def verdict(kind, face_type="", label="", confidence=0.9):
    return {
        "verdict": kind,
        "face_type": face_type or ("human" if kind == "human" else "unsure"),
        "label": label or {"human": "Real human face"}.get(kind, "Unsure (closest: Painting / Sketch)"),
        "confidence": confidence,
        "margin": 0.5,
        "scores": {},
    }


HUMAN = verdict("human")
CAT = verdict("non_human", "animal", "Animal face", 0.88)
UNSURE = verdict("unsure")
NOT_FACE = verdict("not_face", "not_a_face", "Not a face")
UNAVAILABLE = verdict("unavailable", "unsure", "Unsure (verification model unavailable)", None)


def image_type(status="PHOTO_LIKE", label="Real Photo", model="SigLIP2 test"):
    return {
        "status": status,
        "label": label,
        "photo_share": 0.98,
        "synthetic_share": 0.02,
        "predictions": [],
        "model": model,
        "reason": "test reason",
    }


PHOTO = image_type()
NON_PHOTO = image_type("NON_PHOTO", "Painting / Sketch")


def adult(age=34.0, gender="Female", p_under_18=0.0):
    group_start = int(age // 3 * 3)
    return Demographics(
        age_group=f"{group_start} to {group_start + 3}",
        estimated_age=age,
        age_range=[age - 8, age + 8],
        age_group_probability=0.19,
        p_under_18=p_under_18,
        age_status="Estimated",
        gender=gender,
        gender_confidence=0.97,
        model="Age model test",
    )


def entity(box, face_type="animal", label="Dog", source="YOLOX-S (COCO)", confidence=0.9):
    return {"face_type": face_type, "label": label, "confidence": confidence, "box_xyxy": list(box), "source": source}


class Harness:
    """Replaces every model-backed collaborator of the analyzer with fakes the test can steer."""

    def __init__(self):
        self.faces: list[DetectedFace] = []
        self.face_models = ["Face detector test"]
        self.entities: list[dict] = []
        self.persons: list[list[int]] = []
        self.entity_models = ["Entity detector test"]
        self.verdicts: dict[tuple, dict] = {}  # box -> verdict; anything else gets ``default_verdict``
        self.default_verdict = HUMAN
        self.image_type = dict(PHOTO)
        self.demographics = adult()
        self.semantic_available = True
        self.demographics_calls: list[dict] = []

    def _classify_boxes(self, image, boxes):
        return [self.verdicts.get(tuple(box), self.default_verdict) for box in boxes]

    def _predict(self, *args, **kwargs):
        self.demographics_calls.append(kwargs)
        return self.demographics

    def run(self, image: np.ndarray | None = None, **kwargs):
        image = textured(SIZE) if image is None else image
        pil = Image.fromarray(image[:, :, ::-1])
        with (
            patch.object(az, "detect_faces", side_effect=lambda img: (self.faces, self.face_models)),
            patch.object(
                az,
                "detect_entities_and_persons",
                side_effect=lambda img: (self.entities, self.persons, self.entity_models),
            ),
            patch.object(az, "classify_boxes", side_effect=self._classify_boxes),
            patch.object(az, "classify_image_type", side_effect=lambda img: self.image_type),
            patch.object(az, "predict_face_demographics", side_effect=self._predict),
            patch.object(az.semantic_service, "is_available", side_effect=lambda: self.semantic_available),
        ):
            result = az.analyze_profile_photo_image(pil, filename="p.jpg", file_format="image/jpeg", **kwargs)
        AnalysisResponse.model_validate(result)  # every payload must satisfy the public schema
        return result


@pytest.fixture
def h():
    return Harness()


# ---- step 2: no visual information ----------------------------------------------------------------------------


def test_blank_image_stops_before_any_model_runs():
    with patch.object(az, "classify_image_type") as classify, patch.object(az, "detect_faces") as detect:
        result = az.analyze_profile_photo_image(Image.new("RGB", (300, 300), "white"))
    classify.assert_not_called()
    detect.assert_not_called()
    AnalysisResponse.model_validate(result)
    assert result["status"] == Status.NO_INFORMATION
    assert result["accepted"] is False
    assert result["image_info"]["has_information"] is False
    assert result["real_human_face_count"] == result["other_face_count"] == result["unsure_face_count"] == 0
    assert result["image_type"]["status"] == "NOT_CHECKED"
    assert result["diagnostics"]["models_executed"] == [az.MODEL_INFO_CHECK]
    assert [s["status"] for s in result["pipeline_steps"]] == ["completed", "failed", "skipped", "skipped"]


# ---- steps 3 and 4: one real human face ---------------------------------------------------------------------------


def test_single_real_human_passes_with_full_demographics(h):
    h.faces = [detected(BOX_A)]
    result = h.run()
    assert result["status"] == Status.PASSED
    assert result["accepted"] is True
    assert (result["real_human_face_count"], result["other_face_count"], result["unsure_face_count"]) == (1, 0, 0)

    face = result["human_faces"][0]
    assert face["face_id"] == 1
    assert face["verification_confidence"] == 0.9
    assert face["gender"] == "Female"
    assert face["age_group"] == "33 to 36"
    assert face["age_status"] == "Estimated"
    assert face["age_range"] == [26.0, 42.0]
    assert face["p_under_18"] == 0.0
    assert face["possible_minor"] is False
    assert face["quality"]["status"] == "LIKELY_COMPLETE"
    assert face["box_percent"] == [33.33, 33.33, 66.67, 66.67]

    assert result["summary"] == "1 real human face(s); 0 non-human faces; 0 unverified candidate(s)."
    assert [s["status"] for s in result["pipeline_steps"]] == ["completed"] * 4
    models = result["diagnostics"]["models_executed"]
    assert models[0] == az.MODEL_INFO_CHECK
    assert {"SigLIP2 test", "Face detector test", "Age model test"} <= set(models)
    assert result["annotated_image"].startswith("data:image/jpeg")


def test_demographics_run_even_when_the_image_type_is_uncertain(h):
    """Regression: an uncertain image-level label used to switch age/gender off for real faces."""
    h.faces = [detected(BOX_A)]
    h.image_type = image_type("NEEDS_REVIEW")
    result = h.run()
    assert result["human_faces"][0]["age_status"] == "Estimated"
    assert result["status"] == Status.REVIEW_REQUIRED
    assert result["accepted"] is False


def test_non_photo_image_with_a_verified_human_needs_review(h):
    h.faces = [detected(BOX_A)]
    h.image_type = NON_PHOTO
    result = h.run()
    assert result["status"] == Status.REVIEW_REQUIRED
    assert "non-photo image (Painting / Sketch)" in result["reason"]


def test_unavailable_image_type_model_requires_review(h):
    h.faces = [detected(BOX_A)]
    h.image_type = image_type("UNCERTAIN", "Uncertain", model=None)
    result = h.run()
    assert result["status"] == Status.REVIEW_REQUIRED
    assert result["diagnostics"]["models_executed"][0] == az.MODEL_INFO_CHECK
    assert "SigLIP2 test" not in result["diagnostics"]["models_executed"]


def test_blurry_face_is_not_accepted(h):
    h.faces = [detected(BOX_A)]
    flat = textured(SIZE)
    flat[200:400, 200:400] = 140  # a smooth face, while the rest of the image still has detail
    flat[0:100, 0:100] = np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    result = h.run(flat)
    assert result["status"] == Status.POOR_FACE_QUALITY
    assert "blurry" in result["reason"]


def test_unsure_age_is_never_accepted(h):
    h.faces = [detected(BOX_A)]
    h.demographics = Demographics(age_group="Unsure", age_status="Unsure", p_under_18=0.01, gender="Unsure", model="m")
    assert h.run()["status"] == Status.AGE_UNSURE


def test_possible_minor_requires_age_verification(h):
    h.faces = [detected(BOX_A)]
    h.demographics = adult(age=16.0, p_under_18=0.7)
    result = h.run()
    assert result["status"] == Status.POSSIBLE_MINOR
    assert result["accepted"] is False
    assert "70%" in result["reason"]
    assert result["human_faces"][0]["possible_minor"] is True


def test_missing_age_model_reports_not_detected(h):
    h.faces = [detected(BOX_A)]
    h.demographics = Demographics()
    result = h.run()
    assert result["human_faces"][0]["age_group"] == "Not Detected"
    assert result["status"] == Status.AGE_UNSURE
    assert result["pipeline_steps"][3]["status"] == "skipped"
    assert "Age model test" not in result["diagnostics"]["models_executed"]


# ---- step 3: counting every kind of face -----------------------------------------------------------------------------


def test_multiple_real_humans_each_get_demographics_in_reading_order(h):
    h.faces = [detected(BOX_C), detected(BOX_B), detected(BOX_A)]
    result = h.run()
    assert result["real_human_face_count"] == 3
    assert result["status"] == Status.MULTIPLE_HUMAN_FACES
    assert "found 3" in result["reason"]
    assert [f["box_xyxy"] for f in result["human_faces"]] == [BOX_B, BOX_A, BOX_C]  # top to bottom, left to right
    assert [f["face_id"] for f in result["human_faces"]] == [1, 2, 3]
    assert all(f["age_status"] == "Estimated" for f in result["human_faces"])


def test_non_human_faces_are_counted_separately_and_never_aged(h):
    h.faces = [detected(BOX_A), detected(BOX_C)]
    h.verdicts = {tuple(BOX_C): CAT}
    result = h.run()
    assert (result["real_human_face_count"], result["other_face_count"]) == (1, 1)
    other = result["other_faces"][0]
    assert other["face_type"] == "animal"
    assert other["verified_by"] == "SigLIP2"
    assert result["status"] == Status.NON_HUMAN_FACE_PRESENT
    assert "Animal face" in result["reason"]
    assert len(result["human_faces"]) == 1
    assert len(h.demographics_calls) == 1  # the animal was never aged


def test_animals_without_a_face_candidate_are_found_by_the_entity_detector(h):
    h.entities = [entity([100, 100, 300, 300], label="Dog")]
    h.verdicts = {(100, 100, 300, 300): CAT}
    result = h.run()
    assert result["real_human_face_count"] == 0
    assert result["other_faces"][0]["label"] == "Dog"
    assert result["status"] == Status.ONLY_NON_HUMAN_FACES
    assert "No real human face was detected" in result["reason"]
    assert result["human_faces"] == []
    assert result["pipeline_steps"][3]["status"] == "skipped"


def test_person_riding_a_horse_keeps_both(h):
    """A human face whose centre is inside an animal's box is still a human."""
    h.faces = [detected(BOX_A)]
    h.entities = [entity([100, 100, 500, 580], label="Horse")]
    h.verdicts = {(100, 100, 500, 580): CAT}
    result = h.run()
    assert (result["real_human_face_count"], result["other_face_count"]) == (1, 1)


def test_face_candidate_that_is_really_an_animal_merges_with_the_animal_box(h):
    h.faces = [detected([220, 220, 320, 320])]
    h.entities = [entity([100, 100, 500, 500], label="Cat")]
    h.verdicts = {(220, 220, 320, 320): CAT, (100, 100, 500, 500): CAT}
    result = h.run()
    assert result["other_face_count"] == 1
    assert result["other_faces"][0]["label"] == "Cat"
    assert result["other_faces"][0]["source"] == "YOLOX-S (COCO)"


def test_duplicate_animal_reports_collapse_preferring_yolox(h):
    h.entities = [
        entity([100, 100, 500, 500], label="Dog", confidence=0.7),
        entity(
            [200, 200, 300, 300], face_type="anime", label="Anime / Cartoon Character", source="anime", confidence=0.9
        ),
    ]
    h.verdicts = {(100, 100, 500, 500): CAT, (200, 200, 300, 300): CAT}
    result = h.run()
    assert result["other_face_count"] == 1
    assert result["other_faces"][0]["label"] == "Dog"


def test_overlapping_reports_of_the_same_animal_collapse_by_overlap(h):
    h.entities = [entity([100, 100, 300, 300], label="Dog"), entity([105, 105, 295, 295], label="Dog", confidence=0.6)]
    h.verdicts = {(100, 100, 300, 300): CAT, (105, 105, 295, 295): CAT}
    assert h.run()["other_face_count"] == 1


def test_two_separate_animals_stay_separate(h):
    h.entities = [entity([10, 10, 100, 100], label="Cat"), entity([400, 400, 550, 550], label="Dog")]
    h.verdicts = {(10, 10, 100, 100): CAT, (400, 400, 550, 550): CAT}
    result = h.run()
    assert [o["label"] for o in result["other_faces"]] == ["Cat", "Dog"]
    assert [o["face_id"] for o in result["other_faces"]] == [1, 2]


def test_unverifiable_candidates_are_reported_as_unsure_never_guessed(h):
    h.faces = [detected(BOX_A)]
    h.default_verdict = UNSURE
    result = h.run()
    assert (result["real_human_face_count"], result["unsure_face_count"]) == (0, 1)
    assert result["status"] == Status.REVIEW_REQUIRED
    assert result["unsure_faces"][0]["reason"].startswith("The model could not tell")
    assert "could not be verified" in result["legend"]


def test_unavailable_verification_model_is_stated_not_hidden(h):
    h.faces = [detected(BOX_A)]
    h.default_verdict = UNAVAILABLE
    h.semantic_available = False
    result = h.run()
    assert "verification model is unavailable" in result["unsure_faces"][0]["reason"]
    assert az.MODEL_NAME_SIGLIP not in result["diagnostics"]["models_executed"]


def test_one_human_plus_an_unverified_candidate_needs_review(h):
    h.faces = [detected(BOX_A), detected(BOX_C)]
    h.verdicts = {tuple(BOX_C): UNSURE}
    result = h.run()
    assert result["status"] == Status.REVIEW_REQUIRED
    assert (result["real_human_face_count"], result["unsure_face_count"]) == (1, 1)


def test_false_face_candidates_are_dropped_and_counted(h):
    h.faces = [detected(BOX_A), detected(BOX_C)]
    h.verdicts = {tuple(BOX_C): NOT_FACE}
    result = h.run()
    assert (result["real_human_face_count"], result["other_face_count"]) == (1, 0)
    assert result["diagnostics"]["rejected_candidate_count"] == 1
    assert result["accepted"] is True


def test_no_faces_at_all(h):
    result = h.run()
    assert result["status"] == Status.NO_HUMAN_FACE
    assert result["reason"] == "No human face was detected."
    assert result["legend"] == "No face was located in this image."

    h.image_type = NON_PHOTO
    result = h.run()
    assert result["status"] == Status.NO_HUMAN_FACE
    assert "non-photo image (Painting / Sketch)" in result["reason"]


def test_face_detector_failure_is_visible_in_the_pipeline_steps(h):
    h.face_models = []
    h.entity_models = []
    h.semantic_available = False
    result = h.run()
    assert result["pipeline_steps"][2]["status"] == "failed"
    assert "No face detector could run" in result["pipeline_steps"][2]["detail"]
    assert result["diagnostics"]["models_executed"] == [az.MODEL_INFO_CHECK, "SigLIP2 test"]


def test_same_region_reported_by_a_face_candidate_and_the_anime_detector_is_listed_once(h):
    h.faces = [detected([100, 100, 300, 300])]
    h.entities = [entity([95, 95, 305, 305], face_type="anime", label="Anime", source="anime")]
    h.default_verdict = UNSURE
    assert h.run()["unsure_face_count"] == 1


def test_detector_hit_on_a_real_face_is_counted_as_rejected_not_reported(h):
    h.faces = [detected(BOX_A)]
    h.entities = [entity(BOX_A, face_type="anime", label="Anime", source="anime")]
    h.verdicts = {tuple(BOX_A): HUMAN}
    result = h.run()
    assert (result["real_human_face_count"], result["other_face_count"]) == (1, 0)
    assert result["diagnostics"]["rejected_candidate_count"] == 1


def test_anime_detector_only_hit_lands_in_the_unsure_list(h):
    h.entities = [entity([100, 100, 200, 200], face_type="anime", label="Anime", source="anime")]
    h.verdicts = {(100, 100, 200, 200): UNSURE}
    result = h.run()
    assert (result["other_face_count"], result["unsure_face_count"]) == (0, 1)
    assert "detector only" in result["unsure_faces"][0]["label"]
    assert result["status"] == Status.REVIEW_REQUIRED


# ---- how a detector entity is resolved ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["human", "not_face"])
def test_detector_hit_on_a_real_face_or_non_face_is_discarded(kind):
    assert az._resolve_entity(entity([0, 0, 10, 10], face_type="anime"), verdict(kind)) is None


def test_anime_detector_alone_is_unverified_not_confirmed():
    for unverified in (UNSURE, UNAVAILABLE):
        resolved = az._resolve_entity(entity([0, 0, 10, 10], face_type="anime", label="Anime"), unverified)
        assert resolved.other is None
        assert "detector only" in resolved.unsure["label"]


def test_animal_detector_alone_is_kept_when_confident_and_doubted_when_weak():
    resolved = az._resolve_entity(entity([0, 0, 10, 10], label="Dog", confidence=0.9), UNSURE)
    assert resolved.unsure is None
    assert resolved.other["verified_by"] == "detector"
    assert resolved.other["label"] == "Dog"

    weak = entity([0, 0, 10, 10], label="Dog", confidence=0.52)
    resolved = az._resolve_entity(weak, UNSURE)
    assert resolved.other is None
    assert "low confidence" in resolved.unsure["label"]
    assert az._resolve_entity({**weak, "confidence": None}, UNSURE).unsure is not None


def test_verified_type_overrides_the_detector_type_when_they_disagree():
    drawn = verdict("non_human", "cartoon_or_illustration", "Cartoon / illustration", 0.8)
    item = az._resolve_entity(entity([0, 0, 10, 10], label="Dog"), drawn).other
    assert item["face_type"] == "cartoon_or_illustration"
    assert item["verified_by"] == "SigLIP2"

    item = az._resolve_entity(entity([0, 0, 10, 10], face_type="anime", label="Anime"), CAT).other
    assert item["face_type"] == "animal"

    anime_ok = verdict("non_human", "anime", "Anime / manga character", 0.9)
    item = az._resolve_entity(entity([0, 0, 10, 10], face_type="anime", label="Anime"), anime_ok).other
    assert item["label"] == "Anime"
    assert item["verified_by"] == "SigLIP2"

    bird = verdict("non_human", "bird", "Bird", 0.9)
    item = az._resolve_entity(entity([0, 0, 10, 10], face_type="animal", label="Cat"), bird).other
    assert item["label"] == "Cat"
    assert item["verified_by"] == "SigLIP2"


# ---- body crops for the age model -----------------------------------------------------------------------------------


def test_match_body_uses_the_tightest_person_box_that_is_much_taller_than_the_face():
    face = [200, 200, 300, 300]
    tall, taller = [150, 150, 350, 600], [100, 100, 400, 700]
    head_and_shoulders = [180, 180, 320, 330]  # only 1.5x the face height: adds nothing for the age model
    elsewhere = [450, 100, 580, 700]
    assert az._match_body(face, [taller, tall, head_and_shoulders, elsewhere]) == tall
    assert az._match_body(face, [head_and_shoulders, elsewhere]) is None
    assert az._match_body(face, []) is None


def test_each_face_gets_its_own_body_crop(h):
    h.faces = [detected(BOX_A)]
    h.persons = [[150, 150, 450, 590]]
    h.run()
    assert h.demographics_calls[0]["body_box"] == [150, 150, 450, 590]
    h.demographics_calls.clear()
    h.persons = []
    h.run()
    assert h.demographics_calls[0]["body_box"] is None


# ---- the decision, the summary and the diagnostics ---------------------------------------------------------------------


def test_metadata_is_reported_in_diagnostics(h):
    h.faces = [detected(BOX_A)]
    meta = DecodedImage(Image.new("RGB", (4, 4)), "image/jpeg", 6000, 4000, True, True)
    pil = Image.fromarray(textured(SIZE)[:, :, ::-1])
    with (
        patch.object(az, "detect_faces", return_value=(h.faces, ["Face detector test"])),
        patch.object(az, "detect_entities_and_persons", return_value=([], [], [])),
        patch.object(az, "classify_boxes", side_effect=lambda img, boxes: [HUMAN] * len(boxes)),
        patch.object(az, "classify_image_type", return_value=PHOTO),
        patch.object(az, "predict_face_demographics", return_value=adult()),
    ):
        result = az.analyze_profile_photo_image(pil, "x.jpg", "image/jpeg", meta)
    diagnostics = result["diagnostics"]
    assert (diagnostics["original_width"], diagnostics["original_height"]) == (6000, 4000)
    assert diagnostics["exif_orientation_applied"] is True
    assert diagnostics["downscaled_for_analysis"] is True
    assert diagnostics["file_format"] == "image/jpeg"
    assert diagnostics["processing_time_ms"] > 0


# ---- progress events and the optional annotated image ---------------------------------------------------------------------


def stages(events):
    return [(event.step_id, event.status) for event in events]


def test_a_full_run_reports_every_stage_as_it_starts_and_finishes(h):
    h.faces = [detected(BOX_A)]
    events = []
    h.run(report=events.append)
    assert stages(events) == [
        ("validate", "completed"),
        ("information", "running"),
        ("information", "completed"),
        ("detection", "running"),
        ("detection", "completed"),
        ("demographics", "running"),
        ("demographics", "completed"),
    ]
    finished = [event for event in events if event.status == "completed" and event.step_id != "validate"]
    assert all(event.elapsed_ms > 0 for event in finished)
    assert "1 real human face(s)" in events[4].detail
    assert events[6].detail == "Age and gender estimated for 1 face(s)."


def test_a_blank_image_reports_the_skipped_stages():
    events = []
    az.analyze_profile_photo_image(Image.new("RGB", (80, 80), "white"), report=events.append)
    assert stages(events) == [
        ("validate", "completed"),
        ("information", "running"),
        ("information", "failed"),
        ("detection", "skipped"),
        ("demographics", "skipped"),
    ]


def test_stages_that_could_not_run_are_reported_as_failed_or_skipped(h):
    h.face_models = []
    h.entity_models = []
    h.demographics = Demographics()
    events = []
    h.run(report=events.append)
    assert ("detection", "failed") in stages(events)
    assert events[-1].status == "skipped"
    assert events[-1].detail == "No age estimate was made."


def test_the_annotated_image_can_be_left_out(h):
    h.faces = [detected(BOX_A)]
    assert h.run(render_annotation=False)["annotated_image"] is None
    assert h.run()["annotated_image"].startswith("data:image/jpeg")
    blank = az.analyze_profile_photo_image(Image.new("RGB", (80, 80), "white"), render_annotation=False)
    assert blank["annotated_image"] is None
