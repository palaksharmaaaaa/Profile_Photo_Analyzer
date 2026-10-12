from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.schemas.analysis import AnalysisResponse
from app.services import analyzer as az
from app.services.face_service import Demographics, DetectedFace
from tests.conftest import textured

SIZE = 600
BOX_A = [200, 200, 400, 400]  # centred, textured region of the test image
BOX_B = [20, 100, 120, 200]
BOX_C = [450, 450, 550, 550]


def kps_for(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return np.array(
        [
            [x1 + 0.3 * w, y1 + 0.35 * h],
            [x1 + 0.7 * w, y1 + 0.35 * h],
            [x1 + 0.5 * w, y1 + 0.55 * h],
            [x1 + 0.35 * w, y1 + 0.8 * h],
            [x1 + 0.65 * w, y1 + 0.8 * h],
        ],
        dtype=np.float32,
    )


def det(box, score=0.95):
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

PHOTO = {
    "status": "PHOTO_LIKE", "top_class": "Real Photo", "top_class_confidence": 0.97, "photo_score": 0.98,
    "cartoon_score": 0.02, "display_label": "Real Photo", "model_executed": "SigLIP2 test", "all_predictions": [],
    "scores": {}, "reason": "photographic",
}  # fmt: skip
NON_PHOTO = {**PHOTO, "status": "NON_PHOTO", "display_label": "Painting / Sketch", "reason": "painted"}


def adult(age=34.0, gender="Female", p_under_18=0.0):
    group = f"{int(age // 3 * 3)} to {int(age // 3 * 3 + 3)}"
    return Demographics(
        age_group=group, estimated_age=age, age_range=[age - 8, age + 8], age_confidence=0.19,
        p_under_18=p_under_18, age_status="Estimated", gender=gender, gender_confidence=0.97,
        model="Age model test",
    )  # fmt: skip


class Harness:
    """Replaces every model-backed collaborator of the analyzer with controllable fakes."""

    def __init__(self):
        self.faces: list[DetectedFace] = []
        self.face_models: list[str] = ["YuNet test"]
        self.persons: list[list[int]] = []
        self.demo_calls: list[dict] = []
        self.entities: list[dict] = []
        self.entity_models: list[str] = ["YOLOX test"]
        self.verdicts: dict[tuple, dict] = {}
        self.default_verdict = HUMAN
        self.image_type = dict(PHOTO)
        self.demographics = adult()
        self.semantic_available = True

    def _demographics(self, *args, **kwargs):
        self.demo_calls.append(kwargs)
        return self.demographics

    def _classify_boxes(self, image, boxes):
        return [self.verdicts.get(tuple(b), self.default_verdict) for b in boxes]

    def run(self, image: np.ndarray | None = None):
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
            patch.object(az, "predict_face_demographics", side_effect=self._demographics),
            patch.object(az.semantic_service, "is_available", side_effect=lambda: self.semantic_available),
        ):
            result = az.analyze_profile_photo_image(pil, filename="p.jpg", format_detected="image/jpeg")
        AnalysisResponse.model_validate(result)  # the payload must always satisfy the public schema
        return result


@pytest.fixture
def h():
    return Harness()


def entity(box, face_type="animal", label="Dog", source="YOLOX-S (COCO)", confidence=0.9):
    return {"face_type": face_type, "label": label, "confidence": confidence, "box_xyxy": list(box), "source": source}


# ---- step 2: no information -----------------------------------------------------------------------


def test_blank_image_stops_before_any_model_runs(h):
    with patch.object(az, "classify_image_type") as classify, patch.object(az, "detect_faces") as detect:
        pil = Image.new("RGB", (300, 300), "white")
        result = az.analyze_profile_photo_image(pil)
    classify.assert_not_called()
    detect.assert_not_called()
    AnalysisResponse.model_validate(result)
    assert result["posting_status"] == "NO_INFORMATION"
    assert result["has_information"] is False
    assert result["accepted"] is False
    assert result["real_human_face_count"] == result["other_face_count"] == 0
    assert result["image_type_check"]["status"] == "NOT_CHECKED"
    assert result["technical_diagnostics"]["models_executed"] == [az.MODEL_INFO_ANALYZER]
    assert [s["status"] for s in result["pipeline_steps"]] == ["completed", "failed", "skipped", "skipped"]


# ---- steps 3 and 4: one real human face ---------------------------------------------------------


def test_single_real_human_passes_with_full_demographics(h):
    h.faces = [det(BOX_A)]
    result = h.run()
    assert result["accepted"] is True
    assert result["posting_status"] == "INITIAL_CHECKS_PASSED"
    assert result["real_human_face_count"] == 1 and result["other_face_count"] == 0 and result["unsure_face_count"] == 0
    assert result["total_face_count"] == 1
    face = result["human_faces"][0]
    assert face["gender"] == "Female" and face["age_group"] == "33 to 36" and face["age_status"] == "Estimated"
    assert face["age_range"] == [26.0, 42.0] and face["p_under_18"] == 0.0 and face["possible_minor_risk"] is False
    assert face["verification_confidence"] == 0.9 and face["is_primary"] is True
    assert face["face_visibility"]["status"] == "LIKELY_COMPLETE"
    # legacy single-face fields mirror the face
    assert result["estimated_age_group"] == "33 to 36" and result["gender_presentation_prediction"] == "Female"
    assert result["age_gender_prediction_performed"] is True
    assert result["age_prediction_error"] is None and result["gender_prediction_error"] is None
    assert result["summary"].startswith("1 real human face(s); 0 non-human faces")
    assert [s["status"] for s in result["pipeline_steps"]] == ["completed"] * 4
    models = result["technical_diagnostics"]["models_executed"]
    assert models[0] == az.MODEL_INFO_ANALYZER and "Age model test" in models and "YuNet test" in models
    assert result["original_image"].startswith("data:image/jpeg")
    assert result["annotated_image"].startswith("data:image/jpeg")


def test_demographics_run_for_the_face_even_when_image_type_is_unsure(h):
    """Regression: an uncertain image-level label used to switch demographics off for real faces."""
    h.faces = [det(BOX_A)]
    h.image_type = {**PHOTO, "status": "NEEDS_REVIEW"}
    result = h.run()
    assert result["human_faces"][0]["age_status"] == "Estimated"
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"
    assert result["accepted"] is False


def test_non_photo_image_with_a_verified_human_needs_review(h):
    h.faces = [det(BOX_A)]
    h.image_type = NON_PHOTO
    result = h.run()
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"
    assert "non-photo" in result["reason"]
    assert result["image_type_review_required"] is True


def test_unavailable_image_type_model_requires_review(h):
    h.faces = [det(BOX_A)]
    h.image_type = {**PHOTO, "status": "UNCERTAIN", "model_executed": None}
    result = h.run()
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"
    assert az.MODEL_INFO_ANALYZER in result["technical_diagnostics"]["models_executed"]


def test_blurry_face_is_not_accepted(h):
    h.faces = [det(BOX_A)]
    flat = textured(SIZE)
    flat[200:400, 200:400] = 140  # smooth face, but the rest of the image still has detail
    flat[0:100, 0:100] = np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    result = h.run(flat)
    assert result["posting_status"] == "REJECTED_OR_REVIEW"
    assert "blurry" in result["reason"]


def test_unsure_age_is_never_accepted(h):
    h.faces = [det(BOX_A)]
    h.demographics = Demographics(age_group="Unsure", age_status="Unsure", p_under_18=0.01, gender="Unsure", model="m")
    result = h.run()
    assert result["posting_status"] == "AGE_ESTIMATION_UNAVAILABLE"
    assert result["age_prediction_error"] and result["gender_prediction_error"]


def test_possible_minor_requires_age_verification(h):
    h.faces = [det(BOX_A)]
    h.demographics = adult(age=16.0, p_under_18=0.7)
    result = h.run()
    assert result["posting_status"] == "AGE_VERIFICATION_REQUIRED"
    assert result["possible_minor_risk"] is True and result["age_verification_required"] is True
    assert "70%" in result["reason"]
    assert result["human_faces"][0]["possible_minor_risk"] is True


def test_missing_age_model_reports_not_detected(h):
    h.faces = [det(BOX_A)]
    h.demographics = Demographics()
    result = h.run()
    assert result["human_faces"][0]["age_group"] == "Not Detected"
    assert result["posting_status"] == "AGE_ESTIMATION_UNAVAILABLE"
    assert result["age_gender_prediction_performed"] is False
    assert result["pipeline_steps"][3]["status"] == "skipped"
    assert "Age model test" not in result["technical_diagnostics"]["models_executed"]


# ---- step 3: counting every kind of face ----------------------------------------------------------


def test_multiple_real_humans_each_get_demographics_in_reading_order(h):
    h.faces = [det(BOX_C), det(BOX_B), det(BOX_A)]
    result = h.run()
    assert result["real_human_face_count"] == 3
    assert result["posting_status"] == "REJECTED"
    assert "found 3" in result["reason"]
    assert [f["box_xyxy"] for f in result["human_faces"]] == [BOX_B, BOX_A, BOX_C]  # top-to-bottom, left-to-right
    assert [f["face_id"] for f in result["human_faces"]] == [1, 2, 3]
    assert all(f["age_status"] == "Estimated" for f in result["human_faces"])
    assert result["estimated_age_group"] == "Not available" and result["face_count"] == 3
    assert result["age_verification_required"] is True


def test_non_human_faces_are_counted_separately_and_never_aged(h):
    h.faces = [det(BOX_A), det(BOX_C)]
    h.verdicts = {tuple(BOX_C): CAT}
    result = h.run()
    assert result["real_human_face_count"] == 1
    assert result["other_face_count"] == 1 and result["total_face_count"] == 2
    assert result["other_faces"][0]["face_type"] == "animal" and result["other_faces"][0]["verified_by"] == "SigLIP2"
    assert result["posting_status"] == "NON_PHOTO_WARNING"
    assert "Animal face" in result["reason"]
    assert len(result["human_faces"]) == 1


def test_animals_without_a_yunet_face_are_found_by_the_entity_detector(h):
    h.entities = [entity([100, 100, 300, 300], label="Dog")]
    h.verdicts = {(100, 100, 300, 300): CAT}
    result = h.run()
    assert result["real_human_face_count"] == 0
    assert result["other_faces"][0]["label"] == "Dog"
    assert result["posting_status"] == "NON_PHOTO_WARNING"
    assert "No real human face was detected" in result["reason"]
    assert result["human_faces"] == [] and result["estimated_age_group"] == "Not Detected"
    assert result["pipeline_steps"][3]["status"] == "skipped"


def test_person_riding_a_horse_keeps_both(h):
    """A human face whose centre is inside an animal's box is still a human (it used to be reclassified)."""
    h.faces = [det(BOX_A)]
    h.entities = [entity([100, 100, 500, 580], label="Horse")]
    h.verdicts = {(100, 100, 500, 580): CAT}
    result = h.run()
    assert result["real_human_face_count"] == 1 and result["other_face_count"] == 1


def test_yunet_candidate_that_is_really_an_animal_merges_with_the_animal_box(h):
    h.faces = [det([220, 220, 320, 320])]
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


def test_two_separate_animals_stay_separate(h):
    h.entities = [entity([10, 10, 100, 100], label="Cat"), entity([400, 400, 550, 550], label="Dog")]
    h.verdicts = {(10, 10, 100, 100): CAT, (400, 400, 550, 550): CAT}
    result = h.run()
    assert [o["label"] for o in result["other_faces"]] == ["Cat", "Dog"]
    assert [o["face_id"] for o in result["other_faces"]] == [1, 2]


def test_unverifiable_candidates_are_reported_as_unsure_never_guessed(h):
    h.faces = [det(BOX_A)]
    h.default_verdict = UNSURE
    result = h.run()
    assert result["real_human_face_count"] == 0 and result["unsure_face_count"] == 1
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"
    assert result["unsure_faces"][0]["reason"].startswith("The model could not tell")
    assert result["face_detected"] is True
    assert "could not be verified" in result["region_note"]


def test_unavailable_verification_model_is_stated_not_hidden(h):
    h.faces = [det(BOX_A)]
    h.default_verdict = UNAVAILABLE
    h.semantic_available = False
    result = h.run()
    assert "verification model is unavailable" in result["unsure_faces"][0]["reason"]
    assert az.MODEL_NAME_SIGLIP not in result["technical_diagnostics"]["models_executed"]


def test_one_human_plus_unverified_candidate_needs_review(h):
    h.faces = [det(BOX_A), det(BOX_C)]
    h.verdicts = {tuple(BOX_C): UNSURE}
    result = h.run()
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"
    assert result["real_human_face_count"] == 1 and result["unsure_face_count"] == 1


def test_false_face_candidates_are_dropped_and_counted(h):
    h.faces = [det(BOX_A), det(BOX_C)]
    h.verdicts = {tuple(BOX_C): NOT_FACE}
    result = h.run()
    assert result["real_human_face_count"] == 1 and result["other_face_count"] == 0
    assert result["technical_diagnostics"]["rejected_candidate_count"] == 1
    assert result["accepted"] is True


def test_no_faces_at_all(h):
    result = h.run()
    assert result["posting_status"] == "REJECTED"
    assert result["reason"] == "No human face was detected."
    assert result["face_detected"] is False and result["total_face_count"] == 0
    assert result["face_regions_xyxy"] == []
    assert "No face region" in result["region_note"]

    h.image_type = NON_PHOTO
    result = h.run()
    assert result["posting_status"] == "NON_PHOTO_WARNING"
    assert "non-photo image (Painting / Sketch)" in result["reason"]


def test_face_detector_failure_is_visible_in_the_pipeline_steps(h):
    h.face_models = []
    h.entity_models = []
    h.semantic_available = False
    result = h.run()
    assert result["pipeline_steps"][2]["status"] == "failed"
    assert "could not run" in result["pipeline_steps"][2]["detail"]
    assert result["technical_diagnostics"]["models_executed"] == [az.MODEL_INFO_ANALYZER, "SigLIP2 test"]


# ---- entity resolution rules ------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["human", "not_face"])
def test_detector_hit_on_a_real_face_or_non_face_is_discarded(kind):
    assert az._entity_item(entity([0, 0, 10, 10], face_type="anime"), verdict(kind)) is None


def test_anime_detector_alone_is_unverified_not_confirmed():
    item, bucket = az._entity_item(entity([0, 0, 10, 10], face_type="anime", label="Anime"), UNSURE)
    assert bucket == az.BUCKET_UNSURE and "detector only" in item["label"]
    item, bucket = az._entity_item(entity([0, 0, 10, 10], face_type="anime", label="Anime"), UNAVAILABLE)
    assert bucket == az.BUCKET_UNSURE


def test_animal_detector_alone_is_kept_and_labelled_detector_only():
    item, bucket = az._entity_item(entity([0, 0, 10, 10], label="Dog"), UNSURE)
    assert bucket == az.BUCKET_OTHER and item["verified_by"] == "detector" and item["label"] == "Dog"


def test_semantic_verdict_overrides_detector_type_when_they_disagree():
    drawn = verdict("non_human", "cartoon_or_illustration", "Cartoon / illustration", 0.8)
    item, _ = az._entity_item(entity([0, 0, 10, 10], label="Dog"), drawn)
    assert item["face_type"] == "cartoon_or_illustration" and item["verified_by"] == "SigLIP2"

    item, _ = az._entity_item(entity([0, 0, 10, 10], face_type="anime", label="Anime"), CAT)
    assert item["face_type"] == "animal"

    anime_ok = verdict("non_human", "anime", "Anime / manga character", 0.9)
    item, _ = az._entity_item(entity([0, 0, 10, 10], face_type="anime", label="Anime"), anime_ok)
    assert item["label"] == "Anime" and item["verified_by"] == "SigLIP2"

    bird = verdict("non_human", "bird", "Bird", 0.9)
    item, _ = az._entity_item(entity([0, 0, 10, 10], face_type="animal", label="Cat"), bird)
    assert item["label"] == "Cat" and item["verified_by"] == "SigLIP2"


# ---- policy / summary / misc ----------------------------------------------------------------------


def test_summary_groups_non_human_labels():
    others = [{"label": "Dog"}, {"label": "Dog"}, {"label": "Bird"}]
    text = az._summary([{}], others, [{}, {}])
    assert text == "1 real human face(s); 3 non-human face(s): 2 x Dog, 1 x Bird; 2 unverified candidate(s)."


def test_decide_with_non_photo_and_no_faces():
    accepted, status, reason = az._decide([], [], [], {"status": "NON_PHOTO", "display_label": "Diagram"})
    assert (accepted, status) == (False, "NON_PHOTO_WARNING") and "Diagram" in reason


def test_metadata_is_reported_in_diagnostics(h):
    from app.services.image_utils import DecodedImage

    h.faces = [det(BOX_A)]
    meta = DecodedImage(Image.new("RGB", (4, 4)), "image/jpeg", 6000, 4000, True, True)
    bgr = textured(SIZE)
    pil = Image.fromarray(bgr[:, :, ::-1])
    with (
        patch.object(az, "detect_faces", return_value=(h.faces, ["YuNet test"])),
        patch.object(az, "detect_entities_and_persons", return_value=([], [], [])),
        patch.object(az, "classify_boxes", side_effect=lambda img, boxes: [HUMAN] * len(boxes)),
        patch.object(az, "classify_image_type", return_value=PHOTO),
        patch.object(az, "predict_face_demographics", return_value=adult()),
    ):
        result = az.analyze_profile_photo_image(pil, "x.jpg", "image/jpeg", meta)
    diag = result["technical_diagnostics"]
    assert (diag["original_width"], diag["original_height"]) == (6000, 4000)
    assert diag["exif_orientation_applied"] is True and diag["downscaled_for_analysis"] is True
    assert diag["processing_time_ms"] == result["processing_time_ms"] > 0


def test_detector_hit_on_a_real_face_is_counted_as_rejected_not_reported(h):
    h.faces = [det(BOX_A)]
    h.entities = [entity(BOX_A, face_type="anime", label="Anime", source="anime")]
    h.verdicts = {tuple(BOX_A): HUMAN}
    result = h.run()
    assert result["real_human_face_count"] == 1 and result["other_face_count"] == 0
    assert result["technical_diagnostics"]["rejected_candidate_count"] == 1


def test_anime_detector_only_hit_lands_in_the_unsure_list(h):
    h.entities = [entity([100, 100, 200, 200], face_type="anime", label="Anime", source="anime")]
    h.verdicts = {(100, 100, 200, 200): UNSURE}
    result = h.run()
    assert result["other_face_count"] == 0 and result["unsure_face_count"] == 1
    assert "detector only" in result["unsure_faces"][0]["label"]
    assert result["posting_status"] == "MANUAL_REVIEW_REQUIRED"


def test_overlapping_reports_of_the_same_object_collapse_by_iou(h):
    h.entities = [entity([100, 100, 300, 300], label="Dog"), entity([105, 105, 295, 295], label="Dog", confidence=0.6)]
    h.verdicts = {(100, 100, 300, 300): CAT, (105, 105, 295, 295): CAT}
    assert h.run()["other_face_count"] == 1


def test_same_region_reported_by_yunet_and_anime_detector_is_listed_once(h):
    h.faces = [det([100, 100, 300, 300])]
    h.entities = [entity([95, 95, 305, 305], face_type="anime", label="Anime", source="anime")]
    h.default_verdict = UNSURE
    result = h.run()
    assert result["unsure_face_count"] == 1


def test_weak_detector_only_animal_is_unsure_not_confirmed():
    weak = entity([0, 0, 10, 10], label="Dog", confidence=0.52)
    item, bucket = az._entity_item(weak, UNSURE)
    assert bucket == az.BUCKET_UNSURE and "low confidence" in item["label"]
    assert az._entity_item(entity([0, 0, 10, 10], label="Dog", confidence=0.9), UNSURE)[1] == az.BUCKET_OTHER
    assert az._entity_item({**weak, "confidence": None}, UNSURE)[1] == az.BUCKET_UNSURE


# ---- body crops for the age model -----------------------------------------------------------------


def test_match_body_uses_the_tightest_person_box_that_is_much_taller_than_the_face():
    face = [200, 200, 300, 300]
    tall = [150, 150, 350, 600]
    taller = [100, 100, 400, 700]
    head_and_shoulders = [180, 180, 320, 330]  # only 1.5x the face height: no extra information
    elsewhere = [450, 100, 580, 700]
    assert az._match_body(face, [taller, tall, head_and_shoulders, elsewhere]) == tall
    assert az._match_body(face, [head_and_shoulders, elsewhere]) is None
    assert az._match_body(face, []) is None


def test_each_face_gets_its_own_body_crop(h):
    h.faces = [det(BOX_A)]
    h.persons = [[150, 150, 450, 590]]
    h.run()
    assert h.demo_calls[0]["body_box"] == [150, 150, 450, 590]
    h.demo_calls.clear()
    h.persons = []
    h.run()
    assert h.demo_calls[0]["body_box"] is None
