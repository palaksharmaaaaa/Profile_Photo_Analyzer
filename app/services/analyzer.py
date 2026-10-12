"""Profile photo analysis: the orchestrator that ties every model together.

The flow, in the order a request goes through it:

1. the upload is decoded and validated (``image_utils``, called by the API route);
2. an image with no visual information stops here and no model runs;
3. face candidates are found (SCRFD + YuNet) together with animals, birds and people (YOLOX) and anime faces;
   every candidate crop is then verified by SigLIP2 as a real human face, an animal, a cartoon, ... or "unsure";
4. every verified real human face gets a quality check, an age and a gender (MiVOLO v2);
5. the response reports only what the models produced; whatever they cannot support is "Unsure" or "Not Detected".
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import NOT_DETECTED, UNSURE, Status
from app.services import semantic_service
from app.services.classifier_service import (
    MODEL_NAME_SIGLIP,
    MODEL_NAME_YOLOX,
    VERDICT_HUMAN,
    VERDICT_NON_HUMAN,
    VERDICT_NOT_FACE,
    VERDICT_UNAVAILABLE,
    classify_boxes,
    classify_image_type,
    detect_entities_and_persons,
)
from app.services.face_service import DetectedFace, detect_faces, predict_face_demographics
from app.services.image_utils import (
    BoxItem,
    DecodedImage,
    annotate_boxes,
    bgr_to_base64_jpeg,
    box_center_inside,
    box_iou,
    box_to_percent,
    clamp_box,
    pil_to_bgr,
)
from app.services.quality_service import assess_face_quality, check_image_information

# Box colours of the annotated image (BGR).
COLOR_ORANGE = (0, 140, 255)
COLOR_GREEN = (0, 180, 0)
COLOR_PURPLE = (200, 50, 180)

MODEL_INFO_CHECK = "Image information check (entropy and variance)"
NO_INFORMATION_REASON = "This image has no information to check or process (blank or solid colour)."
DEFAULT_FORMAT = "image/jpeg"
DISCLAIMER = (
    "Age and gender are appearance-based estimates; the typical age error is several years. "
    "Keep a separate age-verification process for any strict 18+ policy."
)

YUNET_SOURCE = "YuNet candidate"
CARTOON_FAMILY = ("anime", "cartoon_or_illustration", "render_3d", "painting")
OTHER, UNSURE_BUCKET = "other", "unsure"  # where a verified detector entity is reported

SAME_OBJECT_IOU = 0.30  # overlap at which two reports are the same object
DUPLICATE_IOU = 0.50  # overlap at which two animal reports are duplicates
ROW_HEIGHT_PX = 50  # faces within this vertical distance count as one reading row
IMAGE_TYPES_NEEDING_REVIEW = ("NON_PHOTO", "NEEDS_REVIEW", "UNCERTAIN")


@dataclass
class FaceCensus:
    """Everything found in the image, grouped by what each detection was verified to be."""

    humans: list[tuple[DetectedFace, dict[str, Any]]] = field(default_factory=list)  # (face, verification verdict)
    others: list[dict[str, Any]] = field(default_factory=list)  # animals, birds, cartoons, ...
    unsure: list[dict[str, Any]] = field(default_factory=list)  # could not be verified either way
    persons: list[list[int]] = field(default_factory=list)  # person boxes, used as body crops for the age model
    rejected: int = 0  # detections discarded as false positives
    models: list[str] = field(default_factory=list)  # models that really ran
    detection_ran: bool = False


# --------------------------------------------------------------------------------------
# Building the census: detect, verify, reconcile
# --------------------------------------------------------------------------------------


def _reading_order(box: list[int]) -> tuple[int, int]:
    """Sort key: top to bottom in rows, then left to right."""
    return round(box[1] / ROW_HEIGHT_PX), box[0]


def _other_face(
    box: list[int], face_type: str, label: str, confidence: float | None, source: str, verified_by: str
) -> dict[str, Any]:
    return {
        "face_type": face_type,
        "label": label,
        "confidence": confidence,
        "box_xyxy": box,
        "source": source,
        "verified_by": verified_by,
    }


def _unsure_face(box: list[int], label: str, reason: str, detection_confidence: float | None) -> dict[str, Any]:
    return {"label": label, "reason": reason, "detection_confidence": detection_confidence, "box_xyxy": box}


def _classify_face_candidates(image_bgr: np.ndarray, census: FaceCensus, faces: list[DetectedFace]) -> None:
    """Sort the detector's face candidates into humans, non-human faces, false positives and unsure."""
    verdicts = classify_boxes(image_bgr, [face.bbox for face in faces])
    for face, verdict in zip(faces, verdicts, strict=True):
        kind = verdict["verdict"]
        if kind == VERDICT_HUMAN:
            census.humans.append((face, verdict))
        elif kind == VERDICT_NON_HUMAN:
            census.others.append(
                _other_face(
                    face.bbox, verdict["face_type"], verdict["label"], verdict["confidence"], YUNET_SOURCE, "SigLIP2"
                )
            )
        elif kind == VERDICT_NOT_FACE:
            census.rejected += 1
        else:
            reason = (
                "The verification model is unavailable, so this candidate could not be confirmed as a real human face."
                if kind == VERDICT_UNAVAILABLE
                else "The model could not tell with enough certainty whether this is a real human face."
            )
            census.unsure.append(_unsure_face(face.bbox, verdict["label"], reason, face.det_score))


def _resolve_entity(entity: dict[str, Any], verdict: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """
    Decide what a detector entity really is, using the verdict on its crop.

    Returns ``(item, bucket)`` where bucket says whether it is reported as a non-human face or as unsure, or
    None when the detector fired on a real human face or on something that is not a face.
    """
    kind = verdict["verdict"]
    entity_type = entity["face_type"]
    detector_only = _other_face(
        entity["box_xyxy"], entity_type, entity["label"], entity["confidence"], entity["source"], "detector"
    )

    if kind in (VERDICT_HUMAN, VERDICT_NOT_FACE):
        return None

    if kind == VERDICT_NON_HUMAN:
        verified_type = verdict["face_type"]
        agrees = (entity_type == "anime" and verified_type in CARTOON_FAMILY) or (
            entity_type in ("animal", "bird") and verified_type in ("animal", "bird")
        )
        if agrees:
            return {**detector_only, "verified_by": "SigLIP2"}, OTHER
        # The verifier disagrees with the detector's type (e.g. a "dog" that is really a drawing): trust the verifier.
        item = _other_face(
            entity["box_xyxy"], verified_type, verdict["label"], verdict["confidence"], entity["source"], "SigLIP2"
        )
        return item, OTHER

    # The verifier could not decide (or is unavailable): rely on the detector only when it is trustworthy.
    if entity_type == "anime":
        # The anime detector alone is not trustworthy on photos: it also fires on animals and paintings.
        item = _unsure_face(
            entity["box_xyxy"],
            "Possible anime/cartoon face (detector only, unverified)",
            "Only the anime detector reported this region; the verification model could not confirm it.",
            entity["confidence"],
        )
        return item, UNSURE_BUCKET
    if (entity["confidence"] or 0.0) < settings.ANIMAL_DETECTOR_ONLY_MIN_CONF:
        # A weak COCO "dog/cat" box that the verifier could not confirm is often a face-filling false positive.
        item = _unsure_face(
            entity["box_xyxy"],
            f"Possible {entity['label'].lower()} (detector only, low confidence)",
            "Only the object detector reported this region, with low confidence, and it was not verified.",
            entity["confidence"],
        )
        return item, UNSURE_BUCKET
    return detector_only, OTHER


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two reports are the same object: strong overlap, or the centre of ``a`` lies inside ``b``."""
    return box_iou(a["box_xyxy"], b["box_xyxy"]) > SAME_OBJECT_IOU or box_center_inside(a["box_xyxy"], b["box_xyxy"])


def _is_duplicate_animal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The same animal reported twice, e.g. a body box and a face box, or by two detectors."""
    if box_iou(a["box_xyxy"], b["box_xyxy"]) > DUPLICATE_IOU:
        return True
    creatures = ("animal", "bird")
    same_family = a["face_type"] == b["face_type"] or (a["face_type"] in creatures and b["face_type"] in creatures)
    return same_family and (
        box_center_inside(a["box_xyxy"], b["box_xyxy"]) or box_center_inside(b["box_xyxy"], a["box_xyxy"])
    )


def _merge_entities(image_bgr: np.ndarray, census: FaceCensus, entities: list[dict[str, Any]]) -> None:
    """Verify the detector entities and reconcile them with the face candidates that are already classified."""
    verdicts = classify_boxes(image_bgr, [entity["box_xyxy"] for entity in entities])
    reported: list[dict[str, Any]] = []
    for entity, verdict in zip(entities, verdicts, strict=True):
        resolved = _resolve_entity(entity, verdict)
        if resolved is None:
            census.rejected += 1
            continue
        item, bucket = resolved
        if bucket == OTHER:
            reported.append(item)
        elif not any(_overlaps(item, existing) for existing in census.unsure):
            census.unsure.append(item)  # the same region may already be listed from the face candidates

    # Keep one report per animal; prefer the YOLOX one, then the more confident one.
    reported.sort(key=lambda item: (item["source"] != MODEL_NAME_YOLOX, -(item["confidence"] or 0.0)))
    unique: list[dict[str, Any]] = []
    for item in reported:
        if not any(_is_duplicate_animal(item, kept) for kept in unique):
            unique.append(item)

    # A face candidate already judged non-human/unsure is the same object as an overlapping detector entity.
    for item in unique:
        census.others = [
            other for other in census.others if other["source"] != YUNET_SOURCE or not _overlaps(other, item)
        ]
        census.unsure = [candidate for candidate in census.unsure if not _overlaps(candidate, item)]
        census.others.append(item)


def census_faces(image_bgr: np.ndarray) -> FaceCensus:
    """Detect every face-like region and classify each one."""
    census = FaceCensus()
    faces, face_models = detect_faces(image_bgr)
    entities, persons, entity_models = detect_entities_and_persons(image_bgr)
    census.detection_ran = bool(face_models)
    census.persons = persons
    census.models = [*face_models, *entity_models]

    _classify_face_candidates(image_bgr, census, faces)
    _merge_entities(image_bgr, census, entities)
    if semantic_service.is_available():
        census.models.append(MODEL_NAME_SIGLIP)

    census.humans.sort(key=lambda pair: _reading_order(pair[0].bbox))
    census.others.sort(key=lambda item: _reading_order(item["box_xyxy"]))
    census.unsure.sort(key=lambda item: _reading_order(item["box_xyxy"]))
    return census


# --------------------------------------------------------------------------------------
# Real human faces: quality, age, gender
# --------------------------------------------------------------------------------------


def _match_body(face_box: list[int], persons: list[list[int]]) -> list[int] | None:
    """
    The person box that contains this face and shows a real body (much taller than the face).

    A person box about as tall as the face (a head-only or head-and-shoulders crop) adds nothing for the age
    model and is ignored. Among qualifying boxes the tightest one is used.
    """
    face_height = face_box[3] - face_box[1]
    candidates = [
        box
        for box in persons
        if box_center_inside(face_box, box) and (box[3] - box[1]) >= settings.BODY_MIN_HEIGHT_RATIO * face_height
    ]
    return min(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) if candidates else None


def _build_human_faces(
    image_bgr: np.ndarray, humans: list[tuple[DetectedFace, dict[str, Any]]], persons: list[list[int]]
) -> tuple[list[dict[str, Any]], str | None]:
    """Quality check, age and gender for every verified human face. Also returns the age/gender model if it ran."""
    height, width = image_bgr.shape[:2]
    faces: list[dict[str, Any]] = []
    demographics_model: str | None = None

    for face_id, (face, verdict) in enumerate(humans, start=1):
        box = list(clamp_box(face.bbox, width, height))
        quality = assess_face_quality(image_bgr, box, face.kps)
        demographics = predict_face_demographics(
            image_bgr,
            box,
            usable=quality["usable_for_demographics"],
            caveat=quality["demographics_caveat"],
            body_box=_match_body(box, persons),
        )
        demographics_model = demographics.model or demographics_model
        possible_minor = (
            demographics.p_under_18 is not None and demographics.p_under_18 >= settings.MINOR_RISK_PROBABILITY
        )
        faces.append(
            {
                "face_id": face_id,
                "box_xyxy": box,
                "box_percent": box_to_percent(box, width, height),
                "detection_confidence": face.det_score,
                "verification_confidence": verdict["confidence"],
                "quality": quality,
                "gender": demographics.gender,
                "gender_confidence": demographics.gender_confidence,
                "age_group": demographics.age_group,
                "age_status": demographics.age_status,
                "estimated_age": demographics.estimated_age,
                "age_range": demographics.age_range,
                "age_group_probability": demographics.age_group_probability,
                "p_under_18": demographics.p_under_18,
                "possible_minor": possible_minor,
                "notes": demographics.notes,
            }
        )
    return faces, demographics_model


def _number_faces(items: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """Give non-human and unsure faces their ids (reading order) and percent boxes."""
    for face_id, item in enumerate(items, start=1):
        item["face_id"] = face_id
        item["box_percent"] = box_to_percent(item["box_xyxy"], width, height)
    return items


# --------------------------------------------------------------------------------------
# Decision and response
# --------------------------------------------------------------------------------------


def _decide(
    humans: list[dict[str, Any]],
    others: list[dict[str, Any]],
    unsure: list[dict[str, Any]],
    image_type: dict[str, Any],
) -> tuple[Status, str]:
    """
    The profile-photo policy: accept only exactly one verified real human face, nothing doubtful around it,
    good quality, a determinable age and no sign of a minor.
    """
    other_labels = ", ".join(sorted({other["label"] for other in others}))
    non_photo_note = (
        f"This looks like a non-photo image ({image_type['label']}). " if image_type["status"] == "NON_PHOTO" else ""
    )

    if not humans:
        if others:
            return (
                Status.ONLY_NON_HUMAN_FACES,
                f"{non_photo_note}No real human face was detected. Found {len(others)} other face(s): {other_labels}.",
            )
        if unsure:
            return (
                Status.REVIEW_REQUIRED,
                f"{len(unsure)} face candidate(s) could not be verified as real human faces. Manual review is needed.",
            )
        return Status.NO_HUMAN_FACE, f"{non_photo_note}No human face was detected."

    if len(humans) > 1:
        return (
            Status.MULTIPLE_HUMAN_FACES,
            f"{non_photo_note}A profile photo must contain exactly one real human face; found {len(humans)}.",
        )

    face = humans[0]
    if image_type["status"] in IMAGE_TYPES_NEEDING_REVIEW:
        return (
            Status.REVIEW_REQUIRED,
            f"{non_photo_note}Image type screening ({image_type['reason']}) requires a manual review.",
        )
    if others:
        return (
            Status.NON_HUMAN_FACE_PRESENT,
            f"One real human face was found, but the image also contains {len(others)} non-human face(s): "
            f"{other_labels}.",
        )
    if unsure:
        return (
            Status.REVIEW_REQUIRED,
            f"One real human face was found, but {len(unsure)} other candidate(s) could not be verified.",
        )
    if face["quality"]["status"] != "LIKELY_COMPLETE":
        return (
            Status.POOR_FACE_QUALITY,
            f"{face['quality']['reason']} Please provide a clear, centred, uncropped photo.",
        )
    if face["age_group"] in (UNSURE, NOT_DETECTED):
        return Status.AGE_UNSURE, "The face was verified, but its age could not be estimated with enough certainty."
    if face["possible_minor"]:
        return (
            Status.POSSIBLE_MINOR,
            f"There is an estimated {face['p_under_18']:.0%} chance that this person is under 18 "
            f"(estimated age group {face['age_group']}). Age verification is required.",
        )
    return Status.PASSED, f"All checks passed. Estimated age group: {face['age_group']} (appearance-based estimate)."


def _summary(humans: list[dict[str, Any]], others: list[dict[str, Any]], unsure: list[dict[str, Any]]) -> str:
    """One line with the three counts, naming the kinds of non-human faces."""
    counts: dict[str, int] = {}
    for other in others:
        counts[other["label"]] = counts.get(other["label"], 0) + 1
    non_human = (
        f"{len(others)} non-human face(s): " + ", ".join(f"{n} x {label}" for label, n in counts.items())
        if others
        else "0 non-human faces"
    )
    return f"{len(humans)} real human face(s); {non_human}; {len(unsure)} unverified candidate(s)."


def _annotate(
    image_bgr: np.ndarray,
    humans: list[dict[str, Any]],
    others: list[dict[str, Any]],
    unsure: list[dict[str, Any]],
) -> np.ndarray:
    """Draw the faces: green = real human, purple = non-human, orange = could not be verified."""
    items: list[BoxItem] = []
    for face in humans:
        items.append((face["box_xyxy"], f"#{face['face_id']} {face['gender']} {face['age_group']}", COLOR_GREEN))
    items += [(other["box_xyxy"], other["label"], COLOR_PURPLE) for other in others]
    items += [(candidate["box_xyxy"], "UNSURE", COLOR_ORANGE) for candidate in unsure]
    return annotate_boxes(image_bgr, items)


def _pipeline_steps(
    info_detail: str, has_information: bool, census: FaceCensus | None, demographics_ran: bool
) -> list[dict[str, Any]]:
    """The four stages shown in the UI stepper, with what each one did."""
    if census is None:
        detection_status, detection_detail = "skipped", "Skipped because the image has no visual information."
    elif census.detection_ran:
        detection_status = "completed"
        detection_detail = (
            f"{len(census.humans)} real human face(s), {len(census.others)} non-human face(s), "
            f"{len(census.unsure)} unverified candidate(s)."
        )
    else:
        detection_status, detection_detail = "failed", "No face detector could run."

    max_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
    return [
        {
            "step_id": "validate",
            "step_number": 1,
            "title": "File check",
            "status": "completed",
            "detail": f"The file is a valid image within the {max_mb} MB limit.",
        },
        {
            "step_id": "information",
            "step_number": 2,
            "title": "Visual information",
            "status": "completed" if has_information else "failed",
            "detail": info_detail,
        },
        {
            "step_id": "detection",
            "step_number": 3,
            "title": "Faces: detection and verification",
            "status": detection_status,
            "detail": detection_detail,
        },
        {
            "step_id": "demographics",
            "step_number": 4,
            "title": "Age and gender",
            "status": "completed" if demographics_ran else "skipped",
            "detail": (
                "Age (3-year brackets) and gender estimated for every verified real human face."
                if demographics_ran
                else "Skipped: no verified real human face, or the age/gender model is unavailable."
            ),
        },
    ]


def _diagnostics(
    bgr: np.ndarray,
    models: list[str],
    file_format: str,
    meta: DecodedImage | None,
    rejected: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    height, width = bgr.shape[:2]
    return {
        "processing_time_ms": elapsed_ms,
        "image_width": width,
        "image_height": height,
        "original_width": meta.original_width if meta else None,
        "original_height": meta.original_height if meta else None,
        "exif_orientation_applied": meta.exif_orientation_applied if meta else False,
        "downscaled_for_analysis": meta.downscaled if meta else False,
        "file_format": file_format,
        "rejected_candidate_count": rejected,
        "models_executed": list(dict.fromkeys(models)),
    }


def _no_information_response(
    bgr: np.ndarray, info: dict[str, Any], filename: str | None, file_format: str, meta: DecodedImage | None
) -> dict[str, Any]:
    image = bgr_to_base64_jpeg(bgr)
    return {
        "status": Status.NO_INFORMATION,
        "accepted": False,
        "reason": NO_INFORMATION_REASON,
        "summary": "No visual information: nothing was checked.",
        "real_human_face_count": 0,
        "other_face_count": 0,
        "unsure_face_count": 0,
        "human_faces": [],
        "other_faces": [],
        "unsure_faces": [],
        "image_info": info,
        "image_type": {
            "status": "NOT_CHECKED",
            "label": "Not checked",
            "photo_share": None,
            "synthetic_share": None,
            "predictions": [],
            "model": None,
            "reason": "Not checked because the image has no visual information.",
        },
        "pipeline_steps": _pipeline_steps(
            f"Blank or solid colour (entropy {info['entropy']:.2f}).", False, None, False
        ),
        "diagnostics": _diagnostics(bgr, [MODEL_INFO_CHECK], file_format, meta, 0, 0.0),
        "original_image": image,
        "annotated_image": image,
        "filename": filename,
        "legend": "No visual information or structure was found in this image.",
        "disclaimer": "Please upload a photograph with enough light, contrast and visual content.",
    }


def analyze_profile_photo_image(
    image: Image.Image,
    filename: str | None = "profile.jpg",
    file_format: str = DEFAULT_FORMAT,
    meta: DecodedImage | None = None,
) -> dict[str, Any]:
    """Run the complete analysis on a decoded image and return the response payload (see ``AnalysisResponse``)."""
    started = time.perf_counter()
    bgr = pil_to_bgr(image.convert("RGB"))
    height, width = bgr.shape[:2]

    info = check_image_information(bgr)
    if not info["has_information"]:
        response = _no_information_response(bgr, info, filename, file_format, meta)
        response["diagnostics"]["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response

    image_type = classify_image_type(bgr)
    census = census_faces(bgr)
    humans, demographics_model = _build_human_faces(bgr, census.humans, census.persons)
    others = _number_faces(census.others, width, height)
    unsure = _number_faces(census.unsure, width, height)
    status, reason = _decide(humans, others, unsure, image_type)
    accepted = status == Status.PASSED

    models = [MODEL_INFO_CHECK, *([image_type["model"]] if image_type["model"] else []), *census.models]
    models += [demographics_model] if demographics_model else []
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "status": status,
        "accepted": accepted,
        "reason": reason,
        "summary": _summary(humans, others, unsure),
        "real_human_face_count": len(humans),
        "other_face_count": len(others),
        "unsure_face_count": len(unsure),
        "human_faces": humans,
        "other_faces": others,
        "unsure_faces": unsure,
        "image_info": info,
        "image_type": image_type,
        "pipeline_steps": _pipeline_steps(
            f"The image has visual content (entropy {info['entropy']:.2f}).",
            True,
            census,
            demographics_model is not None,
        ),
        "diagnostics": _diagnostics(bgr, models, file_format, meta, census.rejected, elapsed_ms),
        "original_image": bgr_to_base64_jpeg(bgr),
        "annotated_image": bgr_to_base64_jpeg(_annotate(bgr, humans, others, unsure)),
        "filename": filename,
        "legend": (
            "Green boxes are verified real human faces, purple boxes are non-human faces, "
            "orange boxes could not be verified."
            if humans or others or unsure
            else "No face was located in this image."
        ),
        "disclaimer": DISCLAIMER,
    }
