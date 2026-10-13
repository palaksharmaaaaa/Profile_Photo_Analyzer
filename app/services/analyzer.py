"""Profile photo analysis: the orchestrator that ties every model together.

The flow, in the order a request goes through it:

1. the upload is decoded and validated (``image_utils``, called by the API route);
2. an image with no visual information stops here and no model runs;
3. face candidates are found (SCRFD + YuNet) together with animals, birds and people (YOLOX) and anime faces;
   every candidate crop is then verified by SigLIP2 as a real human face, an animal, a cartoon, ... or "unsure";
4. every verified real human face gets a quality check, an age and a gender (MiVOLO v2);
5. ``policy`` turns the findings into the outcome; the response reports only what the models produced, and whatever
   they cannot support is "Unsure" or "Not Detected".

While it runs the analysis reports each stage through an optional callback (see ``progress``).
"""

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import Status
from app.services import semantic_service
from app.services.classifier_service import (
    MODEL_NAME_SIGLIP,
    MODEL_NAME_YOLOX,
    VERDICT_HUMAN,
    VERDICT_NON_HUMAN,
    VERDICT_NOT_FACE,
    VERDICT_UNAVAILABLE,
    classify_image_type,
    detect_entities_and_persons,
)
from app.services.classifier_service import (
    classify_boxes as classify_boxes,  # re-exported: the evaluation tooling wraps it
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
from app.services.parallel import run_pair
from app.services.policy import decide, is_possible_minor, summarise
from app.services.progress import COMPLETED, FAILED, RUNNING, SKIPPED, STAGES, ProgressReporter, stage_event
from app.services.quality_service import assess_face_quality, check_image_information
from app.services.types import (
    AnalysisResult,
    Box,
    DiagnosticsItem,
    Entity,
    HumanFaceItem,
    ImageInfoItem,
    ImageTypeItem,
    OtherFaceItem,
    PipelineStepItem,
    UnsureFaceItem,
    Verdict,
)

# Box colours of the annotated image (BGR).
COLOR_ORANGE = (0, 140, 255)
COLOR_GREEN = (0, 180, 0)
COLOR_PURPLE = (200, 50, 180)

MODEL_INFO_CHECK = "Image information check (entropy and variance)"
NO_INFORMATION_REASON = "This image has no information to check or process (blank or solid colour)."
SKIPPED_NO_INFORMATION = "Skipped because the image has no visual information."
DEFAULT_FORMAT = "image/jpeg"
DISCLAIMER = (
    "Age and gender are appearance-based estimates; the typical age error is several years. "
    "Keep a separate age-verification process for any strict 18+ policy."
)

YUNET_SOURCE = "YuNet candidate"
CARTOON_FAMILY = ("anime", "cartoon_or_illustration", "render_3d", "painting")

SAME_OBJECT_IOU = 0.30  # overlap at which two reports are the same object
DUPLICATE_IOU = 0.50  # overlap at which two animal reports are duplicates
ROW_HEIGHT_PX = 50  # faces within this vertical distance count as one reading row


@dataclass
class FaceCensus:
    """Everything found in the image, grouped by what each detection was verified to be."""

    humans: list[tuple[DetectedFace, Verdict]] = field(default_factory=list)  # (face, verification verdict)
    others: list[OtherFaceItem] = field(default_factory=list)  # animals, birds, cartoons, ...
    unsure: list[UnsureFaceItem] = field(default_factory=list)  # could not be verified either way
    persons: list[Box] = field(default_factory=list)  # person boxes, used as body crops for the age model
    rejected: int = 0  # detections discarded as false positives
    models: list[str] = field(default_factory=list)  # models that really ran
    detection_ran: bool = False


@dataclass
class Resolved:
    """What a detector entity turned out to be: a non-human face or an unsure candidate (exactly one is set)."""

    other: OtherFaceItem | None = None
    unsure: UnsureFaceItem | None = None


# --------------------------------------------------------------------------------------
# Building the census: detect, verify, reconcile
# --------------------------------------------------------------------------------------


def _reading_order(box: Box) -> tuple[int, int]:
    """Sort key: top to bottom in rows, then left to right."""
    return round(box[1] / ROW_HEIGHT_PX), box[0]


def _other_face(
    box: Box, face_type: str, label: str, confidence: float | None, source: str, verified_by: str
) -> OtherFaceItem:
    return {
        "face_type": face_type,
        "label": label,
        "confidence": confidence,
        "box_xyxy": box,
        "source": source,
        "verified_by": verified_by,
    }


def _unsure_face(box: Box, label: str, reason: str, detection_confidence: float | None) -> UnsureFaceItem:
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


def _resolve_entity(entity: Entity, verdict: Verdict) -> Resolved | None:
    """
    Decide what a detector entity really is, using the verdict on its crop.

    Returns the entity as a non-human face or as an unsure candidate, or None when the detector fired on a
    real human face or on something that is not a face.
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
            return Resolved(other={**detector_only, "verified_by": "SigLIP2"})
        # The verifier disagrees with the detector's type (e.g. a "dog" that is really a drawing): trust the verifier.
        return Resolved(
            other=_other_face(
                entity["box_xyxy"], verified_type, verdict["label"], verdict["confidence"], entity["source"], "SigLIP2"
            )
        )

    # The verifier could not decide (or is unavailable): rely on the detector only when it is trustworthy.
    if entity_type == "anime":
        # The anime detector alone is not trustworthy on photos: it also fires on animals and paintings.
        return Resolved(
            unsure=_unsure_face(
                entity["box_xyxy"],
                "Possible anime/cartoon face (detector only, unverified)",
                "Only the anime detector reported this region; the verification model could not confirm it.",
                entity["confidence"],
            )
        )
    if (entity["confidence"] or 0.0) < settings.detection.animal_detector_only_min_conf:
        # A weak COCO "dog/cat" box that the verifier could not confirm is often a face-filling false positive.
        return Resolved(
            unsure=_unsure_face(
                entity["box_xyxy"],
                f"Possible {entity['label'].lower()} (detector only, low confidence)",
                "Only the object detector reported this region, with low confidence, and it was not verified.",
                entity["confidence"],
            )
        )
    return Resolved(other=detector_only)


def _overlaps(a: Box, b: Box) -> bool:
    """True when two reports are the same object: strong overlap, or the centre of ``a`` lies inside ``b``."""
    return box_iou(a, b) > SAME_OBJECT_IOU or box_center_inside(a, b)


def _is_duplicate_animal(a: OtherFaceItem, b: OtherFaceItem) -> bool:
    """The same animal reported twice, e.g. a body box and a face box, or by two detectors."""
    if box_iou(a["box_xyxy"], b["box_xyxy"]) > DUPLICATE_IOU:
        return True
    creatures = ("animal", "bird")
    same_family = a["face_type"] == b["face_type"] or (a["face_type"] in creatures and b["face_type"] in creatures)
    return same_family and (
        box_center_inside(a["box_xyxy"], b["box_xyxy"]) or box_center_inside(b["box_xyxy"], a["box_xyxy"])
    )


def _merge_entities(image_bgr: np.ndarray, census: FaceCensus, entities: list[Entity]) -> None:
    """Verify the detector entities and reconcile them with the face candidates that are already classified."""
    verdicts = classify_boxes(image_bgr, [entity["box_xyxy"] for entity in entities])
    reported: list[OtherFaceItem] = []
    for entity, verdict in zip(entities, verdicts, strict=True):
        resolved = _resolve_entity(entity, verdict)
        if resolved is None:
            census.rejected += 1
        elif resolved.other is not None:
            reported.append(resolved.other)
        elif resolved.unsure is not None and not any(
            _overlaps(resolved.unsure["box_xyxy"], existing["box_xyxy"]) for existing in census.unsure
        ):
            census.unsure.append(resolved.unsure)  # the same region may already be listed from the face candidates

    # Keep one report per animal; prefer the YOLOX one, then the more confident one.
    reported.sort(key=lambda item: (item["source"] != MODEL_NAME_YOLOX, -(item["confidence"] or 0.0)))
    unique: list[OtherFaceItem] = []
    for item in reported:
        if not any(_is_duplicate_animal(item, kept) for kept in unique):
            unique.append(item)

    # A face candidate already judged non-human/unsure is the same object as an overlapping detector entity.
    for item in unique:
        census.others = [
            other
            for other in census.others
            if other["source"] != YUNET_SOURCE or not _overlaps(other["box_xyxy"], item["box_xyxy"])
        ]
        census.unsure = [
            candidate for candidate in census.unsure if not _overlaps(candidate["box_xyxy"], item["box_xyxy"])
        ]
        census.others.append(item)


def census_faces(image_bgr: np.ndarray) -> FaceCensus:
    """Detect every face-like region and classify each one."""
    census = FaceCensus()
    # The face detectors and the animal / person / anime detectors do not depend on each other.
    (faces, face_models), (entities, persons, entity_models) = run_pair(
        lambda: detect_faces(image_bgr), lambda: detect_entities_and_persons(image_bgr)
    )
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


def _match_body(face_box: Box, persons: list[Box]) -> Box | None:
    """
    The person box that contains this face and shows a real body (much taller than the face).

    A person box about as tall as the face (a head-only or head-and-shoulders crop) adds nothing for the age
    model and is ignored. Among qualifying boxes the tightest one is used.
    """
    face_height = face_box[3] - face_box[1]
    candidates = [
        box
        for box in persons
        if box_center_inside(face_box, box)
        and (box[3] - box[1]) >= settings.detection.body_min_height_ratio * face_height
    ]
    return min(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) if candidates else None


def _build_human_faces(
    image_bgr: np.ndarray, humans: list[tuple[DetectedFace, Verdict]], persons: list[Box]
) -> tuple[list[HumanFaceItem], str | None]:
    """Quality check, age and gender for every verified human face. Also returns the age/gender model if it ran."""
    height, width = image_bgr.shape[:2]
    faces: list[HumanFaceItem] = []
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
                "possible_minor": is_possible_minor(demographics.p_under_18),
                "notes": demographics.notes,
            }
        )
    return faces, demographics_model


def _number_faces[T: (OtherFaceItem, UnsureFaceItem)](items: list[T], width: int, height: int) -> list[T]:
    """Give non-human and unsure faces their ids (reading order) and percent boxes."""
    for face_id, item in enumerate(items, start=1):
        item["face_id"] = face_id
        item["box_percent"] = box_to_percent(item["box_xyxy"], width, height)
    return items


# --------------------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------------------


def _annotate(
    image_bgr: np.ndarray,
    humans: list[HumanFaceItem],
    others: list[OtherFaceItem],
    unsure: list[UnsureFaceItem],
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
) -> list[PipelineStepItem]:
    """The four stages shown in the UI stepper, with what each one did."""
    if census is None:
        detection_status, detection_detail = SKIPPED, SKIPPED_NO_INFORMATION
    elif census.detection_ran:
        detection_status = COMPLETED
        detection_detail = (
            f"{len(census.humans)} real human face(s), {len(census.others)} non-human face(s), "
            f"{len(census.unsure)} unverified candidate(s)."
        )
    else:
        detection_status, detection_detail = FAILED, "No face detector could run."

    max_mb = settings.upload.max_bytes // (1024 * 1024)
    details = (
        (COMPLETED, f"The file is a valid image within the {max_mb} MB limit."),
        (COMPLETED if has_information else FAILED, info_detail),
        (detection_status, detection_detail),
        (
            COMPLETED if demographics_ran else SKIPPED,
            "Age (3-year brackets) and gender estimated for every verified real human face."
            if demographics_ran
            else "Skipped: no verified real human face, or the age/gender model is unavailable.",
        ),
    )
    return [
        {"step_id": step_id, "step_number": number, "title": title, "status": status, "detail": detail}
        for number, ((step_id, title), (status, detail)) in enumerate(zip(STAGES, details, strict=True), start=1)
    ]


def _diagnostics(
    bgr: np.ndarray,
    models: list[str],
    file_format: str,
    meta: DecodedImage | None,
    rejected: int,
    elapsed_ms: float,
) -> DiagnosticsItem:
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


def _not_checked_image_type() -> ImageTypeItem:
    return {
        "status": "NOT_CHECKED",
        "label": "Not checked",
        "photo_share": None,
        "synthetic_share": None,
        "predictions": [],
        "model": None,
        "reason": "Not checked because the image has no visual information.",
    }


def _no_information_response(
    bgr: np.ndarray,
    info: ImageInfoItem,
    filename: str | None,
    file_format: str,
    meta: DecodedImage | None,
    render_annotation: bool,
) -> AnalysisResult:
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
        "image_type": _not_checked_image_type(),
        "pipeline_steps": _pipeline_steps(
            f"Blank or solid colour (entropy {info['entropy']:.2f}).", False, None, False
        ),
        "diagnostics": _diagnostics(bgr, [MODEL_INFO_CHECK], file_format, meta, 0, 0.0),
        "annotated_image": bgr_to_base64_jpeg(bgr) if render_annotation else None,
        "filename": filename,
        "legend": "No visual information or structure was found in this image.",
        "disclaimer": "Please upload a photograph with enough light, contrast and visual content.",
    }


def _emit(report: ProgressReporter | None, step_id: str, status: str, detail: str = "", started: float = 0.0) -> None:
    """Tell the progress callback (if any) that a stage changed state."""
    if report is not None:
        elapsed_ms = (time.perf_counter() - started) * 1000 if started else 0.0
        report(stage_event(step_id, status, detail, elapsed_ms))


def analyze_profile_photo_image(
    image: Image.Image,
    filename: str | None = "profile.jpg",
    file_format: str = DEFAULT_FORMAT,
    meta: DecodedImage | None = None,
    *,
    report: ProgressReporter | None = None,
    render_annotation: bool = True,
) -> AnalysisResult:
    """
    Run the complete analysis on a decoded image and return the response payload (see ``AnalysisResponse``).

    ``report`` is called whenever a stage starts or finishes; ``render_annotation=False`` leaves out the annotated
    image (the largest part of the response).
    """
    started = time.perf_counter()
    bgr = pil_to_bgr(image.convert("RGB"))
    height, width = bgr.shape[:2]
    _emit(report, "validate", COMPLETED, "The upload is a valid image within the limits.")

    stage_started = time.perf_counter()
    _emit(report, "information", RUNNING)
    info = check_image_information(bgr)
    if not info["has_information"]:
        _emit(report, "information", FAILED, info["reason"], stage_started)
        _emit(report, "detection", SKIPPED, SKIPPED_NO_INFORMATION)
        _emit(report, "demographics", SKIPPED, SKIPPED_NO_INFORMATION)
        response = _no_information_response(bgr, info, filename, file_format, meta, render_annotation)
        response["diagnostics"]["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return response
    _emit(report, "information", COMPLETED, info["reason"], stage_started)

    stage_started = time.perf_counter()
    _emit(report, "detection", RUNNING)
    image_type = classify_image_type(bgr)
    census = census_faces(bgr)
    _emit(
        report,
        "detection",
        COMPLETED if census.detection_ran else FAILED,
        f"{len(census.humans)} real human face(s), {len(census.others)} non-human face(s), "
        f"{len(census.unsure)} unverified candidate(s).",
        stage_started,
    )

    stage_started = time.perf_counter()
    _emit(report, "demographics", RUNNING)
    humans, demographics_model = _build_human_faces(bgr, census.humans, census.persons)
    _emit(
        report,
        "demographics",
        COMPLETED if demographics_model else SKIPPED,
        f"Age and gender estimated for {len(humans)} face(s)." if demographics_model else "No age estimate was made.",
        stage_started,
    )

    others = _number_faces(census.others, width, height)
    unsure = _number_faces(census.unsure, width, height)
    decision = decide(humans, others, unsure, image_type)

    models = [MODEL_INFO_CHECK, *([image_type["model"]] if image_type["model"] else []), *census.models]
    models += [demographics_model] if demographics_model else []
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "status": decision.status,
        "accepted": decision.accepted,
        "reason": decision.reason,
        "summary": summarise(humans, others, unsure),
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
        "annotated_image": bgr_to_base64_jpeg(_annotate(bgr, humans, others, unsure)) if render_annotation else None,
        "filename": filename,
        "legend": (
            "Green boxes are verified real human faces, purple boxes are non-human faces, "
            "orange boxes could not be verified."
            if humans or others or unsure
            else "No face was located in this image."
        ),
        "disclaimer": DISCLAIMER,
    }
