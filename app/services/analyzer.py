"""End-to-end profile photo analysis.

Flow: (1) validated upload -> (2) is there any information? -> (3) find face candidates and verify,
per face, whether each is a real human face, an animal, a cartoon, ... -> (4) estimate age and gender
for every verified real human face -> (5) report only what the models actually produced; anything
they cannot support is reported as "Unsure" or "Not Detected".
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
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
    detect_non_human_entities,
)
from app.services.face_service import (
    NOT_AVAILABLE,
    NOT_DETECTED,
    UNSURE,
    DetectedFace,
    detect_faces,
    predict_face_demographics,
)
from app.services.image_utils import (
    BoxItem,
    DecodedImage,
    annotate_boxes,
    bgr_to_base64_jpeg,
    box_center_inside,
    box_iou,
    clamp_box,
    compute_normalized_box,
    pil_to_bgr,
)
from app.services.quality_service import assess_face_quality, check_image_information

logger = logging.getLogger(__name__)

COLOR_ORANGE = (0, 140, 255)  # BGR
COLOR_GREEN = (0, 180, 0)
COLOR_PURPLE = (200, 50, 180)

DEFAULT_IMAGE_FORMAT = "image/jpeg"
MODEL_INFO_ANALYZER = "Image Information & Entropy Analyzer"
NO_INFO_REASON = "This image has no information to check or process (blank or solid color)."
ENTITY_MERGE_IOU = 0.30
ENTITY_DEDUPE_IOU = 0.50
ROW_BUCKET_PX = 50
YUNET_SOURCE = "YuNet candidate"

NON_PHOTO_STATUSES = ("NON_PHOTO", "NEEDS_REVIEW", "UNCERTAIN")


@dataclass
class FaceCensus:
    """Everything found in the image, grouped by what each detection was verified to be."""

    humans: list[tuple[DetectedFace, dict[str, Any]]] = field(default_factory=list)
    others: list[dict[str, Any]] = field(default_factory=list)
    unsure: list[dict[str, Any]] = field(default_factory=list)
    rejected: int = 0
    models: list[str] = field(default_factory=list)
    detection_ran: bool = False


def _reading_key(box: list[int]) -> tuple[int, int]:
    return (round(box[1] / ROW_BUCKET_PX), box[0])


def _other_item(
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


def _unsure_item(box: list[int], verdict: dict[str, Any], det_score: float | None) -> dict[str, Any]:
    if verdict["verdict"] == VERDICT_UNAVAILABLE:
        reason = "The verification model is unavailable, so this candidate could not be confirmed as a real human face."
    else:
        reason = "The model could not tell with enough certainty whether this is a real human face."
    return {
        "label": verdict["label"],
        "reason": reason,
        "detection_confidence": det_score,
        "box_xyxy": box,
    }


def _classify_face_candidates(image_bgr: np.ndarray, census: FaceCensus, faces: list[DetectedFace]) -> None:
    verdicts = classify_boxes(image_bgr, [f.bbox for f in faces])
    for face, verdict in zip(faces, verdicts, strict=True):
        kind = verdict["verdict"]
        if kind == VERDICT_HUMAN:
            census.humans.append((face, verdict))
        elif kind == VERDICT_NON_HUMAN:
            census.others.append(
                _other_item(
                    face.bbox, verdict["face_type"], verdict["label"], verdict["confidence"], YUNET_SOURCE, "SigLIP2"
                )
            )
        elif kind == VERDICT_NOT_FACE:
            census.rejected += 1
        else:
            census.unsure.append(_unsure_item(face.bbox, verdict, face.det_score))


CARTOON_FAMILY = ("anime", "cartoon_or_illustration", "render_3d", "painting")
BUCKET_OTHER = "other"
BUCKET_UNSURE = "unsure"


def _entity_item(entity: dict[str, Any], verdict: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """
    Decide what a detector entity really is, using the verdict of its crop.

    Returns ``(item, bucket)`` or None when the detector fired on a real face / a non-face.
    """
    kind = verdict["verdict"]
    entity_type = entity["face_type"]
    detector_only = _other_item(
        entity["box_xyxy"], entity_type, entity["label"], entity["confidence"], entity["source"], "detector"
    )
    if kind in (VERDICT_HUMAN, VERDICT_NOT_FACE):
        return None

    if kind == VERDICT_NON_HUMAN:
        semantic_type = verdict["face_type"]
        if entity_type == "anime" and semantic_type in CARTOON_FAMILY:
            return {**detector_only, "verified_by": "SigLIP2"}, BUCKET_OTHER
        if entity_type in ("animal", "bird") and semantic_type in ("animal", "bird"):
            return {**detector_only, "verified_by": "SigLIP2"}, BUCKET_OTHER
        return (
            _other_item(
                entity["box_xyxy"], semantic_type, verdict["label"], verdict["confidence"], entity["source"], "SigLIP2"
            ),
            BUCKET_OTHER,
        )

    # unsure / unavailable
    if entity_type == "anime":
        # The anime detector alone is not trustworthy on photos (it fires on animals and paintings).
        return (
            {
                "label": "Possible anime/cartoon face (detector only, unverified)",
                "reason": "Only the anime detector reported this region; the verification model could not confirm it.",
                "detection_confidence": entity["confidence"],
                "box_xyxy": entity["box_xyxy"],
            },
            BUCKET_UNSURE,
        )
    return detector_only, BUCKET_OTHER


def _same_family(a: dict[str, Any], b: dict[str, Any]) -> bool:
    creature = ("animal", "bird")
    return a["face_type"] == b["face_type"] or (a["face_type"] in creature and b["face_type"] in creature)


def _duplicate_entities(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if box_iou(a["box_xyxy"], b["box_xyxy"]) > ENTITY_DEDUPE_IOU:
        return True
    return _same_family(a, b) and (
        box_center_inside(a["box_xyxy"], b["box_xyxy"]) or box_center_inside(b["box_xyxy"], a["box_xyxy"])
    )


def _overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return box_iou(a["box_xyxy"], b["box_xyxy"]) > ENTITY_MERGE_IOU or box_center_inside(a["box_xyxy"], b["box_xyxy"])


def _merge_entities(image_bgr: np.ndarray, census: FaceCensus, entities: list[dict[str, Any]]) -> None:
    verdicts = classify_boxes(image_bgr, [e["box_xyxy"] for e in entities])
    kept: list[dict[str, Any]] = []
    for entity, verdict in zip(entities, verdicts, strict=True):
        resolved = _entity_item(entity, verdict)
        if resolved is None:
            census.rejected += 1
            continue
        item, bucket = resolved
        if bucket == BUCKET_UNSURE:
            if not any(_overlaps(item, other) for other in census.unsure):
                census.unsure.append(item)  # the same region may already be listed from the YuNet candidates
        else:
            kept.append(item)

    # Same animal reported twice (body box + face box, or two detectors): keep the best-sourced one.
    kept.sort(key=lambda it: (it["source"] != MODEL_NAME_YOLOX, -(it["confidence"] or 0.0)))
    unique: list[dict[str, Any]] = []
    for item in kept:
        if not any(_duplicate_entities(item, other) for other in unique):
            unique.append(item)

    # A YuNet candidate already judged non-human/unsure is the same object as an overlapping entity.
    for item in unique:
        box = item["box_xyxy"]

        def same_object(other: dict[str, Any], box: list[int] = box) -> bool:
            return box_iou(box, other["box_xyxy"]) > ENTITY_MERGE_IOU or box_center_inside(other["box_xyxy"], box)

        census.others = [o for o in census.others if o["source"] != YUNET_SOURCE or not same_object(o)]
        census.unsure = [u for u in census.unsure if not same_object(u)]
        census.others.append(item)


def census_faces(image_bgr: np.ndarray) -> FaceCensus:
    """Detect all face-like regions and classify each one."""
    census = FaceCensus()
    faces, face_model = detect_faces(image_bgr)
    entities, entity_models = detect_non_human_entities(image_bgr)
    census.detection_ran = face_model is not None
    census.models = [m for m in (face_model, *entity_models) if m]

    _classify_face_candidates(image_bgr, census, faces)
    _merge_entities(image_bgr, census, entities)

    if semantic_service.is_available():
        census.models.append(MODEL_NAME_SIGLIP)

    census.humans.sort(key=lambda pair: _reading_key(pair[0].bbox))
    census.others.sort(key=lambda item: _reading_key(item["box_xyxy"]))
    census.unsure.sort(key=lambda item: _reading_key(item["box_xyxy"]))
    return census


def _build_human_items(
    image_bgr: np.ndarray, humans: list[tuple[DetectedFace, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], str | None]:
    height, width = image_bgr.shape[:2]
    items: list[dict[str, Any]] = []
    demographics_model: str | None = None

    for idx, (face, verdict) in enumerate(humans, start=1):
        box = list(clamp_box(face.bbox, width, height))
        quality = assess_face_quality(image_bgr, box, face.kps)
        demo = predict_face_demographics(
            image_bgr, box, usable=quality["demographics_usable"], caveat=quality["demographics_caveat"]
        )
        demographics_model = demo.model or demographics_model
        minor_risk = demo.p_under_18 is not None and demo.p_under_18 >= settings.MINOR_RISK_PROBABILITY
        items.append(
            {
                "face_id": idx,
                "box_xyxy": box,
                "box_normalized": compute_normalized_box(box, width, height),
                "detection_confidence": face.det_score,
                "verification_confidence": verdict["confidence"],
                "gender": demo.gender,
                "gender_confidence": demo.gender_confidence,
                "age_group": demo.age_group,
                "age_status": demo.age_status,
                "estimated_age": demo.estimated_age,
                "age_range": demo.age_range,
                "age_confidence": demo.age_confidence,
                "p_under_18": demo.p_under_18,
                "possible_minor_risk": minor_risk,
                "demographics_notes": demo.notes,
                "face_visibility": quality,
                "is_primary": idx == 1,
            }
        )
    return items, demographics_model


def _number_boxes(items: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    for idx, item in enumerate(items, start=1):
        item["face_id"] = idx
        item["box_normalized"] = compute_normalized_box(item["box_xyxy"], width, height)
    return items


def _decide(
    humans: list[dict[str, Any]],
    others: list[dict[str, Any]],
    unsure: list[dict[str, Any]],
    image_type: dict[str, Any],
) -> tuple[bool, str, str]:
    """Profile-photo policy: exactly one verified real human face and nothing doubtful."""
    other_labels = ", ".join(sorted({o["label"] for o in others}))
    type_status = image_type.get("status")
    type_warning = (
        f"Warning: this looks like a non-photo image ({image_type.get('display_label')}). "
        if type_status == "NON_PHOTO"
        else ""
    )

    if not humans:
        if others:
            return (
                False,
                "NON_PHOTO_WARNING",
                f"{type_warning}No real human face was detected. Found {len(others)} other face(s) ({other_labels}).",
            )
        if unsure:
            return (
                False,
                "MANUAL_REVIEW_REQUIRED",
                f"{len(unsure)} face candidate(s) could not be verified as real human faces. Manual review is needed.",
            )
        return False, "NON_PHOTO_WARNING" if type_warning else "REJECTED", f"{type_warning}No human face was detected."

    if len(humans) > 1:
        return (
            False,
            "REJECTED",
            f"{type_warning}A profile photo must contain exactly one real human face; found {len(humans)}.",
        )

    primary = humans[0]
    visibility = primary["face_visibility"]
    if type_status in NON_PHOTO_STATUSES:
        return (
            False,
            "MANUAL_REVIEW_REQUIRED",
            f"{type_warning}Image screening ({image_type.get('reason')}) requires human review.",
        )
    if others:
        return (
            False,
            "NON_PHOTO_WARNING",
            f"One real human face detected, but the image also contains {len(others)} non-human face(s) ({other_labels}).",
        )
    if unsure:
        return (
            False,
            "MANUAL_REVIEW_REQUIRED",
            f"One real human face detected, but {len(unsure)} other candidate(s) could not be verified.",
        )
    if visibility["status"] != "LIKELY_COMPLETE":
        return False, "REJECTED_OR_REVIEW", f"{visibility['reason']} Please provide a clear, centered, uncropped photo."
    if primary["age_group"] in (UNSURE, NOT_DETECTED, NOT_AVAILABLE):
        return (
            False,
            "AGE_ESTIMATION_UNAVAILABLE",
            "The face was verified, but its age could not be estimated with enough certainty (Unsure).",
        )
    if primary["possible_minor_risk"]:
        return (
            False,
            "AGE_VERIFICATION_REQUIRED",
            f"There is a {primary['p_under_18']:.0%} estimated chance this person is under 18 "
            f"(estimated group {primary['age_group']}). Age verification is required.",
        )
    return (
        True,
        "INITIAL_CHECKS_PASSED",
        f"All checks passed. Estimated age group is {primary['age_group']} (appearance-based estimate).",
    )


def _summary(humans: list[dict[str, Any]], others: list[dict[str, Any]], unsure: list[dict[str, Any]]) -> str:
    parts = [f"{len(humans)} real human face(s)"]
    if others:
        by_label: dict[str, int] = {}
        for item in others:
            by_label[item["label"]] = by_label.get(item["label"], 0) + 1
        parts.append(f"{len(others)} non-human face(s): " + ", ".join(f"{n} x {lbl}" for lbl, n in by_label.items()))
    else:
        parts.append("0 non-human faces")
    parts.append(f"{len(unsure)} unverified candidate(s)")
    return "; ".join(parts) + "."


def _annotate(
    image_bgr: np.ndarray, humans: list[dict[str, Any]], others: list[dict[str, Any]], unsure, accepted: bool
):
    items: list[BoxItem] = []
    for h in humans:
        text = f"#{h['face_id']} {h['gender']} {h['age_group']}"
        items.append((h["box_xyxy"], text, COLOR_GREEN if accepted else COLOR_ORANGE))
    for o in others:
        items.append((o["box_xyxy"], f"{o['label']}", COLOR_PURPLE))
    for u in unsure:
        items.append((u["box_xyxy"], "UNSURE", COLOR_ORANGE))
    return annotate_boxes(image_bgr, items)


def _steps(
    has_info: bool, info_detail: str, census: FaceCensus | None, demographics_done: bool
) -> list[dict[str, Any]]:
    if census is None:
        detection_status, detection_detail = "skipped", "Skipped due to lack of visual information."
    else:
        detection_status = "completed" if census.detection_ran else "failed"
        detection_detail = (
            f"{len(census.humans)} real human face(s), {len(census.others)} other face(s), "
            f"{len(census.unsure)} unverified candidate(s)."
            if census.detection_ran
            else "Face detection could not run."
        )
    return [
        {
            "step_id": "validate",
            "step_number": 1,
            "title": "File & Binary Validation",
            "status": "completed",
            "detail": f"Authentic binary image validated & decodable within {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        },
        {
            "step_id": "information",
            "step_number": 2,
            "title": "Visual Information Check",
            "status": "completed" if has_info else "failed",
            "detail": info_detail,
        },
        {
            "step_id": "detection",
            "step_number": 3,
            "title": "Face Detection & Verification",
            "status": detection_status,
            "detail": detection_detail,
        },
        {
            "step_id": "demographics",
            "step_number": 4,
            "title": "Demographics & 3-Year Age Buckets",
            "status": "completed" if demographics_done else "skipped",
            "detail": "Age and gender estimated for every verified real human face."
            if demographics_done
            else "Skipped (no verified real human face, or the age/gender model is unavailable).",
        },
    ]


def _diagnostics(
    bgr: np.ndarray,
    info: dict[str, Any],
    models: list[str],
    format_detected: str,
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
        "file_format_detected": format_detected or DEFAULT_IMAGE_FORMAT,
        "entropy_score": round(float(info.get("entropy", 0.0)), 2),
        "std_dev_score": round(float(info.get("std_dev", 0.0)), 2),
        "laplacian_var": round(float(info.get("laplacian_var", 0.0)), 2),
        "rejected_candidate_count": rejected,
        "models_executed": list(dict.fromkeys(models)),
    }


def _not_checked_image_type() -> dict[str, Any]:
    return {
        "status": "NOT_CHECKED",
        "top_class": NOT_DETECTED,
        "top_class_confidence": None,
        "photo_score": None,
        "cartoon_score": None,
        "display_label": "Not checked",
        "model_executed": None,
        "all_predictions": [],
        "scores": {},
        "reason": "Not checked because the image contains no visual information.",
    }


def _legacy_fields(humans: list[dict[str, Any]], decision_accepted: bool) -> dict[str, Any]:
    """Single-face convenience fields; multi-face results are described by ``human_faces``."""
    if len(humans) == 1:
        h = humans[0]
        unsure_age = h["age_group"] in (UNSURE, NOT_DETECTED)
        return {
            "face_count": 1,
            "face_visibility": h["face_visibility"],
            "estimated_age_group": h["age_group"],
            "age_model_confidence": h["age_confidence"],
            "age_note": "; ".join(h["demographics_notes"])
            or "Appearance-based estimate only; not proof of legal age. Age error is typically several years.",
            "age_prediction_error": "Age could not be determined with enough certainty." if unsure_age else None,
            "possible_minor_risk": h["possible_minor_risk"],
            "age_verification_required": h["possible_minor_risk"] or not decision_accepted,
            "gender_presentation_prediction": h["gender"],
            "gender_note": "Appearance-based model prediction only; not a confirmed gender identity.",
            "gender_prediction_error": "Gender could not be determined with enough certainty."
            if h["gender"] in (UNSURE, NOT_DETECTED)
            else None,
            "face_detection_confidence": h["detection_confidence"],
            "face_box_xyxy": h["box_xyxy"],
            "age_gender_prediction_performed": h["age_status"] != NOT_DETECTED,
        }
    placeholder = NOT_AVAILABLE if humans else NOT_DETECTED
    note = "Multiple faces: see the per-face results." if humans else "No real human face was verified."
    return {
        "face_count": len(humans),
        "face_visibility": None,
        "estimated_age_group": placeholder,
        "age_model_confidence": None,
        "age_note": note,
        "age_prediction_error": note,
        "possible_minor_risk": any(h["possible_minor_risk"] for h in humans),
        "age_verification_required": True,
        "gender_presentation_prediction": placeholder,
        "gender_note": note,
        "gender_prediction_error": note,
        "face_detection_confidence": humans[0]["detection_confidence"] if humans else None,
        "face_box_xyxy": humans[0]["box_xyxy"] if humans else None,
        "age_gender_prediction_performed": any(h["age_status"] != NOT_DETECTED for h in humans),
    }


def _handle_no_info(
    bgr: np.ndarray, info: dict[str, Any], filename: str | None, format_detected: str, meta: DecodedImage | None
) -> dict[str, Any]:
    encoded = bgr_to_base64_jpeg(bgr)
    legacy = _legacy_fields([], False)
    return {
        "accepted": False,
        "posting_status": "NO_INFORMATION",
        "posting_reason": NO_INFO_REASON,
        "reason": NO_INFO_REASON,
        "summary": "No visual information: nothing was checked.",
        "has_information": False,
        "image_info_check": info,
        "image_type_check": _not_checked_image_type(),
        "image_type_review_required": False,
        "face_detected": False,
        "total_face_count": 0,
        "real_human_face_count": 0,
        "other_face_count": 0,
        "unsure_face_count": 0,
        "human_faces": [],
        "other_faces": [],
        "unsure_faces": [],
        "pipeline_steps": _steps(False, f"Blank or solid color (Entropy: {info.get('entropy', 0.0):.2f})", None, False),
        "technical_diagnostics": _diagnostics(bgr, info, [MODEL_INFO_ANALYZER], format_detected, meta, 0, 0.0),
        "processing_time_ms": 0.0,
        **legacy,
        "age_note": "No visual information to analyze.",
        "face_regions_xyxy": [],
        "original_image": encoded,
        "annotated_image": encoded,
        "filename": filename,
        "region_note": "No visual information or structure was found in this image.",
        "note": "Please upload a photograph with sufficient lighting, contrast, and visual content.",
    }


def analyze_profile_photo_image(
    image: Image.Image,
    filename: str | None = "profile.jpg",
    format_detected: str = DEFAULT_IMAGE_FORMAT,
    meta: DecodedImage | None = None,
) -> dict[str, Any]:
    """Run the complete analysis and return the API payload."""
    started = time.perf_counter()
    bgr = pil_to_bgr(image.convert("RGB"))
    height, width = bgr.shape[:2]

    info = check_image_information(bgr)
    if not info["has_information"]:
        result = _handle_no_info(bgr, info, filename, format_detected, meta)
        result["processing_time_ms"] = result["technical_diagnostics"]["processing_time_ms"] = round(
            (time.perf_counter() - started) * 1000, 1
        )
        return result

    image_type = classify_image_type(bgr)
    census = census_faces(bgr)

    humans, demographics_model = _build_human_items(bgr, census.humans)
    others = _number_boxes(census.others, width, height)
    unsure = _number_boxes(census.unsure, width, height)
    accepted, status, reason = _decide(humans, others, unsure, image_type)

    models = [MODEL_INFO_ANALYZER, *([image_type["model_executed"]] if image_type.get("model_executed") else [])]
    models += census.models + ([demographics_model] if demographics_model else [])
    demographics_done = demographics_model is not None
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    total = len(humans) + len(others)
    regions = [h["box_xyxy"] for h in humans] + [o["box_xyxy"] for o in others] + [u["box_xyxy"] for u in unsure]
    return {
        "accepted": accepted,
        "posting_status": status,
        "posting_reason": reason,
        "reason": reason,
        "summary": _summary(humans, others, unsure),
        "has_information": True,
        "image_info_check": info,
        "image_type_check": image_type,
        "image_type_review_required": image_type["status"] in NON_PHOTO_STATUSES or bool(others) or bool(unsure),
        "face_detected": bool(total or unsure),
        "total_face_count": total,
        "real_human_face_count": len(humans),
        "other_face_count": len(others),
        "unsure_face_count": len(unsure),
        "human_faces": humans,
        "other_faces": others,
        "unsure_faces": unsure,
        "pipeline_steps": _steps(
            True, f"Valid visual content (Entropy: {info['entropy']:.2f})", census, demographics_done
        ),
        "technical_diagnostics": _diagnostics(bgr, info, models, format_detected, meta, census.rejected, elapsed_ms),
        "processing_time_ms": elapsed_ms,
        **_legacy_fields(humans, accepted),
        "face_regions_xyxy": regions,
        "original_image": bgr_to_base64_jpeg(bgr),
        "annotated_image": bgr_to_base64_jpeg(_annotate(bgr, humans, others, unsure, accepted)),
        "filename": filename,
        "region_note": (
            "Green boxes are verified real human faces, purple boxes are non-human faces, "
            "orange boxes could not be verified."
            if regions
            else "No face region could be located. The original uploaded photo is shown for review."
        ),
        "note": "Age and gender are appearance-based estimates with a typical age error of several years. "
        "Keep a separate age-verification workflow for any strict 18+ policy.",
    }
