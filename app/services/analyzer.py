import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.image_utils import (
    pil_to_bgr,
    bgr_to_base64_jpeg,
    clamp_box,
    compute_normalized_box,
    annotate_faces,
)
from app.services.quality_service import (
    assess_face_quality,
    check_image_information
)
from app.services.classifier_service import (
    classify_image_type,
    detect_non_human_entities,
    check_is_non_human_face,
)
from app.services.face_service import (
    detect_faces,
    predict_face_demographics,
    possible_minor_age_group,
    sort_faces_reading_order,
    NOT_AVAILABLE,
    NOT_DETECTED,
)

logger = logging.getLogger(__name__)

COLOR_ORANGE = (0, 140, 255)  # BGR
COLOR_GREEN = (0, 180, 0)     # BGR
COLOR_PURPLE = (200, 50, 180) # BGR

MULTIPLE_FACES_REASON = "A profile photo must contain exactly one detected face."
MULTIPLE_FACES_NOTE = "Multiple faces detected."
NO_INFO_REASON = "This image has no information to check or process (blank or solid color)."
NO_FACE_DETECTED = "No face detected."
DEFAULT_IMAGE_FORMAT = "image/jpeg"


def _determine_step3_status(info_ok: bool, human_count: int) -> str:
    """Helper to determine step 3 status without nested conditionals."""
    if not info_ok:
        return "skipped"
    if human_count == 1:
        return "completed"
    return "failed"


def _build_pipeline_steps(
    info_ok: bool,
    info_detail: str,
    human_count: int,
    other_count: int,
    demographics_done: bool,
    demographics_detail: str
) -> List[Dict[str, Any]]:
    """Construct structured status items for the 4-stage pipeline stepper."""
    step3_status = _determine_step3_status(info_ok, human_count)
    step4_status = "completed" if demographics_done else "skipped"

    return [
        {
            "step_id": "validate",
            "step_number": 1,
            "title": "File & Binary Validation",
            "status": "completed",
            "detail": "Authentic binary image validated & decodable within 10 MB limit."
        },
        {
            "step_id": "information",
            "step_number": 2,
            "title": "Visual Information Check",
            "status": "completed" if info_ok else "failed",
            "detail": info_detail
        },
        {
            "step_id": "detection",
            "step_number": 3,
            "title": "Face Detection & Discrimination",
            "status": step3_status,
            "detail": f"{human_count} human face(s), {other_count} other face(s) detected." if info_ok else "Skipped due to lack of visual information."
        },
        {
            "step_id": "demographics",
            "step_number": 4,
            "title": "Demographics & 3-Year Age Buckets",
            "status": step4_status,
            "detail": demographics_detail if demographics_done else "Skipped (requires exactly 1 real human face)."
        }
    ]


def _build_technical_diagnostics(
    width: int,
    height: int,
    format_detected: str,
    info_check: Dict[str, Any],
    models_executed: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Generate technical metadata and computer vision model diagnostics."""
    default_models = [
        "YuNet Face Detector (2023mar)",
        "InsightFace GenderAge (multitask CNN)",
        "DeepGHS Anime Face Detector (v1.4n)",
        "Ultralytics YOLOv8n (COCO)"
    ]
    return {
        "processing_time_ms": 0.0,
        "image_width": width,
        "image_height": height,
        "file_format_detected": format_detected or DEFAULT_IMAGE_FORMAT,
        "entropy_score": round(float(info_check.get("entropy", 0.0)), 2),
        "std_dev_score": round(float(info_check.get("std_dev", 0.0)), 2),
        "laplacian_var": round(float(info_check.get("laplacian_var", 0.0)), 2),
        "models_executed": models_executed if models_executed is not None else default_models
    }


def _handle_no_info(
    image_bgr: np.ndarray,
    info_check: Dict[str, Any],
    image_type: Dict[str, Any],
    filename: Optional[str],
    format_detected: str = DEFAULT_IMAGE_FORMAT
) -> Dict[str, Any]:
    """Generate response when image contains no visual information."""
    height, width = image_bgr.shape[:2]
    img_b64 = bgr_to_base64_jpeg(image_bgr)
    entropy_val = info_check.get("entropy", 0.0)
    steps = _build_pipeline_steps(
        info_ok=False,
        info_detail=f"Blank or solid color (Entropy: {entropy_val:.2f})",
        human_count=0,
        other_count=0,
        demographics_done=False,
        demographics_detail=""
    )
    diagnostics = _build_technical_diagnostics(
        width, height, format_detected, info_check,
        models_executed=["Image Information & Entropy Analyzer"]
    )

    return {
        "accepted": False,
        "posting_status": "NO_INFORMATION",
        "posting_reason": NO_INFO_REASON,
        "reason": NO_INFO_REASON,
        "has_information": False,
        "image_info_check": info_check,
        "image_type_check": image_type,
        "image_type_review_required": True,
        "face_detected": False,
        "total_face_count": 0,
        "real_human_face_count": 0,
        "other_face_count": 0,
        "human_faces": [],
        "other_faces": [],
        "pipeline_steps": steps,
        "technical_diagnostics": diagnostics,
        "processing_time_ms": 0.0,
        "face_count": 0,
        "face_visibility": None,
        "estimated_age_group": NOT_DETECTED,
        "age_model_confidence": None,
        "age_note": "No face detected in blank image.",
        "age_prediction_error": NO_FACE_DETECTED,
        "possible_minor_risk": False,
        "age_verification_required": True,
        "gender_presentation_prediction": NOT_DETECTED,
        "gender_note": NO_FACE_DETECTED,
        "gender_prediction_error": NO_FACE_DETECTED,
        "face_detection_confidence": None,
        "face_box_xyxy": None,
        "face_regions_xyxy": [],
        "age_gender_prediction_performed": False,
        "original_image": img_b64,
        "annotated_image": img_b64,
        "filename": filename,
        "region_note": "No visual information or structure was found in this image.",
        "note": "Please upload an authentic photograph with sufficient lighting, contrast, and visual content."
    }


def _handle_no_human_faces(
    image_bgr: np.ndarray,
    info_check: Dict[str, Any],
    image_type: Dict[str, Any],
    other_faces: List[Dict[str, Any]],
    type_warning: str,
    non_photo: bool,
    filename: Optional[str],
    format_detected: str = DEFAULT_IMAGE_FORMAT
) -> Dict[str, Any]:
    """Generate rejection payload when zero real human faces are detected."""
    height, width = image_bgr.shape[:2]
    other_count = len(other_faces)
    if other_count > 0:
        posting_status = "NON_PHOTO_WARNING"
        types_summary = ", ".join(sorted({f["label"] for f in other_faces}))
        reason = f"{type_warning} No real human face was detected. Found {other_count} other face(s) ({types_summary}).".strip()
        region_note = f"Found {other_count} non-human face(s): {types_summary}. Profile photos must contain a real human face."
        boxes = [f["box_xyxy"] for f in other_faces]
        annotated = annotate_faces(image_bgr, boxes, "NON-HUMAN FACE", COLOR_PURPLE)
    else:
        posting_status = "NON_PHOTO_WARNING" if non_photo else "REJECTED"
        reason = ((type_warning + " ") if type_warning else "") + "No human face was detected."
        region_note = "No face region could be located. The original uploaded photo is shown for review."
        annotated = image_bgr

    img_b64 = bgr_to_base64_jpeg(image_bgr)
    annot_b64 = bgr_to_base64_jpeg(annotated)

    steps = _build_pipeline_steps(
        info_ok=True,
        info_detail=f"Valid visual content (Entropy: {info_check.get('entropy', 0.0):.2f})",
        human_count=0,
        other_count=other_count,
        demographics_done=False,
        demographics_detail=""
    )
    executed = [
        "Image Information & Entropy Analyzer",
        "Resilient Image Screening / CLIP",
        "DeepGHS Anime Face Detector (v1.4n)",
        "Ultralytics YOLOv8n (COCO)",
        "YuNet Face Detector (2023mar)"
    ]
    diagnostics = _build_technical_diagnostics(width, height, format_detected, info_check, executed)

    return {
        "accepted": False,
        "posting_status": posting_status,
        "posting_reason": reason,
        "reason": reason,
        "has_information": True,
        "image_info_check": info_check,
        "image_type_check": image_type,
        "image_type_review_required": non_photo or (other_count > 0),
        "face_detected": other_count > 0,
        "total_face_count": other_count,
        "real_human_face_count": 0,
        "other_face_count": other_count,
        "human_faces": [],
        "other_faces": other_faces,
        "pipeline_steps": steps,
        "technical_diagnostics": diagnostics,
        "processing_time_ms": 0.0,
        "face_count": 0,
        "face_visibility": None,
        "estimated_age_group": NOT_AVAILABLE,
        "age_model_confidence": None,
        "age_note": "No face detected to estimate age.",
        "age_prediction_error": NO_FACE_DETECTED,
        "possible_minor_risk": False,
        "age_verification_required": True,
        "gender_presentation_prediction": NOT_AVAILABLE,
        "gender_note": "No face detected to estimate gender.",
        "gender_prediction_error": NO_FACE_DETECTED,
        "face_detection_confidence": None,
        "face_box_xyxy": None,
        "face_regions_xyxy": [f["box_xyxy"] for f in other_faces],
        "age_gender_prediction_performed": False,
        "original_image": img_b64,
        "annotated_image": annot_b64,
        "filename": filename,
        "region_note": region_note,
        "note": "A clear, well-lit portrait with a single human face is required for profile photos."
    }


def _handle_multiple_human_faces(
    image_bgr: np.ndarray,
    info_check: Dict[str, Any],
    image_type: Dict[str, Any],
    human_faces: List[Dict[str, Any]],
    other_faces: List[Dict[str, Any]],
    type_warning: str,
    non_photo: bool,
    filename: Optional[str],
    format_detected: str = DEFAULT_IMAGE_FORMAT
) -> Dict[str, Any]:
    """Generate rejection payload when multiple real human faces are detected."""
    height, width = image_bgr.shape[:2]
    human_count = len(human_faces)
    total_count = human_count + len(other_faces)
    posting_status = "NON_PHOTO_WARNING" if non_photo else "REJECTED"
    reason = ((type_warning + " ") if type_warning else "") + MULTIPLE_FACES_REASON

    boxes = [f["box_xyxy"] for f in human_faces]
    annotated = annotate_faces(image_bgr, boxes, "MULTIPLE FACES", COLOR_ORANGE)

    steps = _build_pipeline_steps(
        info_ok=True,
        info_detail=f"Valid visual content (Entropy: {info_check.get('entropy', 0.0):.2f})",
        human_count=human_count,
        other_count=len(other_faces),
        demographics_done=False,
        demographics_detail=""
    )
    executed = [
        "Image Information & Entropy Analyzer",
        "Resilient Image Screening / CLIP",
        "DeepGHS Anime Face Detector (v1.4n)",
        "Ultralytics YOLOv8n (COCO)",
        "YuNet Face Detector (2023mar)"
    ]
    if not non_photo:
        executed.append("InsightFace GenderAge (multitask CNN)")
    diagnostics = _build_technical_diagnostics(width, height, format_detected, info_check, executed)

    return {
        "accepted": False,
        "posting_status": posting_status,
        "posting_reason": reason,
        "reason": reason,
        "has_information": True,
        "image_info_check": info_check,
        "image_type_check": image_type,
        "image_type_review_required": non_photo or len(other_faces) > 0,
        "face_detected": True,
        "total_face_count": total_count,
        "real_human_face_count": human_count,
        "other_face_count": len(other_faces),
        "human_faces": human_faces,
        "other_faces": other_faces,
        "pipeline_steps": steps,
        "technical_diagnostics": diagnostics,
        "processing_time_ms": 0.0,
        "face_count": human_count,
        "face_visibility": None,
        "estimated_age_group": NOT_AVAILABLE,
        "age_model_confidence": None,
        "age_note": MULTIPLE_FACES_NOTE,
        "age_prediction_error": MULTIPLE_FACES_NOTE,
        "possible_minor_risk": False,
        "age_verification_required": True,
        "gender_presentation_prediction": NOT_AVAILABLE,
        "gender_note": MULTIPLE_FACES_NOTE,
        "gender_prediction_error": MULTIPLE_FACES_NOTE,
        "face_detection_confidence": human_faces[0]["detection_confidence"] if human_faces else None,
        "face_box_xyxy": boxes[0] if boxes else None,
        "face_regions_xyxy": boxes,
        "age_gender_prediction_performed": False,
        "original_image": bgr_to_base64_jpeg(image_bgr),
        "annotated_image": bgr_to_base64_jpeg(annotated),
        "filename": filename,
        "region_note": f"Found {human_count} faces in image. Profile photos must contain only one face.",
        "note": "A profile photo must contain exactly one person."
    }


def _separate_faces(
    raw_faces: List[Any],
    image_bgr: np.ndarray,
    non_human_entities: List[Dict[str, Any]],
    width: int,
    height: int
) -> Tuple[List[Tuple[Any, List[int]]], List[Dict[str, Any]]]:
    """Separate candidate faces into real human candidates and non-human entities."""
    real_human_raw: List[Tuple[Any, List[int]]] = []
    other_faces_list: List[Dict[str, Any]] = []

    for face in raw_faces:
        raw_box = [int(v) for v in getattr(face, "bbox", [0, 0, width, height])]
        box = list(clamp_box(raw_box, width, height))

        non_human_match = check_is_non_human_face(box, image_bgr, non_human_entities)
        if non_human_match is not None:
            match_copy = dict(non_human_match)
            match_copy["box_normalized"] = compute_normalized_box(box, width, height)
            other_faces_list.append(match_copy)
        else:
            real_human_raw.append((face, box))

    for ent in non_human_entities:
        ent_box = ent["box_xyxy"]
        if not any(f["box_xyxy"] == ent_box for f in other_faces_list):
            ent_copy = dict(ent)
            ent_copy["box_normalized"] = compute_normalized_box(ent_box, width, height)
            other_faces_list.append(ent_copy)

    # Sort real human faces in reading order (top-to-bottom, left-to-right)
    real_human_raw = sorted(
        real_human_raw,
        key=lambda item: (round(item[1][1] / 50), item[1][0])
    )

    # Sort other entities in reading order
    other_faces_list = sorted(
        other_faces_list,
        key=lambda item: (round(item["box_xyxy"][1] / 50), item["box_xyxy"][0])
    )
    for idx, f in enumerate(other_faces_list, start=1):
        f["face_id"] = idx

    return real_human_raw, other_faces_list


def _build_human_face_items(
    real_human_raw: List[Tuple[Any, List[int]]],
    image_bgr: np.ndarray,
    non_photo: bool
) -> List[Dict[str, Any]]:
    """Build detailed analysis items for all detected real human faces."""
    height, width = image_bgr.shape[:2]
    items: List[Dict[str, Any]] = []
    for face_idx, (face_obj, box) in enumerate(real_human_raw, start=1):
        landmarks = getattr(face_obj, "kps", None)
        visibility = assess_face_quality(image_bgr, box, landmarks)
        det_score = round(float(getattr(face_obj, "det_score", 0.95)), 4) if hasattr(face_obj, "det_score") else 0.95

        try:
            if non_photo:
                age_grp, est_age, age_conf = NOT_AVAILABLE, None, None
                gender_str, gender_conf = NOT_AVAILABLE, None
            else:
                age_grp, est_age, age_conf, gender_str, gender_conf = predict_face_demographics(
                    image_bgr, box, face_obj
                )
        except Exception:
            logger.exception("Demographic extraction error")
            age_grp, est_age, age_conf = NOT_AVAILABLE, None, None
            gender_str, gender_conf = NOT_AVAILABLE, None

        minor_risk = possible_minor_age_group(age_grp) if age_grp not in (NOT_AVAILABLE, NOT_DETECTED, "Unsure") else False

        items.append({
            "face_id": face_idx,
            "box_xyxy": box,
            "box_normalized": compute_normalized_box(box, width, height),
            "detection_confidence": det_score,
            "gender": gender_str,
            "gender_confidence": gender_conf,
            "age_group": age_grp,
            "estimated_age": est_age,
            "age_confidence": age_conf,
            "possible_minor_risk": minor_risk,
            "face_visibility": visibility,
            "is_primary": (face_idx == 1)
        })
    return items


def _resolve_attribute_errors(
    age_label: str,
    gender_pred: str,
    non_photo: bool
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve human-readable error messages for demographic estimation."""
    if non_photo:
        age_error = "Skipped because the image appears to be a non-photo image."
    elif age_label in (NOT_AVAILABLE, NOT_DETECTED):
        age_error = "Age model could not estimate the age group."
    else:
        age_error = None

    if non_photo:
        gender_error = "Skipped because the image appears to be a non-photo image."
    elif gender_pred in (NOT_AVAILABLE, NOT_DETECTED):
        gender_error = "Gender prediction was unavailable."
    else:
        gender_error = None

    return age_error, gender_error



def _determine_single_face_decision(
    non_photo: bool,
    other_count: int,
    other_labels: str,
    visibility_status: str,
    visibility_reason: str,
    age_label: str,
    minor_risk: bool,
    type_warning: str
) -> Tuple[bool, str, str]:
    """Determine acceptance and status code for a single human face."""
    if non_photo:
        return (
            False,
            "NON_PHOTO_WARNING",
            f"{type_warning} Face detection found one face. Age and gender estimates on illustrations or computer-generated characters are unreliable."
        )
    if other_count > 0:
        return (
            False,
            "NON_PHOTO_WARNING",
            f"One human face detected, but image also contains {other_count} non-human face(s) ({other_labels})."
        )
    if visibility_status != "LIKELY_COMPLETE":
        return False, "REJECTED_OR_REVIEW", f"{visibility_reason} Please provide a clear, centered, uncropped photo."
    if age_label in (NOT_AVAILABLE, NOT_DETECTED):
        return False, "AGE_ESTIMATION_UNAVAILABLE", "The face was detected, but age group could not be estimated. Please try a clear, front-facing photograph."
    if minor_risk:
        return False, "AGE_VERIFICATION_REQUIRED", f"The estimated age group is {age_label}. This range may include minors under 18. Age verification is required."
    return True, "INITIAL_CHECKS_PASSED", f"All checks passed. Estimated age group is {age_label}."


def _render_annotation(
    image_bgr: np.ndarray,
    face_box: List[int],
    other_faces_list: List[Dict[str, Any]],
    accepted: bool,
    minor_risk: bool,
    posting_status: str,
    age_label: str
) -> str:
    """Render bounding boxes and return Base64-encoded JPEG."""
    box_color = COLOR_GREEN if accepted else COLOR_ORANGE
    if accepted:
        box_label = "FACE DETECTED"
    elif minor_risk:
        box_label = f"AGE GROUP: {age_label}"
    elif posting_status == "NON_PHOTO_WARNING":
        box_label = "WARNING: NON-PHOTO IMAGE"
    else:
        box_label = "REVIEW REQUIRED"

    annotated = annotate_faces(image_bgr, [face_box], box_label, box_color)
    if other_faces_list:
        other_boxes = [f["box_xyxy"] for f in other_faces_list]
        annotated = annotate_faces(annotated, other_boxes, "NON-HUMAN FACE", COLOR_PURPLE)

    return bgr_to_base64_jpeg(annotated)


def analyze_profile_photo_image(
    image: Image.Image,
    filename: Optional[str] = "profile.jpg",
    format_detected: str = DEFAULT_IMAGE_FORMAT
) -> Dict[str, Any]:
    """
    Complete end-to-end profile photo analysis pipeline:
    1. Information & Blank/Solid Image Check
    2. Image Type Screening
    3. Multi-Face & Non-Human Entity Detection (Animal, Bird, Cartoon, Diagram)
    4. Demographic Analysis for All Real Human Faces (3-Year Age Buckets)
    5. Authentic Verification (No Fabrication, 'Unsure' / 'Not Detected' Handling)
    """
    image = image.convert("RGB")
    image_bgr = pil_to_bgr(image)
    height, width = image_bgr.shape[:2]

    # 1. Information / Entropy / Blank check
    info_check = check_image_information(image_bgr)
    image_type = classify_image_type(image)

    if not info_check["has_information"]:
        return _handle_no_info(image_bgr, info_check, image_type, filename, format_detected)

    status = image_type.get("status")
    is_non_photo = (status == "NON_PHOTO")
    is_needs_review = (status == "NEEDS_REVIEW")
    non_photo = is_non_photo or is_needs_review

    if is_non_photo:
        type_warning = (
            f"Warning: this looks like a non-photo image ({image_type.get('display_label', 'Non-photo')}), not a real photograph."
        )
    elif is_needs_review:
        type_warning = (
            f"Warning: image type classification is uncertain ({image_type.get('display_label', 'Needs review')}). Manual verification required."
        )
    else:
        type_warning = ""

    # 2. Multi-Entity and Face Detection
    non_human_entities = detect_non_human_entities(image_bgr)
    raw_faces = detect_faces(image_bgr)
    real_human_raw, other_faces_list = _separate_faces(raw_faces, image_bgr, non_human_entities, width, height)

    real_human_count = len(real_human_raw)
    if real_human_count == 0:
        return _handle_no_human_faces(
            image_bgr, info_check, image_type, other_faces_list, type_warning, non_photo, filename, format_detected
        )

    # 3. Demographics for all real human faces
    human_faces_list = _build_human_face_items(real_human_raw, image_bgr, non_photo)
    if real_human_count > 1:
        return _handle_multiple_human_faces(
            image_bgr, info_check, image_type, human_faces_list, other_faces_list, type_warning, non_photo, filename, format_detected
        )

    # 4. Single real human face evaluation
    primary = human_faces_list[0]
    face_box = primary["box_xyxy"]
    visibility = primary["face_visibility"]
    age_label = primary["age_group"]
    gender_pred = primary["gender"]
    minor_risk = primary["possible_minor_risk"]

    age_error, gender_error = _resolve_attribute_errors(age_label, gender_pred, non_photo)

    other_count = len(other_faces_list)
    other_labels = ", ".join(sorted({f["label"] for f in other_faces_list}))

    accepted, posting_status, posting_reason = _determine_single_face_decision(
        non_photo, other_count, other_labels, visibility["status"], visibility["reason"],
        age_label, minor_risk, type_warning
    )

    annot_b64 = _render_annotation(
        image_bgr, face_box, other_faces_list, accepted, minor_risk, posting_status, age_label
    )

    steps = _build_pipeline_steps(
        info_ok=True,
        info_detail=f"Valid visual content (Entropy: {info_check.get('entropy', 0.0):.2f})",
        human_count=1,
        other_count=other_count,
        demographics_done=True,
        demographics_detail=f"Estimated age group: {age_label}, Gender: {gender_pred}."
    )
    executed = [
        "Image Information & Entropy Analyzer",
        "Resilient Image Screening / CLIP",
        "DeepGHS Anime Face Detector (v1.4n)",
        "Ultralytics YOLOv8n (COCO)",
        "YuNet Face Detector (2023mar)"
    ]
    if not non_photo:
        executed.append("InsightFace GenderAge (multitask CNN)")
    diagnostics = _build_technical_diagnostics(width, height, format_detected, info_check, executed)

    return {
        "accepted": accepted,
        "posting_status": posting_status,
        "posting_reason": posting_reason,
        "reason": posting_reason,
        "has_information": True,
        "image_info_check": info_check,
        "image_type_check": image_type,
        "image_type_review_required": non_photo or other_count > 0,
        "face_detected": True,
        "total_face_count": 1 + other_count,
        "real_human_face_count": 1,
        "other_face_count": other_count,
        "human_faces": human_faces_list,
        "other_faces": other_faces_list,
        "pipeline_steps": steps,
        "technical_diagnostics": diagnostics,
        "processing_time_ms": 0.0,
        "face_count": 1,
        "face_visibility": visibility,
        "estimated_age_group": age_label,
        "age_model_confidence": primary["age_confidence"],
        "age_note": "Appearance-based demographic estimate only; not confirmed proof of legal age." if not age_error else age_error,
        "age_prediction_error": age_error,
        "possible_minor_risk": minor_risk,
        "age_verification_required": minor_risk or not accepted,
        "gender_presentation_prediction": gender_pred,
        "gender_note": "Appearance-based model prediction only; not a confirmed gender identity." if not gender_error else gender_error,
        "gender_prediction_error": gender_error,
        "face_detection_confidence": primary["detection_confidence"],
        "face_box_xyxy": face_box,
        "face_regions_xyxy": [face_box],
        "age_gender_prediction_performed": not non_photo and age_label not in (NOT_AVAILABLE, NOT_DETECTED),
        "original_image": bgr_to_base64_jpeg(image_bgr),
        "annotated_image": annot_b64,
        "filename": filename,
        "region_note": "The outlined area indicates the detected face region.",
        "note": "Age and quality estimates are approximate appearance indicators. Maintain a separate KYC/age-verification workflow for strict compliance."
    }
