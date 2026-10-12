import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.image_utils import (
    pil_to_bgr,
    bgr_to_base64_jpeg,
    clamp_box,
    annotate_faces,
)
from app.services.classifier_service import classify_image_type
from app.services.quality_service import assess_face_quality
from app.services.face_service import (
    detect_faces,
    extract_face_attributes,
    possible_minor_age_group,
    NOT_AVAILABLE,
)

logger = logging.getLogger(__name__)

COLOR_ORANGE = (0, 140, 255)  # BGR
COLOR_GREEN = (0, 180, 0)     # BGR

MULTIPLE_FACES_REASON = "A profile photo must contain exactly one detected face."
MULTIPLE_FACES_NOTE = "Multiple faces detected."


def _handle_no_faces(
    image_bgr: np.ndarray,
    image_type: Dict[str, Any],
    type_warning: str,
    non_photo: bool,
    filename: Optional[str]
) -> Dict[str, Any]:
    """Generate rejection payload when zero faces are detected."""
    posting_status = "NON_PHOTO_WARNING" if non_photo else "REJECTED"
    reason = ((type_warning + " ") if type_warning else "") + "No human face was detected."
    img_b64 = bgr_to_base64_jpeg(image_bgr)
    return {
        "accepted": False,
        "posting_status": posting_status,
        "posting_reason": reason,
        "reason": reason,
        "image_type_check": image_type,
        "image_type_review_required": non_photo,
        "face_detected": False,
        "face_count": 0,
        "face_visibility": None,
        "estimated_age_group": NOT_AVAILABLE,
        "age_model_confidence": None,
        "age_note": "No face detected to estimate age.",
        "age_prediction_error": "No face detected.",
        "possible_minor_risk": False,
        "age_verification_required": True,
        "gender_presentation_prediction": NOT_AVAILABLE,
        "gender_note": "No face detected to estimate gender.",
        "gender_prediction_error": "No face detected.",
        "face_detection_confidence": None,
        "face_box_xyxy": None,
        "face_regions_xyxy": [],
        "age_gender_prediction_performed": False,
        "original_image": img_b64,
        "annotated_image": img_b64,
        "filename": filename,
        "region_note": "No face region could be located. The original uploaded photo is shown for review.",
        "note": "A clear, well-lit portrait with a single human face is required for profile photos."
    }


def _handle_multiple_faces(
    image_bgr: np.ndarray,
    detected_faces: List[Any],
    image_type: Dict[str, Any],
    type_warning: str,
    non_photo: bool,
    width: int,
    height: int,
    filename: Optional[str]
) -> Dict[str, Any]:
    """Generate rejection payload when multiple faces are detected."""
    face_count = len(detected_faces)
    posting_status = "NON_PHOTO_WARNING" if non_photo else "REJECTED"
    reason = ((type_warning + " ") if type_warning else "") + MULTIPLE_FACES_REASON

    boxes = [
        list(clamp_box([int(v) for v in f.bbox], width, height))
        for f in detected_faces
    ]
    annotated = annotate_faces(image_bgr, boxes, "MULTIPLE FACES", COLOR_ORANGE)

    return {
        "accepted": False,
        "posting_status": posting_status,
        "posting_reason": reason,
        "reason": reason,
        "image_type_check": image_type,
        "image_type_review_required": non_photo,
        "face_detected": True,
        "face_count": face_count,
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
        "face_detection_confidence": round(float(detected_faces[0].det_score), 4) if hasattr(detected_faces[0], "det_score") else None,
        "face_box_xyxy": boxes[0] if boxes else None,
        "face_regions_xyxy": boxes,
        "age_gender_prediction_performed": False,
        "original_image": bgr_to_base64_jpeg(image_bgr),
        "annotated_image": bgr_to_base64_jpeg(annotated),
        "filename": filename,
        "region_note": f"Found {face_count} faces in image. Profile photos must contain only one face.",
        "note": "A profile photo must contain exactly one person."
    }


def _extract_attributes_safely(face: Any, non_photo: bool) -> Tuple[str, Optional[float], Optional[str], str, Optional[str], bool]:
    """Safely extract age and gender attributes when appropriate."""
    if non_photo:
        return (
            NOT_AVAILABLE,
            None,
            "Skipped because the image appears to be a non-photo image.",
            NOT_AVAILABLE,
            "Skipped because the image appears to be a non-photo image.",
            False
        )

    try:
        age_label, age_confidence, gender_prediction = extract_face_attributes(face)
        age_error = "Age model could not estimate the age group." if age_label == NOT_AVAILABLE else None
        gender_error = "Gender prediction was unavailable." if gender_prediction == NOT_AVAILABLE else None
        minor_risk = possible_minor_age_group(age_label) if age_label not in (NOT_AVAILABLE, "Not performed") else False
        return age_label, age_confidence, age_error, gender_prediction, gender_error, minor_risk
    except Exception:
        logger.exception("Error extracting face attributes")
        return NOT_AVAILABLE, None, "Attribute extraction failed.", NOT_AVAILABLE, "Attribute extraction failed.", False


def _determine_posting_decision(
    non_photo: bool,
    type_warning: str,
    visibility_status: str,
    visibility_reason: str,
    age_label: str,
    minor_risk: bool
) -> Tuple[bool, str, str]:
    """Determine acceptance boolean, posting status, and detailed reason string."""
    if non_photo:
        return (
            False,
            "NON_PHOTO_WARNING",
            f"{type_warning} Face detection found one face. Age and gender estimates on illustrations or computer-generated characters are unreliable."
        )
    if visibility_status != "LIKELY_COMPLETE":
        return False, "REJECTED_OR_REVIEW", f"{visibility_reason} Please provide a clear, centered, uncropped photo."
    if age_label == NOT_AVAILABLE:
        return False, "AGE_ESTIMATION_UNAVAILABLE", "The face was detected, but age group could not be estimated. Please try a clear, front-facing photograph."
    if minor_risk:
        return False, "AGE_VERIFICATION_REQUIRED", f"The estimated age group is {age_label}. This range may include minors under 18. Age verification is required."
    return True, "INITIAL_CHECKS_PASSED", f"All checks passed. Estimated age group is {age_label}."


def analyze_profile_photo_image(
    image: Image.Image,
    filename: Optional[str] = "profile.jpg"
) -> Dict[str, Any]:
    """
    Complete end-to-end profile photo analysis pipeline without duplicate inference.
    """
    image = image.convert("RGB")
    image_bgr = pil_to_bgr(image)
    height, width = image_bgr.shape[:2]

    # 1. Screen image type
    image_type = classify_image_type(image)
    non_photo = image_type.get("status") == "NON_PHOTO"
    type_warning = (
        f"Warning: this looks like a non-photo image ({image_type.get('display_label', 'Non-photo')}), not a real photograph."
        if non_photo else ""
    )

    # 2. Detect faces
    detected_faces = detect_faces(image_bgr)
    face_count = len(detected_faces)

    if face_count == 0:
        return _handle_no_faces(image_bgr, image_type, type_warning, non_photo, filename)

    if face_count > 1:
        return _handle_multiple_faces(image_bgr, detected_faces, image_type, type_warning, non_photo, width, height, filename)

    # 3. Single face quality analysis
    face = detected_faces[0]
    raw_bbox = [int(v) for v in getattr(face, "bbox", [0, 0, width, height])]
    face_box = list(clamp_box(raw_bbox, width, height))
    landmarks = getattr(face, "kps", None)
    visibility = assess_face_quality(image_bgr, face_box, landmarks)

    # 4. Demographic attributes
    age_label, age_confidence, age_error, gender_pred, gender_error, minor_risk = _extract_attributes_safely(face, non_photo)

    # 5. Decision logic
    accepted, posting_status, posting_reason = _determine_posting_decision(
        non_photo,
        type_warning,
        visibility["status"],
        visibility["reason"],
        age_label,
        minor_risk
    )

    # 6. Preview annotation
    det_score = round(float(getattr(face, "det_score", 0.95)), 4) if hasattr(face, "det_score") else 0.95
    box_color = COLOR_GREEN if accepted else COLOR_ORANGE

    if accepted:
        box_label = "FACE DETECTED"
    elif minor_risk:
        box_label = f"AGE GROUP: {age_label}"
    else:
        box_label = "REVIEW REQUIRED"

    if posting_status == "NON_PHOTO_WARNING":
        box_label = "WARNING: NON-PHOTO IMAGE"

    annotated = annotate_faces(image_bgr, [face_box], box_label, box_color)

    return {
        "accepted": accepted,
        "posting_status": posting_status,
        "posting_reason": posting_reason,
        "reason": posting_reason,
        "image_type_check": image_type,
        "image_type_review_required": non_photo,
        "face_detected": True,
        "face_count": 1,
        "face_visibility": visibility,
        "estimated_age_group": age_label,
        "age_model_confidence": age_confidence,
        "age_note": "Appearance-based demographic estimate only; not confirmed proof of legal age." if not age_error else age_error,
        "age_prediction_error": age_error,
        "possible_minor_risk": minor_risk,
        "age_verification_required": minor_risk or not accepted,
        "gender_presentation_prediction": gender_pred,
        "gender_note": "Appearance-based model prediction only; not a confirmed gender identity." if not gender_error else gender_error,
        "gender_prediction_error": gender_error,
        "face_detection_confidence": det_score,
        "face_box_xyxy": face_box,
        "face_regions_xyxy": [face_box],
        "age_gender_prediction_performed": not non_photo and age_label != NOT_AVAILABLE,
        "original_image": bgr_to_base64_jpeg(image_bgr),
        "annotated_image": bgr_to_base64_jpeg(annotated),
        "filename": filename,
        "region_note": "The outlined area indicates the detected face region.",
        "note": "Age and quality estimates are approximate appearance indicators. Maintain a separate KYC/age-verification workflow for strict compliance."
    }
