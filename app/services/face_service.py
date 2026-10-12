import logging
import re
from typing import List, Any, Optional, Tuple
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "Not available"

_face_app = None


def get_face_app():
    """Lazy initialize the InsightFace FaceAnalysis application."""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
            providers = ["CPUExecutionProvider"] if settings.FORCE_CPU else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("Initializing InsightFace FaceAnalysis model...")
            app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                providers=providers
            )
            app.prepare(ctx_id=-1 if settings.FORCE_CPU else 0, det_size=settings.DET_SIZE)
            _face_app = app
        except Exception as exc:
            logger.warning(f"Could not initialize InsightFace FaceAnalysis: {exc}")
            _face_app = False
    return _face_app if _face_app is not False else None


def age_to_group(age: float) -> str:
    """Format continuous age into a standard demographic age bracket."""
    age_int = int(round(age))
    if age_int <= 2:
        return "0-2"
    if age_int <= 9:
        return "3-9"
    if age_int <= 19:
        return "10-19"
    if age_int <= 29:
        return "20-29"
    if age_int <= 39:
        return "30-39"
    if age_int <= 49:
        return "40-49"
    if age_int <= 59:
        return "50-59"
    if age_int <= 69:
        return "60-69"
    return "70+"


def possible_minor_age_group(age_label: str) -> bool:
    """Determine if the age label indicates a user under 18."""
    label = age_label.lower().replace(" ", "")

    if label in {"0-2", "3-9", "10-19"}:
        return True

    if any(k in label for k in ("under18", "child", "minor", "infant", "toddler", "teen")):
        return True

    # Linear complexity regex with spaces already removed
    match = re.search(r"^(\d{1,3})[-–](\d{1,3})$", label)
    if match:
        lower = int(match.group(1))
        return lower < 18

    # Single digit or explicit number check
    single_match = re.search(r"^(\d{1,3})$", label)
    if single_match:
        return int(single_match.group(1)) < 18

    return False


def detect_faces(image_bgr: np.ndarray) -> List[Any]:
    """Detect faces in BGR image using InsightFace with fallback support."""
    face_app = get_face_app()
    if face_app is None:
        logger.debug("InsightFace not loaded; returning empty face list.")
        return []
    return face_app.get(image_bgr)


def extract_face_attributes(face: Any) -> Tuple[str, Optional[float], str]:
    """
    Extract estimated age group, confidence, and gender presentation
    directly from InsightFace's built-in multitask heads.
    """
    # 1. Age extraction
    age_val = getattr(face, "age", None)
    if age_val is not None:
        estimated_age_group = age_to_group(float(age_val))
        age_confidence = 0.85
    else:
        estimated_age_group = NOT_AVAILABLE
        age_confidence = None

    # 2. Gender presentation extraction (0 = female, 1 = male)
    gender_val = getattr(face, "gender", None)
    if gender_val is not None and int(gender_val) in (0, 1):
        gender_prediction = (
            "female-presentation prediction"
            if int(gender_val) == 0
            else "male-presentation prediction"
        )
    else:
        gender_prediction = NOT_AVAILABLE

    return estimated_age_group, age_confidence, gender_prediction
