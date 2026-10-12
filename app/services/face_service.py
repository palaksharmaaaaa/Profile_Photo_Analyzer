import logging
import math
import re
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services.image_utils import clamp_box
from app.services.model_manager import get_genderage_session, get_yunet_detector

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "Not available"
NOT_DETECTED = "Not Detected"
UNSURE = "Unsure"

MODEL_NAME_INSIGHTFACE = "InsightFace Face Detector"
MODEL_NAME_YUNET = "OpenCV YuNet Face Detector (ONNX)"
MODEL_NAME_GENDERAGE = "InsightFace GenderAge (multitask CNN)"

_face_app = None
_last_face_detector_model: str | None = None
_last_demographics_model: str | None = None


def get_last_face_detector_model() -> str | None:
    """Return the name of the face detector model that executed inference during the last call."""
    return _last_face_detector_model


def get_last_demographics_model() -> str | None:
    """Return the name of the demographic model that executed inference during the last call."""
    return _last_demographics_model


class DetectedFace:
    """Standardized representation of a detected face."""

    def __init__(
        self,
        bbox: list[int],
        det_score: float = 0.95,
        kps: np.ndarray | None = None,
        age: float | None = None,
        gender: int | None = None,
    ):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = float(det_score)
        self.kps = kps
        self.age = age
        self.gender = gender


def get_face_app():
    """Lazy initialize the InsightFace FaceAnalysis application (fallback/mock compatibility)."""
    global _face_app
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CPUExecutionProvider"] if settings.FORCE_CPU else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            logger.info("Initializing InsightFace FaceAnalysis model...")
            app = FaceAnalysis(name=settings.INSIGHTFACE_MODEL_NAME, providers=providers)
            app.prepare(ctx_id=-1 if settings.FORCE_CPU else 0, det_size=settings.DET_SIZE)
            _face_app = app
        except Exception as exc:
            logger.warning(f"Could not initialize InsightFace FaceAnalysis: {exc}")
            _face_app = False
    return _face_app if _face_app is not False else None


def age_to_three_year_group(age: float) -> str:
    """
    Format continuous age into a 3-year bracket interval:
    0 to 3, 3 to 6, 6 to 9, 9 to 12, 12 to 15, 15 to 18, 18 to 21, and so on.
    """
    if age < 0:
        return "0 to 3"
    lower = math.floor(age / 3.0) * 3
    upper = lower + 3
    return f"{lower} to {upper}"


def sort_faces_reading_order(faces: list[Any], row_threshold_px: int = 50) -> list[Any]:
    """
    Sort detected faces in natural reading order: top-to-bottom, left-to-right.
    Groups faces into vertical rows using row_threshold_px, then sorts horizontally.
    """
    if not faces:
        return []

    def sort_key(face: Any) -> tuple[int, int]:
        bbox = face.bbox if hasattr(face, "bbox") else face.get("bbox", [0, 0, 0, 0])
        y1 = int(bbox[1])
        x1 = int(bbox[0])
        row_idx = round(y1 / max(1, row_threshold_px))
        return (row_idx, x1)

    return sorted(faces, key=sort_key)


def age_to_group(age: float) -> str:
    """Standard demographic age group (alias for 3-year brackets)."""
    return age_to_three_year_group(age)


def possible_minor_age_group(age_label: str) -> bool:
    """Determine if the age label indicates a person under 18."""
    label = age_label.lower().replace(" ", "")

    if any(k in label for k in ("under18", "child", "minor", "infant", "toddler", "teen")):
        return True

    # Format: "XtoY" (e.g., "0to3", "15to18", "18to21")
    match_to = re.search(r"^(\d{1,3})to(\d{1,3})$", label)
    if match_to:
        lower = int(match_to.group(1))
        upper = int(match_to.group(2))
        return lower < 18 or upper <= 18

    # Format: "X-Y" (e.g., "0-2", "3-9", "10-19")
    match_dash = re.search(r"^(\d{1,3})[-–](\d{1,3})$", label)
    if match_dash:
        lower = int(match_dash.group(1))
        upper = int(match_dash.group(2))
        return lower < 18 or upper <= 18

    # Single integer
    single_match = re.search(r"^(\d{1,3})$", label)
    if single_match:
        return int(single_match.group(1)) < 18

    return False


def detect_faces(image_bgr: np.ndarray) -> list[Any]:
    """
    Detect faces in BGR image using lightweight YuNet ONNX model,
    with seamless fallback to InsightFace if configured.
    """
    global _last_face_detector_model
    _last_face_detector_model = None

    if image_bgr is None or image_bgr.size == 0:
        return []

    height, width = image_bgr.shape[:2]

    # Check mock or loaded InsightFace first if set
    face_app = get_face_app()
    if face_app is not None:
        try:
            res = face_app.get(image_bgr)
            _last_face_detector_model = MODEL_NAME_INSIGHTFACE
            return res
        except Exception as exc:
            logger.warning(f"InsightFace detection failed: {exc}")

    # YuNet detector execution
    yunet = get_yunet_detector(width=width, height=height)
    if yunet is None:
        return []

    try:
        _, raw_faces = yunet.detect(image_bgr)
        _last_face_detector_model = MODEL_NAME_YUNET
        if raw_faces is None or len(raw_faces) == 0:
            return []

        detected: list[Any] = []
        for face_row in raw_faces:
            x = round(face_row[0])
            y = round(face_row[1])
            w = round(face_row[2])
            h = round(face_row[3])
            score = float(face_row[14])

            box = list(clamp_box([x, y, x + w, y + h], width, height))
            kps = np.array(
                [
                    [face_row[4], face_row[5]],
                    [face_row[6], face_row[7]],
                    [face_row[8], face_row[9]],
                    [face_row[10], face_row[11]],
                    [face_row[12], face_row[13]],
                ],
                dtype=np.float32,
            )

            detected.append(DetectedFace(bbox=box, det_score=score, kps=kps))
        return detected
    except Exception as exc:
        logger.warning(f"YuNet detection exception: {exc}")
        return []


def extract_face_attributes(face: Any) -> tuple[str, float | None, str]:
    """
    Extract estimated age group, confidence, and gender presentation
    for backwards compatibility with existing pipelines.
    """
    age_val = getattr(face, "age", None)
    if age_val is not None:
        estimated_age_group = age_to_three_year_group(float(age_val))
        age_confidence = getattr(face, "age_confidence", None)
    else:
        estimated_age_group = NOT_AVAILABLE
        age_confidence = None

    gender_val = getattr(face, "gender", None)
    if gender_val is not None and int(gender_val) in (0, 1):
        gender_prediction = "female-presentation prediction" if int(gender_val) == 0 else "male-presentation prediction"
    else:
        gender_prediction = NOT_AVAILABLE

    return estimated_age_group, age_confidence, gender_prediction


def _crop_face_with_margin(image_bgr: np.ndarray, face_box: list[int]) -> np.ndarray | None:
    """Validate face bounding box and return cropped region with 15% margin."""
    height, width = image_bgr.shape[:2]
    rx1, ry1, rx2, ry2 = [int(v) for v in face_box[:4]]
    if rx1 >= width or ry1 >= height or rx2 <= 0 or ry2 <= 0 or rx2 <= rx1 or ry2 <= ry1:
        return None

    x1, y1, x2, y2 = clamp_box(face_box, width, height)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * 0.15), int(bh * 0.15)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(width, x2 + mx)
    cy2 = min(height, y2 + my)
    return image_bgr[cy1:cy2, cx1:cx2]


def _parse_gender_prediction(gender_logits: np.ndarray) -> tuple[str, float]:
    """Parse softmax probabilities for gender classification."""
    exp_g = np.exp(gender_logits - np.max(gender_logits))
    probs = exp_g / np.sum(exp_g)
    max_p = float(np.max(probs))

    if max_p < settings.GENDER_SURETY_THRESHOLD:
        return UNSURE, round(max_p, 3)
    if probs[0] > probs[1]:
        return "female-presentation prediction", round(float(probs[0]), 3)
    return "male-presentation prediction", round(float(probs[1]), 3)


def _parse_age_prediction(raw_age_normalized: float) -> tuple[str, float | None, float | None]:
    """
    Parse continuous normalized age and compute 3-year bucket.
    InsightFace GenderAge model performs scalar continuous regression without class probabilities.
    To avoid fabricating confidence by borrowing detector scores, age confidence is set to None.
    """
    raw_age = float(raw_age_normalized * 100)
    if raw_age < 0 or raw_age > 120:
        return UNSURE, None, None
    age_grp = age_to_three_year_group(raw_age)
    est_age = round(raw_age, 1)
    return age_grp, est_age, None


def predict_face_demographics(
    image_bgr: np.ndarray, face_box: list[int], face_obj: Any | None = None
) -> tuple[str, float | None, float | None, str, float | None]:
    """
    Predict age group, continuous age, age confidence, gender presentation,
    and gender confidence for a single face using GenderAge ONNX model.
    """
    global _last_demographics_model
    _last_demographics_model = None

    # 1. Check face_obj via extract_face_attributes if provided with pre-computed attributes
    if face_obj is not None:
        age_grp, age_conf, gender_str = extract_face_attributes(face_obj)
        if age_grp != NOT_AVAILABLE and gender_str != NOT_AVAILABLE:
            age_val = float(getattr(face_obj, "age", 0.0))
            return age_grp, round(age_val, 1), age_conf, gender_str, age_conf

    # 2. Run GenderAge ONNX session on face crop
    crop = _crop_face_with_margin(image_bgr, face_box)
    if crop is None:
        return NOT_DETECTED, None, None, NOT_DETECTED, None

    session = get_genderage_session()
    if session is None:
        return NOT_DETECTED, None, None, NOT_DETECTED, None

    try:
        resized = cv2.resize(crop, (96, 96))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis, :, :, :]
        inp_name = session.get_inputs()[0].name
        out = session.run(None, {inp_name: blob})[0]
        _last_demographics_model = MODEL_NAME_GENDERAGE

        gender_str, gender_conf = _parse_gender_prediction(out[0][:2])
        age_grp, est_age, age_conf = _parse_age_prediction(out[0][2])

        return age_grp, est_age, age_conf, gender_str, gender_conf
    except Exception as exc:
        logger.warning(f"GenderAge prediction error: {exc}")
        return NOT_DETECTED, None, None, NOT_DETECTED, None
