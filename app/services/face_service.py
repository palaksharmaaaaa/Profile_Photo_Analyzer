"""Face candidate detection (YuNet) and age/gender estimation with explicit uncertainty."""

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings
from app.services.image_utils import box_iou, clamp_box, crop_square
from app.services.model_manager import MODEL_SPECS, get_session, get_yunet_detector, yunet_lock

logger = logging.getLogger(__name__)

NOT_AVAILABLE = "Not available"
NOT_DETECTED = "Not Detected"
UNSURE = "Unsure"

MODEL_NAME_YUNET = MODEL_SPECS["yunet"].display_name
MODEL_NAME_AGE_GENDER = MODEL_SPECS["age_gender"].display_name

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_AGE_INPUT = 224
MIN_VISIBLE_FRACTION = 0.25


@dataclass
class DetectedFace:
    """A face *candidate*; whether it is a real human face is decided later by verification."""

    bbox: list[int]
    det_score: float
    kps: np.ndarray | None = None


@dataclass
class Demographics:
    """Age and gender estimate for one face. Fields are None/'Unsure' when the model cannot say."""

    age_group: str = NOT_DETECTED
    estimated_age: float | None = None
    age_range: list[float] | None = None  # 80% interval, years
    age_confidence: float | None = None  # probability that the true age lies in age_group
    p_under_18: float | None = None
    age_status: str = NOT_DETECTED
    gender: str = NOT_DETECTED
    gender_confidence: float | None = None
    model: str | None = None
    notes: list[str] = field(default_factory=list)


def age_to_three_year_group(age: float) -> str:
    """Format an age as a 3-year bracket: '0 to 3', '3 to 6', ..., '18 to 21', and so on."""
    lower = math.floor(max(age, 0.0) / 3.0) * 3
    return f"{lower} to {lower + 3}"


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _group_bounds(group: str) -> tuple[float, float]:
    low, high = group.split(" to ")
    return float(low), float(high)


# --------------------------------------------------------------------------------------
# Face candidate detection
# --------------------------------------------------------------------------------------


def _yunet_pass(image_bgr: np.ndarray, scale: float) -> list[DetectedFace]:
    """One YuNet pass on the image resized by ``scale``; results are mapped back to image coordinates."""
    height, width = image_bgr.shape[:2]
    work = image_bgr if scale == 1.0 else cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    work_h, work_w = work.shape[:2]

    with yunet_lock:
        detector = get_yunet_detector(width=work_w, height=work_h)
        if detector is None:
            raise RuntimeError("YuNet detector unavailable")
        _, rows = detector.detect(work)

    faces: list[DetectedFace] = []
    if rows is None:
        return faces
    for row in rows:
        x, y, w, h = (float(v) / scale for v in row[:4])
        box = list(clamp_box([round(x), round(y), round(x + w), round(y + h)], width, height))
        visible_area = (box[2] - box[0]) * (box[3] - box[1])
        if box[2] <= box[0] or box[3] <= box[1] or visible_area < MIN_VISIBLE_FRACTION * w * h:
            continue  # off-image or mostly outside the frame: clamping would leave a meaningless sliver
        kps = np.array(row[4:14], dtype=np.float32).reshape(5, 2) / scale
        faces.append(DetectedFace(bbox=box, det_score=round(float(row[14]), 4), kps=kps))
    return faces


def _merge_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
    """Greedy non-maximum suppression across detection passes."""
    kept: list[DetectedFace] = []
    for face in sorted(faces, key=lambda f: f.det_score, reverse=True):
        if all(box_iou(face.bbox, other.bbox) <= settings.FACE_NMS_IOU for other in kept):
            kept.append(face)
    return kept


def detect_faces(image_bgr: np.ndarray) -> tuple[list[DetectedFace], str | None]:
    """
    Detect face candidates with YuNet (a second 2x pass is added for small images so small faces are found).

    Returns the candidates and the detector name, or ``None`` as the name when detection could not run.
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], None

    scales = [1.0]
    if max(image_bgr.shape[:2]) < settings.SMALL_IMAGE_UPSCALE_SIDE:
        scales.append(2.0)

    try:
        faces = [face for scale in scales for face in _yunet_pass(image_bgr, scale)]
    except Exception as exc:
        logger.warning("YuNet detection unavailable: %s", exc)
        return [], None
    return _merge_faces(faces), MODEL_NAME_YUNET


# --------------------------------------------------------------------------------------
# Age / gender
# --------------------------------------------------------------------------------------


def _prepare_view(crop_bgr: np.ndarray) -> np.ndarray:
    interp = cv2.INTER_AREA if max(crop_bgr.shape[:2]) > _AGE_INPUT else cv2.INTER_LINEAR
    resized = cv2.resize(crop_bgr, (_AGE_INPUT, _AGE_INPUT), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)


def _build_views(image_bgr: np.ndarray, face_box: list[int]) -> np.ndarray | None:
    """Every configured crop margin, each also mirrored; disagreement between views measures uncertainty."""
    views = []
    for margin in settings.AGE_CROP_MARGINS:
        crop = crop_square(image_bgr, face_box, margin)
        if crop is None or crop.size == 0:
            continue
        views.append(_prepare_view(crop))
        views.append(_prepare_view(cv2.flip(crop, 1)))
    return np.stack(views).astype(np.float32) if views else None


def _estimate_age(ages: np.ndarray, result: Demographics) -> float:
    """Fill the age fields from per-view regressions. Returns the mean age (for gender gating)."""
    mean_age = float(np.clip(np.mean(ages), 0.0, 100.0))
    view_std = float(np.std(ages))
    sigma = math.sqrt((settings.AGE_MODEL_MAE_YEARS * settings.AGE_SIGMA_PER_MAE) ** 2 + view_std**2)
    result.p_under_18 = round(_normal_cdf((settings.ADULT_AGE_YEARS - mean_age) / sigma), 4)

    if view_std > settings.AGE_UNSURE_VIEW_STD:
        result.age_group, result.age_status = UNSURE, UNSURE
        result.notes.append(f"Age views disagree by {view_std:.1f} years, so the age is not reliable.")
    elif not settings.AGE_MIN_CREDIBLE <= mean_age <= settings.AGE_MAX_CREDIBLE:
        result.age_group, result.age_status = UNSURE, UNSURE
        result.notes.append("The estimate lies outside the range the model was trained on.")
    else:
        group = age_to_three_year_group(mean_age)
        low, high = _group_bounds(group)
        result.age_group, result.age_status = group, "Estimated"
        result.estimated_age = round(mean_age, 1)
        result.age_confidence = round(_normal_cdf((high - mean_age) / sigma) - _normal_cdf((low - mean_age) / sigma), 4)
        half = settings.AGE_INTERVAL_Z * sigma
        result.age_range = [round(max(0.0, mean_age - half), 1), round(min(100.0, mean_age + half), 1)]
    return mean_age


def _estimate_gender(female_probs: np.ndarray, mean_age: float, result: Demographics) -> None:
    mean_female = float(np.mean(female_probs))
    confidence = max(mean_female, 1.0 - mean_female)
    sides = {bool(p >= 0.5) for p in female_probs}
    result.gender_confidence = round(confidence, 4)

    if mean_age < settings.GENDER_MIN_AGE_YEARS:
        result.gender = UNSURE
        result.notes.append("Gender is not estimated for apparent children: the model is unreliable below this age.")
    elif len(sides) > 1:
        result.gender = UNSURE
        result.notes.append("Gender views disagree.")
    elif confidence < settings.GENDER_SURETY_THRESHOLD:
        result.gender = UNSURE
    else:
        result.gender = "Female" if mean_female >= 0.5 else "Male"


def predict_face_demographics(
    image_bgr: np.ndarray,
    face_box: list[int],
    usable: bool = True,
    caveat: str | None = None,
) -> Demographics:
    """
    Estimate age (with an 80% interval and P(age < 18)) and gender for one verified face.

    ``usable=False`` (tiny, very blurry or strongly turned faces) skips inference and reports Unsure
    rather than producing numbers the model cannot support.
    """
    result = Demographics()
    if not usable:
        result.age_group = result.age_status = result.gender = UNSURE
        result.notes.append(caveat or "The face is not suitable for reliable age or gender estimation.")
        return result

    session = get_session("age_gender")
    views = _build_views(image_bgr, face_box)
    if session is None or views is None:
        return result

    try:
        output = session.run(None, {session.get_inputs()[0].name: views})[0]
    except Exception:
        logger.exception("Age/gender inference failed")
        return result

    result.model = MODEL_NAME_AGE_GENDER
    mean_age = _estimate_age(output[:, 0], result)
    _estimate_gender(output[:, 1], mean_age, result)
    return result
