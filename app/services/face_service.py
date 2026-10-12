"""Face candidate detection (SCRFD-10G + YuNet) and age/gender estimation (MiVOLO v2) with explicit uncertainty."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import cv2
import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.schemas.analysis import NOT_DETECTED, UNSURE
from app.services.image_utils import box_iou, clamp_box, letterbox_top_left
from app.services.model_manager import MODEL_SPECS, get_session, get_yunet_detector, yunet_lock
from app.services.parallel import run_all

logger = logging.getLogger(__name__)

MODEL_NAME_SCRFD = MODEL_SPECS["scrfd"].display_name
MODEL_NAME_YUNET = MODEL_SPECS["yunet"].display_name
MODEL_NAME_AGE_GENDER = MODEL_SPECS["mivolo"].display_name

# Face detection
MIN_VISIBLE_FRACTION = 0.25  # a detection with less than this share of its box inside the image is dropped
SCRFD_STRIDES = (8, 16, 32)  # SCRFD predicts on three feature maps
SCRFD_MIN_SIZE = 640  # SCRFD input is a square canvas of at least this side...
SCRFD_MAX_SIZE = 1280  # ...and at most this side (a multiple of 32)
SCRFD_NMS_IOU = 0.4  # candidates overlapping more than this are the same face

# MiVOLO v2 age/gender
MIVOLO_SIZE = 384  # face and body crops are each letterboxed to 384 x 384
MIVOLO_AGE_SCALE = 122.0  # the model's raw age output maps to years as raw * 122 + 61
MIVOLO_AGE_OFFSET = 61.0
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class DetectedFace:
    """A face *candidate*; whether it is a real human face is decided later by verification."""

    bbox: list[int]
    det_score: float
    kps: np.ndarray | None = None
    source: str = ""


@dataclass
class Demographics:
    """Age and gender estimate for one face. Fields are None/'Unsure' when the model cannot say."""

    age_group: str = NOT_DETECTED
    estimated_age: float | None = None
    age_range: list[float] | None = None  # 80% interval, years
    age_group_probability: float | None = None  # probability that the true age lies in age_group
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


def _scrfd_input_size(height: int, width: int) -> int:
    longest = max(height, width)
    if longest <= SCRFD_MIN_SIZE:
        return SCRFD_MIN_SIZE
    return min(SCRFD_MAX_SIZE, math.ceil(longest / 32) * 32)


def _decode_scrfd(outputs: list[np.ndarray], size: int, threshold: float) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Decode SCRFD's 9 raw outputs (scores, box distances, landmark offsets for strides 8/16/32, 2 anchors each)."""
    detections: list[tuple[np.ndarray, float, np.ndarray]] = []
    for level, stride in enumerate(SCRFD_STRIDES):
        scores = outputs[level][:, 0]
        distances = outputs[level + 3] * stride
        landmarks = outputs[level + 6] * stride
        cells = size // stride
        xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
        centers = np.stack([xs, ys], axis=-1).astype(np.float32).reshape(-1, 2) * stride
        centers = np.stack([centers] * 2, axis=1).reshape(-1, 2)  # two anchors per cell
        if centers.shape[0] != scores.shape[0]:
            raise ValueError("Unexpected SCRFD output size")
        for i in np.flatnonzero(scores >= threshold):
            cx, cy = centers[i]
            d = distances[i]
            box = np.array([cx - d[0], cy - d[1], cx + d[2], cy + d[3]], dtype=np.float32)
            kps = np.stack([cx + landmarks[i][0::2], cy + landmarks[i][1::2]], axis=-1).astype(np.float32)
            detections.append((box, float(scores[i]), kps))
    return detections


def _scrfd_pass(image_bgr: np.ndarray) -> list[DetectedFace]:
    session = get_session("scrfd")
    if session is None:
        raise RuntimeError("SCRFD detector unavailable")

    height, width = image_bgr.shape[:2]
    size = _scrfd_input_size(height, width)
    canvas, ratio = letterbox_top_left(image_bgr, size)
    blob = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
    outputs = session.run(None, {session.get_inputs()[0].name: blob})

    faces: list[DetectedFace] = []
    for box, score, kps in _decode_scrfd(outputs, size, settings.detection.scrfd_threshold):
        x1, y1, x2, y2 = (float(v) / ratio for v in box)
        clamped = list(clamp_box([round(x1), round(y1), round(x2), round(y2)], width, height))
        visible = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
        if (
            clamped[2] <= clamped[0]
            or clamped[3] <= clamped[1]
            or visible < MIN_VISIBLE_FRACTION * (x2 - x1) * (y2 - y1)
        ):
            continue
        faces.append(DetectedFace(bbox=clamped, det_score=round(score, 4), kps=kps / ratio, source=MODEL_NAME_SCRFD))
    return faces


def _yunet_pass(image_bgr: np.ndarray, scale: float) -> list[DetectedFace]:
    """One YuNet pass on the image resized by ``scale``; results are mapped back to image coordinates."""
    height, width = image_bgr.shape[:2]
    work = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else image_bgr
    work_h, work_w = work.shape[:2]

    with yunet_lock:
        detector = get_yunet_detector(width=work_w, height=work_h)
        if detector is None:
            raise RuntimeError("YuNet detector unavailable")
        rows: np.ndarray | None = detector.detect(work)[1]

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
        faces.append(DetectedFace(bbox=box, det_score=round(float(row[14]), 4), kps=kps, source=MODEL_NAME_YUNET))
    return faces


def _yunet_passes(image_bgr: np.ndarray) -> list[DetectedFace]:
    scales = [1.0]
    if max(image_bgr.shape[:2]) < settings.detection.small_image_upscale_side:
        scales.append(2.0)  # an extra 2x pass finds small faces in small images
    return [face for scale in scales for face in _yunet_pass(image_bgr, scale)]


def _merge_faces(faces: list[DetectedFace]) -> list[DetectedFace]:
    """
    Greedy non-maximum suppression across detectors and passes.

    When both detectors find the same face the YuNet box is kept: on 81 real faces the SigLIP2 verifier accepted
    81/81 YuNet boxes but only 76/81 SCRFD boxes (SCRFD's slightly looser boxes score closer to the threshold).
    SCRFD still adds every face YuNet misses.
    """
    kept: list[DetectedFace] = []
    for face in sorted(faces, key=lambda f: (f.source != MODEL_NAME_YUNET, -f.det_score)):
        if all(box_iou(face.bbox, other.bbox) <= SCRFD_NMS_IOU for other in kept):
            kept.append(face)
    return kept


def _run_detector(
    name: str, run: Callable[[np.ndarray], list[DetectedFace]], image_bgr: np.ndarray
) -> list[DetectedFace] | None:
    """One detector's candidates, or None (logged) when it cannot run."""
    try:
        return run(image_bgr)
    except Exception as exc:
        logger.warning("%s unavailable: %s", name, exc)
        return None


def detect_faces(image_bgr: np.ndarray) -> tuple[list[DetectedFace], list[str]]:
    """
    Detect face candidates with SCRFD-10G and YuNet and merge them (the union raises recall; the verifier
    later rejects non-faces). SCRFD is skipped when ``detection.enable_scrfd`` is off (its weights are
    non-commercial); the two detectors run at the same time when parallel inference is on.

    Returns the candidates and the names of the detectors that really ran (empty when none could run).
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], []

    detectors = [(MODEL_NAME_YUNET, _yunet_passes)]
    if settings.detection.enable_scrfd:
        detectors.insert(0, (MODEL_NAME_SCRFD, _scrfd_pass))
    outcomes = run_all([partial(_run_detector, name, run, image_bgr) for name, run in detectors])

    faces: list[DetectedFace] = []
    ran: list[str] = []
    for (name, _), found in zip(detectors, outcomes, strict=True):
        if found is not None:
            faces.extend(found)
            ran.append(name)
    return _merge_faces(faces), ran


# --------------------------------------------------------------------------------------
# Age / gender (MiVOLO v2)
# --------------------------------------------------------------------------------------


def _crop_rect(image_bgr: np.ndarray, box: list[int] | None, margin: float) -> np.ndarray | None:
    """Rectangular crop of ``box`` enlarged by ``margin`` on every side and clamped to the image."""
    if box is None:
        return None
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if x2 <= x1 or y2 <= y1:
        return None
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(round(x1 - pad_x))), max(0, int(round(y1 - pad_y)))
    right, bottom = min(width, int(round(x2 + pad_x))), min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return image_bgr[top:bottom, left:right]


def _letterbox_normalise(crop_bgr: np.ndarray | None) -> np.ndarray:
    """MiVOLO preprocessing: keep aspect, centre on a black 384 square, RGB, ImageNet-normalise. None -> 'no crop'."""
    canvas = np.zeros((MIVOLO_SIZE, MIVOLO_SIZE, 3), dtype=np.uint8)
    if crop_bgr is not None:
        height, width = crop_bgr.shape[:2]
        ratio = MIVOLO_SIZE / max(height, width)
        new_w, new_h = max(1, round(width * ratio)), max(1, round(height * ratio))
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.cvtColor(cv2.resize(crop_bgr, (new_w, new_h), interpolation=interp), cv2.COLOR_BGR2RGB)
        top, left = (MIVOLO_SIZE - new_h) // 2, (MIVOLO_SIZE - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
    return ((canvas.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)


def _build_views(image_bgr: np.ndarray, face_box: list[int], body_box: list[int] | None) -> list[np.ndarray]:
    """One [1, 6, 384, 384] input per configured (margin, mirrored) view: face channels, then body channels."""
    body = _crop_rect(image_bgr, body_box, 0.0)
    views = []
    for margin, mirrored in settings.age.views:
        face = _crop_rect(image_bgr, face_box, margin)
        if face is None:
            continue
        view_body = body
        if mirrored:
            face = cv2.flip(face, 1)
            view_body = None if body is None else cv2.flip(body, 1)
        stacked = np.concatenate([_letterbox_normalise(face), _letterbox_normalise(view_body)], axis=0)
        views.append(stacked[np.newaxis].astype(np.float32))
    return views


def _estimate_age(ages: np.ndarray, result: Demographics) -> float:
    """Fill the age fields from per-view regressions. Returns the mean age (for gender gating)."""
    mean_age = float(np.clip(np.mean(ages), 0.0, 100.0))
    view_std = float(np.std(ages))
    sigma = math.sqrt((settings.age.model_mae_years * settings.age.sigma_per_mae) ** 2 + view_std**2)
    result.p_under_18 = round(_normal_cdf((settings.age.adult_years - mean_age) / sigma), 4)

    if view_std > settings.age.unsure_view_std:
        result.age_group, result.age_status = UNSURE, UNSURE
        result.notes.append(f"Age views disagree by {view_std:.1f} years, so the age is not reliable.")
    elif not settings.age.min_credible <= mean_age <= settings.age.max_credible:
        result.age_group, result.age_status = UNSURE, UNSURE
        result.notes.append("The estimate lies outside the range the model was trained on.")
    else:
        group = age_to_three_year_group(mean_age)
        low, high = _group_bounds(group)
        result.age_group, result.age_status = group, "Estimated"
        result.estimated_age = round(mean_age, 1)
        result.age_group_probability = round(
            _normal_cdf((high - mean_age) / sigma) - _normal_cdf((low - mean_age) / sigma), 4
        )
        half = settings.age.interval_z * sigma
        result.age_range = [round(max(0.0, mean_age - half), 1), round(min(100.0, mean_age + half), 1)]
    return mean_age


def _estimate_gender(female_probs: np.ndarray, mean_age: float, result: Demographics) -> None:
    mean_female = float(np.mean(female_probs))
    confidence = max(mean_female, 1.0 - mean_female)
    sides = {bool(p >= 0.5) for p in female_probs}
    result.gender_confidence = round(confidence, 4)

    if mean_age < settings.age.gender_min_age:
        result.gender = UNSURE
        result.notes.append("Gender is not estimated for apparent children: it cannot be judged reliably at this age.")
    elif len(sides) > 1:
        result.gender = UNSURE
        result.notes.append("Gender views disagree.")
    elif confidence < settings.age.gender_surety:
        result.gender = UNSURE
    else:
        result.gender = "Female" if mean_female >= 0.5 else "Male"


def _parse_head(head: np.ndarray) -> tuple[float, float]:
    """MiVOLO head [male logit, female logit, raw age] -> (age in years, P(female))."""
    logits = head[:2].astype(np.float64)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return float(head[2]) * MIVOLO_AGE_SCALE + MIVOLO_AGE_OFFSET, float(probs[1])


def _run_view(session: ort.InferenceSession, input_name: str, view: np.ndarray) -> np.ndarray:
    """MiVOLO's [male logit, female logit, raw age] for one view."""
    return np.asarray(session.run(["head"], {input_name: view})[0][0], dtype=np.float32)


def predict_face_demographics(
    image_bgr: np.ndarray,
    face_box: list[int],
    usable: bool = True,
    caveat: str | None = None,
    body_box: list[int] | None = None,
) -> Demographics:
    """
    Estimate age (with an 80% interval and P(age < 18)) and gender for one verified face.

    ``body_box`` (the person box containing the face) gives MiVOLO its second input when the body is visible.
    ``usable=False`` (tiny, very blurred or strongly turned faces) skips inference and reports Unsure rather
    than producing numbers the model cannot support.
    """
    result = Demographics()
    if not usable:
        result.age_group = result.age_status = result.gender = UNSURE
        result.notes.append(caveat or "The face is not suitable for reliable age or gender estimation.")
        return result

    session = get_session("mivolo")
    views = _build_views(image_bgr, face_box, body_box)
    if session is None or not views:
        return result

    try:
        # The exported graph has a fixed batch size of 1, so each view is its own run.
        input_name = session.get_inputs()[0].name
        heads = run_all([partial(_run_view, session, input_name, view) for view in views])
    except Exception:
        logger.exception("Age/gender inference failed")
        return result

    parsed = np.array([_parse_head(head) for head in heads])
    result.model = MODEL_NAME_AGE_GENDER
    mean_age = _estimate_age(parsed[:, 0], result)
    _estimate_gender(parsed[:, 1], mean_age, result)
    return result
