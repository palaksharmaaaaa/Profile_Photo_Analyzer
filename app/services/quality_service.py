import cv2
import numpy as np

from app.core.config import settings
from app.services.image_utils import clamp_box
from app.services.types import Box, ImageInfoItem, QualityItem

INFO_ANALYSIS_MAX_SIDE = 512  # the information check runs on a copy no larger than this
SHARPNESS_REFERENCE_SIZE = 112  # faces are resized to this before measuring sharpness (resolution independent)
DEMOGRAPHICS_MAX_YAW_RATIO = 0.60  # beyond this head turn, age and gender are not estimated at all


def check_image_information(image_bgr: np.ndarray) -> ImageInfoItem:
    """Decide whether the image carries visual information or is blank / a solid or uniform fill."""
    if image_bgr is None or image_bgr.size == 0:
        return {
            "has_information": False,
            "entropy": 0.0,
            "std_dev": 0.0,
            "laplacian_var": 0.0,
            "reason": "Image has no data or dimensions.",
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    longest = max(gray.shape[:2])
    if longest > INFO_ANALYSIS_MAX_SIDE:
        scale = INFO_ANALYSIS_MAX_SIDE / longest
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    std_val = float(np.std(gray))
    counts = np.bincount(gray.ravel().astype(np.int64), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) else 0.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    unique_vals = int(np.count_nonzero(counts))

    is_blank = (
        std_val < settings.information.min_std_dev
        or entropy < settings.information.min_entropy
        or lap_var < settings.information.min_laplacian_var
        or unique_vals < 5
    )
    reason = (
        "This image contains no visual information to check or process (it is blank, solid color, or uniform fill)."
        if is_blank
        else "Image contains valid visual entropy and information."
    )
    return {
        "has_information": not is_blank,
        "entropy": round(entropy, 3),
        "std_dev": round(std_val, 3),
        "laplacian_var": round(lap_var, 3),
        "reason": reason,
    }


def _check_edge_touch(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> bool:
    margin_x = max(settings.quality.min_edge_margin_pixels, int(width * settings.quality.edge_margin_ratio))
    margin_y = max(settings.quality.min_edge_margin_pixels, int(height * settings.quality.edge_margin_ratio))
    return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y


def _check_landmarks(landmarks: np.ndarray | None, width: int, height: int) -> tuple[bool, bool]:
    if landmarks is None:
        return False, False
    points = np.asarray(landmarks)
    if points.ndim != 2 or points.shape[1] < 2:
        return True, False
    inside = bool(
        np.all(points[:, 0] >= 0)
        and np.all(points[:, 0] < width)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 1] < height)
    )
    return True, inside


def estimate_yaw_ratio(landmarks: np.ndarray | None) -> float | None:
    """
    Head-turn proxy from the 5 YuNet landmarks (eye, eye, nose, mouth, mouth).

    The nose tip's horizontal offset from the eye midpoint, relative to the inter-ocular distance:
    about 0 for a frontal face and growing in magnitude as the head turns.
    """
    if landmarks is None:
        return None
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    eye_distance = abs(points[1, 0] - points[0, 0])
    if eye_distance < 1e-6:
        return None
    return float((points[2, 0] - (points[0, 0] + points[1, 0]) / 2.0) / eye_distance)


def _resolve_quality_verdict(
    touches_edge: bool,
    brightness: float,
    face_area_ratio: float,
    sharpness: float,
    landmarks_available: bool,
    landmarks_inside: bool,
    yaw_ratio: float | None,
) -> tuple[str, str]:
    """Determine status and reasoning based on prioritised quality metrics."""
    if touches_edge:
        return "PARTIAL_OR_CROPPED", "The face box is too close to an image edge."
    if brightness < settings.quality.min_brightness:
        return "UNCERTAIN", "The face appears significantly underexposed or in poor lighting."
    if brightness > settings.quality.max_brightness:
        return "UNCERTAIN", "The face appears overexposed or washed out."
    if face_area_ratio < settings.quality.min_face_area_ratio:
        return "UNCERTAIN", "The face is too small in the frame for a profile photo."
    if face_area_ratio > settings.quality.max_face_area_ratio:
        return "UNCERTAIN", "The face fills almost the whole frame; the head may be cropped."
    if sharpness < settings.quality.min_sharpness:
        return (
            "UNCERTAIN",
            f"The face image may be blurry (sharpness score {sharpness:.1f} < {settings.quality.min_sharpness:.1f}).",
        )
    if not landmarks_available or not landmarks_inside:
        return "UNCERTAIN", "Facial landmarks could not be verified reliably."
    if yaw_ratio is not None and abs(yaw_ratio) > settings.quality.max_yaw_ratio:
        return "UNCERTAIN", "The head is turned away from the camera; a frontal photo is expected."
    return "LIKELY_COMPLETE", "The face is well-framed, sharp, frontal, and all key facial landmarks are visible."


def assess_face_quality(image_bgr: np.ndarray, face_box: Box, landmarks: np.ndarray | None = None) -> QualityItem:
    """Evaluate framing, exposure, sharpness (resolution-normalised), pose and landmarks for one face."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(face_box, width, height)
    box_width, box_height = max(0, x2 - x1), max(0, y2 - y1)

    if box_width == 0 or box_height == 0:
        return {
            "status": "UNCERTAIN",
            "reason": "Invalid face bounding box.",
            "touches_image_edge": True,
            "sharpness": 0.0,
            "landmarks_available": False,
            "landmarks_inside_image": False,
            "brightness": 0.0,
            "face_area_ratio": 0.0,
            "yaw_ratio": None,
            "min_side_px": 0,
            "usable_for_demographics": False,
            "demographics_caveat": "Invalid face bounding box.",
        }

    touches_edge = _check_edge_touch(x1, y1, x2, y2, width, height)
    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    reference = cv2.resize(
        gray,
        (SHARPNESS_REFERENCE_SIZE, SHARPNESS_REFERENCE_SIZE),
        interpolation=cv2.INTER_AREA if max(gray.shape) > SHARPNESS_REFERENCE_SIZE else cv2.INTER_LINEAR,
    )
    sharpness = float(cv2.Laplacian(reference, cv2.CV_64F).var())
    face_area_ratio = float(box_width * box_height / (width * height))
    landmarks_available, landmarks_inside = _check_landmarks(landmarks, width, height)
    yaw_ratio = estimate_yaw_ratio(landmarks)
    min_side = min(box_width, box_height)

    status, reason = _resolve_quality_verdict(
        touches_edge, brightness, face_area_ratio, sharpness, landmarks_available, landmarks_inside, yaw_ratio
    )

    caveat = None
    if min_side < settings.quality.min_face_side_px:
        caveat = f"The face is only {min_side}px wide; at least {settings.quality.min_face_side_px}px is needed."
    elif yaw_ratio is not None and abs(yaw_ratio) > DEMOGRAPHICS_MAX_YAW_RATIO:
        caveat = "The head is turned too far from the camera for reliable age or gender estimation."

    return {
        "status": status,
        "reason": reason,
        "touches_image_edge": touches_edge,
        "sharpness": round(sharpness, 2),
        "landmarks_available": landmarks_available,
        "landmarks_inside_image": landmarks_inside,
        "brightness": round(brightness, 2),
        "face_area_ratio": round(face_area_ratio, 4),
        "yaw_ratio": None if yaw_ratio is None else round(yaw_ratio, 3),
        "min_side_px": min_side,
        "usable_for_demographics": caveat is None,
        "demographics_caveat": caveat,
    }
