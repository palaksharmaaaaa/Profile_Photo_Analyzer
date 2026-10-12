from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services.image_utils import clamp_box


def check_image_information(image_bgr: np.ndarray) -> dict[str, Any]:
    """
    Check if the image contains any real visual information vs blank,
    solid color, or uniform tone with zero entropy/structure.
    """
    if image_bgr is None or image_bgr.size == 0:
        return {
            "has_information": False,
            "entropy": 0.0,
            "std_dev": 0.0,
            "laplacian_var": 0.0,
            "reason": "Image has no data or dimensions.",
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std_val = float(np.std(gray))

    # Shannon Entropy calculation
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

    # Edge / Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    unique_vals = len(np.unique(gray))

    is_solid = (
        std_val < settings.MIN_IMAGE_STD_DEV
        or entropy < settings.MIN_IMAGE_ENTROPY
        or lap_var < settings.MIN_LAPLACIAN_VAR
        or unique_vals < 5
    )

    if is_solid:
        reason = (
            "This image contains no visual information to check or process (it is blank, solid color, or uniform fill)."
        )
    else:
        reason = "Image contains valid visual entropy and information."

    return {
        "has_information": not is_solid,
        "entropy": round(entropy, 3),
        "std_dev": round(std_val, 3),
        "laplacian_var": round(lap_var, 3),
        "reason": reason,
    }


def _check_edge_touch(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> bool:
    """Check if bounding box contacts image boundaries."""
    margin_x = max(settings.MIN_EDGE_MARGIN_PIXELS, int(width * settings.EDGE_MARGIN_RATIO))
    margin_y = max(settings.MIN_EDGE_MARGIN_PIXELS, int(height * settings.EDGE_MARGIN_RATIO))
    return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y


def _check_landmarks(landmarks: np.ndarray | None, width: int, height: int) -> tuple[bool, bool]:
    """Verify facial keypoints presence and boundaries."""
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


def _resolve_quality_verdict(
    touches_edge: bool,
    brightness: float,
    face_area_ratio: float,
    sharpness: float,
    landmarks_available: bool,
    landmarks_inside: bool,
) -> tuple[str, str]:
    """Determine status and reasoning based on prioritized quality metrics."""
    if touches_edge:
        return "PARTIAL_OR_CROPPED", "The face box is too close to an image edge."
    if brightness < settings.MIN_BRIGHTNESS:
        return (
            "UNCERTAIN",
            "The face appears significantly underexposed or in poor lighting.",
        )
    if brightness > settings.MAX_BRIGHTNESS:
        return "UNCERTAIN", "The face appears overexposed or washed out."
    if face_area_ratio < settings.MIN_FACE_AREA_RATIO:
        return "UNCERTAIN", "The face is too small in the frame for a profile photo."
    if sharpness < settings.MIN_SHARPNESS_SCORE:
        return (
            "UNCERTAIN",
            f"The face image may be blurry (sharpness score {sharpness:.1f} < {settings.MIN_SHARPNESS_SCORE:.1f}).",
        )
    if not landmarks_available or not landmarks_inside:
        return "UNCERTAIN", "Facial landmarks could not be verified reliably."
    return (
        "LIKELY_COMPLETE",
        "The face is well-framed, sharp, and all key facial landmarks are visible.",
    )


def assess_face_quality(image_bgr: np.ndarray, face_box: list, landmarks: np.ndarray | None = None) -> dict[str, Any]:
    """
    Evaluate face visibility, sharpness, exposure, framing, and landmarks.
    """
    height, width = image_bgr.shape[:2]
    total_area = width * height

    x1, y1, x2, y2 = clamp_box(face_box, width, height)
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    if box_width == 0 or box_height == 0:
        return {
            "status": "UNCERTAIN",
            "reason": "Invalid face bounding box.",
            "face_box_touches_image_edge": True,
            "sharpness_score": 0.0,
            "landmarks_available": False,
            "landmarks_inside_image": False,
            "brightness_score": 0.0,
            "face_area_ratio": 0.0,
        }

    touches_edge = _check_edge_touch(x1, y1, x2, y2, width, height)

    face_crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    face_area = box_width * box_height
    face_area_ratio = float(face_area / total_area) if total_area > 0 else 0.0

    landmarks_available, landmarks_inside = _check_landmarks(landmarks, width, height)

    status, reason = _resolve_quality_verdict(
        touches_edge,
        brightness,
        face_area_ratio,
        sharpness,
        landmarks_available,
        landmarks_inside,
    )

    return {
        "status": status,
        "reason": reason,
        "face_box_touches_image_edge": touches_edge,
        "sharpness_score": round(sharpness, 2),
        "landmarks_available": landmarks_available,
        "landmarks_inside_image": landmarks_inside,
        "brightness_score": round(brightness, 2),
        "face_area_ratio": round(face_area_ratio, 4),
    }
