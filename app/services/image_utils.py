import base64
import io
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_image_bytes(contents: bytes) -> Image.Image:
    """Decode raw bytes into a PIL RGB image."""
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image format: {exc}") from exc


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR numpy array."""
    image_rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_base64_jpeg(image_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image as a JPEG data URL."""
    success, buffer = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise ValueError("Could not encode image to JPEG format.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


def clamp_box(box: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp box coordinates safely within image bounds."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def annotate_faces(
    image_bgr: np.ndarray,
    boxes: List[List[int]],
    label: str,
    box_color: Tuple[int, int, int]
) -> np.ndarray:
    """Draw bounding boxes and formatted tags on image."""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = clamp_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(width, height) / 1000.0)
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        label_y = min(height - baseline - 2, max(text_height + 10, y1))
        rect_top = max(0, label_y - text_height - 8)
        rect_bottom = min(height - 1, label_y + baseline)
        rect_right = min(width - 1, x1 + text_width + 8)

        cv2.rectangle(
            annotated,
            (x1, rect_top),
            (rect_right, rect_bottom),
            box_color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return annotated
