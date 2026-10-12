import base64
import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.core.config import settings

BoxItem = tuple[list[int], str, tuple[int, int, int]]


class InvalidImageError(ValueError):
    """Raised when an upload is not a decodable, in-limits image. ``reason`` is safe to show to users."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class DecodedImage:
    """A decoded working image together with facts about the original upload."""

    image: Image.Image
    file_format: str
    original_width: int
    original_height: int
    exif_orientation_applied: bool
    downscaled: bool


def sniff_image_magic_bytes(contents: bytes) -> str:
    """Identify the real container format from magic bytes (never from the file extension)."""
    if not contents or len(contents) < 4:
        raise ValueError("File payload is empty or too short to identify.")

    if contents.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(contents) >= 12 and contents.startswith(b"RIFF") and contents[8:12] == b"WEBP":
        return "image/webp"
    if contents.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if contents.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if contents.startswith(b"BM"):
        return "image/bmp"

    raise ValueError(
        "Invalid or unsupported binary format. Only authentic JPEG, PNG, WebP, GIF, TIFF and BMP images are supported."
    )


def _flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparent pixels over white instead of exposing arbitrary hidden RGB values."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def decode_image_bytes(contents: bytes) -> DecodedImage:
    """
    Validate and decode an upload into an upright RGB working image.

    Enforces the magic-byte format, pixel/dimension limits, honours the EXIF orientation tag,
    flattens transparency, uses the first frame of animations and caps the working size.
    """
    try:
        file_format = sniff_image_magic_bytes(contents)
    except ValueError as exc:
        raise InvalidImageError(str(exc)) from exc

    # Pillow itself enforces the pixel limit while opening: a warning above it (turned into an error
    # below) and DecompressionBombError at twice the limit.
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(contents))
            width, height = image.size
            if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
                raise InvalidImageError(f"Image dimension exceeds {settings.MAX_IMAGE_DIMENSION}px limit.")
            if image.format in ("JPEG", "MPO"):
                # Let the JPEG decoder produce a reduced-size image directly, saving memory.
                image.draft("RGB", (settings.WORKING_MAX_SIDE * 2, settings.WORKING_MAX_SIDE * 2))
            image.seek(0)
            image.load()
    except InvalidImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InvalidImageError("Image pixel count exceeds the maximum permitted decompression limit.") from exc
    except (OSError, ValueError, SyntaxError, EOFError) as exc:
        raise InvalidImageError("The uploaded file is not a valid supported image.") from exc

    orientation = image.getexif().get(0x0112, 1)
    upright = ImageOps.exif_transpose(image)
    rgb = _flatten_alpha(upright)

    downscaled = False
    longest = max(rgb.size)
    if longest > settings.WORKING_MAX_SIDE:
        scale = settings.WORKING_MAX_SIDE / longest
        rgb = rgb.resize(
            (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
            Image.Resampling.LANCZOS,
        )
        downscaled = True

    return DecodedImage(
        image=rgb,
        file_format=file_format,
        original_width=width,
        original_height=height,
        exif_orientation_applied=orientation not in (0, 1),
        downscaled=downscaled,
    )


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV BGR numpy array."""
    image_rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_base64_jpeg(image_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image as a JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Could not encode image to JPEG format.")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded


def clamp_box(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp box coordinates safely within image bounds."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def crop_square(image_bgr: np.ndarray, box: list[int], margin: float) -> np.ndarray | None:
    """
    Crop a square region centred on ``box`` and enlarged by ``margin`` on each side.

    Regions that extend past the image are filled by replicating edge pixels, so the face
    keeps its aspect ratio instead of being stretched when it is later resized.
    """
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 <= x1 or y2 <= y1 or x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None

    half = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * margin) / 2.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = int(round(cx - half)), int(round(cy - half))
    right, bottom = int(round(cx + half)), int(round(cy + half))
    if right <= left or bottom <= top:
        return None

    pad = max(0, -left, -top, right - width, bottom - height)
    if pad:
        image_bgr = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return image_bgr[top + pad : bottom + pad, left + pad : right + pad]


def letterbox_top_left(image_bgr: np.ndarray, size: int, pad_value: int = 0) -> tuple[np.ndarray, float]:
    """
    Fit the image into a ``size`` x ``size`` square without distorting it.

    The resized image sits in the top-left corner and the rest is filled with ``pad_value``. Returns the canvas
    and the scale factor, so detector boxes can be mapped back with ``box / scale``.
    """
    height, width = image_bgr.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas, scale


def box_to_percent(box: list[int], width: int, height: int) -> list[float]:
    """A pixel box as [x1, y1, x2, y2] in percent of the image, for positioning overlays in the browser."""
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = clamp_box(box, width, height)
    return [
        round(x1 / width * 100.0, 2),
        round(y1 / height * 100.0, 2),
        round(x2 / width * 100.0, 2),
        round(y2 / height * 100.0, 2),
    ]


def annotate_boxes(image_bgr: np.ndarray, items: list[BoxItem]) -> np.ndarray:
    """Draw each ``(box, label, color)`` item on a copy of the image."""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(width, height) / 1000.0)
    thickness = 2

    for box, label, color in items:
        x1, y1, x2, y2 = clamp_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = min(height - baseline - 2, max(text_height + 10, y1))
        rect_top = max(0, label_y - text_height - 8)
        rect_bottom = min(height - 1, label_y + baseline)
        rect_right = min(width - 1, x1 + text_width + 8)
        cv2.rectangle(annotated, (x1, rect_top), (rect_right, rect_bottom), color, -1)
        cv2.putText(annotated, label, (x1 + 4, label_y - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    """Intersection over Union between two [x1, y1, x2, y2] boxes."""
    inter_w = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def box_center_inside(inner: list[int], outer: list[int]) -> bool:
    """True when the centre of ``inner`` lies inside ``outer``."""
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
