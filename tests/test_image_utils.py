import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch
from app.services.image_utils import (
    decode_image_bytes,
    pil_to_bgr,
    bgr_to_base64_jpeg,
    clamp_box,
    annotate_faces,
)


def test_decode_image_bytes_valid():
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = decode_image_bytes(buf.getvalue())
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (50, 50)


def test_decode_image_bytes_invalid():
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        decode_image_bytes(b"not an image byte sequence")


def test_pil_to_bgr():
    img = Image.new("RGB", (20, 20), color=(255, 0, 0))
    bgr = pil_to_bgr(img)
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == (20, 20, 3)
    # Red in RGB is Blue in BGR (0, 0, 255)
    assert bgr[0, 0, 0] == 0
    assert bgr[0, 0, 2] == 255


def test_bgr_to_base64_jpeg():
    arr = np.zeros((30, 30, 3), dtype=np.uint8)
    encoded = bgr_to_base64_jpeg(arr)
    assert encoded.startswith("data:image/jpeg;base64,")


def test_bgr_to_base64_jpeg_failure():
    arr = np.zeros((30, 30, 3), dtype=np.uint8)
    with patch("cv2.imencode", return_value=(False, None)):
        with pytest.raises(ValueError, match="Could not encode"):
            bgr_to_base64_jpeg(arr)


def test_clamp_box():
    # Box exceeds image boundaries
    box = [-10, -5, 600, 700]
    x1, y1, x2, y2 = clamp_box(box, 500, 500)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 500
    assert y2 == 500


def test_annotate_faces_inverted_box():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    # Box with x2 <= x1 (inverted)
    inverted_boxes = [[150, 50, 50, 150]]
    result = annotate_faces(image, inverted_boxes, "TEST", (0, 255, 0))
    assert isinstance(result, np.ndarray)
