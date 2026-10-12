import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.services.image_utils import (
    annotate_faces,
    bgr_to_base64_jpeg,
    clamp_box,
    compute_normalized_box,
    decode_image_bytes,
    pil_to_bgr,
    sniff_image_magic_bytes,
)


def test_decode_image_bytes_valid():
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    decoded = decode_image_bytes(buf.getvalue())
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (50, 50)


def test_decode_image_bytes_invalid():
    with pytest.raises(ValueError, match="Invalid or unsupported"):
        decode_image_bytes(b"not an image byte sequence")


def test_decode_image_bytes_corrupted_payload():
    corrupted = b"\xff\xd8\xff\xe0" + b"\x00" * 30
    with pytest.raises(ValueError, match="Invalid or corrupted"):
        decode_image_bytes(corrupted)


def test_sniff_image_magic_bytes_formats():
    assert sniff_image_magic_bytes(b"\xff\xd8\xff\xe0\x00") == "image/jpeg"
    assert sniff_image_magic_bytes(b"\x89PNG\r\n\x1a\n\x00\x00") == "image/png"
    assert sniff_image_magic_bytes(b"RIFF\x00\x00\x00\x00WEBP\x00") == "image/webp"
    assert sniff_image_magic_bytes(b"BM\x00\x00\x00") == "image/bmp"

    with pytest.raises(ValueError, match="empty or too short"):
        sniff_image_magic_bytes(b"")

    with pytest.raises(ValueError, match="empty or too short"):
        sniff_image_magic_bytes(b"ab")

    with pytest.raises(ValueError, match="Invalid or unsupported"):
        sniff_image_magic_bytes(b"FAKE_IMAGE_HEADER_123")


def test_compute_normalized_box():
    # Zero/negative bounds
    assert compute_normalized_box([10, 10, 50, 50], 0, 100) == [0.0, 0.0, 0.0, 0.0]
    assert compute_normalized_box([10, 10, 50, 50], 100, 0) == [0.0, 0.0, 0.0, 0.0]

    # Standard clamping and conversion: box is [x1, y1, x2, y2] = [20, 10, 50, 60] on 100x200 (width=100, height=200)
    # ymin% = (10/200)*100 = 5.0%, xmin% = (20/100)*100 = 20.0%, ymax% = (60/200)*100 = 30.0%, xmax% = (50/100)*100 = 50.0%
    norm = compute_normalized_box([20, 10, 50, 60], 100, 200)
    assert norm == [5.0, 20.0, 30.0, 50.0]


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
