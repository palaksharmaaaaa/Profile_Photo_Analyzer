import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.services.image_utils import (
    InvalidImageError,
    annotate_boxes,
    bgr_to_base64_jpeg,
    box_center_inside,
    box_iou,
    box_to_percent,
    clamp_box,
    crop_square,
    decode_image_bytes,
    letterbox_top_left,
    pil_to_bgr,
    sniff_image_magic_bytes,
)


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"\xff\xd8\xff\xe0abc", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\nxxxx", "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"GIF89a....", "image/gif"),
        (b"GIF87a....", "image/gif"),
        (b"II*\x00....", "image/tiff"),
        (b"MM\x00*....", "image/tiff"),
        (b"BM......", "image/bmp"),
    ],
)
def test_sniff_recognises_real_formats(payload, expected):
    assert sniff_image_magic_bytes(payload) == expected


@pytest.mark.parametrize("payload", [b"", b"abc", b"NOT_AN_IMAGE_AT_ALL"])
def test_sniff_rejects_unknown_or_short(payload):
    with pytest.raises(ValueError):
        sniff_image_magic_bytes(payload)


def test_decode_plain_jpeg():
    decoded = decode_image_bytes(encode(Image.new("RGB", (64, 48), "red"), "JPEG"))
    assert decoded.image.size == (64, 48)
    assert decoded.image.mode == "RGB"
    assert decoded.file_format == "image/jpeg"
    assert (decoded.original_width, decoded.original_height) == (64, 48)
    assert decoded.exif_orientation_applied is False
    assert decoded.downscaled is False


def test_decode_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, display rotated 90 degrees
    decoded = decode_image_bytes(encode(Image.new("RGB", (40, 20), "blue"), "JPEG", exif=exif))
    assert decoded.image.size == (20, 40)
    assert decoded.exif_orientation_applied is True
    assert (decoded.original_width, decoded.original_height) == (40, 20)


def test_decode_flattens_transparency_onto_white():
    decoded = decode_image_bytes(encode(Image.new("RGBA", (10, 10), (0, 0, 0, 0)), "PNG"))
    assert np.array(decoded.image).min() == 255


def test_decode_palette_transparency():
    palette = Image.new("P", (8, 8), 0)
    palette.putpalette([0, 0, 0] * 256)
    palette.info["transparency"] = 0
    decoded = decode_image_bytes(encode(palette, "PNG", transparency=0))
    assert decoded.image.mode == "RGB"
    assert np.array(decoded.image).min() == 255


def test_decode_animated_gif_uses_first_frame():
    frames = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "green")]
    payload = encode(frames[0], "GIF", save_all=True, append_images=frames[1:])
    decoded = decode_image_bytes(payload)
    assert decoded.file_format == "image/gif"
    assert decoded.image.size == (16, 16)


def test_decode_downscales_to_working_size(monkeypatch):
    monkeypatch.setattr(settings, "WORKING_MAX_SIDE", 100)
    decoded = decode_image_bytes(encode(Image.new("RGB", (300, 200), "grey"), "PNG"))
    assert max(decoded.image.size) == 100
    assert decoded.downscaled is True
    assert (decoded.original_width, decoded.original_height) == (300, 200)


def test_decode_uses_jpeg_draft_for_big_jpegs(monkeypatch):
    monkeypatch.setattr(settings, "WORKING_MAX_SIDE", 64)
    decoded = decode_image_bytes(encode(Image.new("RGB", (1024, 768), "grey"), "JPEG"))
    assert max(decoded.image.size) == 64
    assert (decoded.original_width, decoded.original_height) == (1024, 768)


def test_decode_rejects_oversized_dimension(monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_DIMENSION", 50)
    too_wide = encode(Image.new("RGB", (60, 10), "grey"), "PNG")
    with pytest.raises(InvalidImageError, match="dimension"):
        decode_image_bytes(too_wide)


@pytest.mark.parametrize("pixels", [1500, 50_000])
def test_decode_rejects_decompression_bomb(monkeypatch, pixels):
    monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 1000)
    side = int(pixels**0.5)
    bomb = encode(Image.new("RGB", (side, side), "grey"), "PNG")
    with pytest.raises(InvalidImageError, match="decompression"):
        decode_image_bytes(bomb)


def test_decode_rejects_unknown_format():
    with pytest.raises(InvalidImageError, match="unsupported binary format"):
        decode_image_bytes(b"this is not an image")


def test_decode_rejects_corrupt_data_with_valid_header():
    with pytest.raises(InvalidImageError, match="not a valid supported image"):
        decode_image_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 40)


def test_pil_to_bgr_swaps_channels():
    bgr = pil_to_bgr(Image.new("RGB", (2, 2), (255, 0, 0)))
    assert tuple(bgr[0, 0]) == (0, 0, 255)


def test_bgr_to_base64_jpeg_roundtrip():
    url = bgr_to_base64_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
    assert url.startswith("data:image/jpeg;base64,")


def test_bgr_to_base64_jpeg_encode_failure(monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    with pytest.raises(ValueError, match="Could not encode"):
        bgr_to_base64_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))


def test_clamp_box():
    assert clamp_box([-5, -5, 500, 500], 100, 80) == (0, 0, 100, 80)
    assert clamp_box([150, 150, 160, 160], 100, 80) == (99, 79, 100, 80)


def test_box_to_percent_is_x1_y1_x2_y2():
    assert box_to_percent([10, 20, 50, 60], 100, 200) == [10.0, 10.0, 50.0, 30.0]
    assert box_to_percent([0, 0, 1, 1], 0, 100) == [0.0, 0.0, 0.0, 0.0]


def test_crop_square_is_square_and_expands_by_margin():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_square(image, [80, 80, 120, 120], 0.5)
    assert crop.shape[:2] == (80, 80)


def test_crop_square_replicates_edges_instead_of_stretching():
    image = np.full((100, 100, 3), 77, dtype=np.uint8)
    crop = crop_square(image, [0, 0, 40, 40], 0.5)
    assert crop.shape[:2] == (80, 80)
    assert crop.min() == 77  # padding replicates image pixels, no black fill


@pytest.mark.parametrize("box", [[10, 10, 10, 20], [50, 50, 40, 60], [-20, -20, -5, -5], [300, 300, 400, 400]])
def test_crop_square_rejects_degenerate_boxes(box):
    assert crop_square(np.zeros((100, 100, 3), dtype=np.uint8), box, 0.2) is None


def test_crop_square_rejects_zero_size_after_rounding():
    assert crop_square(np.zeros((100, 100, 3), dtype=np.uint8), [10, 10, 10.0001, 10.0001], -0.5) is None


def test_annotate_boxes_draws_without_mutating_input():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    out = annotate_boxes(image, [([20, 20, 80, 80], "Label", (0, 255, 0)), ([5, 5, 5, 5], "skip", (0, 0, 255))])
    assert image.max() == 0
    assert out.max() > 0


def test_box_iou_and_center():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert 0 < box_iou([0, 0, 10, 10], [5, 5, 15, 15]) < 1
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert box_center_inside([40, 40, 60, 60], [0, 0, 100, 100]) is True
    assert box_center_inside([200, 200, 220, 220], [0, 0, 100, 100]) is False


def test_letterbox_top_left_keeps_aspect_and_reports_the_scale():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    canvas, scale = letterbox_top_left(image, 400, pad_value=114)
    assert canvas.shape == (400, 400, 3)
    assert scale == 2.0
    assert canvas[0, 0, 0] == 255
    assert canvas[199, 399, 0] == 255  # the scaled image fills the top 200 rows
    assert canvas[200, 0, 0] == 114
    assert canvas[399, 399, 0] == 114  # grey padding below
    tiny, tiny_scale = letterbox_top_left(np.zeros((1, 1, 3), np.uint8), 8)
    assert tiny.shape == (8, 8, 3)
    assert tiny_scale == 8.0
