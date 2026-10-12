import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Create a synthetic 640x640 RGB image with a distinct centered square."""
    img = Image.new("RGB", (640, 640), color=(200, 200, 200))
    arr = np.array(img)
    # Add high-frequency noise in the center to pass sharpness check
    noise = np.random.randint(50, 180, (200, 200, 3), dtype=np.uint8)
    arr[200:400, 200:400] = noise
    return Image.fromarray(arr)


@pytest.fixture
def test_image_bytes(test_image_rgb) -> bytes:
    """Convert test RGB image to JPEG bytes."""
    buf = io.BytesIO()
    test_image_rgb.save(buf, format="JPEG")
    return buf.getvalue()


class MockFace:
    """Mock InsightFace detection result."""

    def __init__(self, bbox=(200, 200, 400, 400), det_score=0.98, kps=None, age=28.0, gender=1):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = det_score
        self.kps = (
            kps
            if kps is not None
            else np.array(
                [[250, 260], [350, 260], [300, 300], [260, 350], [340, 350]],
                dtype=np.float32,
            )
        )
        self.age = age
        self.gender = gender
