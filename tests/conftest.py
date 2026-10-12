import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.services import model_manager, semantic_service
from app.services.label_prompts import FACE_CLASSES, IMAGE_CLASSES


@pytest.fixture(autouse=True)
def _isolate_model_state():
    """Every test starts without cached sessions, detectors, hash verdicts or label banks."""
    model_manager._sessions.clear()
    model_manager._validation_cache.clear()
    model_manager._yunet_detector = None
    model_manager._yunet_size = (0, 0)
    semantic_service.reset_cache()
    yield
    model_manager._sessions.clear()
    model_manager._validation_cache.clear()
    model_manager._yunet_detector = None
    semantic_service.reset_cache()


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """A 640x640 grey image with a noisy (textured) square in the middle."""
    arr = np.full((640, 640, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    arr[200:400, 200:400] = rng.integers(60, 190, (200, 200, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def test_image_bytes(test_image_rgb) -> bytes:
    buf = io.BytesIO()
    test_image_rgb.save(buf, format="JPEG")
    return buf.getvalue()


class FakeSession:
    """Minimal stand-in for an onnxruntime InferenceSession driven by a callback."""

    def __init__(self, fn, input_name: str = "input"):
        self._fn = fn
        self._input_name = input_name
        self.calls = 0

    def get_inputs(self):
        return [type("Input", (), {"name": self._input_name})()]

    def run(self, output_names, feeds):
        self.calls += 1
        return self._fn(output_names, feeds)


class FakeDetector:
    """Stand-in for cv2.FaceDetectorYN returning faces expressed in *original* image coordinates."""

    def __init__(self, rows, original_width):
        self.rows = rows
        self.original_width = original_width

    def detect(self, image):
        if self.rows is None:
            return 1, None
        scale = image.shape[1] / self.original_width
        return 1, np.array([[v * scale for v in row[:14]] + [row[14]] for row in self.rows], dtype=np.float32)


def yunet_row(x, y, w, h, score=0.95):
    """A YuNet output row: box, 5 landmarks (eyes, nose, mouth) and score."""
    lm = [x + w * 0.3, y + h * 0.35, x + w * 0.7, y + h * 0.35, x + w * 0.5, y + h * 0.55, x + w * 0.35, y + h * 0.8]
    return [x, y, w, h, *lm, x + w * 0.65, y + h * 0.8, score]


def one_hot(taxonomy: str, name: str, dims: int = 16) -> np.ndarray:
    classes = FACE_CLASSES if taxonomy == "face" else IMAGE_CLASSES
    vec = np.zeros(dims, dtype=np.float32)
    vec[[c.name for c in classes].index(name)] = 1.0
    return vec


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.models, "directory", tmp_path)
    return tmp_path


@pytest.fixture
def label_bank(models_dir):
    """A label bank whose class vectors are orthonormal one-hots, so tests can steer classification exactly."""
    arrays: dict[str, Any] = {}
    for taxonomy, classes in (("face", FACE_CLASSES), ("image", IMAGE_CLASSES)):
        arrays[f"{taxonomy}_names"] = np.array([c.name for c in classes])
        arrays[f"{taxonomy}_vectors"] = np.stack([one_hot(taxonomy, c.name) for c in classes])
    np.savez(models_dir / model_manager.LABEL_BANK_NAME, meta=np.array("{}"), **arrays)
    return models_dir


def textured(size: int = 600, seed: int = 0) -> np.ndarray:
    """BGR image: mid-grey with a noisy square in the middle (passes information and sharpness checks)."""
    rng = np.random.default_rng(seed)
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    lo, hi = size // 3, 2 * size // 3
    image[lo:hi, lo:hi] = rng.integers(60, 190, (hi - lo, hi - lo, 3), dtype=np.uint8)
    return image
