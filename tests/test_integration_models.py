"""
Tests that exercise the real ONNX models. They are skipped automatically when the models are not present
(build them with ``python scripts/build_models.py``). Set ``PPA_TEST_IMAGES_DIR`` to a folder of your own
photos to additionally smoke-test them (the test only checks the response is well-formed and honest, since
ground truth for private photos is not known here).
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.schemas.analysis import AnalysisResponse
from app.services import semantic_service
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import decode_image_bytes
from app.services.model_manager import check_models_status, get_session

pytestmark = pytest.mark.skipif(
    not all(item["ready"] for item in check_models_status().values()),
    reason="models not built; run scripts/build_models.py",
)


def noise_image(size=480, seed=1) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def test_blank_image_is_reported_without_running_any_model():
    result = analyze_profile_photo_image(Image.new("RGB", (400, 400), (30, 90, 160)))
    assert result["posting_status"] == "NO_INFORMATION"
    assert result["technical_diagnostics"]["models_executed"] == ["Image Information & Entropy Analyzer"]


def test_pure_noise_has_information_but_no_faces():
    result = analyze_profile_photo_image(noise_image())
    AnalysisResponse.model_validate(result)
    assert result["has_information"] is True
    assert result["real_human_face_count"] == 0
    assert result["accepted"] is False
    assert result["human_faces"] == []


def test_semantic_embedding_does_not_depend_on_batch_composition():
    """Regression: the INT8 vision model quantises per batch, so crops must be embedded one at a time."""
    import cv2

    a = np.asarray(noise_image(300, 1))[:, :, ::-1].copy()
    b = np.asarray(noise_image(300, 2))[:, :, ::-1].copy()
    alone = semantic_service.classify_images([a], "face")[0]
    together = semantic_service.classify_images([b, a], "face")[1]
    assert alone.probabilities == together.probabilities
    assert cv2 is not None


def test_age_gender_model_contract():
    session = get_session("age_gender")
    out = session.run(None, {session.get_inputs()[0].name: np.zeros((6, 3, 224, 224), np.float32)})[0]
    assert out.shape == (6, 2)  # [age in years, P(female)] per view


def test_label_bank_matches_the_vision_model_dimension():
    session = get_session("siglip_vision")
    embedding = semantic_service.embed_images([np.zeros((64, 64, 3), np.uint8)])
    assert embedding.shape == (1, 768)
    bank = semantic_service._load_bank()
    assert bank["face"][1].shape[1] == embedding.shape[1]
    assert session is not None


@pytest.mark.skipif(not os.environ.get("PPA_TEST_IMAGES_DIR"), reason="PPA_TEST_IMAGES_DIR not set")
def test_user_images_produce_well_formed_honest_results():
    folder = Path(os.environ["PPA_TEST_IMAGES_DIR"])
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    assert files, "no images found"
    for path in files:
        decoded = decode_image_bytes(path.read_bytes())
        result = analyze_profile_photo_image(decoded.image, path.name, decoded.format_detected, decoded)
        AnalysisResponse.model_validate(result)
        for face in result["human_faces"]:
            assert face["age_status"] in ("Estimated", "Unsure", "Not Detected")
            if face["age_status"] != "Estimated":
                assert face["estimated_age"] is None and face["age_range"] is None
