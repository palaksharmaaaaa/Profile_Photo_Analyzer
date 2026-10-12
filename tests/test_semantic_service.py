from unittest.mock import patch

import numpy as np

from app.core.config import settings
from app.services import semantic_service as ss
from app.services.label_prompts import FACE_CLASSES, IMAGE_CLASSES, prompt_templates
from tests.conftest import FakeSession, one_hot


def sigl_session(vector: np.ndarray):
    """A fake SigLIP2 vision session that always 'sees' the given embedding."""
    return FakeSession(lambda names, feeds: [vector[np.newaxis, :] * 3.0], input_name="pixel_values")


def test_taxonomies_are_complete_and_consistent():
    assert {c.name for c in FACE_CLASSES} >= {"real_human", "animal", "bird", "cartoon", "anime", "diagram", "not_face"}
    assert all(len(c.prompts) >= 2 for c in (*FACE_CLASSES, *IMAGE_CLASSES))
    assert len({c.name for c in FACE_CLASSES}) == len(FACE_CLASSES)
    templates = prompt_templates()
    assert len(templates["face"]) == sum(len(c.prompts) for c in FACE_CLASSES)
    assert len(templates["image"]) == sum(len(c.prompts) for c in IMAGE_CLASSES)


def test_preprocess_normalises_to_siglip_range_and_channel_first():
    out = ss.preprocess(np.full((50, 80, 3), 255, dtype=np.uint8))
    assert out.shape == (3, 224, 224)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0)
    big = ss.preprocess(np.zeros((900, 700, 3), dtype=np.uint8))
    assert np.allclose(big, -1.0)


def test_preprocess_swaps_bgr_to_rgb():
    blue_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255
    out = ss.preprocess(blue_bgr)
    assert out[2].min() == 1.0 and out[0].max() == -1.0  # blue ends up in the third (RGB) channel


def test_label_bank_missing_means_unavailable(models_dir):
    assert ss._load_bank() is None
    assert ss.classify_images([np.zeros((8, 8, 3), np.uint8)], "face") is None


def test_label_bank_must_match_taxonomy(models_dir):
    names = np.array(["wrong"] * len(FACE_CLASSES))
    np.savez(
        models_dir / "label_embeddings.npz",
        meta=np.array("{}"),
        face_names=names,
        face_vectors=np.zeros((len(FACE_CLASSES), 4), np.float32),
        image_names=np.array([c.name for c in IMAGE_CLASSES]),
        image_vectors=np.zeros((len(IMAGE_CLASSES), 4), np.float32),
    )
    assert ss._load_bank() is None


def test_label_bank_loads_and_is_cached(label_bank):
    bank = ss._load_bank()
    assert set(bank) == {"face", "image"}
    assert ss._load_bank() is bank


def test_embed_images_unavailable_or_empty():
    with patch.object(ss, "get_session", return_value=None):
        assert ss.embed_images([np.zeros((8, 8, 3), np.uint8)]) is None
    with patch.object(ss, "get_session", return_value=sigl_session(one_hot("face", "animal"))):
        assert ss.embed_images([]) is None


def test_embed_images_runs_each_image_separately_and_normalises():
    session = sigl_session(one_hot("face", "animal"))
    with patch.object(ss, "get_session", return_value=session):
        out = ss.embed_images([np.zeros((8, 8, 3), np.uint8)] * 3)
    assert session.calls == 3  # never batched: INT8 activations are quantised per batch
    assert out.shape == (3, 16)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_images_survives_inference_failure():
    def explode(names, feeds):
        raise RuntimeError("ort failure")

    with patch.object(ss, "get_session", return_value=FakeSession(explode)):
        assert ss.embed_images([np.zeros((8, 8, 3), np.uint8)]) is None


def test_classify_embeddings_scores_margin_and_probabilities(label_bank):
    results = ss.classify_embeddings(np.stack([one_hot("face", "real_human"), one_hot("face", "bird")]), "face")
    human, bird = results
    assert human.name == "real_human" and human.face_type == "human"
    assert human.probability > 0.99 and human.margin > 0.98
    assert bird.name == "bird"
    assert abs(sum(human.probabilities.values()) - 1.0) < 0.01


def test_classify_embeddings_ambiguous_input_has_small_margin(label_bank):
    mix = one_hot("face", "real_human") + one_hot("face", "painting")
    (result,) = ss.classify_embeddings((mix / np.linalg.norm(mix))[np.newaxis], "face")
    assert result.margin < 0.05


def test_temperature_controls_sharpness(label_bank, monkeypatch):
    monkeypatch.setattr(settings, "SEMANTIC_TEMPERATURE", 1.0)
    (result,) = ss.classify_embeddings(one_hot("face", "animal")[np.newaxis], "face")
    assert result.probability < 0.5


def test_classify_images_end_to_end_and_is_available(label_bank):
    session = sigl_session(one_hot("image", "diagram"))
    with patch.object(ss, "get_session", return_value=session):
        (result,) = ss.classify_images([np.zeros((8, 8, 3), np.uint8)], "image")
        assert result.name == "diagram"
        assert ss.is_available() is True

    with patch.object(ss, "get_session", return_value=None):
        assert ss.is_available() is False
        assert ss.classify_images([np.zeros((8, 8, 3), np.uint8)], "image") is None


def test_classify_embeddings_without_a_bank_is_none(models_dir):
    assert ss.classify_embeddings(np.zeros((1, 16), np.float32), "face") is None
