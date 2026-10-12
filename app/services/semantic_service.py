"""SigLIP2 zero-shot verification of what an image or an image crop actually shows."""

import json
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings
from app.services.label_prompts import FACE_CLASSES, IMAGE_CLASSES, LabelClass
from app.services.model_manager import LABEL_BANK_NAME, MODEL_SPECS, get_session

logger = logging.getLogger(__name__)

MODEL_NAME_SIGLIP = MODEL_SPECS["siglip_vision"].display_name
_INPUT_SIZE = 224
_TAXONOMIES: dict[str, tuple[LabelClass, ...]] = {"face": FACE_CLASSES, "image": IMAGE_CLASSES}

_bank_lock = threading.Lock()
_bank: dict[str, tuple[list[str], np.ndarray]] | None = None


@dataclass(frozen=True)
class SemanticResult:
    """Result of comparing one image against a taxonomy of prompts."""

    name: str
    display: str
    kind: str
    probability: float  # softmax share among the prompts (a relative score, not a calibrated probability)
    margin: float  # probability gap between the best and the second-best class
    probabilities: dict[str, float]


def reset_cache() -> None:
    """Forget the cached label bank (used by tests and after rebuilding models)."""
    global _bank
    with _bank_lock:
        _bank = None


def _load_bank() -> dict[str, tuple[list[str], np.ndarray]] | None:
    global _bank
    with _bank_lock:
        if _bank is not None:
            return _bank
        path = settings.MODELS_DIR / LABEL_BANK_NAME
        try:
            with np.load(path, allow_pickle=False) as data:
                bank: dict[str, tuple[list[str], np.ndarray]] = {}
                for taxonomy, classes in _TAXONOMIES.items():
                    names = [str(n) for n in data[f"{taxonomy}_names"]]
                    vectors = np.asarray(data[f"{taxonomy}_vectors"], dtype=np.float32)
                    expected = [c.name for c in classes]
                    if names != expected or vectors.shape[0] != len(expected):
                        raise ValueError(f"Label bank '{taxonomy}' does not match the current taxonomy; rebuild it.")
                    bank[taxonomy] = (names, vectors)
                json.loads(str(data["meta"]))
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Label bank unavailable (%s): %s", path.name, exc)
            return None
        _bank = bank
        return _bank


def is_available() -> bool:
    """True when both the vision model and a matching label bank can be used."""
    return _load_bank() is not None and get_session("siglip_vision") is not None


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """SigLIP preprocessing: direct 224x224 bilinear resize, RGB, (x/255 - 0.5)/0.5, CHW."""
    height, width = image_bgr.shape[:2]
    interp = cv2.INTER_AREA if max(height, width) > _INPUT_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(image_bgr, (_INPUT_SIZE, _INPUT_SIZE), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - 0.5) / 0.5).transpose(2, 0, 1)


def embed_images(images_bgr: list[np.ndarray]) -> np.ndarray | None:
    """
    L2-normalised SigLIP2 image embeddings, one row per image; None if the model is unavailable.

    Images are embedded one at a time on purpose: the INT8 model quantises activations per batch, so
    batching would make one image's result depend on which other images happened to be in the batch.
    """
    session = get_session("siglip_vision")
    if session is None or not images_bgr:
        return None
    input_name = session.get_inputs()[0].name
    try:
        rows = [
            session.run(["pooler_output"], {input_name: preprocess(img)[np.newaxis].astype(np.float32)})[0][0]
            for img in images_bgr
        ]
    except Exception:
        logger.exception("SigLIP2 embedding failed")
        return None
    outputs = np.stack(rows)
    return outputs / np.maximum(np.linalg.norm(outputs, axis=1, keepdims=True), 1e-12)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def classify_embeddings(embeddings: np.ndarray, taxonomy: str) -> list[SemanticResult] | None:
    """Score each embedding against the pre-computed prompt vectors of a taxonomy."""
    bank = _load_bank()
    if bank is None:
        return None
    names, vectors = bank[taxonomy]
    classes = {c.name: c for c in _TAXONOMIES[taxonomy]}

    results: list[SemanticResult] = []
    for emb in embeddings:
        probs = _softmax((vectors @ emb) * settings.SEMANTIC_TEMPERATURE)
        order = np.argsort(-probs)
        best = classes[names[int(order[0])]]
        runner_up = float(probs[order[1]]) if len(order) > 1 else 0.0
        results.append(
            SemanticResult(
                name=best.name,
                display=best.display,
                kind=best.kind,
                probability=round(float(probs[order[0]]), 4),
                margin=round(float(probs[order[0]]) - runner_up, 4),
                probabilities={n: round(float(p), 4) for n, p in zip(names, probs, strict=True)},
            )
        )
    return results


def classify_images(images_bgr: list[np.ndarray], taxonomy: str) -> list[SemanticResult] | None:
    """Embed and classify a batch of images; None when SigLIP2 or the label bank is unavailable."""
    if _load_bank() is None:
        return None
    embeddings = embed_images(images_bgr)
    if embeddings is None:
        return None
    return classify_embeddings(embeddings, taxonomy)
