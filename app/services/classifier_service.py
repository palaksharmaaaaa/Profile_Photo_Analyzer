"""
What is in the image: whole-image type, animals / people / anime faces (detectors), and the verdict on each
crop (real human face, animal, cartoon, ...) from SigLIP2.
"""

import logging
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.services import semantic_service
from app.services.image_utils import clamp_box, crop_square
from app.services.model_manager import MODEL_SPECS, get_session

logger = logging.getLogger(__name__)

MODEL_NAME_YOLOX = MODEL_SPECS["yolox"].display_name
MODEL_NAME_ANIME_FACE = MODEL_SPECS["anime_face"].display_name
MODEL_NAME_SIGLIP = semantic_service.MODEL_NAME_SIGLIP

# COCO class id -> (api face_type, display label). Only animals that can have a face are of interest.
COCO_ANIMAL_CLASSES: dict[int, tuple[str, str]] = {
    14: ("bird", "Bird"),
    15: ("animal", "Cat"),
    16: ("animal", "Dog"),
    17: ("animal", "Horse"),
    18: ("animal", "Sheep"),
    19: ("animal", "Cow"),
    20: ("animal", "Elephant"),
    21: ("animal", "Bear"),
    22: ("animal", "Zebra"),
    23: ("animal", "Giraffe"),
}

PERSON_CLASS: dict[int, tuple[str, str]] = {0: ("person", "Person")}
DETECTOR_INPUT = 640  # YOLOX and the anime detector take a 640 x 640 letterboxed image
LETTERBOX_PAD = 114  # grey padding value used when letterboxing
# The two groups of whole-image types (see label_prompts.IMAGE_CLASSES)
PHOTO_GROUP = "photo"
SYNTHETIC_GROUP = "synthetic"

# Verdict on one crop
VERDICT_HUMAN = "human"
VERDICT_NON_HUMAN = "non_human"
VERDICT_NOT_FACE = "not_face"
VERDICT_UNSURE = "unsure"
VERDICT_UNAVAILABLE = "unavailable"


# --------------------------------------------------------------------------------------
# Image-type screening (informational; per-face verdicts are decided on the face crops)
# --------------------------------------------------------------------------------------


def _unavailable_image_type() -> dict[str, Any]:
    return {
        "status": "UNCERTAIN",
        "label": "Uncertain",
        "photo_share": None,
        "synthetic_share": None,
        "predictions": [],
        "model": None,
        "reason": "The SigLIP2 image-type model is unavailable, so the image type was not checked.",
    }


def classify_image_type(image_bgr: np.ndarray) -> dict[str, Any]:
    """Classify the whole upload as a photograph or as a cartoon, anime, painting, 3D render or diagram."""
    results = semantic_service.classify_images([image_bgr], "image")
    if not results:
        return _unavailable_image_type()

    result = results[0]
    classes = {c.name: c for c in semantic_service.IMAGE_CLASSES}
    photo_share = sum(p for name, p in result.probabilities.items() if classes[name].kind == PHOTO_GROUP)
    synthetic_share = sum(p for name, p in result.probabilities.items() if classes[name].kind == SYNTHETIC_GROUP)

    if photo_share >= settings.IMAGE_TYPE_MIN_SHARE:
        status, reason = "PHOTO_LIKE", "Most of the probability is on photographic image types."
    elif synthetic_share >= settings.IMAGE_TYPE_MIN_SHARE:
        status, reason = "NON_PHOTO", "Most of the probability is on drawn, painted or computer-made image types."
    else:
        status, reason = "NEEDS_REVIEW", "The image type could not be determined confidently."

    ranked = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
    predictions = [{"label": classes[name].display, "probability": probability} for name, probability in ranked]

    # The label names the best class inside the winning group, so it never contradicts the status.
    winning_group = {"PHOTO_LIKE": PHOTO_GROUP, "NON_PHOTO": SYNTHETIC_GROUP}.get(status)
    shown = next((classes[name] for name, _ in ranked if classes[name].kind == winning_group), classes[result.name])
    return {
        "status": status,
        "label": shown.display,
        "photo_share": round(photo_share, 4),
        "synthetic_share": round(synthetic_share, 4),
        "predictions": predictions,
        "model": MODEL_NAME_SIGLIP,
        "reason": reason,
    }


# --------------------------------------------------------------------------------------
# Crop-level verification of what a detected box contains
# --------------------------------------------------------------------------------------


def _verdict_from_probabilities(result: semantic_service.SemanticResult) -> dict[str, Any]:
    """Turn SigLIP2 prompt scores into a verdict, preferring ``unsure`` over guessing."""
    probs = result.probabilities
    human = probs.get("real_human", 0.0)
    base = {"scores": probs, "margin": result.margin}

    if result.name == "real_human" and human >= settings.HUMAN_MIN_PROB and result.margin >= settings.HUMAN_MIN_MARGIN:
        return {**base, "verdict": VERDICT_HUMAN, "face_type": "human", "label": result.display, "confidence": human}

    if human < settings.NONHUMAN_MAX_HUMAN_PROB:
        if result.name == "not_face":
            if result.probability >= settings.NOT_FACE_MIN_PROB:
                return {
                    **base,
                    "verdict": VERDICT_NOT_FACE,
                    "face_type": result.kind,
                    "label": result.display,
                    "confidence": result.probability,
                }
        else:
            return {
                **base,
                "verdict": VERDICT_NON_HUMAN,
                "face_type": result.kind,
                "label": result.display,
                "confidence": result.probability,
            }

    return {
        **base,
        "verdict": VERDICT_UNSURE,
        "face_type": "unsure",
        "label": f"Unsure (closest: {result.display})",
        "confidence": result.probability,
    }


def classify_boxes(image_bgr: np.ndarray, boxes: list[list[int]]) -> list[dict[str, Any]]:
    """Verify what each box contains (real human, animal, cartoon, ...). One verdict per box, in order."""
    if not boxes:
        return []

    crops, index = [], []
    for i, box in enumerate(boxes):
        crop = crop_square(image_bgr, box, settings.FACE_CROP_MARGIN)
        if crop is not None and crop.size:
            crops.append(crop)
            index.append(i)

    unavailable = {
        "verdict": VERDICT_UNAVAILABLE,
        "face_type": "unsure",
        "label": "Unsure (verification model unavailable)",
        "confidence": None,
        "margin": None,
        "scores": {},
    }
    verdicts = [dict(unavailable) for _ in boxes]
    results = semantic_service.classify_images(crops, "face") if crops else None
    if results:
        for i, result in zip(index, results, strict=True):
            verdicts[i] = _verdict_from_probabilities(result)
    return verdicts


# --------------------------------------------------------------------------------------
# Non-human entity localisation (animals/birds via YOLOX, anime faces via DeepGHS)
# --------------------------------------------------------------------------------------


def _letterbox(image_bgr: np.ndarray, to_rgb: bool) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio into a top-left aligned DETECTOR_INPUT square padded with grey."""
    height, width = image_bgr.shape[:2]
    ratio = min(DETECTOR_INPUT / height, DETECTOR_INPUT / width)
    new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))
    canvas = np.full((DETECTOR_INPUT, DETECTOR_INPUT, 3), LETTERBOX_PAD, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas, ratio


@lru_cache(maxsize=1)
def _yolox_grid() -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for stride in (8, 16, 32):
        n = DETECTOR_INPUT // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), stride))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


def _nms(boxes_xywh: list[list[float]], scores: list[float], score_threshold: float) -> list[int]:
    if not boxes_xywh:
        return []
    kept = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold, 0.45)
    return [int(i) for i in np.asarray(kept).flatten()] if len(kept) else []


def _to_entities(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    labels: list[tuple[str, str]],
    threshold: float,
    source: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    boxes_xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes_xyxy]
    entities = []
    for i in _nms(boxes_xywh, [float(s) for s in scores], threshold):
        x1, y1, x2, y2 = clamp_box([int(round(v)) for v in boxes_xyxy[i]], width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        face_type, label = labels[i]
        entities.append(
            {
                "face_type": face_type,
                "label": label,
                "confidence": round(float(scores[i]), 3),
                "box_xyxy": [x1, y1, x2, y2],
                "source": source,
            }
        )
    return entities


def _decode_yolox(
    output: np.ndarray,
    ratio: float,
    width: int,
    height: int,
    threshold: float,
    classes: dict[int, tuple[str, str]] = COCO_ANIMAL_CLASSES,
) -> list[dict[str, Any]]:
    """Decode raw YOLOX output [1, 8400, 85] (grid/stride decoding, objectness x class score) for ``classes``."""
    if output.ndim != 3 or output.shape[2] < 6:
        return []
    preds = output[0]
    grid, stride = _yolox_grid()
    if preds.shape[0] != grid.shape[0]:
        return []

    centers = (preds[:, :2] + grid) * stride
    sizes = np.exp(np.clip(preds[:, 2:4], -10, 10)) * stride
    class_ids = np.array(sorted(classes))
    class_scores = preds[:, 4:5] * preds[:, 5:][:, class_ids]
    best = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)
    keep = best_score >= threshold
    if not np.any(keep):
        return []

    boxes = np.concatenate([centers - sizes / 2.0, centers + sizes / 2.0], axis=1)[keep] / ratio
    labels = [classes[int(class_ids[b])] for b in best[keep]]
    return _to_entities(boxes, best_score[keep], labels, threshold, MODEL_NAME_YOLOX, width, height)


def _decode_anime(output: np.ndarray, ratio: float, width: int, height: int, threshold: float) -> list[dict[str, Any]]:
    """Decode the single-class Ultralytics-style anime face output [1, 5, 8400]."""
    if output.ndim != 3 or output.shape[1] < 5:
        return []
    feats = output[0].T
    keep = feats[:, 4] >= threshold
    if not np.any(keep):
        return []
    rows = feats[keep]
    boxes = np.stack(
        [
            rows[:, 0] - rows[:, 2] / 2.0,
            rows[:, 1] - rows[:, 3] / 2.0,
            rows[:, 0] + rows[:, 2] / 2.0,
            rows[:, 1] + rows[:, 3] / 2.0,
        ],
        axis=1,
    )
    boxes = boxes / ratio
    labels = [("anime", "Anime / Cartoon Character")] * len(rows)
    return _to_entities(boxes, rows[:, 4], labels, threshold, MODEL_NAME_ANIME_FACE, width, height)


def detect_entities_and_persons(
    image_bgr: np.ndarray,
) -> tuple[list[dict[str, Any]], list[list[int]], list[str]]:
    """
    Locate animals/birds (YOLOX), people (YOLOX, same pass) and anime faces (DeepGHS).

    Returns the non-human candidate entities, the person boxes (used as body crops for age/gender) and the names
    of the models that really ran without error.
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], [], []

    height, width = image_bgr.shape[:2]
    entities: list[dict[str, Any]] = []
    persons: list[list[int]] = []
    executed: list[str] = []

    yolox = get_session("yolox")
    if yolox is not None:
        try:
            canvas, ratio = _letterbox(image_bgr, to_rgb=False)
            blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            output = yolox.run(None, {yolox.get_inputs()[0].name: blob})[0]
            entities.extend(_decode_yolox(output, ratio, width, height, settings.ANIMAL_CONF_THRESHOLD))
            people = _decode_yolox(output, ratio, width, height, settings.PERSON_CONF_THRESHOLD, PERSON_CLASS)
            persons.extend(person["box_xyxy"] for person in people)
            executed.append(MODEL_NAME_YOLOX)
        except Exception:
            logger.exception("Animal/person detection failed")

    anime = get_session("anime_face")
    if anime is not None:
        try:
            canvas, ratio = _letterbox(image_bgr, to_rgb=True)
            blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
            output = anime.run(None, {anime.get_inputs()[0].name: blob})[0]
            entities.extend(_decode_anime(output, ratio, width, height, settings.ANIME_FACE_CONF_THRESHOLD))
            executed.append(MODEL_NAME_ANIME_FACE)
        except Exception:
            logger.exception("Anime face detection failed")

    return entities, persons, executed
