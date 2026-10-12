import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.image_utils import clamp_box
from app.services.model_manager import get_anime_face_session, get_yolo_session

logger = logging.getLogger(__name__)

LABEL_REAL_PHOTO = "a real photograph of a real human person"
LABEL_CARTOON = "a cartoon drawing of a person"
LABEL_ANIME = "an anime or Ghibli-style illustration"
LABEL_DIGITAL_PAINTING = "a digital painting of a fictional character"
LABEL_3D = "a computer-generated 3D character"
LABEL_OBJECT = "a photograph or picture of an object with no person"
LABEL_DIAGRAM = "a biological or scientific diagram"
LABEL_UNCERTAIN = "uncertain or unverified image type"

IMAGE_LABELS = [
    LABEL_REAL_PHOTO,
    LABEL_CARTOON,
    LABEL_ANIME,
    LABEL_DIGITAL_PAINTING,
    LABEL_3D,
    LABEL_OBJECT,
    LABEL_DIAGRAM,
]

SYNTHETIC_KEYWORDS = (
    "cartoon",
    "anime",
    "illustration",
    "digital painting",
    "computer-generated",
    "diagram",
)

KEYWORD_REAL_PHOTO = "real photograph"

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

_clip_pipeline = None


def get_clip_pipeline():
    """Lazy initialize the CLIP zero-shot classification pipeline (legacy/fallback)."""
    global _clip_pipeline
    if _clip_pipeline is None:
        try:
            from transformers import pipeline

            device = -1 if settings.FORCE_CPU else 0
            logger.info("Initializing zero-shot image classifier pipeline...")
            _clip_pipeline = pipeline(
                "zero-shot-image-classification",
                model=settings.CLIP_MODEL_NAME,
                device=device,
            )
        except Exception as exc:
            logger.warning(f"Could not load Hugging Face CLIP pipeline: {exc}")
            _clip_pipeline = False
    return _clip_pipeline if _clip_pipeline is not False else None


def friendly_image_type(top_class: str) -> str:
    """Map candidate prompt to a clean user-facing title."""
    norm = (top_class or "").lower()
    if KEYWORD_REAL_PHOTO in norm:
        return "Real Photo"
    if "anime" in norm:
        return "Anime / Ghibli-style Image"
    if "cartoon" in norm:
        return "Cartoon Image"
    if "digital painting" in norm:
        return "Digital Painting (Fictional Character)"
    if "computer-generated" in norm:
        return "3D / Computer-generated Character"
    if "diagram" in norm:
        return "Biological / Scientific Diagram"
    if "object with no person" in norm:
        return "Object / No Person"
    return "Uncertain"


def _apply_nms(boxes: list[list[int]], scores: list[float], score_threshold: float, nms_threshold: float = 0.45) -> Any:
    """Run OpenCV non-maximum suppression on detected boxes."""
    return cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)


def _decode_yolo_predictions(
    output_tensor: np.ndarray,
    orig_w: int,
    orig_h: int,
    score_threshold: float,
    is_anime: bool = False,
) -> list[dict[str, Any]]:
    """Decode YOLO output tensor (shape [1, C, 8400]) into boxes and scores with NMS."""
    if output_tensor is None or len(output_tensor.shape) != 3:
        return []

    # Transpose [1, C, 8400] -> [8400, C]
    features = output_tensor[0].T
    boxes: list[list[int]] = []
    scores: list[float] = []
    labels: list[tuple[str, str]] = []

    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    if is_anime:
        confidences = features[:, 4]
        mask = confidences >= score_threshold
        valid_rows = features[mask]
        for row in valid_rows:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x1 = round((cx - w / 2.0) * scale_x)
            y1 = round((cy - h / 2.0) * scale_y)
            bw = round(w * scale_x)
            bh = round(h * scale_y)
            boxes.append([x1, y1, bw, bh])
            scores.append(float(row[4]))
            labels.append(("cartoon_or_illustration", "Anime / Cartoon Character"))
    else:
        class_scores = features[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        max_classes = np.argmax(class_scores, axis=1)

        mask = max_scores >= score_threshold
        valid_rows = features[mask]
        valid_scores = max_scores[mask]
        valid_classes = max_classes[mask]

        for row, score, cls_id in zip(valid_rows, valid_scores, valid_classes):
            cid = int(cls_id)
            if cid in COCO_ANIMAL_CLASSES:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = round((cx - w / 2.0) * scale_x)
                y1 = round((cy - h / 2.0) * scale_y)
                bw = round(w * scale_x)
                bh = round(h * scale_y)
                boxes.append([x1, y1, bw, bh])
                scores.append(float(score))
                labels.append(COCO_ANIMAL_CLASSES[cid])

    if not boxes:
        return []

    indices = _apply_nms(boxes, scores, score_threshold, 0.45)
    if len(indices) == 0:
        return []

    results = []
    for idx in np.asarray(indices).flatten():
        i = int(idx)
        bx, by, bw, bh = boxes[i]
        clamped = list(clamp_box([bx, by, bx + bw, by + bh], orig_w, orig_h))
        ftype, flabel = labels[i]
        results.append(
            {
                "face_type": ftype,
                "label": flabel,
                "confidence": round(scores[i], 3),
                "box_xyxy": clamped,
            }
        )
    return results


def detect_non_human_entities(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """
    Run lightweight YOLOv8 models to detect:
    1. Animals & Birds (COCO classes)
    2. Anime & Cartoon faces (DeepGHS model)
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    h, w = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, (640, 640))
    blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :, :, :]

    entities: list[dict[str, Any]] = []

    # 1. Animals & Birds via YOLOv8n
    yolo_session = get_yolo_session()
    if yolo_session is not None:
        try:
            inp_name = yolo_session.get_inputs()[0].name
            out = yolo_session.run(None, {inp_name: blob})[0]
            animals = _decode_yolo_predictions(out, w, h, settings.ANIMAL_CONF_THRESHOLD, is_anime=False)
            entities.extend(animals)
        except Exception as exc:
            logger.warning(f"Animal detection failed: {exc}")

    # 2. Anime & Cartoon faces via Anime YOLOv8
    anime_session = get_anime_face_session()
    if anime_session is not None:
        try:
            inp_name = anime_session.get_inputs()[0].name
            out = anime_session.run(None, {inp_name: blob})[0]
            cartoons = _decode_yolo_predictions(out, w, h, settings.ANIME_FACE_CONF_THRESHOLD, is_anime=True)
            entities.extend(cartoons)
        except Exception as exc:
            logger.warning(f"Anime face detection failed: {exc}")

    return entities


def is_biological_diagram_or_drawing(crop_bgr: np.ndarray) -> bool:
    """
    Check if a face crop is a biological diagram, schematic, line art,
    or textbook illustration instead of a photographic face.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return False

    # 1. White or near-white background ratio
    white_mask = (crop_bgr[:, :, 0] > 230) & (crop_bgr[:, :, 1] > 230) & (crop_bgr[:, :, 2] > 230)
    white_ratio = float(np.mean(white_mask))

    # 2. Check skin color coverage in YCrCb color space
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    # Human skin usually falls in Cr: 133-173, Cb: 77-127
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    skin_ratio = float(np.mean(skin_mask))

    # If >65% white background and almost no natural skin tones, it's a diagram/drawing
    if white_ratio > 0.65 and skin_ratio < 0.15:
        return True

    # 3. High edge sparsity (black lines on solid/light ground)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.mean(edges > 0))

    return bool(white_ratio > 0.5 and edge_ratio > 0.05 and skin_ratio < 0.1)


def _compute_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = box_a_area + box_b_area - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def check_is_non_human_face(
    face_box: list[int], image_bgr: np.ndarray, non_human_entities: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """
    Check if a face box corresponds to an animal, bird, cartoon, or diagram.
    Returns entity dict if non-human, or None if genuine human candidate.
    """
    # 1. Overlap with detected animal, bird, or anime face
    for ent in non_human_entities:
        ent_box = ent["box_xyxy"]
        iou = _compute_iou(face_box, ent_box)
        # Check center inside entity box
        cx = (face_box[0] + face_box[2]) // 2
        cy = (face_box[1] + face_box[3]) // 2
        center_inside = (ent_box[0] <= cx <= ent_box[2]) and (ent_box[1] <= cy <= ent_box[3])

        if iou > 0.25 or center_inside:
            return {
                "face_type": ent["face_type"],
                "label": ent["label"],
                "confidence": ent["confidence"],
                "box_xyxy": face_box,
            }

    # 2. Biological diagram check on face crop
    x1, y1, x2, y2 = clamp_box(face_box, image_bgr.shape[1], image_bgr.shape[0])
    crop = image_bgr[y1:y2, x1:x2]
    if is_biological_diagram_or_drawing(crop):
        return {
            "face_type": "biological_diagram",
            "label": "Biological Diagram / Illustration",
            "confidence": 0.90,
            "box_xyxy": face_box,
        }

    return None


def is_cartoon_or_illustration(image_bgr: np.ndarray) -> bool:
    """Detect whether an image exhibits cartoon or digital illustration traits."""
    if image_bgr.size == 0:
        return False
    small = cv2.resize(image_bgr, (128, 128))
    quantized = small // 32
    flat_colors = quantized.reshape(-1, 3)
    unique_colors = len(np.unique(flat_colors, axis=0))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = float(np.mean(edges > 0))

    non_edge_mask = edges == 0
    flat_variance = float(np.var(gray[non_edge_mask])) if np.sum(non_edge_mask) > 100 else 0.0

    is_flat_palette = unique_colors < 120
    is_hard_outline_flat_texture = edge_ratio > 0.06 and flat_variance < 180.0
    return bool(is_flat_palette or is_hard_outline_flat_texture)


def classify_image_type(image: Image.Image) -> dict[str, Any]:
    """
    Classify whether image is a real photo, cartoon, anime, 3D character, diagram, or object.
    Uses zero-shot CLIP classification when available, with resilient computer vision screening.
    """
    pipeline_fn = get_clip_pipeline()

    if pipeline_fn is None:
        image_rgb = image.convert("RGB")
        arr = np.array(image_rgb)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # 1. Check biological diagram
        if is_biological_diagram_or_drawing(arr_bgr):
            return {
                "status": "NON_PHOTO",
                "top_class": LABEL_DIAGRAM,
                "top_class_confidence": 0.90,
                "photo_score": 0.05,
                "cartoon_score": 0.90,
                "display_label": "Biological / Scientific Diagram",
                "model_executed": "Heuristic CV Screener",
                "all_predictions": [
                    {"label": LABEL_DIAGRAM, "confidence": 0.90},
                    {"label": LABEL_REAL_PHOTO, "confidence": 0.05},
                ],
                "scores": {LABEL_DIAGRAM: 0.90, LABEL_REAL_PHOTO: 0.05},
                "reason": "Image has high line sparsity and vector characteristics of a diagram.",
            }

        # 2. Check cartoon / digital illustration traits
        if is_cartoon_or_illustration(arr_bgr):
            return {
                "status": "NON_PHOTO",
                "top_class": LABEL_CARTOON,
                "top_class_confidence": 0.85,
                "photo_score": 0.10,
                "cartoon_score": 0.85,
                "display_label": "Cartoon / Illustrated Image",
                "model_executed": "Heuristic CV Screener",
                "all_predictions": [
                    {"label": LABEL_CARTOON, "confidence": 0.85},
                    {"label": LABEL_REAL_PHOTO, "confidence": 0.10},
                ],
                "scores": {LABEL_CARTOON: 0.85, LABEL_REAL_PHOTO: 0.10},
                "reason": "Image exhibits flat color palette and hard vector/illustration outlines.",
            }

        # 3. Fallback when CLIP is unavailable: Treat heuristic-only output as review-required
        return {
            "status": "NEEDS_REVIEW",
            "top_class": LABEL_UNCERTAIN,
            "top_class_confidence": 0.50,
            "photo_score": 0.50,
            "cartoon_score": 0.50,
            "display_label": "Heuristic Screening (Review Required)",
            "model_executed": "Heuristic CV Screener",
            "all_predictions": [
                {"label": LABEL_UNCERTAIN, "confidence": 0.50},
                {"label": LABEL_REAL_PHOTO, "confidence": 0.50},
            ],
            "scores": {LABEL_UNCERTAIN: 0.50, LABEL_REAL_PHOTO: 0.50},
            "reason": "Zero-shot neural classifier (CLIP) unavailable; heuristic screening requires manual review.",
        }

    try:
        raw_predictions = pipeline_fn(image.convert("RGB"), candidate_labels=IMAGE_LABELS)

        if not isinstance(raw_predictions, list) or not raw_predictions:
            raise ValueError("No predictions returned by classifier.")

        predictions = sorted(raw_predictions, key=lambda item: float(item["score"]), reverse=True)

        all_predictions = [
            {"label": str(item["label"]), "confidence": round(float(item["score"]), 4)}
            for item in predictions
            if isinstance(item, dict) and "label" in item and "score" in item
        ]

        if not all_predictions:
            raise ValueError("Invalid prediction format returned.")

        top = all_predictions[0]
        top_label = top["label"]
        normalized = top_label.lower()

        photo_score = max(
            (item["confidence"] for item in all_predictions if KEYWORD_REAL_PHOTO in item["label"].lower()),
            default=0.0,
        )

        synthetic_score = max(
            (
                item["confidence"]
                for item in all_predictions
                if any(k in item["label"].lower() for k in SYNTHETIC_KEYWORDS)
            ),
            default=0.0,
        )

        if KEYWORD_REAL_PHOTO in normalized and photo_score >= settings.PHOTO_SCORE_THRESHOLD:
            status = "PHOTO_LIKE"
            reason = "The highest-scoring category is a real human photograph."
        elif "object with no person" in normalized:
            status = "NON_PHOTO"
            reason = "The image was classified as an object or scene without a person."
        elif any(term in normalized for term in SYNTHETIC_KEYWORDS):
            status = "NON_PHOTO"
            reason = "The highest-scoring category is an illustrated, diagrammatic, or computer-generated image."
        else:
            status = "NEEDS_REVIEW"
            reason = "The image type could not be determined confidently."

        display_label = friendly_image_type(top_label)

        return {
            "status": status,
            "top_class": top_label,
            "top_class_confidence": top["confidence"],
            "photo_score": round(photo_score, 4),
            "cartoon_score": round(synthetic_score, 4),
            "display_label": display_label,
            "model_executed": "OpenAI CLIP ViT-B/32",
            "all_predictions": all_predictions,
            "scores": {item["label"]: item["confidence"] for item in all_predictions},
            "reason": reason,
        }

    except Exception:
        logger.exception("Image classification failed")
        return {
            "status": "UNCERTAIN",
            "top_class": "Uncertain",
            "top_class_confidence": 0.0,
            "photo_score": 0.0,
            "cartoon_score": 0.0,
            "display_label": "Uncertain",
            "model_executed": "Heuristic CV Screener",
            "all_predictions": [],
            "scores": {},
            "reason": "Image classification failed due to an internal error.",
        }
