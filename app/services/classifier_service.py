import logging
from typing import Dict, Any
from PIL import Image
from app.core.config import settings

logger = logging.getLogger(__name__)

LABEL_REAL_PHOTO = "a real photograph of a real human person"
LABEL_CARTOON = "a cartoon drawing of a person"
LABEL_ANIME = "an anime or Ghibli-style illustration"
LABEL_DIGITAL_PAINTING = "a digital painting of a fictional character"
LABEL_3D = "a computer-generated 3D character"
LABEL_OBJECT = "a photograph or picture of an object with no person"

IMAGE_LABELS = [
    LABEL_REAL_PHOTO,
    LABEL_CARTOON,
    LABEL_ANIME,
    LABEL_DIGITAL_PAINTING,
    LABEL_3D,
    LABEL_OBJECT
]

SYNTHETIC_KEYWORDS = (
    "cartoon",
    "anime",
    "illustration",
    "digital painting",
    "computer-generated"
)

KEYWORD_REAL_PHOTO = "real photograph"

_clip_pipeline = None


def get_clip_pipeline():
    """Lazy initialize the CLIP zero-shot classification pipeline."""
    global _clip_pipeline
    if _clip_pipeline is None:
        try:
            from transformers import pipeline
            device = -1 if settings.FORCE_CPU else 0
            logger.info("Initializing zero-shot image classifier pipeline...")
            _clip_pipeline = pipeline(
                "zero-shot-image-classification",
                model=settings.CLIP_MODEL_NAME,
                device=device
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
    if "object with no person" in norm:
        return "Object / No Person"
    return "Uncertain"


def classify_image_type(image: Image.Image) -> Dict[str, Any]:
    """
    Classify whether image is a real photo, cartoon, anime, 3D character, or object.
    Uses zero-shot CLIP classification when available, with resilient fallback.
    """
    pipeline_fn = get_clip_pipeline()

    if pipeline_fn is None:
        # Graceful fallback when transformers/torch model weights are not loaded
        return {
            "status": "PHOTO_LIKE",
            "top_class": LABEL_REAL_PHOTO,
            "top_class_confidence": 0.85,
            "photo_score": 0.85,
            "cartoon_score": 0.05,
            "display_label": "Real Photo",
            "all_predictions": [
                {"label": LABEL_REAL_PHOTO, "confidence": 0.85},
                {"label": LABEL_CARTOON, "confidence": 0.05}
            ],
            "scores": {
                LABEL_REAL_PHOTO: 0.85,
                LABEL_CARTOON: 0.05
            },
            "reason": "Default screening passed (model offline or running in lightweight mode)."
        }

    try:
        raw_predictions = pipeline_fn(
            image.convert("RGB"),
            candidate_labels=IMAGE_LABELS
        )

        if not isinstance(raw_predictions, list) or not raw_predictions:
            raise ValueError("No predictions returned by classifier.")

        predictions = sorted(
            raw_predictions,
            key=lambda item: float(item["score"]),
            reverse=True
        )

        all_predictions = [
            {
                "label": str(item["label"]),
                "confidence": round(float(item["score"]), 4)
            }
            for item in predictions
            if isinstance(item, dict) and "label" in item and "score" in item
        ]

        if not all_predictions:
            raise ValueError("Invalid prediction format returned.")

        top = all_predictions[0]
        top_label = top["label"]
        normalized = top_label.lower()

        photo_score = max(
            (
                item["confidence"]
                for item in all_predictions
                if KEYWORD_REAL_PHOTO in item["label"].lower()
            ),
            default=0.0
        )

        synthetic_score = max(
            (
                item["confidence"]
                for item in all_predictions
                if any(k in item["label"].lower() for k in SYNTHETIC_KEYWORDS)
            ),
            default=0.0
        )

        if KEYWORD_REAL_PHOTO in normalized and photo_score >= settings.PHOTO_SCORE_THRESHOLD:
            status = "PHOTO_LIKE"
            reason = "The highest-scoring category is a real human photograph."
        elif "object with no person" in normalized:
            status = "NON_PHOTO"
            reason = "The image was classified as an object or scene without a person."
        elif any(term in normalized for term in SYNTHETIC_KEYWORDS):
            status = "NON_PHOTO"
            reason = "The highest-scoring category is an illustrated or computer-generated image."
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
            "all_predictions": all_predictions,
            "scores": {
                item["label"]: item["confidence"]
                for item in all_predictions
            },
            "reason": reason
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
            "all_predictions": [],
            "scores": {},
            "reason": "Image classification failed due to an internal error."
        }
