import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import cv2
import onnxruntime as ort

from app.core.config import settings

logger = logging.getLogger(__name__)

_yunet_detector: Optional[cv2.FaceDetectorYN] = None
_yunet_size: Tuple[int, int] = (0, 0)
_genderage_session: Optional[ort.InferenceSession] = None
_anime_session: Optional[ort.InferenceSession] = None
_yolo_session: Optional[ort.InferenceSession] = None


def ensure_model_file(model_path: Path, download_url: str) -> bool:
    """Ensure that the requested ONNX model file exists locally; download if missing."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return True

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading pre-trained model to {model_path} from {download_url}...")
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "ProfilePhotoAnalyzer/3.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(model_path, "wb") as f_out:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
        logger.info(f"Successfully downloaded {model_path.name} ({model_path.stat().st_size} bytes)")
        return True
    except Exception as exc:
        logger.warning(f"Could not download model {model_path.name} from {download_url}: {exc}")
        if model_path.exists() and model_path.stat().st_size == 0:
            try:
                model_path.unlink()
            except OSError:
                pass
        return False


def get_yunet_detector(width: int = 640, height: int = 640) -> Optional[cv2.FaceDetectorYN]:
    """Retrieve or lazily initialize the OpenCV YuNet face detector."""
    global _yunet_detector, _yunet_size
    path = settings.MODELS_DIR / settings.YUNET_MODEL_NAME
    if not ensure_model_file(path, settings.YUNET_URL):
        return None

    try:
        if _yunet_detector is None:
            _yunet_detector = cv2.FaceDetectorYN.create(
                str(path),
                "",
                (width, height),
                score_threshold=settings.HUMAN_FACE_CONF_THRESHOLD,
                nms_threshold=0.3
            )
            _yunet_size = (width, height)
        elif _yunet_size != (width, height):
            _yunet_detector.setInputSize((width, height))
            _yunet_size = (width, height)
        return _yunet_detector
    except Exception as exc:
        logger.warning(f"Failed to initialize YuNet detector: {exc}")
        return None


def get_genderage_session() -> Optional[ort.InferenceSession]:
    """Retrieve or lazily initialize the InsightFace GenderAge ONNX session."""
    global _genderage_session
    if _genderage_session is not None:
        return _genderage_session

    path = settings.MODELS_DIR / settings.GENDERAGE_MODEL_NAME
    if not ensure_model_file(path, settings.GENDERAGE_URL):
        return None

    try:
        _genderage_session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"]
        )
        return _genderage_session
    except Exception as exc:
        logger.warning(f"Failed to initialize GenderAge session: {exc}")
        return None


def get_anime_face_session() -> Optional[ort.InferenceSession]:
    """Retrieve or lazily initialize the Anime/Cartoon Face Detector ONNX session."""
    global _anime_session
    if _anime_session is not None:
        return _anime_session

    path = settings.MODELS_DIR / settings.ANIME_FACE_MODEL_NAME
    if not ensure_model_file(path, settings.ANIME_FACE_URL):
        return None

    try:
        _anime_session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"]
        )
        return _anime_session
    except Exception as exc:
        logger.warning(f"Failed to initialize Anime Face session: {exc}")
        return None


def get_yolo_session() -> Optional[ort.InferenceSession]:
    """Retrieve or lazily initialize the YOLOv8n ONNX session."""
    global _yolo_session
    if _yolo_session is not None:
        return _yolo_session

    path = settings.MODELS_DIR / settings.YOLO_MODEL_NAME
    if not ensure_model_file(path, settings.YOLO_URL):
        return None

    try:
        _yolo_session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"]
        )
        return _yolo_session
    except Exception as exc:
        logger.warning(f"Failed to initialize YOLOv8n session: {exc}")
        return None
