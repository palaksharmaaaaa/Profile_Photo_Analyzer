import logging
import os
import urllib.request
from pathlib import Path

import cv2
import onnxruntime as ort

from app.core.config import settings

logger = logging.getLogger(__name__)

_yunet_detector: cv2.FaceDetectorYN | None = None
_yunet_size: tuple[int, int] = (0, 0)
_genderage_session: ort.InferenceSession | None = None
_anime_session: ort.InferenceSession | None = None
_yolo_session: ort.InferenceSession | None = None


def is_valid_model_file(model_path: Path, min_size: int = settings.MIN_MODEL_FILE_SIZE) -> bool:
    """Check if model file exists, meets minimum size, and contains a valid ONNX binary protobuf header."""
    if not (model_path.is_file() and model_path.stat().st_size >= min_size):
        return False
    try:
        with open(model_path, "rb") as f:
            header = f.read(32)
        return bool(header and header[0] in (0x08, 0x12, 0x0A))
    except OSError:
        return False


def ensure_model_file(model_path: Path, download_url: str, min_size: int = settings.MIN_MODEL_FILE_SIZE) -> bool:
    """Ensure that the requested ONNX model file exists locally; download atomically if missing."""
    if is_valid_model_file(model_path, min_size):
        return True

    # Remove stale, truncated, or corrupted non-ONNX file if present
    if model_path.is_file() and not is_valid_model_file(model_path, min_size):
        try:
            model_path.unlink()
        except OSError:
            pass

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    try:
        logger.info(f"Downloading pre-trained model to {model_path} from {download_url}...")
        req = urllib.request.Request(download_url, headers={"User-Agent": "ProfilePhotoAnalyzer/3.0"})
        with (
            urllib.request.urlopen(req, timeout=30) as resp,
            open(temp_path, "wb") as f_out,
        ):
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)

        downloaded_size = temp_path.stat().st_size
        if downloaded_size < min_size:
            raise ValueError(f"Downloaded model file is too small ({downloaded_size} bytes < {min_size} bytes)")

        os.replace(temp_path, model_path)
        logger.info(f"Successfully downloaded {model_path.name} ({downloaded_size} bytes)")
        return True
    except Exception as exc:
        logger.warning(f"Could not download model {model_path.name} from {download_url}: {exc}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def check_models_status() -> dict:
    """Check readiness of all required local ONNX models."""
    models = {
        "yunet": {
            "name": settings.YUNET_MODEL_NAME,
            "path": settings.MODELS_DIR / settings.YUNET_MODEL_NAME,
        },
        "genderage": {
            "name": settings.GENDERAGE_MODEL_NAME,
            "path": settings.MODELS_DIR / settings.GENDERAGE_MODEL_NAME,
        },
        "anime_face": {
            "name": settings.ANIME_FACE_MODEL_NAME,
            "path": settings.MODELS_DIR / settings.ANIME_FACE_MODEL_NAME,
        },
        "yolov8n": {
            "name": settings.YOLO_MODEL_NAME,
            "path": settings.MODELS_DIR / settings.YOLO_MODEL_NAME,
        },
    }
    result = {}
    for key, info in models.items():
        p: Path = info["path"]
        ready = is_valid_model_file(p)
        size_bytes = p.stat().st_size if ready else 0
        result[key] = {
            "model_name": info["name"],
            "ready": ready,
            "size_bytes": size_bytes,
        }
    return result


def get_yunet_detector(width: int = 640, height: int = 640) -> cv2.FaceDetectorYN | None:
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
                nms_threshold=0.3,
            )
            _yunet_size = (width, height)
        elif _yunet_size != (width, height):
            _yunet_detector.setInputSize((width, height))
            _yunet_size = (width, height)
        return _yunet_detector
    except Exception as exc:
        logger.warning(f"Failed to initialize YuNet detector: {exc}")
        return None


def get_genderage_session() -> ort.InferenceSession | None:
    """Retrieve or lazily initialize the InsightFace GenderAge ONNX session."""
    global _genderage_session
    if _genderage_session is not None:
        return _genderage_session

    path = settings.MODELS_DIR / settings.GENDERAGE_MODEL_NAME
    if not ensure_model_file(path, settings.GENDERAGE_URL):
        return None

    try:
        _genderage_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _genderage_session
    except Exception as exc:
        logger.warning(f"Failed to initialize GenderAge session: {exc}")
        return None


def get_anime_face_session() -> ort.InferenceSession | None:
    """Retrieve or lazily initialize the Anime/Cartoon Face Detector ONNX session."""
    global _anime_session
    if _anime_session is not None:
        return _anime_session

    path = settings.MODELS_DIR / settings.ANIME_FACE_MODEL_NAME
    if not ensure_model_file(path, settings.ANIME_FACE_URL):
        return None

    try:
        _anime_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _anime_session
    except Exception as exc:
        logger.warning(f"Failed to initialize Anime Face session: {exc}")
        return None


def get_yolo_session() -> ort.InferenceSession | None:
    """Retrieve or lazily initialize the YOLOv8n ONNX session."""
    global _yolo_session
    if _yolo_session is not None:
        return _yolo_session

    path = settings.MODELS_DIR / settings.YOLO_MODEL_NAME
    if not ensure_model_file(path, settings.YOLO_URL):
        return None

    try:
        _yolo_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _yolo_session
    except Exception as exc:
        logger.warning(f"Failed to initialize YOLOv8n session: {exc}")
        return None
