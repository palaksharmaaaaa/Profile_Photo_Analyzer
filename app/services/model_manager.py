import hashlib
import logging
import os
import urllib.request
from pathlib import Path

import cv2
import onnxruntime as ort

from app.core.config import settings

logger = logging.getLogger(__name__)

MODEL_PINNED_HASHES: dict[str, str] = {
    settings.YUNET_MODEL_NAME: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    settings.GENDERAGE_MODEL_NAME: "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    settings.ANIME_FACE_MODEL_NAME: "fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9",
    settings.YOLO_MODEL_NAME: "dd48a79dd7fec8ca25fde4eca742ff7bca23b27e2e903eb23bc1d9f83a459bd2",
}

_validation_cache: dict[str, tuple[float, bool]] = {}
_yunet_detector: cv2.FaceDetectorYN | None = None
_yunet_size: tuple[int, int] = (0, 0)
_genderage_session: ort.InferenceSession | None = None
_anime_session: ort.InferenceSession | None = None
_yolo_session: ort.InferenceSession | None = None


def _calculate_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_model_loadable(model_path: Path) -> bool:
    """Verify that an ONNX model file can be parsed and loaded without protobuf corruption."""
    try:
        with open(model_path, "rb") as f:
            header = f.read(32)
        if not (header and header[0] in (0x08, 0x12, 0x0A)):
            return False

        if model_path.name == settings.YUNET_MODEL_NAME:
            detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320))
            return detector is not None
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return session is not None
    except Exception:
        return False


def is_valid_model_file(
    model_path: Path,
    min_size: int = settings.MIN_MODEL_FILE_SIZE,
    expected_name: str | None = None,
) -> bool:
    """
    Check if model file exists, meets minimum size, and passes cryptographic
    pinned hash verification or deep protobuf load integrity check.
    """
    try:
        st = model_path.stat()
    except OSError:
        return False

    if not (model_path.is_file() and st.st_size >= min_size):
        return False

    cache_key = str(model_path.resolve())
    mtime = st.st_mtime

    cached = _validation_cache.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    lookup_name = expected_name or model_path.name
    pinned_hash = MODEL_PINNED_HASHES.get(lookup_name)
    if pinned_hash:
        try:
            actual_hash = _calculate_file_sha256(model_path)
            is_valid = actual_hash == pinned_hash
        except OSError:
            is_valid = False
    else:
        is_valid = _verify_model_loadable(model_path)

    _validation_cache[cache_key] = (mtime, is_valid)
    return is_valid


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

        # Verify cryptographic hash and model loadability before moving into destination path
        if not is_valid_model_file(temp_path, min_size=min_size, expected_name=model_path.name):
            raise ValueError(
                f"Downloaded file for {model_path.name} failed cryptographic integrity or verification check."
            )

        os.replace(temp_path, model_path)
        logger.info(f"Successfully downloaded and verified {model_path.name} ({downloaded_size} bytes)")
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
