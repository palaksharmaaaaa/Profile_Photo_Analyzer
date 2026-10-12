"""The model registry: where each ONNX model lives, integrity checks, lazy loading and downloading."""

import hashlib
import json
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import onnxruntime as ort

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ORT_THREADS = 4
MANIFEST_NAME = "manifest.json"
LABEL_BANK_NAME = "label_embeddings.npz"


@dataclass(frozen=True)
class ModelSpec:
    """A model the pipeline can use.

    ``url`` is set for models that can be downloaded as they are, pinned by ``sha256``. Models built
    locally (SigLIP2 and its label embeddings, MiVOLO, SCRFD) have ``url=None``: ``scripts/build_models.py``
    creates them and records their SHA-256 in ``models/manifest.json``, which is checked when they load.
    """

    key: str
    filename: str
    display_name: str
    role: str
    license: str
    url: str | None = None
    sha256: str | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "yunet": ModelSpec(
        key="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        display_name="OpenCV YuNet Face Detector (2023mar)",
        role="face candidate detection",
        license="MIT (OpenCV Zoo)",
        url="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "anime_face": ModelSpec(
        key="anime_face",
        filename="anime_face_v1.4_n.onnx",
        display_name="DeepGHS Anime Face Detector (v1.4n)",
        role="anime/cartoon face localisation",
        license="MIT (model card)",
        url="https://huggingface.co/deepghs/anime_face_detection/resolve/main/face_detect_v1.4_n/model.onnx",
        sha256="fd860b650a4377046842c3cd80d01b0b408bdfbdb4acee5759630f82c6ef04a9",
    ),
    "yolox": ModelSpec(
        key="yolox",
        filename="yolox_s.onnx",
        display_name="YOLOX-S (COCO)",
        role="animal and bird detection",
        license="Apache-2.0 (Megvii YOLOX)",
        url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
        sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
    "siglip_vision": ModelSpec(
        key="siglip_vision",
        filename="siglip2_base_vision_int8.onnx",
        display_name="SigLIP2 Base/16-224 vision tower (INT8)",
        role="real-human vs non-human face and image-type verification",
        license="Apache-2.0 (Google SigLIP2)",
    ),
    "scrfd": ModelSpec(
        key="scrfd",
        filename="scrfd_10g.onnx",
        display_name="SCRFD-10G Face Detector (InsightFace)",
        role="primary face candidate detection",
        license="InsightFace pretrained weights: non-commercial research only (see NOTICE.md)",
    ),
    "mivolo": ModelSpec(
        key="mivolo",
        filename="mivolo_v2_age_gender.onnx",
        display_name="MiVOLO v2 Age/Gender (face + body transformer)",
        role="age regression and gender classification",
        license="Apache-2.0 per model card; training data only partly disclosed (see NOTICE.md)",
    ),
}

_lock = threading.RLock()
_validation_cache: dict[str, tuple[tuple[int, int], bool]] = {}
_yunet_detector: cv2.FaceDetectorYN | None = None
_yunet_size: tuple[int, int] = (0, 0)
_sessions: dict[str, ort.InferenceSession] = {}


def model_path(key: str) -> Path:
    return settings.MODELS_DIR / MODEL_SPECS[key].filename


def _load_manifest() -> dict[str, dict]:
    path = settings.MODELS_DIR / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("files", {}) if isinstance(data, dict) else {}
    except OSError, ValueError:
        return {}


def expected_sha256(filename: str) -> str | None:
    """Pinned SHA-256 for a model file: from the registry, otherwise from the build manifest."""
    for spec in MODEL_SPECS.values():
        if spec.filename == filename and spec.sha256:
            return spec.sha256
    entry = _load_manifest().get(filename)
    return entry.get("sha256") if isinstance(entry, dict) else None


def _calculate_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_model_loadable(model_path_: Path) -> bool:
    """Fallback integrity check for files without a pinned hash: the file must parse as a model."""
    try:
        with open(model_path_, "rb") as f:
            header = f.read(32)
        if not (header and header[0] in (0x08, 0x12, 0x0A)):
            return False
        ort.InferenceSession(str(model_path_), providers=["CPUExecutionProvider"])
        return True
    except Exception:
        return False


def is_valid_model_file(
    path: Path,
    min_size: int = settings.MIN_MODEL_FILE_SIZE,
    expected_name: str | None = None,
) -> bool:
    """A model file is valid when it exists, is large enough, and matches its pinned hash (or loads)."""
    try:
        st = path.stat()
    except OSError:
        return False

    if not (path.is_file() and st.st_size >= min_size):
        return False

    cache_key = str(path.resolve())
    fingerprint = (st.st_mtime_ns, st.st_size)
    with _lock:
        cached = _validation_cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    pinned = expected_sha256(expected_name or path.name)
    if pinned:
        try:
            is_valid = _calculate_file_sha256(path) == pinned
        except OSError:
            is_valid = False
    else:
        is_valid = _verify_model_loadable(path)

    with _lock:
        _validation_cache[cache_key] = (fingerprint, is_valid)
    return is_valid


def _download_file_stream(download_url: str, temp_path: Path) -> None:
    """Download a binary payload in chunks to a temporary destination."""
    if not download_url.startswith("https://"):
        raise ValueError("Refusing to download a model over a non-HTTPS URL.")
    req = urllib.request.Request(download_url, headers={"User-Agent": f"ProfilePhotoAnalyzer/{settings.APP_VERSION}"})
    with (
        urllib.request.urlopen(req, timeout=settings.DOWNLOAD_TIMEOUT_SECONDS) as resp,  # noqa: S310
        open(temp_path, "wb") as f_out,
    ):
        while chunk := resp.read(1024 * 1024):
            f_out.write(chunk)


def ensure_model_file(path: Path, download_url: str | None, min_size: int = settings.MIN_MODEL_FILE_SIZE) -> bool:
    """Return True when a verified model file is in place, downloading it atomically if it is missing."""
    if is_valid_model_file(path, min_size):
        return True

    if not download_url:
        # Locally built models are never deleted automatically: a mismatch needs a human to look at it.
        logger.warning("Model %s is missing or failed verification; run scripts/build_models.py.", path.name)
        return False

    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        logger.info("Downloading %s from %s ...", path.name, download_url)
        _download_file_stream(download_url, temp_path)
        if temp_path.stat().st_size < min_size:
            raise ValueError("Downloaded model file is too small.")
        if not is_valid_model_file(temp_path, min_size=min_size, expected_name=path.name):
            raise ValueError(f"Downloaded file for {path.name} failed integrity verification.")
        os.replace(temp_path, path)
        return True
    except Exception as exc:
        logger.warning("Could not download model %s: %s", path.name, exc)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def check_models_status() -> dict[str, dict]:
    """Report readiness of every model, plus whether the pre-embedded label bank is present."""
    result: dict[str, dict] = {}
    for key, spec in MODEL_SPECS.items():
        path = model_path(key)
        ready = is_valid_model_file(path)
        result[key] = {
            "model_name": spec.display_name,
            "file": spec.filename,
            "role": spec.role,
            "license": spec.license,
            "ready": ready,
            "size_bytes": path.stat().st_size if ready else 0,
        }
    bank = settings.MODELS_DIR / LABEL_BANK_NAME
    result["label_bank"] = {
        "model_name": "SigLIP2 pre-computed prompt embeddings",
        "file": LABEL_BANK_NAME,
        "role": "zero-shot label vectors",
        "license": "derived from Apache-2.0 SigLIP2 text tower",
        "ready": bank.is_file() and bank.stat().st_size > 0,
        "size_bytes": bank.stat().st_size if bank.is_file() else 0,
    }
    return result


def _providers() -> list[str]:
    if settings.FORCE_CPU:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    return [*preferred, "CPUExecutionProvider"]


def get_session(key: str) -> ort.InferenceSession | None:
    """Lazily create (once, thread-safely) the ONNX Runtime session for a registered model."""
    with _lock:
        session = _sessions.get(key)
        if session is not None:
            return session

        spec = MODEL_SPECS[key]
        path = model_path(key)
        if not ensure_model_file(path, spec.url):
            return None
        try:
            options = ort.SessionOptions()
            # Measured: onnxruntime's "use every core" default was ~4x slower than 4 threads on a 14-core CPU.
            options.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS or min(
                DEFAULT_ORT_THREADS, os.cpu_count() or 1
            )
            session = ort.InferenceSession(str(path), sess_options=options, providers=_providers())
        except Exception as exc:
            logger.warning("Failed to initialise %s session: %s", spec.display_name, exc)
            return None
        _sessions[key] = session
        return session


def get_yunet_detector(width: int = 640, height: int = 640) -> cv2.FaceDetectorYN | None:
    """Return the shared YuNet detector sized for the current image (callers must hold ``yunet_lock``)."""
    global _yunet_detector, _yunet_size
    spec = MODEL_SPECS["yunet"]
    path = model_path("yunet")
    with _lock:
        if not ensure_model_file(path, spec.url):
            return None
        try:
            if _yunet_detector is None:
                _yunet_detector = cv2.FaceDetectorYN.create(
                    str(path),
                    "",
                    (width, height),
                    score_threshold=settings.HUMAN_FACE_CONF_THRESHOLD,
                    nms_threshold=settings.FACE_NMS_IOU,
                )
                _yunet_size = (width, height)
            elif _yunet_size != (width, height):
                _yunet_detector.setInputSize((width, height))
                _yunet_size = (width, height)
            return _yunet_detector
        except Exception as exc:
            logger.warning("Failed to initialise YuNet detector: %s", exc)
            return None


yunet_lock = threading.Lock()  # the YuNet detector holds mutable input-size state, so detection is serialised
