import io
import logging
import time
from pathlib import Path
from typing import Annotated, Optional, Tuple
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from app.core.config import settings
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import sniff_image_magic_bytes
from app.services.model_manager import check_models_status

logger = logging.getLogger(__name__)

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


@router.get("/", response_class=FileResponse)
def home():
    """Serve the single-page application frontend securely via FileResponse."""
    if INDEX_HTML_PATH.is_file():
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return HTMLResponse("<h1>Profile Photo Analyzer</h1><p>Frontend template not found.</p>")


@router.get("/health")
def health():
    """Health check endpoint reporting application and model readiness."""
    models_status = check_models_status()
    all_ready = all(info["ready"] for info in models_status.values())
    return {
        "status": "ready" if all_ready else "degraded",
        "version": settings.APP_VERSION,
        "models": models_status
    }


def _validate_upload_extension(filename: str) -> Optional[JSONResponse]:
    """Validate that filename has an allowed extension."""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in settings.ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": f"Only {', '.join(sorted(settings.ALLOWED_EXTENSIONS))} files are allowed."
            }
        )
    return None


def _read_and_validate_upload_bytes(file: UploadFile) -> Tuple[Optional[bytes], Optional[JSONResponse]]:
    """Read upload contents and validate file size limits."""
    contents = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if not contents:
        return None, JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "The uploaded file is empty."}
        )
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        max_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        return None, JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"accepted": False, "reason": f"Maximum upload size is {max_mb} MB."}
        )
    return contents, None


def _decode_and_validate_image(contents: bytes) -> Tuple[Optional[Image.Image], Optional[str], Optional[JSONResponse]]:
    """Verify magic bytes and decode PIL image with dimension limits."""
    try:
        format_detected = sniff_image_magic_bytes(contents)
    except ValueError as exc:
        logger.warning("Magic bytes validation failed: %s", exc)
        return None, None, JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": str(exc)}
        )

    try:
        Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
        image = Image.open(io.BytesIO(contents))
        if image.width > settings.MAX_IMAGE_DIMENSION or image.height > settings.MAX_IMAGE_DIMENSION:
            return None, None, JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "accepted": False,
                    "reason": f"Image dimension exceeds {settings.MAX_IMAGE_DIMENSION}px limit ({image.width}x{image.height})."
                }
            )
        if (image.width * image.height) > settings.MAX_IMAGE_PIXELS:
            return None, None, JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "accepted": False,
                    "reason": f"Image pixel count exceeds maximum permitted ({settings.MAX_IMAGE_PIXELS} pixels)."
                }
            )
        image.load()
        return image.convert("RGB"), format_detected, None
    except (OSError, ValueError) as exc:
        logger.warning("Failed to decode image: %s", exc)
        return None, None, JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"accepted": False, "reason": "The uploaded file is not a valid supported image."}
        )


@router.post("/analyze-profile-photo")
def analyze_profile_photo(file: Annotated[UploadFile, File()]):
    """
    Synchronous route handler executed on thread pool to avoid blocking async event loop.
    Validates file format, size, decodes image, and runs the end-to-end analyzer.
    """
    filename = file.filename or "upload.jpg"
    ext_err = _validate_upload_extension(filename)
    if ext_err is not None:
        return ext_err

    contents, size_err = _read_and_validate_upload_bytes(file)
    if size_err is not None or contents is None:
        return size_err

    start_time = time.perf_counter()
    image, format_detected, decode_err = _decode_and_validate_image(contents)
    if decode_err is not None or image is None or format_detected is None:
        return decode_err

    try:
        result = analyze_profile_photo_image(image, filename=filename, format_detected=format_detected)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        result["processing_time_ms"] = elapsed_ms
        if "technical_diagnostics" in result and result["technical_diagnostics"] is not None:
            result["technical_diagnostics"]["processing_time_ms"] = elapsed_ms
        return JSONResponse(content=result)
    except Exception:
        logger.exception("Analysis error processing uploaded image")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "accepted": False,
                "reason": "Analysis failed. Check server logs."
            }
        )
