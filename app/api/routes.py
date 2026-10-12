import io
import logging
import time
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from app.core.config import settings
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import sniff_image_magic_bytes

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
    """Health check endpoint."""
    return {"status": "running", "version": settings.APP_VERSION}


@router.post("/analyze-profile-photo")
def analyze_profile_photo(file: Annotated[UploadFile, File()]):
    """
    Synchronous route handler executed on thread pool to avoid blocking async event loop.
    Validates file format, size, decodes image, and runs the end-to-end analyzer.
    """
    filename = file.filename or "upload.jpg"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in settings.ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": f"Only {', '.join(sorted(settings.ALLOWED_EXTENSIONS))} files are allowed."
            }
        )

    # Read uploaded bytes with size limit guard
    contents = file.file.read(settings.MAX_UPLOAD_BYTES + 1)

    if not contents:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "The uploaded file is empty."
            }
        )

    if len(contents) > settings.MAX_UPLOAD_BYTES:
        max_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        return JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={
                "accepted": False,
                "reason": f"Maximum upload size is {max_mb} MB."
            }
        )

    start_time = time.perf_counter()

    try:
        format_detected = sniff_image_magic_bytes(contents)
    except ValueError as exc:
        logger.warning("Magic bytes validation failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": str(exc)
            }
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = image.convert("RGB")
    except (OSError, ValueError) as exc:
        logger.warning("Failed to decode image: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "The uploaded file is not a valid supported image."
            }
        )

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
