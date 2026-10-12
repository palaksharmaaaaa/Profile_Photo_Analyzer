import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.core.config import settings
from app.schemas.analysis import AnalysisResponse, ErrorResponse
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import InvalidImageError, decode_image_bytes
from app.services.model_manager import check_models_status

logger = logging.getLogger(__name__)

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"


def _error(status_code: int, reason: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"accepted": False, "reason": reason})


@router.get("/", response_class=FileResponse)
def home():
    """Serve the single-page frontend."""
    if INDEX_HTML_PATH.is_file():
        return FileResponse(INDEX_HTML_PATH, media_type="text/html")
    return HTMLResponse("<h1>Profile Photo Analyzer</h1><p>Frontend template not found.</p>")


@router.get("/health")
def health():
    """Report application version and the readiness of every model (nothing is assumed ready)."""
    models_status = check_models_status()
    all_ready = all(info["ready"] for info in models_status.values())
    return {
        "status": "ready" if all_ready else "degraded",
        "version": settings.APP_VERSION,
        "models": models_status,
    }


def _validate_upload_extension(filename: str) -> JSONResponse | None:
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in settings.ALLOWED_EXTENSIONS:
        return _error(
            status.HTTP_400_BAD_REQUEST,
            f"Only {', '.join(sorted(settings.ALLOWED_EXTENSIONS))} files are allowed.",
        )
    return None


def _read_and_validate_upload_bytes(file: UploadFile) -> tuple[bytes | None, JSONResponse | None]:
    contents = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if not contents:
        return None, _error(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        max_mb = settings.MAX_UPLOAD_BYTES // (1024 * 1024)
        return None, _error(status.HTTP_413_CONTENT_TOO_LARGE, f"Maximum upload size is {max_mb} MB.")
    return contents, None


@router.post(
    "/analyze-profile-photo",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def analyze_profile_photo(file: Annotated[UploadFile, File()]):
    """
    Synchronous handler (runs in the FastAPI threadpool so inference never blocks the event loop).

    Validates the upload, decodes it (EXIF-upright, alpha-flattened, size-capped) and runs the analyzer.
    """
    filename = file.filename or "upload.jpg"
    ext_err = _validate_upload_extension(filename)
    if ext_err is not None:
        return ext_err

    contents, size_err = _read_and_validate_upload_bytes(file)
    if size_err is not None or contents is None:
        return size_err

    try:
        decoded = decode_image_bytes(contents)
    except InvalidImageError as exc:
        logger.warning("Rejected upload %r: %s", filename, exc.reason)
        return _error(status.HTTP_400_BAD_REQUEST, exc.reason)

    try:
        return analyze_profile_photo_image(
            decoded.image, filename=filename, file_format=decoded.file_format, meta=decoded
        )
    except Exception:
        logger.exception("Analysis error processing uploaded image")
        return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Analysis failed. Check server logs.")
