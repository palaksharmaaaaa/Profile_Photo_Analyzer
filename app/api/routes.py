"""
The HTTP API.

* ``POST /v1/analyze-profile-photo``  analyse a photo and answer with the result (the legacy path without ``/v1`` is
  kept, marked deprecated)
* ``POST /v1/jobs``                   start an analysis and follow it: ``GET /v1/jobs/{id}`` and the event stream
  ``GET /v1/jobs/{id}/events``
* ``GET /health`` ``GET /ready`` ``GET /metrics``  liveness with model files, readiness after warm-up, Prometheus
* ``GET /``                           the single-page frontend
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from app.core import metrics
from app.core.config import settings
from app.core.errors import ApiError, ErrorCode
from app.schemas.analysis import (
    AnalysisResponse,
    ErrorResponse,
    JobCreated,
    JobStatus,
    StageEventModel,
)
from app.schemas.ops import HealthResponse, Limits, ReadyResponse
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import SUPPORTED_MIME_TYPES, DecodedImage, InvalidImageError, decode_image_bytes
from app.services.jobs import Job, QueueFull, get_job_manager
from app.services.model_manager import check_models_status, commercial_use_blockers
from app.services.types import AnalysisResult

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"
PAGE_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "Cache-Control": "no-cache",
}
EVENT_WAIT_SECONDS = 15.0  # how long the event stream waits for news before it sends a keep-alive comment
MAX_FILENAME_LENGTH = 255
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status_code: {"model": ErrorResponse} for status_code in (400, 401, 413, 422, 429, 500, 503)
}

router = APIRouter()
v1 = APIRouter(prefix="/v1")


# --------------------------------------------------------------------------------------
# Reading an upload
# --------------------------------------------------------------------------------------


def _clean_filename(raw: str | None) -> str:
    """The upload's base name without path parts or control characters, at most 255 characters."""
    name = CONTROL_CHARACTERS.sub("", (raw or "").replace("\\", "/").rsplit("/", 1)[-1]).strip()
    return name[:MAX_FILENAME_LENGTH] or "upload"


def _read_upload(file: UploadFile) -> bytes:
    contents = file.file.read(settings.upload.max_bytes + 1)
    if not contents:
        raise ApiError(status.HTTP_400_BAD_REQUEST, ErrorCode.EMPTY_FILE, "The uploaded file is empty.")
    if len(contents) > settings.upload.max_bytes:
        limit_mb = settings.upload.max_bytes // (1024 * 1024)
        raise ApiError(
            status.HTTP_413_CONTENT_TOO_LARGE, ErrorCode.FILE_TOO_LARGE, f"Maximum upload size is {limit_mb} MB."
        )
    return contents


def _decode_upload(file: UploadFile) -> tuple[str, DecodedImage]:
    """Validate and decode an upload (content, not file name, decides what is acceptable)."""
    filename = _clean_filename(file.filename)
    try:
        return filename, decode_image_bytes(_read_upload(file))
    except InvalidImageError as exc:
        logger.warning("Rejected upload: %s", exc.reason)
        metrics.REJECTED_UPLOADS.labels(ErrorCode.INVALID_IMAGE.value).inc()
        raise ApiError(status.HTTP_400_BAD_REQUEST, ErrorCode.INVALID_IMAGE, exc.reason) from exc
    except ApiError as exc:
        metrics.REJECTED_UPLOADS.labels(exc.code.value).inc()
        raise


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def _analyze(file: UploadFile, annotated_image: bool) -> AnalysisResult:
    filename, decoded = _decode_upload(file)
    try:
        result = analyze_profile_photo_image(
            decoded.image,
            filename=filename,
            file_format=decoded.file_format,
            meta=decoded,
            report=metrics.stage_reporter(),
            render_annotation=annotated_image,
        )
    except Exception as exc:
        logger.exception("Analysis error processing uploaded image")
        raise ApiError(
            status.HTTP_500_INTERNAL_SERVER_ERROR, ErrorCode.ANALYSIS_FAILED, "Analysis failed. Check server logs."
        ) from exc
    metrics.record_analysis(result)
    return result


AnnotatedQuery = Annotated[bool, Query(description="Include the annotated image (the largest part of the answer).")]


@v1.post(
    "/analyze-profile-photo",
    response_model=AnalysisResponse,
    responses=ERROR_RESPONSES,
    summary="Analyse a profile photo and answer with the result",
)
def analyze_profile_photo(
    file: Annotated[UploadFile, File()], annotated_image: AnnotatedQuery = True
) -> AnalysisResult:
    """
    A synchronous handler (it runs in the threadpool, so inference never blocks the event loop): validates the
    upload, decodes it (EXIF-upright, alpha-flattened, size-capped) and runs the analyzer.
    """
    return _analyze(file, annotated_image)


@router.post(
    "/analyze-profile-photo",
    response_model=AnalysisResponse,
    responses=ERROR_RESPONSES,
    deprecated=True,
    summary="Deprecated alias of POST /v1/analyze-profile-photo",
)
def analyze_profile_photo_legacy(
    file: Annotated[UploadFile, File()], response: Response, annotated_image: AnnotatedQuery = True
) -> AnalysisResult:
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</v1/analyze-profile-photo>; rel="successor-version"'
    return _analyze(file, annotated_image)


# --------------------------------------------------------------------------------------
# Jobs: start an analysis, follow its stages, collect the result
# --------------------------------------------------------------------------------------


def _job_status(job: Job) -> JobStatus:
    error = None
    if job.error is not None:
        error = ErrorResponse(code=job.error[0].value, reason=job.error[1])
    return JobStatus(
        job_id=job.job_id,
        status=job.state,
        filename=job.filename,
        stages=[StageEventModel.model_validate(event.as_dict()) for event in job.events],
        result=AnalysisResponse.model_validate(job.result) if job.result is not None else None,
        error=error,
    )


def _find_job(job_id: str) -> Job:
    job = get_job_manager().get(job_id)
    if job is None:
        raise ApiError(
            status.HTTP_404_NOT_FOUND, ErrorCode.JOB_NOT_FOUND, "There is no such job (it may have expired)."
        )
    return job


@v1.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreated,
    responses=ERROR_RESPONSES,
    summary="Start an analysis and follow it with the returned links",
)
def create_job(file: Annotated[UploadFile, File()], annotated_image: AnnotatedQuery = True) -> JobCreated:
    filename, decoded = _decode_upload(file)
    try:
        job = get_job_manager().submit(decoded.image, filename, decoded.file_format, decoded, annotated_image)
    except QueueFull as exc:
        metrics.REJECTED_UPLOADS.labels(ErrorCode.QUEUE_FULL.value).inc()
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCode.QUEUE_FULL,
            "Every worker is busy and the waiting list is full. Try again shortly.",
            {"Retry-After": "5"},
        ) from exc
    return JobCreated(
        job_id=job.job_id,
        status=job.state,
        status_url=f"/v1/jobs/{job.job_id}",
        events_url=f"/v1/jobs/{job.job_id}/events",
    )


@v1.get(
    "/jobs/{job_id}", response_model=JobStatus, responses=ERROR_RESPONSES, summary="State, stages and result of a job"
)
def get_job(job_id: str) -> JobStatus:
    return _job_status(_find_job(job_id))


def _sse(event: str, data: object, event_id: int | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {json.dumps(data)}\n\n"


async def _event_stream(job: Job) -> AsyncIterator[str]:
    """Server-sent events: a ``stage`` event per change, then one ``done`` event with the final state."""
    seen = 0
    while True:
        events, finished = await asyncio.to_thread(job.wait_for_events, seen, EVENT_WAIT_SECONDS)
        for event in events:
            seen += 1
            yield _sse("stage", event.as_dict(), seen)
        if finished and not events:
            yield _sse("done", {"status": job.state.value})
            return
        if not events:
            yield ": keep-alive\n\n"


@v1.get("/jobs/{job_id}/events", responses={200: {"content": {"text/event-stream": {}}}, 404: {"model": ErrorResponse}})
async def job_events(job_id: str) -> StreamingResponse:
    """The stages of a job as server-sent events, ending with a ``done`` event."""
    job = _find_job(job_id)
    return StreamingResponse(
        _event_stream(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@v1.get("/limits", response_model=Limits, summary="What the service accepts")
def limits() -> Limits:
    upload = settings.upload
    return Limits(
        max_upload_bytes=upload.max_bytes,
        max_pixels=upload.max_pixels,
        max_dimension=upload.max_dimension,
        working_max_side=upload.working_max_side,
        formats=list(SUPPORTED_MIME_TYPES),
    )


# --------------------------------------------------------------------------------------
# Operations and the page
# --------------------------------------------------------------------------------------


@router.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(INDEX_HTML_PATH, media_type="text/html", headers=PAGE_SECURITY_HEADERS)


@router.get("/health", response_model=HealthResponse, summary="Liveness, with the state of every model file")
def health() -> dict[str, object]:
    """Report application version and the readiness of every model (nothing is assumed ready)."""
    models = check_models_status()
    all_ready = all(info["ready"] for info in models.values() if info["enabled"])
    blockers = commercial_use_blockers()
    return {
        "status": "ready" if all_ready else "degraded",
        "version": settings.app_version,
        "models": models,
        "licensing": {"commercial_use_cleared": not blockers, "blocking_models": blockers},
    }


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
    summary="Readiness: every enabled model is loaded and has answered an inference",
)
def ready(request: Request, response: Response) -> dict[str, object]:
    warm_up = getattr(request.app.state, "warm_up", None)
    if warm_up is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "warming_up", "models": {}}
    models_ok = all(entry["loaded"] for entry in warm_up.values())
    if not models_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if models_ok else "not_ready", "models": warm_up}


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")
