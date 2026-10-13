"""One error shape for the whole API: ``{"accepted": false, "code": ..., "reason": ..., "request_id": ...}``."""

from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var


class ErrorCode(StrEnum):
    """Machine-readable reasons; ``reason`` carries the sentence for people."""

    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    INVALID_IMAGE = "invalid_image"
    ANALYSIS_FAILED = "analysis_failed"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    BUSY = "busy"
    QUEUE_FULL = "queue_full"
    JOB_NOT_FOUND = "job_not_found"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"


class ApiError(Exception):
    """An error the API reports to the client with a status code and a stable code."""

    def __init__(self, status_code: int, code: ErrorCode, reason: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.code = code
        self.reason = reason
        self.headers = headers or {}


def error_response(
    status_code: int, code: ErrorCode, reason: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"accepted": False, "code": code.value, "reason": reason, "request_id": request_id_var.get()},
        headers=headers,
    )


_HTTP_CODES = {
    404: (ErrorCode.NOT_FOUND, "The requested resource does not exist."),
    405: (ErrorCode.METHOD_NOT_ALLOWED, "This method is not allowed for the requested resource."),
}


def _handle_api_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.reason, exc.headers)


def _handle_http_exception(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code, reason = _HTTP_CODES.get(exc.status_code, (ErrorCode.INTERNAL_ERROR, str(exc.detail)))
    return error_response(exc.status_code, code, reason)


def _handle_validation_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    fields = ", ".join(
        sorted({".".join(str(part) for part in error["loc"][1:]) or "request" for error in exc.errors()})
    )
    return error_response(422, ErrorCode.VALIDATION_ERROR, f"The request is not valid: {fields}.")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
