"""
The ASGI middleware around every request: request id, security headers, API-key authentication, rate limiting,
a body-size cap, a concurrency guard for the synchronous analysis endpoint and request metrics.
"""

import hmac
import logging
import re
import time
import uuid

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.limiter import RateLimiter, default_limiter
from app.core import metrics
from app.core.config import settings
from app.core.errors import ErrorCode, error_response
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/", "/health", "/ready", "/v1/limits", "/docs", "/openapi.json", "/redoc"})
STATIC_PREFIX = "/static/"  # the page's own stylesheet and script: never authenticated or rate limited
UNLIMITED_PATHS = frozenset({"/metrics"})  # authenticated, but a scraper must not be rate limited
GUARDED_PATHS = frozenset({"/v1/analyze-profile-photo", "/analyze-profile-photo"})  # one request = one analysis
BODY_SLACK_BYTES = 1024 * 1024  # multipart framing around the file
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def resolve_client_ip(scope: Scope) -> str:
    """The caller's address; ``X-Forwarded-For`` is believed only when the direct peer is a trusted proxy."""
    client = scope.get("client")
    peer = client[0] if client else "unknown"
    if peer in settings.security.trusted_proxies:
        forwarded = Headers(scope=scope).get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return peer


def _api_key_from(headers: Headers) -> str | None:
    key = headers.get(settings.security.api_key_header)
    if not key:
        authorization = headers.get("authorization", "")
        if authorization.startswith("Bearer "):
            key = authorization[7:].strip()
    return key or None


def _api_key_is_valid(headers: Headers) -> bool:
    expected, supplied = settings.security.api_key, _api_key_from(headers)
    return bool(expected and supplied and hmac.compare_digest(supplied, expected))


def _too_large() -> JSONResponse:
    limit_mb = settings.upload.max_bytes // (1024 * 1024)
    return error_response(413, ErrorCode.FILE_TOO_LARGE, f"Maximum upload size is {limit_mb} MB.")


class RequestScope:
    """What is known about one request while it is served: its headers, answer state and the size of its body so far."""

    def __init__(self, scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        self.scope = scope
        self._receive = receive
        self._send = send
        self.extra = {"X-Request-ID": request_id, **SECURITY_HEADERS}
        self.status = 500
        self.response_started = False
        self.refused = False  # the body proved too large and the 413 has been sent: what the app sends now is dropped
        self._received = 0

    async def send(self, message: Message) -> None:
        """Pass a message on, adding the id and security headers to the response start."""
        if self.refused:
            return
        if message["type"] == "http.response.start":
            self.response_started = True
            self.status = message["status"]
            response_headers = MutableHeaders(scope=message)
            for name, value in self.extra.items():
                response_headers[name] = value
        await self._send(message)

    async def receive(self) -> Message:
        """``receive`` that answers 413 and ends the request once more than an upload may contain has arrived."""
        message = await self._receive()
        if message["type"] == "http.request" and not self.refused:
            self._received += len(message.get("body", b""))
            if self._received > settings.upload.max_bytes + BODY_SLACK_BYTES:
                await _too_large()(self.scope, self._receive, self.send)
                self.refused = True
                return {"type": "http.disconnect"}
        return message

    async def respond(self, response: JSONResponse) -> None:
        await response(self.scope, self._receive, self.send)


class SecurityMiddleware:
    """Pure ASGI middleware (no ``BaseHTTPMiddleware``), so streaming responses and cancellation behave."""

    def __init__(self, app: ASGIApp, limiter: RateLimiter | None = None) -> None:
        self.app = app
        self.limiter: RateLimiter = limiter or default_limiter
        self.active_analyses = 0

    @staticmethod
    def _request_id(headers: Headers) -> str:
        supplied = headers.get("x-request-id", "")
        return supplied if REQUEST_ID_PATTERN.match(supplied) else uuid.uuid4().hex

    def _refusal(self, scope: Scope, headers: Headers, extra: dict[str, str]) -> JSONResponse | None:
        """The response that stops this request before it reaches the application, if there is one."""
        path = scope["path"]
        if scope["method"] == "OPTIONS" or path in PUBLIC_PATHS or path.startswith(STATIC_PREFIX):
            return None  # a browser sends CORS preflight requests without credentials
        if settings.security.api_key and not _api_key_is_valid(headers):
            return error_response(401, ErrorCode.UNAUTHORIZED, "Unauthorized: invalid or missing API key.")

        if settings.security.rate_limit_enabled and path not in UNLIMITED_PATHS:
            decision = self.limiter.check(resolve_client_ip(scope), time.time())
            extra["X-RateLimit-Limit"] = str(decision.limit)
            extra["X-RateLimit-Remaining"] = str(decision.remaining)
            if not decision.allowed:
                return error_response(
                    429,
                    ErrorCode.RATE_LIMITED,
                    "Too many requests: the rate limit was exceeded.",
                    {"Retry-After": str(decision.retry_after_seconds)},
                )
        if scope["method"] == "POST":
            declared = headers.get("content-length", "")
            if declared.isdigit() and int(declared) > settings.upload.max_bytes + BODY_SLACK_BYTES:
                return _too_large()
        return None

    def _admit(self, scope: Scope) -> tuple[JSONResponse | None, bool]:
        """Count a synchronous analysis; a refusal when too many run at once. Other requests are not counted."""
        if scope["method"] != "POST" or scope["path"] not in GUARDED_PATHS:
            return None, False
        if self.active_analyses >= settings.security.max_concurrent_requests:
            return error_response(503, ErrorCode.BUSY, "The server is busy: the concurrency limit was reached."), False
        self.active_analyses += 1
        return None, True

    async def _serve(self, request: RequestScope, headers: Headers) -> bool:
        """Answer or pass on the request; True when it was counted as a running analysis."""
        refusal = self._refusal(request.scope, headers, request.extra)
        counted = False
        if refusal is None:
            refusal, counted = self._admit(request.scope)
        if refusal is not None:
            await request.respond(refusal)
            return False
        try:
            await self.app(request.scope, request.receive, request.send)
        except Exception:
            logger.exception("Unhandled error in %s %s", request.scope["method"], request.scope["path"])
            if not request.response_started:
                await request.respond(
                    error_response(500, ErrorCode.INTERNAL_ERROR, "Something went wrong. Check the server logs.")
                )
        return counted

    @staticmethod
    def _record(request: RequestScope, started: float) -> None:
        scope = request.scope
        route = getattr(scope.get("route"), "path", None)
        label = route or ("/static" if scope["path"].startswith(STATIC_PREFIX) else "unmatched")
        metrics.HTTP_REQUESTS.labels(scope["method"], label, str(request.status)).inc()
        metrics.HTTP_SECONDS.labels(scope["method"], label).observe(time.perf_counter() - started)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = self._request_id(headers)
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        request = RequestScope(scope, receive, send, request_id)
        counted = False
        try:
            counted = await self._serve(request, headers)
        finally:
            if counted:
                self.active_analyses -= 1
            self._record(request, started)
            request_id_var.reset(token)
