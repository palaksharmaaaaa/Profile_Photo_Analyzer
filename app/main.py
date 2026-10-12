import logging
import time
from collections import defaultdict
from threading import RLock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_rate_limit_lock = RLock()
_request_records: dict[str, list[float]] = defaultdict(list)
_concurrency_lock = RLock()
_active_requests: int = 0


EXEMPT_PATHS: frozenset[str] = frozenset({"/", "/health", "/docs", "/openapi.json"})


class ProductionSecurityMiddleware(BaseHTTPMiddleware):
    """Enforce API authentication, rate limiting, and concurrency limits."""

    @staticmethod
    def _verify_api_key(request: Request) -> JSONResponse | None:
        """Validate API key from custom header or Bearer authorization."""
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key:
            auth_hdr = request.headers.get("Authorization", "")
            if auth_hdr.startswith("Bearer "):
                api_key = auth_hdr[7:].strip()
        if api_key != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: Invalid or missing API key."},
            )
        return None

    @staticmethod
    def _resolve_client_ip(request: Request) -> str:
        """Resolve client IP address respecting trusted reverse proxy headers."""
        peer_ip = request.client.host if request.client else "unknown"
        if peer_ip in settings.TRUSTED_PROXIES:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_part = forwarded.split(",")[0].strip()
                if client_part:
                    return client_part
        return peer_ip

    @staticmethod
    def _prune_rate_limit_records(now: float) -> None:
        """Prune expired rate limit entries to prevent unbounded memory growth."""
        window_start = now - 60.0
        expired_keys = [ip for ip, recs in _request_records.items() if not recs or recs[-1] <= window_start]
        for ip in expired_keys:
            del _request_records[ip]
        if len(_request_records) > settings.MAX_TRACKED_IPS:
            excess = len(_request_records) - settings.MAX_TRACKED_IPS
            for ip in list(_request_records.keys())[:excess]:
                del _request_records[ip]

    @staticmethod
    def _check_rate_limit(client_ip: str, now: float) -> tuple[JSONResponse | None, int]:
        """Enforce sliding-window rate limit per client IP with bounded cleanup."""
        with _rate_limit_lock:
            if len(_request_records) > 100:
                ProductionSecurityMiddleware._prune_rate_limit_records(now)
            window_start = now - 60.0
            timestamps = [t for t in _request_records[client_ip] if t > window_start]
            if len(timestamps) >= settings.RATE_LIMIT_PER_MINUTE:
                _request_records[client_ip] = timestamps
                return (
                    JSONResponse(
                        status_code=429,
                        content={"detail": "Too Many Requests: Rate limit exceeded."},
                        headers={"Retry-After": "60"},
                    ),
                    0,
                )
            timestamps.append(now)
            _request_records[client_ip] = timestamps
            remaining = max(0, settings.RATE_LIMIT_PER_MINUTE - len(timestamps))
            return None, remaining

    async def dispatch(self, request: Request, call_next):
        global _active_requests

        path = request.url.path
        is_exempt = path in EXEMPT_PATHS

        # 1. API key authentication check (if configured)
        if settings.API_KEY and not is_exempt:
            auth_error = self._verify_api_key(request)
            if auth_error:
                return auth_error

        # 2. Rate limiting (sliding window per client IP)
        client_ip = self._resolve_client_ip(request)
        now = time.time()
        if settings.RATE_LIMIT_ENABLED and not is_exempt:
            rate_error, remaining = self._check_rate_limit(client_ip, now)
            if rate_error:
                return rate_error
        else:
            remaining = settings.RATE_LIMIT_PER_MINUTE

        # 3. Concurrency guard
        if not is_exempt:
            with _concurrency_lock:
                if _active_requests >= settings.MAX_CONCURRENT_REQUESTS:
                    return JSONResponse(
                        status_code=503,
                        content={"detail": "Service Unavailable: Server concurrency limit reached."},
                    )
                _active_requests += 1

        try:
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
        finally:
            if not is_exempt:
                with _concurrency_lock:
                    _active_requests = max(0, _active_requests - 1)


def create_app() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
        description="Automated profile photo quality, facial presence, and suitability analyzer.",
    )

    application.add_middleware(ProductionSecurityMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()
