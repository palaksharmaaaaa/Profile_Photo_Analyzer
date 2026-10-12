"""Request rate limiting. The in-memory limiter is per process; with several processes limit at the edge instead."""

import threading
from dataclasses import dataclass
from typing import Protocol

from app.core.config import settings

WINDOW_SECONDS = 60.0
PRUNE_ABOVE = 100  # expired clients are dropped once more than this many are tracked


@dataclass(frozen=True)
class RateDecision:
    """The verdict on one request: whether it may proceed and what to tell the client."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int = 0


class RateLimiter(Protocol):
    """Anything that can say whether a client may make another request now."""

    def check(self, client: str, now: float) -> RateDecision: ...


class SlidingWindowLimiter:
    """At most ``security.rate_limit_per_minute`` requests per client in any 60-second window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def _prune(self, now: float) -> None:
        """Forget clients with no recent request, and the oldest ones beyond ``max_tracked_ips``."""
        window_start = now - WINDOW_SECONDS
        for client in [c for c, hits in self._hits.items() if not hits or hits[-1] <= window_start]:
            del self._hits[client]
        excess = len(self._hits) - settings.security.max_tracked_ips
        for client in list(self._hits)[: max(0, excess)]:
            del self._hits[client]

    def check(self, client: str, now: float) -> RateDecision:
        limit = settings.security.rate_limit_per_minute
        with self._lock:
            if len(self._hits) > PRUNE_ABOVE:
                self._prune(now)
            recent = [t for t in self._hits.get(client, []) if t > now - WINDOW_SECONDS]
            if len(recent) >= limit:
                self._hits[client] = recent
                return RateDecision(False, limit, 0, retry_after_seconds=int(WINDOW_SECONDS))
            recent.append(now)
            self._hits[client] = recent
            return RateDecision(True, limit, max(0, limit - len(recent)))

    def clear(self) -> None:
        with self._lock:
            self._hits.clear()

    def tracked(self) -> set[str]:
        with self._lock:
            return set(self._hits)


default_limiter = SlidingWindowLimiter()  # the limiter the application uses
