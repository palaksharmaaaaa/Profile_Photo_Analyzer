"""Prometheus metrics, kept in their own registry and exposed at ``GET /metrics``."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from app.services.progress import COMPLETED, FAILED, ProgressReporter, StageEvent
from app.services.types import AnalysisResult

registry = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "ppa_http_requests_total",
    "HTTP requests by method, route and status.",
    ["method", "route", "status"],
    registry=registry,
)
HTTP_SECONDS = Histogram(
    "ppa_http_request_seconds",
    "HTTP request duration in seconds.",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=registry,
)
ANALYSES = Counter("ppa_analyses_total", "Finished analyses by outcome status.", ["status"], registry=registry)
STAGE_SECONDS = Histogram(
    "ppa_stage_seconds",
    "Duration of each analysis stage in seconds.",
    ["stage"],
    buckets=(0.005, 0.02, 0.1, 0.25, 0.5, 1, 2, 4, 8),
    registry=registry,
)
REJECTED_UPLOADS = Counter(
    "ppa_rejected_uploads_total", "Uploads refused before analysis, by code.", ["code"], registry=registry
)
JOBS = Gauge("ppa_jobs_active", "Jobs that are queued or running.", registry=registry)
JOBS_SUBMITTED = Counter("ppa_jobs_total", "Jobs by final state.", ["state"], registry=registry)


def stage_reporter() -> ProgressReporter:
    """A progress callback that records how long every finished stage took."""

    def record(event: StageEvent) -> None:
        if event.status in (COMPLETED, FAILED) and event.elapsed_ms > 0:
            STAGE_SECONDS.labels(event.step_id).observe(event.elapsed_ms / 1000)

    return record


def record_analysis(result: AnalysisResult) -> None:
    ANALYSES.labels(str(result["status"])).inc()


def render() -> bytes:
    return generate_latest(registry)
