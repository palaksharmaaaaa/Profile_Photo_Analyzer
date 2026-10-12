"""
Asynchronous analyses: a bounded worker pool in front of the models.

A job is submitted with an already decoded image, runs on one of ``jobs.workers`` threads and reports its stages as it
goes. At most ``jobs.workers + jobs.queue_size`` jobs exist unfinished at any time: beyond that ``submit`` raises
``QueueFull`` instead of letting work pile up. Finished jobs are forgotten after ``jobs.result_ttl_seconds``.
This is in-process: to scale out, run several instances behind a load balancer or put an external queue here.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from PIL import Image

from app.core import metrics
from app.core.config import settings
from app.core.errors import ErrorCode
from app.schemas.analysis import JobState
from app.services.analyzer import analyze_profile_photo_image
from app.services.image_utils import DecodedImage
from app.services.progress import StageEvent
from app.services.types import AnalysisResult

logger = logging.getLogger(__name__)

Analyzer = Callable[..., AnalysisResult]


class QueueFull(Exception):
    """Every worker is busy and the waiting list is full."""


@dataclass
class Job:
    """One analysis: its state, the stages seen so far and, when finished, the result or the error."""

    job_id: str
    filename: str
    state: JobState = JobState.QUEUED
    events: list[StageEvent] = field(default_factory=list)
    result: AnalysisResult | None = None
    error: tuple[ErrorCode, str] | None = None
    finished_at: float | None = None
    _changed: threading.Condition = field(default_factory=threading.Condition, repr=False)

    @property
    def finished(self) -> bool:
        return self.state in (JobState.SUCCEEDED, JobState.FAILED)

    def add_event(self, event: StageEvent) -> None:
        with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    def finish(
        self, state: JobState, result: AnalysisResult | None = None, error: tuple[ErrorCode, str] | None = None
    ) -> None:
        with self._changed:
            self.state, self.result, self.error = state, result, error
            self.finished_at = time.monotonic()
            self._changed.notify_all()

    def start(self) -> None:
        with self._changed:
            self.state = JobState.RUNNING
            self._changed.notify_all()

    def wait_for_events(self, seen: int, timeout: float) -> tuple[list[StageEvent], bool]:
        """Block until there are events beyond the first ``seen`` or the job is finished (or ``timeout``)."""
        with self._changed:
            self._changed.wait_for(lambda: len(self.events) > seen or self.finished, timeout)
            return list(self.events[seen:]), self.finished


class JobManager:
    """Owns the worker pool and the table of jobs."""

    def __init__(self, analyzer: Analyzer | None = None) -> None:
        self._analyzer: Analyzer = analyzer or analyze_profile_photo_image
        self._pool = ThreadPoolExecutor(max_workers=settings.jobs.workers, thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        ttl = settings.jobs.result_ttl_seconds
        expired = [
            key for key, job in self._jobs.items() if job.finished_at is not None and now - job.finished_at > ttl
        ]
        for key in expired:
            del self._jobs[key]

    def _unfinished(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.finished)

    def submit(
        self, image: Image.Image, filename: str, file_format: str, meta: DecodedImage, render_annotation: bool = True
    ) -> Job:
        """Queue an analysis. Raises ``QueueFull`` when no worker is free and the waiting list is full."""
        with self._lock:
            self._purge(time.monotonic())
            if self._unfinished() >= settings.jobs.workers + settings.jobs.queue_size:
                raise QueueFull
            job = Job(job_id=uuid.uuid4().hex, filename=filename)
            self._jobs[job.job_id] = job
        metrics.JOBS.inc()
        self._pool.submit(self._run, job, image, file_format, meta, render_annotation)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge(time.monotonic())
            return self._jobs.get(job_id)

    def _run(self, job: Job, image: Image.Image, file_format: str, meta: DecodedImage, render_annotation: bool) -> None:
        job.start()
        record_stage = metrics.stage_reporter()

        def report(event: StageEvent) -> None:
            record_stage(event)
            job.add_event(event)

        try:
            result = self._analyzer(
                image,
                filename=job.filename,
                file_format=file_format,
                meta=meta,
                report=report,
                render_annotation=render_annotation,
            )
        except Exception:
            logger.exception("Analysis job %s failed", job.job_id)
            job.finish(JobState.FAILED, error=(ErrorCode.ANALYSIS_FAILED, "Analysis failed. Check server logs."))
            metrics.JOBS_SUBMITTED.labels(JobState.FAILED.value).inc()
        else:
            metrics.record_analysis(result)
            job.finish(JobState.SUCCEEDED, result=result)
            metrics.JOBS_SUBMITTED.labels(JobState.SUCCEEDED.value).inc()
        finally:
            metrics.JOBS.dec()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    """The process-wide job manager (created on first use)."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager


def reset_job_manager() -> None:
    """Stop and forget the process-wide job manager (used at shutdown and by tests)."""
    global _manager
    with _manager_lock:
        manager, _manager = _manager, None
    if manager is not None:
        manager.shutdown()
