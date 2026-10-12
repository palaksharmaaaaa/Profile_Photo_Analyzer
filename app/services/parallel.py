"""Run independent model calls at the same time on a small shared thread pool."""

import atexit
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from app.core.config import settings

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_inside_pool = threading.local()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=settings.models.inference_workers, thread_name_prefix="inference")
            atexit.register(shutdown)
        return _pool


def shutdown() -> None:
    """Stop the pool (the next call creates a fresh one)."""
    global _pool
    with _lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _in_worker[T](task: Callable[[], T]) -> T:
    _inside_pool.active = True
    try:
        return task()
    finally:
        _inside_pool.active = False


def run_all[T](tasks: Sequence[Callable[[], T]]) -> list[T]:
    """
    Run the tasks and return their results in order; the first exception is raised once every task has finished.

    The tasks run concurrently when ``models.parallel_inference`` is on and there is more than one. A call made from
    inside a pool thread runs its tasks one after the other, so nested calls can never wait for each other.
    """
    if len(tasks) < 2 or not settings.models.parallel_inference or getattr(_inside_pool, "active", False):
        return [task() for task in tasks]
    futures = [_get_pool().submit(_in_worker, task) for task in tasks]
    results: list[T] = []
    first_error: BaseException | None = None
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as exc:  # noqa: BLE001 - re-raised below after every task has finished
            first_error = first_error or exc
    if first_error is not None:
        raise first_error
    return results


def run_pair[A, B](first: Callable[[], A], second: Callable[[], B]) -> tuple[A, B]:
    """``run_all`` for two tasks that return different types."""
    results = run_all([cast(Callable[[], object], first), cast(Callable[[], object], second)])
    return cast(A, results[0]), cast(B, results[1])
