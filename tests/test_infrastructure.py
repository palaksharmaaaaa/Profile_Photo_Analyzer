"""Rate limiter, client addresses, logging, progress events, parallel execution and warm-up."""

import json
import logging
import threading
import time

import numpy as np
import pytest
from starlette.requests import Request

from app.api.limiter import RateDecision, SlidingWindowLimiter
from app.api.middleware import resolve_client_ip
from app.core.config import settings
from app.core.logging import JsonFormatter, RequestIdFilter, configure_logging, request_id_var
from app.services import model_manager, parallel
from app.services.progress import STAGES, stage_event
from tests.conftest import FakeSession

# ---- the sliding-window limiter -----------------------------------------------------------------------------------------


def test_limiter_allows_up_to_the_limit_then_refuses_with_a_retry_hint(monkeypatch):
    monkeypatch.setattr(settings.security, "rate_limit_per_minute", 2)
    limiter = SlidingWindowLimiter()
    assert limiter.check("a", 100.0) == RateDecision(True, 2, 1)
    assert limiter.check("a", 101.0).remaining == 0
    refused = limiter.check("a", 102.0)
    assert (refused.allowed, refused.remaining, refused.retry_after_seconds) == (False, 0, 60)
    assert limiter.check("b", 102.0).allowed  # another client is unaffected
    assert limiter.check("a", 161.5).allowed  # the window has moved on


def test_limiter_forgets_idle_clients_and_caps_the_number_it_tracks(monkeypatch):
    limiter = SlidingWindowLimiter()
    now = time.time()
    limiter._hits.update({"old": [now - 120.0], "recent": [now], "empty": []})
    limiter._prune(now)
    assert limiter.tracked() == {"recent"}

    monkeypatch.setattr(settings.security, "max_tracked_ips", 2)
    limiter._hits.update({"x": [now], "y": [now], "z": [now]})
    limiter._prune(now)
    assert len(limiter.tracked()) == 2

    limiter.clear()
    for index in range(105):
        limiter._hits[f"10.0.0.{index}"] = [now - 120.0]
    assert limiter.check("10.0.0.200", now).allowed  # many tracked clients trigger the clean-up
    assert limiter.tracked() == {"10.0.0.200"}


# ---- which address a request came from -------------------------------------------------------------------------------


def scope(peer=("127.0.0.1", 1), forwarded=None):
    headers = [(b"x-forwarded-for", forwarded)] if forwarded is not None else []
    result = {"type": "http", "headers": headers}
    if peer:
        result["client"] = peer
    return result


def test_forwarded_addresses_are_believed_only_from_trusted_proxies(monkeypatch):
    monkeypatch.setattr(settings.security, "trusted_proxies", ["127.0.0.1"])
    assert resolve_client_ip(scope(forwarded=b"203.0.113.50, 10.0.0.1")) == "203.0.113.50"
    assert resolve_client_ip(scope()) == "127.0.0.1"
    assert resolve_client_ip(scope(forwarded=b"   ")) == "127.0.0.1"
    assert resolve_client_ip(scope(peer=("8.8.8.8", 1), forwarded=b"1.2.3.4")) == "8.8.8.8"
    assert resolve_client_ip(scope(peer=None)) == "unknown"
    assert Request(scope()).client.host == "127.0.0.1"


# ---- logging ---------------------------------------------------------------------------------------------------------------


def record(message="hello", exc_info=None, **extra):
    logger = logging.getLogger("tests.logging")
    return logger.makeRecord("tests.logging", logging.WARNING, __file__, 1, message, (), exc_info, extra=extra)


def test_the_filter_stamps_the_current_request_id():
    token = request_id_var.set("req-123456")
    try:
        stamped = record()
        RequestIdFilter().filter(stamped)
    finally:
        request_id_var.reset(token)
    assert stamped.request_id == "req-123456"
    unstamped = record()
    RequestIdFilter().filter(unstamped)
    assert unstamped.request_id == "-"


def test_json_logs_carry_the_request_id_extras_and_tracebacks():
    entry = record("uploaded a.jpg", request_id="req-1", size=12)
    parsed = json.loads(JsonFormatter().format(entry))
    assert parsed["message"] == "uploaded a.jpg"
    assert parsed["level"] == "WARNING"
    assert parsed["request_id"] == "req-1"
    assert parsed["size"] == 12
    assert parsed["time"].endswith("+00:00")

    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        failing = record("failed", exc_info=sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(failing))["exception"]


@pytest.mark.parametrize("json_output", [False, True])
def test_logging_is_configured_once_for_either_format(json_output, capsys):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("debug", json_output)
        configure_logging("debug", json_output)
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
        token = request_id_var.set("req-777777")
        try:
            logging.getLogger("tests.logging").warning("visible")
        finally:
            request_id_var.reset(token)
        err = capsys.readouterr().err
        assert "req-777777" in err
        assert ("{" in err) is json_output
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ---- progress events -------------------------------------------------------------------------------------------------------


def test_stage_events_know_their_number_and_title():
    event = stage_event("detection", "running", "working", 12.345)
    assert (event.step_number, event.title, event.elapsed_ms) == (3, "Faces: detection and verification", 12.3)
    assert event.as_dict()["status"] == "running"
    assert [stage_event(step_id, "completed").step_number for step_id, _ in STAGES] == [1, 2, 3, 4]
    with pytest.raises(KeyError):
        stage_event("unknown", "running")


# ---- parallel execution --------------------------------------------------------------------------------------------------


def test_tasks_run_concurrently_and_come_back_in_order():
    barrier = threading.Barrier(2, timeout=5)

    def task(value):
        def run():
            barrier.wait()  # only passes if both tasks run at the same time
            return value

        return run

    assert parallel.run_all([task("a"), task("b")]) == ["a", "b"]
    assert parallel.run_pair(lambda: 1, lambda: "two") == (1, "two")


def test_tasks_run_one_after_the_other_when_parallelism_is_off_or_only_one_exists(monkeypatch):
    names = []

    def task():
        names.append(threading.current_thread().name)
        return 1

    assert parallel.run_all([task]) == [1]
    monkeypatch.setattr(settings.models, "parallel_inference", False)
    assert parallel.run_all([task, task]) == [1, 1]
    assert set(names) == {threading.current_thread().name}


def test_nested_calls_cannot_deadlock(monkeypatch):
    monkeypatch.setattr(settings.models, "inference_workers", 1)
    parallel.shutdown()

    def outer():
        return parallel.run_all([lambda: "x", lambda: "y"])  # runs inline inside a pool thread

    assert parallel.run_all([outer, outer]) == [["x", "y"], ["x", "y"]]
    parallel.shutdown()


def test_the_first_error_is_raised_after_every_task_finished():
    finished = []

    def slow():
        time.sleep(0.1)
        finished.append("slow")

    def broken():
        raise RuntimeError("first")

    with pytest.raises(RuntimeError, match="first"):
        parallel.run_all([broken, slow])
    assert finished == ["slow"]


def test_shutting_the_pool_down_twice_is_harmless():
    parallel.run_all([lambda: 1, lambda: 2])
    parallel.shutdown()
    parallel.shutdown()
    assert parallel.run_all([lambda: 3, lambda: 4]) == [3, 4]  # a new pool is created on demand
    parallel.shutdown()


# ---- model status and warm-up ------------------------------------------------------------------------------------------------


class Meta:
    def __init__(self, name, shape, kind="tensor(float)"):
        self.name, self.shape, self.type = name, shape, kind


class WarmSession(FakeSession):
    def __init__(self, inputs, fail=False):
        super().__init__(lambda names, feeds: self._answer(feeds))
        self.inputs, self.fail, self.fed = inputs, fail, None

    def _answer(self, feeds):
        self.fed = feeds
        if self.fail:
            raise RuntimeError("no inference")
        return [np.zeros(1)]

    def get_inputs(self):
        return self.inputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_dummy_inputs_replace_dynamic_dimensions():
    session = WarmSession([Meta("pixels", ["batch", 3, "h", 224]), Meta("ids", [1, 8], "tensor(int64)")])
    feeds = model_manager._dummy_input(session)
    assert feeds["pixels"].shape == (1, 3, model_manager.WARM_UP_SIDE, 224)
    assert feeds["pixels"].dtype == np.float32
    assert feeds["ids"].dtype == np.int64


def test_warm_up_reports_every_model(monkeypatch):
    sessions = {
        "scrfd": WarmSession([Meta("input", [1, 3, "h", "w"])]),
        "mivolo": WarmSession([Meta("faces_bodies", [1, 6, 384, 384])], fail=True),
        "yolox": None,
    }
    monkeypatch.setattr(model_manager, "get_session", lambda key: sessions.get(key))

    class Detector:
        def detect(self, image):
            return 1, None

    monkeypatch.setattr(model_manager, "get_yunet_detector", lambda width, height: Detector())
    report = model_manager.warm_up()
    assert report["scrfd"]["loaded"] is True
    assert report["scrfd"]["providers"] == ["CPUExecutionProvider"]
    assert report["scrfd"]["warm_up_ms"] >= 0
    assert report["mivolo"] == {"loaded": False, "warm_up_ms": None, "providers": ["CPUExecutionProvider"]}
    assert report["yolox"] == {"loaded": False, "warm_up_ms": None, "providers": []}
    assert report["yunet"]["providers"] == ["OpenCV DNN"]
    assert report["yunet"]["loaded"] is True


def test_warm_up_skips_a_disabled_model_and_reports_a_missing_yunet(monkeypatch):
    monkeypatch.setattr(settings.detection, "enable_scrfd", False)
    monkeypatch.setattr(model_manager, "get_session", lambda key: None)
    monkeypatch.setattr(model_manager, "get_yunet_detector", lambda width, height: None)
    report = model_manager.warm_up()
    assert "scrfd" not in report
    assert report["yunet"] == {"loaded": False, "warm_up_ms": None, "providers": []}


def test_status_reports_licensing_and_what_is_switched_off(models_dir, monkeypatch):
    status = model_manager.check_models_status()
    assert status["scrfd"]["commercial_use_cleared"] is False
    assert status["yunet"]["commercial_use_cleared"] is True
    assert status["label_bank"]["enabled"] is True
    assert all(entry["enabled"] for entry in status.values())
    assert len(model_manager.commercial_use_blockers()) == 2

    monkeypatch.setattr(settings.detection, "enable_scrfd", False)
    assert model_manager.check_models_status()["scrfd"]["enabled"] is False
    assert model_manager.commercial_use_blockers() == [model_manager.MODEL_SPECS["mivolo"].display_name]
