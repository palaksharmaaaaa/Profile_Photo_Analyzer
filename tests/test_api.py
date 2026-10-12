import io
import json
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from app.api import routes
from app.api.limiter import default_limiter
from app.core.config import Settings, settings
from app.main import create_app
from app.schemas.analysis import AnalysisResponse
from app.services import jobs as jobs_module
from app.services.analyzer import analyze_profile_photo_image
from tests.test_analyzer import HUMAN, PHOTO, adult, detected

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_state():
    default_limiter.clear()
    jobs_module.reset_job_manager()
    yield
    jobs_module.reset_job_manager()


def jpeg_bytes(size=(64, 64), color="grey") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def valid_payload(**kwargs) -> dict:
    """A complete, schema-valid response (the no-information path needs no model)."""
    return analyze_profile_photo_image(Image.new("RGB", (64, 64), "white"), **kwargs)


@pytest.fixture
def stub_analysis():
    with patch.object(routes, "analyze_profile_photo_image", side_effect=lambda *a, **k: valid_payload()) as stub:
        yield stub


def post(name="p.jpg", data=None, headers=None, ctype="image/jpeg", path="/v1/analyze-profile-photo", **params):
    return client.post(
        path,
        headers=headers or {},
        params=params,
        files={"file": (name, jpeg_bytes() if data is None else data, ctype)},
    )


def assert_error(response, status, code):
    assert response.status_code == status
    body = response.json()
    assert body["accepted"] is False
    assert body["code"] == code
    assert body["reason"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    return body


# ---- the page ------------------------------------------------------------------------------------------------------


def test_home_serves_the_page_with_a_content_security_policy():
    response = client.get("/")
    assert response.status_code == 200
    assert "Profile Photo Analyzer" in response.text
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_starting_without_the_page_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.INDEX_HTML_PATH", tmp_path / "missing.html")
    with pytest.raises(RuntimeError, match="frontend page is missing"):
        create_app()


def test_page_assets_are_served_and_never_need_the_api_key(monkeypatch):
    monkeypatch.setattr(settings.security, "api_key", "secret-test-token")
    assert client.get("/").status_code == 200
    for asset, kind in (("analyzer.css", "text/css"), ("analyzer.js", "javascript")):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200
        assert kind in response.headers["content-type"]


# ---- health, readiness, metrics -------------------------------------------------------------------------------------


def test_health_reports_every_model_and_degrades_honestly():
    data = client.get("/health").json()
    assert data["status"] in ("ready", "degraded")
    assert data["version"] == settings.app_version
    assert {"yunet", "yolox", "anime_face", "siglip_vision", "scrfd", "mivolo", "label_bank"} <= set(data["models"])

    def model(ready, enabled=True):
        return {
            "model_name": "m", "file": "m.onnx", "role": "r", "license": "l", "ready": ready,
            "size_bytes": 1, "enabled": enabled, "commercial_use_cleared": True,
        }  # fmt: skip

    not_ready, ready, disabled = model(False), model(True), model(False, enabled=False)
    for models, expected in (
        ({"a": not_ready}, "degraded"),
        ({"a": ready}, "ready"),
        ({"a": ready, "b": disabled}, "ready"),  # a model that is switched off cannot make the service degraded
    ):
        with patch.object(routes, "check_models_status", return_value=models):
            assert client.get("/health").json()["status"] == expected


def test_health_states_the_licensing_position():
    licensing = client.get("/health").json()["licensing"]
    assert licensing["commercial_use_cleared"] is False
    assert any("MiVOLO" in name for name in licensing["blocking_models"])
    assert any("SCRFD" in name for name in licensing["blocking_models"])


def test_switching_scrfd_off_removes_it_from_the_licensing_blockers(monkeypatch):
    monkeypatch.setattr(settings.detection, "enable_scrfd", False)
    data = client.get("/health").json()
    assert not any("SCRFD" in name for name in data["licensing"]["blocking_models"])
    assert data["models"]["scrfd"]["enabled"] is False


def test_ready_follows_the_warm_up():
    fresh = create_app()
    probe = TestClient(fresh)
    warming = probe.get("/ready")
    assert warming.status_code == 503
    assert warming.json()["status"] == "warming_up"

    fresh.state.warm_up = {"mivolo": {"loaded": True, "warm_up_ms": 12.5, "providers": ["CPUExecutionProvider"]}}
    ok = probe.get("/ready")
    assert ok.status_code == 200
    assert ok.json() == {"status": "ready", "models": fresh.state.warm_up}

    fresh.state.warm_up = {"mivolo": {"loaded": False, "warm_up_ms": None, "providers": []}}
    broken = probe.get("/ready")
    assert broken.status_code == 503
    assert broken.json()["status"] == "not_ready"


def test_lifespan_warms_the_models_up_and_cleans_up():
    report = {"mivolo": {"loaded": True, "warm_up_ms": 1.0, "providers": ["CPUExecutionProvider"]}}
    with patch("app.main.warm_up", return_value=report):
        with TestClient(create_app()) as probe:
            for _ in range(100):
                if probe.get("/ready").status_code == 200:
                    break
                time.sleep(0.05)
            assert probe.get("/ready").json()["models"] == report


def test_lifespan_can_skip_the_warm_up(monkeypatch):
    monkeypatch.setattr(settings.models, "warm_up_on_start", False)
    with patch("app.main.warm_up") as warm:
        with TestClient(create_app()) as probe:
            assert probe.get("/ready").json() == {"status": "ready", "models": {}}
    warm.assert_not_called()


def test_metrics_count_requests_analyses_and_stages(test_image_bytes):
    from app.services import analyzer as az

    with (
        patch.object(az, "detect_faces", return_value=([detected([200, 200, 400, 400])], ["Detector test"])),
        patch.object(az, "detect_entities_and_persons", return_value=([], [], [])),
        patch.object(az, "classify_boxes", side_effect=lambda img, boxes: [HUMAN] * len(boxes)),
        patch.object(az, "classify_image_type", return_value=PHOTO),
        patch.object(az, "predict_face_demographics", return_value=adult()),
    ):
        assert post(data=test_image_bytes).status_code == 200
    assert post(data=b"").status_code == 400
    text = client.get("/metrics").text
    assert 'ppa_http_requests_total{method="POST",route="/v1/analyze-profile-photo",status="200"}' in text
    assert 'ppa_analyses_total{status="PASSED"}' in text
    assert 'ppa_stage_seconds_count{stage="detection"}' in text
    assert 'ppa_rejected_uploads_total{code="empty_file"}' in text


# ---- analysing ---------------------------------------------------------------------------------------------------


def test_successful_analysis_passes_decoded_metadata(stub_analysis):
    response = post()
    assert response.status_code == 200
    assert response.json()["status"] == "NO_INFORMATION"
    kwargs = stub_analysis.call_args.kwargs
    assert kwargs["filename"] == "p.jpg"
    assert kwargs["file_format"] == "image/jpeg"
    assert kwargs["meta"].original_width == 64
    assert kwargs["render_annotation"] is True


def test_the_annotated_image_can_be_left_out(stub_analysis):
    assert post(annotated_image="false").status_code == 200
    assert stub_analysis.call_args.kwargs["render_annotation"] is False
    assert post(path="/analyze-profile-photo", annotated_image="false").status_code == 200


def test_the_response_does_not_echo_the_original_image():
    body = post().json()
    assert "original_image" not in body
    assert body["annotated_image"].startswith("data:image/jpeg")


def test_real_pipeline_output_satisfies_public_schema(test_image_bytes):
    """Full route -> decode -> analyzer with only the model calls replaced."""
    from app.services import analyzer as az

    with (
        patch.object(az, "detect_faces", return_value=([detected([200, 200, 400, 400])], ["Detector test"])),
        patch.object(az, "detect_entities_and_persons", return_value=([], [], [])),
        patch.object(az, "classify_boxes", side_effect=lambda img, boxes: [HUMAN] * len(boxes)),
        patch.object(az, "classify_image_type", return_value=PHOTO),
        patch.object(az, "predict_face_demographics", return_value=adult()),
    ):
        response = post(data=test_image_bytes)
    assert response.status_code == 200
    validated = AnalysisResponse(**response.json())
    assert validated.real_human_face_count == 1
    assert validated.human_faces[0].gender == "Female"


def test_the_legacy_path_still_works_and_is_marked_deprecated(stub_analysis):
    response = post(path="/analyze-profile-photo")
    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
    assert 'rel="successor-version"' in response.headers["Link"]
    assert "Deprecation" not in post().headers


@pytest.mark.parametrize("name", ["a.jpg", "a.exe", "no_extension", "A.PNG", "../../etc/passwd", "x" * 400 + ".jpg"])
def test_the_file_name_never_decides_what_is_accepted(stub_analysis, name):
    assert post(name=name).status_code == 200


def test_the_file_name_is_cleaned_before_it_is_used():
    clean = routes._clean_filename
    assert clean("C:\\evil\\..\\pho\x00to\x1f.jpg") == "photo.jpg"
    assert clean("../../etc/passwd") == "passwd"
    assert len(clean("x" * 400)) == 255
    assert clean("/") == "upload"
    assert clean(None) == "upload"
    assert clean("  \t ") == "upload"


def test_rejects_empty_oversized_and_non_image_payloads():
    assert_error(post(data=b""), 400, "empty_file")
    assert_error(post(data=b"0" * (settings.upload.max_bytes + 10)), 413, "file_too_large")
    assert_error(post(data=b"NOT_AN_IMAGE_DATA_HEADER"), 400, "invalid_image")

    corrupt = post(data=b"\xff\xd8\xff\xe0" + b"\x00" * 40)
    assert "not a valid supported image" in assert_error(corrupt, 400, "invalid_image")["reason"].lower()


def test_rejects_oversized_dimensions(monkeypatch):
    monkeypatch.setattr(settings.upload, "max_dimension", 50)
    response = post(data=jpeg_bytes((100, 10)))
    assert "limit" in assert_error(response, 400, "invalid_image")["reason"]


def test_a_declared_huge_body_is_refused_before_it_is_read(monkeypatch):
    monkeypatch.setattr(settings.upload, "max_bytes", 1000)
    big = post(data=b"0" * 2_000_000)
    assert_error(big, 413, "file_too_large")


def test_a_streamed_body_without_a_length_is_cut_off(monkeypatch):
    monkeypatch.setattr(settings.upload, "max_bytes", 1000)

    def chunks():
        for _ in range(300):
            yield b"x" * 10_000

    response = client.post(
        "/v1/analyze-profile-photo", content=chunks(), headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert_error(response, 413, "file_too_large")


def test_internal_errors_do_not_leak_details(test_image_bytes):
    with patch.object(routes, "analyze_profile_photo_image", side_effect=RuntimeError("secret internals")):
        response = post(data=test_image_bytes)
    assert "secret internals" not in response.text
    assert "analysis failed" in assert_error(response, 500, "analysis_failed")["reason"].lower()


def test_unexpected_exceptions_are_answered_in_the_same_shape():
    probe = create_app()

    @probe.get("/boom")
    def boom():
        raise RuntimeError("secret internals")

    response = TestClient(probe, raise_server_exceptions=False).get("/boom")
    assert "secret internals" not in response.text
    assert_error(response, 500, "internal_error")


def test_an_error_after_the_response_has_started_is_logged_not_answered_twice(capsys):
    from fastapi.responses import StreamingResponse

    probe = create_app()

    @probe.get("/half")
    def half():
        def chunks():
            yield b"start"
            raise RuntimeError("late failure")

        return StreamingResponse(chunks())

    response = TestClient(probe, raise_server_exceptions=False).get("/half")
    assert response.status_code == 200
    assert "Unhandled error in GET /half" in capsys.readouterr().err


def test_unknown_paths_wrong_methods_and_missing_files_use_the_error_shape():
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/v1/analyze-profile-photo"), 405, "method_not_allowed")
    missing = client.post("/v1/analyze-profile-photo")
    assert "file" in assert_error(missing, 422, "validation_error")["reason"]


def test_an_http_error_with_another_status_keeps_its_detail():
    probe = create_app()

    @probe.get("/teapot")
    def teapot():
        from fastapi import HTTPException

        raise HTTPException(status_code=418, detail="short and stout")

    response = TestClient(probe).get("/teapot")
    assert assert_error(response, 418, "internal_error")["reason"] == "short and stout"


# ---- request ids, headers, CORS -----------------------------------------------------------------------------------------


def test_every_response_carries_a_request_id_and_security_headers():
    first, second = client.get("/health"), client.get("/health")
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert first.headers["X-Content-Type-Options"] == "nosniff"
    assert first.headers["Referrer-Policy"] == "no-referrer"
    assert first.headers["X-Frame-Options"] == "DENY"


def test_a_sensible_request_id_from_the_caller_is_kept_and_a_bad_one_replaced():
    assert client.get("/health", headers={"X-Request-ID": "trace-12345678"}).headers["X-Request-ID"] == "trace-12345678"
    replaced = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces"}).headers["X-Request-ID"]
    assert replaced != "bad id\twith spaces"
    assert len(replaced) == 32


def test_cors_exposes_the_headers_a_browser_client_needs():
    response = client.options(
        "/v1/analyze-profile-photo",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    simple = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "X-Request-ID" in simple.headers["access-control-expose-headers"]


# ---- authentication, rate limit, concurrency ---------------------------------------------------------------------------


def test_api_key_authentication(stub_analysis, monkeypatch):
    monkeypatch.setattr(settings.security, "api_key", "secret-test-token")
    assert_error(post(), 401, "unauthorized")
    assert post(headers={"Authorization": "Bearer secret-test-token"}).status_code == 200
    assert post(headers={"X-API-Key": "secret-test-token"}).status_code == 200
    assert_error(post(headers={"X-API-Key": "wrong-key"}), 401, "unauthorized")
    assert_error(post(headers={"Authorization": "Basic abc"}), 401, "unauthorized")
    assert client.get("/health").status_code == 200  # public
    assert client.get("/ready").status_code in (200, 503)  # public
    assert_error(client.get("/metrics"), 401, "unauthorized")
    assert client.get("/metrics", headers={"X-API-Key": "secret-test-token"}).status_code == 200


def test_rate_limiting(stub_analysis, monkeypatch):
    monkeypatch.setattr(settings.security, "rate_limit_per_minute", 2)
    first = post()
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert first.headers["X-RateLimit-Remaining"] == "1"
    assert post().status_code == 200
    limited = post()
    assert_error(limited, 429, "rate_limited")
    assert limited.headers["Retry-After"] == "60"
    assert limited.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limiting_can_be_switched_off_and_never_touches_public_paths(stub_analysis, monkeypatch):
    monkeypatch.setattr(settings.security, "rate_limit_per_minute", 1)
    for _ in range(3):
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200  # a scraper is not rate limited
    monkeypatch.setattr(settings.security, "rate_limit_enabled", False)
    for _ in range(3):
        assert post().status_code == 200


def test_concurrency_limit_applies_to_the_synchronous_endpoint_only(stub_analysis, monkeypatch):
    monkeypatch.setattr(settings.security, "max_concurrent_requests", 0)
    assert client.get("/health").status_code == 200
    assert_error(post(), 503, "busy")
    assert_error(post(path="/analyze-profile-photo"), 503, "busy")


def test_the_concurrency_slot_is_released_after_each_request(stub_analysis, monkeypatch):
    monkeypatch.setattr(settings.security, "max_concurrent_requests", 1)
    for _ in range(3):
        assert post().status_code == 200


def test_production_mode_requires_api_key():
    with pytest.raises(ValueError, match="requires PPA_SECURITY__API_KEY"):
        Settings(environment="production", security={"api_key": None, "require_api_key_in_production": True})
    assert (
        Settings(environment="production", security={"api_key": "prod-secret-key"}).security.api_key
        == "prod-secret-key"
    )
    assert Settings(environment="development").environment == "development"
    assert (
        Settings(environment="production", security={"require_api_key_in_production": False}).environment
        == "production"
    )


def test_settings_read_nested_environment_variables(monkeypatch):
    monkeypatch.setenv("PPA_AGE__MINOR_RISK_PROBABILITY", "0.33")
    monkeypatch.setenv("PPA_ENVIRONMENT", "staging")
    loaded = Settings()
    assert loaded.age.minor_risk_probability == 0.33
    assert loaded.environment == "staging"
    assert loaded.age.model_mae_years == 5.1  # the rest of the group keeps its defaults


# ---- jobs ----------------------------------------------------------------------------------------------------------


def wait_for(job_id, state="succeeded"):
    for _ in range(200):
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job never reached {state}: {body}")


def slow_analyzer(release):
    def run(image, **kwargs):
        release.wait(5)
        return valid_payload(**{k: v for k, v in kwargs.items() if k in ("filename", "file_format", "meta")})

    return run


def test_a_job_runs_and_reports_its_stages(stub_analysis):
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=lambda image, **kw: _staged(kw)):
        created = post(path="/v1/jobs", name="job.jpg")
        assert created.status_code == 202
        body = created.json()
        assert body["status_url"] == f"/v1/jobs/{body['job_id']}"
        assert body["events_url"] == f"/v1/jobs/{body['job_id']}/events"
        done = wait_for(body["job_id"])
    assert done["filename"] == "job.jpg"
    assert [stage["step_id"] for stage in done["stages"]] == ["validate", "information"]
    assert done["result"]["status"] == "NO_INFORMATION"
    assert done["error"] is None


def _staged(kwargs):
    from app.services.progress import stage_event

    kwargs["report"](stage_event("validate", "completed", "ok"))
    kwargs["report"](stage_event("information", "completed", "fine", 5.0))
    return valid_payload()


def test_a_failing_job_reports_a_clean_error():
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=RuntimeError("secret internals")):
        job_id = post(path="/v1/jobs").json()["job_id"]
        failed = wait_for(job_id, "failed")
    assert failed["result"] is None
    assert failed["error"]["code"] == "analysis_failed"
    assert "secret internals" not in json.dumps(failed)


def test_a_job_that_cannot_start_is_refused_like_any_bad_upload():
    assert_error(post(path="/v1/jobs", data=b"NOT_AN_IMAGE_DATA_HEADER"), 400, "invalid_image")
    assert_error(post(path="/v1/jobs", data=b""), 400, "empty_file")


def test_unknown_jobs_are_not_found():
    assert_error(client.get("/v1/jobs/unknown"), 404, "job_not_found")
    assert_error(client.get("/v1/jobs/unknown/events"), 404, "job_not_found")


def test_a_full_queue_answers_503_with_retry_after(monkeypatch):
    import threading

    monkeypatch.setattr(settings.jobs, "workers", 1)
    monkeypatch.setattr(settings.jobs, "queue_size", 1)
    release = threading.Event()
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=slow_analyzer(release)):
        first, second = post(path="/v1/jobs"), post(path="/v1/jobs")
        assert (first.status_code, second.status_code) == (202, 202)
        full = post(path="/v1/jobs")
        assert full.headers["Retry-After"] == "5"
        assert_error(full, 503, "queue_full")
        release.set()
        wait_for(first.json()["job_id"])
        wait_for(second.json()["job_id"])
        assert post(path="/v1/jobs").status_code == 202  # room again


def test_the_event_stream_delivers_every_stage_then_done():
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=lambda image, **kw: _staged(kw)):
        job_id = post(path="/v1/jobs").json()["job_id"]
        wait_for(job_id)
        response = client.get(f"/v1/jobs/{job_id}/events")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Accel-Buffering"] == "no"
    events = [
        dict(line.split(": ", 1) for line in block.splitlines()) for block in response.text.split("\n\n") if block
    ]
    assert [event["event"] for event in events] == ["stage", "stage", "done"]
    assert [event.get("id") for event in events] == ["1", "2", None]
    assert json.loads(events[0]["data"])["step_id"] == "validate"
    assert json.loads(events[-1]["data"]) == {"status": "succeeded"}


def test_the_event_stream_sends_keep_alives_while_it_waits(monkeypatch):
    import threading

    monkeypatch.setattr(routes, "EVENT_WAIT_SECONDS", 0.05)
    release = threading.Event()
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=slow_analyzer(release)):
        job_id = post(path="/v1/jobs").json()["job_id"]
        timer = threading.Timer(0.4, release.set)
        timer.start()
        response = client.get(f"/v1/jobs/{job_id}/events")
        timer.join()
    assert ": keep-alive" in response.text
    assert response.text.rstrip().endswith('"succeeded"}')


def test_finished_jobs_are_forgotten_after_their_time_to_live(monkeypatch):
    monkeypatch.setattr(settings.jobs, "result_ttl_seconds", 0.3)
    with patch.object(jobs_module, "analyze_profile_photo_image", side_effect=lambda image, **kw: valid_payload()):
        job_id = post(path="/v1/jobs").json()["job_id"]
        wait_for(job_id)
        time.sleep(0.4)
        assert_error(client.get(f"/v1/jobs/{job_id}"), 404, "job_not_found")


# ---- the documented contract -------------------------------------------------------------------------------------------


def test_openapi_documents_versions_errors_jobs_and_the_deprecated_alias():
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operation = paths["/v1/analyze-profile-photo"]["post"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/AnalysisResponse")
    assert {"400", "401", "413", "422", "429", "500", "503"} <= set(operation["responses"])
    assert operation["responses"]["400"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert paths["/analyze-profile-photo"]["post"]["deprecated"] is True
    assert paths["/v1/jobs"]["post"]["responses"]["202"]
    assert {"/v1/jobs/{job_id}", "/v1/jobs/{job_id}/events", "/health", "/ready"} <= set(paths)
    assert "original_image" not in schema["components"]["schemas"]["AnalysisResponse"]["properties"]


def test_module_entrypoint_launches_uvicorn():
    import app.__main__ as entry

    with patch("uvicorn.run") as run:
        entry.main()
    run.assert_called_once()
    assert run.call_args.args[0] == "app.main:app"
    assert run.call_args.kwargs["reload"] is False


def test_package_exposes_single_version():
    import app as package

    assert package.__version__ == settings.app_version


def test_python_dash_m_entrypoint():
    import runpy

    with patch("uvicorn.run") as run:
        runpy.run_module("app", run_name="__main__")
    run.assert_called_once()
