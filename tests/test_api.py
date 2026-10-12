import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from app.core.config import Settings, settings
from app.main import ProductionSecurityMiddleware
from app.schemas.analysis import AnalysisResponse
from app.services.analyzer import analyze_profile_photo_image
from tests.test_analyzer import HUMAN, PHOTO, adult, detected

client = TestClient(app)


def jpeg_bytes(size=(64, 64), color="grey") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def valid_payload() -> dict:
    """A complete, schema-valid response (the no-information path needs no model)."""
    return analyze_profile_photo_image(Image.new("RGB", (64, 64), "white"))


@pytest.fixture
def stub_analysis():
    with patch("app.api.routes.analyze_profile_photo_image", side_effect=lambda *a, **k: valid_payload()) as stub:
        yield stub


def post(name="p.jpg", data=None, headers=None, ctype="image/jpeg"):
    return client.post(
        "/analyze-profile-photo",
        headers=headers or {},
        files={"file": (name, jpeg_bytes() if data is None else data, ctype)},
    )


def test_home_and_missing_template():
    response = client.get("/")
    assert response.status_code == 200
    assert "Profile Photo Analyzer" in response.text
    with patch("pathlib.Path.is_file", return_value=False):
        assert "Frontend template not found" in client.get("/").text


def test_health_reports_every_model_and_degrades_honestly():
    data = client.get("/health").json()
    assert data["status"] in ("ready", "degraded")
    assert data["version"] == settings.APP_VERSION
    assert {"yunet", "yolox", "anime_face", "siglip_vision", "scrfd", "mivolo", "label_bank"} <= set(data["models"])
    with patch("app.api.routes.check_models_status", return_value={"yunet": {"ready": False, "size_bytes": 0}}):
        assert client.get("/health").json()["status"] == "degraded"
    with patch("app.api.routes.check_models_status", return_value={"yunet": {"ready": True, "size_bytes": 1}}):
        assert client.get("/health").json()["status"] == "ready"


def test_successful_analysis_passes_decoded_metadata(stub_analysis):
    response = post()
    assert response.status_code == 200
    assert response.json()["status"] == "NO_INFORMATION"
    kwargs = stub_analysis.call_args.kwargs
    assert kwargs["filename"] == "p.jpg"
    assert kwargs["file_format"] == "image/jpeg"
    assert kwargs["meta"].original_width == 64


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


@pytest.mark.parametrize(
    "name", ["a.jpg", "a.jpeg", "a.jfif", "a.png", "a.webp", "a.gif", "a.tif", "a.tiff", "a.bmp", "A.JPG"]
)
def test_all_documented_extensions_are_accepted(stub_analysis, name):
    assert post(name=name).status_code == 200


def test_rejects_unlisted_extension():
    response = post(name="malicious.exe", data=b"binarycontent", ctype="application/octet-stream")
    assert response.status_code == 400
    assert response.json()["accepted"] is False
    assert "Only" in response.json()["reason"]
    assert post(name="noextension", data=b"x").status_code == 400


def test_rejects_empty_oversized_and_non_image_payloads():
    assert "empty" in post(data=b"").json()["reason"].lower()

    too_big = post(data=b"0" * (settings.MAX_UPLOAD_BYTES + 10))
    assert too_big.status_code == 413
    assert "maximum upload size" in too_big.json()["reason"].lower()

    fake = post(data=b"NOT_AN_IMAGE_DATA_HEADER")
    assert fake.status_code == 400
    assert "unsupported binary format" in fake.json()["reason"].lower()

    corrupt = post(data=b"\xff\xd8\xff\xe0" + b"\x00" * 40)
    assert corrupt.status_code == 400
    assert "not a valid supported image" in corrupt.json()["reason"].lower()


def test_rejects_oversized_dimensions(monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_DIMENSION", 50)
    response = post(data=jpeg_bytes((100, 10)))
    assert response.status_code == 400
    assert "limit" in response.json()["reason"]


def test_internal_errors_do_not_leak_details(test_image_bytes):
    with patch("app.api.routes.analyze_profile_photo_image", side_effect=RuntimeError("secret internals")):
        response = post(data=test_image_bytes)
    assert response.status_code == 500
    assert "secret internals" not in response.text
    assert "analysis failed" in response.json()["reason"].lower()


def test_api_key_authentication(stub_analysis):
    with patch.object(settings, "API_KEY", "secret-test-token"):
        assert post().status_code == 401
        assert post(headers={"Authorization": "Bearer secret-test-token"}).status_code == 200
        assert post(headers={"X-API-Key": "secret-test-token"}).status_code == 200
        assert post(headers={"X-API-Key": "wrong-key"}).status_code == 401
        assert client.get("/health").status_code == 200  # exempt


def test_rate_limiting(stub_analysis):
    import app.main as app_main

    with patch.object(settings, "RATE_LIMIT_ENABLED", True), patch.object(settings, "RATE_LIMIT_PER_MINUTE", 2):
        with app_main._rate_limit_lock:
            app_main._request_records.clear()
        assert post().status_code == 200
        assert post().status_code == 200
        limited = post()
        assert limited.status_code == 429
        assert "Rate limit exceeded" in limited.json()["detail"]
        assert limited.headers["Retry-After"] == "60"


def test_concurrency_limit(stub_analysis):
    with patch.object(settings, "MAX_CONCURRENT_REQUESTS", 0):
        assert client.get("/health").status_code == 200
        busy = post()
        assert busy.status_code == 503
        assert "concurrency limit" in busy.json()["detail"]


def test_rate_limit_headers_present(stub_analysis):
    import app.main as app_main

    with app_main._rate_limit_lock:
        app_main._request_records.clear()
    response = post()
    assert response.headers["X-RateLimit-Limit"] == str(settings.RATE_LIMIT_PER_MINUTE)


def test_production_mode_requires_api_key():
    with pytest.raises(ValueError, match="Production mode .* requires PPA_API_KEY"):
        Settings(ENVIRONMENT="production", API_KEY=None, REQUIRE_API_KEY_IN_PRODUCTION=True)
    assert Settings(ENVIRONMENT="production", API_KEY="prod-secret-key").API_KEY == "prod-secret-key"
    assert Settings(ENVIRONMENT="development", API_KEY=None).ENVIRONMENT == "development"


def test_trusted_proxy_ip_resolution():
    from starlette.requests import Request

    def request(client_addr=("127.0.0.1", 1), forwarded=None):
        headers = [(b"x-forwarded-for", forwarded)] if forwarded is not None else []
        scope = {"type": "http", "headers": headers}
        if client_addr:
            scope["client"] = client_addr
        return Request(scope)

    resolve = ProductionSecurityMiddleware._resolve_client_ip
    with patch.object(settings, "TRUSTED_PROXIES", ["127.0.0.1"]):
        assert resolve(request(forwarded=b"203.0.113.50, 10.0.0.1")) == "203.0.113.50"
        assert resolve(request()) == "127.0.0.1"
        assert resolve(request(forwarded=b"   ")) == "127.0.0.1"
        assert resolve(request(client_addr=("8.8.8.8", 1), forwarded=b"1.2.3.4")) == "8.8.8.8"
    assert resolve(request(client_addr=None)) == "unknown"


def test_rate_limit_record_pruning():
    import time

    import app.main as app_main

    now = time.time()
    with app_main._rate_limit_lock:
        app_main._request_records.clear()
        app_main._request_records.update({"1.1.1.1": [now - 120.0], "2.2.2.2": [now], "3.3.3.3": []})
    ProductionSecurityMiddleware._prune_rate_limit_records(now)
    assert set(app_main._request_records) == {"2.2.2.2"}

    with patch.object(settings, "MAX_TRACKED_IPS", 2):
        app_main._request_records.update({"4.4.4.4": [now], "5.5.5.5": [now]})
        ProductionSecurityMiddleware._prune_rate_limit_records(now)
        assert len(app_main._request_records) <= 2

    app_main._request_records.clear()
    for i in range(105):
        app_main._request_records[f"10.0.0.{i}"] = [now - 120.0]
    ProductionSecurityMiddleware._check_rate_limit("10.0.0.200", now)
    assert len(app_main._request_records) <= 2


def test_module_entrypoint_launches_uvicorn():
    import app.__main__ as entry

    with patch("uvicorn.run") as run:
        entry.main()
    run.assert_called_once()
    assert run.call_args.args[0] == "app.main:app"
    assert run.call_args.kwargs["reload"] is False


def test_package_exposes_single_version():
    import app as package

    assert package.__version__ == settings.APP_VERSION


def test_python_dash_m_entrypoint():
    import runpy

    with patch("uvicorn.run") as run:
        runpy.run_module("app", run_name="__main__")
    run.assert_called_once()


def test_page_assets_are_served_and_never_need_the_api_key():
    with patch.object(settings, "API_KEY", "secret-test-token"):
        assert client.get("/").status_code == 200
        css, script = client.get("/static/analyzer.css"), client.get("/static/analyzer.js")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert script.status_code == 200
    assert "analyze-profile-photo" in script.text


def test_openapi_documents_the_response_contract():
    schema = client.get("/openapi.json").json()
    post_operation = schema["paths"]["/analyze-profile-photo"]["post"]
    ok = post_operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert ok["$ref"].endswith("/AnalysisResponse")
    assert {"400", "413", "500"} <= set(post_operation["responses"])
