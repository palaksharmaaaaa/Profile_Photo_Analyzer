import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app import app
from app.schemas.analysis import AnalysisResponse
from tests.conftest import MockFace

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Profile Photo Analyzer" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "degraded")
    assert "version" in data
    assert "models" in data


def test_health_endpoint_degraded():
    with patch(
        "app.api.routes.check_models_status",
        return_value={"yunet": {"ready": False, "size_bytes": 0}},
    ):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"


def test_analyze_endpoint_valid_image(test_image_bytes):
    faces = [MockFace(bbox=[150, 150, 350, 350], age=25.0)]
    mock_type = {
        "status": "PHOTO_LIKE",
        "top_class": "a real photograph of a real human person",
        "top_class_confidence": 0.95,
        "photo_score": 0.95,
        "cartoon_score": 0.05,
        "display_label": "Real Photo",
        "reason": "Highest scoring category is real photo.",
        "model_executed": "OpenAI CLIP ViT-B/32",
    }
    with (
        patch("app.services.analyzer.classify_image_type", return_value=mock_type),
        patch("app.services.analyzer.detect_faces", return_value=faces),
    ):
        response = client.post(
            "/analyze-profile-photo",
            files={"file": ("profile.jpg", test_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        validated = AnalysisResponse(**data)
        assert validated.face_detected is True
        assert validated.face_count == 1
        assert validated.original_image is not None
        assert validated.annotated_image is not None


def test_analyze_endpoint_invalid_extension():
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "Only" in data["reason"]


def test_analyze_endpoint_empty_file():
    response = client.post("/analyze-profile-photo", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "empty" in data["reason"].lower()


def test_analyze_endpoint_corrupted_image():
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA_HEADER", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "invalid or unsupported binary format" in data["reason"].lower()


def test_analyze_endpoint_corrupt_data_valid_header():
    corrupted_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 40
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("corrupt.jpg", corrupted_jpeg, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "not a valid supported image" in data["reason"].lower()


def test_analyze_endpoint_file_too_large():
    from app.core.config import settings

    # 9 MB dummy payload
    oversized = b"0" * (settings.MAX_UPLOAD_BYTES + 10)
    response = client.post("/analyze-profile-photo", files={"file": ("huge.jpg", oversized, "image/jpeg")})
    assert response.status_code == 413
    data = response.json()
    assert data["accepted"] is False
    assert "maximum upload size" in data["reason"].lower()


def test_analyze_endpoint_internal_error(test_image_bytes):
    with patch(
        "app.api.routes.analyze_profile_photo_image",
        side_effect=RuntimeError("Unexpected crash"),
    ):
        response = client.post(
            "/analyze-profile-photo",
            files={"file": ("profile.jpg", test_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["accepted"] is False
        assert "analysis failed" in data["reason"].lower()


def test_home_template_not_found():
    with patch("pathlib.Path.is_file", return_value=False):
        response = client.get("/")
        assert response.status_code == 200
        assert "Frontend template not found" in response.text


def test_app_main_block():
    with patch("uvicorn.run") as mock_uvicorn:
        import runpy

        runpy.run_path("app.py", run_name="__main__")
        mock_uvicorn.assert_called_once()


def test_analyze_endpoint_dimension_too_large():
    img = Image.new("RGB", (5000, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    payload = buf.getvalue()

    response = client.post("/analyze-profile-photo", files={"file": ("wide.jpg", payload, "image/jpeg")})
    assert response.status_code == 400
    assert "limit" in response.json()["reason"]


def test_analyze_endpoint_pixel_count_too_large():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        img = Image.new("RGB", (4000, 4001), color="white")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        payload = buf.getvalue()

    response = client.post("/analyze-profile-photo", files={"file": ("bomb.jpg", payload, "image/jpeg")})
    assert response.status_code == 400
    assert "pixel count" in response.json()["reason"]


def test_analyze_endpoint_decompression_bomb_error(test_image_bytes):
    with patch(
        "PIL.Image.open",
        side_effect=Image.DecompressionBombError("Decompression bomb DOS attack"),
    ):
        response = client.post("/analyze-profile-photo", files={"file": ("bomb.jpg", test_image_bytes, "image/jpeg")})
        assert response.status_code == 400
        assert "decompression limit" in response.json()["reason"]


def test_security_middleware_api_key_auth(test_image_bytes):
    from app.core.config import settings

    with patch.object(settings, "API_KEY", "secret-test-token"):
        # Without key -> 401
        res_no_key = client.post("/analyze-profile-photo", files={"file": ("p.jpg", test_image_bytes, "image/jpeg")})
        assert res_no_key.status_code == 401

        # With Bearer header -> 200
        res_auth = client.post(
            "/analyze-profile-photo",
            headers={"Authorization": "Bearer secret-test-token"},
            files={"file": ("p.jpg", test_image_bytes, "image/jpeg")},
        )
        assert res_auth.status_code == 200

        # With X-API-Key header -> 200
        res_header = client.post(
            "/analyze-profile-photo",
            headers={"X-API-Key": "secret-test-token"},
            files={"file": ("p.jpg", test_image_bytes, "image/jpeg")},
        )
        assert res_header.status_code == 200

        # Wrong key -> 401
        res_wrong = client.post(
            "/analyze-profile-photo",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("p.jpg", test_image_bytes, "image/jpeg")},
        )
        assert res_wrong.status_code == 401


def test_security_middleware_rate_limiting():
    import app.main as app_main
    from app.core.config import settings

    with (
        patch.object(settings, "RATE_LIMIT_ENABLED", True),
        patch.object(settings, "RATE_LIMIT_PER_MINUTE", 2),
    ):
        with app_main._rate_limit_lock:
            app_main._request_records.clear()

        # Request 1 & 2: pass
        r1 = client.post("/analyze-profile-photo")
        assert r1.status_code in (200, 400, 422)
        r2 = client.post("/analyze-profile-photo")
        assert r2.status_code in (200, 400, 422)

        # Request 3: rate limited
        r3 = client.post("/analyze-profile-photo")
        assert r3.status_code == 429
        assert "Rate limit exceeded" in r3.json()["detail"]


def test_security_middleware_concurrency_limit():
    from app.core.config import settings

    with patch.object(settings, "MAX_CONCURRENT_REQUESTS", 0):
        r_health = client.get("/health")
        assert r_health.status_code == 200  # exempt

        r_busy = client.post("/analyze-profile-photo")
        assert r_busy.status_code == 503
        assert "concurrency limit" in r_busy.json()["detail"]
