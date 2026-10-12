import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.schemas.analysis import AnalysisResponse, ErrorResponse
from app import app
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
    assert data["status"] == "running"
    assert "version" in data


def test_analyze_endpoint_valid_image(test_image_bytes):
    faces = [MockFace(bbox=[150, 150, 350, 350], age=25.0)]
    with patch("app.services.analyzer.detect_faces", return_value=faces):
        response = client.post(
            "/analyze-profile-photo",
            files={"file": ("profile.jpg", test_image_bytes, "image/jpeg")}
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
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "Only" in data["reason"]


def test_analyze_endpoint_empty_file():
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "empty" in data["reason"].lower()


def test_analyze_endpoint_corrupted_image():
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA_HEADER", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "invalid or unsupported binary format" in data["reason"].lower()


def test_analyze_endpoint_corrupt_data_valid_header():
    corrupted_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 40
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("corrupt.jpg", corrupted_jpeg, "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["accepted"] is False
    assert "not a valid supported image" in data["reason"].lower()


def test_analyze_endpoint_file_too_large():
    from app.core.config import settings
    # 9 MB dummy payload
    oversized = b"0" * (settings.MAX_UPLOAD_BYTES + 10)
    response = client.post(
        "/analyze-profile-photo",
        files={"file": ("huge.jpg", oversized, "image/jpeg")}
    )
    assert response.status_code == 413
    data = response.json()
    assert data["accepted"] is False
    assert "maximum upload size" in data["reason"].lower()


def test_analyze_endpoint_internal_error(test_image_bytes):
    with patch("app.api.routes.analyze_profile_photo_image", side_effect=RuntimeError("Unexpected crash")):
        response = client.post(
            "/analyze-profile-photo",
            files={"file": ("profile.jpg", test_image_bytes, "image/jpeg")}
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

