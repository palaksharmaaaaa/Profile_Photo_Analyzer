import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from app.services import model_manager as mm


def test_ensure_model_file_already_exists(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"dummy_model_bytes")
    assert mm.ensure_model_file(f, "http://example.com/model.onnx") is True


def test_ensure_model_file_download_success(tmp_path):
    target = tmp_path / "subdir" / "downloaded.onnx"

    mock_resp = io.BytesIO(b"fake_model_content_12345")
    mock_resp.status = 200

    class DummyContext:
        def __enter__(self):
            return mock_resp
        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=DummyContext()):
        assert mm.ensure_model_file(target, "http://example.com/downloaded.onnx") is True
        assert target.is_file()
        assert target.read_bytes() == b"fake_model_content_12345"


def test_ensure_model_file_download_failure(tmp_path):
    target = tmp_path / "fail.onnx"

    def fail_with_empty_file(*args, **kwargs):
        target.touch()
        raise RuntimeError("Network down")

    with patch("urllib.request.urlopen", side_effect=fail_with_empty_file):
        assert mm.ensure_model_file(target, "http://example.com/fail.onnx") is False
        assert not target.is_file()


def test_ensure_model_file_cleanup_oserror(tmp_path):
    target = tmp_path / "fail_oserror.onnx"
    target.touch()
    with patch("urllib.request.urlopen", side_effect=RuntimeError("Network down")), \
         patch.object(Path, "unlink", side_effect=OSError("Cannot unlink")):
        assert mm.ensure_model_file(target, "http://example.com/fail.onnx") is False


def test_get_yunet_detector(monkeypatch):
    monkeypatch.setattr(mm, "_yunet_detector", None)
    monkeypatch.setattr(mm, "_yunet_size", (0, 0))

    # ensure_model_file fails
    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_yunet_detector(640, 640) is None

    # initialization succeeds
    mock_yn = MagicMock()
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("cv2.FaceDetectorYN.create", return_value=mock_yn):
        det = mm.get_yunet_detector(640, 640)
        assert det == mock_yn

        # change dimensions
        mm.get_yunet_detector(800, 800)
        mock_yn.setInputSize.assert_called_with((800, 800))

    # exception in creation
    monkeypatch.setattr(mm, "_yunet_detector", None)
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("cv2.FaceDetectorYN.create", side_effect=RuntimeError("cv2 error")):
        assert mm.get_yunet_detector(640, 640) is None


def test_get_genderage_session(monkeypatch):
    monkeypatch.setattr(mm, "_genderage_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_genderage_session() is None

    mock_sess = MagicMock()
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", return_value=mock_sess):
        sess = mm.get_genderage_session()
        assert sess == mock_sess
        # cached return
        assert mm.get_genderage_session() == mock_sess

    monkeypatch.setattr(mm, "_genderage_session", None)
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")):
        assert mm.get_genderage_session() is None


def test_get_anime_face_session(monkeypatch):
    monkeypatch.setattr(mm, "_anime_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_anime_face_session() is None

    mock_sess = MagicMock()
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", return_value=mock_sess):
        sess = mm.get_anime_face_session()
        assert sess == mock_sess
        assert mm.get_anime_face_session() == mock_sess

    monkeypatch.setattr(mm, "_anime_session", None)
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")):
        assert mm.get_anime_face_session() is None


def test_get_yolo_session(monkeypatch):
    monkeypatch.setattr(mm, "_yolo_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_yolo_session() is None

    mock_sess = MagicMock()
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", return_value=mock_sess):
        sess = mm.get_yolo_session()
        assert sess == mock_sess
        assert mm.get_yolo_session() == mock_sess

    monkeypatch.setattr(mm, "_yolo_session", None)
    with patch("app.services.model_manager.ensure_model_file", return_value=True), \
         patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")):
        assert mm.get_yolo_session() is None
