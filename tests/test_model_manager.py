import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services import model_manager as mm


def test_ensure_model_file_already_exists(tmp_path):
    f = tmp_path / "model.onnx"
    f.write_bytes(b"\x08" + b"x" * 59999)
    with patch("app.services.model_manager._verify_model_loadable", return_value=True):
        assert mm.ensure_model_file(f, "http://example.com/model.onnx") is True


def test_ensure_model_file_download_success(tmp_path):
    target = tmp_path / "subdir" / "downloaded.onnx"

    mock_resp = io.BytesIO(b"\x08" + b"x" * 59999)
    mock_resp.status = 200

    class DummyContext:
        def __enter__(self):
            return mock_resp

        def __exit__(self, *args):
            pass

    with (
        patch("urllib.request.urlopen", return_value=DummyContext()),
        patch("app.services.model_manager._verify_model_loadable", return_value=True),
    ):
        assert mm.ensure_model_file(target, "http://example.com/downloaded.onnx") is True
        assert target.is_file()
        assert target.stat().st_size == 60000


def test_ensure_model_file_download_failure(tmp_path):
    target = tmp_path / "fail.onnx"
    temp_target = target.with_suffix(".onnx.tmp")

    def fail_with_temp_file(*args, **kwargs):
        temp_target.touch()
        raise RuntimeError("Network down")

    with patch("urllib.request.urlopen", side_effect=fail_with_temp_file):
        assert mm.ensure_model_file(target, "http://example.com/fail.onnx") is False
        assert not temp_target.is_file()


def test_ensure_model_file_cleanup_oserror(tmp_path):
    target = tmp_path / "fail_oserror.onnx"
    temp_target = target.with_suffix(".onnx.tmp")
    temp_target.touch()
    with (
        patch("urllib.request.urlopen", side_effect=RuntimeError("Network down")),
        patch.object(Path, "unlink", side_effect=OSError("Cannot unlink")),
    ):
        assert mm.ensure_model_file(target, "http://example.com/fail.onnx") is False


def test_ensure_model_file_removes_truncated_file(tmp_path):
    f = tmp_path / "truncated.onnx"
    f.write_bytes(b"short")
    with patch("urllib.request.urlopen", side_effect=RuntimeError("Download aborted")):
        assert mm.ensure_model_file(f, "http://example.com/truncated.onnx") is False
        assert not f.is_file()


def test_ensure_model_file_truncated_unlink_oserror(tmp_path):
    f = tmp_path / "truncated_err.onnx"
    f.write_bytes(b"short")
    with (
        patch.object(Path, "unlink", side_effect=OSError("Permission denied")),
        patch("urllib.request.urlopen", side_effect=RuntimeError("Aborted")),
    ):
        assert mm.ensure_model_file(f, "http://example.com/truncated_err.onnx") is False


def test_ensure_model_file_download_too_small(tmp_path):
    target = tmp_path / "too_small.onnx"
    mock_resp = io.BytesIO(b"short")

    class DummyContext:
        def __enter__(self):
            return mock_resp

        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=DummyContext()):
        assert mm.ensure_model_file(target, "http://example.com/too_small.onnx") is False
        assert not target.is_file()


def test_ensure_model_file_download_hash_mismatch(tmp_path):
    target = tmp_path / mm.settings.YUNET_MODEL_NAME
    # Download payload has correct size but wrong cryptographic hash
    mock_resp = io.BytesIO(b"\x08" + b"z" * 60000)

    class DummyContext:
        def __enter__(self):
            return mock_resp

        def __exit__(self, *args):
            pass

    with patch("urllib.request.urlopen", return_value=DummyContext()):
        assert mm.ensure_model_file(target, "http://example.com/fake_yunet.onnx") is False
        assert not target.is_file()
        assert not target.with_suffix(f"{target.suffix}.tmp").is_file()


def test_is_valid_model_file_rejects_non_onnx_signature(tmp_path):
    f = tmp_path / "fake_non_onnx.onnx"
    f.write_bytes(b"This is a 72 KB text file masquerading as a model..." + b"x" * 72000)
    assert mm.is_valid_model_file(f) is False
    with patch("urllib.request.urlopen", side_effect=RuntimeError("Aborted")):
        assert mm.ensure_model_file(f, "http://example.com/model.onnx") is False
        assert not f.is_file()


def test_is_valid_model_file_ioerror(tmp_path):
    f = tmp_path / "valid.onnx"
    f.write_bytes(b"\x08" + b"x" * 60000)
    with patch("builtins.open", side_effect=OSError("Read error")):
        assert mm.is_valid_model_file(f) is False


def test_is_valid_model_file_nonexistent_and_dir(tmp_path):
    # Non-existent file raises FileNotFoundError (caught by OSError handler)
    non_existent = tmp_path / "missing.onnx"
    assert mm.is_valid_model_file(non_existent) is False

    # Directory is not a regular file
    a_dir = tmp_path / "subfolder"
    a_dir.mkdir()
    assert mm.is_valid_model_file(a_dir) is False


def test_check_models_status():
    status = mm.check_models_status()
    assert isinstance(status, dict)
    for model_key in ("yunet", "genderage", "anime_face", "yolov8n"):
        assert model_key in status
        assert "ready" in status[model_key]
        assert "size_bytes" in status[model_key]


def test_get_yunet_detector(monkeypatch):
    monkeypatch.setattr(mm, "_yunet_detector", None)
    monkeypatch.setattr(mm, "_yunet_size", (0, 0))

    # ensure_model_file fails
    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_yunet_detector(640, 640) is None

    # initialization succeeds
    mock_yn = MagicMock()
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("cv2.FaceDetectorYN.create", return_value=mock_yn),
    ):
        det = mm.get_yunet_detector(640, 640)
        assert det == mock_yn

        # change dimensions
        mm.get_yunet_detector(800, 800)
        mock_yn.setInputSize.assert_called_with((800, 800))

    # exception in creation
    monkeypatch.setattr(mm, "_yunet_detector", None)
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("cv2.FaceDetectorYN.create", side_effect=RuntimeError("cv2 error")),
    ):
        assert mm.get_yunet_detector(640, 640) is None


def test_get_genderage_session(monkeypatch):
    monkeypatch.setattr(mm, "_genderage_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_genderage_session() is None

    mock_sess = MagicMock()
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", return_value=mock_sess),
    ):
        sess = mm.get_genderage_session()
        assert sess == mock_sess
        # cached return
        assert mm.get_genderage_session() == mock_sess

    monkeypatch.setattr(mm, "_genderage_session", None)
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")),
    ):
        assert mm.get_genderage_session() is None


def test_get_anime_face_session(monkeypatch):
    monkeypatch.setattr(mm, "_anime_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_anime_face_session() is None

    mock_sess = MagicMock()
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", return_value=mock_sess),
    ):
        sess = mm.get_anime_face_session()
        assert sess == mock_sess
        assert mm.get_anime_face_session() == mock_sess

    monkeypatch.setattr(mm, "_anime_session", None)
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")),
    ):
        assert mm.get_anime_face_session() is None


def test_get_yolo_session(monkeypatch):
    monkeypatch.setattr(mm, "_yolo_session", None)

    with patch("app.services.model_manager.ensure_model_file", return_value=False):
        assert mm.get_yolo_session() is None

    mock_sess = MagicMock()
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", return_value=mock_sess),
    ):
        sess = mm.get_yolo_session()
        assert sess == mock_sess
        assert mm.get_yolo_session() == mock_sess

    monkeypatch.setattr(mm, "_yolo_session", None)
    with (
        patch("app.services.model_manager.ensure_model_file", return_value=True),
        patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")),
    ):
        assert mm.get_yolo_session() is None


def test_is_valid_model_file_pinned_hash(tmp_path):
    f = tmp_path / mm.settings.YUNET_MODEL_NAME
    f.write_bytes(b"\x08" + b"x" * 59999)
    correct_hash = mm.MODEL_PINNED_HASHES[mm.settings.YUNET_MODEL_NAME]

    with patch("app.services.model_manager._calculate_file_sha256", return_value=correct_hash):
        assert mm.is_valid_model_file(f) is True
        # Second call hits cache
        assert mm.is_valid_model_file(f) is True

    # Pinned hash mismatch
    bad_file = tmp_path / mm.settings.GENDERAGE_MODEL_NAME
    bad_file.write_bytes(b"\x08" + b"bad" * 20000)
    with patch("app.services.model_manager._calculate_file_sha256", return_value="deadbeef"):
        assert mm.is_valid_model_file(bad_file) is False

    # OSError in hash calculation
    oserror_file = tmp_path / "oserror_model.onnx"
    oserror_file.write_bytes(b"\x08" + b"err" * 20000)
    with patch("app.services.model_manager._calculate_file_sha256", side_effect=OSError("Read error")):
        assert mm.is_valid_model_file(oserror_file, expected_name=mm.settings.GENDERAGE_MODEL_NAME) is False


def test_verify_model_loadable_branches(tmp_path):
    # Invalid header
    bad_header = tmp_path / "bad_head.onnx"
    bad_header.write_bytes(b"\x00" * 32)
    assert mm._verify_model_loadable(bad_header) is False

    # YuNet model loading success & fail
    yunet_path = tmp_path / mm.settings.YUNET_MODEL_NAME
    yunet_path.write_bytes(b"\x08" + b"x" * 32)
    with patch("cv2.FaceDetectorYN.create", return_value=MagicMock()):
        assert mm._verify_model_loadable(yunet_path) is True
    with patch("cv2.FaceDetectorYN.create", side_effect=RuntimeError("cv2 error")):
        assert mm._verify_model_loadable(yunet_path) is False

    # ONNX session loading success & fail
    onnx_path = tmp_path / "custom.onnx"
    onnx_path.write_bytes(b"\x08" + b"x" * 32)
    with patch("onnxruntime.InferenceSession", return_value=MagicMock()):
        assert mm._verify_model_loadable(onnx_path) is True
    with patch("onnxruntime.InferenceSession", side_effect=RuntimeError("ort error")):
        assert mm._verify_model_loadable(onnx_path) is False


def test_calculate_file_sha256(tmp_path):
    import hashlib

    test_file = tmp_path / "hash_test.bin"
    content = b"ProfilePhotoAnalyzerIntegrityCheck"
    test_file.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert mm._calculate_file_sha256(test_file) == expected
