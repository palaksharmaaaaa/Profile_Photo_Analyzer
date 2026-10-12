import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import cv2
import pytest

from app.core.config import settings
from app.services import model_manager as mm


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_manifest(models_dir, name: str, data: bytes):
    (models_dir / mm.MANIFEST_NAME).write_text(json.dumps({"files": {name: {"sha256": sha(data)}}}), encoding="utf-8")


def test_registry_declares_every_model_with_licence_and_role():
    assert set(mm.MODEL_SPECS) == {"yunet", "anime_face", "yolox", "siglip_vision", "scrfd", "mivolo"}
    for spec in mm.MODEL_SPECS.values():
        assert spec.license and spec.role and spec.filename.endswith(".onnx")


def test_downloadable_models_are_hash_pinned_and_built_models_are_not_downloadable():
    for key in ("yunet", "anime_face", "yolox"):
        assert mm.MODEL_SPECS[key].url.startswith("https://")
        assert len(mm.MODEL_SPECS[key].sha256) == 64
    for key in ("siglip_vision", "scrfd", "mivolo"):
        assert mm.MODEL_SPECS[key].url is None


def test_expected_sha256_prefers_registry_then_manifest(models_dir):
    assert mm.expected_sha256(mm.MODEL_SPECS["yolox"].filename) == mm.MODEL_SPECS["yolox"].sha256
    assert mm.expected_sha256("age_gender_vit_int8.onnx") is None
    write_manifest(models_dir, "age_gender_vit_int8.onnx", b"abc")
    assert mm.expected_sha256("age_gender_vit_int8.onnx") == sha(b"abc")


def test_manifest_with_garbage_is_ignored(models_dir):
    (models_dir / mm.MANIFEST_NAME).write_text("not json", encoding="utf-8")
    assert mm.expected_sha256("x.onnx") is None
    (models_dir / mm.MANIFEST_NAME).write_text('{"files": {"x.onnx": 5}}', encoding="utf-8")
    assert mm.expected_sha256("x.onnx") is None
    (models_dir / mm.MANIFEST_NAME).write_text("[1, 2]", encoding="utf-8")
    assert mm.expected_sha256("x.onnx") is None


def test_valid_file_requires_existence_size_and_matching_hash(models_dir):
    path = models_dir / "built.onnx"
    assert mm.is_valid_model_file(path) is False  # missing

    path.write_bytes(b"tiny")
    assert mm.is_valid_model_file(path, min_size=100) is False  # too small

    data = b"x" * 200
    path.write_bytes(data)
    write_manifest(models_dir, "built.onnx", data)
    assert mm.is_valid_model_file(path, min_size=100) is True

    path.write_bytes(b"y" * 200)  # tampered, same size
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
    assert mm.is_valid_model_file(path, min_size=100) is False


def test_validity_is_cached_until_file_changes(models_dir):
    path = models_dir / "built.onnx"
    data = b"x" * 200
    path.write_bytes(data)
    write_manifest(models_dir, "built.onnx", data)
    with patch.object(mm, "_calculate_file_sha256", wraps=mm._calculate_file_sha256) as spy:
        assert mm.is_valid_model_file(path, min_size=100)
        assert mm.is_valid_model_file(path, min_size=100)
        assert spy.call_count == 1


def test_hash_read_error_is_invalid(models_dir):
    path = models_dir / "built.onnx"
    path.write_bytes(b"x" * 200)
    write_manifest(models_dir, "built.onnx", b"x" * 200)
    with patch.object(mm, "_calculate_file_sha256", side_effect=OSError("disk")):
        assert mm.is_valid_model_file(path, min_size=100) is False


def test_unpinned_file_must_at_least_load(models_dir):
    path = models_dir / "unknown.onnx"
    path.write_bytes(b"\x08" + b"x" * 200)
    with patch.object(mm.ort, "InferenceSession", return_value=object()):
        assert mm.is_valid_model_file(path, min_size=100) is True

    path.write_bytes(b"\x08" + b"y" * 200)  # new mtime content -> re-evaluated
    with patch.object(mm.ort, "InferenceSession", side_effect=RuntimeError("bad")):
        mm._validation_cache.clear()
        assert mm.is_valid_model_file(path, min_size=100) is False


def test_loadable_check_rejects_bad_header(models_dir):
    path = models_dir / "unknown.onnx"
    path.write_bytes(b"\xff" * 200)
    assert mm._verify_model_loadable(path) is False
    assert mm._verify_model_loadable(models_dir / "missing.onnx") is False


def fake_download(payload: bytes):
    def _download(url, dest):
        dest.write_bytes(payload)

    return _download


def test_ensure_returns_true_for_valid_existing_file(models_dir):
    path = models_dir / "built.onnx"
    path.write_bytes(b"x" * 200)
    write_manifest(models_dir, "built.onnx", b"x" * 200)
    assert mm.ensure_model_file(path, None, min_size=100) is True


def test_ensure_never_deletes_or_downloads_locally_built_models(models_dir):
    path = models_dir / "built.onnx"
    path.write_bytes(b"corrupt" * 100)
    write_manifest(models_dir, "built.onnx", b"the real file")
    with patch.object(mm, "_download_file_stream") as download:
        assert mm.ensure_model_file(path, None, min_size=100) is False
    download.assert_not_called()
    assert path.exists()


def test_ensure_downloads_missing_model_atomically(models_dir):
    path = models_dir / "sub" / "fetched.onnx"
    payload = b"z" * 300
    write_manifest(models_dir, "fetched.onnx", payload)
    with patch.object(mm, "_download_file_stream", side_effect=fake_download(payload)):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is True
    assert path.read_bytes() == payload
    assert not path.with_suffix(".onnx.tmp").exists()


def test_ensure_replaces_corrupt_downloadable_model(models_dir):
    path = models_dir / "fetched.onnx"
    path.write_bytes(b"old" * 100)
    payload = b"z" * 300
    write_manifest(models_dir, "fetched.onnx", payload)
    with patch.object(mm, "_download_file_stream", side_effect=fake_download(payload)):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is True
    assert path.read_bytes() == payload


def test_ensure_replacement_survives_unlink_failure(models_dir):
    path = models_dir / "fetched.onnx"
    path.write_bytes(b"old" * 100)
    payload = b"z" * 300
    write_manifest(models_dir, "fetched.onnx", payload)
    with (
        patch("pathlib.Path.unlink", side_effect=OSError("locked")),
        patch.object(mm, "_download_file_stream", side_effect=fake_download(payload)),
    ):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is True


@pytest.mark.parametrize(
    ("payload", "expected_hash"),
    [(b"s" * 10, None), (b"w" * 300, "0" * 64)],
    ids=["too-small", "hash-mismatch"],
)
def test_ensure_rejects_bad_downloads_and_cleans_up(models_dir, payload, expected_hash):
    path = models_dir / "fetched.onnx"
    if expected_hash:
        (models_dir / mm.MANIFEST_NAME).write_text(
            json.dumps({"files": {"fetched.onnx": {"sha256": expected_hash}}}), encoding="utf-8"
        )
    with patch.object(mm, "_download_file_stream", side_effect=fake_download(payload)):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is False
    assert not path.exists()
    assert not path.with_suffix(".onnx.tmp").exists()


def test_ensure_handles_network_error_and_cleanup_failure(models_dir):
    path = models_dir / "fetched.onnx"

    def boom(url, dest):
        dest.write_bytes(b"partial")
        raise OSError("network down")

    with (
        patch.object(mm, "_download_file_stream", side_effect=boom),
        patch("pathlib.Path.unlink", side_effect=OSError("locked")),
    ):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is False


def test_download_refuses_plain_http(models_dir):
    with pytest.raises(ValueError, match="non-HTTPS"):
        mm._download_file_stream("http://example.test/m.onnx", models_dir / "x")


def test_download_streams_response_to_file(models_dir):
    response = MagicMock()
    response.read.side_effect = [b"abc", b"def", b""]
    response.__enter__.return_value = response
    with patch.object(mm.urllib.request, "urlopen", return_value=response) as urlopen:
        mm._download_file_stream("https://example.test/m.onnx", models_dir / "out.bin")
    assert (models_dir / "out.bin").read_bytes() == b"abcdef"
    assert urlopen.call_args.kwargs["timeout"] == settings.DOWNLOAD_TIMEOUT_SECONDS


def test_check_models_status_reports_each_model_honestly(models_dir):
    status = mm.check_models_status()
    assert set(status) == {*mm.MODEL_SPECS, "label_bank"}
    assert all(item["ready"] is False and item["size_bytes"] == 0 for item in status.values())

    (models_dir / mm.LABEL_BANK_NAME).write_bytes(b"npz")
    assert mm.check_models_status()["label_bank"]["ready"] is True

    path = mm.model_path("yunet")
    with patch.object(mm, "is_valid_model_file", return_value=True), patch("pathlib.Path.stat") as stat:
        stat.return_value.st_size = 1234
        stat.return_value.st_mtime = 1.0
        assert mm.check_models_status()["yunet"]["size_bytes"] == 1234
    assert path.name == mm.MODEL_SPECS["yunet"].filename


def test_get_session_none_when_model_missing(models_dir):
    assert mm.get_session("mivolo") is None


def test_get_session_creates_once_with_bounded_threads(models_dir):
    created = []

    class Options:
        intra_op_num_threads = 0

    def factory(path, sess_options=None, providers=None):
        created.append((path, sess_options.intra_op_num_threads, providers))
        return object()

    with (
        patch.object(mm, "ensure_model_file", return_value=True),
        patch.object(mm.ort, "InferenceSession", side_effect=factory),
        patch.object(mm.ort, "SessionOptions", Options),
    ):
        first = mm.get_session("yolox")
        second = mm.get_session("yolox")
    assert first is second
    assert len(created) == 1
    assert 1 <= created[0][1] <= mm.DEFAULT_ORT_THREADS
    assert created[0][2] == ["CPUExecutionProvider"]


def test_get_session_respects_explicit_thread_setting(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "ORT_INTRA_OP_THREADS", 2)
    seen = {}

    def factory(path, sess_options=None, providers=None):
        seen["threads"] = sess_options.intra_op_num_threads
        return object()

    with patch.object(mm, "ensure_model_file", return_value=True), patch.object(mm.ort, "InferenceSession", factory):
        mm.get_session("yolox")
    assert seen["threads"] == 2


def test_get_session_returns_none_when_runtime_fails(models_dir):
    with (
        patch.object(mm, "ensure_model_file", return_value=True),
        patch.object(mm.ort, "InferenceSession", side_effect=RuntimeError("bad model")),
    ):
        assert mm.get_session("yolox") is None


def test_providers_prefer_accelerators_only_when_cpu_not_forced(monkeypatch):
    assert mm._providers() == ["CPUExecutionProvider"]
    monkeypatch.setattr(settings, "FORCE_CPU", False)
    with patch.object(
        mm.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]
    ):
        assert mm._providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with patch.object(mm.ort, "get_available_providers", return_value=["CPUExecutionProvider"]):
        assert mm._providers() == ["CPUExecutionProvider"]


def test_yunet_detector_lifecycle(models_dir):
    detector = MagicMock()
    with (
        patch.object(mm, "ensure_model_file", return_value=True),
        patch.object(cv2.FaceDetectorYN, "create", return_value=detector) as create,
    ):
        assert mm.get_yunet_detector(100, 100) is detector
        assert mm.get_yunet_detector(100, 100) is detector  # same size: no resize
        assert mm.get_yunet_detector(200, 150) is detector
    create.assert_called_once()
    detector.setInputSize.assert_called_once_with((200, 150))


def test_yunet_detector_unavailable_or_failing(models_dir):
    with patch.object(mm, "ensure_model_file", return_value=False):
        assert mm.get_yunet_detector() is None
    with (
        patch.object(mm, "ensure_model_file", return_value=True),
        patch.object(cv2.FaceDetectorYN, "create", side_effect=cv2.error("bad")),
    ):
        assert mm.get_yunet_detector() is None


def test_ensure_handles_failure_before_any_file_was_written(models_dir):
    path = models_dir / "fetched.onnx"
    with patch.object(mm, "_download_file_stream", side_effect=OSError("dns")):
        assert mm.ensure_model_file(path, "https://example.test/m.onnx", min_size=100) is False
