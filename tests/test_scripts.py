"""The command-line tools: model fetching and building, evaluation data download and the latency benchmark."""

import csv
import hashlib
import importlib.util
import io
import json
import runpy
import sys
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.label_prompts import FACE_CLASSES, IMAGE_CLASSES

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import benchmark_latency  # noqa: E402
import fetch_eval_data  # noqa: E402
import fetch_models  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_models = load_script("build_models")


class Download(io.BytesIO):
    """What ``urllib.request.urlopen`` returns: a readable context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---- fetch_models -------------------------------------------------------------------------------------------------------


def test_fetch_models_downloads_only_what_has_a_url(monkeypatch, capsys):
    asked = []

    def ensure(path, url):
        asked.append((path.name, url))
        return path.name != "yolox_s.onnx"

    monkeypatch.setattr(fetch_models, "ensure_model_file", ensure)
    assert fetch_models.main() == 1
    assert {name for name, _ in asked} == {
        "face_detection_yunet_2023mar.onnx",
        "anime_face_v1.4_n.onnx",
        "yolox_s.onnx",
    }
    out = capsys.readouterr().out
    assert "FAILED yolox_s.onnx" in out
    assert "ok     face_detection_yunet_2023mar.onnx" in out

    monkeypatch.setattr(fetch_models, "ensure_model_file", lambda path, url: True)
    assert fetch_models.main() == 0


# ---- the latency benchmark ------------------------------------------------------------------------------------------------


def test_time_stage_summarises_the_durations():
    calls = []
    summary = benchmark_latency.time_stage(lambda: calls.append(1), iterations=5, warmup=2)
    assert len(calls) == 7
    assert set(summary) == {"min", "mean", "p50", "p90", "max"}
    assert summary["min"] <= summary["p50"] <= summary["p90"] <= summary["max"]


def test_measure_times_every_stage_on_a_portrait(test_image_bytes):
    from tests.test_analyzer import detected

    with (
        patch.object(benchmark_latency, "detect_faces", return_value=([detected([200, 200, 400, 400])], ["x"])),
        patch.object(benchmark_latency, "check_image_information", return_value={}),
        patch.object(benchmark_latency, "classify_image_type", return_value={}),
        patch.object(benchmark_latency, "detect_entities_and_persons", return_value=([], [], [])),
        patch.object(benchmark_latency, "classify_boxes", return_value=[]),
        patch.object(benchmark_latency, "predict_face_demographics", return_value=None),
        patch.object(benchmark_latency, "analyze_profile_photo_image", return_value={}),
    ):
        report = benchmark_latency.measure(test_image_bytes, iterations=2)
    assert len(report["latency_ms"]) == 7
    assert "END-TO-END" in report["latency_ms"]
    assert report["environment"]["iterations"] == 2
    assert report["environment"]["parallel_inference"] is False


def test_measure_refuses_an_image_without_a_face(test_image_bytes):
    with patch.object(benchmark_latency, "detect_faces", return_value=([], [])):
        with pytest.raises(SystemExit, match="No face found"):
            benchmark_latency.measure(test_image_bytes, iterations=1)


def test_benchmark_command_line(test_image_bytes, tmp_path, monkeypatch, capsys):
    from app.core.config import settings

    image = tmp_path / "p.jpg"
    image.write_bytes(test_image_bytes)
    out = tmp_path / "report.json"
    monkeypatch.setattr(settings.models, "parallel_inference", False)
    with patch.object(benchmark_latency, "measure", return_value={"environment": {}, "latency_ms": {}}) as measure:
        assert (
            benchmark_latency.main(["--image", str(image), "--iterations", "3", "--parallel", "--json", str(out)]) == 0
        )
        assert settings.models.parallel_inference is True
        assert benchmark_latency.main(["--image", str(image)]) == 0
    assert measure.call_args_list[0].args[1] == 3
    assert measure.call_args_list[1].args[1] == 15
    assert json.loads(out.read_text(encoding="utf-8")) == {"environment": {}, "latency_ms": {}}
    assert '"latency_ms"' in capsys.readouterr().out
    monkeypatch.setattr(settings.models, "parallel_inference", False)


# ---- fetch_eval_data --------------------------------------------------------------------------------------------------------


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "boom", {}, None)


def test_open_retries_busy_answers_and_gives_up_on_real_errors(monkeypatch):
    monkeypatch.setattr(fetch_eval_data.time, "sleep", lambda seconds: None)
    answers = [http_error(429), TimeoutError(), Download(b"payload")]

    def urlopen(request, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    with patch.object(fetch_eval_data.urllib.request, "urlopen", urlopen):
        assert fetch_eval_data._open("https://x") == b"payload"
    with patch.object(fetch_eval_data.urllib.request, "urlopen", side_effect=http_error(404)):
        with pytest.raises(urllib.error.HTTPError):
            fetch_eval_data._open("https://x")
    with patch.object(fetch_eval_data.urllib.request, "urlopen", side_effect=http_error(503)):
        with pytest.raises(urllib.error.HTTPError):
            fetch_eval_data._open("https://x")  # still failing after every retry
    with patch.object(fetch_eval_data.urllib.request, "urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(urllib.error.URLError):
            fetch_eval_data._open("https://x")


def test_api_helpers_build_the_right_requests():
    requested = []

    def fake_open(url):
        requested.append(url)
        if "/size" in url:
            return json.dumps({"size": {"config": {"num_rows": 42}}}).encode()
        return json.dumps({"rows": [{"row_idx": 3}]}).encode()

    with patch.object(fetch_eval_data, "_open", fake_open):
        assert fetch_eval_data._total("a/b", "default") == 42
        assert fetch_eval_data._rows("a/b", "default", "train", 5, 10) == [{"row_idx": 3}]
    assert "dataset=a%2Fb" in requested[1] or "dataset=a/b" in requested[1]
    assert "offset=5" in requested[1]
    assert "length=10" in requested[1]


def parquet_with_faces(path: Path, count: int = 6) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "image": [{"bytes": f"jpeg-{index}".encode()} for index in range(count)],
            "age": [index % 9 for index in range(count)],
            "gender": [index % 2 for index in range(count)],
            "race": [index % 7 for index in range(count)],
        }
    )
    pq.write_table(table, path)


def test_fetch_fairface_samples_the_parquet_and_labels_each_face(tmp_path):
    parquet_with_faces(tmp_path / "fairface_validation.parquet")
    rows: list[list[str]] = []

    class Writer:
        def writerow(self, row):
            rows.append(row)

    import random

    fetch_eval_data.fetch_fairface(tmp_path, 3, random.Random(1), Writer(), known={"fairface/ff_00000.jpg"})
    written = sorted((tmp_path / "fairface").iterdir())
    assert 2 <= len(written) <= 3
    assert all(row[1:3] == ["fairface", "human"] for row in rows)
    first = rows[0]
    assert first[3] in fetch_eval_data.FAIRFACE_AGES
    assert first[4] in fetch_eval_data.FAIRFACE_GENDERS
    assert first[5] in fetch_eval_data.FAIRFACE_RACES
    assert (tmp_path / first[0]).read_bytes().startswith(b"jpeg-")


def test_fetch_fairface_downloads_the_parquet_when_it_is_missing(tmp_path):
    source = tmp_path / "source.parquet"
    parquet_with_faces(source)
    with patch.object(fetch_eval_data, "_open", return_value=source.read_bytes()) as opened:
        import random

        fetch_eval_data.fetch_fairface(tmp_path, 1, random.Random(1), MagicMock(), known=set())
    assert opened.call_args.args[0] == fetch_eval_data.FAIRFACE_PARQUET
    assert (tmp_path / "fairface_validation.parquet").is_file()


def row_of(index, genre=6):
    return {"row_idx": index, "row": {"genre": genre, "image": {"src": f"https://img/{index}"}}}


class InOrder:
    """A stand-in for ``random.Random`` that keeps the windows in their natural order."""

    def shuffle(self, items):
        items.sort()


def test_fetch_non_human_takes_images_until_it_has_enough_and_skips_known_ones(tmp_path, capsys):
    rows: list[list[str]] = []
    writer = MagicMock()
    writer.writerow.side_effect = rows.append
    windows = {0: [row_of(0), row_of(1)], 2: [row_of(2), row_of(3)], 4: [row_of(4), row_of(5)]}

    def fake_rows(dataset, config, split, offset, length):
        if offset == 2:
            raise http_error(500)  # an unavailable window is skipped
        return windows[offset]

    known = {"pets/pets_000001.jpg"}
    spec = ("a/b", "default", "train", None)
    with (
        patch.object(fetch_eval_data, "WINDOW", 2),
        patch.object(fetch_eval_data, "_total", return_value=6),
        patch.object(fetch_eval_data, "_rows", fake_rows),
        patch.object(fetch_eval_data, "_open", return_value=b"jpeg"),
    ):
        fetch_eval_data.fetch_non_human(tmp_path, "pets", spec, 4, InOrder(), writer, known)
    assert [row[0] for row in rows] == ["pets/pets_000000.jpg", "pets/pets_000004.jpg", "pets/pets_000005.jpg"]
    assert all(row[1:3] == ["pets", "not_human"] for row in rows)
    assert (tmp_path / rows[0][0]).read_bytes() == b"jpeg"
    assert "skipping rows 2-4 (500)" in capsys.readouterr().out

    rows.clear()  # enough images already: nothing more is fetched
    with (
        patch.object(fetch_eval_data, "WINDOW", 2),
        patch.object(fetch_eval_data, "_total", return_value=6),
        patch.object(fetch_eval_data, "_rows", side_effect=AssertionError("must not be asked")),
    ):
        fetch_eval_data.fetch_non_human(tmp_path, "pets", spec, 1, InOrder(), writer, known)
    assert rows == []


def test_fetch_non_human_filters_rows_by_a_column_and_stops_at_the_count(tmp_path):
    import random

    writer = MagicMock()
    windows = {0: [row_of(0, genre=1), row_of(1, genre=6), row_of(2, genre=6)]}
    spec = ("a/b", "default", "train", ("genre", 6))
    with (
        patch.object(fetch_eval_data, "WINDOW", 3),
        patch.object(fetch_eval_data, "_total", return_value=3),
        patch.object(fetch_eval_data, "_rows", lambda *args: windows[args[3]]),
        patch.object(fetch_eval_data, "_open", return_value=b"jpeg"),
    ):
        fetch_eval_data.fetch_non_human(tmp_path, "paintings", spec, 1, random.Random(1), writer, set())
    assert writer.writerow.call_count == 1
    assert writer.writerow.call_args.args[0][0] == "paintings/paintings_000001.jpg"


def test_fetch_eval_data_command_line_resumes_and_survives_one_failing_source(tmp_path, capsys):
    out = tmp_path / "data"
    out.mkdir()
    (out / "labels.csv").write_text(
        "file,set,expectation,age,gender,race\npets/old.jpg,pets,not_human,,,\n", encoding="utf-8"
    )

    def fake_non_human(path, name, spec, count, rng, writer, known):
        assert "pets/old.jpg" in known
        writer.writerow([f"{name}/new.jpg", name, "not_human", "", "", ""])
        if name == "birds":
            raise RuntimeError("source down")

    with (
        patch.object(fetch_eval_data, "fetch_non_human", fake_non_human),
        patch.object(fetch_eval_data, "fetch_fairface", lambda *args: None),
        patch.object(sys, "argv", ["fetch", "--out", str(out), "--only", "pets", "birds", "fairface"]),
    ):
        assert fetch_eval_data.main() == 0
    text = capsys.readouterr().out
    assert "birds: FAILED (RuntimeError: source down)" in text
    with (out / "labels.csv").open(encoding="utf-8") as handle:
        files = [row["file"] for row in csv.DictReader(handle)]
    assert files == ["pets/old.jpg", "pets/new.jpg", "birds/new.jpg"]

    fresh = tmp_path / "fresh"
    with (
        patch.object(fetch_eval_data, "fetch_non_human", lambda *args: None),
        patch.object(sys, "argv", ["fetch", "--out", str(fresh), "--only", "pets"]),
    ):
        fetch_eval_data.main()
    assert (fresh / "labels.csv").read_text(encoding="utf-8").startswith("file,set,expectation")


# ---- build_models -----------------------------------------------------------------------------------------------------------


def test_the_prompt_taxonomy_must_be_loadable(monkeypatch):
    monkeypatch.setattr(build_models.importlib.util, "spec_from_file_location", lambda name, path: None)
    with pytest.raises(ImportError, match="Cannot load the prompt taxonomy"):
        build_models._load_label_prompts()


def test_sha256_of_hashes_a_file_in_chunks(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc" * 1000)
    assert build_models.sha256_of(path) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_download_verifies_the_pinned_hash(tmp_path, capsys):
    payload = b"model bytes"
    expected = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "m.onnx"
    with patch.object(build_models.urllib.request, "urlopen", return_value=Download(payload)) as urlopen:
        build_models.download("https://x/m.onnx", dest, expected)
        assert dest.read_bytes() == payload
        build_models.download("https://x/m.onnx", dest, expected)  # cached: not downloaded again
    assert urlopen.call_count == 1
    assert "ok (cached)" in capsys.readouterr().out

    bad = tmp_path / "bad.onnx"
    with patch.object(build_models.urllib.request, "urlopen", return_value=Download(b"tampered")):
        with pytest.raises(SystemExit, match="SHA-256 mismatch"):
            build_models.download("https://x/bad.onnx", bad, expected)
    assert not bad.exists()
    assert not (tmp_path / "bad.onnx.part").exists()


def mivolo_stand_in(path: Path, age_offset: float = 61.0) -> None:
    """A tiny graph with MiVOLO's contract: input ``faces_bodies``, tensor ``add_45`` [1, 3] and output ``age``."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("faces_bodies", TensorProto.FLOAT, [1, 6, 384, 384])
    age = helper.make_tensor_value_info("age", TensorProto.FLOAT, [1, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [2, 3])
    starts0 = helper.make_tensor("s0", TensorProto.INT64, [1], [0])
    ends3 = helper.make_tensor("e3", TensorProto.INT64, [1], [3])
    axis1 = helper.make_tensor("a1", TensorProto.INT64, [1], [1])
    starts2 = helper.make_tensor("s2", TensorProto.INT64, [1], [2])
    ends6 = helper.make_tensor("e6", TensorProto.INT64, [1], [3])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [122.0])
    offset = helper.make_tensor("offset", TensorProto.FLOAT, [], [age_offset])
    nodes = [
        helper.make_node("ReduceMean", ["faces_bodies", "axes"], ["pooled"], keepdims=0),
        helper.make_node("Slice", ["pooled", "s0", "e3", "a1"], ["add_45"]),
        helper.make_node("Slice", ["add_45", "s2", "e6", "a1"], ["raw_age"]),
        helper.make_node("Mul", ["raw_age", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "offset"], ["age"]),
    ]
    graph = helper.make_graph(
        nodes, "mivolo", [x], [age], initializer=[axes, starts0, ends3, axis1, starts2, ends6, scale, offset]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


def test_build_mivolo_exposes_the_head_and_checks_it_against_the_original_age(tmp_path):
    source, target = tmp_path / "src.onnx", tmp_path / "out.onnx"
    mivolo_stand_in(source)
    validation = build_models.build_mivolo(source, target)
    assert validation["max_age_difference_years"] < 1e-3
    import onnxruntime as ort

    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["age", "head"]


def test_build_mivolo_refuses_a_changed_upstream_export(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "other.onnx"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    onnx.save(helper.make_model(graph), str(path))
    with pytest.raises(SystemExit, match="upstream export has changed"):
        build_models.build_mivolo(path, tmp_path / "out.onnx")


def test_build_mivolo_refuses_a_head_that_does_not_reproduce_the_age(tmp_path):
    source = tmp_path / "src.onnx"
    mivolo_stand_in(source, age_offset=50.0)  # the age output no longer equals the head's raw age * 122 + 61
    with pytest.raises(SystemExit, match="does not reproduce the original age output"):
        build_models.build_mivolo(source, tmp_path / "out.onnx")


def test_build_mivolo_refuses_a_head_of_the_wrong_shape(tmp_path, monkeypatch):
    source = tmp_path / "src.onnx"
    mivolo_stand_in(source)
    monkeypatch.setattr(build_models, "MIVOLO_HEAD_TENSOR", "scaled")  # the wrong tensor: it has shape (1, 1)
    with pytest.raises(SystemExit, match=r"shape \(1, 1\), not \(1, 3\)"):
        build_models.build_mivolo(source, tmp_path / "out.onnx")


def test_extract_scrfd_checks_the_pinned_hash(tmp_path, monkeypatch):
    archive = tmp_path / "buffalo_l.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("det_10g.onnx", b"detector")
    with pytest.raises(SystemExit, match="does not match the pinned SHA-256"):
        build_models.extract_scrfd(archive, tmp_path / "scrfd.onnx")
    monkeypatch.setattr(build_models, "SCRFD_SHA256", hashlib.sha256(b"detector").hexdigest())
    build_models.extract_scrfd(archive, tmp_path / "scrfd.onnx")
    assert (tmp_path / "scrfd.onnx").read_bytes() == b"detector"


def text_stand_ins(tmp_path: Path) -> tuple[Path, Path]:
    """A word-level tokenizer and a text model whose pooled output is the first 8 token ids."""
    import onnx
    from onnx import TensorProto, helper
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "<unk>": 1}
    words = {
        word
        for classes in (FACE_CLASSES, IMAGE_CLASSES)
        for c in classes
        for prompt in c.prompts
        for word in prompt.lower().split()
    }
    vocab.update({word: index + 2 for index, word in enumerate(sorted(words))})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["n", 64])
    pooled = helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["n", 8])
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["as_float"], to=TensorProto.FLOAT),
        helper.make_node("Slice", ["as_float", "s", "e", "a"], ["pooler_output"]),
    ]
    inits = [
        helper.make_tensor("s", TensorProto.INT64, [1], [0]),
        helper.make_tensor("e", TensorProto.INT64, [1], [8]),
        helper.make_tensor("a", TensorProto.INT64, [1], [1]),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "text", [ids], [pooled], initializer=inits),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    model_path = tmp_path / "text.onnx"
    onnx.save(model, str(model_path))
    return model_path, tokenizer_path


def test_prompts_are_embedded_and_averaged_per_class(tmp_path):
    model, tokenizer = text_stand_ins(tmp_path)
    vectors = build_models.embed_prompts(model, tokenizer, ["a photo of a dog", "a photo of a cat"])
    assert vectors.shape == (2, 8)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    out = tmp_path / "bank.npz"
    build_models.build_bank(model, tokenizer, out)
    with np.load(out) as bank:
        assert list(bank["face_names"]) == [c.name for c in FACE_CLASSES]
        assert bank["image_vectors"].shape == (len(IMAGE_CLASSES), 8)
        assert np.allclose(np.linalg.norm(bank["face_vectors"], axis=1), 1.0, atol=1e-5)
        assert json.loads(str(bank["meta"]))["tokens"] == 64


def test_main_builds_the_requested_stages_and_merges_the_manifest(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "manifest.json").write_text(
        json.dumps(
            {"schema": 2, "files": {"old.onnx": {"sha256": "x", "size": 1}}, "upstream_sources": {}, "validation": {}}
        ),
        encoding="utf-8",
    )
    downloads = []

    def fake_download(url, dest, expected):
        downloads.append(dest.name)
        dest.write_bytes(b"x" * 10)

    def fake_build_mivolo(src, dst):
        dst.write_bytes(b"mivolo")
        return {"max_age_difference_years": 0.0}

    def fake_extract(archive, dst):
        dst.write_bytes(b"scrfd")

    def fake_bank(text_model, tokenizer, out):
        out.write_bytes(b"bank")

    monkeypatch.setattr(build_models, "download", fake_download)
    monkeypatch.setattr(build_models, "build_mivolo", fake_build_mivolo)
    monkeypatch.setattr(build_models, "extract_scrfd", fake_extract)
    monkeypatch.setattr(build_models, "build_bank", fake_bank)
    work = tmp_path / "work"
    argv = ["build", "--work-dir", str(work), "--models-dir", str(models), "--cleanup"]

    with patch.object(sys, "argv", argv + ["--only", "mivolo", "scrfd"]):
        build_models.main()
    assert downloads == ["mivolo_v2_age.onnx", "buffalo_l.zip"]
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {"old.onnx", build_models.OUT_MIVOLO, build_models.OUT_SCRFD}
    assert manifest["validation"]["mivolo_exposed_head"] == {"max_age_difference_years": 0.0}
    assert not work.exists()  # --cleanup
    assert "scratch folder removed" in capsys.readouterr().out

    (work / "sig_vision_int8.onnx").parent.mkdir(parents=True, exist_ok=True)
    (work / "sig_vision_int8.onnx").write_bytes(b"vision")
    monkeypatch.setattr(build_models, "download", lambda url, dest, expected: None)
    with patch.object(sys, "argv", ["build", "--work-dir", str(work), "--models-dir", str(models), "--only", "siglip"]):
        build_models.main()
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_bank"]["file"] == build_models.OUT_BANK
    assert build_models.OUT_VISION in manifest["files"]

    fresh = tmp_path / "fresh-models"
    with patch.object(sys, "argv", ["build", "--work-dir", str(work), "--models-dir", str(fresh), "--only", "scrfd"]):
        build_models.main()
    assert (fresh / "manifest.json").is_file()


# ---- every tool can be run as a program ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["build_models", "fetch_eval_data", "benchmark_latency", "evaluate"])
def test_the_tools_describe_themselves_when_run_as_programs(script, capsys):
    with patch.object(sys, "argv", [script, "--help"]), pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPTS / f"{script}.py"), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_fetch_models_runs_as_a_program():
    with patch("app.services.model_manager.ensure_model_file", return_value=True), pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(SCRIPTS / "fetch_models.py"), run_name="__main__")
    assert stopped.value.code == 0
