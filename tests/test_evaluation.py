"""The evaluation tooling: metrics on hand-made rows, the runner with a fake pipeline, and the command line."""

import json
import math
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation import metrics, runner

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402


def face(gender="Male", age=25.0, status="Estimated", p_under_18=0.0, area=100):
    return {
        "area": area, "gender": gender, "gender_confidence": 0.95, "age_group": "x", "age_status": status,
        "estimated_age": age if status == "Estimated" else None, "age_range": None, "p_under_18": p_under_18,
        "verification": 0.9,
    }  # fmt: skip


def row(file, humans=(), others=0, unsure=0, status="PASSED", candidates=(), ms=100.0, error=None):
    return {
        "file": file, "error": error, "status": status, "humans": list(humans), "others": others,
        "unsure": unsure, "candidates": list(candidates), "ms": ms,
    }  # fmt: skip


def label(file, expectation="human", set_name="fairface", age="20-29", gender="Male", race="White"):
    return {"file": file, "set": set_name, "expectation": expectation, "age": age, "gender": gender, "race": race}


def candidate(p_human, margin, best_is_human=True):
    return {"kind": "human", "p_human": p_human, "margin": margin, "best_is_human": best_is_human}


# ---- bins and errors ----------------------------------------------------------------------------------------------------


def test_residual_to_a_bin_is_zero_inside_and_the_distance_outside():
    assert metrics.bin_residual(25, 20, 30) == 0
    assert metrics.bin_residual(31.5, 20, 30) == 1.5
    assert metrics.bin_residual(17, 20, 30) == 3
    assert metrics.bin_residual(95, 70, None) == 0
    assert metrics.bin_residual(65, 70, None) == 5


def test_expected_error_assumes_a_uniform_age_inside_the_bin():
    assert metrics.expected_error_in_bin(25, 20, 30) == pytest.approx(2.5)  # centre of a 10-year bin
    assert metrics.expected_error_in_bin(20, 20, 30) == pytest.approx(5.0)  # at the edge
    assert metrics.expected_error_in_bin(40, 20, 30) == pytest.approx(15.0)  # outside: distance to the centre


def test_the_main_face_is_the_largest_one():
    assert metrics.main_face(row("a", humans=[face(area=5), face(area=50, gender="Female")]))["gender"] == "Female"
    assert metrics.main_face(row("a")) is None


# ---- the reports --------------------------------------------------------------------------------------------------------


def portraits():
    rows = [
        row("a", [face("Male", 25)]),  # right
        row("b", [face("Female", 26)]),  # wrong gender (label is Male)
        row("c", [face("Unsure", 24)]),
        row("d", [], status="NO_HUMAN_FACE"),
        row("e", [face(), face()], status="MULTIPLE_HUMAN_FACES", others=1),
        row("f", error="ValueError: bad"),
    ]
    labels = {r["file"]: label(r["file"]) for r in rows}
    return metrics.human_set(rows, labels)


def test_detection_report_counts_every_outcome():
    report = metrics.detection_report(portraits())
    assert report["images"] == 6
    assert report["errors"] == 1
    assert report["exactly_one_human_rate"] == 0.6
    assert report["at_least_one_human_rate"] == 0.8
    assert report["no_human_rate"] == 0.2
    assert report["several_humans_rate"] == 0.2
    assert report["reported_a_non_human_face_rate"] == 0.2
    assert report["status_counts"]["NO_HUMAN_FACE"] == 1


def test_gender_report_by_group():
    report = metrics.gender_report(portraits())
    assert report["ALL"]["faces"] == 4
    assert report["ALL"]["accuracy_when_answered"] == pytest.approx(0.6667, abs=1e-4)
    assert report["ALL"]["unsure_rate"] == 0.25
    assert report["White"] == report["ALL"]
    assert report["ALL"]["unsure_rate_labelled_20_plus"] == 0.25  # the Unsure face is labelled 20-29
    assert report["ALL"]["unsure_rate_labelled_0_to_9"] is None  # no child in this set
    assert report["ALL"]["unsure_without_a_score_rate"] == 0.0  # the Unsure face still has a model score


def test_age_report_separates_a_lower_bound_from_an_assumption():
    pairs = [
        (row("a", [face(age=25)]), label("a", age="20-29")),
        (row("b", [face(age=34)]), label("b", age="20-29")),
        (row("c", [face(status="Unsure")]), label("c", age="20-29")),
        (row("d", [face(age=80)]), label("d", age="more than 70")),
        (row("e", [face(age=50)]), label("e", age="unknown bin")),
        (row("f", []), label("f", age="20-29")),
    ]
    report = metrics.age_report(pairs)
    assert report["faces"] == 4
    assert report["unsure_rate"] == 0.25
    assert report["inside_labelled_bin_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert report["mean_error_to_bin_years_lower_bound"] == pytest.approx(1.33, abs=0.01)
    assert report["mean_error_years_if_ages_uniform_in_bin"] == pytest.approx((2.5 + 9.0) / 2, abs=0.01)
    assert set(report["per_bin"]) == {"20-29"}


def test_the_error_is_reported_per_band_and_small_bands_borrow_the_pooled_spread(monkeypatch):
    monkeypatch.setattr(metrics, "MIN_FACES_PER_BAND", 4)
    by_bin = {
        "20-29": [18.0, 22.0, 28.0, 32.0, 25.0],  # five faces: a band of its own
        "30-39": [35.0, 36.0],  # two faces: too few
    }
    table = metrics.error_by_band(by_bin)
    assert [entry["below_age"] for entry in table] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 200.0]
    own = table[2]
    assert (own["faces"], own["source"], own["bias_years"]) == (5, "band", 0.0)
    assert own["spread_years"] == pytest.approx(math.sqrt(statistics.pvariance([-7, -3, 3, 7, 0]) - 100 / 12), abs=0.01)
    borrowed = table[3]
    assert (borrowed["faces"], borrowed["source"]) == (2, "pooled")
    assert borrowed["bias_years"] == 0.5
    empty = table[0]
    assert (empty["faces"], empty["bias_years"], empty["source"]) == (0, None, "pooled")
    assert borrowed["spread_years"] == empty["spread_years"] > 0
    assert metrics.error_by_band({})[0]["spread_years"] == 0.0


def test_the_spread_has_the_width_of_the_bin_taken_out():
    estimates = [23.0, 24.0, 26.0, 27.0]  # spread well inside what a uniform 10-year bin explains by itself
    spread = metrics._bin_spread("20-29", estimates)
    assert spread["bias_years"] == 0
    assert spread["faces"] == 4
    assert spread["model_spread_years"] == 0  # variance 17 is below the bin's own 100/12
    wide = metrics._bin_spread("20-29", [10.0, 40.0])
    assert wide["model_spread_years"] > 10
    assert metrics._bin_spread("20-29", [25.0])["model_spread_years"] == 0


def test_minor_report_uses_the_unambiguous_bins_only():
    pairs = [
        (row("a", [face(p_under_18=0.9)]), label("a", age="3-9")),
        (row("b", [face(p_under_18=0.01)]), label("b", age="0-2")),
        (row("c", [face(p_under_18=0.5)]), label("c", age="10-19")),
        (row("d", [face(p_under_18=0.2)]), label("d", age="30-39")),
        (row("e", [face(p_under_18=0.0)]), label("e", age="more than 70")),
        (row("f", [face(p_under_18=None)]), label("f", age="30-39")),
        (row("g", [face(p_under_18=0.9)]), label("g", age="20-29")),
    ]
    report = metrics.minor_report(pairs, 0.1)
    assert report["minor_recall_labelled_0_to_9"] == 0.5
    assert report["false_flag_rate_labelled_30_plus"] == 0.5
    assert report["flag_rate_labelled_10_to_19"] == 1.0
    assert (report["minors_scored"], report["adults_scored"]) == (2, 2)


def test_age_is_reported_per_race_group():
    pairs = [
        (row("a", [face(age=25)]), label("a", age="20-29", race="White")),
        (row("b", [face(age=40, p_under_18=0.5)]), label("b", age="3-9", race="Black")),
        (row("c", [face(age=34)]), label("c", age="20-29", race="Black")),
    ]
    report = metrics.age_by_group(pairs, 0.1)
    assert list(report) == ["ALL", "Black", "White"]
    assert report["ALL"]["faces"] == 3
    assert report["White"]["inside_labelled_bin_rate"] == 1.0
    assert report["Black"]["inside_labelled_bin_rate"] == 0.0
    assert report["Black"]["minor_recall_labelled_0_to_9"] == 1.0
    assert report["White"]["minor_recall_labelled_0_to_9"] is None


def test_minor_sweep_reports_every_threshold():
    pairs = [
        (row("a", [face(p_under_18=0.3)]), label("a", age="3-9")),
        (row("b", [face(p_under_18=0.05)]), label("b", age="40-49")),
    ]
    sweep = metrics.minor_sweep(pairs, [0.02, 0.1, 0.5])
    assert [entry["threshold"] for entry in sweep] == [0.02, 0.1, 0.5]
    assert [entry["minor_recall_labelled_0_to_9"] for entry in sweep] == [1.0, 1.0, 0.0]
    assert [entry["false_flag_rate_labelled_30_plus"] for entry in sweep] == [1.0, 0.0, 0.0]


def test_not_human_report_classifies_each_image_once():
    rows = [
        row("a", [face()]),  # reported as a real human and passed the whole policy
        row("b", [], others=1, status="ONLY_NON_HUMAN_FACES"),
        row("c", [], unsure=2, status="REVIEW_REQUIRED"),
        row("d", [], status="NO_HUMAN_FACE"),
        row("e", error="x"),
        row("f", [], others=1, status="ONLY_NON_HUMAN_FACES"),
    ]
    labels = {r["file"]: label(r["file"], "not_human", "pets" if r["file"] < "d" else "birds") for r in rows}
    report = metrics.not_human_report(metrics.not_human_set(rows, labels))
    assert report["ALL"] == {
        "images": 5, "reported_as_real_human_rate": 0.2, "passed_the_whole_policy_rate": 0.2,
        "reported_as_non_human_rate": 0.4,
        "left_unsure_rate": 0.2, "nothing_reported_rate": 0.2,
    }  # fmt: skip
    assert report["pets"]["images"] == 3
    assert report["birds"]["images"] == 2


# ---- the verification thresholds ------------------------------------------------------------------------------------------------


def test_threshold_sweep_and_the_best_choice():
    positives = [(row("a", candidates=[candidate(0.8, 0.5)]), label("a")), (row("b", candidates=[candidate(0.55, 0.2)]), label("b")),
                 (row("c", error="x"), label("c"))]  # fmt: skip
    negatives = [
        (row("d", candidates=[candidate(0.62, 0.3)]), label("d", "not_human", "pets")),
        (row("e", candidates=[candidate(0.9, 0.9, best_is_human=False)]), label("e", "not_human", "pets")),
    ]
    table = metrics.threshold_sweep(positives, negatives, [0.5, 0.6, 0.7], [0.0, 0.25])
    by_pair = {(e["human_min_prob"], e["human_min_margin"]): e for e in table}
    assert by_pair[(0.5, 0.0)]["portraits_verified_rate"] == 1.0
    assert by_pair[(0.5, 0.0)]["non_human_images_verified_as_human_rate"] == 0.5  # the cat photo at 0.62
    assert by_pair[(0.7, 0.25)]["portraits_verified_rate"] == 0.5
    assert by_pair[(0.7, 0.25)]["non_human_images_verified_as_human_rate"] == 0.0

    best = metrics.best_threshold(table, max_false_rate=0.0)
    assert (best["human_min_prob"], best["human_min_margin"]) == (0.7, 0.0)
    assert metrics.best_threshold(table, max_false_rate=-1) is None
    assert (
        metrics.best_threshold([{"non_human_images_verified_as_human_rate": None, "portraits_verified_rate": 1}], 1)
        is None
    )
    assert metrics.accepted_as_human([], 0.1, 0.0) is False


# ---- the report and the gates ----------------------------------------------------------------------------------------------------


def test_build_report_and_gates():
    rows = [row("a", [face()], ms=100), row("b", [], ms=300), row("c", error="x", ms=1)]
    labels = {"a": label("a"), "b": label("b", "not_human", "pets"), "c": label("c")}
    report = metrics.build_report(rows, labels, 0.1)
    assert report["speed_ms"] == {"p50": 300, "p90": 300}
    assert metrics.build_report([], labels, 0.1)["speed_ms"] == {"p50": None, "p90": None}
    assert metrics.lookup(report, "portraits.detection.exactly_one_human_rate") == 1.0

    gates = {
        "portraits.detection.exactly_one_human_rate": {"min": 0.9},
        "not_human.ALL.reported_as_real_human_rate": {"max": 0.0},
    }
    assert metrics.check_gates(report, gates) == []
    failing = {
        "portraits.detection.exactly_one_human_rate": {"min": 1.5},
        "not_human.ALL.images": {"max": 0},
        "portraits.nothing.here": {"min": 0},
        "portraits.minor.false_flag_rate_labelled_30_plus": {"min": 0},
    }
    problems = metrics.check_gates(report, failing)
    assert len(problems) == 4
    assert "below the minimum" in problems[0]
    assert "above the maximum" in problems[1]
    assert problems[2].endswith("not measured")
    assert problems[3].endswith("not measured")  # no adult was scored, so the rate is None


# ---- the runner -------------------------------------------------------------------------------------------------------------------


def fake_pipeline(path):
    if path.name == "broken.jpg":
        raise ValueError("cannot read")
    result = {
        "status": "PASSED", "reason": "All checks passed.", "accepted": True, "other_face_count": 0, "unsure_face_count": 0, "other_faces": [{"label": "Cat"}],
        "human_faces": [
            {"box_xyxy": [0, 0, 10, 20], "gender": "Male", "gender_confidence": 0.9, "age_group": "30 to 33",
             "age_status": "Estimated", "estimated_age": 31.0, "age_range": [25, 37], "p_under_18": 0.0,
             "verification_confidence": 0.8}
        ],
        "image_type": {"status": "PHOTO_LIKE"}, "diagnostics": {"processing_time_ms": 42.0},
    }  # fmt: skip
    return result, [candidate(0.8, 0.5)]


def make_dataset(tmp_path):
    (tmp_path / "pets").mkdir()
    (tmp_path / "fairface").mkdir()
    names = ["fairface/a.jpg", "fairface/b.jpg", "pets/broken.jpg", "pets/c.jpg"]
    lines = ["file,set,expectation,age,gender,race"] + [
        f"{n},{n.split('/')[0]},{'human' if n.startswith('fair') else 'not_human'},20-29,Male,White" for n in names
    ]
    (tmp_path / "labels.csv").write_text("\n".join(lines), encoding="utf-8")
    return tmp_path


def test_analyse_file_flattens_the_result_and_records_failures():
    ok = runner.analyse_file(Path("data/fairface/a.jpg"), fake_pipeline)
    assert ok["file"] == "fairface/a.jpg"
    assert ok["humans"][0]["area"] == 200
    assert ok["other_labels"] == ["Cat"]
    assert ok["ms"] == 42.0
    assert ok["reason"] == "All checks passed."
    assert ok["candidates"] == [candidate(0.8, 0.5)]

    failed = runner.analyse_file(Path("data/pets/broken.jpg"), fake_pipeline)
    assert failed["error"] == "ValueError: cannot read"
    assert failed["ms"] >= 0


def test_runner_resumes_where_it_stopped_and_honours_limits(tmp_path):
    data = make_dataset(tmp_path)
    results = tmp_path / "out" / "results.jsonl"

    def work(path):
        return runner.analyse_file(Path(path), fake_pipeline)

    first = runner.run_folder(data, results, workers=0, limit_per_set=1, work=work)
    assert [r["file"] for r in first] == ["fairface/a.jpg", "pets/broken.jpg"]
    second = runner.run_folder(data, results, workers=0, work=work)
    assert sorted(r["file"] for r in second) == ["fairface/a.jpg", "fairface/b.jpg", "pets/broken.jpg", "pets/c.jpg"]
    only = runner.run_folder(data, tmp_path / "pets.jsonl", workers=0, sets=["pets"], work=work)
    assert {r["file"] for r in only} == {"pets/broken.jpg", "pets/c.jpg"}
    assert runner.load_results(tmp_path / "nothing.jsonl") == []
    assert runner.load_labels(data / "labels.csv")["pets/c.jpg"]["expectation"] == "not_human"


def test_runner_reports_its_progress(tmp_path, capsys):
    (tmp_path / "pets").mkdir()
    names = [f"pets/{index}.jpg" for index in range(26)]
    lines = ["file,set,expectation,age,gender,race"] + [f"{name},pets,not_human,,," for name in names]
    (tmp_path / "labels.csv").write_text("\n".join(lines), encoding="utf-8")
    rows = runner.run_folder(tmp_path, tmp_path / "r.jsonl", workers=0, work=pool_work)
    assert len(rows) == 26
    assert "25/26 analysed" in capsys.readouterr().out


def test_runner_can_use_a_process_pool(tmp_path):
    data = make_dataset(tmp_path)
    rows = runner.run_folder(data, tmp_path / "pool.jsonl", workers=2, threads=1, sets=["fairface"], work=pool_work)
    assert {r["file"] for r in rows} == {"fairface/a.jpg", "fairface/b.jpg"}


def pool_work(path):  # module level so it can be pickled for the pool
    return {"file": Path(path).parent.name + "/" + Path(path).name, "error": None}


def test_worker_initialiser_sets_the_environment(monkeypatch):
    monkeypatch.delenv("PPA_MODELS__ORT_THREADS", raising=False)
    runner._init_worker(2)
    import os

    assert os.environ["PPA_MODELS__ORT_THREADS"] == "2"
    assert os.environ["PPA_MODELS__WARM_UP_ON_START"] == "false"
    monkeypatch.delenv("PPA_MODELS__ORT_THREADS")
    monkeypatch.delenv("PPA_MODELS__WARM_UP_ON_START")


def test_the_default_worker_reports_a_missing_file_as_an_error(tmp_path):
    result = runner._work(str(tmp_path / "pets" / "missing.jpg"))
    assert result["file"] == "pets/missing.jpg"
    assert result["error"].startswith("FileNotFoundError")


def test_run_pipeline_records_every_verdict_it_verifies(test_image_bytes, tmp_path):
    from app.services import analyzer as az
    from tests.test_analyzer import HUMAN, PHOTO, adult, detected

    path = tmp_path / "p.jpg"
    path.write_bytes(test_image_bytes)
    scored = {**HUMAN, "scores": {"real_human": 0.9, "cat": 0.1}, "margin": 0.8}
    with (
        patch.object(az, "detect_faces", return_value=([detected([200, 200, 400, 400])], ["Detector test"])),
        patch.object(az, "detect_entities_and_persons", return_value=([], [], [])),
        patch("app.services.analyzer.classify_boxes", side_effect=lambda img, boxes: [scored] * len(boxes)) as original,
        patch.object(az, "classify_image_type", return_value=PHOTO),
        patch.object(az, "predict_face_demographics", return_value=adult()),
    ):
        # recorded_candidates wraps whatever classify_boxes is when the block starts
        result, candidates = runner.run_pipeline(path)
    assert original.called
    assert result["status"] == "PASSED"
    assert result["annotated_image"] is None
    assert candidates == [{"kind": "human", "p_human": 0.9, "margin": 0.8, "best_is_human": True}]


def test_unavailable_verdicts_are_not_recorded():
    from app.services import analyzer as az

    with patch.object(az, "classify_boxes", return_value=[{"verdict": "unavailable", "scores": {}, "margin": None}]):
        with runner.recorded_candidates() as records:
            az.classify_boxes(None, [[0, 0, 1, 1]])
    assert records == []


# ---- the command line -------------------------------------------------------------------------------------------------------------


def test_command_line_runs_reports_and_checks_gates(tmp_path, capsys):
    (tmp_path / ".eval_data").mkdir()
    data = make_dataset(tmp_path / ".eval_data")

    def work(path):
        return runner.analyse_file(Path(path), fake_pipeline)

    (tmp_path / "evaluation").mkdir()
    gates = tmp_path / "evaluation" / "gates.json"
    gates.write_text(json.dumps({"portraits.detection.exactly_one_human_rate": {"min": 0.9}}), encoding="utf-8")
    with patch.object(runner, "_work", work):
        code = evaluate.main(["--workers", "0", "--gates"], workspace=tmp_path)
    assert code == 0
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    assert report["portraits"]["detection"]["images"] == 2
    assert report["thresholds"]
    assert len(report["minor_thresholds"]) == 7
    assert "best verification thresholds" in capsys.readouterr().out

    gates.write_text(json.dumps({"portraits.detection.exactly_one_human_rate": {"min": 1.1}}), encoding="utf-8")
    assert evaluate.main(["--skip-run", "--gates"], workspace=tmp_path) == 1
    assert "GATE FAILED" in capsys.readouterr().out
    assert evaluate.main(["--skip-run", "--minor-threshold", "0.3"], workspace=tmp_path) == 0


# ---- the generated tables of the evaluation document ---------------------------------------------------------------------


def sample_report():
    rows = [
        row("a", [face("Male", 25)], candidates=[candidate(0.8, 0.5)]),
        row("b", [face("Female", 34)], candidates=[candidate(0.65, 0.4)]),
        row("c", [], others=1, status="ONLY_NON_HUMAN_FACES", candidates=[candidate(0.2, 0.5, best_is_human=False)]),
    ]
    labels = {
        "a": label("a", age="20-29"),
        "b": label("b", age="3-9", gender="Male"),
        "c": label("c", "not_human", "pets"),
    }
    report = metrics.build_report(rows, labels, 0.1)
    report["run"] = metrics.run_info(
        rows,
        labels,
        {"human_min_prob": 0.7, "human_min_margin": 0.25, "minor_risk_probability": 0.1, "enable_scrfd": False},
        "2026-10-13",
    )
    report["thresholds"] = metrics.threshold_sweep(
        metrics.human_set(rows, labels), metrics.not_human_set(rows, labels), [0.5, 0.55, 0.6, 0.7], [0.0, 0.25]
    )
    report["minor_thresholds"] = metrics.minor_sweep(metrics.human_set(rows, labels), [0.1, 0.5])
    return report


def test_run_info_counts_the_images_of_each_set():
    info = sample_report()["run"]
    assert info["images_per_set"] == {"fairface": 2, "pets": 1}
    assert info["date"] == "2026-10-13"


def test_every_section_of_the_document_is_generated_from_the_report():
    from evaluation import render

    text = render.render(sample_report())
    assert "Run of 2026-10-13. Images: fairface 2, pets 1." in text
    assert "SCRFD off" in text
    assert "| At least one real human face found | 100.0 % |" in text
    assert "| ALL | 1 | 0.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % |" in text
    assert "`human_min_prob`" in text
    assert "0.7 (used)" in text
    assert "0.5 |" in text
    assert "0.6 |" in text
    assert "#### Age by race group" in text
    assert "| Group | Faces | Age Unsure |" in text
    assert "Faces scored: 1 labelled 0-9, 0 labelled 30 or older." in text
    assert "| any |" in text
    assert "median 100 ms and 90th percentile 100 ms" in text
    assert "n/a" in text  # no adult was scored, so that rate is not available
    assert "| Set | Images |" in text


def test_numbers_and_percentages_have_a_not_available_form():
    from evaluation import render

    assert render.percent(None) == "n/a"
    assert render.percent(0.1234) == "12.3 %"
    assert render.percent(0.1234, 0) == "12 %"
    assert render.number(None) == "n/a"
    assert render.number(1.236) == "1.24"


def test_only_the_text_between_the_markers_is_replaced():
    from evaluation import render

    document = f"intro\n{render.START_MARKER}\nold numbers\n{render.END_MARKER}\noutro\n"
    updated = render.replace_between_markers(document, "NEW")
    assert updated == f"intro\n{render.START_MARKER}\n\nNEW\n\n{render.END_MARKER}\noutro\n"
    assert render.replace_between_markers(updated, "NEWER").count("NEWER") == 1
    with pytest.raises(ValueError):
        render.replace_between_markers("no markers here", "x")


def test_render_script_updates_the_document(tmp_path, capsys):
    from evaluation import render

    render_script = __import__("render_evaluation")
    (tmp_path / ".eval_data").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / ".eval_data" / "report.json").write_text(json.dumps(sample_report()), encoding="utf-8")
    doc = tmp_path / "docs" / "EVALUATION_AND_FAIRNESS.md"
    doc.write_text(
        f"hand written\n{render.START_MARKER}\nSTALE-NUMBERS\n{render.END_MARKER}\nmore hand written\n",
        encoding="utf-8",
    )
    assert render_script.main([], workspace=tmp_path) == 0
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("hand written")
    assert text.rstrip().endswith("more hand written")
    assert "Run of 2026-10-13" in text
    assert "STALE-NUMBERS" not in text
    assert "updated" in capsys.readouterr().out


def benchmark_output(p50):
    stage = {"min": 1, "mean": 2, "p50": p50, "p90": p50 + 5, "max": 99}
    return {
        "environment": {
            "python": "3.14.5", "onnxruntime": "1.31.0", "logical_cpus": 14, "onnx_threads_per_model": "min(4, cores)",
            "analysed_resolution": [178, 218], "iterations": 5,
        },
        "latency_ms": {"face detection": stage, "END-TO-END": {**stage, "p50": p50 * 2}},
    }  # fmt: skip


def test_the_speed_table_compares_parallel_and_sequential_runs(tmp_path, capsys):
    from evaluation import render

    text = render.render_benchmark(benchmark_output(100), benchmark_output(250))
    assert "14 logical CPUs" in text
    assert "178 x 218 px" in text
    assert "| face detection | 250 | 255 | 100 | 105 |" in text
    assert "| END-TO-END | 500 | 255 | 200 | 105 |" in text

    render_script = __import__("render_evaluation")
    (tmp_path / ".eval_data").mkdir()
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "EVALUATION_AND_FAIRNESS.md"
    doc.write_text(
        f"a\n{render.BENCHMARK_START_MARKER}\nSTALE-SPEED\n{render.BENCHMARK_END_MARKER}\nb\n", encoding="utf-8"
    )
    data = tmp_path / ".eval_data"
    (data / "benchmark_parallel.json").write_text(json.dumps(benchmark_output(100)), encoding="utf-8")
    (data / "benchmark_sequential.json").write_text(json.dumps(benchmark_output(250)), encoding="utf-8")
    assert render_script.main(["--benchmark"], workspace=tmp_path) == 0  # no report.json here: only the speed table
    assert "STALE-SPEED" not in doc.read_text(encoding="utf-8")
    assert "| face detection | 250 |" in doc.read_text(encoding="utf-8")
    capsys.readouterr()


def test_the_shipped_gates_are_well_formed_and_name_things_a_report_contains():
    gates = json.loads((ROOT / "evaluation" / "gates.json").read_text(encoding="utf-8"))
    assert gates
    for path, limits in gates.items():
        assert limits, path
        assert limits.keys() <= {"min", "max"}, path
        assert all(isinstance(value, int | float) for value in limits.values()), path

    rows = [row("a", [face()])] + [row(name, [], others=1) for name in ("pets", "birds", "anime")]
    labels = {"a": label("a")} | {name: label(name, "not_human", name) for name in ("pets", "birds", "anime")}
    report = metrics.build_report(rows, labels, 0.1)
    for path in gates:
        metrics.lookup(report, path)  # raises when the report has no such value
    assert metrics.check_gates(report, gates) != []  # one portrait and no minors: some limits cannot be met
