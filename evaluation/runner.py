"""
Run the real pipeline over labelled images and record what it said, one JSON line per image.

Every verification verdict is recorded as well (the share given to "real human face" and the lead over the runner-up),
so the verification thresholds can be re-evaluated later without running the models again.
"""

import csv
import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evaluation.metrics import Row

Analyzer = Callable[[Path], tuple[dict[str, Any], list[Row]]]


def load_labels(path: Path) -> dict[str, Row]:
    """``labels.csv`` as ``{file: row}``."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def load_results(path: Path) -> list[Row]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def recorded_candidates() -> Iterator[list[Row]]:
    """Collect the verdict on every crop the analyzer verifies while the block runs."""
    from app.services import analyzer

    records: list[Row] = []
    original = analyzer.classify_boxes

    def recording(image: Any, boxes: list[list[int]]) -> Any:
        verdicts = original(image, boxes)
        for verdict in verdicts:
            scores = verdict["scores"]
            if scores:
                best = max(scores, key=scores.__getitem__)
                records.append(
                    {
                        "kind": verdict["verdict"],
                        "p_human": scores.get("real_human", 0.0),
                        "margin": verdict["margin"],
                        "best_is_human": best == "real_human",
                    }
                )
        return verdicts

    with patch.object(analyzer, "classify_boxes", recording):
        yield records


def run_pipeline(path: Path) -> tuple[dict[str, Any], list[Row]]:
    """The real analysis of one image file, with the verification verdicts it produced."""
    from app.services.analyzer import analyze_profile_photo_image
    from app.services.image_utils import decode_image_bytes

    decoded = decode_image_bytes(path.read_bytes())
    with recorded_candidates() as candidates:
        result = analyze_profile_photo_image(
            decoded.image, path.name, decoded.file_format, decoded, render_annotation=False
        )
    return dict(result), candidates


def analyse_file(path: Path, analyze: Analyzer = run_pipeline) -> Row:
    """One result row; a failure is recorded in ``error`` instead of stopping the run."""
    started = time.perf_counter()
    row: Row = {"file": path.parent.name + "/" + path.name}
    try:
        result, candidates = analyze(path)
    except Exception as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}", "ms": round((time.perf_counter() - started) * 1000, 1)}

    humans = []
    for face in result["human_faces"]:
        x1, y1, x2, y2 = face["box_xyxy"]
        humans.append(
            {
                "area": (x2 - x1) * (y2 - y1),
                "gender": face["gender"],
                "gender_confidence": face["gender_confidence"],
                "age_group": face["age_group"],
                "age_status": face["age_status"],
                "estimated_age": face["estimated_age"],
                "age_range": face["age_range"],
                "p_under_18": face["p_under_18"],
                "verification": face["verification_confidence"],
            }
        )
    return {
        **row,
        "error": None,
        "status": str(result["status"]),
        "reason": result["reason"],
        "accepted": result["accepted"],
        "humans": humans,
        "others": result["other_face_count"],
        "unsure": result["unsure_face_count"],
        "other_labels": sorted({other["label"] for other in result["other_faces"]}),
        "candidates": candidates,
        "image_type": result["image_type"]["status"],
        "ms": result["diagnostics"]["processing_time_ms"],
    }


def _init_worker(threads: int) -> None:
    os.environ["PPA_MODELS__ORT_THREADS"] = str(threads)
    os.environ["PPA_MODELS__WARM_UP_ON_START"] = "false"


def _work(path: str) -> Row:
    return analyse_file(Path(path))


def run_folder(
    data: Path,
    results_path: Path,
    workers: int = 3,
    threads: int = 4,
    limit_per_set: int | None = None,
    sets: list[str] | None = None,
    work: Callable[[str], Row] | None = None,
) -> list[Row]:
    """
    Analyse every labelled image that has no result yet and append each result to ``results_path`` as it arrives, so an
    interrupted run continues where it stopped. ``workers=0`` runs in this process.
    """
    work = work or _work
    labels = load_labels(data / "labels.csv")
    done = {row["file"] for row in load_results(results_path)}
    taken: dict[str, int] = {}
    todo: list[str] = []
    for file, label in labels.items():
        if sets and label["set"] not in sets:
            continue
        if limit_per_set is not None and taken.get(label["set"], 0) >= limit_per_set:
            continue
        taken[label["set"]] = taken.get(label["set"], 0) + 1
        if file not in done:
            todo.append(str(data / file))

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as out:
        if workers <= 0:
            stream: Iterator[Row] = (work(path) for path in todo)
            pool = None
        else:
            pool = Pool(workers, initializer=_init_worker, initargs=(threads,))
            stream = pool.imap_unordered(work, todo)
        try:
            for index, row in enumerate(stream, start=1):
                out.write(json.dumps(row) + "\n")
                out.flush()
                if index % 25 == 0:
                    print(f"  {index}/{len(todo)} analysed", flush=True)
        finally:
            if pool is not None:
                pool.terminate()
    return load_results(results_path)
