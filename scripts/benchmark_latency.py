"""
Benchmark harness for measuring CPU operational latency across Profile Photo Analyzer pipeline stages.
Runs repeatable iterations on synthetic test inputs and reports min, mean, median (p50), p90, p95, and max.
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from app.services.analyzer import analyze_profile_photo_image
from app.services.classifier_service import (
    _classify_via_heuristics,
    detect_non_human_entities,
    is_biological_diagram_or_drawing,
    is_cartoon_or_illustration,
)
from app.services.face_service import (
    detect_faces,
    predict_face_demographics,
)
from app.services.quality_service import check_image_information


def _time_stage(name: str, func: Callable[[], Any], iterations: int = 15) -> dict[str, float]:
    print(f"--> Benchmarking: {name} ({iterations} iterations)...", flush=True)
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    sorted_lat = sorted(latencies)
    p50_idx = int(0.50 * len(sorted_lat))
    p90_idx = min(int(0.90 * len(sorted_lat)), len(sorted_lat) - 1)
    p95_idx = min(int(0.95 * len(sorted_lat)), len(sorted_lat) - 1)
    return {
        "min": round(sorted_lat[0], 2),
        "mean": round(statistics.mean(sorted_lat), 2),
        "p50": round(sorted_lat[p50_idx], 2),
        "p90": round(sorted_lat[p90_idx], 2),
        "p95": round(sorted_lat[p95_idx], 2),
        "max": round(sorted_lat[-1], 2),
    }


def run_benchmark(iterations: int = 15, warmup: int = 2) -> dict[str, dict[str, float]]:
    # Create realistic test portrait array
    arr = np.zeros((600, 600, 3), dtype=np.uint8)
    arr[100:500, 150:450] = (210, 180, 140)  # Center face tone
    cv2.circle(arr, (300, 260), 120, (190, 160, 120), -1)
    cv2.circle(arr, (260, 240), 15, (40, 40, 40), -1)  # Left eye
    cv2.circle(arr, (340, 240), 15, (40, 40, 40), -1)  # Right eye
    cv2.ellipse(arr, (300, 310), (35, 15), 0, 0, 180, (60, 40, 40), 5)  # Smile

    img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    face_box = [150, 100, 450, 500]

    print("Warming up models...", flush=True)
    for _ in range(warmup):
        check_image_information(arr)
        is_biological_diagram_or_drawing(arr)
        is_cartoon_or_illustration(arr)
        detect_faces(arr)
        detect_non_human_entities(arr)
        predict_face_demographics(arr, face_box)
        analyze_profile_photo_image(img)

    results: dict[str, dict[str, float]] = {}

    results["Entropy & Information Check"] = _time_stage(
        "Entropy & Information", lambda: check_image_information(arr), iterations
    )
    results["Biological Diagram Check"] = _time_stage(
        "Biological Diagram", lambda: is_biological_diagram_or_drawing(arr), iterations
    )
    results["Cartoon & Palette Screening"] = _time_stage(
        "Cartoon Screening", lambda: is_cartoon_or_illustration(arr), iterations
    )
    results["Heuristic Pipeline Screener"] = _time_stage(
        "Heuristic Screener", lambda: _classify_via_heuristics(img), iterations
    )
    results["YuNet Face Detection"] = _time_stage("YuNet Face Detection", lambda: detect_faces(arr), iterations)
    results["Non-Human & Animal Screening (YOLO + Anime)"] = _time_stage(
        "Non-Human & Animal Screening", lambda: detect_non_human_entities(arr), iterations
    )
    results["GenderAge Demographics ONNX"] = _time_stage(
        "GenderAge Demographics", lambda: predict_face_demographics(arr, face_box), iterations
    )
    results["End-to-End Analysis Pipeline"] = _time_stage(
        "End-to-End Pipeline", lambda: analyze_profile_photo_image(img), iterations
    )

    return results


def print_results(results: dict[str, dict[str, float]]) -> None:
    header = (
        f"| {'Pipeline Stage':<42} | {'min':>7} | {'mean':>7} | {'p50':>7} | {'p90':>7} | {'p95':>7} | {'max':>7} |"
    )
    divider = f"|:{'-' * 42} | {'-' * 7}:| {'-' * 7}:| {'-' * 7}:| {'-' * 7}:| {'-' * 7}:| {'-' * 7}:|"
    print("\n### Empirical Operational Latency Benchmark Results (CPU)")
    print(header)
    print(divider)
    for stage, metrics in results.items():
        print(
            f"| {stage:<42} | {metrics['min']:>5.2f}ms | {metrics['mean']:>5.2f}ms | "
            f"{metrics['p50']:>5.2f}ms | {metrics['p90']:>5.2f}ms | {metrics['p95']:>5.2f}ms | {metrics['max']:>5.2f}ms |"
        )


if __name__ == "__main__":
    print("Running latency benchmark across 15 iterations on CPU...", flush=True)
    benchmarks = run_benchmark(iterations=15)
    print_results(benchmarks)
