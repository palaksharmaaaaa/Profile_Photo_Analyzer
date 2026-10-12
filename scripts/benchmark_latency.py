"""
Measure CPU latency of each pipeline stage and of the whole analysis on one image.

    python scripts/benchmark_latency.py --image path/to/portrait.jpg [--iterations 15] [--sequential] [--json out.json]

Use a real portrait: with a synthetic image the face-dependent stages (verification, age/gender) never run
and would not be measured. ``--sequential`` turns parallel inference off, to measure what running the detectors and
the age views at the same time gains. Results depend entirely on the machine; nothing here is a guarantee.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.analyzer import analyze_profile_photo_image  # noqa: E402
from app.services.classifier_service import classify_boxes, classify_image_type, detect_entities_and_persons  # noqa: E402
from app.services.face_service import detect_faces, predict_face_demographics  # noqa: E402
from app.services.image_utils import decode_image_bytes, pil_to_bgr  # noqa: E402
from app.services.quality_service import check_image_information  # noqa: E402


def time_stage(func: Callable[[], Any], iterations: int, warmup: int = 2) -> dict[str, float]:
    """Run ``func`` ``warmup`` times unmeasured, then ``iterations`` times, and summarise the durations in ms."""
    for _ in range(warmup):
        func()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()

    def pick(q: float) -> float:
        return samples[min(len(samples) - 1, int(q * len(samples)))]

    return {
        "min": round(samples[0], 1),
        "mean": round(statistics.mean(samples), 1),
        "p50": round(pick(0.50), 1),
        "p90": round(pick(0.90), 1),
        "max": round(samples[-1], 1),
    }


def measure(image_bytes: bytes, iterations: int) -> dict[str, Any]:
    """Time every stage on one image; raises ``SystemExit`` when the image has no face to exercise the later stages."""
    decoded = decode_image_bytes(image_bytes)
    bgr = pil_to_bgr(decoded.image)
    faces, _ = detect_faces(bgr)
    if not faces:
        raise SystemExit("No face found in the benchmark image; use a portrait so every stage is exercised.")
    box = faces[0].bbox

    stages = {
        "information check": lambda: check_image_information(bgr),
        "image type (SigLIP2)": lambda: classify_image_type(bgr),
        "face detection (SCRFD + YuNet)": lambda: detect_faces(bgr),
        "animal + person + anime detection": lambda: detect_entities_and_persons(bgr),
        "face verification (1 crop)": lambda: classify_boxes(bgr, [box]),
        "age + gender (MiVOLO, 3 views)": lambda: predict_face_demographics(bgr, box),
        "END-TO-END": lambda: analyze_profile_photo_image(decoded.image, "bench", decoded.file_format, decoded),
    }
    return {
        "environment": {
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "onnx_threads_per_model": settings.models.ort_threads or "min(4, cores)",
            "parallel_inference": settings.models.parallel_inference,
            "analysed_resolution": list(decoded.image.size),
            "iterations": iterations,
        },
        "latency_ms": {name: time_stage(function, iterations) for name, function in stages.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--sequential", action="store_true", help="turn parallel inference off")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    if args.sequential:
        settings.models.parallel_inference = False
    report = measure(args.image.read_bytes(), args.iterations)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
