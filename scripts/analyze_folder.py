"""
Run the full analysis pipeline over a folder of images and write one CSV row per image.

    python scripts/analyze_folder.py <folder> --out results.csv [--sample 300 --seed 1] [--list names.txt] [--workers 2]

Use it to check detection and verification on a folder where you know what every image contains. For a folder of
real single-face portraits, expect 1 real human face and 0 non-human faces everywhere. It reports what the
pipeline produced; it cannot know the true ages or genders unless you provide labels (see evaluate_fairface.py).
When a result has several human faces, the columns describe the first one (reading order).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
COLUMNS = [
    "file", "error", "status", "accepted", "humans", "others", "unsure", "rejected", "other_labels",
    "quality_status", "quality_reason", "face_w", "face_h", "detection_confidence", "verification_confidence",
    "age_group", "age_status", "estimated_age", "age_low", "age_high", "age_group_probability", "p_under_18",
    "possible_minor", "gender", "gender_confidence", "image_type", "photo_share", "ms",
]  # fmt: skip


def _init_worker(threads: int) -> None:
    os.environ["PPA_ORT_INTRA_OP_THREADS"] = str(threads)


def analyze_one(path: str) -> dict:
    """Analyse one image and flatten the result into a CSV row (a failure is reported in the ``error`` column)."""
    from app.services.analyzer import analyze_profile_photo_image
    from app.services.image_utils import InvalidImageError, decode_image_bytes

    row: dict = {"file": Path(path).name}
    started = time.perf_counter()
    try:
        decoded = decode_image_bytes(Path(path).read_bytes())
        result = analyze_profile_photo_image(decoded.image, row["file"], decoded.file_format, decoded)
    except InvalidImageError as exc:
        return {**row, "error": exc.reason}
    except Exception as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}"}

    face = result["human_faces"][0] if result["human_faces"] else {}
    quality = face.get("quality") or {}
    x1, y1, x2, y2 = face.get("box_xyxy") or [0, 0, 0, 0]
    age_low, age_high = face.get("age_range") or [None, None]
    row.update(
        status=result["status"],
        accepted=result["accepted"],
        humans=result["real_human_face_count"],
        others=result["other_face_count"],
        unsure=result["unsure_face_count"],
        rejected=result["diagnostics"]["rejected_candidate_count"],
        other_labels="|".join(other["label"] for other in result["other_faces"]),
        quality_status=quality.get("status"),
        quality_reason=quality.get("reason"),
        face_w=x2 - x1,
        face_h=y2 - y1,
        detection_confidence=face.get("detection_confidence"),
        verification_confidence=face.get("verification_confidence"),
        age_group=face.get("age_group"),
        age_status=face.get("age_status"),
        estimated_age=face.get("estimated_age"),
        age_low=age_low,
        age_high=age_high,
        age_group_probability=face.get("age_group_probability"),
        p_under_18=face.get("p_under_18"),
        possible_minor=face.get("possible_minor"),
        gender=face.get("gender"),
        gender_confidence=face.get("gender_confidence"),
        image_type=result["image_type"]["label"],
        photo_share=result["image_type"]["photo_share"],
        ms=round((time.perf_counter() - started) * 1000),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="CSV file to write")
    parser.add_argument("--sample", type=int, help="analyse a random sample of this many images")
    parser.add_argument("--seed", type=int, default=1, help="seed for --sample")
    parser.add_argument("--list", type=Path, help="text file of image file names (one per line) to analyse")
    parser.add_argument("--workers", type=int, default=2, help="parallel worker processes")
    parser.add_argument("--threads", type=int, default=2, help="ONNX threads per worker")
    args = parser.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTENSIONS)
    if args.list:
        wanted = set(args.list.read_text(encoding="utf-8").split())
        files = [p for p in files if p.name in wanted]
    if args.sample and args.sample < len(files):
        files = sorted(random.Random(args.seed).sample(files, args.sample))
    print(f"analysing {len(files)} images with {args.workers} workers", flush=True)

    started = time.perf_counter()
    with (
        open(args.out, "w", newline="", encoding="utf-8") as handle,
        Pool(args.workers, initializer=_init_worker, initargs=(args.threads,)) as pool,
    ):
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for done, row in enumerate(pool.imap(analyze_one, [str(p) for p in files], chunksize=4), start=1):
            writer.writerow(row)
            if done % 100 == 0:
                handle.flush()
                print(f"{done}/{len(files)}  {done / (time.perf_counter() - started):.2f} img/s", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
