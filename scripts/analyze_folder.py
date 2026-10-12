"""
Run the full analysis pipeline over a folder of images and write one CSV row per image.

    python scripts/analyze_folder.py <folder> --out results.csv [--sample 300 --seed 1] [--workers 3]

Use it to check detection/verification on a folder where you know what every image contains (for example a folder
of real single-face portraits: expect 1 real human face and 0 other faces everywhere). It reports what the pipeline
produced; it cannot know ages or genders unless you provide labels (see evaluate_fairface.py).
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
    "quality_status", "quality_reason", "face_w", "face_h", "det_conf", "verif_conf", "age_group", "age_status",
    "est_age", "age_lo", "age_hi", "age_bin_prob", "p_under_18", "minor_flag", "gender", "gender_conf",
    "image_type", "photo_score", "ms",
]  # fmt: skip


def _init_worker(threads: int) -> None:
    os.environ["PPA_ORT_INTRA_OP_THREADS"] = str(threads)


def analyze_one(path: str) -> dict:
    from app.services.analyzer import analyze_profile_photo_image
    from app.services.image_utils import InvalidImageError, decode_image_bytes

    row: dict = {"file": Path(path).name}
    started = time.perf_counter()
    try:
        decoded = decode_image_bytes(Path(path).read_bytes())
        result = analyze_profile_photo_image(decoded.image, row["file"], decoded.format_detected, decoded)
    except InvalidImageError as exc:
        return {**row, "error": exc.reason}
    except Exception as exc:  # keep going; report the failure in the CSV
        return {**row, "error": f"{type(exc).__name__}: {exc}"}

    humans = result["human_faces"]
    face = humans[0] if humans else {}
    quality = face.get("face_visibility") or {}
    box = face.get("box_xyxy") or [0, 0, 0, 0]
    rng = face.get("age_range") or [None, None]
    row.update(
        status=result["posting_status"],
        accepted=result["accepted"],
        humans=result["real_human_face_count"],
        others=result["other_face_count"],
        unsure=result["unsure_face_count"],
        rejected=result["technical_diagnostics"]["rejected_candidate_count"],
        other_labels="|".join(o["label"] for o in result["other_faces"]),
        quality_status=quality.get("status"),
        quality_reason=quality.get("reason"),
        face_w=box[2] - box[0],
        face_h=box[3] - box[1],
        det_conf=face.get("detection_confidence"),
        verif_conf=face.get("verification_confidence"),
        age_group=face.get("age_group"),
        age_status=face.get("age_status"),
        est_age=face.get("estimated_age"),
        age_lo=rng[0],
        age_hi=rng[1],
        age_bin_prob=face.get("age_confidence"),
        p_under_18=face.get("p_under_18"),
        minor_flag=face.get("possible_minor_risk"),
        gender=face.get("gender"),
        gender_conf=face.get("gender_confidence"),
        image_type=result["image_type_check"]["display_label"],
        photo_score=result["image_type_check"]["photo_score"],
        ms=round((time.perf_counter() - started) * 1000),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, help="analyse a random sample of this many images")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2, help="ONNX threads per worker")
    args = parser.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTENSIONS)
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
                rate = done / (time.perf_counter() - started)
                print(f"{done}/{len(files)}  {rate:.2f} img/s", flush=True)
    print(f"done in {time.perf_counter() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
