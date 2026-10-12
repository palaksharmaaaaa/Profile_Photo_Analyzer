"""
Measure the age/gender stage on labelled data (FairFace validation set, CC BY 4.0).

    python scripts/evaluate_fairface.py --csv fairface_label_val.csv --images <folder containing val/> [--limit 1000]

FairFace labels age in 9 bins ("0-2", "3-9", "10-19", "20-29", ..., "more than 70") and gender as Male/Female.
The model was trained on UTKFace, so FairFace is an out-of-distribution test. Reported numbers:

  * gender accuracy among the faces where a gender was given, and the share reported "Unsure"
  * share of faces whose estimated age lies inside the labelled FairFace bin (+-1 year tolerance)
  * mean absolute error against the bin centre (an approximation: the true age is not in the dataset)
  * minor recall: of faces labelled under 18, the share flagged possible-minor (P(age<18) >= threshold);
    plus the false-alarm rate on faces labelled 30 or older
  * everything broken down by the dataset's race groups

Nothing in this script is a promise about production accuracy; it only reports what it measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.face_service import predict_face_demographics  # noqa: E402

BINS = {
    "0-2": (0, 2), "3-9": (3, 9), "10-19": (10, 19), "20-29": (20, 29), "30-39": (30, 39),
    "40-49": (40, 49), "50-59": (50, 59), "60-69": (60, 69), "more than 70": (70, 90),
}  # fmt: skip


def evaluate(csv_path: Path, root: Path, limit: int | None) -> dict:
    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    for row in rows:
        image = cv2.imread(str(root / row["file"]))
        if image is None or row["age"] not in BINS:
            continue
        height, width = image.shape[:2]
        demo = predict_face_demographics(image, [0, 0, width, height])  # FairFace images are pre-cropped faces
        low, high = BINS[row["age"]]
        for group in ("ALL", row["race"]):
            s = stats[group]
            s["n"] += 1
            if demo.gender == "Unsure":
                s["gender_unsure"] += 1
            elif demo.gender in ("Male", "Female"):
                s["gender_answered"] += 1
                s["gender_correct"] += demo.gender == row["gender"]
            if demo.age_status != "Estimated":
                s["age_unsure"] += 1
            else:
                s["age_answered"] += 1
                s["age_in_bin"] += low - 1 <= demo.estimated_age <= high + 1
                s["age_abs_err"] += abs(demo.estimated_age - (low + high) / 2)
            if high < 18:
                s["minors"] += 1
                s["minors_flagged"] += (demo.p_under_18 or 0.0) >= settings.MINOR_RISK_PROBABILITY
            elif low >= 30:
                s["adults30"] += 1
                s["adults30_flagged"] += (demo.p_under_18 or 0.0) >= settings.MINOR_RISK_PROBABILITY

    def ratio(a: float, b: float) -> float | None:
        return round(a / b, 4) if b else None

    return {
        group: {
            "faces": int(s["n"]),
            "gender_accuracy_when_answered": ratio(s["gender_correct"], s["gender_answered"]),
            "gender_unsure_rate": ratio(s["gender_unsure"], s["n"]),
            "age_unsure_rate": ratio(s["age_unsure"], s["n"]),
            "age_inside_labelled_bin": ratio(s["age_in_bin"], s["age_answered"]),
            "age_mae_vs_bin_centre_years": ratio(s["age_abs_err"], s["age_answered"]),
            "minor_recall": ratio(s["minors_flagged"], s["minors"]),
            "false_minor_flag_rate_age_30plus": ratio(s["adults30_flagged"], s["adults30"]),
        }
        for group, s in sorted(stats.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = evaluate(args.csv, args.images, args.limit)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
