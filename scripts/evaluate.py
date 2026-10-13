"""
Measure the whole service on labelled images and report what it gets right and wrong.

    python scripts/fetch_eval_data.py                       # once: downloads about 2,000 labelled images
    python scripts/evaluate.py --workers 3                  # run the pipeline, write results.jsonl and report.json
    python scripts/evaluate.py --skip-run --gates evaluation/gates.json   # only re-compute the report and check the gates

The pipeline results are cached in ``results.jsonl`` (one line per image), so an interrupted run continues and
a report or threshold sweep can be recomputed without running the models again.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import metrics, runner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(".eval_data"))
    parser.add_argument("--results", type=Path, help="default: <data>/results.jsonl")
    parser.add_argument("--report", type=Path, help="default: <data>/report.json")
    parser.add_argument("--workers", type=int, default=3, help="processes (0 = run in this process)")
    parser.add_argument("--threads", type=int, default=4, help="ONNX threads per process")
    parser.add_argument("--limit-per-set", type=int)
    parser.add_argument("--sets", nargs="+")
    parser.add_argument("--skip-run", action="store_true", help="do not run the models, only report")
    parser.add_argument("--gates", type=Path, help="JSON file of limits; the exit code is 1 when one is violated")
    parser.add_argument("--minor-threshold", type=float, help="default: the configured minor_risk_probability")
    args = parser.parse_args(argv)

    from app.core.config import settings

    results_path = args.results or args.data / "results.jsonl"
    report_path = args.report or args.data / "report.json"
    labels = runner.load_labels(args.data / "labels.csv")
    if args.skip_run:
        rows = runner.load_results(results_path)
    else:
        rows = runner.run_folder(args.data, results_path, args.workers, args.threads, args.limit_per_set, args.sets)

    report = metrics.build_report(rows, labels, args.minor_threshold or settings.age.minor_risk_probability)
    report["run"] = metrics.run_info(
        rows,
        labels,
        {
            "human_min_prob": settings.verification.human_min_prob,
            "human_min_margin": settings.verification.human_min_margin,
            "minor_risk_probability": settings.age.minor_risk_probability,
            "enable_scrfd": settings.detection.enable_scrfd,
        },
        datetime.date.today().isoformat(),
    )
    report["thresholds"] = metrics.threshold_sweep(
        metrics.human_set(rows, labels),
        metrics.not_human_set(rows, labels),
        [p / 100 for p in range(30, 96, 5)],
        [m / 100 for m in range(0, 61, 5)],
    )
    report["minor_thresholds"] = metrics.minor_sweep(
        metrics.human_set(rows, labels), [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("thresholds", "minor_thresholds", "run")},
            indent=2,
        )
    )
    best = metrics.best_threshold(report["thresholds"], max_false_rate=0.01)
    print("best verification thresholds with at most 1 % non-human images verified as human:", best)

    if args.gates:
        problems = metrics.check_gates(report, json.loads(args.gates.read_text(encoding="utf-8")))
        for problem in problems:
            print("GATE FAILED:", problem)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
