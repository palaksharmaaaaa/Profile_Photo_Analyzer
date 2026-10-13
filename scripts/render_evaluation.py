"""
Write the measured tables of an evaluation report into docs/EVALUATION_AND_FAIRNESS.md.

    python scripts/evaluate.py --skip-run               # (re)compute .eval_data/report.json from the cached results
    python scripts/render_evaluation.py                 # replace the text between the markers in the document
    python scripts/render_evaluation.py --benchmark-parallel a.json --benchmark-sequential b.json   # the speed table

Only the text between ``<!-- measured:start -->`` and ``<!-- measured:end -->`` (and between the benchmark markers) is
replaced; the rest of the document is written by hand and says what the numbers mean and do not mean.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=Path(".eval_data/report.json"))
    parser.add_argument("--doc", type=Path, default=ROOT / "docs" / "EVALUATION_AND_FAIRNESS.md")
    parser.add_argument("--benchmark-parallel", type=Path, help="output of benchmark_latency.py --parallel --json")
    parser.add_argument("--benchmark-sequential", type=Path, help="output of benchmark_latency.py --json")
    args = parser.parse_args(argv)

    document = args.doc.read_text(encoding="utf-8")
    if args.report.is_file():
        report = json.loads(args.report.read_text(encoding="utf-8"))
        document = render.replace_between_markers(document, render.render(report))
    if args.benchmark_parallel and args.benchmark_sequential:
        parallel = json.loads(args.benchmark_parallel.read_text(encoding="utf-8"))
        sequential = json.loads(args.benchmark_sequential.read_text(encoding="utf-8"))
        document = render.replace_between(
            document,
            render.BENCHMARK_START_MARKER,
            render.BENCHMARK_END_MARKER,
            render.render_benchmark(parallel, sequential),
        )
    args.doc.write_text(document, encoding="utf-8")
    print(f"updated {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
