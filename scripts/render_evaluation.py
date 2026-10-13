"""
Write the measured tables of an evaluation report into docs/EVALUATION_AND_FAIRNESS.md.

    python scripts/evaluate.py --skip-run               # (re)compute .eval_data/report.json from the cached results
    python scripts/render_evaluation.py                 # replace the text between the markers in the document

    python scripts/benchmark_latency.py --image p.jpg --json .eval_data/benchmark_sequential.json
    python scripts/benchmark_latency.py --image p.jpg --parallel --json .eval_data/benchmark_parallel.json
    python scripts/render_evaluation.py --benchmark     # also fill in the speed table from those two files

Only the text between ``<!-- measured:start -->`` and ``<!-- measured:end -->`` (and between the benchmark markers) is
replaced; the rest of the document is written by hand and says what the numbers mean and do not mean. All files are
under the project folder: ``.eval_data/`` and ``docs/``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None, *, workspace: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--benchmark", action="store_true", help="fill in the speed table from .eval_data/benchmark_*.json"
    )
    args = parser.parse_args(argv)

    data = workspace / ".eval_data"
    doc = workspace / "docs" / "EVALUATION_AND_FAIRNESS.md"
    document = doc.read_text(encoding="utf-8")
    report_path = data / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        document = render.replace_between_markers(document, render.render(report))
    if args.benchmark:
        parallel = json.loads((data / "benchmark_parallel.json").read_text(encoding="utf-8"))
        sequential = json.loads((data / "benchmark_sequential.json").read_text(encoding="utf-8"))
        document = render.replace_between(
            document,
            render.BENCHMARK_START_MARKER,
            render.BENCHMARK_END_MARKER,
            render.render_benchmark(parallel, sequential),
        )
    doc.write_text(document, encoding="utf-8")
    print(f"updated {doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
